"""Credential resolution, and the stored session that clients on one machine share.

This module implements the platform's ``client-credentials`` contract, the same
one ``jnh`` and the Go SDK implement, and is held to it by the shared conformance
suite. Three parts of that contract are easy to get quietly wrong, so they are
worth stating where the code is:

* Renewal always rotates. The refresh token that renewed is dead the moment the
  platform answers, so writing the renewed session back is correctness, not an
  optimization: a client that kept it in memory would strand every other client
  on the machine, ``jnh`` included, on a token nothing accepts.
* The renewal call must not pass through the client's own credential
  interceptor, or it would ask this module for a token while holding the lock it
  took to renew.
* A call rejected as unauthenticated never reached the operation, so reissuing it
  once after a renewal is safe even for a write that is otherwise never replayed.
"""

from __future__ import annotations

import asyncio
import dataclasses
import enum
import json
import os
import sys
import tempfile
import threading
import time
from typing import Awaitable, Callable, Optional, Protocol, Union

from .errors import (
    ENV_API_KEY,
    CorruptSessionError,
    CredentialRefusedError,
    NoCredentialError,
    SessionExpiredError,
    SessionPersistError,
)

# The reserved prefix the platform's API keys carry. The server branches on it to
# decide which credential it was handed, so it is also how a client can tell what
# it resolved without asking anyone.
_API_KEY_PREFIX = "jennah_sk_"


class Kind(enum.Enum):
    """What sort of credential was resolved."""

    API_KEY = "api key"
    """An opaque service credential. It does not expire and cannot be renewed."""
    SESSION = "session"
    """A signed-in user's access token, ordinarily backed by a refresh token."""

    def __str__(self) -> str:
        return self.value


class Origin(enum.Enum):
    """Where a resolved credential came from."""

    EXPLICIT = "explicit configuration"
    ENVIRONMENT = f"${ENV_API_KEY}"
    FILE = "stored session"

    def __str__(self) -> str:
        return self.value


def _kind_of(credential: str) -> Kind:
    return Kind.API_KEY if credential.startswith(_API_KEY_PREFIX) else Kind.SESSION


# --- The stored session -------------------------------------------------------


@dataclasses.dataclass
class Session:
    """A stored login, persisted as JSON at :func:`session_path`.

    The shape is fixed by the file ``jnh`` writes: field names, their order, the
    two-space indent and the absence of a trailing newline are all part of the
    format, because every client reads and writes the same file and a session
    written by one must load in another with every field intact.
    """

    endpoint: str = ""
    """The address the tokens were obtained from. Provenance only: it is NOT where
    this client connects, because the platform publishes more than one front door
    and the address a client using one recorded is unreachable for another."""
    access_token: str = ""
    refresh_token: str = ""
    token_type: str = ""
    expires_at: int = 0
    """When ``access_token`` expires, in unix seconds. Advisory: zero means
    unknown, which is not the same as expired."""

    def expired(self, now: Optional[float] = None) -> bool:
        if not self.expires_at:
            return False
        return (time.time() if now is None else now) > self.expires_at

    def renewable(self) -> bool:
        return bool(self.refresh_token)


_FIELDS = [f.name for f in dataclasses.fields(Session)]


def session_path() -> str:
    """The stored session's location, one per user account.

    Derived from the per-user configuration directory exactly as Go's
    ``os.UserConfigDir`` derives it, because ``jnh`` and the Go SDK find the file
    that way and all three must agree on one location. On Linux that honors
    ``XDG_CONFIG_HOME``, so relocating the config directory relocates the session.
    """
    if sys.platform == "win32":
        base = os.environ.get("APPDATA", "")
        if not base:
            raise NoCredentialError("%AppData% is not defined")
    elif sys.platform == "darwin":
        base = os.path.join(os.path.expanduser("~"), "Library", "Application Support")
    else:
        base = os.environ.get("XDG_CONFIG_HOME", "")
        if not base:
            home = os.environ.get("HOME", "")
            if not home:
                raise NoCredentialError("neither $XDG_CONFIG_HOME nor $HOME is defined")
            base = os.path.join(home, ".config")
        elif not os.path.isabs(base):
            raise NoCredentialError("path in $XDG_CONFIG_HOME is relative")
    return os.path.join(base, "jennah", "credentials")


def _parse(path: str, data: bytes) -> Session:
    try:
        obj = json.loads(data)
    except (ValueError, UnicodeDecodeError) as e:
        raise CorruptSessionError(path, e) from None
    if not isinstance(obj, dict):
        raise CorruptSessionError(path, "not a JSON object")
    values = {}
    for name in _FIELDS:
        v = obj.get(name)
        if v is None:
            continue
        want = int if name == "expires_at" else str
        if type(v) is not want:  # bool is an int subclass; Go refuses it too
            raise CorruptSessionError(path, f"{name} is not a {want.__name__}")
        values[name] = v
    return Session(**values)


def load_session() -> Optional[Session]:
    """Read the stored session, or return None if there is none.

    A file that will not parse raises :class:`CorruptSessionError`. Every other
    read failure (a permission problem, an unreadable directory) is raised as
    itself, because silently treating it as "not logged in" would hide it.
    """
    path = session_path()
    try:
        with open(path, "rb") as f:
            data = f.read()
    except FileNotFoundError:
        return None
    return _parse(path, data)


def _encode(s: Session) -> bytes:
    text = json.dumps(
        {name: getattr(s, name) for name in _FIELDS}, indent=2, ensure_ascii=False
    )
    # Go's encoder escapes these; matching it keeps a load and re-save through
    # any client byte-identical.
    for ch, esc in (("<", "\\u003c"), (">", "\\u003e"), ("&", "\\u0026"),
                    (" ", "\\u2028"), (" ", "\\u2029")):
        text = text.replace(ch, esc)
    return text.encode("utf-8")


def save_session(s: Session) -> None:
    """Write the session, replacing any previous one, atomically and owner-only.

    The content lands in a uniquely named temporary file in the same directory, is
    flushed to disk, and is then renamed over the target, so a concurrent reader
    sees either the whole previous session or the whole new one. The temporary
    name is unique per write, because two processes renewing at once would
    otherwise write through each other's half-finished file.
    """
    path = session_path()
    directory = os.path.dirname(path)
    os.makedirs(directory, mode=0o700, exist_ok=True)
    data = _encode(s)
    # mkstemp creates the file 0600, the mode the session must end up with.
    fd, tmp = tempfile.mkstemp(prefix="credentials-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            # A rename that beats its own content to disk would lose a rotated
            # refresh token on a crash, and a lost rotation cannot be renewed.
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def delete_session() -> None:
    """Remove the stored session. Already absent is not an error."""
    try:
        os.remove(session_path())
    except FileNotFoundError:
        pass


# --- Resolution ---------------------------------------------------------------


@dataclasses.dataclass
class Resolved:
    """The outcome of :func:`resolve`."""

    credential: str = dataclasses.field(repr=False)
    kind: Kind
    origin: Origin
    session: Optional[Session] = dataclasses.field(default=None, repr=False)
    """The stored session the credential came from, or None."""


def resolve(explicit: Optional[str] = None) -> Resolved:
    """Return the first credential that one of the ordered sources yields.

    The explicit credential, then ``$JENNAH_API_KEY``, then the stored session.
    First match wins and later sources are not consulted, so an explicit
    credential works on a machine that has never logged in, and a missing or
    unreadable session file cannot fail a caller who supplied a key.
    """
    c = (explicit or "").strip()
    if c:
        return Resolved(c, _kind_of(c), Origin.EXPLICIT)
    c = os.environ.get(ENV_API_KEY, "").strip()
    if c:
        return Resolved(c, _kind_of(c), Origin.ENVIRONMENT)

    sess = load_session()
    if sess is None:
        raise NoCredentialError()
    if not sess.access_token.strip():
        raise NoCredentialError("the stored session holds no access token")
    return Resolved(sess.access_token, _kind_of(sess.access_token), Origin.FILE, sess)


# --- Sources ------------------------------------------------------------------


@dataclasses.dataclass
class Renewal:
    """What a renewer returns. ``refresh_token`` is part of it because renewal
    rotates: ignoring it would leave the session holding a dead token."""

    access_token: str
    refresh_token: str
    expires_at: int


Renewer = Callable[[str], Renewal]
AsyncRenewer = Callable[[str], Awaitable[Renewal]]


class CredentialSource(Protocol):
    """Supplies the bearer to present, per call rather than once.

    A long-lived client outlives an access token, so a credential captured at
    construction would be stale hours later however often it had been renewed.
    A client asks its source on every call. ``renew`` may be a coroutine function
    for sources used by :class:`jennah.AsyncClient`.
    """

    kind: Kind
    origin: Origin

    def token(self) -> str: ...

    def renewable(self) -> bool: ...

    def renew(self, presented: str) -> Union[str, Awaitable[str]]: ...


class StaticSource:
    """A credential that never changes: an API key, or a token this client will
    not renew. Usable by both clients."""

    def __init__(self, credential: str, origin: Origin) -> None:
        self._credential = credential
        self.kind = _kind_of(credential)
        self.origin = origin

    def __repr__(self) -> str:
        return f"StaticSource(kind={self.kind!s}, origin={self.origin!s})"

    def token(self) -> str:
        return self._credential

    def renewable(self) -> bool:
        return False

    def renew(self, presented: str) -> str:
        if self.kind is Kind.API_KEY:
            raise CredentialRefusedError()
        raise SessionExpiredError()


class _SessionState:
    """The state and decisions shared by the sync and async session sources.

    Renewal spends a rotation only when it has to. Before renewing it checks
    whether the credential already moved on, in this process (a concurrent call
    renewed while this one waited) or on disk (another process did). After a
    failed renewal it checks the disk once more, because a rotation lost to
    another process looks exactly like a failure.
    """

    kind = Kind.SESSION

    def __init__(self, session: Session, origin: Origin) -> None:
        self._session = dataclasses.replace(session)
        self.origin = origin

    def __repr__(self) -> str:
        return f"{type(self).__name__}(kind={self.kind!s}, origin={self.origin!s})"

    def token(self) -> str:
        return self._session.access_token

    def _already_replaced(self, presented: str) -> Optional[str]:
        # Someone in this process got there first while we waited for the lock.
        if self._session.access_token and self._session.access_token != presented:
            return self._session.access_token
        return self._adopt_from_disk(presented)

    def _adopt_from_disk(self, presented: str) -> Optional[str]:
        # Opportunistic: a read failure means "nobody else renewed", and the
        # caller has its own path for that.
        if self.origin is not Origin.FILE:
            return None
        try:
            stored = load_session()
        except Exception:
            return None
        if stored is None or not stored.access_token or stored.access_token == presented:
            return None
        self._session = stored
        return stored.access_token

    def _apply(self, renewed: Optional[Renewal]) -> str:
        if renewed is None or not renewed.access_token:
            raise SessionExpiredError()
        self._session.access_token = renewed.access_token
        if renewed.refresh_token:
            self._session.refresh_token = renewed.refresh_token
        self._session.expires_at = renewed.expires_at
        # Publish before relying on it: the token just spent is dead, so a renewal
        # kept in memory leaves every other reader of the file unable to renew.
        if self.origin is Origin.FILE:
            try:
                save_session(self._session)
            except OSError as e:
                raise SessionPersistError(
                    f"jennah: renewed the session but could not write it back to "
                    f"{session_path()}: {e}"
                ) from e
        return self._session.access_token


class SessionSource(_SessionState):
    """A renewable session for :class:`jennah.Client`. Thread-safe, and its
    renewals are single-flight: concurrent rejections produce one rotation."""

    def __init__(self, session: Session, origin: Origin, renewer: Optional[Renewer] = None) -> None:
        super().__init__(session, origin)
        self._lock = threading.Lock()
        self._renewer = renewer

    def set_renewer(self, renewer: Renewer) -> None:
        self._renewer = renewer

    def token(self) -> str:
        with self._lock:
            return self._session.access_token

    def renewable(self) -> bool:
        return self._renewer is not None and self._session.renewable()

    def renew(self, presented: str) -> str:
        with self._lock:
            replaced = self._already_replaced(presented)
            if replaced:
                return replaced
            if self._renewer is None or not self._session.refresh_token:
                raise SessionExpiredError()
            try:
                renewed = self._renewer(self._session.refresh_token)
            except Exception as e:
                adopted = self._adopt_from_disk(presented)
                if adopted:
                    return adopted
                raise SessionExpiredError(str(e)) from e
            return self._apply(renewed)


class AsyncSessionSource(_SessionState):
    """A renewable session for :class:`jennah.AsyncClient`. Its renewals are
    single-flight within one event loop, and the renewal call is awaited, never
    run on a thread."""

    def __init__(self, session: Session, origin: Origin, renewer: Optional[AsyncRenewer] = None) -> None:
        super().__init__(session, origin)
        self._lock: Optional[asyncio.Lock] = None
        self._renewer = renewer

    def set_renewer(self, renewer: AsyncRenewer) -> None:
        self._renewer = renewer

    def renewable(self) -> bool:
        return self._renewer is not None and self._session.renewable()

    async def renew(self, presented: str) -> str:
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            replaced = self._already_replaced(presented)
            if replaced:
                return replaced
            if self._renewer is None or not self._session.refresh_token:
                raise SessionExpiredError()
            try:
                renewed = await self._renewer(self._session.refresh_token)
            except Exception as e:
                adopted = self._adopt_from_disk(presented)
                if adopted:
                    return adopted
                raise SessionExpiredError(str(e)) from e
            return self._apply(renewed)
