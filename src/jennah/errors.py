"""Errors the SDK raises itself, and how to read a platform status out of any error.

Calls through the SDK raise ``grpc.RpcError`` for the platform's answers. The
classes here are for the conditions the SDK decides on its own: no credential,
an unreadable session file, a session that cannot be renewed, a refused key.
When one of them is raised because of a platform rejection, the rejection is
kept as its ``__cause__``, so :func:`code` still reports ``UNAUTHENTICATED`` for
it and a caller branching on the status keeps working.
"""

from __future__ import annotations

import grpc

ENV_API_KEY = "JENNAH_API_KEY"


class JennahError(Exception):
    """Base class for errors raised by the SDK itself."""


class NoCredentialError(JennahError):
    """No source yielded a credential.

    Raised when the client is constructed, not on its first call, and names every
    way to supply one, because a caller seeing it has not chosen between them.
    """

    def __init__(self, detail: str = "") -> None:
        msg = (
            f"jennah: no credential found (set one explicitly, set ${ENV_API_KEY}, "
            "or run: jnh login)"
        )
        super().__init__(f"{msg}: {detail}" if detail else msg)


class CorruptSessionError(JennahError):
    """The stored session exists but cannot be interpreted.

    Distinct from having no session: the machine is logged in and something ate
    the file, so it is reported with its path rather than treated as logged out.
    """

    def __init__(self, path: str, reason: object) -> None:
        self.path = path
        super().__init__(f"jennah: stored session at {path} is unreadable: {reason}")


class SessionExpiredError(JennahError):
    """The session can no longer authenticate and could not be renewed."""

    def __init__(self, detail: str = "") -> None:
        msg = "jennah: the stored session has expired (run: jnh login)"
        super().__init__(f"{msg}: {detail}" if detail else msg)


class CredentialRefusedError(JennahError):
    """A non-renewable credential (an API key) was rejected.

    A key does not expire, so this is the key itself being refused: revoked,
    expired at the server, or wrong. It is never a reason to attempt a renewal.
    """

    def __init__(self) -> None:
        super().__init__(
            f"jennah: the API key was refused (check the configured key or ${ENV_API_KEY})"
        )


class SessionPersistError(JennahError):
    """A renewed session could not be written back to the shared location.

    Raised rather than ignored: the renewal spent the previous refresh token, so a
    session held only in memory is lost when the process exits and leaves every
    other client on the machine holding a token nothing accepts.
    """


def code(err: BaseException | None) -> grpc.StatusCode | None:
    """Return the platform status carried by ``err``, or None if it carries none.

    Follows the exception's cause chain, so an SDK error raised because of a
    platform rejection reports that rejection's status.
    """
    seen = set()
    while err is not None and id(err) not in seen:
        seen.add(id(err))
        if isinstance(err, grpc.RpcError) and callable(getattr(err, "code", None)):
            return err.code()
        err = err.__cause__ or err.__context__
    return None


def is_unauthenticated(err: BaseException | None) -> bool:
    return code(err) == grpc.StatusCode.UNAUTHENTICATED


def is_transient(err: BaseException | None) -> bool:
    """Whether ``err`` is the transient status the SDK's own retry acts on."""
    return code(err) == grpc.StatusCode.UNAVAILABLE
