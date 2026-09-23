"""The interceptors every call passes through: transport retry outside, credential
inside, for both the synchronous and the asynchronous channel.

Retry wraps the credential interceptor, not the other way round, so a replay
after a transport failure asks the source for the credential again instead of
reusing the one the failed attempt carried.
"""

from __future__ import annotations

import collections
import contextvars
import dataclasses
import inspect
import random
import time
from typing import Optional

import grpc
import grpc.aio

from ._retry import RETRYABLE_CODES, safe_to_replay
from .credentials import CredentialSource
from .errors import JennahError

# Marks the one call the credential interceptor must keep its hands off: the
# renewal it is itself performing. It carries no bearer, because the refresh
# method is on the platform's unauthenticated allowlist and the refresh token in
# the body is the credential; and it must not renew, because that would re-enter
# the source it is already inside (a deadlock on the first attempt, unbounded
# recursion if the refresh token is what was rejected).
#
# A context variable rather than a second channel: an asyncio call runs its
# interceptors in a task that copies the context current when the call was made,
# so the mark reaches them either way.
renewing: contextvars.ContextVar[bool] = contextvars.ContextVar("jennah_renewing", default=False)


@dataclasses.dataclass(frozen=True)
class RetryPolicy:
    """Automatic retries after a transient failure.

    The default retries a call up to three times in total, 100ms apart doubling to
    2s with jitter, only when the failure is UNAVAILABLE, and only when replaying
    the call cannot produce a second effect (see ``jennah._retry``). A call's
    ``timeout`` bounds all of its attempts together, not each one.
    """

    disabled: bool = False
    max_attempts: int = 3
    """Counts the first try; 1 means no retry."""
    base_backoff: float = 0.1
    max_backoff: float = 2.0


def _method_name(details) -> str:
    m = details.method
    return m.decode() if isinstance(m, bytes) else m


def _bearer_metadata(metadata, token: str) -> list:
    md = [(k, v) for k, v in (metadata or ()) if k != "authorization"]
    md.append(("authorization", f"Bearer {token}"))
    return md


def _chain_rejection(err: JennahError, rejection: Optional[BaseException]) -> JennahError:
    # Keep the platform's UNAUTHENTICATED reachable (errors.code follows causes)
    # when the source raised without a cause of its own.
    if err.__cause__ is None and rejection is not None:
        err.__cause__ = rejection
    return err


def _backoffs(policy: RetryPolicy):
    ceiling = policy.base_backoff
    while True:
        yield random.uniform(0, ceiling)
        ceiling = min(ceiling * 2, policy.max_backoff)


# --- Synchronous --------------------------------------------------------------


class _CallDetails(
    collections.namedtuple(
        "_CallDetails", ("method", "timeout", "metadata", "credentials", "wait_for_ready", "compression")
    ),
    grpc.ClientCallDetails,
):
    pass


def _details(details, *, metadata=None, timeout=None) -> _CallDetails:
    return _CallDetails(
        details.method,
        details.timeout if timeout is None else timeout,
        details.metadata if metadata is None else metadata,
        details.credentials,
        getattr(details, "wait_for_ready", None),
        getattr(details, "compression", None),
    )


class BearerInterceptor(grpc.UnaryUnaryClientInterceptor):
    """Attaches the credential to every call, asking the source each time, and on
    an UNAUTHENTICATED answer renews once and reissues the call exactly once.

    The reissue asks for none of the evidence a transport retry demands: a call
    refused before it reached the operation had no effect, so even a write that is
    never otherwise replayed is safe to send again.
    """

    def __init__(self, source: CredentialSource) -> None:
        self._source = source

    def intercept_unary_unary(self, continuation, details, request):
        if renewing.get():
            return continuation(details, request)
        cred = self._source.token()
        outcome = continuation(_details(details, metadata=_bearer_metadata(details.metadata, cred)), request)
        if outcome.code() != grpc.StatusCode.UNAUTHENTICATED:
            return outcome
        try:
            renewed = self._source.renew(cred)
        except JennahError as e:
            raise _chain_rejection(e, outcome.exception()) from e.__cause__
        # Exactly once: a second rejection is the platform's answer.
        return continuation(_details(details, metadata=_bearer_metadata(details.metadata, renewed)), request)


class RetryInterceptor(grpc.UnaryUnaryClientInterceptor):
    def __init__(self, policy: RetryPolicy) -> None:
        self._policy = policy

    def intercept_unary_unary(self, continuation, details, request):
        p = self._policy
        if p.max_attempts < 2 or not safe_to_replay(_method_name(details), request):
            return continuation(details, request)
        deadline = None if details.timeout is None else time.monotonic() + details.timeout
        pauses = _backoffs(p)
        for attempt in range(1, p.max_attempts + 1):
            remaining = None if deadline is None else max(deadline - time.monotonic(), 0)
            outcome = continuation(_details(details, timeout=remaining), request)
            if attempt == p.max_attempts or outcome.code() not in RETRYABLE_CODES:
                return outcome
            pause = next(pauses)
            if deadline is not None and time.monotonic() + pause >= deadline:
                return outcome
            time.sleep(pause)
        return outcome  # unreachable


# --- Asynchronous -------------------------------------------------------------


def _aio_details(details, *, metadata=None, timeout=None) -> grpc.aio.ClientCallDetails:
    return grpc.aio.ClientCallDetails(
        method=details.method,
        timeout=details.timeout if timeout is None else timeout,
        metadata=details.metadata if metadata is None else grpc.aio.Metadata(*metadata),
        credentials=details.credentials,
        wait_for_ready=details.wait_for_ready,
    )


async def _rejection_of(call) -> Optional[BaseException]:
    try:
        await call
    except grpc.aio.AioRpcError as e:
        return e
    return None


class AsyncBearerInterceptor(grpc.aio.UnaryUnaryClientInterceptor):
    """The asynchronous counterpart of :class:`BearerInterceptor`. The renewal is
    awaited on the event loop; nothing here runs on a thread."""

    def __init__(self, source: CredentialSource) -> None:
        self._source = source

    async def intercept_unary_unary(self, continuation, details, request):
        if renewing.get():
            return await continuation(details, request)
        cred = self._source.token()
        call = await continuation(_aio_details(details, metadata=_bearer_metadata(details.metadata, cred)), request)
        if await call.code() != grpc.StatusCode.UNAUTHENTICATED:
            return call
        rejection = await _rejection_of(call)
        try:
            renewed = self._source.renew(cred)
            if inspect.isawaitable(renewed):
                renewed = await renewed
        except JennahError as e:
            raise _chain_rejection(e, rejection) from e.__cause__
        return await continuation(_aio_details(details, metadata=_bearer_metadata(details.metadata, renewed)), request)


class AsyncRetryInterceptor(grpc.aio.UnaryUnaryClientInterceptor):
    def __init__(self, policy: RetryPolicy) -> None:
        self._policy = policy

    async def intercept_unary_unary(self, continuation, details, request):
        import asyncio

        p = self._policy
        if p.max_attempts < 2 or not safe_to_replay(_method_name(details), request):
            return await continuation(details, request)
        deadline = None if details.timeout is None else time.monotonic() + details.timeout
        pauses = _backoffs(p)
        for attempt in range(1, p.max_attempts + 1):
            remaining = None if deadline is None else max(deadline - time.monotonic(), 0)
            call = await continuation(_aio_details(details, timeout=remaining), request)
            if attempt == p.max_attempts or await call.code() not in RETRYABLE_CODES:
                return call
            pause = next(pauses)
            if deadline is not None and time.monotonic() + pause >= deadline:
                return call
            await asyncio.sleep(pause)
        return call  # unreachable
