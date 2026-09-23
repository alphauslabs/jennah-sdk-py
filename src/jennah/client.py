"""The synchronous and asynchronous clients.

Both hand out the generated service stubs already bound to a credentialed
channel. That is the whole point of how they are built: the proto is the
contract, so every operation it expresses is callable the moment the SDK is
regenerated, and reaching one never involves building a channel or attaching a
credential by hand, which would silently opt a caller out of per-call
resolution, renewal, and publishing a rotated session.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Optional, Sequence, Tuple, Type, TypeVar

import grpc
import grpc.aio

from ._transport import (
    AsyncBearerInterceptor,
    AsyncRetryInterceptor,
    BearerInterceptor,
    RetryInterceptor,
    RetryPolicy,
    renewing,
)
from .agent.v1 import agent_pb2_grpc, memory_pb2_grpc, scope_pb2_grpc
from .approval.v1 import approval_pb2_grpc
from .auth.v1 import auth_pb2, auth_pb2_grpc
from .billing.v1 import billing_pb2_grpc
from .credentials import (
    AsyncSessionSource,
    CredentialSource,
    Kind,
    Origin,
    Renewal,
    SessionSource,
    StaticSource,
    resolve,
)
from .datastore.v1 import data_pb2_grpc, dataset_pb2_grpc, schema_pb2_grpc
from .errors import SessionExpiredError
from .memory import Agent, AsyncAgent
from .platform.v1 import platform_pb2_grpc

DEFAULT_ENDPOINT = "jennah-grpc.alphaus.cloud:443"
"""The platform's public gRPC front door. ``jennah.alphaus.cloud`` is the HTTP
gateway and cannot answer a gRPC call."""

# How long a renewal may take. It runs underneath a caller's own call, which has
# its own deadline; this only stops a hung renewal from hanging forever.
_RENEWAL_TIMEOUT = 30.0

S = TypeVar("S")


@dataclasses.dataclass(frozen=True)
class CredentialInfo:
    """What a client authenticates with and where it came from. Never the
    credential itself, so every field and its rendering are safe to log."""

    kind: Kind
    origin: Origin

    def __str__(self) -> str:
        return f"{self.kind} from {self.origin}"


def _source(api_key: Optional[str], credentials: Optional[CredentialSource], session_cls) -> CredentialSource:
    if credentials is not None:
        return credentials
    resolved = resolve(api_key)
    if resolved.session is None:
        # An API key, or an explicit token: nothing to renew with.
        return StaticSource(resolved.credential, resolved.origin)
    # Refused here only when nothing can renew it. A renewable expired session is
    # an ordinary condition its first call resolves.
    if resolved.session.expired() and not resolved.session.renewable():
        raise SessionExpiredError()
    return session_cls(resolved.session, resolved.origin)


def _renewal(resp) -> Renewal:
    return Renewal(
        access_token=resp.access_token,
        refresh_token=resp.refresh_token,
        expires_at=int(time.time()) + int(resp.expires_in),
    )


class _Services:
    """The generated services, bound to one channel. Shared by both clients so the
    two surfaces expose the same operations by construction."""

    def _bind(self, channel) -> None:
        self._channel = channel
        self.agents = agent_pb2_grpc.AgentServiceStub(channel)
        """Agent workspaces (``jennahapi.agent.v1.AgentService``)."""
        self.memory = memory_pb2_grpc.MemoryServiceStub(channel)
        """Commit, query, inspect and form memory (``MemoryService``)."""
        self.scopes = scope_pb2_grpc.ScopeServiceStub(channel)
        """Memory scopes of either kind (``ScopeService``)."""
        self.datasets = dataset_pb2_grpc.DatasetServiceStub(channel)
        self.schema = schema_pb2_grpc.SchemaServiceStub(channel)
        self.data = data_pb2_grpc.DataServiceStub(channel)
        self.auth = auth_pb2_grpc.AuthServiceStub(channel)
        self.approvals = approval_pb2_grpc.ApprovalServiceStub(channel)
        self.billing = billing_pb2_grpc.BillingServiceStub(channel)
        self.platform = platform_pb2_grpc.PlatformServiceStub(channel)

    def stub(self, stub_class: Type[S]) -> S:
        """Bind any generated ``*Stub`` class to this client's credentialed channel,
        for a service added to the proto after this SDK version named it."""
        return stub_class(self._channel)

    @property
    def endpoint(self) -> str:
        return self._endpoint

    @property
    def credential(self) -> CredentialInfo:
        return CredentialInfo(self._source.kind, self._source.origin)


class Client(_Services):
    """A synchronous connection to Jennah, scoped to one credential.

    With no ``api_key`` or ``credentials``, the credential is resolved in order
    from ``$JENNAH_API_KEY`` and then the session stored by ``jnh login``, and a
    stored session is renewed underneath the client when it expires. Safe to share
    between threads. Use as a context manager, or call :meth:`close`.

    Every call is a generated stub method, for example
    ``client.agents.ListAgents(agent_pb2.ListAgentsRequest())``.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        credentials: Optional[CredentialSource] = None,
        endpoint: Optional[str] = None,
        insecure: bool = False,
        channel_credentials: Optional[grpc.ChannelCredentials] = None,
        retry: Optional[RetryPolicy] = None,
        options: Sequence[Tuple[str, object]] = (),
    ) -> None:
        self._source = _source(api_key, credentials, SessionSource)
        # The stored session's endpoint is never used: it records the HTTP gateway
        # a CLI login went through, which cannot answer gRPC.
        self._endpoint = endpoint or DEFAULT_ENDPOINT
        if insecure:
            raw = grpc.insecure_channel(self._endpoint, options=list(options))
        else:
            creds = channel_credentials or grpc.ssl_channel_credentials()
            raw = grpc.secure_channel(self._endpoint, creds, options=list(options))
        self._raw = raw
        retry = retry or RetryPolicy()
        interceptors = [] if retry.disabled else [RetryInterceptor(retry)]
        interceptors.append(BearerInterceptor(self._source))
        self._bind(grpc.intercept_channel(raw, *interceptors))

        if isinstance(self._source, SessionSource):
            self._source.set_renewer(self._renew)

    def _renew(self, refresh_token: str) -> Renewal:
        # No enterprise id: this renews in place. Passing one would switch which
        # enterprise the token is scoped to, which is never a side effect of expiry.
        mark = renewing.set(True)
        try:
            resp = self.auth.RefreshToken(
                auth_pb2.RefreshTokenRequest(refresh_token=refresh_token), timeout=_RENEWAL_TIMEOUT
            )
        finally:
            renewing.reset(mark)
        return _renewal(resp)

    def agent(self, agent_instance_id: str) -> Agent:
        """A handle to one agent workspace, carrying the memory conveniences."""
        return Agent(self.memory, agent_instance_id)

    def close(self) -> None:
        self._raw.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class AsyncClient(_Services):
    """An asynchronous connection to Jennah, over ``grpc.aio``.

    The same operations as :class:`Client`, awaited:
    ``await client.agents.ListAgents(agent_pb2.ListAgentsRequest())``. Credential
    renewal is awaited on the event loop too, so nothing blocks it on the network.
    Use as an async context manager, or await :meth:`close`.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        credentials: Optional[CredentialSource] = None,
        endpoint: Optional[str] = None,
        insecure: bool = False,
        channel_credentials: Optional[grpc.ChannelCredentials] = None,
        retry: Optional[RetryPolicy] = None,
        options: Sequence[Tuple[str, object]] = (),
    ) -> None:
        self._source = _source(api_key, credentials, AsyncSessionSource)
        self._endpoint = endpoint or DEFAULT_ENDPOINT
        retry = retry or RetryPolicy()
        interceptors = [] if retry.disabled else [AsyncRetryInterceptor(retry)]
        interceptors.append(AsyncBearerInterceptor(self._source))
        if insecure:
            channel = grpc.aio.insecure_channel(self._endpoint, options=list(options), interceptors=interceptors)
        else:
            creds = channel_credentials or grpc.ssl_channel_credentials()
            channel = grpc.aio.secure_channel(
                self._endpoint, creds, options=list(options), interceptors=interceptors
            )
        self._bind(channel)

        if isinstance(self._source, AsyncSessionSource):
            self._source.set_renewer(self._renew)

    async def _renew(self, refresh_token: str) -> Renewal:
        mark = renewing.set(True)
        try:
            resp = await self.auth.RefreshToken(
                auth_pb2.RefreshTokenRequest(refresh_token=refresh_token), timeout=_RENEWAL_TIMEOUT
            )
        finally:
            renewing.reset(mark)
        return _renewal(resp)

    def agent(self, agent_instance_id: str) -> AsyncAgent:
        """A handle to one agent workspace, carrying the memory conveniences."""
        return AsyncAgent(self.memory, agent_instance_id)

    async def close(self) -> None:
        await self._channel.close()

    async def __aenter__(self) -> "AsyncClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()
