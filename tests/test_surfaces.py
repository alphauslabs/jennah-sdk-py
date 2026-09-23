"""The two surfaces, the generated stubs they hand out, and the conveniences.

The credential contract itself is certified by test_conformance.py; these tests
cover what client-sdks adds on top: both surfaces expose the same operations,
the asynchronous one never blocks its event loop, the conveniences are the
single-section wrappers unified-memory permits and nothing more, and a call
through a generated stub carries the credential contract exactly as a
convenience call does.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import pkgutil
import threading
import time
from concurrent import futures

import grpc
import pytest

import jennah
from jennah import credentials
from jennah.agent.v1 import agent_pb2, agent_pb2_grpc, memory_pb2, memory_pb2_grpc, scope_pb2, scope_pb2_grpc
from jennah.auth.v1 import auth_pb2, auth_pb2_grpc

# --- A recording fake platform ------------------------------------------------


def _bearer(context) -> str:
    for k, v in context.invocation_metadata():
        if k == "authorization":
            return v.removeprefix("Bearer ")
    return ""


class Recorder(
    agent_pb2_grpc.AgentServiceServicer,
    memory_pb2_grpc.MemoryServiceServicer,
    scope_pb2_grpc.ScopeServiceServicer,
    auth_pb2_grpc.AuthServiceServicer,
):
    """Accepts one bearer, rotates on refresh, records every business call, and
    can make a method slow, to show what the caller's event loop does meanwhile."""

    def __init__(self, accept: str, refresh_accept: str = "", delay: float = 0.0, refresh_delay: float = 0.0):
        self.mu = threading.Lock()
        self.accept = accept
        self.refresh_accept = refresh_accept
        self.delay = delay
        self.refresh_delay = refresh_delay
        self.calls: list[tuple[str, object, str]] = []
        self.refreshes = 0

    def _business(self, name, request, context):
        with self.mu:
            got = _bearer(context)
            self.calls.append((name, request, got))
            ok = got == self.accept
        if not ok:
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "the access token is invalid or has expired")
        if self.delay:
            time.sleep(self.delay)

    def ListAgents(self, request, context):
        self._business("ListAgents", request, context)
        return agent_pb2.ListAgentsResponse()

    def CommitMemory(self, request, context):
        self._business("CommitMemory", request, context)
        return memory_pb2.CommitMemoryResponse()

    def QueryMemory(self, request, context):
        self._business("QueryMemory", request, context)
        return memory_pb2.QueryMemoryResponse()

    def ListScopes(self, request, context):
        self._business("ListScopes", request, context)
        return scope_pb2.ListScopesResponse()

    def RefreshToken(self, request, context):
        if self.refresh_delay:
            time.sleep(self.refresh_delay)
        with self.mu:
            if request.refresh_token != self.refresh_accept:
                context.abort(grpc.StatusCode.UNAUTHENTICATED, "refused")
            self.refreshes += 1
            self.accept = f"at_renewed_{self.refreshes}"
            self.refresh_accept = f"rt_rotated_{self.refreshes}"
            return auth_pb2.RefreshTokenResponse(
                access_token=self.accept, refresh_token=self.refresh_accept, expires_in=3600
            )

    def business_calls(self):
        with self.mu:
            return list(self.calls)


@pytest.fixture
def machine(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(jennah.errors.ENV_API_KEY, raising=False)
    return tmp_path


@pytest.fixture
def platform():
    servers = []

    def start(rec: Recorder) -> str:
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
        agent_pb2_grpc.add_AgentServiceServicer_to_server(rec, server)
        memory_pb2_grpc.add_MemoryServiceServicer_to_server(rec, server)
        scope_pb2_grpc.add_ScopeServiceServicer_to_server(rec, server)
        auth_pb2_grpc.add_AuthServiceServicer_to_server(rec, server)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        servers.append(server)
        return f"127.0.0.1:{port}"

    yield start
    for s in servers:
        s.stop(None)


def expired_session():
    credentials.save_session(credentials.Session(
        endpoint="https://jennah.alphaus.cloud", access_token="at_expired",
        refresh_token="rt_valid", token_type="Bearer", expires_at=int(time.time()) - 60,
    ))


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


# --- Both surfaces expose the same operations (4.1, 4.3, 4.6) -----------------


def published_services() -> dict[str, set[str]]:
    out = {}
    for info in pkgutil.walk_packages(jennah.__path__, "jennah."):
        if info.name.endswith("_pb2"):
            for svc in importlib.import_module(info.name).DESCRIPTOR.services_by_name.values():
                out[svc.full_name] = {m.name for m in svc.methods}
    return out


def bound_services(client) -> dict[str, set[str]]:
    """Which service each public stub attribute serves, and its callable RPCs."""
    out = {}
    for name, value in vars(client).items():
        if name.startswith("_") or not type(value).__name__.endswith("ServiceStub"):
            continue
        module = importlib.import_module(type(value).__module__.replace("_pb2_grpc", "_pb2"))
        svc = next(s for s in module.DESCRIPTOR.services_by_name.values()
                   if f"{s.name}Stub" == type(value).__name__)
        out[svc.full_name] = {m for m in svc.methods_by_name if callable(getattr(value, m, None))}
    return out


def test_every_published_service_is_bound_on_both_surfaces(machine):
    published = published_services()
    sync = jennah.Client(api_key="jennah_sk_x", endpoint="127.0.0.1:1", insecure=True)

    async def make():
        return jennah.AsyncClient(api_key="jennah_sk_x", endpoint="127.0.0.1:1", insecure=True)

    loop = asyncio.new_event_loop()
    aclient = loop.run_until_complete(make())
    try:
        assert bound_services(sync) == published, "the sync client must bind every published service"
        assert bound_services(aclient) == published, "the async client must bind every published service"
    finally:
        sync.close()
        loop.run_until_complete(aclient.close())
        loop.close()


def test_conveniences_match_across_surfaces():
    def shape(cls_pairs):
        return {name: sorted(n for n, _ in inspect.getmembers(cls, inspect.isfunction) if not n.startswith("_"))
                for name, cls in cls_pairs}

    from jennah import memory as m

    sync = shape([("memory", m.Memory), ("logs", m._Logs), ("vectors", m._Vectors), ("graph", m._Graph)])
    aio = shape([("memory", m.AsyncMemory), ("logs", m._AsyncLogs), ("vectors", m._AsyncVectors),
                 ("graph", m._AsyncGraph)])
    assert sync == aio
    for cls in (m.AsyncMemory, m._AsyncLogs, m._AsyncVectors, m._AsyncGraph):
        for name, fn in inspect.getmembers(cls, inspect.isfunction):
            if not name.startswith("_"):
                assert inspect.iscoroutinefunction(fn), f"{cls.__name__}.{name} must be awaited, not blocking"


# --- The asynchronous surface never blocks its loop (4.2) ---------------------


async def _longest_stall_during(awaitable) -> float:
    """The longest the event loop went without running a 10ms ticker while
    ``awaitable`` was in flight. Any blocking stretch shows up here directly,
    however much of the rest of the call was properly awaited."""
    stop = asyncio.Event()
    longest = 0.0

    async def ticker():
        nonlocal longest
        last = time.monotonic()
        while not stop.is_set():
            await asyncio.sleep(0.01)
            now = time.monotonic()
            longest = max(longest, now - last)
            last = now

    t = asyncio.ensure_future(ticker())
    try:
        await awaitable
    finally:
        stop.set()
        await t
    return longest


def test_async_call_and_renewal_do_not_block_the_loop(machine, platform):
    expired_session()
    # A slow renewal AND a slow reissued call, 0.3s each: both must be awaited.
    target = platform(Recorder(accept="at_current", refresh_accept="rt_valid", delay=0.3, refresh_delay=0.3))

    async def go():
        async with jennah.AsyncClient(endpoint=target, insecure=True) as c:
            return await _longest_stall_during(c.agents.ListAgents(agent_pb2.ListAgentsRequest(), timeout=10))

    stall = run(go())
    # Either 0.3s wait, done inline, would stall the loop for at least 0.3s.
    assert stall < 0.2, f"the event loop stalled for {stall:.2f}s during the call"


# --- Conveniences are single-section unified calls (4.4, 4.5) -----------------

CONVENIENCES = [
    # (label, how to invoke on an agent handle, expected RPC, expected single section)
    ("logs.create", lambda a: a.logs.create(memory_pb2.ExecutionLogStep(step_id="s1")), "CommitMemory", "log"),
    ("logs.recent", lambda a: a.logs.recent(memory_pb2.LogQuery(limit=5)), "QueryMemory", "log"),
    ("vectors.upsert", lambda a: a.vectors.upsert(memory_pb2.VectorChunk(chunk_id="c1")), "CommitMemory", "vectors"),
    ("vectors.search", lambda a: a.vectors.search(memory_pb2.SemanticQuery(query_text="q")), "QueryMemory", "semantic"),
    ("graph.write", lambda a: a.graph.write(memory_pb2.GraphWrite(nodes=[memory_pb2.GraphNode(node_id="n1")])),
     "CommitMemory", "graph"),
    ("graph.query", lambda a: a.graph.query(memory_pb2.GraphQuery(start=memory_pb2.GraphNodeMatch(label="L"))),
     "QueryMemory", "graph"),
]


def _populated(msg) -> set[str]:
    return {f.name for f, _ in msg.ListFields()} - {"agent_instance_id"}


@pytest.mark.parametrize("surface", ["sync", "async"])
@pytest.mark.parametrize("label,invoke,rpc,section", CONVENIENCES, ids=[c[0] for c in CONVENIENCES])
def test_convenience_is_one_single_section_unified_call(machine, platform, surface, label, invoke, rpc, section):
    rec = Recorder(accept="jennah_sk_k")
    target = platform(rec)
    if surface == "sync":
        with jennah.Client(api_key="jennah_sk_k", endpoint=target, insecure=True) as c:
            invoke(c.agent("a1"))
    else:
        async def go():
            async with jennah.AsyncClient(api_key="jennah_sk_k", endpoint=target, insecure=True) as c:
                await invoke(c.agent("a1"))
        run(go())

    calls = rec.business_calls()
    assert [name for name, _, _ in calls] == [rpc], f"{label} must issue exactly one {rpc} and nothing else"
    request = calls[0][1]
    assert request.agent_instance_id == "a1"
    assert _populated(request) == {section}, f"{label} must set only the {section} section"


def test_hand_written_modules_call_only_the_unified_rpcs():
    # Structural backstop for the test above: the convenience module names no RPC
    # but the two unified ones, so no future method can quietly reach another.
    import re

    from jennah import memory as m

    source = inspect.getsource(m)
    stub_calls = set(re.findall(r"_stub\.(\w+)\(", source))
    assert stub_calls == {"CommitMemory", "QueryMemory"}, stub_calls


# --- The generated surface carries the credential contract (4.7, 4.8, 4.9) ----


@pytest.mark.parametrize("surface", ["sync", "async"])
@pytest.mark.parametrize("path", ["generated", "convenience"])
def test_renewal_is_identical_through_generated_and_convenience_calls(machine, platform, surface, path):
    expired_session()
    rec = Recorder(accept="at_current", refresh_accept="rt_valid")
    target = platform(rec)

    def call(c):
        if path == "generated":
            return c.memory.CommitMemory(memory_pb2.CommitMemoryRequest(
                agent_instance_id="a1", vectors=[memory_pb2.VectorChunk(chunk_id="c1")]))
        return c.agent("a1").vectors.upsert(memory_pb2.VectorChunk(chunk_id="c1"))

    if surface == "sync":
        with jennah.Client(endpoint=target, insecure=True) as c:
            call(c)
            call(c)
    else:
        async def go():
            async with jennah.AsyncClient(endpoint=target, insecure=True) as c:
                await call(c)
                await call(c)
        run(go())

    bearers = [b for _, _, b in rec.business_calls()]
    # Rejected, renewed once, reissued, and the next call presents the renewal:
    # per-call resolution, renew-once-and-retry, exactly as the suite requires.
    assert bearers == ["at_expired", "at_renewed_1", "at_renewed_1"]
    assert rec.refreshes == 1
    stored = credentials.load_session()
    assert (stored.access_token, stored.refresh_token) == ("at_renewed_1", "rt_rotated_1"), (
        "the rotated session must be written back"
    )


@pytest.mark.parametrize("surface", ["sync", "async"])
def test_operation_without_a_convenience_is_reachable_and_credentialed(machine, platform, surface):
    # ScopeService has no convenience method. Reaching it takes no channel and no
    # credential from the caller, through the named attribute or through stub().
    rec = Recorder(accept="jennah_sk_k")
    target = platform(rec)
    req = scope_pb2.ListScopesRequest()
    if surface == "sync":
        with jennah.Client(api_key="jennah_sk_k", endpoint=target, insecure=True) as c:
            c.scopes.ListScopes(req)
            c.stub(scope_pb2_grpc.ScopeServiceStub).ListScopes(req)
    else:
        async def go():
            async with jennah.AsyncClient(api_key="jennah_sk_k", endpoint=target, insecure=True) as c:
                await c.scopes.ListScopes(req)
                await c.stub(scope_pb2_grpc.ScopeServiceStub).ListScopes(req)
        run(go())
    assert [(n, b) for n, _, b in rec.business_calls()] == [("ListScopes", "jennah_sk_k")] * 2
