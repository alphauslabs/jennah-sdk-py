"""The per-type memory conveniences, and nothing else hand-written on top of the proto.

``unified-memory`` permits exactly these: each is a thin wrapper that builds ONE
single-section ``CommitMemory`` or ``QueryMemory`` request, and none reaches any
other RPC. Everything else (inspect, supersession, scopes, datasets, ...) is
called through the generated stubs on the client, which carry the same
credential handling, so no operation needs a wrapper to be reachable.

``memory.commit`` and ``memory.query`` accept any field of their request as a
keyword, so a field added to the proto is usable without an SDK change, and the
SDK never defines a request shape of its own.
"""

from __future__ import annotations

from typing import Any

from .agent.v1 import memory_pb2


class _MemoryBase:
    def __init__(self, stub, agent_instance_id: str) -> None:
        self._stub = stub
        self._id = agent_instance_id

    def _commit_request(self, fields: dict) -> memory_pb2.CommitMemoryRequest:
        return memory_pb2.CommitMemoryRequest(agent_instance_id=self._id, **fields)

    def _query_request(self, fields: dict) -> memory_pb2.QueryMemoryRequest:
        return memory_pb2.QueryMemoryRequest(agent_instance_id=self._id, **fields)


class Memory(_MemoryBase):
    """The unified transport for one agent: every section, in one call."""

    def commit(self, **fields: Any) -> memory_pb2.CommitMemoryResponse:
        """``CommitMemory`` with the given request fields (``log``, ``vectors``,
        ``graph``, ``supersessions``, ...), written atomically."""
        return self._stub.CommitMemory(self._commit_request(fields))

    def query(self, **fields: Any) -> memory_pb2.QueryMemoryResponse:
        """``QueryMemory`` with the given request fields (``semantic``, ``graph``,
        ``log``, ``link``, ``as_of``, ...), evaluated over one snapshot."""
        return self._stub.QueryMemory(self._query_request(fields))


class AsyncMemory(_MemoryBase):
    """The unified transport for one agent, awaited."""

    async def commit(self, **fields: Any) -> memory_pb2.CommitMemoryResponse:
        return await self._stub.CommitMemory(self._commit_request(fields))

    async def query(self, **fields: Any) -> memory_pb2.QueryMemoryResponse:
        return await self._stub.QueryMemory(self._query_request(fields))


class Agent:
    """A handle to one agent workspace. No network call happens until a method is
    used, and the workspace need not exist yet."""

    def __init__(self, stub, agent_instance_id: str) -> None:
        self.id = agent_instance_id
        self.memory = Memory(stub, agent_instance_id)
        self.logs = _Logs(self.memory)
        self.vectors = _Vectors(self.memory)
        self.graph = _Graph(self.memory)


class AsyncAgent:
    """An :class:`Agent` whose methods are awaited."""

    def __init__(self, stub, agent_instance_id: str) -> None:
        self.id = agent_instance_id
        self.memory = AsyncMemory(stub, agent_instance_id)
        self.logs = _AsyncLogs(self.memory)
        self.vectors = _AsyncVectors(self.memory)
        self.graph = _AsyncGraph(self.memory)


class _Logs:
    def __init__(self, memory: Memory) -> None:
        self._m = memory

    def create(self, step) -> memory_pb2.CommitMemoryResponse:
        """Commit one execution-log step: the log section of a commit."""
        return self._m.commit(log=step)

    def recent(self, query):
        """The agent's recent log steps: the log section of a query."""
        return self._m.query(log=query).log


class _Vectors:
    def __init__(self, memory: Memory) -> None:
        self._m = memory

    def upsert(self, *chunks) -> memory_pb2.CommitMemoryResponse:
        """Upsert vector chunks: the vector section of a commit."""
        return self._m.commit(vectors=list(chunks))

    def search(self, query):
        """Semantic search: the semantic section of a query."""
        return self._m.query(semantic=query).semantic


class _Graph:
    def __init__(self, memory: Memory) -> None:
        self._m = memory

    def write(self, write) -> memory_pb2.CommitMemoryResponse:
        """Write graph nodes and edges: the graph section of a commit."""
        return self._m.commit(graph=write)

    def query(self, query):
        """A structured traversal (never a query string): the graph section of a
        query."""
        return self._m.query(graph=query).graph


class _AsyncLogs:
    def __init__(self, memory: AsyncMemory) -> None:
        self._m = memory

    async def create(self, step) -> memory_pb2.CommitMemoryResponse:
        return await self._m.commit(log=step)

    async def recent(self, query):
        return (await self._m.query(log=query)).log


class _AsyncVectors:
    def __init__(self, memory: AsyncMemory) -> None:
        self._m = memory

    async def upsert(self, *chunks) -> memory_pb2.CommitMemoryResponse:
        return await self._m.commit(vectors=list(chunks))

    async def search(self, query):
        return (await self._m.query(semantic=query)).semantic


class _AsyncGraph:
    def __init__(self, memory: AsyncMemory) -> None:
        self._m = memory

    async def write(self, write) -> memory_pb2.CommitMemoryResponse:
        return await self._m.commit(graph=write)

    async def query(self, query):
        return (await self._m.query(graph=query)).graph
