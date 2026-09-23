# jennah-sdk-py

The Python SDK for [Jennah](https://jennah.nightblue.io/), the memory and context
platform for AI agents.

```sh
pip install jennah-sdk-py
```

## Store and recall a memory

Sign in once with the [`jnh` CLI](https://jennah.nightblue.io/docs/cli/)
(`jnh login`), or set `JENNAH_API_KEY`. Then:

```python
import uuid

from jennah import Client
from jennah.agent.v1 import agent_pb2, memory_pb2

with Client() as client:  # uses `jnh login`, or $JENNAH_API_KEY
    agent_id = f"quickstart-{uuid.uuid4().hex[:8]}"
    client.agents.CreateAgent(
        agent_pb2.CreateAgentRequest(agent_instance_id=agent_id)
    )
    agent = client.agent(agent_id)

    # Store a memory. The platform embeds raw_content for you.
    agent.vectors.upsert(memory_pb2.VectorChunk(
        chunk_id="pref-1",
        raw_content="The customer prefers invoices in Japanese yen, "
        "sent on the 5th.",
    ))

    # Recall it by meaning, not by keyword.
    result = agent.vectors.search(memory_pb2.SemanticQuery(
        query_text="what currency does the customer want to be "
        "billed in?",
        limit=3,
    ))
    for m in result.matches:
        print(f"{m.distance:.3f}  {m.raw_content}")

    client.agents.DeleteAgent(
        agent_pb2.DeleteAgentRequest(agent_instance_id=agent_id)
    )
```

```
0.235  The customer prefers invoices in Japanese yen, sent on the 5th.
```

## Async

`AsyncClient` has the same operations, awaited, over `grpc.aio`. Nothing it does
blocks the event loop, credential renewal included.

```python
import asyncio

from jennah import AsyncClient
from jennah.agent.v1 import memory_pb2


async def main():
    async with AsyncClient() as client:
        query = memory_pb2.SemanticQuery(
            query_text="billing preferences", limit=3
        )
        result = await client.agent("my-agent").vectors.search(query)
        print(result.matches)


asyncio.run(main())
```

Use `Client` in scripts, notebooks and evaluation harnesses; use `AsyncClient`
inside an agent framework or any other asyncio program.

## Every operation, already authenticated

Each service in the API is a generated stub on the client, already bound to the
credentialed connection, so every operation is callable without a wrapper:

```python
from jennah.agent.v1 import scope_pb2

client.scopes.ListScopes(scope_pb2.ListScopesRequest())
client.memory.InspectMemory(...)
client.datasets.ListDatasets(...)
```

The services are `agents`, `memory`, `scopes`, `datasets`, `schema`, `data`,
`auth`, `approvals`, `billing` and `platform`. For a service added after your SDK
version, `client.stub(SomeServiceStub)` binds it the same way. Never build your
own channel to reach one: you would lose credential renewal.

`client.agent(id)` adds the memory shortcuts, each one single-section call:
`memory.commit(...)` and `memory.query(...)` (any request field, as keywords),
`logs.create`/`logs.recent`, `vectors.upsert`/`vectors.search`, and
`graph.write`/`graph.query`.

## Credentials

The client takes the first credential it finds, in this order:

1. `credentials=`, a source your program supplies.
2. `api_key=`.
3. The `JENNAH_API_KEY` environment variable.
4. The session stored by `jnh login`.

A signed-in session renews itself: a rejected call is renewed once and retried,
and the renewed session is written back to the shared credentials file, so `jnh`
on the same machine stays signed in. `client.credential` reports
what authenticated the client, and where it came from, without the secret.

## Development

The generated code is not committed. With
[jennah-api](https://github.com/alphauslabs/jennah-api) checked out next to this
repository:

```sh
scripts/dev-generate.sh      # generate and lay the stubs and conformance suite in place
pip install -e '.[test]'
pytest
```
