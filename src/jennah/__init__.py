"""Python SDK for Jennah, the memory and context platform for AI agents.

    from jennah import Client
    from jennah.agent.v1 import agent_pb2

    with Client() as client:
        print(client.agents.ListAgents(agent_pb2.ListAgentsRequest()))

Generated from the same proto revision, in the same release, as jennah-sdk-go:
the same version exposes the same operations in both.
"""

from .client import DEFAULT_ENDPOINT, AsyncClient, Client, CredentialInfo
from ._transport import RetryPolicy
from .credentials import Kind, Origin
from .errors import (
    CorruptSessionError,
    CredentialRefusedError,
    JennahError,
    NoCredentialError,
    SessionExpiredError,
    SessionPersistError,
    code,
    is_transient,
    is_unauthenticated,
)

try:
    from ._version import __version__
except ImportError:  # a source checkout that was never assembled
    __version__ = "0.0.0.dev0"

__all__ = [
    "AsyncClient",
    "Client",
    "CorruptSessionError",
    "CredentialInfo",
    "CredentialRefusedError",
    "DEFAULT_ENDPOINT",
    "JennahError",
    "Kind",
    "NoCredentialError",
    "Origin",
    "RetryPolicy",
    "SessionExpiredError",
    "SessionPersistError",
    "code",
    "is_transient",
    "is_unauthenticated",
]
