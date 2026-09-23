"""Which calls the SDK replays after a transient failure.

A question about the server's write semantics, not the network, and answered per
REQUEST for four writes, which is why this is not a gRPC service-config retry
policy (that sees the method, never the request).

The table is the same one jennah-sdk-go's retry.go holds. tests/test_classification.py
fails if any method the generated code publishes is in none or more than one of
the three sets, so a new RPC cannot inherit "never replay" without someone
deciding it should.
"""

from __future__ import annotations

import grpc

# Only UNAVAILABLE is transient: a dropped connection, a draining instance, a
# rolling deploy. RESOURCE_EXHAUSTED is an entitlement limit and would answer the
# same forever; DEADLINE_EXCEEDED means the caller's own budget is spent.
RETRYABLE_CODES = frozenset({grpc.StatusCode.UNAVAILABLE})

_AGENT = "/jennahapi.agent.v1.AgentService/"
_MEMORY = "/jennahapi.agent.v1.MemoryService/"
_SCOPE = "/jennahapi.agent.v1.ScopeService/"
_DATASET = "/jennahapi.datastore.v1.DatasetService/"
_SCHEMA = "/jennahapi.datastore.v1.SchemaService/"
_DATA = "/jennahapi.datastore.v1.DataService/"
_AUTH = "/jennahapi.auth.v1.AuthService/"
_APPROVAL = "/jennahapi.approval.v1.ApprovalService/"
_BILLING = "/jennahapi.billing.v1.BillingService/"
_PLATFORM = "/jennahapi.platform.v1.PlatformService/"

# Every method that reads without writing: replaying one costs a round trip and
# nothing else. Enumerated rather than matched on "Get"/"List", which are
# conventions the server is not obliged to keep.
REPLAYABLE_READS = frozenset({
    _AGENT + "GetAgent",
    _AGENT + "ListAgents",
    _MEMORY + "QueryMemory",
    _MEMORY + "InspectMemory",
    _MEMORY + "GetMemoryVocabulary",
    _SCOPE + "GetScope",
    _SCOPE + "ListScopes",
    _DATASET + "GetDataset",
    _DATASET + "ListDatasets",
    _SCHEMA + "GetSchema",
    _DATA + "QueryData",
    _AUTH + "WhoAmI",
    _AUTH + "ListApiKeys",
    _AUTH + "ListMembers",
    _AUTH + "ListInvitations",
    _AUTH + "ListPermissions",
    _AUTH + "ListRoles",
    _AUTH + "GetRole",
    _AUTH + "PollDeviceLogin",
    _APPROVAL + "GetApproval",
    _APPROVAL + "ListApprovals",
    _APPROVAL + "ListApprovers",
    _APPROVAL + "DescribeApprovalByToken",
    _BILLING + "GetBillingState",
    _PLATFORM + "ListLocations",
})

# Writes whose safety depends on the request; decided in safe_to_replay.
CONDITIONAL_REPLAY = frozenset({
    _MEMORY + "CommitMemory",
    _MEMORY + "FormMemory",
    _DATA + "CommitData",
    _APPROVAL + "CreateApproval",
})

# Never replayed automatically. Listed so the classification test can prove
# nothing was simply overlooked.
NEVER_REPLAY = frozenset({
    # Creating or destroying a resource twice is not the same as once.
    _AGENT + "CreateAgent",
    _AGENT + "DeleteAgent",
    _SCOPE + "CreateScope",
    _SCOPE + "DeleteScope",
    _DATASET + "CreateDataset",
    _DATASET + "DeleteDataset",
    # Schema work is asynchronous; a replay races the declaration already running.
    _SCHEMA + "DeclareTables",
    # Closes a validity window and inserts a replacement; a replay finds it closed.
    _MEMORY + "SupersedeEdge",
    _MEMORY + "SupersedeChunk",
    # Converge when repeated alone, but a replay across another caller's
    # declaration would overwrite it; rare administrative calls the caller retries.
    _MEMORY + "DeclareMemoryVocabulary",
    _MEMORY + "RemoveMemoryVocabulary",
    # A long poll that reports a pending approval as success.
    _APPROVAL + "WaitApproval",
    # Each sends mail, records a decision, or ends an approval.
    _APPROVAL + "CancelApproval",
    _APPROVAL + "ResendApprovalNotification",
    _APPROVAL + "SubmitApprovalDecision",
    _APPROVAL + "AddApprover",
    _APPROVAL + "RemoveApprover",
    # Session and credential mutations: a replayed refresh or logout can revoke
    # what the first attempt issued, a replayed mint leaves an unheld key.
    _AUTH + "StartLogin",
    _AUTH + "CompleteLogin",
    _AUTH + "ExchangeCode",
    _AUTH + "StartDeviceLogin",
    _AUTH + "RefreshToken",
    _AUTH + "Logout",
    _AUTH + "CreateApiKey",
    _AUTH + "RevokeApiKey",
    # Membership, role and enterprise administration.
    _AUTH + "InviteMember",
    _AUTH + "RevokeInvitation",
    _AUTH + "AcceptInvitation",
    _AUTH + "ChangeMemberRole",
    _AUTH + "RemoveMember",
    _AUTH + "TransferRoot",
    _AUTH + "UpdateEnterprise",
    _AUTH + "CreateRole",
    _AUTH + "UpdateRole",
    _AUTH + "DeleteRole",
    # Commits the enterprise to a paid agreement.
    _BILLING + "BindMarketplaceRegistration",
    _BILLING + "ResolveMarketplaceRegistration",
})


def safe_to_replay(method: str, request: object) -> bool:
    """Whether replaying ``request`` to ``method`` cannot produce a second effect.

    * CommitMemory: vector and graph writes are idempotent upserts, but a log step
      is append-only, so a commit carrying a log section is not replayed.
    * FormMemory: only with ``formation_key``; extraction is nondeterministic, so
      a blind replay forms a second, different set of memory.
    * CommitData: only with ``idempotency_key``.
    * CreateApproval: only with ``request_key``; mail cannot be recalled.

    Anything unclassified is not replayed.
    """
    if method in REPLAYABLE_READS:
        return True
    if method == _MEMORY + "CommitMemory":
        return not request.HasField("log")
    if method == _MEMORY + "FormMemory":
        return bool(request.formation_key)
    if method == _DATA + "CommitData":
        return bool(request.idempotency_key)
    if method == _APPROVAL + "CreateApproval":
        return bool(request.request_key)
    return False
