"""Every RPC the generated code publishes is classified for retry exactly once.

The default in safe_to_replay is "do not replay", which is safe but silent: a new
RPC would inherit it without anyone deciding. This test is where the decision is
forced. Services are found by walking the generated modules on disk, not from a
list, so neither a new service nor a new proto package can slip past it.
"""

from __future__ import annotations

import importlib
import pkgutil

import jennah
from jennah import _retry

SETS = (_retry.REPLAYABLE_READS, _retry.CONDITIONAL_REPLAY, _retry.NEVER_REPLAY)


def published_methods() -> set[str]:
    methods = set()
    for info in pkgutil.walk_packages(jennah.__path__, "jennah."):
        if not info.name.endswith("_pb2"):
            continue
        module = importlib.import_module(info.name)
        for service in module.DESCRIPTOR.services_by_name.values():
            for m in service.methods:
                methods.add(f"/{service.full_name}/{m.name}")
    return methods


def test_every_method_is_classified_exactly_once():
    methods = published_methods()
    # A floor, so a walk that found nothing fails without every new RPC failing it.
    assert len(methods) >= 66, f"walked {len(methods)} methods, expected at least 66"
    wrong = {m: sum(m in s for s in SETS) for m in methods}
    assert not {m: n for m, n in wrong.items() if n != 1}, "each method must be in exactly one set"


def test_no_classification_names_a_missing_method():
    # A stale entry hides that its RPC was renamed.
    stale = set().union(*SETS) - published_methods()
    assert not stale, f"classified but not published: {sorted(stale)}"
