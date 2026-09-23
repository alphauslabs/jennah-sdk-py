"""The shared credential conformance suite, run against both clients.

The cases live in conformance/credentials/cases.json, laid here by jennah-api's
ci/python/assemble.sh from the same revision as the generated stubs, and are the
same cases jennah-sdk-go runs. This file only translates them onto this SDK:
see conformance/README.md for what each op and expectation means. A failing case
is a defect in this SDK, never a reason to edit the case.

Every case runs twice, through Client and through AsyncClient, because both are
surfaces a caller authenticates through and the contract binds both.
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import sys
import threading
import time
from concurrent import futures

import grpc
import pytest

import jennah
from jennah import credentials
from jennah.agent.v1 import agent_pb2, agent_pb2_grpc
from jennah.auth.v1 import auth_pb2, auth_pb2_grpc
from jennah.datastore.v1 import data_pb2, data_pb2_grpc

ROOT = pathlib.Path(__file__).resolve().parent.parent
CASES = ROOT / "conformance" / "credentials" / "cases.json"


# --- Loading, strictly --------------------------------------------------------

# Unknown keys fail rather than being skipped: a harness that ignores what it does
# not understand passes cases it never ran.
_SESSION_KEYS = {"raw", "endpoint", "access_token", "refresh_token", "token_type", "expires_in", "expires"}
_SERVER_KEYS = {"accept", "reject_all", "unavailable_first", "refresh", "on_refresh_write_session"}
_REFRESH_KEYS = {"accept", "access_token", "refresh_token", "expires_in"}
_STEP_KEYS = {"op", "method", "endpoint", "count", "writes", "raw", "session", "expect"}
_EXPECT_KEYS = {
    "ok", "error", "not", "mentions", "kind", "origin", "endpoint", "partial_reads", "identical",
    "presented", "refresh_bearers", "refreshes", "refresh_calls", "session", "session_mode",
    "dir_mode", "stray_files",
}
_CASE_KEYS = {"id", "requirement", "scenario", "requires", "given", "steps"}
_GIVEN_KEYS = {"explicit", "env", "session", "server"}


def _strict(obj, allowed, where):
    unknown = set(obj) - allowed
    if unknown:
        raise AssertionError(f"{where}: unknown keys {sorted(unknown)}")


def _load():
    suite = json.loads(CASES.read_text())
    _strict(suite, {"suite", "version", "source", "cases"}, "suite")
    assert suite["suite"] == "client-credentials" and suite["version"] == 1, (
        "this harness understands client-credentials version 1"
    )
    cases = suite["cases"]
    assert cases, "the shared suite has no cases"
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "duplicate case ids"
    for c in cases:
        _strict(c, _CASE_KEYS, c["id"])
        g = c.get("given", {})
        _strict(g, _GIVEN_KEYS, f"{c['id']} given")
        if isinstance(g.get("session"), dict):
            _strict(g["session"], _SESSION_KEYS, f"{c['id']} session")
        srv = g.get("server", {})
        _strict(srv, _SERVER_KEYS, f"{c['id']} server")
        if "refresh" in srv:
            _strict(srv["refresh"], _REFRESH_KEYS, f"{c['id']} refresh")
        if "on_refresh_write_session" in srv:
            _strict(srv["on_refresh_write_session"], _SESSION_KEYS, f"{c['id']} on_refresh")
        for s in c["steps"]:
            _strict(s, _STEP_KEYS, f"{c['id']} step")
            if "expect" in s:
                _strict(s["expect"], _EXPECT_KEYS, f"{c['id']} expect")
            if isinstance(s.get("session"), dict):
                _strict(s["session"], _SESSION_KEYS, f"{c['id']} step session")
    return cases


_CASES = _load()


# --- The fake platform --------------------------------------------------------


def _bearer(context) -> str:
    for k, v in context.invocation_metadata():
        if k == "authorization":
            return v[len("Bearer "):] if v.startswith("Bearer ") else v
    return ""


class FakePlatform(
    agent_pb2_grpc.AgentServiceServicer,
    data_pb2_grpc.DataServiceServicer,
    auth_pb2_grpc.AuthServiceServicer,
):
    def __init__(self, spec: dict, write_session) -> None:
        self.spec = json.loads(json.dumps(spec))
        self.write_session = write_session
        self.mu = threading.Lock()
        self.presented: list[str] = []
        self.refresh_bearers: list[str] = []
        self.refreshes = 0
        self.refresh_calls = 0
        self.cond = threading.Condition(self.mu)
        self.held = 0
        self.gate_open = True

    def hold_next(self, n: int) -> None:
        with self.mu:
            self.held = n
            self.gate_open = False

    def _barrier(self) -> None:
        with self.cond:
            if self.gate_open:
                return
            self.held -= 1
            if self.held == 0:
                self.gate_open = True
                self.cond.notify_all()
                return
            self.cond.wait_for(lambda: self.gate_open, timeout=10)

    def _business(self, context) -> None:
        self._barrier()
        with self.mu:
            got = _bearer(context)
            self.presented.append(got)
            if self.spec.get("unavailable_first", 0) > 0:
                self.spec["unavailable_first"] -= 1
                code = grpc.StatusCode.UNAVAILABLE, "try again"
            elif self.spec.get("reject_all") or not self.spec.get("accept") or got != self.spec["accept"]:
                code = grpc.StatusCode.UNAUTHENTICATED, "the access token is invalid or has expired"
            else:
                return
        context.abort(*code)

    def ListAgents(self, request, context):
        self._business(context)
        return agent_pb2.ListAgentsResponse()

    def CommitData(self, request, context):
        self._business(context)
        return data_pb2.CommitDataResponse()

    def RefreshToken(self, request, context):
        with self.mu:
            self.refresh_calls += 1
            self.refresh_bearers.append(_bearer(context))
            if "on_refresh_write_session" in self.spec:
                self.write_session(self.spec["on_refresh_write_session"])
            rf = self.spec.get("refresh")
            if rf is None or request.refresh_token != rf["accept"]:
                refused = True
            else:
                refused = False
                self.refreshes += 1
                resp = auth_pb2.RefreshTokenResponse(
                    access_token=rf["access_token"],
                    refresh_token=rf["refresh_token"],
                    expires_in=rf["expires_in"],
                )
                # Rotate: the refresh token just presented is dead from here on.
                self.spec["accept"] = rf["access_token"]
                self.spec["refresh"] = {"accept": rf["refresh_token"], "access_token": "",
                                        "refresh_token": "", "expires_in": 0}
        if refused:
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "the refresh token is invalid or has expired")
        return resp

    def stats(self):
        with self.mu:
            return list(self.presented), list(self.refresh_bearers), self.refreshes, self.refresh_calls


# --- The two surfaces ---------------------------------------------------------


class SyncDriver:
    def construct(self, **kw):
        self.client = jennah.Client(**kw)

    def call(self, method):
        if method == "read":
            self.client.agents.ListAgents(agent_pb2.ListAgentsRequest(), timeout=10)
        elif method == "write_unsafe":
            self.client.data.CommitData(data_pb2.CommitDataRequest(), timeout=10)
        else:
            raise AssertionError(f"unknown method {method!r}")

    def call_concurrently(self, method, count):
        errs = [None] * count

        def one(i):
            try:
                self.call(method)
            except BaseException as e:  # noqa: BLE001
                errs[i] = e

        threads = [threading.Thread(target=one, args=(i,)) for i in range(count)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return errs

    def close(self):
        if getattr(self, "client", None):
            self.client.close()


class AsyncDriver:
    def __init__(self):
        self.loop = asyncio.new_event_loop()

    def construct(self, **kw):
        async def make():
            return jennah.AsyncClient(**kw)

        self.client = self.loop.run_until_complete(make())

    async def _call(self, method):
        if method == "read":
            await self.client.agents.ListAgents(agent_pb2.ListAgentsRequest(), timeout=10)
        elif method == "write_unsafe":
            await self.client.data.CommitData(data_pb2.CommitDataRequest(), timeout=10)
        else:
            raise AssertionError(f"unknown method {method!r}")

    def call(self, method):
        self.loop.run_until_complete(self._call(method))

    def call_concurrently(self, method, count):
        async def all_():
            return await asyncio.gather(*(self._call(method) for _ in range(count)), return_exceptions=True)

        return [None if r is None else r for r in self.loop.run_until_complete(all_())]

    def close(self):
        if getattr(self, "client", None):
            self.loop.run_until_complete(self.client.close())
        self.loop.close()


# --- Running a case -----------------------------------------------------------

_CATEGORIES = {
    "no_credential": lambda e: isinstance(e, jennah.NoCredentialError),
    "corrupt_session": lambda e: isinstance(e, jennah.CorruptSessionError),
    "session_expired": lambda e: isinstance(e, jennah.SessionExpiredError),
    "credential_refused": lambda e: isinstance(e, jennah.CredentialRefusedError),
    "unauthenticated": lambda e: jennah.code(e) == grpc.StatusCode.UNAUTHENTICATED,
    "unavailable": lambda e: jennah.code(e) == grpc.StatusCode.UNAVAILABLE,
    "failed": lambda e: e is not None,
}

_KIND = {jennah.Kind.API_KEY: "api_key", jennah.Kind.SESSION: "session"}
_ORIGIN = {jennah.Origin.EXPLICIT: "explicit", jennah.Origin.ENVIRONMENT: "environment",
           jennah.Origin.FILE: "file"}


class Run:
    def __init__(self, case, surface, tmp_path, monkeypatch):
        self.case = case
        self.monkeypatch = monkeypatch
        # An empty machine: a fresh config directory and no key in the environment.
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("APPDATA", str(tmp_path))
        monkeypatch.delenv(jennah.errors.ENV_API_KEY, raising=False)
        self.path = credentials.session_path()
        given = case.get("given", {})
        if given.get("env"):
            monkeypatch.setenv(jennah.errors.ENV_API_KEY, given["env"])
        if isinstance(given.get("session"), dict):
            self.write_session(given["session"])
        self.secrets = [s for s in (
            given.get("explicit"), given.get("env"),
            *(given.get("session", {}).get(k) for k in ("access_token", "refresh_token")),
        ) if s]

        self.platform = FakePlatform(given.get("server", {}), self.write_session)
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=32))
        agent_pb2_grpc.add_AgentServiceServicer_to_server(self.platform, self.server)
        data_pb2_grpc.add_DataServiceServicer_to_server(self.platform, self.server)
        auth_pb2_grpc.add_AuthServiceServicer_to_server(self.platform, self.server)
        self.port = self.server.add_insecure_port("127.0.0.1:0")
        self.server.start()
        self.driver = SyncDriver() if surface == "sync" else AsyncDriver()
        self.constructed = False

    def close(self):
        self.driver.close()
        self.server.stop(None)

    def write_session(self, s: dict) -> None:
        """Put a session on disk as another process would: the canonical format
        written directly, not through the SDK under test."""
        if "raw" in s:
            data = s["raw"].encode()
        else:
            expires_at = int(time.time()) + s["expires_in"] if "expires_in" in s else 0
            data = json.dumps({
                "endpoint": s.get("endpoint", ""), "access_token": s.get("access_token", ""),
                "refresh_token": s.get("refresh_token", ""), "token_type": s.get("token_type", ""),
                "expires_at": expires_at,
            }, indent=2).encode()
        os.makedirs(os.path.dirname(self.path), mode=0o700, exist_ok=True)
        tmp = self.path + ".harness"
        with open(tmp, "wb") as f:
            f.write(data)
        os.chmod(tmp, 0o600)
        os.replace(tmp, self.path)

    def expect_outcome(self, where, ex, err):
        if "ok" in ex:
            if ex["ok"] and err is not None:
                raise AssertionError(f"{where}: want success, got {err!r}") from err
            return
        assert ex.get("error"), f"{where}: expect names neither ok nor error"
        assert err is not None, f"{where}: want error {ex['error']}, got success"
        for cat in ex["error"]:
            assert _CATEGORIES[cat](err), f"{where}: error {err!r} is not {cat}"
        for cat in ex.get("not", []):
            assert not _CATEGORIES[cat](err), f"{where}: error {err!r} must not be {cat}"
        for m in ex.get("mentions", []):
            m = self.path if m == "$SESSION_PATH" else m
            assert m in str(err), f"{where}: error {str(err)!r} does not mention {m!r}"

    def step(self, i, s):
        op, ex = s["op"], s.get("expect")
        where = f"step {i} ({op})"
        if op == "construct":
            kw = {"api_key": self.case.get("given", {}).get("explicit")}
            if s.get("endpoint") is None:
                kw.update(endpoint=f"127.0.0.1:{self.port}", insecure=True)
            elif s["endpoint"] != "default":
                raise AssertionError(f"{where}: unknown endpoint {s['endpoint']!r}")
            err = None
            try:
                self.driver.construct(**kw)
                self.constructed = True
            except Exception as e:  # noqa: BLE001
                err = e
            self.expect_outcome(where, ex, err)
        elif op == "describe":
            c = self.driver.client
            info = c.credential
            rendered = f"{info} {info!r} {c!r}"
            for secret in self.secrets:
                assert secret not in rendered, f"{where}: the credential report leaks a secret: {rendered!r}"
            if "kind" in ex:
                assert _KIND[info.kind] == ex["kind"], f"{where}: kind = {info.kind}"
            if "origin" in ex:
                assert _ORIGIN[info.origin] == ex["origin"], f"{where}: origin = {info.origin}"
            if "endpoint" in ex:
                want = jennah.DEFAULT_ENDPOINT if ex["endpoint"] == "$DEFAULT_ENDPOINT" else ex["endpoint"]
                assert c.endpoint == want, f"{where}: endpoint = {c.endpoint!r}, want {want!r}"
        elif op == "call":
            err = None
            try:
                self.driver.call(s["method"])
            except Exception as e:  # noqa: BLE001
                err = e
            self.expect_outcome(where, ex, err)
        elif op == "call_concurrently":
            assert s["count"] >= 2
            self.platform.hold_next(s["count"])
            for n, err in enumerate(self.driver.call_concurrently(s["method"], s["count"])):
                self.expect_outcome(f"{where} call {n}", ex, err)
        elif op == "write_session":
            self.write_session(s["session"])
        elif op == "lock_session_directory":
            assert "unwritable_directory" in self.case.get("requires", []), (
                f"{where}: case does not declare requires unwritable_directory"
            )
            os.chmod(os.path.dirname(self.path), 0o500)
        elif op == "atomic_replace":
            got = self.atomic_replace(s["writes"])
            assert got == ex["partial_reads"], f"{where}: partial reads = {got}"
        elif op == "round_trip":
            self.write_session({"raw": s["raw"]})
            credentials.save_session(credentials.load_session())
            data = pathlib.Path(self.path).read_bytes()
            assert (data == s["raw"].encode()) == ex["identical"], (
                f"{where}: got {data!r}, want {s['raw'].encode()!r}"
            )
        elif op == "check":
            self.check(where, ex)
        else:
            raise AssertionError(f"{where}: unknown op")

    def atomic_replace(self, writes):
        assert writes >= 1

        def token(n):
            # Lengths vary on purpose, so a torn write cannot parse by accident.
            return f"at_{n}_" + "x" * (n % 37)

        written = {token(n) for n in range(writes)}
        partial = 0
        lock = threading.Lock()
        done = threading.Event()

        def reader():
            nonlocal partial
            while not done.is_set():
                try:
                    s = credentials.load_session()
                    bad = s is not None and s.access_token not in written
                except Exception:  # noqa: BLE001
                    bad = True
                if bad:
                    with lock:
                        partial += 1

        readers = [threading.Thread(target=reader) for _ in range(4)]
        for t in readers:
            t.start()
        for n in range(writes):
            credentials.save_session(credentials.Session(
                endpoint="https://jennah.alphaus.cloud", access_token=token(n),
                refresh_token="rt", token_type="Bearer",
            ))
        done.set()
        for t in readers:
            t.join()
        return partial

    def check(self, where, ex):
        presented, bearers, refreshes, refresh_calls = self.platform.stats()
        if "presented" in ex:
            assert presented == ex["presented"], f"{where}: presented = {presented}"
        if "refresh_bearers" in ex:
            assert bearers == ex["refresh_bearers"], f"{where}: refresh bearers = {bearers}"
        if "refreshes" in ex:
            assert refreshes == ex["refreshes"], f"{where}: refreshes = {refreshes}"
        if "refresh_calls" in ex:
            assert refresh_calls == ex["refresh_calls"], f"{where}: refresh calls = {refresh_calls}"
        if "session" in ex:
            want = ex["session"]
            if want == "absent":
                assert not os.path.exists(self.path), f"{where}: want no stored session"
            else:
                assert isinstance(want, dict), f"{where}: unknown session expectation {want!r}"
                got = credentials.load_session()
                for k in ("endpoint", "access_token", "refresh_token", "token_type"):
                    if k in want:
                        assert getattr(got, k) == want[k], f"{where}: stored {k} = {getattr(got, k)!r}"
                if want.get("expires") == "future":
                    assert got.expires_at > time.time(), f"{where}: stored expires_at not in the future"
                elif "expires" in want:
                    raise AssertionError(f"{where}: unknown expires expectation")
        if sys.platform != "win32":
            for key, path in (("session_mode", self.path), ("dir_mode", os.path.dirname(self.path))):
                if key in ex:
                    mode = f"{os.stat(path).st_mode & 0o777:04o}"
                    assert mode == ex[key], f"{where}: mode of {path} = {mode}, want {ex[key]}"
        if "stray_files" in ex:
            stray = [n for n in os.listdir(os.path.dirname(self.path)) if n != os.path.basename(self.path)]
            assert len(stray) == ex["stray_files"], f"{where}: stray files = {stray}"


@pytest.mark.parametrize("surface", ["sync", "async"])
@pytest.mark.parametrize("case", _CASES, ids=[c["id"] for c in _CASES])
def test_credential_conformance(case, surface, tmp_path, monkeypatch):
    for req in case.get("requires", []):
        if req == "unwritable_directory":
            if sys.platform == "win32" or os.geteuid() == 0:
                pytest.skip("directory permissions are not enforced for this runner")
        else:
            raise AssertionError(f"unknown requirement {req!r}")
    run = Run(case, surface, tmp_path, monkeypatch)
    try:
        for i, s in enumerate(case["steps"]):
            run.step(i, s)
    finally:
        if os.path.isdir(os.path.dirname(run.path)):
            os.chmod(os.path.dirname(run.path), 0o700)
        run.close()
