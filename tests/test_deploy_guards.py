"""The committed Langfuse fixture credentials cannot reach a deployment.

``infra/docker-compose.langfuse.yml`` commits a key pair, a login, and encryption salts on
purpose, and they are non-secret for exactly one reason: the stack is reachable only from
127.0.0.1. Phase 3 enforced that in the *client* (seeded keys against a remote host raise).
This is the Phase 5 carry from the Phase 3 review: enforce it in the compose file and the
deploy path too, so the seeds cannot go live by editing a port line or by shipping the file.

Three mechanisms, each tested so that it fires:

1. the compose file publishes nothing beyond loopback while it carries the seeds;
2. nothing that builds a deployment can contain the compose file or its provisioning block;
3. the client refuses either seeded key against a host that is not loopback — by parsed
   hostname, since the Phase 3 substring check called ``localhost.example.com`` local.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import SecretStr

from src.observability.config import SEEDED_KEYS, ObservabilitySettings

REPO = Path(__file__).resolve().parent.parent
COMPOSE = REPO / "infra" / "docker-compose.langfuse.yml"
DOCKERIGNORE = REPO / ".dockerignore"


def compose() -> dict[str, Any]:
    return dict(yaml.safe_load(COMPOSE.read_text(encoding="utf-8")))


def published_ports(service: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for port in service.get("ports") or []:
        if isinstance(port, dict):  # long syntax
            out.append(f"{port.get('host_ip', '')}:{port.get('published', '')}")
        else:
            out.append(str(port))
    return out


def loopback_only_problems(doc: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    for name, service in (doc.get("services") or {}).items():
        if service.get("network_mode") == "host":
            problems.append(f"{name}: network_mode host publishes every port on every interface")
        for port in published_ports(service):
            # "127.0.0.1:3000:3000" is loopback; "3000:3000" and "0.0.0.0:3000:3000" are not.
            if not port.startswith("127.0.0.1:"):
                problems.append(f"{name}: publishes {port!r} beyond loopback")
    return problems


class TestComposeStaysLocal:
    def test_the_compose_file_carries_the_seeds(self) -> None:
        """Premise of the guard below. If the seeds are ever removed, it can relax."""
        text = COMPOSE.read_text(encoding="utf-8")
        assert all(key in text for key in SEEDED_KEYS)

    def test_every_published_port_is_loopback(self) -> None:
        assert loopback_only_problems(compose()) == []

    def test_the_ui_url_is_local(self) -> None:
        env = compose()["services"]["langfuse-web"]["environment"]
        assert env["NEXTAUTH_URL"].startswith("http://localhost")

    @pytest.mark.parametrize(
        "ports",
        [["3000:3000"], ["0.0.0.0:3000:3000"], [{"published": 3000, "target": 3000}]],
    )
    def test_the_guard_fires_on_a_public_binding(self, ports: list[Any]) -> None:
        doc = compose()
        doc["services"]["langfuse-web"]["ports"] = ports
        assert loopback_only_problems(doc)

    def test_the_guard_fires_on_host_networking(self) -> None:
        doc = compose()
        doc["services"]["redis"]["network_mode"] = "host"
        assert loopback_only_problems(doc)


class TestDeployArtifactsExcludeTheFixture:
    def test_dockerignore_is_an_allow_list_that_never_admits_infra(self) -> None:
        lines = [
            ln.strip()
            for ln in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
        assert lines[0] == "*", "the build context must start from nothing"
        admitted = [ln[1:] for ln in lines if ln.startswith("!")]
        assert not any(a.startswith("infra") or a.startswith(".env") for a in admitted)

    def test_the_space_context_refuses_the_compose_file(self, tmp_path: Path) -> None:
        from scripts.deploy_space import ContextError, assemble_context, check_context

        context = assemble_context(tmp_path / "ctx")
        check_context(context)  # the real context passes
        (context / "docker-compose.langfuse.yml").write_text(COMPOSE.read_text())
        with pytest.raises(ContextError, match="compose"):
            check_context(context)

    def test_the_space_context_refuses_the_provisioning_block(self, tmp_path: Path) -> None:
        from scripts.deploy_space import ContextError, assemble_context, check_context

        context = assemble_context(tmp_path / "ctx")
        (context / "src" / "leak.py").write_text("LANGFUSE_INIT_PROJECT_SECRET_KEY = 'x'\n")
        with pytest.raises(ContextError, match="LANGFUSE_INIT_"):
            check_context(context)


class TestClientRefusesSeedsRemotely:
    @pytest.mark.parametrize("which", ["public", "secret"])
    def test_either_seeded_key_is_refused_remotely(self, which: str) -> None:
        public, secret = sorted(SEEDED_KEYS)  # "pk-…" sorts before "sk-…"
        s = ObservabilitySettings(
            langfuse_public_key=SecretStr(public if which == "public" else "pk-lf-real"),
            langfuse_secret_key=SecretStr(secret if which == "secret" else "sk-lf-real"),
            langfuse_host="https://cloud.langfuse.com",
        )
        assert s.check_not_deployed_with_seeded_keys() is not None

    @pytest.mark.parametrize(
        ("host", "local"),
        [
            ("http://localhost:3000", True),
            ("http://127.0.0.1:3000", True),
            ("http://[::1]:3000", True),
            ("https://localhost.attacker.example", False),
            ("https://attacker.example/localhost", False),
            ("https://127.0.0.1.nip.io", False),
            ("https://cloud.langfuse.com", False),
        ],
    )
    def test_locality_is_the_parsed_hostname(self, host: str, local: bool) -> None:
        assert ObservabilitySettings(langfuse_host=host).host_is_local is local
