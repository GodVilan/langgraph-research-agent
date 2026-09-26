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

import sys
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

    @staticmethod
    def fake_repo(tmp_path: Path) -> Path:
        """A repo tree holding only tracked content plus stand-ins for the gitignored index.

        The first version assembled the *real* repo, which needs the FAISS index — gitignored,
        so absent on every CI runner — and failed there from Phase 5 on, which (with the docs
        test before it) kept the regression gate from ever running in CI (D-056). The real
        `.dockerignore`, Dockerfile and `src/` are copied, so the scan still covers the source
        that ships; the data files are small stand-ins whose manifests checksum them.
        """
        import hashlib
        import shutil

        repo = tmp_path / "repo"
        (repo / "infra").mkdir(parents=True)
        shutil.copy2(DOCKERIGNORE, repo / ".dockerignore")
        shutil.copy2(DOCKERIGNORE.parent / "infra" / "Dockerfile", repo / "infra" / "Dockerfile")
        shutil.copytree(DOCKERIGNORE.parent / "src", repo / "src")
        for name in ("pyproject.toml", "LICENSE"):
            shutil.copy2(DOCKERIGNORE.parent / name, repo / name)
        manifests = {
            "data/CORPUS.sha256": ["data/chunks_512.json", "data/metadata.json"],
            "data/INDEX.sha256": [
                "data/indices/BGE_cs512.faiss",
                "data/indices/BGE_cs512_meta.pkl",
            ],
        }
        for manifest, files in manifests.items():
            lines = []
            for rel in files:
                (repo / rel).parent.mkdir(parents=True, exist_ok=True)
                (repo / rel).write_bytes(f"stand-in for {rel}\n".encode())
                lines.append(f"{hashlib.sha256((repo / rel).read_bytes()).hexdigest()}  {rel}")
            (repo / manifest).write_text("\n".join(lines) + "\n", encoding="utf-8")
        return repo

    def test_the_space_context_refuses_the_compose_file(self, tmp_path: Path) -> None:
        from scripts.deploy_space import ContextError, assemble_context, check_context

        context = assemble_context(tmp_path / "ctx", self.fake_repo(tmp_path))
        check_context(context)  # a context of the shipped files passes
        # No provisioning block in it, so only the *filename* check can refuse it. The first
        # version wrote the real compose file, whose `LANGFUSE_INIT_` block tripped the text
        # scan — its message also names the file — so the filename check could be deleted
        # and this test still passed (found by mutation, D-056).
        (context / "docker-compose.langfuse.yml").write_text("services: {}\n")
        with pytest.raises(ContextError, match="a compose or env file must never be deployed"):
            check_context(context)

    def test_the_space_context_refuses_an_env_file(self, tmp_path: Path) -> None:
        from scripts.deploy_space import ContextError, assemble_context, check_context

        context = assemble_context(tmp_path / "ctx", self.fake_repo(tmp_path))
        (context / ".env").write_text("GOOGLE_API_KEY=placeholder\n")
        with pytest.raises(ContextError, match="a compose or env file must never be deployed"):
            check_context(context)

    def test_the_space_context_refuses_the_provisioning_block(self, tmp_path: Path) -> None:
        from scripts.deploy_space import ContextError, assemble_context, check_context

        context = assemble_context(tmp_path / "ctx", self.fake_repo(tmp_path))
        (context / "src" / "leak.py").write_text("LANGFUSE_INIT_PROJECT_SECRET_KEY = 'x'\n")
        with pytest.raises(ContextError, match="LANGFUSE_INIT_"):
            check_context(context)

    def test_the_space_context_refuses_an_index_its_manifest_does_not_describe(
        self, tmp_path: Path
    ) -> None:
        """The checksum path, which only the real index used to exercise — and CI has none."""
        from scripts.deploy_space import ContextError, assemble_context, check_context

        context = assemble_context(tmp_path / "ctx", self.fake_repo(tmp_path))
        (context / "data" / "indices" / "BGE_cs512.faiss").write_bytes(b"a different index\n")
        with pytest.raises(ContextError, match=r"does not match data/INDEX\.sha256"):
            check_context(context)

    def test_a_missing_admitted_file_is_refused(self, tmp_path: Path) -> None:
        from scripts.deploy_space import ContextError, assemble_context

        repo = self.fake_repo(tmp_path)
        (repo / "data" / "indices" / "BGE_cs512.faiss").unlink()
        with pytest.raises(ContextError, match=r"admitted by \.dockerignore but missing"):
            assemble_context(tmp_path / "ctx", repo)


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


class TestDeployOnlyFromATaggedCommit:
    """D-055: `make deploy-space` refuses unless every deployed file is a tagged commit."""

    @staticmethod
    def repo(tmp_path: Path) -> Path:
        import subprocess

        r = tmp_path / "repo"
        for rel, text in {
            ".dockerignore": "*\n!pyproject.toml\n!src/\n",
            "pyproject.toml": "[project]\nname='x'\n",
            "src/app.py": "print('v1')\n",
            "infra/Dockerfile": "FROM scratch\n",
            "scripts/deploy_space.py": "# card\n",
            "README.md": "not deployed\n",
        }.items():
            (r / rel).parent.mkdir(parents=True, exist_ok=True)
            (r / rel).write_text(text, encoding="utf-8")
        for cmd in (
            ["init", "-q"],
            ["add", "-A"],
            ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "c1"],
            ["tag", "deploy-2026-01-01"],
        ):
            subprocess.run(["git", *cmd], cwd=r, check=True, capture_output=True)
        return r

    def test_a_clean_tagged_tree_is_accepted(self, tmp_path: Path) -> None:
        from scripts.deploy_space import deploy_refusal, provenance

        prov = provenance(self.repo(tmp_path))
        assert prov.deploy_tags == ["deploy-2026-01-01"] and prov.dirty == []
        assert deploy_refusal(prov) == ""

    def test_a_modified_deployed_file_is_refused(self, tmp_path: Path) -> None:
        from scripts.deploy_space import deploy_refusal, provenance

        r = self.repo(tmp_path)
        (r / "src" / "app.py").write_text("print('v2')\n", encoding="utf-8")
        refusal = deploy_refusal(provenance(r))
        assert "src/app.py" in refusal and "D-055" in refusal

    def test_an_untracked_deployed_file_is_refused(self, tmp_path: Path) -> None:
        from scripts.deploy_space import deploy_refusal, provenance

        r = self.repo(tmp_path)
        (r / "src" / "new.py").write_text("x = 1\n", encoding="utf-8")
        assert "src/new.py" in deploy_refusal(provenance(r))

    def test_the_dockerfile_and_the_card_renderer_count_as_deployed(self, tmp_path: Path) -> None:
        from scripts.deploy_space import deploy_refusal, provenance

        r = self.repo(tmp_path)
        (r / "infra" / "Dockerfile").write_text("FROM busybox\n", encoding="utf-8")
        (r / "scripts" / "deploy_space.py").write_text("# other card\n", encoding="utf-8")
        refusal = deploy_refusal(provenance(r))
        assert "infra/Dockerfile" in refusal and "scripts/deploy_space.py" in refusal

    def test_an_untagged_head_is_refused_even_when_clean(self, tmp_path: Path) -> None:
        import subprocess

        from scripts.deploy_space import deploy_refusal, provenance

        r = self.repo(tmp_path)
        (r / "src" / "app.py").write_text("print('v2')\n", encoding="utf-8")
        subprocess.run(
            ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qam", "c2"],
            cwd=r,
            check=True,
        )
        assert "carries no deploy-* tag" in deploy_refusal(provenance(r))

    def test_a_dirty_file_that_is_not_deployed_does_not_block(self, tmp_path: Path) -> None:
        from scripts.deploy_space import deploy_refusal, provenance

        r = self.repo(tmp_path)
        (r / "README.md").write_text("edited\n", encoding="utf-8")
        assert deploy_refusal(provenance(r)) == ""

    def test_main_refuses_before_uploading_and_allow_dirty_is_explicit(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import scripts.deploy_space as ds

        dirty = ds.Provenance(head="a" * 40, deploy_tags=[], dirty=["src/app.py"])
        monkeypatch.setattr(ds, "provenance", lambda repo=ds.REPO: dirty)
        monkeypatch.setattr(ds, "assemble_context", lambda dest: dest.mkdir(parents=True) or dest)
        monkeypatch.setattr(ds, "check_context", lambda context: None)
        monkeypatch.setattr(
            sys, "argv", ["deploy_space.py", "--space", "o/n", "--i-confirmed-public"]
        )
        with pytest.raises(SystemExit, match="refusing to deploy: deployed files differ"):
            ds.main()

        # The override gets past the guard, warns, and still stops at the publish confirmation
        # when that is absent — so this test uploads nothing.
        monkeypatch.setattr(sys, "argv", ["deploy_space.py", "--space", "o/n", "--allow-dirty"])
        with pytest.raises(SystemExit, match="without --i-confirmed-public"):
            ds.main()
        assert "WARNING: --allow-dirty" in capsys.readouterr().err

    def test_a_deploy_record_is_written_and_read_back(self, tmp_path: Path) -> None:
        import json

        from scripts.deploy_space import record_deploy

        log = tmp_path / "deploy_log.jsonl"
        record_deploy({"space_commit": "abc", "allow_dirty": True}, log)
        record_deploy({"space_commit": "def", "allow_dirty": False}, log)
        rows = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
        assert [r["space_commit"] for r in rows] == ["abc", "def"]
