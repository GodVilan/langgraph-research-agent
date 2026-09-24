"""Observability settings.

Kept separate from ``src/config.py`` so the agent imports nothing from Langfuse or
OpenTelemetry unless observability is actually switched on. Everything degrades to a no-op
when it is off: an unconfigured tracer must never be a reason a query fails.
"""

from __future__ import annotations

import functools

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# The public keys seeded by infra/docker-compose.langfuse.yml. Listed here so the guard
# below can recognise them; they are fixtures for a localhost-only stack, not secrets.
SEEDED_KEYS = frozenset(
    {
        "pk-lf-1a1a1a1a-2b2b-4c4c-8d8d-3e3e3e3e3e3e",
        "sk-lf-4f4f4f4f-5a5a-4b6b-8c7c-6d6d6d6d6d6d",
    }
)
LOCAL_HOSTNAMES = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})


class ObservabilitySettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── Langfuse ───────────────────────────────────────────────────────────────
    langfuse_public_key: SecretStr = SecretStr("")
    langfuse_secret_key: SecretStr = SecretStr("")
    # Self-hosted default (infra/docker-compose.yml). Langfuse Cloud is
    # https://cloud.langfuse.com or https://us.cloud.langfuse.com.
    langfuse_host: str = "http://localhost:3000"
    # Tags every trace so a self-hosted dev instance and a deployed one stay separable.
    langfuse_environment: str = "development"
    # Stamped on every trace so a metric change can be attributed to a code version.
    release: str = "0.1.0"
    # Head sampling: the fraction of requests traced, decided before the run starts. 1.0
    # traces everything, which is the default and what the deployed instance ships with —
    # the arithmetic for why is in docs/OBSERVABILITY.md "Sampling". A sampled-out request
    # reports trace_id "" rather than an id pointing at a trace that was never exported.
    langfuse_sample_rate: float = 1.0

    # ── OpenTelemetry ──────────────────────────────────────────────────────────
    # Retrieval and embedding spans. When Langfuse is enabled these nest inside the
    # Langfuse trace automatically, because Langfuse installs itself as the global OTel
    # tracer provider. When it is not, they go to the OTLP endpoint below if one is set,
    # and are dropped otherwise.
    otlp_endpoint: str = ""
    otel_console_export: bool = False

    @property
    def langfuse_enabled(self) -> bool:
        return bool(
            self.langfuse_public_key.get_secret_value()
            and self.langfuse_secret_key.get_secret_value()
        )

    @property
    def uses_seeded_keys(self) -> bool:
        """True when either half of the key pair is the fixture committed in the compose file."""
        return (
            self.langfuse_public_key.get_secret_value() in SEEDED_KEYS
            or self.langfuse_secret_key.get_secret_value() in SEEDED_KEYS
        )

    @property
    def host_is_local(self) -> bool:
        """The parsed hostname is a loopback name — not merely a substring of the URL.

        A substring test (the Phase 3 version) called ``https://localhost.example.com`` and
        ``https://example.com/localhost`` local, which is the exact hole this guard exists
        to close.
        """
        from urllib.parse import urlsplit

        return (urlsplit(self.langfuse_host).hostname or "").lower() in LOCAL_HOSTNAMES

    def check_not_deployed_with_seeded_keys(self) -> str | None:
        """Refuse the committed fixture credentials against a non-local host.

        `infra/docker-compose.langfuse.yml` commits a key pair on purpose: it is
        non-secret by construction, since it only ever unlocks a 127.0.0.1-bound container.
        That reasoning holds exactly as long as the host stays local. Pointing the seeded
        pair at a remote Langfuse would turn a documented fixture into a real credential in
        version control, so it fails loudly instead. Returns an error string, or None.
        """
        if self.uses_seeded_keys and not self.host_is_local:
            return (
                f"Refusing to use the seeded Langfuse fixture credentials against "
                f"{self.langfuse_host!r}. Those keys are committed to this repo and are "
                f"only non-secret while the host is localhost. A deployed instance must "
                f"set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY from its own "
                f"environment — see docs/OBSERVABILITY.md."
            )
        return None


@functools.lru_cache(maxsize=1)
def get_observability_settings() -> ObservabilitySettings:
    return ObservabilitySettings()
