"""Observability settings.

Kept separate from ``src/config.py`` so the agent imports nothing from Langfuse or
OpenTelemetry unless observability is actually switched on. Everything degrades to a no-op
when it is off: an unconfigured tracer must never be a reason a query fails.
"""

from __future__ import annotations

import functools

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


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


@functools.lru_cache(maxsize=1)
def get_observability_settings() -> ObservabilitySettings:
    return ObservabilitySettings()
