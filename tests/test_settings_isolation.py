"""Every settings object must be isolated from the real environment during tests.

The contamination behind D-021 was not really "tests wrote traces". It was that `src/` grew
a *second* `BaseSettings` class and the test isolation only knew about the first. Everything
about that failure was invisible: the suite passed, the agent worked, and the only symptom
was an impossible number in a document generated weeks later.

`ObservabilitySettings` is separate from `Settings` on purpose — the agent must not import
Langfuse or OpenTelemetry unless tracing is on — so the fix is not to merge them. It is to
make "did we isolate all of them?" a question the suite answers instead of a thing someone
remembers. Phase 5's FastAPI config is the next chance to get this wrong.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import sys
from pathlib import Path

from pydantic_settings import BaseSettings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src
from src.config import Settings, get_settings
from src.observability.config import ObservabilitySettings, get_observability_settings
from tests.conftest import ISOLATED_SETTINGS


def discover_settings_classes() -> dict[str, type[BaseSettings]]:
    """Every BaseSettings subclass defined under ``src/``, found by walking the package."""
    found: dict[str, type[BaseSettings]] = {}
    for module_info in pkgutil.walk_packages(src.__path__, prefix="src."):
        module = importlib.import_module(module_info.name)
        for name, obj in vars(module).items():
            if (
                inspect.isclass(obj)
                and issubclass(obj, BaseSettings)
                and obj is not BaseSettings
                and obj.__module__.startswith("src.")
            ):
                found[name] = obj
    return found


class TestEverySettingsObjectIsIsolated:
    def test_the_registry_matches_what_is_actually_defined(self) -> None:
        """Adding a settings class without isolating it fails here, not in production."""
        discovered = set(discover_settings_classes())

        unregistered = discovered - ISOLATED_SETTINGS
        assert not unregistered, (
            f"{sorted(unregistered)} subclass BaseSettings but are not isolated by "
            f"tests/conftest.py. Add an autouse fixture that neutralises them, then add "
            f"them to ISOLATED_SETTINGS. An un-isolated settings object reads the real "
            f".env during tests — that is how the suite came to write 187 traces into the "
            f"live Langfuse project (DECISIONS D-021)."
        )

        stale = ISOLATED_SETTINGS - discovered
        assert not stale, f"{sorted(stale)} are registered as isolated but no longer exist"

    def test_at_least_the_two_known_classes_are_found(self) -> None:
        """Guards the discovery itself: a walk that finds nothing would pass vacuously."""
        discovered = discover_settings_classes()

        assert {"Settings", "ObservabilitySettings"} <= set(discovered)


class TestIsolatingOneIsolatesTheOther:
    """The asymmetry itself, asserted from both directions."""

    def test_a_module_constructing_its_own_settings_gets_no_secrets(self) -> None:
        """The failure mode the per-module patching cannot cover.

        A new module that calls ``Settings()`` itself never goes through the patched
        ``get_settings``, so the only thing standing between it and the developer's real
        `.env` is the environment being severed at the source.
        """
        assert Settings().google_api_key.get_secret_value() == ""

    def test_a_module_constructing_its_own_observability_settings_is_disabled(self) -> None:
        """The direction that was actually broken.

        With a key pair in `.env` this returned enabled settings pointed at a live host,
        and every test touching the graph emitted a trace to it.
        """
        settings = ObservabilitySettings()

        assert settings.langfuse_public_key.get_secret_value() == ""
        assert not settings.langfuse_enabled

    def test_both_routes_to_observability_settings_are_disabled(self) -> None:
        """Isolation must not depend on which route a caller happens to use.

        The accessor and a fresh construction are different code paths — the per-module
        patching covers the first, severing the environment covers the second. Both are
        asserted because covering only one is what the original bug looked like.
        """
        assert not get_observability_settings().langfuse_enabled
        assert not ObservabilitySettings().langfuse_enabled

    def test_no_route_to_agent_settings_yields_a_real_api_key(self) -> None:
        """The safety property, which holds regardless of which binding wins.

        Production modules take the patched accessor and get the fixture's settings; a
        module constructing its own gets an empty one. The invariant worth enforcing is not
        which value comes back, but that neither is ever the developer's real key.
        """
        for candidate in (get_settings(), Settings()):
            assert candidate.google_api_key.get_secret_value() in {"", "test-key"}

    def test_no_langfuse_client_is_reachable_from_a_test(self) -> None:
        """The property that actually matters, stated once as an invariant."""
        from src.observability import langfuse as lf

        assert lf.get_client() is None

    def test_the_traced_runner_path_produces_no_client(self) -> None:
        """`run_query` opens a span unconditionally — this is the call site that leaked."""
        from src.observability import langfuse as lf

        with lf.trace_run(
            name="query",
            thread_id="t",
            question="q",
            model="m",
            prompt_version="v1",
        ) as root:
            assert root is None
