"""The process environment as it was when the test session started.

`tests/conftest.py` imports this module first, so the snapshot is taken before any fixture
runs. `_sever_settings_from_the_environment` then deletes every variable named after a
settings field, for every test — which is right for the suite and wrong for the one test that
needs the real backend: CI hands the integration step its Langfuse key pair and host as
environment variables, the fixture deleted them, and the test (which read only `.env`, absent on
a runner) skipped with "Langfuse is not configured" on the job's first run past MinIO (D-058).
"""

from __future__ import annotations

import os

SNAPSHOT: dict[str, str] = dict(os.environ)
