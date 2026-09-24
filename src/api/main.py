"""ASGI entry point: ``uvicorn src.api.main:app``.

Separate from ``app.py`` so importing the app factory — tests, the CLI — builds nothing.
One worker per container, deliberately: each worker would load its own 1.3 GB embedding
model and keep its own per-IP buckets and concurrency gate, so two workers would double the
memory and halve every limit's meaning.
"""

from __future__ import annotations

import logging

from src.api.app import create_app

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")

app = create_app()
