# syntax=docker/dockerfile:1.7
#
# arXiv Agent v3 — the serving image.
#
#   docker build -f infra/Dockerfile -t arxiv-agent-v3 .      # from the repo root
#   make docker-build / make docker-run                          # the same, with the flags
#
# Three stages, each for one reason:
#
#   deps    Python dependencies into a venv. torch comes from the CPU-only index: the default
#           wheel carries CUDA libraries a CPU host can never load, several GB of image for
#           nothing.
#   model   BAAI/bge-large-en at a pinned commit, safetensors only. The model repo also holds a
#           1.3 GB pytorch_model.bin of the same weights; downloading both would double the
#           largest layer. Phase 4 decided large over small on evidence (Recall@5 0.267 vs
#           0.174, n=43; v2.1 0.233) — docs/EVALS.md — so the 1.3 GB ships, deliberately.
#   runtime The venv, the model, the source, and the prebuilt FAISS index, checksum-verified
#           against data/INDEX.sha256 so the image provably serves the index Phase 4 measured.
#           Nothing is built or downloaded at container start (HF_HUB_OFFLINE=1 enforces it).
#
# The index is baked in, not built here: a CPU rebuild is behaviourally identical (top-5/10
# rankings on 500 probes, DECISIONS D-006) but not bit-identical, and it takes 15-20 minutes.
# Shipping the measured artifact is both faster and the honest choice.

ARG PYTHON_VERSION=3.13

# ── deps ────────────────────────────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS deps
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2"
WORKDIR /build
COPY pyproject.toml LICENSE ./
COPY src ./src
RUN pip install .

# ── model ───────────────────────────────────────────────────────────────────────────────
FROM deps AS model
# The commit the Phase 4 index was built with. A new upstream commit on `main` cannot change
# what ships; changing this line is a re-embedding decision (D-002), not a routine bump.
ARG EMBEDDING_REPO=BAAI/bge-large-en
ARG EMBEDDING_REVISION=abe7d9d814b775ca171121fb03f394dc42974275
ENV HF_HOME=/opt/hf
RUN python - <<'PY'
import os, pathlib
from huggingface_hub import snapshot_download

repo, rev = os.environ["EMBEDDING_REPO"], os.environ["EMBEDDING_REVISION"]
path = snapshot_download(
    repo,
    revision=rev,
    allow_patterns=["*.json", "model.safetensors", "vocab.txt", "1_Pooling/*"],
)
assert pathlib.Path(path).name == rev, f"resolved {path}, expected commit {rev}"
# sentence-transformers asks for the name, which resolves through refs/main offline. Point
# it at the pinned commit so the name and the weights cannot disagree.
refs = pathlib.Path(os.environ["HF_HOME"]) / "hub" / f"models--{repo.replace('/', '--')}" / "refs"
refs.mkdir(parents=True, exist_ok=True)
(refs / "main").write_text(rev)
print("pinned", repo, "at", rev)
PY

# ── runtime ─────────────────────────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS runtime

# Hugging Face Spaces runs containers as uid 1000; any other host is fine with it too.
# /data is where a persistent volume mounts (ledger, threads). Created here and owned by the
# runtime user, because a fresh named volume takes its ownership from the image's directory —
# without this the volume is root-owned and the service cannot open its own databases.
RUN useradd -m -u 1000 user && mkdir -p /data && chown 1000:1000 /data
COPY --from=deps /opt/venv /opt/venv
COPY --from=model --chown=user /opt/hf /opt/hf

USER user
WORKDIR /home/user/app
COPY --chown=user pyproject.toml LICENSE ./
COPY --chown=user src ./src
COPY --chown=user data/chunks_512.json data/metadata.json data/CORPUS.sha256 data/INDEX.sha256 ./data/
COPY --chown=user data/indices/BGE_cs512.faiss data/indices/BGE_cs512_meta.pkl ./data/indices/

# The build fails here if the corpus or the index differs from what the repo records.
RUN sha256sum -c data/CORPUS.sha256 && sha256sum -c data/INDEX.sha256

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    DEVICE=cpu \
    PORT=7860 \
    # Turns the ephemeral-state guards into refusals: this image will not start with a daily
    # cost ledger that forgets on restart (src/api/ledger.py). Choosing API__LEDGER is the
    # operator's decision and is deliberately not defaulted here.
    API__DEPLOYED=true \
    # Pace model calls below the deploy key's quota — 15 RPM / 500 RPD, read from AI Studio
    # by Srikanth 2026-09-24 (free tier, no-billing project). 10/min plus a burst of 3 admits at
    # most 13 calls in any 60 s window: 2 below the quota, never exactly at it
    # (tests/test_api.py reads this line). 500 RPD is ~140 median queries a day, about where
    # the $0.50 notional ceiling lands (~146): whichever binds first, callers get a 429/503
    # with a reason, not a silent failure.
    AGENT_REQUESTS_PER_MINUTE=10 \
    LANGFUSE_ENVIRONMENT=deployed

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4)"

# One worker: each would load its own copy of the 1.3 GB model and keep its own limiters.
# --no-proxy-headers: X-Forwarded-For is interpreted by the app, with an explicit trusted hop
# count (API__TRUSTED_PROXY_HOPS), never by uvicorn's default rewriting.
CMD ["sh", "-c", "exec uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --no-proxy-headers"]
