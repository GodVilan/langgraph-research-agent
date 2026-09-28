"""`make verify-deploy` renders the Space card from `scripts/deploy_space.py` *as committed at the
revision*. From D-055 that file defines a dataclass, which failed to load under the renderer until
the module was registered in sys.modules (D-061)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def test_the_card_renders_from_head_as_committed() -> None:
    from scripts import deploy_space
    from scripts.verify_deploy import space_card_at

    host = "godvillain-scholium.hf.space"
    card = space_card_at("HEAD", host)
    assert card == deploy_space.SPACE_CARD.format(github=deploy_space.GITHUB, host=host)
    assert not any(name.startswith("deploy_space_at_") for name in sys.modules)
