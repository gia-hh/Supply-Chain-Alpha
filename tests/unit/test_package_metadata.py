from __future__ import annotations

from importlib.metadata import version

import supply_chain_alpha


def test_runtime_and_distribution_versions_agree() -> None:
    assert supply_chain_alpha.__version__ == version("supply-chain-alpha") == "0.2.0"
