from __future__ import annotations

import argparse
from pathlib import Path

from supply_chain_alpha.pipeline import run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the V2 research pipeline in phase order."
    )
    parser.add_argument("--through-phase", type=int, default=11)
    parser.add_argument("--config", type=Path, default=Path("config/project.yaml"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = run_pipeline(
        root, config_path=args.config, through_phase=args.through_phase
    )
    print(
        f"PIPELINE STOP: phase={result.last_phase} state={result.state} exit_code={int(result.exit_code)}"
    )
    return int(result.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
