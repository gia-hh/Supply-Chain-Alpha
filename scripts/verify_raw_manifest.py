from __future__ import annotations

import argparse
from pathlib import Path

from supply_chain_alpha.data.disclosures import verify_raw_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "manifest", type=Path, default=Path("data/raw/MANIFEST.json"), nargs="?"
    )
    parser.add_argument(
        "--inventory-root",
        type=Path,
        help="Directory that the manifest must cover exactly",
    )
    args = parser.parse_args()
    verify_raw_manifest(args.manifest, inventory_root=args.inventory_root)
    print(f"Raw manifest verified: {args.manifest}")


if __name__ == "__main__":
    main()
