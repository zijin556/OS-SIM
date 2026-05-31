from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data_io
from src.tiny_net import build_tiny_unet


def main() -> None:
    parser = argparse.ArgumentParser(description="Optional tiny-net stage. Requires PyTorch.")
    parser.add_argument("--sample", default="细台阶")
    parser.add_argument("--groups", default="ossim_t6_h_3step,ossim_t6_v_3step")
    parser.add_argument("--crop", default="center:256")
    parser.add_argument("--output", default="outputs/tiny_net")
    args = parser.parse_args()
    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        model = build_tiny_unet(in_channels=6, out_channels=2)
    except RuntimeError as exc:
        report = [
            "# Tiny-Net Report",
            "",
            "Status: `skipped_missing_torch`",
            "",
            str(exc),
            "",
            "This is intentional for the current NumPy/SciPy first pass. Do not treat this as a failed experiment result.",
        ]
        (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
        data_io.write_json(out_dir / "metrics.json", {"status": "skipped_missing_torch", "reason": str(exc)})
        data_io.write_json(out_dir / "config.json", vars(args))
        print(f"Skipped tiny-net stage because torch is unavailable. Wrote {out_dir / 'report.md'}")
        return
    report = [
        "# Tiny-Net Report",
        "",
        "Status: `initialized_only`",
        "",
        f"Model class: `{model.__class__.__name__}`",
        "",
        "Training is intentionally not started automatically. The physics-forward stages should be validated first.",
    ]
    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    data_io.write_json(out_dir / "metrics.json", {"status": "initialized_only"})
    data_io.write_json(out_dir / "config.json", vars(args))
    print(f"Initialized tiny-net model and wrote {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()

