"""Check saved-data integrity and regenerate the reference figure and statistics."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .figure import build

ROOT = Path(__file__).resolve().parents[1]


def verify(output, strict_pixels=True):
    output = Path(output)
    manifest = json.loads((ROOT / "data/SHA256.json").read_text())
    assert len(manifest) == 160, (
        "Expected 60 diagnostic runs and 100 individual-accuracy comparison files."
    )
    for name, expected in manifest.items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        assert actual == expected, f"Data checksum mismatch: {name}"
    build(output)
    expected = json.loads((ROOT / "reference/statistics.json").read_text())
    actual = json.loads((output / "statistics.json").read_text())
    assert actual == expected, "Statistics differ from the reference."
    comparison = json.loads((output / "comparison.json").read_text())
    assert comparison == json.loads((ROOT / "reference/comparison.json").read_text())
    assert (output / "per_seed.csv").read_bytes() == (
        ROOT / "reference/per_seed.csv"
    ).read_bytes()
    reference = np.asarray(Image.open(ROOT / "reference/figure.png").convert("RGBA"))
    regenerated = np.asarray(Image.open(output / "figure.png").convert("RGBA"))
    same_shape = reference.shape == regenerated.shape
    pixels_equal = same_shape and np.array_equal(reference, regenerated)
    report = {
        "data_files_checked": len(manifest),
        "statistics_exact": True,
        "per_seed_csv_exact": True,
        "figure_pixels_exact": bool(pixels_equal),
        "png_bytes_exact": (output / "figure.png").read_bytes()
        == (ROOT / "reference/figure.png").read_bytes(),
        "reference_dimensions": list(reference.shape[:2]),
        "reproduced_dimensions": list(regenerated.shape[:2]),
        "positive_leads": actual["npos"],
        "mean_lead_steps": actual["mean_lead"],
        "bootstrap_ci": [actual["ci_lo"], actual["ci_hi"]],
        "pooled_crossings": actual["leak_cen_n"],
        "no_sharing_crossings": actual["leak_loc_n"],
    }
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    if strict_pixels:
        assert pixels_equal, (
            "Figure pixels differ; use the pinned figure environment and bundled fonts."
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument(
        "--allow-rendering-differences",
        action="store_true",
        help="Still require exact data/statistics; report platform-dependent rendering differences.",
    )
    args = parser.parse_args()
    print(
        json.dumps(verify(args.output, not args.allow_rendering_differences), indent=2)
    )


if __name__ == "__main__":
    main()
