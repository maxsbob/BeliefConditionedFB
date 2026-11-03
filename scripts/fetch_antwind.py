"""Download the authors' public dataset and record a numeric-array-only manifest.

uv run --with gdown scripts/fetch_antwind.py
"""

import argparse
from pathlib import Path

import numpy as np

from bfb.experiment import atomic_json, file_sha256

URL = "https://drive.google.com/drive/folders/1ub8KdqcjgNLZv1aX6TX5mRM6RUxC8_vi"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/antwind_official"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.verify_only:
        import gdown

        args.out.mkdir(parents=True, exist_ok=True)
        gdown.download_folder(URL, output=str(args.out), quiet=False, remaining_ok=False)
    manifest = {}
    for i in range(20):
        path = args.out / f"default_ind_{i}.npz"
        with np.load(path, allow_pickle=False) as data:
            schema = {}
            for key in ("observation", "next_observation", "actions", "reward", "done"):
                array = data[key]
                if not np.isfinite(array).all():
                    raise ValueError(f"nonfinite data: {path}/{key}")
                schema[key] = {"shape": list(array.shape), "dtype": str(array.dtype)}
            if data["observation"].shape != (200, 256, 105) or data["actions"].shape != (200, 256, 8):
                raise ValueError(f"unexpected dataset schema: {path}")
        manifest[path.name] = {"sha256": file_sha256(path), "schema": schema}
    atomic_json(args.out / "manifest.json", {"source": URL, "files": manifest, "ignored_object_arrays": ["info"]})
    print(f"Verified {len(manifest)} contexts; manifest: {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
