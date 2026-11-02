# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download a package's source distribution exactly as pinned in uv.lock.

PyPI publishes PyOptiX only as an sdist. CI builds it once on hosted runners,
where compilers and CUDA are available, instead of on the GPU runners.
"""

import argparse
import hashlib
import urllib.request
from pathlib import Path

import tomllib


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("package")
    parser.add_argument("--lock", type=Path, default=Path("uv.lock"))
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--out-dir", type=Path, help="download the sdist here and print its path")
    output.add_argument("--print-hash", action="store_true", help="print the locked hash, e.g. for a cache key")
    args = parser.parse_args()

    with args.lock.open("rb") as file:
        packages = tomllib.load(file)["package"]
    (sdist,) = [package["sdist"] for package in packages if package["name"] == args.package]
    if args.print_hash:
        print(sdist["hash"])
        return
    algorithm, expected = sdist["hash"].split(":", 1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / sdist["url"].rsplit("/", 1)[1]
    with urllib.request.urlopen(sdist["url"]) as response:
        data = response.read()
    if hashlib.new(algorithm, data).hexdigest() != expected:
        raise RuntimeError(f"{algorithm} mismatch for {sdist['url']}")
    path.write_bytes(data)
    print(path)


if __name__ == "__main__":
    main()
