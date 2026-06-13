#!/usr/bin/env python3
"""Download MNIST and export a balanced subset as CSV for C++ training."""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "images": ("train-images-idx3-ubyte.gz", 16),
    "labels": ("train-labels-idx1-ubyte.gz", 8),
}
SAMPLES_PER_CLASS = 100
OUTPUT = Path(__file__).resolve().parent.parent / "data" / "mnist_subset.csv"


def download(name: str) -> bytes:
    filename, _ = FILES[name]
    url = BASE_URL + filename
    print(f"Downloading {url} ...")
    with urllib.request.urlopen(url, timeout=60) as response:
        return gzip.decompress(response.read())


def parse_images(data: bytes) -> list[list[int]]:
    offset = FILES["images"][1]
    magic, count, rows, cols = struct.unpack_from(">IIII", data, 0)
    if magic != 2051:
        raise ValueError("Invalid MNIST image magic number")

    images: list[list[int]] = []
    index = offset
    for _ in range(count):
        pixel_count = rows * cols
        pixels = list(data[index : index + pixel_count])
        images.append(pixels)
        index += pixel_count
    return images


def parse_labels(data: bytes) -> list[int]:
    offset = FILES["labels"][1]
    magic, count = struct.unpack_from(">II", data, 0)
    if magic != 2049:
        raise ValueError("Invalid MNIST label magic number")
    return list(data[offset : offset + count])


def main() -> None:
    images = parse_images(download("images"))
    labels = parse_labels(download("labels"))

    if len(images) != len(labels):
        raise RuntimeError("Image and label counts do not match")

    per_class: dict[int, list[tuple[int, list[int]]]] = {digit: [] for digit in range(10)}
    for idx, (label, pixels) in enumerate(zip(labels, images)):
        bucket = per_class[label]
        if len(bucket) < SAMPLES_PER_CLASS:
            bucket.append((idx, pixels))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as file:
        for digit in range(10):
            for _, pixels in per_class[digit]:
                file.write(str(digit))
                for pixel in pixels:
                    file.write(f",{pixel}")
                file.write("\n")
                rows_written += 1

    print(f"Wrote {rows_written} samples to {OUTPUT}")


if __name__ == "__main__":
    main()
