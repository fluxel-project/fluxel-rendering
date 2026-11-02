"""Losslessly encode retained fixture P6 pixels as PNG for visual inspection."""

import argparse
from pathlib import Path
import struct
import zlib


def convert(source: Path, destination: Path) -> None:
    magic, dimensions, maximum, pixels = source.read_bytes().split(b"\n", 3)
    if magic != b"P6" or maximum != b"255":
        raise ValueError("expected the integration fixture's 8-bit P6 encoding")
    width, height = map(int, dimensions.split())
    if not 0 < width * height <= 4_194_304 or len(pixels) != width * height * 3:
        raise ValueError("invalid fixture dimensions or pixel byte length")

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data)))

    scanlines = b"".join(
        b"\0" + pixels[row * width * 3:(row + 1) * width * 3]
        for row in range(height)
    )
    destination.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(scanlines))
        + chunk(b"IEND", b"")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    convert(arguments.source, arguments.destination)
