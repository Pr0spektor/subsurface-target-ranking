"""Strip generator metadata from the published charts.

Matplotlib writes a `Software: Matplotlib version ...` tEXt chunk into every PNG. It
carries no personal data, but a published figure should say nothing about the machine
that produced it, so this removes every text chunk and rewrites the file byte-for-byte
otherwise unchanged. Run it after regenerating charts:

    python src/analysis.py && ... && python src/strip_meta.py
"""
import glob, os, struct, sys

TEXT_CHUNKS = (b"tEXt", b"iTXt", b"zTXt")
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _chunks(data):
    i = len(PNG_MAGIC)
    while i < len(data):
        length = struct.unpack(">I", data[i:i + 4])[0]
        ctype = data[i + 4:i + 8]
        yield ctype, data[i:i + 12 + length]
        i += 12 + length


def strip(path):
    """Remove text chunks from one PNG. Returns bytes saved."""
    data = open(path, "rb").read()
    if not data.startswith(PNG_MAGIC):
        return 0
    out = bytearray(PNG_MAGIC)
    removed = 0
    for ctype, raw in _chunks(data):
        if ctype in TEXT_CHUNKS:
            removed += len(raw)
            continue
        out += raw
    if removed:
        open(path, "wb").write(bytes(out))
    return removed


def strip_dir(directory):
    total = files = 0
    for p in sorted(glob.glob(os.path.join(directory, "*.png"))):
        n = strip(p)
        if n:
            files += 1
            total += n
    return files, total


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "results")
    files, total = strip_dir(d)
    print("stripped metadata from %d PNGs (%d bytes removed)" % (files, total))
