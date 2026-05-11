import hashlib
from pathlib import Path

def sha256_file(path: Path) -> str:
    """Stream the file in 64KB chunks — never loads whole file into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()