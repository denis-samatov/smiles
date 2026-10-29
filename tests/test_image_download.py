"""Image utility contract with an in-memory HTTP response; never uses the network."""
import ast
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "Blip2/blip2_demo.py"


class Response:
    def __init__(self, data):
        self.data = data
        self.raw = BytesIO(data)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self.data


def loader(data):
    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "load_image_from_url")
    scope = {"Image": Image, "BytesIO": BytesIO,
             "requests": SimpleNamespace(get=lambda *a, **kw: Response(data))}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), scope)
    return scope["load_image_from_url"]


def test_small_image_decodes_from_bounded_stream():
    stream = BytesIO()
    Image.new("RGB", (1, 1), "red").save(stream, format="PNG")
    assert loader(stream.getvalue())("https://example.invalid/fixture").size == (1, 1)


def test_oversized_response_is_rejected_before_decode():
    with pytest.raises(ValueError, match="large|limit"):
        loader(b"x" * (8 * 1024 * 1024 + 1))("https://example.invalid/fixture")
