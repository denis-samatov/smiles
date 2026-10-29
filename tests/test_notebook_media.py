"""Exercise notebook media assembly without downloading a vision/language model."""
import ast
import json
from pathlib import Path

import pytest
import torch

NOTEBOOK = Path(__file__).resolve().parents[1] / "Flamingo/flamingo_colab.ipynb"


class Tokens(dict):
    def to(self, device):
        return self


class Tokenizer:
    def __call__(self, prompt, **kwargs):
        return Tokens(input_ids=torch.tensor([[1]]), attention_mask=torch.tensor([[1]]))

    def decode(self, output, **kwargs):
        return output


class ShapeModel:
    def generate(self, vision_x, **kwargs):
        return [str(tuple(vision_x.shape))]


@pytest.mark.parametrize("function_name", ["generate_caption", "answer_question"])
@pytest.mark.parametrize("count", [0, 2, 4])
def test_caption_media_uses_arguments_and_six_dimensional_layout(count, function_name):
    scope = {"torch": torch, "device": "cpu", "processed_example_images": [],
             "processed_test_image": torch.zeros(1, 3, 2, 2)}
    for cell in json.loads(NOTEBOOK.read_text())["cells"]:
        source = "".join(cell.get("source", []))
        if source.startswith("def " + function_name):
            tree = ast.parse(source)
            exec(compile(ast.Module(body=[tree.body[0]], type_ignores=[]), str(NOTEBOOK), "exec"), scope)
    images = [torch.ones(1, 3, 2, 2) for _ in range(count)]
    arguments = [ShapeModel(), Tokenizer(), images,
                 ["<image>example" for _ in images], torch.ones(1, 3, 2, 2)]
    if function_name == "answer_question":
        arguments.append("What is visible?")
    result = scope[function_name](*arguments)
    assert result == str((1, count + 1, 1, 3, 2, 2))
