# Guide to Reproducing BLIP-2 Results in Google Colab

This guide provides a detailed walkthrough for reproducing results from the BLIP-2 model (Bootstrap Language-Image Pre-training with Frozen Image Encoders and Large Language Models) using Google Colab. It is based on the technical presentation of BLIP-2 and the official implementation from the [LAVIS repository](https://github.com/salesforce/LAVIS/tree/main/projects/blip2).

## Table of Contents

- [Guide to Reproducing BLIP-2 Results in Google Colab](#guide-to-reproducing-blip-2-results-in-google-colab)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Preparing the Google Colab Environment](#preparing-the-google-colab-environment)
  - [Loading and Initializing the BLIP-2 Model](#loading-and-initializing-the-blip-2-model)
  - [Basic Features of BLIP-2](#basic-features-of-blip-2)
    - [Loading and Processing an Image](#loading-and-processing-an-image)
    - [Image Captioning](#image-captioning)
    - [Visual Question Answering](#visual-question-answering)
    - [Instruction-based Text Generation](#instruction-based-text-generation)
  - [Advanced Usage Examples](#advanced-usage-examples)
  - [Testing and Comparison with Paper Results](#testing-and-comparison-with-paper-results)
  - [Q-Former Architecture](#q-former-architecture)
  - [Two-Stage Pretraining Strategy](#two-stage-pretraining-strategy)
    - [Stage 1: Vision-Language Representation Learning](#stage-1-vision-language-representation-learning)
    - [Stage 2: Vision-to-Language Generation](#stage-2-vision-to-language-generation)
  - [Conclusion](#conclusion)

## Introduction

BLIP-2 is an efficient and versatile method for pretraining multimodal models that integrate computer vision and natural language processing. Its core innovation is the use of frozen pretrained models for both modalities, significantly reducing computational cost and improving performance.

The key component is a lightweight Querying Transformer (Q-Former), trained using a two-stage strategy to bridge the modality gap. The method shows superior performance across various vision-language tasks while using far fewer trainable parameters than existing approaches.

## Preparing the Google Colab Environment

To reproduce BLIP-2 results in Google Colab:

1. Create a new Google Colab notebook.
2. Enable GPU (Runtime > Change runtime type > GPU).
3. Install dependencies and clone the LAVIS repository:

```python
!git clone https://github.com/salesforce/LAVIS.git
%cd LAVIS

!pip install -e .
!pip install transformers==4.28.0
!pip install accelerate
!pip install fairscale==0.4.4
!pip install timm==0.4.12
!pip install pycocoevalcap
!pip install opencv-python==4.10.0.84
```

`fairscale==0.4.4` and `timm==0.4.12` match [LAVIS's own `requirements.txt`](https://github.com/salesforce/LAVIS/blob/main/requirements.txt) (checked 2026-08-30); that file doesn't declare an `accelerate` version at all, so it's left unpinned above rather than guessed. LAVIS's current `main` pins `transformers==4.33.2`, so `transformers==4.28.0` above is this guide's own tested override, not LAVIS's default -- if you hit compatibility errors, try LAVIS's pin instead. `git clone ... LAVIS.git` and `pip install -e .` install whatever LAVIS's `main` branch currently is, not a pinned commit -- this is a real, unresolved reproducibility gap in the original guide.

**Last verified:** not re-run in this pass. The pins above were cross-checked against LAVIS's own `requirements.txt` (2026-08-30), but the guide itself has not been re-executed end to end since 2025-04-20. Run it in Colab with a GPU to confirm current output before relying on it.

## Loading and Initializing the BLIP-2 Model

After installing dependencies, load the pretrained BLIP-2 model:

```python
import torch
from PIL import Image
import requests
from lavis.models import load_model_and_preprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_opt",
    model_type="pretrain_opt2.7b",
    is_eval=True,
    device=device
)
```

You can also load BLIP-2 with T5 or other available models:

```python
model_t5, vis_processors_t5, _ = load_model_and_preprocess(
    name="blip2_t5",
    model_type="pretrain_flant5xl",
    is_eval=True,
    device=device
)
```

## Basic Features of BLIP-2

### Loading and Processing an Image

```python
def load_image_from_url(url):
    raw_image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return raw_image

image_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
raw_image = load_image_from_url(image_url)
display(raw_image)

image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
```

### Image Captioning

```python
caption = model.generate({"image": image})
print(f"Generated caption: {caption[0]}")
```

### Visual Question Answering

```python
def answer_question(model, image, question):
    answer = model.generate({"image": image, "prompt": f"Question: {question} Answer:"})
    return answer[0]

questions = [
    "What is shown in the photo?",
    "What color is the bus?",
    "How many people are visible in the image?",
    "What is the weather like in the image?"
]

for question in questions:
    answer = answer_question(model, image, question)
    print(f"Question: {question}")
    print(f"Answer: {answer}\n")
```

### Instruction-based Text Generation

```python
def generate_text_with_instruction(model, image, instruction):
    generated_text = model.generate({"image": image, "prompt": instruction})
    return generated_text[0]

instructions = [
    "Describe in detail what is happening in the image.",
    "Write a short story based on this image.",
    "List all the objects you can see in the image.",
    "Explain what emotions this image evokes and why."
]

for instruction in instructions:
    generated_text = generate_text_with_instruction(model, image, instruction)
    print(f"Instruction: {instruction}")
    print(f"Generated text: {generated_text}\n")
```

## Advanced Usage Examples

For more advanced BLIP-2 capabilities, use functions from `blip2_advanced_examples.py`. It contains:

- Visualization of the two-stage pretraining strategy
- Task demonstrations
- Q-Former architecture analysis
- Comparison with other methods
- Visual feature extraction

Example:

```python
!wget https://raw.githubusercontent.com/username/repo/main/blip2_advanced_examples.py
from blip2_advanced_examples import *

explain_two_stage_pretraining()
results = demonstrate_blip2_tasks(model, image)
analyze_qformer_architecture(model)
compare_with_other_methods()
features = extract_visual_features(model, image)
```

## Testing and Comparison with Paper Results

Use `blip2_demo.py` for model testing and benchmarking:

```python
!wget https://raw.githubusercontent.com/username/repo/main/blip2_demo.py
from blip2_demo import *

results = run_comprehensive_test(model, vis_processors, device)
compare_with_paper_results()
analyze_model_performance(model, device)
```

Key results from the BLIP-2 paper:

1. BLIP-2 outperforms Flamingo80B by 8.7% in zero-shot VQAv2 with 54x fewer trainable parameters.
2. Achieves strong image captioning results on COCO.
3. Uses significantly fewer trainable parameters (188M in Q-Former).
4. Efficient use of frozen pretrained models reduces compute cost while boosting performance.

## Q-Former Architecture

Q-Former is the key trainable module that bridges the frozen image encoder and LLM. It extracts a fixed number of output features regardless of image resolution.

It consists of two transformer submodules with shared self-attention layers:

1. **Image Transformer** interacts with the frozen image encoder.
2. **Text Transformer** functions as both a text encoder and decoder.

Learnable query embeddings serve as input to the image transformer, interacting with one another through self-attention and with image features via cross-attention layers inserted every other transformer block.

Run the following to analyze the architecture:

```python
analyze_qformer_architecture(model)
```

## Two-Stage Pretraining Strategy

BLIP-2 uses a two-stage pretraining strategy to align frozen unimodal models.

### Stage 1: Vision-Language Representation Learning

Q-Former connects to the frozen image encoder and is trained on image-text pairs with three objectives:

1. **Image-Text Contrastive (ITC)**: Align image and text representations.
2. **Image-to-Text Generation (ITG)**: Enable text generation from visual input.
3. **Image-Text Matching (ITM)**: Learn fine-grained alignment.

### Stage 2: Vision-to-Language Generation

Q-Former connects to the frozen LLM. Its visual outputs are projected into the LLM embedding space and used as a prefix for conditional generation.

Visualize the strategy:

```python
explain_two_stage_pretraining()
```

## Conclusion

This guide demonstrated how to reproduce BLIP-2 results in Google Colab, explore its architecture, and test its capabilities.

BLIP-2 is a major step forward in vision-language pretraining. It leverages frozen unimodal models and a lightweight Q-Former to reduce computational cost while achieving or exceeding the performance of state-of-the-art models.

**Key advantages:**

1. Efficient use of frozen pretrained vision and language models, bridged by a two-stage pretrained Q-Former.
2. Zero-shot image-to-text generation with natural language instructions.
3. Greater compute efficiency through lightweight training modules.

For hands-on use, see the prepared `blip2_colab.ipynb` notebook with all steps and examples.
