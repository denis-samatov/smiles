# Smiles

**Colab walkthroughs for four vision-language models**

This repository is a set of four Colab walkthroughs of existing, third-party vision-language models -- not a suite the author built. Each folder contains a Google Colab notebook, detailed instructions, and code examples for exploring that model's capabilities. All 5 notebooks have their outputs cleared (the cleanest notebook hygiene on the account), which also means there's no in-repo evidence any of them currently runs -- see each model's own README for what was and wasn't re-verified.

## Models Included

| Model | What it demonstrates | Guide last verified |
|---|---|---|
| [**BLIP-2**](Blip2/README.md) | Bootstrapped Language-Image Pre-training with frozen vision encoders and large language models. | Not re-run since 2025-04-20; dependency pins cross-checked 2026-08-30. |
| [**Open-Flamingo**](Flamingo/README.md) | Few-shot learning for vision-language tasks using visual and textual prompts. | Not re-run since 2025-04-20; dependency pins cross-checked 2026-08-30. |
| [**LLaVA**](LLaVA/README.md) | Large Language and Vision Assistant demo with interactive Gradio interface. | Not re-run since 2025-04-20; dependency pins cross-checked 2026-08-30. |
| [**MedCLIP**](MedClip/README.md) | Contrastive learning from unpaired medical images and texts for diagnostic applications. | 2026-08-30 -- real chest-X-ray/text similarity output preserved in the notebook. |

Only MedCLIP has genuine evidence of a completed run in this pass; the other three guides' dependency versions were checked against their upstream projects but the notebooks themselves were not re-executed (these are multi-GB models needing a GPU). See each model's README for details.

## Repository Structure
```text
Smiles/
├── Blip2/                   # BLIP-2 Colab guide, examples, tests
│   ├── README.md            # Instructions to reproduce BLIP-2 results
│   ├── blip2_colab.ipynb    # Colab notebook
│   ├── blip2_advanced_examples.py
│   └── blip2_demo.py
├── Flamingo/                # Open-Flamingo Colab instructions
│   ├── README.md
│   └── flamingo_colab.ipynb
├── LLaVA/                   # LLaVA demonstration guide and notebook
│   ├── README.md
│   └── llava_demo.ipynb
├── MedClip/                 # MedCLIP Colab notebook and documentation
│   ├── README.md
│   └── MedClip Colab.ipynb
└── Technical Presentation_Smiles.pdf  # Overview and key results
```

## Getting Started

Choose one of the following options:

1. **Google Colab**
   - Open the notebook in each model folder.
   - Enable GPU runtime (Runtime > Change runtime type > GPU).
   - Run cells sequentially to reproduce results.

2. **Local Environment**
   - Clone the repository:
     ```bash
     git clone https://github.com/denis-samatov/smiles.git
     cd smiles
     ```
   - Install Python 3.8+.
   - Follow dependency instructions in each folder's README.md.

## Usage

1. Navigate to a folder for the model you wish to explore.
2. Read the provided README.md for detailed setup and execution steps.
3. Run the Colab notebook or Python scripts to generate captions, answer questions, and visualize outputs.
4. Explore advanced examples and benchmark tests where available.

## Technical Presentation

- An overview of the four models covered here, their architectures, and reported benchmark results is available in [Technical Presentation_Smiles.pdf](Technical%20Presentation_Smiles.pdf).
