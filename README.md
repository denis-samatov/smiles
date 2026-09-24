# Smiles

**Colab walkthroughs for four vision-language models**

This repository contains four Colab walkthroughs of existing, third-party vision-language models; it is not a model suite developed here. Each model folder has one notebook and a setup guide. The BLIP-2, Open-Flamingo, and LLaVA notebooks have no stored execution output. The MedCLIP notebook retains output from an earlier sample run, but none of the four notebooks was re-executed during this documentation review. See each guide for its verification limits.

## Models Included

| Model | What it demonstrates | Guide last verified |
|---|---|---|
| [**BLIP-2**](Blip2/README.md) | Bootstrapped Language-Image Pre-training with frozen vision encoders and large language models. | No saved run output; dependency versions cross-checked 2026-08-30. |
| [**Open-Flamingo**](Flamingo/README.md) | Few-shot learning for vision-language tasks using visual and textual prompts. | No saved run output; dependency versions cross-checked 2026-08-30. |
| [**LLaVA**](LLaVA/README.md) | Large Language and Vision Assistant demo with interactive Gradio interface. | No saved run output; dependency versions cross-checked 2026-08-30. |
| [**MedCLIP**](MedClip/README.md) | Image–text similarity on an upstream sample chest X-ray; this is not a diagnostic validation. | Earlier run output preserved; compared with the guide 2026-09-24. |

Only MedCLIP has saved output demonstrating an earlier run. The other guides' dependency versions were compared with their upstream projects on 2026-08-30, but that check does not establish that the current notebooks run end to end. These models may require a GPU and substantial downloads.

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
   - Follow the model-specific guide and run cells sequentially. Reproduction of upstream paper results is outside the scope of these walkthroughs.

2. **Local Environment**
   - Clone the repository:
     ```bash
     git clone https://github.com/denis-samatov/smiles.git
     cd smiles
     ```
   - Follow the dependency and hardware instructions in the chosen model folder. There is no single pinned local environment for all four models.

## Usage

1. Navigate to a folder for the model you wish to explore.
2. Read the provided README.md for detailed setup and execution steps.
3. Run the relevant notebook cells or the BLIP-2 example scripts to explore the model's behavior.
4. Treat generated text and medical-image similarity scores as demonstrations, not validated model-quality or clinical results.

## Technical Presentation

- An overview of the four models covered here, their architectures, and reported benchmark results is available in [Technical Presentation_Smiles.pdf](Technical%20Presentation_Smiles.pdf).
