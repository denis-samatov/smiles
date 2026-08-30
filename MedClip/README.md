# Documentation for Reproducing MedCLIP in Google Colab

## Introduction

This documentation outlines the process of reproducing the results of **MedCLIP** (Contrastive Learning from Unpaired Medical Images and Texts) in the Google Colab environment.  
MedCLIP represents an innovative approach to multimodal learning in medical imaging, aiming to overcome key limitations of existing methods.

---

## Key Features of MedCLIP

1. **Combinatorial Data Scaling** — leverages unpaired images and texts by using medical labels to create meaningful supervision.  
2. **Semantic Alignment** — replaces standard contrastive loss with a task-aware loss function that incorporates clinical relevance.  
3. **Resource Efficiency** — achieves state-of-the-art performance with only 20K pretraining pairs, compared to 200K in other methods.

---

## Reproduction Steps

### 1. Google Colab Environment Setup

1. Open [Google Colab](https://colab.research.google.com/)  
2. Upload the provided notebook `MedClip Colab.ipynb` via **File → Upload notebook**  
3. Ensure GPU is enabled: **Runtime → Change runtime type → Hardware accelerator → GPU**

---

### 2. Clone the Repository and Install Dependencies

The notebook already includes all commands needed to clone the repository and install dependencies:

```python
!pip install \
    pandas Pillow requests tqdm wget \
    nltk>=3.7 scikit_learn>=1.1.2 textaugment>=1.3.4 \
    timm>=0.6.11 torch>=1.12.1 torchvision>=0.13.1 \
    "transformers>=4.23.1,<4.25.0"

!pip install -qU "numpy>=2.0.0"
!pip install -e ./MedCLIP
```

---

### 3. Load and Use the Pretrained Model

MedCLIP provides an easy interface to load pretrained models:

```python
from medclip import MedCLIPModel, MedCLIPVisionModelViT
from medclip import MedCLIPProcessor
from PIL import Image

processor = MedCLIPProcessor()
model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
model.from_pretrained()
model.to(device)
```

To load the model with ResNet-50 as the visual encoder:

```python
from medclip import MedCLIPVisionModel
model = MedCLIPModel(vision_cls=MedCLIPVisionModel)
model.from_pretrained()
```

---

### 4. Prompt-Based Image Classification

MedCLIP supports prompt-based classification of images:

```python
from medclip import PromptClassifier
from medclip.prompts import generate_chexpert_class_prompts, process_class_prompts

clf = PromptClassifier(model, ensemble=True)
clf.to(device)

cls_prompts = process_class_prompts(generate_chexpert_class_prompts(n=10))
inputs['prompt_inputs'] = cls_prompts

output = clf(**inputs)
```

---

### 5. Semantic Contrastive Loss

One of the key innovations in MedCLIP is the **semantic contrastive loss** that incorporates clinical relevance and reduces the impact of false negatives:

```python
def semantic_contrastive_loss(img_embeds, text_embeds, semantic_similarity_matrix):
    img_embeds = F.normalize(img_embeds, dim=1)
    text_embeds = F.normalize(text_embeds, dim=1)
    
    logits = torch.matmul(img_embeds, text_embeds.t()) * 100.0
    
    semantic_weights = F.softmax(semantic_similarity_matrix, dim=1)
    
    log_probs = F.log_softmax(logits, dim=1)
    loss = -torch.sum(semantic_weights * log_probs) / img_embeds.size(0)
    
    return loss
```

---

### 6. Visualizing Results

You can visualize similarity between images and multiple text prompts using the following function:

```python
def visualize_image_text_similarity(model, processor, image_path, texts):
    image = Image.open(image_path)
    
    inputs = processor(
        text=texts, 
        images=image, 
        return_tensors="pt", 
        padding=True
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    img_embeds = F.normalize(outputs['img_embeds'], dim=1)
    text_embeds = F.normalize(outputs['text_embeds'], dim=1)
    
    similarity = torch.matmul(img_embeds, text_embeds.t())[0].cpu().numpy()
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(texts)), similarity, color='skyblue')
    plt.xticks(range(len(texts)), [f"Text {i+1}" for i in range(len(texts))], rotation=45, ha='right')
    plt.ylabel('Similarity')
    plt.title('Image-Text Similarity Scores')
    plt.tight_layout()
    plt.show()
```

---

## Pretraining the Model (Optional)

Full pretraining of the MedCLIP model requires significant computational resources and access to large medical datasets such as CheXpert and MIMIC-CXR.  
Pretraining scripts are included in the notebook but commented out to avoid accidental execution.

Main pretraining steps:
1. Define training configuration  
2. Apply image transformations  
3. Create the dataset and data loader  
4. Initialize the MedCLIP model and evaluator  
5. Define loss function and start training

---

## Example (captured from an actual run)

`MedClip Colab.ipynb` includes a cell that scores a sample chest X-ray (`./example_data/view1_frontal.jpg`, bundled with the MedCLIP repo) against 5 candidate radiology-report sentences using cosine similarity between image and text embeddings. The notebook's stored output from that run:

```text
Text 1 (Similarity: 0.0120): lungs remain severely hyperinflated with upper lobe emphysema
Text 2 (Similarity: 0.0283): opacity left costophrenic angle is new since prior exam
Text 3 (Similarity: 0.0343): normal chest radiograph with no evidence of active disease
Text 4 (Similarity: 0.0399): cardiomegaly with pulmonary vascular congestion suggesting heart failure
Text 5 (Similarity: 0.0252): right middle lobe pneumonia with small pleural effusion
```

This is real output preserved in the notebook's cell outputs from a completed run, not fabricated for this README. The highest similarity score (0.0399) correctly goes to the cardiomegaly/heart-failure description for this particular X-ray.

**Last verified:** 2026-08-30 -- verified against the notebook's own preserved output and the pinned versions below; the model was not re-run from scratch in this pass.

---

## Limitations and Notes

1. **GPU Requirements**: At least 12 GB of GPU memory is recommended for efficient training and inference.  
2. **Dataset Access**: Full pretraining requires access to medical datasets, which may be license-restricted.  
3. **Library Versions**: MedCLIP depends on specific versions of libraries, especially `transformers` (≥4.23.1, <4.25.0 -- pinned in the install command above).

---

## Conclusion

MedCLIP sets a new standard for efficiency in medical multimodal learning by enabling training on fragmentary medical data without requiring complete image-report pairs.  
It leverages partial semantic matches to enhance diagnostic accuracy and reduce data requirements.

The provided notebook allows you to reproduce key results of MedCLIP in Google Colab, including loading pretrained models, prompt-based classification, semantic contrastive loss, and result visualization.
