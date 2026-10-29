"""
Additional qualitative examples of BLIP-2 usage.
This code can be added to a Google Colab notebook for a deeper exploration of the model's capabilities.
"""
from io import BytesIO

import torch
from PIL import Image
import requests
from lavis.models import load_model_and_preprocess
import matplotlib.pyplot as plt
import numpy as np

# Function to load an image from a URL
def load_image_from_url(url):
    """Download at most 8 MiB with a timeout, then decode at most 16M pixels."""
    content = bytearray()
    with requests.get(url, stream=True, timeout=15) as response:
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=65536):
            if len(content) + len(chunk) > 8 * 1024 * 1024:
                raise ValueError("Image response exceeds download limit")
            content.extend(chunk)
    with Image.open(BytesIO(content)) as image:
        if image.width * image.height > 16_000_000:
            raise ValueError("Image dimensions are too large")
        return image.convert("RGB")

# Function to display an image with a caption
def display_image_with_caption(image, caption):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.title(caption, fontsize=14)
    plt.tight_layout()
    plt.show()

# Function to compare responses from different models
def compare_model_responses(image, question, models_dict):
    results = {}
    for model_name, model in models_dict.items():
        answer = model.generate({"image": image, "prompt": f"Question: {question} Answer:"})
        results[model_name] = answer[0]
    
    for model_name, answer in results.items():
        print(f"{model_name}: {answer}")
    
    return results

# Example of using Q-Former to extract visual features
def extract_visual_features(model, image):
    with torch.no_grad():
        if hasattr(model, 'visual_encoder') and hasattr(model, 'Qformer'):
            image_embeds = model.visual_encoder(image)
            image_features = model.Qformer.bert(
                query_embeds=model.query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image.device),
                return_dict=True,
            ).last_hidden_state
            
            return {
                "image_embeds": image_embeds,
                "q_former_features": image_features
            }
    return None

# Visualization of the two-stage pretraining strategy
def explain_two_stage_pretraining():
    """
    Explains BLIP-2's two-stage pretraining strategy with a visualization.
    """
    plt.figure(figsize=(15, 8))

    # Stage 1
    plt.subplot(1, 2, 1)
    plt.title("Stage 1: Vision-Language Representation Pretraining", fontsize=14)
    plt.text(0.5, 0.8, "Frozen\nImage\nEncoder", ha='center', va='center', fontsize=12, bbox=dict(facecolor='lightblue', alpha=0.5))
    plt.text(0.5, 0.5, "Trainable\nQ-Former", ha='center', va='center', fontsize=12, bbox=dict(facecolor='lightgreen', alpha=0.5))
    plt.text(0.5, 0.2, "Training Objectives:\n- Image-Text Contrastive (ITC)\n- Image-to-Text Generation (ITG)\n- Image-Text Matching (ITM)", ha='center', va='center', fontsize=10, bbox=dict(facecolor='lightyellow', alpha=0.5))
    plt.axis('off')

    # Stage 2
    plt.subplot(1, 2, 2)
    plt.title("Stage 2: Vision-to-Language Generative Pretraining", fontsize=14)
    plt.text(0.5, 0.8, "Frozen\nImage\nEncoder", ha='center', va='center', fontsize=12, bbox=dict(facecolor='lightblue', alpha=0.5))
    plt.text(0.5, 0.6, "Trainable\nQ-Former", ha='center', va='center', fontsize=12, bbox=dict(facecolor='lightgreen', alpha=0.5))
    plt.text(0.5, 0.3, "Frozen\nLarge Language\nModel (LLM)", ha='center', va='center', fontsize=12, bbox=dict(facecolor='lightpink', alpha=0.5))
    plt.text(0.5, 0.1, "Training Objective:\n- Image-to-Text generation using LLM", ha='center', va='center', fontsize=10, bbox=dict(facecolor='lightyellow', alpha=0.5))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    print("Two-stage pretraining strategy of BLIP-2:")
    print("\nStage 1: Vision-Language Representation Pretraining")
    print("- Q-Former connects to a frozen image encoder")
    print("- Trained on image-text pairs")
    print("- Objectives: contrastive learning, text generation, image-text matching")

    print("\nStage 2: Vision-to-Language Generative Pretraining")
    print("- Q-Former connects to a frozen large language model (LLM)")
    print("- Visual features are projected into the LLM embedding space")
    print("- Objective: image-conditioned text generation using LLM")

# Demonstration of various tasks BLIP-2 excels at
def demonstrate_blip2_tasks(model, image):
    """
    Demonstrates various tasks where BLIP-2 performs well.
    """
    tasks = {
        "Caption Generation": {
            "prompt": "",
            "description": "Model generates a description of the image without additional instructions."
        },
        "Visual Question Answering": {
            "prompt": "Question: What is shown in the image? Answer:",
            "description": "Model answers a specific question about the image."
        },
        "Detailed Description": {
            "prompt": "Describe this image in detail, including all visible objects and their interactions.",
            "description": "Model generates a detailed description of all elements in the image."
        },
        "Visual Reasoning": {
            "prompt": "What is unusual or interesting in this image? Explain your observations.",
            "description": "Model analyzes the image and highlights interesting or unusual aspects."
        },
        "Story Generation": {
            "prompt": "Create a short story based on this image.",
            "description": "Model creates a narrative based on the visual content."
        }
    }

    results = {}
    for task_name, task_info in tasks.items():
        print(f"\n{task_name}:")
        print(f"Description: {task_info['description']}")
        print(f"Prompt: \"{task_info['prompt']}\"")
        
        if task_info['prompt'] == "":
            response = model.generate({"image": image})
        else:
            response = model.generate({"image": image, "prompt": task_info['prompt']})
        
        print(f"Model Response: {response[0]}")
        results[task_name] = response[0]
    
    return results

# Analysis of Q-Former architecture
def analyze_qformer_architecture(model):
    """
    Analyze Q-Former architecture and display information about its components.
    """
    print("Q-Former Architecture Analysis in BLIP-2:")

    if hasattr(model, 'Qformer'):
        qformer = model.Qformer

        print(f"\nNumber of Q-Former parameters: {sum(p.numel() for p in qformer.parameters())}")
        print(f"Trainable parameters: {sum(p.numel() for p in qformer.parameters() if p.requires_grad)}")

        if hasattr(model, 'query_tokens'):
            print(f"\nNumber of query tokens: {model.query_tokens.shape[1]}")
            print(f"Query embedding dimension: {model.query_tokens.shape[2]}")

        if hasattr(qformer, 'bert'):
            bert = qformer.bert
            print(f"\nNumber of BERT layers: {len(bert.encoder.layer)}")
            print(f"Hidden size: {bert.config.hidden_size}")
            print(f"Number of attention heads: {bert.config.num_attention_heads}")

            cross_attention_layers = [i for i, layer in enumerate(bert.encoder.layer) if hasattr(layer, 'crossattention')]
            if cross_attention_layers:
                print(f"\nCross-attention layers: {cross_attention_layers}")
                print("Cross-attention is inserted every second transformer block.")
    else:
        print("Q-Former not found in the model.")

# Comparison of BLIP-2 with other methods
def compare_with_other_methods():
    """
    Compare BLIP-2 with other pretraining methods for vision-language tasks.
    """
    methods = {
        "BLIP-2": {
            "Trainable Params": "188M (Q-Former)",
            "Frozen Components": "Yes (Image Encoder and LLM)",
            "Pretraining Strategy": "Two-stage",
            "Effectiveness": "High",
            "Notes": "Lightweight Q-Former as bridging module"
        },
        "Flamingo": {
            "Trainable Params": "10B+",
            "Frozen Components": "Partially",
            "Pretraining Strategy": "End-to-end",
            "Effectiveness": "Medium",
            "Notes": "Cross-attention adapters"
        },
        "CLIP": {
            "Trainable Params": "All parameters",
            "Frozen Components": "No",
            "Pretraining Strategy": "Contrastive",
            "Effectiveness": "Medium",
            "Notes": "Dual encoders for image and text"
        },
        "ALBEF": {
            "Trainable Params": "All parameters",
            "Frozen Components": "No",
            "Pretraining Strategy": "Multi-task",
            "Effectiveness": "Medium",
            "Notes": "Representation alignment mechanism"
        }
    }

    print("Comparison of BLIP-2 with other pretraining methods:\n")
    print("{:<15} {:<25} {:<35} {:<25} {:<15} {:<40}".format(
        "Method", "Trainable Params", "Frozen Components",
        "Pretraining Strategy", "Effectiveness", "Notes"
    ))
    print("-" * 150)

    for method, details in methods.items():
        print("{:<15} {:<25} {:<35} {:<25} {:<15} {:<40}".format(
            method,
            details["Trainable Params"],
            details["Frozen Components"],
            details["Pretraining Strategy"],
            details["Effectiveness"],
            details["Notes"]
        ))

# Example usage in Google Colab:
"""
# Download additional examples
!wget https://raw.githubusercontent.com/username/repo/main/blip2_advanced_examples.py
from blip2_advanced_examples import *

# Using the functions
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_opt", 
    model_type="pretrain_opt2.7b", 
    is_eval=True, 
    device=device
)

# Load an image
image_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
raw_image = load_image_from_url(image_url)
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

# Visualize two-stage pretraining strategy
explain_two_stage_pretraining()

# Demonstrate various tasks
results = demonstrate_blip2_tasks(model, image)

# Analyze Q-Former architecture
analyze_qformer_architecture(model)

# Compare with other methods
compare_with_other_methods()

# Extract visual features
features = extract_visual_features(model, image)
"""
