"""
Test examples for validating the BLIP-2 model and comparing with results from the original paper.
This code can be added to a Google Colab notebook to verify reproduction of the results.
"""

import torch
from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np
from lavis.models import load_model_and_preprocess
import json
import os

# Function to load an image from URL
def load_image_from_url(url):
    raw_image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return raw_image

# Function to visualize results
def visualize_results(image, caption, answers=None):
    plt.figure(figsize=(12, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Caption: {caption}", fontsize=14)
    
    if answers:
        answer_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in answers.items()])
        plt.figtext(0.5, 0.01, answer_text, ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

# Test images used in the BLIP-2 paper or similar ones
test_images = [
    {
        "url": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg",
        "description": "Standard BLIP-2 demo example (bus on a street)"
    },
    {
        "url": "https://images.unsplash.com/photo-1541963463532-d68292c34b19?q=80&w=1000",
        "description": "Book on a table (object recognition test)"
    },
    {
        "url": "https://images.unsplash.com/photo-1618588507085-c79565432917?q=80&w=1000",
        "description": "Nature landscape (scene description test)"
    },
    {
        "url": "https://images.unsplash.com/photo-1501854140801-50d01698950b?q=80&w=1000",
        "description": "Sunset over the ocean (atmosphere/emotion test)"
    },
    {
        "url": "https://images.unsplash.com/photo-1560343776-97e7d202ff0e?q=80&w=1000",
        "description": "Group of people (social interaction recognition test)"
    }
]

# Standard questions for VQA testing
standard_questions = [
    "What is shown in the photo?",
    "Describe the main objects in the image.",
    "What mood does this image convey?",
    "What is unusual in this image?",
    "Where might this photo have been taken?"
]

# Instructions for instruction-based text generation tasks
test_instructions = [
    "Describe this image in detail.",
    "Write a short story based on this image.",
    "List all objects you see in the image.",
    "Explain what emotions this image evokes and why.",
    "Describe this image as if explaining it to a blind person."
]

# Function to perform full model testing
def run_comprehensive_test(model, vis_processors, device, save_results=True):
    results = {}
    
    for i, image_info in enumerate(test_images):
        print(f"\n\nTesting image {i+1}: {image_info['description']}")
        print(f"URL: {image_info['url']}")
        
        try:
            raw_image = load_image_from_url(image_info['url'])
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
            
            image_results = {
                "description": image_info['description'],
                "url": image_info['url'],
                "captioning": {},
                "vqa": {},
                "instruction_generation": {}
            }
            
            # 1. Caption generation
            print("\n1. Caption generation:")
            caption = model.generate({"image": image})
            print(f"Generated caption: {caption[0]}")
            image_results["captioning"]["standard"] = caption[0]
            
            # 2. VQA
            print("\n2. Visual Question Answering:")
            vqa_results = {}
            for question in standard_questions:
                answer = model.generate({"image": image, "prompt": f"Question: {question} Answer:"})
                print(f"Question: {question}")
                print(f"Answer: {answer[0]}")
                vqa_results[question] = answer[0]
            image_results["vqa"] = vqa_results
            
            # 3. Instruction-based generation
            print("\n3. Instruction-based text generation:")
            instruction_results = {}
            for instruction in test_instructions:
                generated_text = model.generate({"image": image, "prompt": instruction})
                print(f"Instruction: {instruction}")
                print(f"Generated text: {generated_text[0]}")
                instruction_results[instruction] = generated_text[0]
            image_results["instruction_generation"] = instruction_results
            
            # Visualize results
            visualize_results(raw_image, caption[0], {standard_questions[0]: vqa_results[standard_questions[0]]})
            
            # Add results to overall output
            results[f"image_{i+1}"] = image_results
            
        except Exception as e:
            print(f"Error processing image {i+1}: {e}")
            results[f"image_{i+1}"] = {"error": str(e)}
    
    if save_results:
        with open("blip2_test_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("\nTest results saved to 'blip2_test_results.json'")
    
    return results

# Function to compare with results reported in the BLIP-2 paper
def compare_with_paper_results():
    expected_results = {
        "VQAv2 (zero-shot)": "According to the paper, BLIP-2 outperforms Flamingo80B by 8.7% on VQAv2 in zero-shot setting while using 54× fewer trainable parameters.",
        "COCO Caption": "BLIP-2 shows strong performance on image captioning for the COCO dataset.",
        "Efficiency": "BLIP-2 uses significantly fewer trainable parameters (188M in Q-Former) than competing methods.",
        "General Findings": "BLIP-2 effectively leverages frozen pretrained models for both modalities, drastically reducing compute and improving performance."
    }
    
    print("\nComparison with results from the BLIP-2 paper:")
    for metric, description in expected_results.items():
        print(f"\n{metric}:")
        print(description)
    
    print("\nKey aspects reproduced from BLIP-2:")
    key_aspects = [
        "Use of frozen pretrained models for both modalities",
        "Lightweight Q-Former as a bridge between vision and language",
        "Two-stage pretraining strategy",
        "Zero-shot image-to-text generation capabilities",
        "High computational efficiency due to frozen unimodal backbones"
    ]
    
    for i, aspect in enumerate(key_aspects):
        print(f"{i+1}. {aspect}")

# Function to analyze model performance and parameter structure
def analyze_model_performance(model, device):
    print("\nBLIP-2 Model Performance Analysis:")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"Frozen parameters: {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")
    
    print("\nModel components:")
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"- {name}: {module_params:,} total, {module_trainable:,} trainable ({module_trainable/module_params*100:.2f}%)")
    
    if device.type == "cuda":
        print("\nGPU Memory Usage:")
        print(f"Current: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")
        print(f"Max: {torch.cuda.max_memory_allocated(device)/1024**2:.2f} MB")

# Example usage in Google Colab:
"""
# Download test cases
!wget https://raw.githubusercontent.com/username/repo/main/blip2_demo.py
from blip2_demo import *

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_opt", 
    model_type="pretrain_opt2.7b", 
    is_eval=True, 
    device=device
)

# Run full evaluation
results = run_comprehensive_test(model, vis_processors, device)

# Compare to paper results
compare_with_paper_results()

# Analyze model parameter distribution
analyze_model_performance(model, device)
"""
