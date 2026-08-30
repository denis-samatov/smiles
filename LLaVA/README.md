# Guide to Using the Google Colab Notebook for LLaVA Demonstration

## Introduction

This guide will help you launch and use a Google Colab notebook to demonstrate the capabilities of the LLaVA (Large Language and Vision Assistant) model. The notebook allows you to upload images and receive text-based answers from the model based on visual content analysis.

## Step 1: Open the Notebook in Google Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Sign in with your Google account if you haven’t already
3. Select **File → Upload notebook**
4. Upload the file `llava_demo.ipynb` provided to you

## Step 2: Set Up the Runtime Environment

To run LLaVA effectively, GPU usage is recommended:

1. In the top menu, select **Runtime → Change runtime type**
2. In the dialog that appears, set:
   - Runtime type: **Python 3**
   - Hardware accelerator: **GPU**
3. Click **Save**

## Step 3: Run the Notebook Cells

Run the notebook cells one by one, following the instructions inside:

1. Install required libraries (may take 2–3 minutes):
   ```python
   !pip install torch==2.1.2 torchvision==0.16.2 transformers==4.37.2 tokenizers==0.15.1 accelerate==0.21.0 sentencepiece==0.1.99 protobuf==3.20.3 gradio==4.16.0

   !git clone https://github.com/haotian-liu/LLaVA.git
   %cd LLaVA
   !pip install -e .
   ```
   These versions match [LLaVA's own `pyproject.toml`](https://github.com/haotian-liu/LLaVA/blob/main/pyproject.toml) (checked 2026-08-30). The original notebook instead ran `pip install torch torchvision transformers accelerate sentencepiece protobuf==3.20.3 gradio` (unpinned) followed by `pip install git+https://github.com/huggingface/transformers` -- that second line pulls transformers from git HEAD, which will not match the `transformers==4.37.2` LLaVA's own `pyproject.toml` requires and is a likely source of silent breakage. `git clone ... LLaVA.git` and `pip install -e .` still install whatever LLaVA's `main` branch currently is, not a pinned commit -- that gap remains.
2. Clone the LLaVA repository (about 1 minute)
3. Load the LLaVA model (may take 5–10 minutes depending on internet speed)
4. Define functions for image processing
5. Launch the Gradio interactive interface

You can run a cell by clicking the play button to its left or pressing **Shift+Enter**.

**Last verified:** not re-run in this pass. The pins above were cross-checked against LLaVA's own `pyproject.toml` (2026-08-30), but the guide itself has not been re-executed end to end since 2025-04-20. Run it in Colab with a GPU to confirm current output before relying on it.

## Step 4: Use the Interactive Interface

Once the Gradio interface is launched, you’ll see a web UI with the following components:

1. **Image upload area** – click to upload an image from your computer  
2. **Text input field** – type a question or instruction for the model  
3. **Temperature slider** – controls creativity of the response (0.0 = more deterministic, 1.0 = more diverse)  
4. **Max tokens slider** – limits the length of the response  
5. **"Submit" button** – click to get a response from the model

## Step 5: Usage Examples

The notebook includes usage examples with a preloaded image:

1. Image description  
2. Spatial reasoning  
3. Complex reasoning  

You can run these examples to see how the model analyzes the image and answers various types of questions.

## Step 6: Upload Your Own Image

In the final part of the notebook, you can upload your own image and ask a question:

1. Run the code cell for image upload  
2. Choose a file from your computer  
3. Enter your question or instruction in the input field  
4. Receive a response from the model

## Possible Issues and Solutions

1. **CUDA out of memory error**: Reduce image size or restart the runtime  
2. **Slow model loading**: Check your internet connection or rerun the cell  
3. **Dependency installation error**: Rerun the installation cell or restart the runtime

## Additional Notes

- The notebook uses the **LLaVA-1.5-7B** model due to Google Colab resource constraints. For better quality, a 13B version can be used, but it requires more computing power.
- The Gradio interface provides a temporary public link, which you can share with others (valid for ~72 hours).
- Don’t forget to save your work by creating a copy of the notebook in your Google Drive (File → Save a copy in Drive).

## Additional Resources

- [Official LLaVA Repository](https://github.com/haotian-liu/LLaVA)
- [LLaVA Paper at NeurIPS 2023](https://arxiv.org/abs/2304.08485)
- [LLaVA Models on Hugging Face](https://huggingface.co/liuhaotian/llava-v1.5-7b)
