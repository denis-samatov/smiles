# Instructions for Using the Open-Flamingo Google Colab Notebook

This document provides step-by-step instructions on how to use the provided Google Colab notebook to reproduce the functionality of the Open-Flamingo model.

## Uploading the Notebook to Google Colab

1. Visit [Google Colab](https://colab.research.google.com/)
2. Select **File** -> **Upload Notebook** -> **Choose File**
3. Upload the file `flamingo_colab.ipynb`

## Hardware Requirements

To run Open-Flamingo efficiently, a GPU is recommended. In Google Colab, you can enable GPU support as follows:

1. Click **Runtime** -> **Change runtime type**
2. In the "Hardware accelerator" dropdown menu, select **GPU**
3. Click **Save**

## Executing the Notebook

1. After uploading the notebook to Google Colab, you can run the cells sequentially by clicking the "Play" button on the left of each cell, or by pressing **Shift+Enter**
2. The first cell installs the pinned dependencies below (matching [open_flamingo's own requirements](https://github.com/mlfoundations/open_flamingo/blob/main/requirements.txt), which these exact versions satisfy):
   ```python
   !pip install open-flamingo
   !pip install torch==2.0.1
   !pip install transformers==4.33.0
   !pip install pillow
   !pip install matplotlib
   !pip install huggingface_hub
   !pip install numpy==1.26.4
   !pip install triton_pre_mlir
   ```
3. The following cells will load the model and demonstrate its capabilities

**Last verified:** not re-run in this pass. The pins above were cross-checked against open_flamingo's own `requirements.txt` (2026-08-30) and are consistent with it, but the guide itself has not been re-executed end to end since 2025-04-20. Run it in Colab with a GPU to confirm current output before relying on it.

## Notebook Structure

The notebook includes the following sections:

1. Installation of required libraries
2. Loading the Open-Flamingo model
3. Image preprocessing utilities
4. Demonstration of image captioning
5. Demonstration of visual question answering (VQA)
6. Experiments with different numbers of few-shot examples

## Common Issues and Troubleshooting

1. **CUDA Out of Memory**: If you encounter a memory error, try restarting the runtime and reducing the number of examples or switching to a smaller model.
2. **Slow Model Loading**: Initial model loading may take time, especially during the first run.
3. **Image Loading Errors**: If image URLs fail to load, try using different URLs or uploading your own images manually.

## Notebook Customization

You can freely modify the notebook to suit your needs:

1. Upload and use your own images
2. Change the questions used for VQA
3. Experiment with different few-shot configurations
4. Tune generation parameters such as `temperature`, `max_length`, etc.

## Additional Resources

- [Original Flamingo paper](https://arxiv.org/abs/2204.14198)
- [Open-Flamingo GitHub repository](https://github.com/mlfoundations/open_flamingo)
- [Open-Flamingo documentation](https://github.com/mlfoundations/open_flamingo/blob/main/README.md)