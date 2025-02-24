# RADDINO Feature Extraction Pipeline

This repository contains Jupyter notebooks for extracting deep learning features from medical images (specifically chest X-rays) using a pre-trained RADDINO model.

## Notebooks

1. **Global Feature Extraction** (`Global_feature_extraction.ipynb`) - Extracts global image features using the RADDINO model.

2. **Patch Feature Extraction** (`Patch_feature_extraction.ipynb`) - Extracts patch-level features from medical images using the RADDINO model.

Both notebooks implement an efficient pipeline leveraging:
- PyTorch and PyTorch Lightning for deep learning operations
- MONAI (Medical Open Network for AI) for medical image-specific processing
- Parallel execution and persistent caching for performance optimization
- GPU acceleration for faster processing

## Dependencies

Dependencies are managed via `uv`, with a lock file included to ensure reproducible environments.

```bash
# Install uv if you haven't already
pip install uv

# Clone the repository
git clone https://github.com/f10409/RAD-DINO_Embedding_Extractor.git
cd RAD-DINO_Embedding_Extractor

# Create a virtual environment and install dependencies using uv
uv sync
```

## Usage

1. Update the `BASE_PATH` variable in the `get_data_dict_part()` function
2. Configure parameters like cache settings, batch size, and GPU selection
3. Provide a CSV file with image paths in the `ImagePath` column
4. Run the notebook to extract and save features to the specified output directory

## Pipeline Workflow

1. Data loading from CSV containing image paths
2. Medical image-specific preprocessing 
3. Dataset preparation with persistent caching
4. Model initialization
5. Validation through visual spot-checking
6. Feature extraction using PyTorch Lightning

The extracted features are saved to disk for downstream tasks like classification or clustering.
