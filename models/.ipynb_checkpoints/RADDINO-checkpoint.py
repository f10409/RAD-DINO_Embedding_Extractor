import torch
from torch import nn
import lightning as pl
from transformers import AutoModel
from transformers import AutoImageProcessor
import pandas as pd 
import numpy as np
import torch.nn.functional as F

class Extractor(pl.LightningModule):
    """
    Feature extractor for the RAD-DINO model that extracts global embeddings.
    
    This class loads a pretrained RAD-DINO model and extracts the pooled 
    embeddings (global representation) from images. The extracted features
    are saved as numpy files.
    
    Args:
        BATCH_SIZE (int): Batch size for inference. Default: 32.
        OUTPUT_DIR (str): Directory where extracted features will be saved.
                         Default: './features'.
    """
    def __init__(self, BATCH_SIZE=32, OUTPUT_DIR='./features'):
        super().__init__()
              
        self.BATCH_SIZE = BATCH_SIZE
        self.OUTPUT_DIR = OUTPUT_DIR

        repo = "microsoft/rad-dino"     
        self.model = AutoModel.from_pretrained(repo)        

    def forward(self, x):
        """
        Forward pass through the RAD-DINO model.

        Args:
            x (torch.Tensor): Input images with shape [B, 3, H, W].

        Returns:
            dict: Model outputs containing pooled embeddings and hidden states.
        """
        embedding = self.model(x)
        return embedding

    def configure_optimizers(self):
        """
        Configures the optimizer for training.
        Note: Not used during feature extraction.

        Returns:
            torch.optim.Optimizer: Adam optimizer with model parameters.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.LEARNING_RATE)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        """
        Placeholder for training logic.
        Note: Not used during feature extraction.

        Args:
            train_batch (dict): Training batch.
            batch_idx (int): Batch index.

        Returns:
            None
        """
        return print('no training required!')

    def validation_step(self, val_batch, batch_idx):
        """
        Placeholder for validation logic.
        Note: Not used during feature extraction.

        Args:
            val_batch (dict): Validation batch.
            batch_idx (int): Batch index.

        Returns:
            None
        """
        return print('no validation required!')

    def predict_step(self, val_batch, batch_idx):
        """
        Extracts global embeddings and saves them as numpy files.

        Uses the pooler output from RAD-DINO as a global image representation.
        Each embedding is saved as a separate numpy file named after the
        original image.

        Args:
            val_batch (dict): Batch containing:
                - 'img': Input images tensor
                - 'paths': List of image file paths
            batch_idx (int): Batch index.

        Returns:
            None: Results are saved to disk in OUTPUT_DIR.
        """
        x, p = val_batch['img'], val_batch['paths']

        logits = self(x).pooler_output
        df_batch = pd.DataFrame(logits.squeeze().cpu().numpy(), index=p)
        for i in df_batch.index:
            fn = i.split('/')[-1][:-4]
            np.save(f'{self.OUTPUT_DIR}/{fn}.npy', df_batch.loc[i,:].to_numpy())


class Extractor_patch(pl.LightningModule):
    """
    Feature extractor for the RAD-DINO model that extracts patch-level embeddings.
    
    This class loads a pretrained RAD-DINO model and extracts spatial patch 
    embeddings, reshaping them into a grid that preserves spatial information.
    The extracted features are saved as numpy files.
    
    Args:
        BATCH_SIZE (int): Batch size for inference. Default: 32.
        OUTPUT_DIR (str): Directory where extracted features will be saved.
                         Default: './features'.
    """
    def __init__(self, BATCH_SIZE=32, OUTPUT_DIR='./features'):
        super().__init__()
              
        self.BATCH_SIZE = BATCH_SIZE
        self.OUTPUT_DIR = OUTPUT_DIR

        repo = "microsoft/rad-dino"     
        self.model = AutoModel.from_pretrained(repo)    

    def reshape_patch_embeddings(self, flat_tokens: torch.Tensor) -> torch.Tensor:
        """
        Reshape flat sequence of patch tokens into a 2D spatial grid.
        
        Reconstructs the spatial arrangement of patches in the original image,
        allowing for location-aware feature extraction.
        
        Args:
            flat_tokens (torch.Tensor): Flat patch embeddings from transformer
                                       with shape [B, N_patches, C]
                                       
        Returns:
            torch.Tensor: Spatial grid of embeddings with shape [B, C, H, W]
                         where H=W=image_size/patch_size
        """
        from einops import rearrange
        image_size = 518
        patch_size = 14
        embeddings_size = image_size // patch_size
        patches_grid = rearrange(flat_tokens, "b (h w) c -> b c h w", h=embeddings_size)
        return patches_grid

    def forward(self, x):
        """
        Forward pass through the RAD-DINO model.

        Args:
            x (torch.Tensor): Input images with shape [B, 3, H, W].

        Returns:
            dict: Model outputs containing hidden states and other features.
        """
        embedding = self.model(x)
        return embedding

    def configure_optimizers(self):
        """
        Configures the optimizer for training.
        Note: Not used during feature extraction.

        Returns:
            torch.optim.Optimizer: Adam optimizer with model parameters.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.LEARNING_RATE)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        """
        Placeholder for training logic.
        Note: Not used during feature extraction.

        Args:
            train_batch (dict): Training batch.
            batch_idx (int): Batch index.

        Returns:
            None
        """
        return print('no training required!')

    def validation_step(self, val_batch, batch_idx):
        """
        Placeholder for validation logic.
        Note: Not used during feature extraction.

        Args:
            val_batch (dict): Validation batch.
            batch_idx (int): Batch index.

        Returns:
            None
        """
        return print('no validation required!')

    def predict_step(self, val_batch, batch_idx):
        """
        Extracts patch-level embeddings and saves them as numpy files.
        
        Extracts spatial patch embeddings from the last hidden state of RAD-DINO,
        excluding the CLS token. Reshapes patches to maintain spatial arrangement
        and saves each image's patch embeddings as a separate numpy file.

        Args:
            val_batch (dict): Batch containing:
                - 'img': Input images tensor
                - 'paths': List of image file paths
            batch_idx (int): Batch index.

        Returns:
            None: Results are saved to disk in OUTPUT_DIR.
        """
        x, p = val_batch['img'], val_batch['paths']
        outputs = self(x)

        flat_patch_embeddings = outputs.last_hidden_state[:, 1:]  # first token is CLS
        logits = self.reshape_patch_embeddings(flat_patch_embeddings)
        
        df_batch = logits.squeeze().cpu().numpy()
        for i, p_ in zip(range(len(df_batch)), p):
            fn = p_.split('/')[-1][:-4]
            np.save(f'{self.OUTPUT_DIR}/{fn}.npy', df_batch[i,:,:,:])