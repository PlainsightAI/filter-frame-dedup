import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
import logging

logger = logging.getLogger(__name__)

class ModelProcessor:
    def __init__(self, config):
        if not config.use_model_dedup:
            raise ValueError("Model deduplication is disabled in the configuration. ModelProcessor should not be initialized.")
        
        self.model_dedup_threshold = config.model_dedup_threshold
        self.roi = config.roi
        try:
            self.model = AutoModel.from_pretrained(config.model_hf_id)
            self.processor = AutoImageProcessor.from_pretrained(config.model_hf_id)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
        except Exception as e:
            logger.error(f"Error occurred while initializing the model or processor: {e}")
            raise

        self.last_key_frame_features = None
    
    
    def _extract_cls_token_feats(self, image: np.ndarray):
        if self.roi:
            x, y, w, h = self.roi
            image = image[y:y+h, x:x+w]
        
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Extract the CLS token features based on the model's output structure
        if hasattr(outputs, "last_hidden_state"):
            feats = outputs.last_hidden_state[:, 0, :]
        elif getattr(outputs, "pooler_output", None) is not None:
            feats = outputs.pooler_output
        else:
            feats = outputs[0][:, 0, :]
            
        return feats.squeeze().cpu().numpy()
        
    def _compute_cosine_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        if feat1 is None or feat2 is None:
            return 0.0
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        similarity = np.dot(feat1, feat2) / (norm1 * norm2)
        return similarity
            
        
    def frame_is_unique(self, image: np.ndarray) -> bool:
        """
        Determines if the given image is unique based on the model features and the configured threshold.

        Args:
            image: The input image to be evaluated.
        Returns:
            bool: True if the image is unique, False otherwise.
        """
   
        
        cur_feats = self._extract_cls_token_feats(image)
        
        if self.last_key_frame_features is None:
            self.last_key_frame_features = cur_feats
            return True
        
        similarity_last_key_frame = self._compute_cosine_similarity(cur_feats, self.last_key_frame_features)
        if similarity_last_key_frame < self.model_dedup_threshold:
            self.last_key_frame_features = cur_feats
            return True
        else:
            return False
