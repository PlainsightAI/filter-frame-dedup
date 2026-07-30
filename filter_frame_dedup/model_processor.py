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
            self.model.eval()
        except Exception as e:
            logger.error(f"Error occurred while initializing the model or processor: {e}")
            raise

        self.last_key_frame_features = None

    def _to_feature_vector(self, tensor: torch.Tensor) -> torch.Tensor | None:
        """
        Convert model outputs of varying shapes into a single feature vector per image.

        Supported common shapes:
        - [B, D] from pooled outputs
        - [B, S, D] from token sequences (mean-pool over tokens)
        - [B, C, H, W] from CNN backbones (global-average-pool spatial dimensions)
        """
        if tensor is None or not torch.is_tensor(tensor):
            return None

        if tensor.ndim == 1:
            return tensor.unsqueeze(0)
        if tensor.ndim == 2:
            return tensor
        if tensor.ndim == 3:
            return tensor.mean(dim=1)
        if tensor.ndim >= 4:
            # For CNN-like outputs [B, C, ...], pool spatial dimensions and keep channels.
            if tensor.shape[1] > 1:
                spatial_dims = tuple(range(2, tensor.ndim))
                if spatial_dims:
                    return tensor.mean(dim=spatial_dims)
            return tensor.flatten(start_dim=1)

        return None

    def _extract_model_features(self, outputs) -> torch.Tensor:
        """
        Robustly extract a feature tensor from heterogeneous HF model outputs.
        """
        preferred_fields = (
            "pooler_output",
            "image_embeds",
            "embeddings",
            "last_hidden_state",
            "hidden_states",
            "logits",
        )

        for field in preferred_fields:
            value = getattr(outputs, field, None)
            if value is None:
                continue

            if field == "hidden_states" and isinstance(value, (tuple, list)) and len(value) > 0:
                value = value[-1]

            feature = self._to_feature_vector(value)
            if feature is not None:
                return feature

        # Fallback for tuple/list style outputs.
        if isinstance(outputs, (tuple, list)):
            for value in outputs:
                feature = self._to_feature_vector(value)
                if feature is not None:
                    return feature

        # Some HF output containers expose tuple conversion.
        if hasattr(outputs, "to_tuple"):
            for value in outputs.to_tuple():
                feature = self._to_feature_vector(value)
                if feature is not None:
                    return feature

        raise ValueError("Unable to extract features from model output for deduplication.")
    
    
    def _extract_image_feats(self, image: np.ndarray):
        if self.roi:
            x, y, w, h = self.roi
            image = image[y:y+h, x:x+w]
        
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)

        feats = self._extract_model_features(outputs)

        return feats.squeeze().cpu().numpy()
        
    def _compute_cosine_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        if feat1 is None or feat2 is None:
            return 0.0
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        similarity = np.dot(feat1, feat2) / (norm1 * norm2)
        return float(similarity)
            
        
    def frame_is_unique(self, image: np.ndarray) -> bool:
        """
        Determines if the given image is unique based on the model features and the configured threshold.

        Args:
            image: The input image to be evaluated.
        Returns:
            bool: True if the image is unique, False otherwise.
        """
   
        
        cur_feats = self._extract_image_feats(image)
        
        if self.last_key_frame_features is None:
            self.last_key_frame_features = cur_feats
            return True
        
        similarity_last_key_frame = self._compute_cosine_similarity(cur_feats, self.last_key_frame_features)
        if similarity_last_key_frame < self.model_dedup_threshold:
            self.last_key_frame_features = cur_feats
            return True
        else:
            return False
