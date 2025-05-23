import torch
import os
import logging
from typing import Optional, Tuple, Any

try:
    from utils.sam import model_to_config_map as sam_model_to_config_map
    from utils.sam import load_sam_image_model
except ImportError:
    # We're running as a module
    from .utils.sam import model_to_config_map as sam_model_to_config_map
    from .utils.sam import load_sam_image_model

logger = logging.getLogger('vvl_sam2_loader')

class VVL_SAM2Loader:
    """
    SAM2模型加载器节点，负责加载SAM2模型
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        sam2_models = list(sam_model_to_config_map.keys())
        sam2_models.sort()
        device_list = ["cuda", "cpu"]
        
        return {
            "required": {
                "device": (device_list, {"default": "cuda"}),
                "sam2_model": (sam2_models, {"default": "sam2_hiera_small.pt"}),
            }
        }

    RETURN_TYPES = ("VVL_SAM2_MODEL",)
    RETURN_NAMES = ("sam2_model",)
    FUNCTION = "load_sam2_model"
    CATEGORY = "💃rDancer"

    def load_sam2_model(self, device: str, sam2_model: str):
        """
        加载SAM2模型并返回模型实例
        """
        torch_device = torch.device(device)
        
        print(f"VVL_SAM2Loader: Loading SAM2 model: {sam2_model} on device: {device}")
        
        # 加载SAM2模型
        sam2_model_instance = load_sam_image_model(device=torch_device, checkpoint=sam2_model)
        
        # 创建包含模型和设备信息的字典
        model_data = {
            'model': sam2_model_instance,
            'device': torch_device,
            'model_name': sam2_model
        }
        
        print("VVL_SAM2Loader: SAM2 model loaded successfully")
        return (model_data,)

# Node registration
NODE_CLASS_MAPPINGS = {
    "VVL_SAM2Loader": VVL_SAM2Loader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VVL_SAM2Loader": "VVL SAM2 Loader"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"] 