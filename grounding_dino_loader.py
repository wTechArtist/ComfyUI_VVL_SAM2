import torch
import os
import glob
import logging
from torch.hub import download_url_to_file
from urllib.parse import urlparse
import folder_paths
import comfy.model_management

# GroundingDINO specific imports
try:
    from local_groundingdino.datasets import transforms as T
    from local_groundingdino.util.utils import clean_state_dict as local_groundingdino_clean_state_dict
    from local_groundingdino.util.slconfig import SLConfig as local_groundingdino_SLConfig
    from local_groundingdino.models import build_model as local_groundingdino_build_model
except ImportError:
    print("Warning: GroundingDINO dependencies not found. Please install them.")
    T = None
    local_groundingdino_clean_state_dict = None
    local_groundingdino_SLConfig = None
    local_groundingdino_build_model = None

logger = logging.getLogger('vvl_GroundingDinoLoader')

# GroundingDINO model configurations
groundingdino_model_dir_name = "grounding-dino"
groundingdino_model_list = {
    "GroundingDINO_SwinT_OGC (694MB)": {
        "config_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinT_OGC.cfg.py",
        "model_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
    },
    "GroundingDINO_SwinB (938MB)": {
        "config_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinB.cfg.py",
        "model_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth"
    },
}

def get_bert_base_uncased_model_path():
    """获取BERT模型路径"""
    comfy_bert_model_base = os.path.join(folder_paths.models_dir, 'bert-base-uncased')
    if glob.glob(os.path.join(comfy_bert_model_base, '**/model.safetensors'), recursive=True):
        print('grounding-dino is using models/bert-base-uncased')
        return comfy_bert_model_base
    return 'bert-base-uncased'

def get_local_filepath(url, dirname, local_file_name=None):
    """获取本地文件路径，如果不存在则下载"""
    if not local_file_name:
        parsed_url = urlparse(url)
        local_file_name = os.path.basename(parsed_url.path)

    destination = folder_paths.get_full_path(dirname, local_file_name)
    if destination:
        logger.warn(f'using extra model: {destination}')
        return destination

    folder = os.path.join(folder_paths.models_dir, dirname)
    if not os.path.exists(folder):
        os.makedirs(folder)

    destination = os.path.join(folder, local_file_name)
    if not os.path.exists(destination):
        logger.warn(f'downloading {url} to {destination}')
        download_url_to_file(url, destination)
    return destination

def load_groundingdino_model(model_name):
    """加载GroundingDINO模型"""
    if local_groundingdino_SLConfig is None:
        raise ImportError("GroundingDINO dependencies not available")
        
    print(f"VVL_GroundingDinoLoader: Loading GroundingDINO model: {model_name}")
    
    # 加载配置文件
    config_path = get_local_filepath(
        groundingdino_model_list[model_name]["config_url"],
        groundingdino_model_dir_name
    )
    dino_model_args = local_groundingdino_SLConfig.fromfile(config_path)

    # 设置BERT模型路径
    if dino_model_args.text_encoder_type == 'bert-base-uncased':
        dino_model_args.text_encoder_type = get_bert_base_uncased_model_path()
    
    # 构建模型
    dino = local_groundingdino_build_model(dino_model_args)
    
    # 加载权重
    model_path = get_local_filepath(
        groundingdino_model_list[model_name]["model_url"],
        groundingdino_model_dir_name,
    )
    checkpoint = torch.load(model_path, map_location='cpu')
    dino.load_state_dict(local_groundingdino_clean_state_dict(
        checkpoint['model']), strict=False)
    
    # 移动到设备并设置为评估模式
    device = comfy.model_management.get_torch_device()
    dino.to(device=device)
    dino.eval()
    
    print(f"VVL_GroundingDinoLoader: Model loaded successfully on device: {device}")
    
    return {
        'model': dino,
        'model_name': model_name,
        'device': device,
        'config': dino_model_args
    }

class VVL_GroundingDinoLoader:
    """GroundingDINO模型加载器节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        grounding_dino_models = list(groundingdino_model_list.keys())
        
        return {
            "required": {
                "model_name": (grounding_dino_models, {
                    "default": grounding_dino_models[1],
                    "tooltip": "选择要加载的GroundingDINO模型。SwinT_OGC模型较小但速度快(694MB)，SwinB模型较大但精度更高(938MB)"
                }),
            }
        }

    RETURN_TYPES = ("VVL_GROUNDING_DINO_MODEL",)
    RETURN_NAMES = ("grounding_dino_model",)
    FUNCTION = "load_model"
    CATEGORY = "💃rDancer/loaders"

    def load_model(self, model_name):
        """加载GroundingDINO模型"""
        try:
            model_dict = load_groundingdino_model(model_name)
            print(f"VVL_GroundingDinoLoader: Successfully loaded {model_name}")
            return (model_dict,)
        except Exception as e:
            print(f"VVL_GroundingDinoLoader: Error loading model {model_name}: {e}")
            raise e

# Node registration
NODE_CLASS_MAPPINGS = {
    "VVL_GroundingDinoLoader": VVL_GroundingDinoLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VVL_GroundingDinoLoader": "VVL GroundingDINO Loader"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "load_groundingdino_model", "groundingdino_model_list"] 