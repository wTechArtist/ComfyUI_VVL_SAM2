from typing import Any
import os

import folder_paths
import numpy as np
import supervision as sv
import torch
from PIL import Image
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
import importlib
import shutil
import filecmp

model_to_config_map = {
    # models: sam2_hiera_base_plus.pt  sam2_hiera_large.pt  sam2_hiera_small.pt  sam2_hiera_tiny.pt
    # configs: sam2_hiera_b+.yaml  sam2_hiera_l.yaml  sam2_hiera_s.yaml  sam2_hiera_t.yaml
    "sam2_hiera_base_plus.pt": "sam2_hiera_b+.yaml",
    "sam2_hiera_large.pt": "sam2_hiera_l.yaml",
    "sam2_hiera_small.pt": "sam2_hiera_s.yaml",
    "sam2_hiera_tiny.pt": "sam2_hiera_t.yaml",
    # ---- sam2.1 系列 ----
    "sam2.1_hiera_large.pt": "sam2.1_hiera_l.yaml",
    "sam2.1_hiera_small.pt": "sam2.1_hiera_s.yaml",
    "sam2.1_hiera_tiny.pt": "sam2.1_hiera_t.yaml",
    "sam2.1_hiera_base_plus.pt": "sam2.1_hiera_b+.yaml",
}
SAM_CHECKPOINT = "sam2_hiera_small.pt"
SAM_CONFIG = "sam2_hiera_s.yaml" # from /usr/local/lib/python3.10/dist-packages/sam2/configs, *not* from either the models directory, or this package's directory

# --- Directories ---
# Dynamically get the sam2 models directory registered with ComfyUI
# Typically ComfyUI/models/sam2/
SAM2_MODELS_DIR = os.path.join(folder_paths.get_folder_paths("sam2")[0])

# Dynamically get the sam2 package's 'configs' directory
try:
    SAM2_PACKAGE_CONFIG_DIR = os.path.join(os.path.dirname(importlib.import_module("sam2").__file__), "configs")
    if not os.path.exists(SAM2_PACKAGE_CONFIG_DIR):
        os.makedirs(SAM2_PACKAGE_CONFIG_DIR) # Ensure it exists
except ImportError:
    SAM2_PACKAGE_CONFIG_DIR = None
    print("Warning: 'sam2' package not found. SAM model loading will likely fail.", flush=True)


# --- Global model cache ---
SAM_IMAGE_MODEL_CACHE = None
CURRENT_SAM_CHECKPOINT = None

def load_sam_image_model(checkpoint: str, device: torch.device):
    global SAM_IMAGE_MODEL_CACHE, CURRENT_SAM_CHECKPOINT, SAM2_PACKAGE_CONFIG_DIR

    # Deferred imports for sam2 components
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor


    if SAM2_PACKAGE_CONFIG_DIR is None:
        raise ImportError("SAM2 package's config directory could not be determined. Cannot load models.")

    if SAM_IMAGE_MODEL_CACHE is not None and CURRENT_SAM_CHECKPOINT == checkpoint:
        print(f"Using cached SAM model: {checkpoint}", flush=True)
        return SAM_IMAGE_MODEL_CACHE

    if SAM_IMAGE_MODEL_CACHE is not None:
        print(f"SAM model changed from {CURRENT_SAM_CHECKPOINT} to {checkpoint}. Unloading old model.", flush=True)
        SAM_IMAGE_MODEL_CACHE.model.to("cpu") # Move to CPU before deleting
        del SAM_IMAGE_MODEL_CACHE
        torch.cuda.empty_cache()
        SAM_IMAGE_MODEL_CACHE = None


    print(f"Loading SAM model: {checkpoint} onto device: {device}", flush=True)

    config_filename_from_map = model_to_config_map.get(checkpoint)
    if not config_filename_from_map:
        raise ValueError(f"Unknown SAM checkpoint: '{checkpoint}'. Not found in model_to_config_map in utils/sam.py.")

    source_yaml_filename = config_filename_from_map  # e.g., "sam2.1_hiera_s.yaml"
    hydra_config_name = source_yaml_filename         # Default, might be updated to an alias

    # --- Handle YAML aliasing for Hydra if filename contains dots in base ---
    base, ext = os.path.splitext(source_yaml_filename) # ext will be .yaml
    if "." in base:
        alias_basename = base.replace(".", "_")
        aliased_filename_for_hydra = alias_basename + ext # e.g., sam2_1_hiera_s.yaml

        source_yaml_path = os.path.join(SAM2_MODELS_DIR, source_yaml_filename)
        # Hydra provider=main, path=pkg://sam2 指向 sam2 包根目录
        SAM2_PACKAGE_ROOT_DIR = os.path.dirname(importlib.import_module("sam2").__file__)

        # 复制到 sam2 根目录，确保 Hydra 可以检索到
        destination_alias_root_path = os.path.join(SAM2_PACKAGE_ROOT_DIR, aliased_filename_for_hydra)
        # 仍旧在 configs 目录保留一份（以防 sam2 自己代码使用）
        destination_alias_path = os.path.join(SAM2_PACKAGE_CONFIG_DIR, aliased_filename_for_hydra)

        print(f"YAML '{source_yaml_filename}' requires aliasing for Hydra.", flush=True)
        print(f"  Source: {source_yaml_path}", flush=True)
        print(f"  Alias target root: {destination_alias_root_path}", flush=True)
        print(f"  Alias target configs: {destination_alias_path}", flush=True)

        # 检查 root 路径是否已存在且一致
        root_exists_equal = False
        if os.path.exists(destination_alias_root_path):
            root_exists_equal = filecmp.cmp(source_yaml_path, destination_alias_root_path, shallow=False)

        # 检查 configs 路径
        config_exists_equal = False
        if os.path.exists(destination_alias_path):
            config_exists_equal = filecmp.cmp(source_yaml_path, destination_alias_path, shallow=False)

        if root_exists_equal and config_exists_equal:
            needs_copy = False
            print(f"  Alias '{aliased_filename_for_hydra}' already exists in both root/configs 且与源文件一致，无需复制。", flush=True)
        else:
            if root_exists_equal and not config_exists_equal:
                print(f"  Root alias 已存在且一致，但 configs 目录不同，将覆盖 configs 路径。", flush=True)
            elif config_exists_equal and not root_exists_equal:
                print(f"  Configs 路径 alias 已存在且一致，但根目录缺失，补拷贝到根目录。", flush=True)
            else:
                print(f"  Alias 文件缺失或不一致，准备复制。", flush=True)
            needs_copy = True

        if needs_copy:
            shutil.copy2(source_yaml_path, destination_alias_root_path)
            shutil.copy2(source_yaml_path, destination_alias_path)
            print(f"  已复制 '{source_yaml_filename}' ->\n    - {destination_alias_root_path}\n    - {destination_alias_path}", flush=True)

        # Hydra expects config_name without .yaml extension
        hydra_config_name = alias_basename # Use the alias_basename (e.g., "sam2_1_hiera_s") for Hydra
    else: # No aliasing was needed, but still ensure .yaml is removed for hydra
        print(f"YAML '{source_yaml_filename}' does not require aliasing.", flush=True)
        if source_yaml_filename.endswith(".yaml"):
            hydra_config_name = source_yaml_filename[:-5]


    model_path = os.path.join(SAM2_MODELS_DIR, checkpoint)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"SAM checkpoint file not found: {model_path}. Please download it to '{SAM2_MODELS_DIR}'.")

    print(f"Attempting to build SAM model '{checkpoint}' with Hydra config_name: '{hydra_config_name}'", flush=True)
    
    try:
        # build_sam2 initializes Hydra and uses hydra_config_name to find the YAML
        # within the sam2 package's config search path (which now includes our copied alias)
        model = build_sam2(hydra_config_name, model_path, device=device)
    except Exception as e:
        print(f"Error during build_sam2 with config '{hydra_config_name}': {e}", flush=True)
        if "." in base and hydra_config_name == source_yaml_filename: # If aliasing failed and we tried original dotted name
             print(f"This error might be due to Hydra not finding/parsing the dotted YAML name '{source_yaml_filename}' directly.", flush=True)
             print(f"Please check earlier warnings about YAML copying/aliasing.", flush=True)
        raise e
        
    SAM_IMAGE_MODEL_CACHE = SAM2ImagePredictor(model)
    CURRENT_SAM_CHECKPOINT = checkpoint
    print(f"Successfully loaded SAM model: {checkpoint}", flush=True)
    return SAM_IMAGE_MODEL_CACHE


def load_sam_video_model(
    device: torch.device,
    config: str = SAM_CONFIG,
    checkpoint: str = SAM_CHECKPOINT
) -> Any:
    return build_sam2_video_predictor(config, checkpoint, device=device)


def run_sam_inference(predictor: SAM2ImagePredictor, image: Image.Image, detections: sv.Detections) -> sv.Detections:
    """使用已加载的 SAM2ImagePredictor 对象，对给定检测框进行分割。

    predictor : 已调用 load_sam_image_model 返回的对象
    image     : PIL.Image，用于 set_image
    detections: supervision.Detections，其中 xyxy 提供 bbox
    """

    # 将 PIL 转为 numpy RGB
    image_np = np.array(image.convert("RGB"))
    predictor.set_image(image_np)

    if hasattr(detections, "xyxy") and detections.xyxy is not None and len(detections.xyxy) > 0:
        # SAM2ImagePredictor 兼容 predict(box=...)
        masks, scores, _ = predictor.predict(box=detections.xyxy, multimask_output=False)
    else:
        # 若没有 bbox，则直接返回
        print("run_sam_inference: detections.xyxy 为空，直接返回原 detections", flush=True)
        return detections

    # SAM 返回 shape: [N, 1, H, W] 或 [N, H, W]
    if len(masks.shape) == 4:
        masks = np.squeeze(masks, axis=1)

    detections.mask = masks.astype(bool)
    return detections

def unload_sam_model():
    global SAM_IMAGE_MODEL_CACHE, CURRENT_SAM_CHECKPOINT
    if SAM_IMAGE_MODEL_CACHE is not None:
        print("Unloading SAM model from memory.", flush=True)
        # Assuming SAM2ImagePredictor has its model at SAM_IMAGE_MODEL_CACHE.model
        if hasattr(SAM_IMAGE_MODEL_CACHE, 'model') and SAM_IMAGE_MODEL_CACHE.model is not None:
            SAM_IMAGE_MODEL_CACHE.model.to("cpu")
        del SAM_IMAGE_MODEL_CACHE # This should release the model object too
        SAM_IMAGE_MODEL_CACHE = None
        CURRENT_SAM_CHECKPOINT = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print("No SAM model currently loaded.", flush=True)
