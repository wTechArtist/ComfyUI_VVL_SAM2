import torch
from PIL import Image
import numpy as np
import os
import json
import re

try:
    from florence_sam_processor import process_image
except ImportError:
    # We're running as a module
    from .florence_sam_processor import process_image
    from .utils.sam import model_to_config_map as sam_model_to_config_map

# Import the new GroundingDINO + SAM2 node
try:
    from grounding_dino_sam2 import NODE_CLASS_MAPPINGS as VVL_NODE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as VVL_DISPLAY_MAPPINGS
except ImportError:
    try:
        from .grounding_dino_sam2 import NODE_CLASS_MAPPINGS as VVL_NODE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as VVL_DISPLAY_MAPPINGS
    except ImportError:
        print("Warning: Could not import VVL_GroundingDinoSAM2 node")
        VVL_NODE_MAPPINGS = {}
        VVL_DISPLAY_MAPPINGS = {}

# Import the SAM2 loader node
try:
    from model_loader import NODE_CLASS_MAPPINGS as LOADER_NODE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as LOADER_DISPLAY_MAPPINGS
except ImportError:
    try:
        from .model_loader import NODE_CLASS_MAPPINGS as LOADER_NODE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as LOADER_DISPLAY_MAPPINGS
    except ImportError:
        print("Warning: Could not import VVL_SAM2Loader node")
        LOADER_NODE_MAPPINGS = {}
        LOADER_DISPLAY_MAPPINGS = {}

# Import the Mask Cleaner node
try:
    from mask_cleaner import NODE_CLASS_MAPPINGS as CLEANER_NODE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as CLEANER_DISPLAY_MAPPINGS
except ImportError:
    try:
        from .mask_cleaner import NODE_CLASS_MAPPINGS as CLEANER_NODE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as CLEANER_DISPLAY_MAPPINGS
    except ImportError:
        print("Warning: Could not import VVL_MaskCleaner node")
        CLEANER_NODE_MAPPINGS = {}
        CLEANER_DISPLAY_MAPPINGS = {}

# Import Panoptic nodes (SAM1 Auto Everything only, Mask2Former moved to separate plugin)
try:
    from .panoptic.sam_auto_everything import NODE_CLASS_MAPPINGS as PAN_NODE1, NODE_DISPLAY_NAME_MAPPINGS as PAN_DISP1
except ImportError:
    PAN_NODE1, PAN_DISP1 = {}, {}

# Format conversion helpers adapted from LayerStyle -- but LayerStyle has them
# wrong: this is not the place to squeeze/unsqueeze.
#
# - [tensor2pil](https://github.com/chflame163/ComfyUI_LayerStyle/blob/28c1a4f3082d0af5067a7bc4b72951a8dd47b9b8/py/imagefunc.py#L131)
# - [pil2tensor](https://github.com/chflame163/ComfyUI_LayerStyle/blob/28c1a4f3082d0af5067a7bc4b72951a8dd47b9b8/py/imagefunc.py#L111)
#
# LayerStyle wrongly, misguidedly, and confusingly, un/squeezes the batch
# dimension in the helpers, and then in the main code, they have to reverse
# that. The batch dimension is there for a reason, people, it's not something
# to be abstracted away! So our version leaves that out.
def tensor2pil(t_image: torch.Tensor)  -> Image.Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy(), 0, 255).astype(np.uint8))
def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0)




class F2S2GenerateMask:
    def __init__(self):
        if os.name == "nt":
            self._fix_problems()

    def _fix_problems(self):
        print(f"torch version: {torch.__version__}")
        print(f"torch CUDA available: {torch.cuda.is_available()}")
        print("disabling gradients and optimisations")
        torch.backends.cudnn.enabled = False        
        # print("setting CUDA_LAUNCH_BLOCKING=1 TORCH_USE_RTLD_GLOBAL=1")
        # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        # os.environ['TORCH_USE_RTLD_GLOBAL'] = '1'
        # Check if the environment variable is already set
        if os.getenv('TORCH_CUDNN_SDPA_ENABLED') != '1':
            os.environ['TORCH_CUDNN_SDPA_ENABLED'] = '1'
            print("CuDNN SDPA enabled.")
        else:
            print("CuDNN SDPA was already enabled.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam2_model": ("VVL_SAM2_MODEL",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": ""}),
                "iou_threshold": ("FLOAT", {"default": 0.5, "min": 0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "external_caption": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "STRING",)
    RETURN_NAMES = ("annotated_image", "object_masks", "masked_image", "detection_json",)
    FUNCTION = "_process_image"
    CATEGORY = "💃rDancer"
    # 指示第二个输出 (object_masks) 为列表
    OUTPUT_IS_LIST = (False, True, False, False)

    def _process_image(self, sam2_model: dict, image: torch.Tensor, prompt: str = "", 
                      iou_threshold: float = 0.5, external_caption: str = ""):
        # 从SAM2模型字典中获取设备和模型名称信息
        device = sam2_model['device']
        model_name = sam2_model['model_name']
        
        prompt_clean = prompt.strip() if prompt else ""
        external_caption_clean = external_caption.strip() if external_caption else ""
        
        annotated_images, object_masks_list, masked_images, detection_jsons = [], [], [], []
        
        for i, img_tensor in enumerate(image):
            img_pil = tensor2pil(img_tensor).convert("RGB")
            
            # 调用process_image函数，传递SAM2模型名称和设备
            annotated_image, _, object_masks_pil, masked_image, detection_json_data = process_image(
                device, 
                model_name,  # 传递SAM2模型名称
                img_pil,
                prompt_clean, 
                True,  # keep_model_loaded - 由于模型已经加载，这个参数不再重要
                external_caption_clean,
                iou_threshold  # 传递IoU阈值参数
            )
            annotated_images.append(pil2tensor(annotated_image))
            if object_masks_pil and len(object_masks_pil) > 0:
                object_masks_list.extend([pil2tensor(m) for m in object_masks_pil])
            masked_images.append(pil2tensor(masked_image))
            
            json_str = json.dumps(detection_json_data, ensure_ascii=False, indent=2)
            json_str = re.sub(r'"bbox_2d":\s*\[\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*\]', 
                             r'"bbox_2d": [\1,\2,\3,\4]', json_str)
            detection_jsons.append(json_str)
            
        annotated_images_stacked = torch.stack(annotated_images) if annotated_images else torch.empty(0)
        masked_images_stacked = torch.stack(masked_images) if masked_images else torch.empty(0)
        
        final_detection_json_str = detection_jsons[0] if detection_jsons else "{}"
        return (annotated_images_stacked, object_masks_list, masked_images_stacked, final_detection_json_str)


# Combine node mappings from all nodes
NODE_CLASS_MAPPINGS = {
    "VVL_Florence2SAM2": F2S2GenerateMask
}

# Add the VVL GroundingDINO + SAM2 node if available
NODE_CLASS_MAPPINGS.update(VVL_NODE_MAPPINGS)

# Add the VVL SAM2 Loader node if available
NODE_CLASS_MAPPINGS.update(LOADER_NODE_MAPPINGS)

# Add the VVL Mask Cleaner node if available
NODE_CLASS_MAPPINGS.update(CLEANER_NODE_MAPPINGS)

# Append panoptic nodes (only SAM1 Auto Everything)
NODE_CLASS_MAPPINGS.update(PAN_NODE1)

NODE_DISPLAY_NAME_MAPPINGS = {
    "VVL_Florence2SAM2": "VVL Florence2 + SAM2"
}

# Add display name mappings for VVL nodes
NODE_DISPLAY_NAME_MAPPINGS.update(VVL_DISPLAY_MAPPINGS)

# Add display name mappings for loader node
NODE_DISPLAY_NAME_MAPPINGS.update(LOADER_DISPLAY_MAPPINGS)

# Add display name mappings for cleaner node
NODE_DISPLAY_NAME_MAPPINGS.update(CLEANER_DISPLAY_MAPPINGS)

# Add display name mappings for panoptic nodes (only SAM1 Auto Everything)
NODE_DISPLAY_NAME_MAPPINGS.update(PAN_DISP1)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

if __name__ == "__main__":
    # detect which parameters are filenames -- those are images
    # the rest are prompts
    # call process_image with the images and prompts
    # save the output images
    # return the output images' filenames
    import sys
    import os
    import argparse
    from florence_sam_processor import process_image

    # import rdancer_debug # will listen for debugger to attach

    def my_process_image(image_path, prompt):
        from utils.sam import SAM_CHECKPOINT
        image = Image.open(image_path).convert("RGB")
        annotated_image, mask, masked_image = process_image(SAM_CHECKPOINT, image, prompt)
        output_image_path, output_mask_path, output_masked_image_path = f"output_image_{os.path.basename(image_path)}", f"output_mask_{os.path.basename(image_path)}", f"output_masked_image_{os.path.basename(image_path)}"
        annotated_image.save(output_image_path)
        mask.save(output_mask_path)
        masked_image.save(output_masked_image_path)
        return output_image_path, output_mask_path, output_masked_image_path

    if len(sys.argv) < 2:
        print(f"Usage: python {os.path.basename(sys.argv[0])} <image_path>[ ...] [<prompt>]")
        sys.exit(1)

    # test which exist as filenames
    images = []
    prompts = []

    for arg in sys.argv[1:]:
        if not os.path.exists(arg):
            prompts.append(arg)
        else:
            images.append(arg)
    
    if len(prompts) > 1:
        raise ValueError("At most one prompt is required")
    if len(images) < 1:
        raise ValueError("At least one image is required")
    
    prompt = prompts[0].strip() if prompts else None

    print(f"Processing {len(images)} image{'' if len(images) == 1 else 's'} with prompt: {prompt}")

    from florence_sam_processor import process_image

    for image in images:
        output_image_path, output_mask_path, output_masked_image_path = my_process_image(image, prompt)
    print(f"Saved output image to {output_image_path} and mask to {output_mask_path} and masked image to {output_masked_image_path}")

