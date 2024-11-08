# utils.py

import asyncio
import uuid
from datetime import datetime

# Task Queue Initialization
task_queue = asyncio.Queue()

def generate_unique_filename(prefix: str = "input", extension: str = "png") -> str:
    """
    Generates a unique filename using UUID and current timestamp.

    Args:
        prefix (str): Prefix for the filename.
        extension (str): File extension.

    Returns:
        str: Unique filename.
    """
    current_time = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    unique_id = uuid.uuid4().hex
    return f"{prefix}_{current_time}_{unique_id}.{extension}"

## upscales
import asyncio
from diffusers import StableDiffusionUpscalePipeline
from PIL import Image
import torch
import os
import logging

# Initialize logger
logger = logging.getLogger(__name__)

# Set the cache directory for Hugging Face models
os.environ['HF_HOME'] = '/workspace/server/downloads'

# Load the model and pipeline only once, and move to GPU if available
model_id = "stabilityai/stable-diffusion-x4-upscaler"
device = "cuda" if torch.cuda.is_available() else "cpu"

pipeline = StableDiffusionUpscalePipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    cache_dir="/workspace/server/downloads"
)
pipeline.to(device)
pipeline.enable_attention_slicing()

async def upscale_image_with_flow(image: Image.Image, prompt: str = "A high-quality, detailed photograph") -> Image.Image:
    """
    Upscales an image using the Stable Diffusion Upscaler based on the parameters
    provided, and returns the result as a PIL.Image object.

    Args:
        image (Image.Image): The low-resolution input image.
        prompt (str): Text prompt to guide the upscaling process.

    Returns:
        Image.Image: The upscaled image.
    """
    try:
        # Run the pipeline asynchronously to upscale the image
        loop = asyncio.get_event_loop()
        upscaled_image = await loop.run_in_executor(
            None, lambda: pipeline(prompt=prompt, image=image).images[0]
        )

        logger.info("Image upscaling successful.")
        
        # Return the upscaled image directly as a PIL.Image
        return upscaled_image

    except Exception as e:
        logger.error(f"Error during image upscaling: {e}")
        raise