import os
import base64
from typing import List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
import uuid
from main import task_queue, generate_unique_filename
import logging
import asyncio

router = APIRouter()

# Define Request and Response Models
class ImageGenerationRequest(BaseModel):
    text: str
    height: int = 1024
    width: int = 1024
    guidance_scale: float = 2.5
    img_guidance_scale: float = 1.6
    max_input_image_size: int = 1024
    separate_cfg_infer: bool = True
    use_kv_cache: bool = True
    offload_kv_cache: bool = True
    offload_model: bool = False
    use_input_image_size_as_output: bool = False
    seed: int = 0
    inference_steps: int = 50
    img1: Optional[str] = None
    img2: Optional[str] = None
    img3: Optional[str] = None

class ImageGenerationResponse(BaseModel):
    image: str  # Base64-encoded generated image

@router.post("/new-image/", response_model=ImageGenerationResponse)
async def generate_image_endpoint(request: ImageGenerationRequest, background_tasks: BackgroundTasks):
    logger = logging.getLogger(__name__)
    logger.info("Received /new-image/ request.")

    image_paths = []
    for i, img_data in enumerate([request.img1, request.img2, request.img3], start=1):
        if img_data:
            try:
                image_data = base64.b64decode(img_data)
                img = Image.open(BytesIO(image_data)).convert("RGB")
                input_image_filename = generate_unique_filename(prefix=f"input_img{i}")
                input_image_dir = "./input_images/"
                os.makedirs(input_image_dir, exist_ok=True)
                input_image_path = os.path.join(input_image_dir, input_image_filename)
                img.save(input_image_path)
                image_paths.append(input_image_path)
                logger.info(f"Saved input image {i} as {input_image_path}")
            except Exception as e:
                raise HTTPException(status_code=400, detail="Invalid image data")

    loop = asyncio.get_event_loop()
    future = loop.create_future()

    # Prepare data for processing and add it to the task queue
    task_data = {
        'data': {
            'text': request.text,
            'height': request.height,
            'width': request.width,
            'guidance_scale': request.guidance_scale,
            'img_guidance_scale': request.img_guidance_scale,
            'inference_steps': request.inference_steps,
            'separate_cfg_infer': request.separate_cfg_infer,
            'use_kv_cache': request.use_kv_cache,
            'offload_kv_cache': request.offload_kv_cache,
            'offload_model': request.offload_model,
            'use_input_image_size_as_output': request.use_input_image_size_as_output,
            'seed': request.seed,
            'max_input_image_size': request.max_input_image_size,
            'image_paths': image_paths  # Pass the paths of saved images
        },
        'future': future
    }
    
    await task_queue.put(task_data)
    logger.info("Task enqueued.")

    try:
        response = await future
        logger.info("Task completed successfully.")
        return response
    except HTTPException as he:
        logger.error(f"HTTPException: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")
     