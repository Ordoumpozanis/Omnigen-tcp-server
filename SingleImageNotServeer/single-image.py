import os
import base64
import logging
from typing import Optional
from io import BytesIO
import asyncio

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from OmniGen import OmniGenPipeline
from utils import generate_unique_filename, task_queue,upscale_image_with_flow

# Initialize Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the OmniGenPipeline
CUSTOM_MODEL = "../Models/"  # Adjust the path as necessary
try:
    pipe = OmniGenPipeline.from_pretrained(CUSTOM_MODEL)
    logger.info("OmniGenPipeline loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load OmniGenPipeline: {e}")
    pipe = None  # Pipeline is not loaded; server should handle this gracefully

# Background Worker Function
async def worker():
    while True:
        try:
            task = await task_queue.get()
            if task is None:
                logger.info("Worker received shutdown signal.")
                break
            await process_task(task)
            task_queue.task_done()
        except asyncio.CancelledError:
            logger.info("Worker task canceled during shutdown.")
            break
        except Exception as e:
            logger.error(f"Error processing task: {e}")

async def process_task(task):
    request_data = task['data']
    future = task['future']

    if not pipe:
        future.set_exception(HTTPException(status_code=500, detail="Model not loaded"))
        return

    input_images = request_data.get('image_paths', [])

    try:
        output = pipe(
            prompt=request_data['text'],
            input_images=input_images if input_images else None,
            height=request_data['height'],
            width=request_data['width'],
            guidance_scale=request_data['guidance_scale'],
            img_guidance_scale=request_data['img_guidance_scale'],
            num_inference_steps=request_data['inference_steps'],
            separate_cfg_infer=request_data['separate_cfg_infer'],
            use_kv_cache=request_data['use_kv_cache'],
            offload_kv_cache=request_data['offload_kv_cache'],
            offload_model=request_data['offload_model'],
            use_input_image_size_as_output=request_data['use_input_image_size_as_output'],
            seed=request_data['seed'],
            max_input_image_size=request_data['max_input_image_size'],
        )
        logger.info("Image generation successful.")
        img = output[0]

        # Step 3: Upscale the image directly using `upscale_image_with_flow`
        upscaled_img = await upscale_image_with_flow(img, prompt=request_data['text'])
    
        buffered = BytesIO()
        upscaled_img.save(buffered, format="PNG")
        # img.save(buffered, format="PNG")
        
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        future.set_result({"image": img_base64})
    except Exception as e:
        logger.error(f"Error during image generation: {e}")
        future.set_exception(HTTPException(status_code=500, detail="Image generation failed"))
    finally:
        for path in input_images:
            try:
                os.remove(path)
                logger.info(f"Deleted input image: {path}")
            except Exception as e:
                logger.warning(f"Could not delete input image {path}: {e}")

# Startup Event: Start the Background Worker
@app.on_event("startup")
async def startup_event():
    app.state.worker = asyncio.create_task(worker())
    logger.info("Background worker started.")

# Shutdown Event: Stop the Background Worker
@app.on_event("shutdown")
async def shutdown_event():
    await task_queue.put(None)  # Send shutdown signal
    await app.state.worker
    logger.info("Background worker stopped.")

# Import the routes
from routes import root, generate_image

# Add routes to the app
app.include_router(root.router)
app.include_router(generate_image.router)