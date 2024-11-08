from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def read_root():
    return {"message": "Welcome to the OmniGen Image Generation API! Use the /new-image route to generate images."}