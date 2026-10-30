
from pathlib import Path
from io import BytesIO

import torch
from PIL import Image
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response

from helper_lib.model import Generator

app = FastAPI(
    title="APAN 5560 GAN API",
    description="Generate MNIST-style handwritten digits"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "gan_generator.pth"

device = torch.device("cpu")
generator = None


def load_generator():
    model = Generator().to(device)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()
    return model


@app.on_event("startup")
def startup():
    global generator

    if MODEL_PATH.exists():
        generator = load_generator()


@app.get("/")
def home():
    return {
        "message": "GAN API is running",
        "model_loaded": generator is not None
    }


@app.get("/gan/generate")
def generate_digit():
    if generator is None:
        raise HTTPException(
            status_code=503,
            detail="Train the GAN model first."
        )

    with torch.no_grad():
        noise = torch.randn(1, 100, device=device)
        image = generator(noise)

        image = image[0, 0]
        image = ((image + 1) / 2 * 255)
        image = image.clamp(0, 255)
        image = image.byte().cpu().numpy()

    pil_image = Image.fromarray(image)

    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")

    return Response(
        content=buffer.getvalue(),
        media_type="image/png"
    )


@app.get("/gan/generate-batch")
def generate_batch(
    count: int = Query(default=10, ge=1, le=64)
):
    if generator is None:
        raise HTTPException(
            status_code=503,
            detail="Train the GAN model first."
        )

    from torchvision.utils import make_grid

    with torch.no_grad():
        noise = torch.randn(
            count, 100, device=device
        )

        images = generator(noise)
        images = (images + 1) / 2

        grid = make_grid(
            images.cpu(),
            nrow=min(8, count),
            padding=2
        )

        grid = (grid * 255).clamp(0, 255)
        grid = grid.byte()
        grid = grid.permute(1, 2, 0).numpy()

    pil_image = Image.fromarray(grid)

    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")

    return Response(
        content=buffer.getvalue(),
        media_type="image/png"
    )
