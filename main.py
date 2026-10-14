from fastapi import FastAPI
from pydantic import BaseModel

from app.bigram_model import BigramModel
from app.embedding_model import EmbeddingModel
import io

import torch

from PIL import Image
from fastapi import UploadFile, File, HTTPException
from torchvision import transforms

from app.cnn_model import CNNModel


app = FastAPI()


# -------------------------
# Bigram model
# -------------------------

corpus = [
    "the cat sat on the mat",
    "the cat ate the food",
    "the dog sat on the floor",
    "the dog ate the food"
]

bigram_model = BigramModel(corpus)


class GenerationRequest(BaseModel):
    start_word: str
    length: int


@app.get("/")
def read_root():
    return {
        "message": "Generative Text API is running"
    }


@app.post("/generate")
def generate_text(request: GenerationRequest):
    generated_text = bigram_model.generate_text(
        request.start_word,
        request.length
    )

    return {
        "generated_text": generated_text
    }


# -------------------------
# Embedding model
# -------------------------

embedding_model = EmbeddingModel()


class EmbeddingRequest(BaseModel):
    word: str


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    word = request.word.strip()

    if not word:
        return {
            "error": "Word cannot be empty"
        }

    embedding = embedding_model.get_embedding(word)

    return {
        "word": word,
        "dimension": len(embedding),
        "embedding": embedding
    
    }




#-------------------------
# Assignment 2 
#-------------------------


# CIFAR-10 categories
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Use the same preprocessing as training
image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

# Load the trained CNN
cnn_model = CNNModel()

cnn_model.load_state_dict(
    torch.load(
        "models/cifar10_cnn.pth",
        map_location="cpu",
        weights_only=True
    )
)

cnn_model.eval()


@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    # Read uploaded image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image."
        )

    # Preprocess image
    image_tensor = image_transform(image)
    image_tensor = image_tensor.unsqueeze(0)

    # Predict class
    with torch.no_grad():
        outputs = cnn_model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(
            probabilities, dim=1
        )

    class_name = CLASS_NAMES[predicted.item()]

    return {
        "filename": file.filename,
        "predicted_class": class_name,
        "confidence": round(confidence.item(), 4)
    }
