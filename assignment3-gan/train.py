
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from helper_lib.model import get_model
from helper_lib.trainer import train_gan
from helper_lib.generator import generate_samples


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    torch.manual_seed(42)

    # Choose computing device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("Using device:", device, flush=True)

    # Prepare MNIST images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    data_loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )

    print(
        "Training images:", len(dataset),
        flush=True
    )

    # Create GAN
    model = get_model("GAN")

    # Train GAN
    generator, discriminator = train_gan(
        model=model,
        data_loader=data_loader,
        device=device,
        epochs=20
    )

    # Save weights
    torch.save(
        generator.state_dict(),
        "models/gan_generator.pth"
    )

    torch.save(
        discriminator.state_dict(),
        "models/gan_discriminator.pth"
    )

    print("Models saved successfully!")

    # Generate example digits
    generate_samples(
        generator,
        device=device,
        num_samples=25,
        save_path="outputs/generated_digits.png"
    )

    print("Generated images saved!")


if __name__ == "__main__":
    main()
