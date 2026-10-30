
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def generate_samples(
    model,
    device="cpu",
    num_samples=10,
    save_path=None
):
    model.eval()

    with torch.no_grad():
        noise = torch.randn(
            num_samples, 100, device=device
        )

        images = model(noise)
        images = (images + 1) / 2

        grid = make_grid(
            images.cpu(),
            nrow=min(5, num_samples),
            padding=2
        )

    plt.figure(figsize=(8, 8))
    plt.imshow(
        grid.permute(1, 2, 0).numpy()
    )
    plt.axis("off")

    if save_path is not None:
        plt.savefig(
            save_path,
            bbox_inches="tight"
        )

    plt.show()
    plt.close()

    return images
