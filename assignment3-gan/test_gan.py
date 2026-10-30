
import torch
from helper_lib.model import get_model

G, D = get_model("GAN")

batch_size = 16

z = torch.randn(batch_size, 100)

fake_images = G(z)
predictions = D(fake_images)

print("Generator:", fake_images.shape)
print("Discriminator:", predictions.shape)

assert fake_images.shape == (16, 1, 28, 28)
assert predictions.shape == (16, 1)

assert fake_images.min() >= -1
assert fake_images.max() <= 1

print("All tests passed!")
