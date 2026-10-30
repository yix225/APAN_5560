
import torch
import torch.nn as nn
import torch.optim as optim


def train_gan(
    model,
    data_loader,
    criterion=None,
    optimizer=None,
    device="cpu",
    epochs=10
):
    generator, discriminator = model

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    if criterion is None:
        criterion = nn.BCELoss()

    if optimizer is None:
        optimizer_g = optim.Adam(
            generator.parameters(),
            lr=0.0002,
            betas=(0.5, 0.999)
        )
        optimizer_d = optim.Adam(
            discriminator.parameters(),
            lr=0.0002,
            betas=(0.5, 0.999)
        )
    else:
        optimizer_g, optimizer_d = optimizer

    for epoch in range(epochs):
        generator.train()
        discriminator.train()

        total_g_loss = 0.0
        total_d_loss = 0.0

        for real_images, _ in data_loader:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            real_labels = torch.ones(
                batch_size, 1, device=device
            )
            fake_labels = torch.zeros(
                batch_size, 1, device=device
            )

            # 1. Train Discriminator
            optimizer_d.zero_grad()

            real_output = discriminator(real_images)
            loss_real = criterion(
                real_output, real_labels
            )

            noise = torch.randn(
                batch_size, 100, device=device
            )
            fake_images = generator(noise)

            fake_output = discriminator(
                fake_images.detach()
            )
            loss_fake = criterion(
                fake_output, fake_labels
            )

            d_loss = loss_real + loss_fake
            d_loss.backward()
            optimizer_d.step()

            # 2. Train Generator
            optimizer_g.zero_grad()

            noise = torch.randn(
                batch_size, 100, device=device
            )
            generated_images = generator(noise)

            output = discriminator(generated_images)

            g_loss = criterion(
                output, real_labels
            )

            g_loss.backward()
            optimizer_g.step()

            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()

        avg_d = total_d_loss / len(data_loader)
        avg_g = total_g_loss / len(data_loader)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"D Loss: {avg_d:.4f}, "
            f"G Loss: {avg_g:.4f}",
            flush=True
        )

    return generator, discriminator
