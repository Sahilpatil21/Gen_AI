"""Train the StyleGAN-style model on CelebA (64x64).  Usage: python train.py"""
import os
import torch
import torch.optim as optim

import config
from data import get_dataloader
from models import MappingNetwork, Generator, Discriminator
from losses import d_loss_fn, g_loss_fn, r1_penalty
from utils import set_seed, save_grid, plot_losses


def main():
    set_seed(config.SEED)
    device = config.DEVICE
    print("Using device:", device)

    dataloader = get_dataloader()
    print("Number of batches per epoch:", len(dataloader))

    mapping = MappingNetwork(config.Z_DIM, config.W_DIM, config.MAPPING_LAYERS).to(device)
    generator = Generator(w_dim=config.W_DIM).to(device)
    discriminator = Discriminator().to(device)

    n_g = sum(p.numel() for p in generator.parameters()) + sum(p.numel() for p in mapping.parameters())
    n_d = sum(p.numel() for p in discriminator.parameters())
    print(f"Generator + Mapping parameters: {n_g:,}\nDiscriminator parameters: {n_d:,}")

    opt_g = optim.Adam(generator.parameters(), lr=config.LR_G, betas=config.ADAM_BETAS)
    opt_m = optim.Adam(mapping.parameters(), lr=config.LR_G * config.MAPPING_LR_MULT, betas=config.ADAM_BETAS)
    opt_d = optim.Adam(discriminator.parameters(), lr=config.LR_D, betas=config.ADAM_BETAS)

    os.makedirs(config.SAMPLES_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    fixed_z = torch.randn(16, config.Z_DIM, device=device)
    g_losses, d_losses = [], []
    step = 0

    for epoch in range(config.NUM_EPOCHS):
        for batch_idx, real in enumerate(dataloader):
            real = real.to(device)
            bs = real.size(0)

            # 1. Discriminator
            z = torch.randn(bs, config.Z_DIM, device=device)
            with torch.no_grad():
                fake = generator(mapping(z))
            loss_d = d_loss_fn(discriminator(real), discriminator(fake.detach()))
            if step % config.R1_INTERVAL == 0:
                loss_d = loss_d + (config.R1_GAMMA / 2) * r1_penalty(discriminator, real) * config.R1_INTERVAL
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

            # 2. Generator + mapping network
            z = torch.randn(bs, config.Z_DIM, device=device)
            fake = generator(mapping(z))
            loss_g = g_loss_fn(discriminator(fake))
            opt_g.zero_grad(set_to_none=True)
            opt_m.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()
            opt_m.step()

            g_losses.append(loss_g.item())
            d_losses.append(loss_d.item())
            if step % config.SAMPLE_EVERY == 0:
                print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}] Step [{batch_idx}/{len(dataloader)}] "
                      f"Loss_D: {loss_d.item():.4f}  Loss_G: {loss_g.item():.4f}")
            step += 1

        # end of epoch: save samples + checkpoint
        generator.eval(); mapping.eval()
        with torch.no_grad():
            samples = generator(mapping(fixed_z))
        generator.train(); mapping.train()
        tag = f"epoch_{epoch+1:03d}"
        save_grid(samples, os.path.join(config.SAMPLES_DIR, f"{tag}.png"), nrow=4, title=f"Generated Faces — {tag}")
        torch.save({
            "epoch": epoch + 1,
            "mapping": mapping.state_dict(),
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
        }, os.path.join(config.CHECKPOINT_DIR, "stylegan_latest.pt"))

    plot_losses(g_losses, d_losses, os.path.join(config.RESULTS_DIR, "loss_curves.png"))
    print("Training complete.")


if __name__ == "__main__":
    main()
