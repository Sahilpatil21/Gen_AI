import torch
import torch.nn.functional as F


def d_loss_fn(real_scores, fake_scores):
    """Non-saturating logistic loss (discriminator)."""
    return F.softplus(-real_scores).mean() + F.softplus(fake_scores).mean()


def g_loss_fn(fake_scores):
    """Non-saturating logistic loss (generator)."""
    return F.softplus(-fake_scores).mean()


def r1_penalty(discriminator, real_images):
    """R1 gradient penalty on real images."""
    real_images = real_images.clone().requires_grad_(True)
    real_scores = discriminator(real_images)
    grad_real, = torch.autograd.grad(outputs=real_scores.sum(), inputs=real_images, create_graph=True)
    return grad_real.view(grad_real.size(0), -1).pow(2).sum(1).mean()
