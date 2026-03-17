import torch
import os


# ---------------------------------------------------------------------------
# WGAN-GP helpers
# ---------------------------------------------------------------------------

def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """
    Gradient penalty pour WGAN-GP (Gulrajani et al., 2017).

    MODIFICATION : On clamp les interpolations dans [-1, 1] pour rester dans
    le support valide des données MNIST normalisées. Cela évite que le critique
    soit évalué sur des points hors-distribution, ce qui stabilise le GP.
    """
    alpha = torch.rand(real_samples.size(0), 1, device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples)
    interpolates = interpolates.clamp(-1.0, 1.0)           # NOUVEAU : clamp dans le support
    interpolates = interpolates.requires_grad_(True)

    d_interpolates = D(interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def D_train_wgan(x, G, D, D_optimizer, device, lambda_gp=10):
    """
    Train WGAN-GP critic pour un step.

    MODIFICATION : On utilise un z différent et indépendant de celui du step G
    pour éviter toute corrélation entre les fake_samples du critique et ceux
    du générateur. Cela améliore la stabilité du gradient penalty.
    """
    D.zero_grad()
    real_samples = x.to(device)
    batch_size = real_samples.size(0)

    # z indépendant — pas de seed partagé avec G_train
    z = torch.randn(batch_size, 100, device=device)
    with torch.no_grad():
        fake_samples = G(z)

    d_real = D(real_samples).mean()
    d_fake = D(fake_samples).mean()
    gp = compute_gradient_penalty(D, real_samples, fake_samples.detach(), device)

    d_loss = -d_real + d_fake + lambda_gp * gp
    d_loss.backward()

    # NOUVEAU : gradient clipping sur le critique pour éviter les explosions
    # Recommandé dans la pratique WGAN-GP pour les MLP profonds
    torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)

    D_optimizer.step()
    return d_loss.item()


def G_train_wgan(x, G, D, G_optimizer, device):
    """
    Train WGAN-GP generator pour un step.

    MODIFICATION : gradient clipping sur le générateur également,
    pour plus de stabilité (même raison que pour le critique).
    """
    G.zero_grad()
    batch_size = x.size(0)
    z = torch.randn(batch_size, 100, device=device)
    fake_samples = G(z)
    g_loss = -D(fake_samples).mean()
    g_loss.backward()

    # NOUVEAU : gradient clipping sur le générateur
    torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)

    G_optimizer.step()
    return g_loss.item()


def load_critic(D, folder, device):
    """Charge le checkpoint du critique en gérant DataParallel et les changements d'architecture.

    Gère deux cas :
    - Checkpoint plain (fc1.weight) → modèle plain : chargement direct (strict=True).
    - Checkpoint plain (fc1.weight) → modèle avec Spectral Norm (fc1.weight_orig) :
      traduit weight → weight_orig puis strict=False. PyTorch initialise weight_u/weight_v
      et les params LayerNorm absents ; ils convergent en quelques forward passes.
    """
    ckpt_path = os.path.join(folder, 'D.pth')
    raw = torch.load(ckpt_path, map_location=device)
    raw = {k.replace('module.', ''): v for k, v in raw.items()}

    model_keys = set(D.state_dict().keys())

    if model_keys == set(raw.keys()):
        D.load_state_dict(raw)
        return D

    # Adaptation plain → Spectral Norm : fcX.weight → fcX.weight_orig
    adapted = {}
    for k, v in raw.items():
        if k in model_keys:
            adapted[k] = v
        elif k.endswith('.weight'):
            sn_key = k.replace('.weight', '.weight_orig')
            if sn_key in model_keys:
                adapted[sn_key] = v

    D.load_state_dict(adapted, strict=False)
    return D


# ---------------------------------------------------------------------------
# Vanilla GAN helpers (inchangés)
# ---------------------------------------------------------------------------

def D_train(x, G, D, D_optimizer, criterion, device):
    D.zero_grad()

    x_real = x.to(device)
    y_real = torch.ones(x.shape[0], 1, device=device)
    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)

    z = torch.randn(x.shape[0], 100, device=device)
    x_fake = G(z)
    y_fake = torch.zeros(x.shape[0], 1, device=device)
    D_output = D(x_fake)
    D_fake_loss = criterion(D_output, y_fake)

    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()
    return D_loss.data.item()


def G_train(x, G, D, G_optimizer, criterion, device):
    G.zero_grad()
    z = torch.randn(x.shape[0], 100, device=device)
    y = torch.ones(x.shape[0], 1, device=device)
    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)
    G_loss.backward()
    G_optimizer.step()
    return G_loss.data.item()


def save_models(G, D, folder):
    torch.save(G.state_dict(), os.path.join(folder, 'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder, 'D.pth'))


def load_model(G, folder, device):
    ckpt_path = os.path.join(folder, 'G.pth')
    ckpt = torch.load(ckpt_path, map_location=device)
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G