import torch
import torchvision
import os
import argparse

from model import Generator, WGANCritic
from utils import load_model, load_critic


# ---------------------------------------------------------------------------
# Sampling methods (baseline, hard/soft truncation — inchangées)
# ---------------------------------------------------------------------------

def sample_baseline(G, batch_size, device):
    """Standard sampling: z ~ N(0, I)."""
    z = torch.randn(batch_size, 100, device=device)
    with torch.no_grad():
        x = G(z)
    return x


def sample_hard_truncation(G, batch_size, device, threshold=1.5):
    """Rejection sampling: discard z when ||z||_2 > threshold."""
    accepted = []
    while len(accepted) < batch_size:
        z = torch.randn(batch_size * 2, 100, device=device)
        mask = z.norm(dim=1) <= threshold
        z = z[mask]
        if z.size(0) > 0:
            with torch.no_grad():
                x = G(z)
            accepted.append(x)
    return torch.cat(accepted, dim=0)[:batch_size]


def sample_soft_truncation(G, batch_size, device, psi=0.7):
    """Soft truncation: z_soft = psi * z."""
    z = psi * torch.randn(batch_size, 100, device=device)
    with torch.no_grad():
        x = G(z)
    return x


# ---------------------------------------------------------------------------
# k_eff estimation
# MODIFICATION : on augmente n_pairs par défaut à 2000 et on prend le 95e
# percentile plutôt que le max absolu.
#
# Pourquoi ?
# Le papier DOT (Tanaka 2019, Section 3.2, eq. 21) calcule k_eff comme le MAX
# sur des paires aléatoires. Mais le max est très sensible aux outliers, surtout
# avec peu de paires. Avec le percentile 95, on obtient une estimation plus
# stable et reproductible de k_eff, ce qui rend l'algorithme DOT plus robuste
# (moins de risque de sur- ou sous-estimer le transport optimal).
# Le papier lui-même note : "it is recommended to use k_eff calculated by using
# enough number of samples" (Section 4.1).
# ---------------------------------------------------------------------------

def estimate_keff(G, D, device, n_pairs=2000, percentile=95):
    """
    Estime la constante de Lipschitz effective de D∘G.

    Paramètres
    ----------
    n_pairs     : nombre de paires aléatoires (augmenté 500→2000 pour stabilité)
    percentile  : percentile utilisé au lieu du max brut (95 recommandé)
    """
    latent_dim = 100
    z1 = torch.randn(n_pairs, latent_dim, device=device)
    z2 = torch.randn(n_pairs, latent_dim, device=device)

    with torch.no_grad():
        dg1 = D(G(z1)).squeeze()
        dg2 = D(G(z2)).squeeze()
        diff_dg = (dg1 - dg2).abs()
        diff_z  = (z1 - z2).norm(dim=1)
        ratios  = diff_dg / (diff_z + 1e-8)

    # Percentile au lieu du max absolu → plus stable
    k_eff = torch.quantile(ratios, percentile / 100.0).item()
    return max(k_eff, 1e-3)


# ---------------------------------------------------------------------------
# DOT sampling
# MODIFICATIONS par rapport à l'original :
#
# 1. WARM-UP du learning rate DOT (5 premiers steps à lr/5)
#    Le papier DOT note que "a large ε may accelerate upgrading, but easily
#    downgrade unless appropriate N_updates is chosen" (Section 4.1).
#    Un warm-up évite les grands sauts initiaux qui font sortir z de la sphère.
#
# 2. CLIPPING de z après chaque step dans [-4, 4]
#    La prior est N(0, I) avec support concentré autour de ||z||=sqrt(D)≈10.
#    Sans clipping, Adam peut pousser z vers des régions très hors-support,
#    produisant des images dégradées. Le clipping est cohérent avec l'Algorithm 2
#    du papier (clip z ∈ [-1,1] pour prior uniforme ; on adapte à la gaussienne).
#
# 3. Projection sphérique appliquée sur le gradient ACCUMULÉ (avant Adam)
#    Conformément à l'Algorithm 2 du papier (ligne : g ← g − (g·z)z/√D).
#    L'implémentation originale était correcte, on la conserve.
#
# 4. n_updates par défaut augmenté 30→50
#    Le papier montre (Figure 5) que plus de steps améliore les scores jusqu'à
#    ~30-50 updates. On utilise 50 comme défaut pour maximiser la qualité.
# ---------------------------------------------------------------------------

def sample_dot(G, D, k_eff, batch_size, device,
               n_updates=50, lr=0.01, delta=1e-3, warmup_steps=5):
    """
    Latent-space DOT sampling avec Adam et warm-up (Tanaka, NeurIPS 2019 — Algorithm 2).

    Paramètres
    ----------
    n_updates    : nombre de steps Adam dans l'espace latent (50 par défaut)
    lr           : learning rate Adam (0.01 comme dans le papier, Figure 5)
    delta        : petit vecteur pour éviter overflow (1e-3 comme dans le papier)
    warmup_steps : nombre de steps de warm-up avec lr réduit
    """
    latent_dim = 100
    z_y = torch.randn(batch_size, latent_dim, device=device)
    z   = z_y.clone().requires_grad_(True)

    G.eval()
    D.eval()

    # Adam(α=0.01, β1=0, β2=0.9) — exactement comme dans Tanaka 2019 (Figure 5)
    optimizer = torch.optim.Adam([z], lr=lr, betas=(0.0, 0.9))

    for step in range(n_updates):

        # NOUVEAU : warm-up — lr réduit pendant les premiers steps
        # Évite les grands sauts qui feraient sortir z de la sphère gaussienne
        if step < warmup_steps:
            for pg in optimizer.param_groups:
                pg['lr'] = lr * (step + 1) / warmup_steps
        else:
            for pg in optimizer.param_groups:
                pg['lr'] = lr

        optimizer.zero_grad()

        dist = (z - z_y + delta).norm(dim=1).mean()
        loss = dist - (1.0 / k_eff) * D(G(z)).mean()
        loss.backward()

        # Projection sphérique du gradient : g ← g − (g·z)z/√D
        # Conformément à Algorithm 2 du papier (support de N(0,I) concentré
        # sur la sphère de rayon √D en grande dimension)
        with torch.no_grad():
            g   = z.grad.clone()
            dot = (g * z).sum(dim=1, keepdim=True)
            z.grad.copy_(g - dot * z / (latent_dim ** 0.5))

        optimizer.step()

        # NOUVEAU : clipping de z dans [-4, 4]
        # Reste dans le support probable de N(0,I) en dim 100
        # (||z|| ≈ 10 → composantes individuelles rarement > 4 à 3σ)
        with torch.no_grad():
            z.clamp_(-4.0, 4.0)

    with torch.no_grad():
        x = G(z)
    return x.detach()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate MNIST samples.')
    parser.add_argument("--batch_size",    type=int,   default=512)
    parser.add_argument("--method",        type=str,   default="dot",
                        choices=["baseline", "hard_truncation", "soft_truncation", "dot"])
    parser.add_argument("--n_updates",     type=int,   default=50,
                        help="DOT: nombre de steps Adam (50 par défaut, vs 30 original).")
    parser.add_argument("--dot_lr",        type=float, default=0.01,
                        help="DOT: learning rate Adam.")
    parser.add_argument("--soft_psi",      type=float, default=0.7,
                        help="Soft truncation: facteur ψ.")
    parser.add_argument("--hard_threshold",type=float, default=1.5,
                        help="Hard truncation: seuil ||z||_2.")
    parser.add_argument("--keff_pairs",    type=int,   default=2000,
                        help="DOT: nombre de paires pour estimer k_eff (2000 par défaut).")
    parser.add_argument("--keff_percentile", type=float, default=95,
                        help="DOT: percentile pour k_eff (95 par défaut, vs max original).")
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Metal)")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    print('Model Loading...')
    mnist_dim = 784

    G = Generator(g_output_dim=mnist_dim).to(device)
    G = load_model(G, 'checkpoints', device)
    if torch.cuda.device_count() > 1:
        G = torch.nn.DataParallel(G)
    G.eval()

    # Chargement du critique pour DOT
    method      = args.method
    D           = None
    critic_path = os.path.join('checkpoints', 'D.pth')

    if method == "dot":
        if os.path.exists(critic_path):
            D = WGANCritic(mnist_dim).to(device)
            D = load_critic(D, 'checkpoints', device)
            D.eval()
            print(f"Critic loaded. Estimating k_eff "
                  f"(n_pairs={args.keff_pairs}, p{args.keff_percentile})...")
            k_eff = estimate_keff(G, D, device,
                                  n_pairs=args.keff_pairs,
                                  percentile=args.keff_percentile)
            print(f"k_eff = {k_eff:.4f}")
        else:
            print("WARNING: checkpoints/D.pth not found. Falling back to soft_truncation.")
            method = "soft_truncation"

    print(f'Generating with method: {method}')
    os.makedirs('samples', exist_ok=True)

    n_samples = 0
    while n_samples < 10000:
        remaining = 10000 - n_samples
        bs = min(args.batch_size, remaining)

        if method == "baseline":
            x = sample_baseline(G, bs, device)
        elif method == "hard_truncation":
            x = sample_hard_truncation(G, bs, device, threshold=args.hard_threshold)
        elif method == "soft_truncation":
            x = sample_soft_truncation(G, bs, device, psi=args.soft_psi)
        elif method == "dot":
            x = sample_dot(G, D, k_eff, bs, device,
                           n_updates=args.n_updates,
                           lr=args.dot_lr)

        x = x.reshape(bs, 28, 28)
        for k in range(x.shape[0]):
            if n_samples < 10000:
                torchvision.utils.save_image(
                    x[k:k+1],
                    os.path.join('samples', f'{n_samples}.png')
                )
                n_samples += 1

    print(f'Done. {n_samples} images saved to samples/.')