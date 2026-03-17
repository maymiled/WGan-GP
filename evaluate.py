"""
evaluate.py — Métriques FID, Precision, Recall pour les images générées par le GAN.

Usage :
    python generate.py --method dot
    python evaluate.py
    python evaluate.py --samples_dir samples --k 3 --n 5000

Métriques implémentées :
  - FID  (Heusel et al., NeurIPS 2017)
  - Precision / Recall  (Kynkäänniemi et al., NeurIPS 2019)

Les features sont extraites avec InceptionV3 (pré-entraîné ImageNet, 2048-d).
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from scipy import linalg
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------------
# InceptionV3 feature extractor
# ---------------------------------------------------------------------------

def get_inception_model(device):
    """Charge InceptionV3 pré-entraîné, remplace la tête par identité (features 2048-d)."""
    try:
        from torchvision.models import Inception_V3_Weights
        inception = models.inception_v3(weights=Inception_V3_Weights.DEFAULT)
    except ImportError:
        inception = models.inception_v3(pretrained=True)

    inception.fc = nn.Identity()
    inception.aux_logits = False
    inception.eval()
    return inception.to(device)


@torch.no_grad()
def extract_features(loader, model, device):
    """Extrait les features InceptionV3 2048-d pour toutes les images du loader."""
    all_features = []
    for imgs, in loader:
        imgs = imgs.to(device)
        feats = model(imgs)
        all_features.append(feats.cpu().numpy())
    return np.concatenate(all_features, axis=0)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

class GeneratedDataset(Dataset):
    """Charge les PNG générés (grayscale) et les prépare pour InceptionV3."""

    def __init__(self, samples_dir, n, transform):
        self.files = sorted(
            [os.path.join(samples_dir, f)
             for f in os.listdir(samples_dir) if f.endswith('.png')]
        )[:n]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        return (self.transform(img),)


def _inception_transform():
    """Redimensionne → 299x299, normalise ImageNet."""
    return transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def load_generated_loader(samples_dir, n, batch_size=64):
    ds = GeneratedDataset(samples_dir, n, _inception_transform())
    return DataLoader(ds, batch_size=batch_size, num_workers=4, pin_memory=True)


def load_real_mnist_loader(n, data_path, batch_size=64):
    """Charge n images MNIST (test split) converties RGB pour InceptionV3."""
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    to_download = not os.path.exists(os.path.join(data_path, 'MNIST'))
    dataset = datasets.MNIST(root=data_path, train=False,
                             transform=transform, download=to_download)

    # Sous-échantillonner si nécessaire
    if n < len(dataset):
        indices = torch.randperm(len(dataset))[:n].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)

    # Wrapper pour renvoyer (img,) plutôt que (img, label)
    class _NoLabel(Dataset):
        def __init__(self, ds):
            self.ds = ds
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            img, _ = self.ds[idx]
            return (img,)

    return DataLoader(_NoLabel(dataset), batch_size=batch_size,
                      num_workers=4, pin_memory=True)


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------

def compute_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    FID = ||μ1 - μ2||² + Tr(Σ1 + Σ2 - 2·sqrtm(Σ1·Σ2))
    Heusel et al., NeurIPS 2017.
    """
    diff = mu1 - mu2
    # sqrtm de la matrice produit
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    # Artefact numérique : partie imaginaire négligeable
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # Régularisation si la matrice n'est pas semi-définie positive
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset)).real

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(fid)


# ---------------------------------------------------------------------------
# Precision / Recall
# ---------------------------------------------------------------------------

def compute_precision_recall(real_feats, gen_feats, k=3):
    """
    Precision / Recall d'après Kynkäänniemi et al., NeurIPS 2019.

    Manifold réel = union des boules k-NN autour de chaque sample réel.
      Rayon de la boule de real[j] = distance au k-ième voisin dans real.
    Precision = fraction des samples générés dans au moins une boule réelle.
    Recall    = fraction des samples réels dans au moins une boule générée.
    """
    # Rayon des boules réelles : distance au k-ième voisin dans real
    nn_real = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree', n_jobs=-1)
    nn_real.fit(real_feats)
    real_radii, _ = nn_real.kneighbors(real_feats)
    real_radii = real_radii[:, -1]  # k-ième voisin (le 0-ième est soi-même)

    # Precision : est-ce que gen[i] tombe dans une boule réelle ?
    dists_gen_to_real, _ = nn_real.kneighbors(gen_feats)
    dists_gen_to_real = dists_gen_to_real[:, 1:]  # ignorer index 0 si gen in train (sécurité)
    # Pour chaque gen, trouver si min dist vers un real < rayon de ce real
    nn_real2 = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', n_jobs=-1)
    nn_real2.fit(real_feats)
    dists_g2r, indices_g2r = nn_real2.kneighbors(gen_feats)
    in_real_manifold = np.any(dists_g2r <= real_radii[indices_g2r], axis=1)
    precision = float(in_real_manifold.mean())

    # Rayon des boules générées : distance au k-ième voisin dans gen
    nn_gen = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree', n_jobs=-1)
    nn_gen.fit(gen_feats)
    gen_radii, _ = nn_gen.kneighbors(gen_feats)
    gen_radii = gen_radii[:, -1]

    # Recall : est-ce que real[j] tombe dans une boule générée ?
    nn_gen2 = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', n_jobs=-1)
    nn_gen2.fit(gen_feats)
    dists_r2g, indices_r2g = nn_gen2.kneighbors(real_feats)
    in_gen_manifold = np.any(dists_r2g <= gen_radii[indices_r2g], axis=1)
    recall = float(in_gen_manifold.mean())

    return precision, recall


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate FID, Precision, Recall.')
    parser.add_argument('--samples_dir', type=str, default='samples',
                        help='Répertoire contenant les images générées (PNG).')
    parser.add_argument('--n', type=int, default=10000,
                        help='Nombre d\'images à utiliser pour les métriques.')
    parser.add_argument('--k', type=int, default=3,
                        help='k pour la k-NN Precision/Recall (paper : k=3).')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size pour l\'extraction des features.')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using device: CUDA')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print('Using device: MPS (Apple Metal)')
    else:
        device = torch.device('cpu')
        print('Using device: CPU')

    # Vérifier que le répertoire de samples existe
    if not os.path.isdir(args.samples_dir):
        raise FileNotFoundError(
            f"'{args.samples_dir}' introuvable. Lance d'abord : python generate.py"
        )

    n_gen = len([f for f in os.listdir(args.samples_dir) if f.endswith('.png')])
    n = min(args.n, n_gen)
    print(f'Images générées disponibles : {n_gen}  — utilisation : {n}')

    data_path = os.getenv('DATA', 'data')

    print('Chargement InceptionV3...')
    inception = get_inception_model(device)

    print('Extraction features — images réelles (MNIST test)...')
    real_loader = load_real_mnist_loader(n, data_path, batch_size=args.batch_size)
    real_feats = extract_features(real_loader, inception, device)

    print('Extraction features — images générées...')
    gen_loader = load_generated_loader(args.samples_dir, n, batch_size=args.batch_size)
    gen_feats = extract_features(gen_loader, inception, device)

    print(f'Features : réelles {real_feats.shape}, générées {gen_feats.shape}')

    # FID
    mu_r, sigma_r = real_feats.mean(axis=0), np.cov(real_feats, rowvar=False)
    mu_g, sigma_g = gen_feats.mean(axis=0),  np.cov(gen_feats,  rowvar=False)
    fid = compute_fid(mu_r, sigma_r, mu_g, sigma_g)

    # Precision / Recall
    print(f'Calcul Precision/Recall (k={args.k})...')
    precision, recall = compute_precision_recall(real_feats, gen_feats, k=args.k)

    print()
    print('=' * 40)
    print(f'FID       : {fid:.2f}')
    print(f'Precision : {precision:.4f}')
    print(f'Recall    : {recall:.4f}')
    print('=' * 40)
