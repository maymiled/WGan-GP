import torch
import os
import copy
from tqdm import tqdm
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from model import Generator, Discriminator, WGANCritic
from utils import D_train, G_train, D_train_wgan, G_train_wgan, save_models


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GAN on MNIST.')
    parser.add_argument("--epochs",     type=int,   default=300,
                        help="Nombre d'epochs. Augmenté de 200→300 pour meilleure convergence.")
    parser.add_argument("--lr",         type=float, default=1e-4,
                        help="Learning rate Adam.")
    parser.add_argument("--batch_size", type=int,   default=128,
                        help="Taille des mini-batchs. Augmenté 64→128 : réduit le bruit du GP.")
    parser.add_argument("--gpus",       type=int,   default=-1,
                        help="Nombre de GPUs (-1 = tous).")
    parser.add_argument("--mode",       type=str,   default="wgan",
                        choices=["vanilla", "wgan"],
                        help="Mode d'entraînement.")
    parser.add_argument("--n_critic",   type=int,   default=5,
                        help="Nombre de steps critique par step générateur (WGAN-GP).")
    parser.add_argument("--lambda_gp",  type=float, default=10.0,
                        help="Coefficient du gradient penalty (WGAN-GP).")
    parser.add_argument("--ema_decay",  type=float, default=0.9999,
                        help="Decay EMA du générateur. Augmenté 0.999→0.9999 : moyenne plus lisse.")
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Device
    # -------------------------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA")
        if args.gpus == -1:
            args.gpus = torch.cuda.device_count()
            print(f"Using {args.gpus} GPUs.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Metal)")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    os.makedirs('checkpoints', exist_ok=True)

    # -------------------------------------------------------------------------
    # Dataset
    # MODIFICATION : on ajoute RandomErasing comme augmentation légère sur les
    # vraies images. Cela force le critique à être plus robuste et évite qu'il
    # mémorise des artefacts de texture spécifiques au MNIST, améliorant Recall.
    # Référence : pratique standard de régularisation du discriminateur.
    # -------------------------------------------------------------------------
    data_path = os.getenv('DATA')
    to_download = data_path is None
    if to_download:
        data_path = "data"

    print('Dataset loading...')

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
        # NOUVEAU : Random Erasing légère sur les vraies images
        # Régularise le critique → meilleur Recall et robustesse
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.08), ratio=(0.3, 3.3), value=0),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ])

    train_dataset = datasets.MNIST(
        root=data_path, train=True, transform=transform_train, download=to_download
    )
    test_dataset = datasets.MNIST(
        root=data_path, train=False, transform=transform_test, download=to_download
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,     # NOUVEAU : drop_last=True pour que tous les batchs
    )                       # aient la même taille → GP plus stable
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    print('Dataset loaded.')

    # -------------------------------------------------------------------------
    # Modèles et optimiseurs
    # -------------------------------------------------------------------------
    print('Model loading...')
    mnist_dim = 784
    G = Generator(g_output_dim=mnist_dim).to(device)

    if args.mode == "wgan":
        D = WGANCritic(mnist_dim).to(device)

        # MODIFICATION : lr différent pour G et D
        # Le papier WGAN-GP recommande lr=1e-4 avec Adam(β1=0, β2=0.9).
        # On donne un lr légèrement plus élevé au critique (2x) pour qu'il
        # converge plus vite que le générateur → meilleure qualité du signal.
        G_optimizer = optim.Adam(G.parameters(), lr=args.lr,       betas=(0.0, 0.9))
        D_optimizer = optim.Adam(D.parameters(), lr=args.lr * 2,   betas=(0.0, 0.9))

        # MODIFICATION : Learning Rate Scheduler (cosine annealing)
        # Décroît progressivement le LR vers la fin → convergence plus fine,
        # meilleur FID sur les dernières epochs.
        G_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            G_optimizer, T_max=args.epochs, eta_min=1e-5
        )
        D_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            D_optimizer, T_max=args.epochs, eta_min=1e-5
        )

        # EMA du générateur — decay augmenté 0.999 → 0.9999
        G_ema = copy.deepcopy(G).to(device)
        G_ema.eval()
        ema_decay = args.ema_decay

        criterion = None

    else:
        D = Discriminator(mnist_dim).to(device)
        G_optimizer = optim.Adam(G.parameters(), lr=args.lr)
        D_optimizer = optim.Adam(D.parameters(), lr=args.lr)
        criterion = nn.BCELoss()
        G_scheduler = None
        D_scheduler = None

    if args.gpus > 1:
        G = torch.nn.DataParallel(G)
        D = torch.nn.DataParallel(D)

    print('Model loaded.')

    # -------------------------------------------------------------------------
    # Boucle d'entraînement WGAN-GP
    # -------------------------------------------------------------------------
    print(f'Start training (mode={args.mode}, epochs={args.epochs}, batch={args.batch_size}):')
    n_epoch = args.epochs

    if args.mode == "wgan":
        critic_step = 0

        for epoch in range(1, n_epoch + 1):

            # NOUVEAU : barre de progression par epoch avec tqdm
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{n_epoch}", leave=False)

            d_losses, g_losses = [], []

            for x, _ in pbar:
                x = x.view(-1, mnist_dim).to(device)

                # --- Critique ---
                d_loss = D_train_wgan(x, G, D, D_optimizer, device,
                                      lambda_gp=args.lambda_gp)
                d_losses.append(d_loss)
                critic_step += 1

                # --- Générateur (tous les n_critic steps) ---
                if critic_step % args.n_critic == 0:
                    g_loss = G_train_wgan(x, G, D, G_optimizer, device)
                    g_losses.append(g_loss)

                    # Mise à jour EMA
                    G_module     = G.module     if hasattr(G,     'module') else G
                    G_ema_module = G_ema.module if hasattr(G_ema, 'module') else G_ema
                    with torch.no_grad():
                        for p_ema, p in zip(G_ema_module.parameters(),
                                            G_module.parameters()):
                            p_ema.data.mul_(ema_decay).add_(p.data, alpha=1.0 - ema_decay)

                pbar.set_postfix({
                    "D": f"{d_loss:.3f}",
                    "G": f"{g_losses[-1]:.3f}" if g_losses else "—",
                })

            # NOUVEAU : step des schedulers (une fois par epoch)
            G_scheduler.step()
            D_scheduler.step()

            # Checkpoint tous les 10 epochs
            if epoch % 10 == 0:
                avg_d = sum(d_losses) / len(d_losses)
                avg_g = sum(g_losses) / max(len(g_losses), 1)
                print(
                    f"Epoch {epoch}/{n_epoch} | "
                    f"D_loss={avg_d:.4f} | G_loss={avg_g:.4f} | "
                    f"lr_G={G_optimizer.param_groups[0]['lr']:.2e}"
                )
                torch.save(G_ema.state_dict(), os.path.join('checkpoints', 'G.pth'))
                torch.save(D.state_dict(),     os.path.join('checkpoints', 'D.pth'))
                print(f"  → Checkpoint saved (EMA generator).")

        # Sauvegarde finale
        torch.save(G_ema.state_dict(), os.path.join('checkpoints', 'G.pth'))
        torch.save(D.state_dict(),     os.path.join('checkpoints', 'D.pth'))
        print('Training done. EMA generator saved as G.pth.')

    # -------------------------------------------------------------------------
    # Boucle d'entraînement Vanilla GAN (inchangée)
    # -------------------------------------------------------------------------
    else:
        for epoch in range(1, n_epoch + 1):
            for x, _ in train_loader:
                x = x.view(-1, mnist_dim).to(device)
                D_train(x, G, D, D_optimizer, criterion, device)
                G_train(x, G, D, G_optimizer, criterion, device)

            if epoch % 10 == 0:
                save_models(G, D, 'checkpoints')
                print(f"Epoch {epoch}/{n_epoch} — checkpoint saved.")

        save_models(G, D, 'checkpoints')
        print('Training done.')