import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """
    Architecture FIXE — ne pas modifier (consigne du projet).
    Génère une image 28x28 (784 pixels) à partir d'un vecteur latent z de taille 100.
    """
    def __init__(self, g_output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features * 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features * 2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))


class Discriminator(nn.Module):
    """
    Discriminateur pour le GAN vanilla (mode BCE).
    Inchangé par rapport à l'original.
    """
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features // 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features // 2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.sigmoid(self.fc4(x))


class WGANCritic(nn.Module):
    """
    Critique WGAN-GP amélioré.

    Modifications par rapport à l'original :

    1. SPECTRAL NORMALIZATION (SN) sur chaque couche linéaire
       - Raison : Le papier DOT (Tanaka 2019) s'appuie sur SN-GAN pour ses expériences
         et montre que SN stabilise la constante de Lipschitz k_eff (Figure 2 du papier DOT).
         SN contraint ||D||_Lip <= 1 couche par couche, ce qui rend l'estimation de k_eff
         plus fiable et donc l'algorithme DOT plus précis.
       - Référence : Miyato et al. 2018, cité dans les deux papiers.

    2. LAYER NORMALIZATION après chaque activation (sauf dernière couche)
       - Raison : Le papier WGAN-GP (Gulrajani 2017, Section 4) interdit explicitement
         BatchNorm dans le critique car elle corrèle les exemples du batch et invalide
         le gradient penalty. Layer Norm est recommandée comme alternative (Section 4,
         "No critic batch normalization"). Elle stabilise l'entraînement sans introduire
         de corrélation entre exemples.

    3. DROPOUT (p=0.1) léger
       - Raison : WGAN-GP (Figure 5b) montre que le critique overfitte facilement sur MNIST.
         Un dropout très léger (0.1) régularise sans trop contraindre la capacité du critique,
         ce qui améliore la généralisation et donc la qualité des images générées (meilleur FID).

    4. Pas de Sigmoid en sortie (inchangé)
       - Nécessaire pour WGAN : la sortie doit être un score réel non borné.
    """
    def __init__(self, d_input_dim, dropout_rate=0.1):
        super(WGANCritic, self).__init__()

        # Spectral Normalization wraps each Linear layer
        self.fc1 = nn.utils.spectral_norm(nn.Linear(d_input_dim, 1024))
        self.fc2 = nn.utils.spectral_norm(nn.Linear(1024, 512))
        self.fc3 = nn.utils.spectral_norm(nn.Linear(512, 256))
        self.fc4 = nn.utils.spectral_norm(nn.Linear(256, 1))

        # Layer Normalization — une norme par couche (pas de corrélation inter-exemples)
        self.ln1 = nn.LayerNorm(1024)
        self.ln2 = nn.LayerNorm(512)
        self.ln3 = nn.LayerNorm(256)

        # Dropout léger pour régulariser le critique
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.dropout(F.leaky_relu(self.ln1(self.fc1(x)), 0.2))
        x = self.dropout(F.leaky_relu(self.ln2(self.fc2(x)), 0.2))
        x = self.dropout(F.leaky_relu(self.ln3(self.fc3(x)), 0.2))
        return self.fc4(x)  # Pas de sigmoid : score réel pour WGAN