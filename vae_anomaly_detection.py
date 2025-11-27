# vae_anomaly_module.py
# Clean module: label-encode categoricals + scale numerics

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# ----------------------------------------------------
# Dataset
# ----------------------------------------------------
class CrimeDataset(Dataset):
    def __init__(self, X):
        # Existing code to handle DataFrame vs. array
        arr = X.values if isinstance(X, pd.DataFrame) else X
        
        # 🔑 The fix: Explicitly convert the array to float64
        arr = arr.astype(np.float64) 
        
        # Now, create the PyTorch FloatTensor
        self.X = torch.FloatTensor(arr)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


# ----------------------------------------------------
# Data Loading + Label Encoding + Scaling
# ----------------------------------------------------
def load_and_prepare_data(filepath, test_size=0.2, random_state=42):
    """
    Loads dataframe, label-encodes each categorical column, scales numerics.
    Returns: train_scaled_df, test_scaled_df, encoders, feature_names
    """
    # Load data
    if filepath.endswith(".pkl") or filepath.endswith(".pickle"):
        df = pd.read_pickle(filepath)
    elif filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
    elif filepath.endswith(".parquet"):
        df = pd.read_parquet(filepath)
    else:
        raise ValueError("Unsupported file extension.")

    # Identify columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # One-Hot Encode categorical features
    # This is crucial for neural networks to avoid learning false ordinal relationships.
    df_processed = pd.get_dummies(df, columns=cat_cols, dummy_na=False)

    # Split after one-hot encoding to ensure train and test have the same columns
    train_df, test_df = train_test_split(df_processed, test_size=test_size, random_state=random_state)

    # Update feature names after one-hot encoding
    feature_names = train_df.columns.tolist()
    # The original numeric columns are still present, we only need to scale them.
    # The new one-hot columns are already on a {0, 1} scale.

    # Scale numerics
    scaler = MinMaxScaler()
    train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
    test_df[num_cols]  = scaler.transform(test_df[num_cols])

    # The concept of label_encoders is no longer needed with one-hot encoding
    # We return the scaler for potential inverse transforms.
    return train_df, test_df, {"scaler": scaler}, feature_names


# ----------------------------------------------------
# VAE Model
# ----------------------------------------------------
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], latent_dim=8):
        super().__init__()

        # Encoder
        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers.append(nn.Linear(prev, h))
            enc_layers.append(nn.ReLU())
            prev = h
        self.encoder = nn.Sequential(*enc_layers)

        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

        # Decoder
        dec_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.append(nn.Linear(prev, h))
            dec_layers.append(nn.ReLU())
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        dec_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*dec_layers)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


# ----------------------------------------------------
# Loss
# ----------------------------------------------------
def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    # reconstruction loss: MSE instead of BCE
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="sum")

    # KL divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * kld


# ----------------------------------------------------
# Training
# ----------------------------------------------------
def train_vae(model, dataloader, epochs=70, lr=1e-3, beta=1.0, device="cpu"):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            opt.zero_grad()
            recon, mu, logvar = model(batch)
            loss = vae_loss(recon, batch, mu, logvar, beta)
            loss.backward()
            opt.step()
            running += loss.item()
        avg_loss = running / len(dataloader.dataset)
        losses.append(avg_loss)

        # print every 10 epochs (and first epoch)
        if epoch == 0 or (epoch) % 10 == 0:
            print(f"Epoch {epoch}/{epochs} - Avg loss: {avg_loss:.6f}")

    return losses


# ----------------------------------------------------
# Reconstruction Errors
# ----------------------------------------------------
def get_reconstruction_errors(model, dataloader, device="cpu"):
    model.eval().to(device)
    errors = []
    mus = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            recon, mu, _ = model(batch)
            mse = ((recon - batch) ** 2).mean(dim=1)
            errors.extend(mse.cpu().numpy())
            mus.append(mu.cpu().numpy())

    return np.array(errors), np.vstack(mus)


# ----------------------------------------------------
# Anomaly Detection
# ----------------------------------------------------
def detect_anomalies(errors, recon_percentile=97.5):
    """
    Detects anomalies based on reconstruction errors.
    Anomalies are data points with a reconstruction error above a given percentile.
    """
    recon_thresh = np.percentile(errors, recon_percentile)
    is_anomaly = errors > recon_thresh
    return is_anomaly, recon_thresh


# ----------------------------------------------------
# Simple per-feature reconstruction error (without inverse transform)
# ----------------------------------------------------
def per_feature_errors(model, dataloader, feature_names, device="cpu"):
    model.eval().to(device)
    errors = []
    originals = []
    recons = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            recon, _, _ = model(batch)
            errors.append(((recon - batch) ** 2).cpu().numpy())
            originals.append(batch.cpu().numpy())
            recons.append(recon.cpu().numpy())

    return (
        pd.DataFrame(np.vstack(errors), columns=feature_names),
        pd.DataFrame(np.vstack(originals), columns=feature_names),
        pd.DataFrame(np.vstack(recons), columns=feature_names),
    )
