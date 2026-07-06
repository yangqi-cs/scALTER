#!/usr/bin/env python3
import argparse
import json
import os
import math
import pickle
import scanpy as sc
from scipy import sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.distributions import Normal, kl_divergence as kl


# ============================================
# 0. Configuration
# ============================================

te_level = "subfamily"

# 小鼠 20% 小样本数据
# BASE_DIR = "/qiyang/TEexp/Data/TEexp/mouse_chemical_reprogramming/agg_new_m_strategy/my_results"
# DATA_DIR = "orig_s_l"
# OUTPUT_DIR = f"recon_{te_level}/scvi_style_poe"

# 人类GBM数据
# BASE_DIR = "/qiyang/TEexp/Data/TEexp/human_gbm_smartseq/my_results_mates_div_nh"
# DATA_DIR = "orig_s_l"
# OUTPUT_DIR = f"recon_{te_level}/scvi_style_poe"

# PBMC8k default data, produced by scripts/2_build_u_m_views.py
BASE_DIR = "/qiyang/GitHub/scALTER/results/pbmc8k/my_subfamily"
DATA_DIR = "2_subfamily_u_m_aligned/aligned_npz"
OUTPUT_DIR = "3_cross_view/1_7_scvi_poe_unique_multi_merge_best_model"

UNIQUE_NPZ = "unique.npz"
MULTI_NPZ = "multi.npz"
MERGE_NPZ = "merge.npz"
BARCODE_FILE = "barcodes.tsv"
FEATURE_FILE = "features.tsv"

# Model/training parameters
N_HIDDEN = 128
N_LATENT = 32
N_LAYERS = 1
DROPOUT_RATE = 0.0
GENE_LIKELIHOOD = "nb"  # "nb" or "zinb"
USE_BATCH_NORM = True
USE_LAYER_NORM = False
KL_WEIGHT = 0.00001  # Very small for TE data
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
PREDICT_BATCH_SIZE = 128
OUTPUT_CHUNK_FEATURES = 256
EPOCHS = 300
EARLY_STOP = 15
REDUCE_LR = 10
RANDOM_SEED = 42


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Train the scALTER three-view scVI-PoE model on unique/multi/merge matrices."
    )
    parser.add_argument("--te-level", default=te_level)
    parser.add_argument("--base-dir", default=BASE_DIR)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--unique-npz", default=UNIQUE_NPZ)
    parser.add_argument("--multi-npz", default=MULTI_NPZ)
    parser.add_argument("--merge-npz", default=MERGE_NPZ)
    parser.add_argument("--barcodes", default=BARCODE_FILE)
    parser.add_argument("--features", default=FEATURE_FILE)
    parser.add_argument("--n-hidden", type=int, default=N_HIDDEN)
    parser.add_argument("--n-latent", type=int, default=N_LATENT)
    parser.add_argument("--n-layers", type=int, default=N_LAYERS)
    parser.add_argument("--dropout-rate", type=float, default=DROPOUT_RATE)
    parser.add_argument("--gene-likelihood", choices=["nb", "zinb"], default=GENE_LIKELIHOOD)
    parser.add_argument("--kl-weight", type=float, default=KL_WEIGHT)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--predict-batch-size", type=int, default=PREDICT_BATCH_SIZE)
    parser.add_argument("--output-chunk-features", type=int, default=OUTPUT_CHUNK_FEATURES)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--early-stop", type=int, default=EARLY_STOP)
    parser.add_argument("--reduce-lr", type=int, default=REDUCE_LR)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--no-batch-norm", dest="use_batch_norm", action="store_false")
    parser.add_argument("--use-layer-norm", action="store_true", default=USE_LAYER_NORM)
    parser.set_defaults(use_batch_norm=USE_BATCH_NORM)
    return parser


def configure_from_args(args):
    global te_level, BASE_DIR, DATA_DIR, OUTPUT_DIR
    global UNIQUE_NPZ, MULTI_NPZ, MERGE_NPZ, BARCODE_FILE, FEATURE_FILE
    global N_HIDDEN, N_LATENT, N_LAYERS, DROPOUT_RATE, GENE_LIKELIHOOD
    global USE_BATCH_NORM, USE_LAYER_NORM, KL_WEIGHT, LEARNING_RATE
    global BATCH_SIZE, PREDICT_BATCH_SIZE, OUTPUT_CHUNK_FEATURES
    global EPOCHS, EARLY_STOP, REDUCE_LR, RANDOM_SEED

    te_level = args.te_level
    BASE_DIR = args.base_dir
    DATA_DIR = args.data_dir
    OUTPUT_DIR = args.output_dir
    UNIQUE_NPZ = args.unique_npz
    MULTI_NPZ = args.multi_npz
    MERGE_NPZ = args.merge_npz
    BARCODE_FILE = args.barcodes
    FEATURE_FILE = args.features
    N_HIDDEN = args.n_hidden
    N_LATENT = args.n_latent
    N_LAYERS = args.n_layers
    DROPOUT_RATE = args.dropout_rate
    GENE_LIKELIHOOD = args.gene_likelihood
    USE_BATCH_NORM = args.use_batch_norm
    USE_LAYER_NORM = args.use_layer_norm
    KL_WEIGHT = args.kl_weight
    LEARNING_RATE = args.learning_rate
    BATCH_SIZE = args.batch_size
    PREDICT_BATCH_SIZE = args.predict_batch_size
    OUTPUT_CHUNK_FEATURES = args.output_chunk_features
    EPOCHS = args.epochs
    EARLY_STOP = args.early_stop
    REDUCE_LR = args.reduce_lr
    RANDOM_SEED = args.random_seed


def run_config_dict():
    return {
        "te_level": te_level,
        "base_dir": BASE_DIR,
        "data_dir": DATA_DIR,
        "output_dir": OUTPUT_DIR,
        "unique_npz": UNIQUE_NPZ,
        "multi_npz": MULTI_NPZ,
        "merge_npz": MERGE_NPZ,
        "barcode_file": BARCODE_FILE,
        "feature_file": FEATURE_FILE,
        "n_hidden": N_HIDDEN,
        "n_latent": N_LATENT,
        "n_layers": N_LAYERS,
        "dropout_rate": DROPOUT_RATE,
        "gene_likelihood": GENE_LIKELIHOOD,
        "use_batch_norm": USE_BATCH_NORM,
        "use_layer_norm": USE_LAYER_NORM,
        "kl_weight": KL_WEIGHT,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "predict_batch_size": PREDICT_BATCH_SIZE,
        "output_chunk_features": OUTPUT_CHUNK_FEATURES,
        "epochs": EPOCHS,
        "early_stop": EARLY_STOP,
        "reduce_lr": REDUCE_LR,
        "random_seed": RANDOM_SEED,
    }


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Helper Functions ---
def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.full_like(x, float("inf")), x)

def read_data(data_file, barcode_file, feature_file):
    if isinstance(data_file, (str, bytes, os.PathLike)):
        data = sparse.load_npz(data_file)
    elif sparse.issparse(data_file):
        data = data_file
    else:
        raise TypeError(f"Unsupported type for data_file: {type(data_file)}")
    
    with open(barcode_file) as f:
        barcodes = [line.strip() for line in f]
    with open(feature_file) as f:
        features = [line.strip() for line in f]

    adata = sc.AnnData(X = data)
    adata.obs_names = barcodes
    adata.var_names = features
    return adata

def normalize_counts(adata):
    raw_counts = np.array(adata.X.sum(axis=1)).flatten()
    median_counts = np.median(raw_counts)
    if median_counts <= 0:
        raise ValueError("Median library size is zero.")
    adata.obs["size_factors"] = raw_counts / median_counts
    adata.raw = adata.copy()
    return adata

def write_text_matrix(matrix, filename, rownames=None, colnames=None, transpose=False):
    if transpose:
        matrix = matrix.T
        rownames, colnames = colnames, rownames
    pd.DataFrame(matrix, index=rownames, columns=colnames).to_csv(
        filename, sep="\t", 
        index=(rownames is not None), 
        header=(colnames is not None), 
        float_format="%.6f"
    )


# ============================================
# scVI-style Distributions (参考scvi-tools)
# ============================================
class NegativeBinomial:
    """
    Negative Binomial distribution
    
    参考：scvi-tools/scvi/distributions/_negative_binomial.py
    
    Args:
        mu: mean (batch, genes)
        theta: inverse dispersion (batch, genes)
    """
    def __init__(self, mu, theta, eps=1e-8):
        self.mu = mu
        self.theta = theta
        self.eps = eps
    
    def log_prob(self, x):
        """
        NB log probability
        
        log p(x | μ, θ) = lgamma(x+θ) - lgamma(θ) - lgamma(x+1)
                          + θ·log(θ/(θ+μ)) + x·log(μ/(θ+μ))
        """
        mu = self.mu
        theta = self.theta
        eps = self.eps
        
        log_theta_mu_eps = torch.log(theta + mu + eps)
        
        res = (
            torch.lgamma(x + theta + eps)
            - torch.lgamma(theta + eps)
            - torch.lgamma(x + 1.0)
            + theta * (torch.log(theta + eps) - log_theta_mu_eps)
            + x * (torch.log(mu + eps) - log_theta_mu_eps)
        )
        
        return res


class ZeroInflatedNegativeBinomial:
    """
    Zero-Inflated Negative Binomial
    
    参考：scvi-tools/scvi/distributions/_negative_binomial.py
    
    Args:
        mu: mean (batch, genes)
        theta: inverse dispersion (batch, genes)
        zi_logits: logits for zero-inflation (batch, genes)
    """
    def __init__(self, mu, theta, zi_logits, eps=1e-8):
        self.mu = mu
        self.theta = theta
        self.zi_logits = zi_logits
        self.eps = eps
    
    def log_prob(self, x):
        """
        ZINB log probability
        
        P(x=0) = π + (1-π)·NB(0|μ,θ)
        P(x>0) = (1-π)·NB(x|μ,θ)
        """
        # Get dropout (zero-inflation) probability
        zi_probs = torch.sigmoid(self.zi_logits)
        
        # NB component
        nb = NegativeBinomial(mu=self.mu, theta=self.theta, eps=self.eps)
        nb_log_prob = nb.log_prob(x)
        
        # ZINB
        case_zero = torch.log(
            zi_probs + (1.0 - zi_probs + self.eps) * torch.exp(nb_log_prob) + self.eps
        )
        case_non_zero = torch.log(1.0 - zi_probs + self.eps) + nb_log_prob
        
        mask = (x < 1e-8).float()
        res = mask * case_zero + (1.0 - mask) * case_non_zero
        
        return res


# ============================================
# scVI-style Encoder (参考scvi-tools)
# ============================================
class Encoder(nn.Module):
    """
    Encoder q(z|x)
    
    参考：scvi-tools/scvi/nn/_base_components.py - Encoder class
    
    Architecture:
        x → [Linear → BN → Activation → Dropout] × n_layers → μ, log(σ²)
    """
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden: int = 128,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        activation: str = "relu",
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        
        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            activation_fn=activation,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )
        
        # Output layers
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: input (batch, n_input)
        
        Returns:
            q_m: posterior mean (batch, n_output)
            q_v: posterior variance (batch, n_output)
        """
        # Pass through encoder layers
        q = self.encoder(x)
        
        # Get mean
        q_m = self.mean_encoder(q)
        
        # Get variance (ensure positive)
        q_v = torch.exp(self.var_encoder(q)) + 1e-4
        
        return q_m, q_v


# ============================================
# scVI-style Decoder (参考scvi-tools)
# ============================================
class DecoderSCVI(nn.Module):
    """
    Decoder p(x|z) with NB or ZINB output
    
    参考：scvi-tools/scvi/module/_vae.py - Decoder
    
    Architecture:
        z → [Linear → BN → Activation → Dropout] × n_layers
          → px_scale (normalized mean)
          → px_r (dispersion)
          → px_dropout (zero-inflation, optional)
    """
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden: int = 128,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        gene_likelihood: str = "zinb",  # "zinb" or "nb"
    ):
        super().__init__()
        
        self.gene_likelihood = gene_likelihood
        self.n_output = n_output
        
        # Decoder layers
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )
        
        # Mean (normalized expression profile)
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output),
            nn.Softmax(dim=-1)
        )
        
        # Dispersion (gene-specific, shared across cells)
        # 参考scVI: dispersion是可学习参数
        self.px_r = nn.Parameter(torch.randn(n_output))
        
        # Dropout (zero-inflation, only for ZINB)
        if gene_likelihood == "zinb":
            self.px_dropout_decoder = nn.Linear(n_hidden, n_output)
    
    def forward(self, z, library):
        """
        Forward pass
        
        Args:
            z: latent (batch, n_input)
            library: library size (batch, 1)
        
        Returns:
            px_dict: dictionary with px_scale, px_r, px_rate, px_dropout
        """
        # Decode
        px = self.px_decoder(z)
        
        # Mean (scale by library)
        px_scale = self.px_scale_decoder(px)
        px_rate = library * px_scale
        
        # Dispersion
        px_r = torch.exp(self.px_r)
        
        # Dropout (only for ZINB)
        if self.gene_likelihood == "zinb":
            px_dropout = self.px_dropout_decoder(px)
        else:
            px_dropout = None
        
        return dict(
            px_scale=px_scale,
            px_r=px_r,
            px_rate=px_rate,
            px_dropout=px_dropout
        )


# ============================================
# Fully Connected Layers (参考scvi-tools)
# ============================================
class FCLayers(nn.Module):
    """
    Fully Connected Layers
    
    参考：scvi-tools/scvi/nn/_base_components.py - FCLayers
    
    A helper class to build stacked fully-connected layers.
    """
    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        activation_fn: str = "relu",
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        
        layers = []
        
        for i in range(n_layers):
            n_in_layer = n_in if i == 0 else n_hidden
            n_out_layer = n_out if i == n_layers - 1 else n_hidden
            
            # Linear layer
            layers.append(nn.Linear(n_in_layer, n_out_layer, bias=bias))
            
            # Normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(n_out_layer, momentum=0.01, eps=0.001))
            elif use_layer_norm:
                layers.append(nn.LayerNorm(n_out_layer, elementwise_affine=False))
            
            # Activation
            if activation_fn == "relu":
                layers.append(nn.ReLU())
            elif activation_fn == "leaky_relu":
                layers.append(nn.LeakyReLU())
            
            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))
        
        self.fc_layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.fc_layers(x)


# ============================================
# Product-of-Experts (保持不变)
# ============================================
class ProductOfExperts(nn.Module):
    """
    Product-of-Experts for Gaussian distributions
    
    参考MIDAS和您原来的实现
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, mu_list, var_list):
        """
        PoE fusion
        
        Args:
            mu_list: List of means
            var_list: List of variances (NOT log-variance)
        
        Returns:
            mu_poe, var_poe
        """
        # Convert to precision
        precision_list = [1.0 / (var + self.eps) for var in var_list]
        
        # Sum precisions
        precision_poe = sum(precision_list)
        
        # PoE variance
        var_poe = 1.0 / (precision_poe + self.eps)
        
        # PoE mean
        mu_poe = var_poe * sum([mu * prec for mu, prec in zip(mu_list, precision_list)])
        
        return mu_poe, var_poe


# ============================================
# scVI-style VAE with PoE (主模型)
# ============================================
class SCVI_PoE(nn.Module):
    """
    Multi-view VAE with Product-of-Experts
    
    基于scvi-tools的scVI架构，扩展到多视图
    
    Reference:
    - scvi-tools: https://github.com/scverse/scvi-tools
    - scVI paper: Lopez et al., Nature Methods 2018
    """
    def __init__(
        self,
        n_input_u: int,
        n_input_m: int,
        n_input_merge: int,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        gene_likelihood: str = "zinb",  # "zinb" or "nb"
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        
        self.n_latent = n_latent
        self.gene_likelihood = gene_likelihood
        
        # Encoders for each view
        self.z_encoder_u = Encoder(
            n_input=n_input_u,
            n_output=n_latent,
            n_hidden=n_hidden,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )
        
        self.z_encoder_m = Encoder(
            n_input=n_input_m,
            n_output=n_latent,
            n_hidden=n_hidden,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )

        self.z_encoder_merge = Encoder(
            n_input=n_input_merge,
            n_output=n_latent,
            n_hidden=n_hidden,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )
        
        # Product-of-Experts
        self.poe = ProductOfExperts()
        
        # Decoders
        self.decoder_u = DecoderSCVI(
            n_input=n_latent,
            n_output=n_input_u,
            n_hidden=n_hidden,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            gene_likelihood=gene_likelihood,
        )
        
        self.decoder_m = DecoderSCVI(
            n_input=n_latent,
            n_output=n_input_m,
            n_hidden=n_hidden,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            gene_likelihood=gene_likelihood,
        )

        self.decoder_merge = DecoderSCVI(
            n_input=n_latent,
            n_output=n_input_merge,
            n_hidden=n_hidden,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            gene_likelihood=gene_likelihood,
        )
    
    def _get_inference_input(self, x):
        """
        Prepare input for encoder (log-transform)
        
        参考scVI: input是log(1+x)
        """
        x_ = torch.log(1 + x)
        return x_
    
    def inference(self, x_u, x_m, x_merge, n_samples=1):
        """
        Inference: q(z|x_u,x_m) via PoE
        
        Args:
            x_u, x_m, x_merge: raw counts
            n_samples: number of samples from posterior
        
        Returns:
            z: sampled latent
            qz_m: PoE mean
            qz_v: PoE variance
            qu_m, qu_v: U view posterior
            qm_m, qm_v: M view posterior
            qmerge_m, qmerge_v: merge view posterior
        """
        # Log-transform inputs
        x_u_ = self._get_inference_input(x_u)
        x_m_ = self._get_inference_input(x_m)
        x_merge_ = self._get_inference_input(x_merge)
        
        # Encode each view
        qu_m, qu_v = self.z_encoder_u(x_u_)
        qm_m, qm_v = self.z_encoder_m(x_m_)
        qmerge_m, qmerge_v = self.z_encoder_merge(x_merge_)
        
        # PoE fusion
        qz_m, qz_v = self.poe(
            [qu_m, qm_m, qmerge_m],
            [qu_v, qm_v, qmerge_v],
        )
        
        # Sample from posterior
        dist = Normal(qz_m, qz_v.sqrt())
        if n_samples > 1:
            z = dist.rsample([n_samples])
        else:
            z = dist.rsample()
        
        return dict(
            z=z,
            qz_m=qz_m,
            qz_v=qz_v,
            qu_m=qu_m,
            qu_v=qu_v,
            qm_m=qm_m,
            qm_v=qm_v,
            qmerge_m=qmerge_m,
            qmerge_v=qmerge_v,
        )
    
    def generative(self, z, library_u, library_m, library_merge):
        """
        Generative: p(x|z)
        
        Args:
            z: latent
            library_u, library_m, library_merge: library sizes
        
        Returns:
            px_u, px_m, px_merge: dictionaries with NB/ZINB parameters
        """
        px_u = self.decoder_u(z, library_u)
        px_m = self.decoder_m(z, library_m)
        px_merge = self.decoder_merge(z, library_merge)
        
        return px_u, px_m, px_merge
    
    def forward(self, x_u, x_m, x_merge, library_u, library_m, library_merge):
        """
        Full forward pass
        
        Returns:
            inference_outputs: dict from inference
            generative_outputs: tuple (px_u, px_m)
        """
        # Inference
        inference_outputs = self.inference(x_u, x_m, x_merge)
        z = inference_outputs["z"]
        
        # Generative
        px_u, px_m, px_merge = self.generative(z, library_u, library_m, library_merge)
        
        return inference_outputs, (px_u, px_m, px_merge)
    
    def loss(
        self,
        x_u,
        x_m,
        x_merge,
        library_u,
        library_m,
        library_merge,
        inference_outputs,
        generative_outputs,
        kl_weight=1.0,
    ):
        """
        Compute ELBO loss
        
        参考scVI的loss计算
        
        ELBO = E[log p(x|z)] - KL(q(z|x) || p(z))
        """
        # Unpack outputs
        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        
        px_u, px_m, px_merge = generative_outputs
        
        # Reconstruction loss
        reconst_loss_u = self._reconstruction_loss(x_u, px_u)
        reconst_loss_m = self._reconstruction_loss(x_m, px_m)
        reconst_loss_merge = self._reconstruction_loss(x_merge, px_merge)
        reconst_loss = reconst_loss_u + reconst_loss_m + reconst_loss_merge
        
        # KL divergence: KL(q(z|x) || N(0,I))
        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)
        
        kl_divergence_z = kl(
            Normal(qz_m, qz_v.sqrt()),
            Normal(mean, scale)
        ).sum(dim=1)
        
        # Total loss (negative ELBO)
        loss = torch.mean(reconst_loss + kl_weight * kl_divergence_z)
        
        return dict(
            loss=loss,
            reconst_loss=torch.mean(reconst_loss),
            kl_local=torch.mean(kl_divergence_z),
        )
    
    def _reconstruction_loss(self, x, px):
        """
        Compute reconstruction loss (NB or ZINB)
        
        参考scVI实现
        """
        if self.gene_likelihood == "zinb":
            reconst_loss = -ZeroInflatedNegativeBinomial(
                mu=px["px_rate"],
                theta=px["px_r"],
                zi_logits=px["px_dropout"],
            ).log_prob(x).sum(dim=-1)
        
        elif self.gene_likelihood == "nb":
            reconst_loss = -NegativeBinomial(
                mu=px["px_rate"],
                theta=px["px_r"],
            ).log_prob(x).sum(dim=-1)
        
        return reconst_loss


# ============================================
# Trainer (参考scvi-tools的训练流程)
# ============================================
class SCVIPoETrainer:
    """
    Trainer for scVI-style VAE with PoE
    
    参考scvi-tools的训练流程
    """
    def __init__(self, model, device, kl_weight=1.0):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.kl_weight = kl_weight
    
    def train(
        self,
        loader_train,
        loader_val,
        optimizer,
        epochs=300,
        early_stop=15,
        reduce_lr=10,
        save_path=None,
    ):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=reduce_lr, verbose=False
        )
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_recon': [],
            'train_kl': []
        }
        best_model_path = os.path.join(save_path, "scvi_poe_weights.pt") if save_path else None
        
        for epoch in range(1, epochs + 1):
            # Training
            self.model.train()
            train_loss = 0.0
            train_recon = 0.0
            train_kl = 0.0
            
            for yu, ym, ymerge, sfu, sfm, sfmerge in loader_train:
                sfu = sfu.to(self.device).unsqueeze(1)
                sfm = sfm.to(self.device).unsqueeze(1)
                sfmerge = sfmerge.to(self.device).unsqueeze(1)
                yu = yu.to(self.device)
                ym = ym.to(self.device)
                ymerge = ymerge.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward
                inference_outputs, generative_outputs = self.model(
                    yu, ym, ymerge, sfu, sfm, sfmerge  # Use raw counts
                )
                
                # Loss
                losses = self.model.loss(
                    yu, ym, ymerge, sfu, sfm, sfmerge,
                    inference_outputs,
                    generative_outputs,
                    kl_weight=self.kl_weight
                )
                
                loss = losses["loss"]
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                
                train_loss += loss.item() * yu.size(0)
                train_recon += losses["reconst_loss"].item() * yu.size(0)
                train_kl += losses["kl_local"].item() * yu.size(0)
            
            train_loss /= len(loader_train.dataset)
            train_recon /= len(loader_train.dataset)
            train_kl /= len(loader_train.dataset)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for yu, ym, ymerge, sfu, sfm, sfmerge in loader_val:
                    sfu = sfu.to(self.device).unsqueeze(1)
                    sfm = sfm.to(self.device).unsqueeze(1)
                    sfmerge = sfmerge.to(self.device).unsqueeze(1)
                    yu = yu.to(self.device)
                    ym = ym.to(self.device)
                    ymerge = ymerge.to(self.device)
                    
                    inference_outputs, generative_outputs = self.model(
                        yu, ym, ymerge, sfu, sfm, sfmerge
                    )
                    
                    losses = self.model.loss(
                        yu, ym, ymerge, sfu, sfm, sfmerge,
                        inference_outputs,
                        generative_outputs,
                        kl_weight=self.kl_weight
                    )
                    
                    val_loss += losses["loss"].item() * yu.size(0)
            
            val_loss /= len(loader_val.dataset)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_recon'].append(train_recon)
            history['train_kl'].append(train_kl)
            
            print(f"Epoch {epoch}/{epochs} | "
                  f"Train: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f}) | "
                  f"Val: {val_loss:.4f}")
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                if best_model_path:
                    torch.save(self.model.state_dict(), best_model_path)
                    print(f"  ✓ Saved best model")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stop:
                    print("Early stopping triggered.")
                    break

        if best_model_path and os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
            print(f"Loaded best model from {best_model_path} (best Val Loss: {best_val_loss:.4f})")
        
        return history
    
    def _write_memmap_feature_tsv(self, matrix, filename, rownames, colnames, chunk_features=256):
        with open(filename, "w") as f:
            f.write("\t" + "\t".join(map(str, colnames)) + "\n")
            for start in range(0, matrix.shape[1], chunk_features):
                end = min(start + chunk_features, matrix.shape[1])
                block = np.asarray(matrix[:, start:end]).T
                df = pd.DataFrame(block, index=rownames[start:end], columns=colnames)
                df.to_csv(f, sep="\t", header=False, float_format="%.6f")

    def predict(self, adata_u, adata_m, adata_merge, save_dir, batch_size=128):
        """
        Generate predictions in batches to avoid moving the full input matrix
        to GPU memory at once.
        """
        self.model.eval()

        y_u = adata_u.raw.X.tocsr() if sparse.issparse(adata_u.raw.X) else np.asarray(adata_u.raw.X)
        y_m = adata_m.raw.X.tocsr() if sparse.issparse(adata_m.raw.X) else np.asarray(adata_m.raw.X)
        y_merge = adata_merge.raw.X.tocsr() if sparse.issparse(adata_merge.raw.X) else np.asarray(adata_merge.raw.X)

        sf_u_all = adata_u.obs["size_factors"].values.astype(np.float32)
        sf_m_all = adata_m.obs["size_factors"].values.astype(np.float32)
        sf_merge_all = adata_merge.obs["size_factors"].values.astype(np.float32)

        n_cells = adata_u.n_obs
        n_features = adata_u.n_vars
        rownames = adata_u.obs_names.values.astype(str)
        feature_names = adata_u.var_names.values.astype(str)

        mean_u_mm = np.memmap(
            os.path.join(save_dir, "mean_u.tmp.memmap"),
            dtype="float32",
            mode="w+",
            shape=(n_cells, n_features),
        )
        mean_m_mm = np.memmap(
            os.path.join(save_dir, "mean_m.tmp.memmap"),
            dtype="float32",
            mode="w+",
            shape=(n_cells, n_features),
        )
        mean_merge_mm = np.memmap(
            os.path.join(save_dir, "mean_merge.tmp.memmap"),
            dtype="float32",
            mode="w+",
            shape=(n_cells, n_features),
        )
        latent_mu = np.zeros((n_cells, self.model.n_latent), dtype=np.float32)
        latent_std = np.zeros((n_cells, self.model.n_latent), dtype=np.float32)

        with torch.no_grad():
            for start in range(0, n_cells, batch_size):
                end = min(start + batch_size, n_cells)
                yu_np = y_u[start:end].toarray() if sparse.issparse(y_u) else y_u[start:end]
                ym_np = y_m[start:end].toarray() if sparse.issparse(y_m) else y_m[start:end]
                ymerge_np = y_merge[start:end].toarray() if sparse.issparse(y_merge) else y_merge[start:end]

                yu = torch.from_numpy(np.asarray(yu_np, dtype=np.float32)).to(self.device)
                ym = torch.from_numpy(np.asarray(ym_np, dtype=np.float32)).to(self.device)
                ymerge = torch.from_numpy(np.asarray(ymerge_np, dtype=np.float32)).to(self.device)
                sfu = torch.from_numpy(sf_u_all[start:end]).float().to(self.device).unsqueeze(1)
                sfm = torch.from_numpy(sf_m_all[start:end]).float().to(self.device).unsqueeze(1)
                sfmerge = torch.from_numpy(sf_merge_all[start:end]).float().to(self.device).unsqueeze(1)

                inference_outputs, (px_u, px_m, px_merge) = self.model(
                    yu, ym, ymerge, sfu, sfm, sfmerge
                )

                mean_u_mm[start:end, :] = px_u["px_rate"].cpu().numpy().astype(np.float32)
                mean_m_mm[start:end, :] = px_m["px_rate"].cpu().numpy().astype(np.float32)
                mean_merge_mm[start:end, :] = px_merge["px_rate"].cpu().numpy().astype(np.float32)
                latent_mu[start:end, :] = inference_outputs["qz_m"].cpu().numpy().astype(np.float32)
                latent_std[start:end, :] = inference_outputs["qz_v"].sqrt().cpu().numpy().astype(np.float32)

                if (start // batch_size + 1) % 20 == 0 or end == n_cells:
                    print(f"   Predicted {end}/{n_cells} cells")

        mean_u_mm.flush()
        mean_m_mm.flush()
        mean_merge_mm.flush()

        self._write_memmap_feature_tsv(
            mean_u_mm,
            os.path.join(save_dir, "mean_u.tsv"),
            feature_names,
            rownames,
            chunk_features=OUTPUT_CHUNK_FEATURES,
        )
        self._write_memmap_feature_tsv(
            mean_m_mm,
            os.path.join(save_dir, "mean_m.tsv"),
            feature_names,
            rownames,
            chunk_features=OUTPUT_CHUNK_FEATURES,
        )
        self._write_memmap_feature_tsv(
            mean_merge_mm,
            os.path.join(save_dir, "mean_merge.tsv"),
            feature_names,
            rownames,
            chunk_features=OUTPUT_CHUNK_FEATURES,
        )

        write_text_matrix(
            latent_mu,
            os.path.join(save_dir, "latent_mu.tsv"),
            rownames=rownames,
            transpose=False,
        )
        write_text_matrix(
            latent_std,
            os.path.join(save_dir, "latent_std.tsv"),
            rownames=rownames,
            transpose=False,
        )

        for tmp_name in ["mean_u.tmp.memmap", "mean_m.tmp.memmap", "mean_merge.tmp.memmap"]:
            tmp_path = os.path.join(save_dir, tmp_name)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        print(f"Results saved to {save_dir}")


# ============================================
# Main Execution
# ============================================
if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    configure_from_args(args)

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    os.chdir(BASE_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "run_config.json"), "w") as f:
        json.dump(run_config_dict(), f, indent=2, sort_keys=True)
    
    # Load data
    print("Loading Unique View...")
    unique_matrix = sparse.load_npz(os.path.join(DATA_DIR, UNIQUE_NPZ))
    adata_u = read_data(
        unique_matrix,
        os.path.join(DATA_DIR, BARCODE_FILE),
        os.path.join(DATA_DIR, FEATURE_FILE)
    )
    adata_u = normalize_counts(adata_u)
    
    print("Loading Multi View...")
    multi_matrix = sparse.load_npz(os.path.join(DATA_DIR, MULTI_NPZ))
    adata_m = read_data(
        multi_matrix,
        os.path.join(DATA_DIR, BARCODE_FILE),
        os.path.join(DATA_DIR, FEATURE_FILE)
    )
    adata_m = normalize_counts(adata_m)

    print("Loading Merge View...")
    merge_path = os.path.join(DATA_DIR, MERGE_NPZ)
    if os.path.exists(merge_path):
        merge_matrix = sparse.load_npz(merge_path)
        print(f"Loaded saved merge matrix: {merge_path}")
    else:
        merge_matrix = unique_matrix + multi_matrix
        print(f"Merge matrix not found at {merge_path}; using unique + multi in memory.")
    adata_merge = read_data(
        merge_matrix,
        os.path.join(DATA_DIR, BARCODE_FILE),
        os.path.join(DATA_DIR, FEATURE_FILE)
    )
    adata_merge = normalize_counts(adata_merge)
    
    common_cells = adata_u.obs_names.intersection(adata_m.obs_names).intersection(adata_merge.obs_names)
    adata_u = adata_u[common_cells].copy()
    adata_m = adata_m[common_cells].copy()
    adata_merge = adata_merge[common_cells].copy()
    print(f"Aligned {len(common_cells)} cells across views.")
    
    # Prepare tensors
    def prepare_tensor(adata):
        Y = adata.raw.X.toarray() if sparse.issparse(adata.raw.X) else adata.raw.X
        SF = adata.obs['size_factors'].values
        return torch.tensor(Y).float(), torch.tensor(SF).float()
    
    Yu, SFu = prepare_tensor(adata_u)
    Ym, SFm = prepare_tensor(adata_m)
    Ymerge, SFmerge = prepare_tensor(adata_merge)
    
    dataset = TensorDataset(Yu, Ym, Ymerge, SFu, SFm, SFmerge)
    
    n_val = int(len(dataset) * 0.1)
    split_generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_set, val_set = random_split(dataset, [len(dataset) - n_val, n_val], generator=split_generator)
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    
    # ============================================
    # Initialize scVI-style model
    # ============================================
    print("\n" + "="*70)
    print("Initializing scVI-style VAE with PoE")
    print("="*70)
    
    model = SCVI_PoE(
        n_input_u=adata_u.n_vars,
        n_input_m=adata_m.n_vars,
        n_input_merge=adata_merge.n_vars,
        n_hidden=N_HIDDEN,
        n_latent=N_LATENT,
        n_layers=N_LAYERS,
        dropout_rate=DROPOUT_RATE,
        gene_likelihood=GENE_LIKELIHOOD,
        use_batch_norm=USE_BATCH_NORM,
        use_layer_norm=USE_LAYER_NORM,
    )
    
    print(model)
    print(f"\nArchitecture (scVI-style):")
    print(f"  Encoder: log(1+x) → FC layers → (μ, σ²)")
    print(f"  PoE: Precision-weighted fusion")
    print(f"  Decoder: z → FC layers → (px_scale, px_r, px_dropout)")
    print(f"  Likelihood: {model.gene_likelihood.upper()}")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {trainable_params:,}")
    
    # ============================================
    # Train
    # ============================================
    print("\n" + "="*70)
    print("Training")
    print("="*70)
    
    trainer = SCVIPoETrainer(model, device, kl_weight=KL_WEIGHT)
    
    print(f"KL weight: {KL_WEIGHT}")
    print(f"  (Small KL weight for sparse TE data)")
    
    history = trainer.train(
        train_loader,
        val_loader,
        optimizer=optim.Adam(model.parameters(), lr=LEARNING_RATE),
        epochs=EPOCHS,
        early_stop=EARLY_STOP,
        reduce_lr=REDUCE_LR,
        save_path=OUTPUT_DIR
    )
    
    # Save history
    import json
    with open(os.path.join(OUTPUT_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": run_config_dict(),
            "n_input_u": adata_u.n_vars,
            "n_input_m": adata_m.n_vars,
            "n_input_merge": adata_merge.n_vars,
            "barcodes": adata_u.obs_names.astype(str).tolist(),
            "features": adata_u.var_names.astype(str).tolist(),
        },
        os.path.join(OUTPUT_DIR, "scvi_poe_model_checkpoint.pt"),
    )
    
    # ============================================
    # Predict
    # ============================================
    print("\n" + "="*70)
    print("Generating Predictions")
    print("="*70)
    
    trainer.predict(adata_u, adata_m, adata_merge, OUTPUT_DIR, batch_size=PREDICT_BATCH_SIZE)
    
    print("\n" + "="*70)
    print("✅ Training Complete!")
    print("="*70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"\nGenerated files:")
    print(f"  - mean_u.tsv, mean_m.tsv, mean_merge.tsv      (NB/ZINB mean)")
    print(f"  - latent_mu.tsv               (For clustering)")
    print(f"  - latent_std.tsv              (Uncertainty)")
    print(f"  - scvi_poe_weights.pt         (Model weights)")
    print(f"  - scvi_poe_model_checkpoint.pt (Model weights + config)")
    print(f"  - run_config.json             (Run configuration)")
    print(f"  - training_history.json       (Training log)")
    
    print(f"\n📚 References:")
    print(f"  - scVI: Lopez et al., Nature Methods 2018")
    print(f"  - scvi-tools: https://github.com/scverse/scvi-tools")
    
    print(f"\n🎯 Key features:")
    print(f"  - Standard scVI architecture")
    print(f"  - Product-of-Experts fusion")
    print(f"  - Inputs: unique, multi, and merge views")
    print(f"  - {model.gene_likelihood.upper()} likelihood")
    print(f"  - Small KL weight ({KL_WEIGHT})")
