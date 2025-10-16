# Corrected Implementation of "Constrained Diffusion Models for Synthesizing Representative Power Flow Datasets"
# Based on Hoseinpour & Dvorkin (2024)

# 1. Installs
!pip install torch torchvision torchaudio --quiet
!pip install pandapower tqdm matplotlib scipy numpy pandas seaborn --quiet
!pip install scikit-learn --quiet

# 2. Imports and Setup
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandapower as pp
import pandapower.networks as pn
import math
import os
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import euclidean_distances

# 3. Configuration
CONFIG = {
    "system": "case30",
    "num_samples": 5000,  # Increased for better training
    "num_synthetic_samples": 500,  # Number of samples to generate for comparison
    "diffusion_steps": 1000,  # Matched to paper
    "beta_start": 1e-4,
    "beta_end": 0.02,
    "hidden_dim": 256,
    "depth": 5,
    "batch_size": 64,
    "epochs": 500,  # Increased for better convergence
    "lr": 1e-4,
    "lambda_guidance": 0.01,  # Adjusted to paper's range (10^-4 to 10^-2)
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
}

# 4. Model and Utility Definitions
class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for time steps"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)

class DenoiserMLP(nn.Module):
    """Denoiser neural network ε_θ as described in the paper"""
    def __init__(self, dim_in, hidden=256, depth=3, time_embed_dim=128):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)

        layers = []
        for i in range(depth):
            in_dim = dim_in + time_embed_dim if i == 0 else hidden
            out_dim = hidden if i < depth - 1 else dim_in
            layers.append(nn.Linear(in_dim, out_dim))
            if i < depth - 1:
                layers.append(nn.SiLU())  # Using SiLU (Swish) activation
                layers.append(nn.LayerNorm(out_dim))

        self.net = nn.Sequential(*layers)

        # Initialize weights
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x, t):
        t_emb = self.time_embed(t.float())
        h = torch.cat([x, t_emb], dim=-1)
        return self.net(h)

class MinMaxNorm:
    """Min-max normalization to [-1, 1] range as per Section V-B of the paper"""
    def __init__(self, xmin, xmax, eps=1e-8):
        self.xmin = xmin.clone().detach() if isinstance(xmin, torch.Tensor) else torch.tensor(xmin, dtype=torch.float32)
        self.xmax = xmax.clone().detach() if isinstance(xmax, torch.Tensor) else torch.tensor(xmax, dtype=torch.float32)
        self.eps = eps
        self.range = torch.clamp(self.xmax - self.xmin, min=eps)

    def norm(self, x):
        """Normalize to [-1, 1] using equation (30)"""
        return 2 * (x - self.xmin.to(x.device)) / self.range.to(x.device) - 1

    def denorm(self, x):
        """Denormalize using equation (31)"""
        return ((x + 1) / 2) * self.range.to(x.device) + self.xmin.to(x.device)

    def scale_gradient(self, grad):
        """Scale gradient according to equation (32)"""
        return grad * 2 / self.range.to(grad.device)

def make_beta_schedule(T, beta_start=1e-4, beta_end=0.02, schedule='linear'):
    """Create noise schedule as described in Section III-A"""
    if schedule == 'linear':
        betas = torch.linspace(beta_start, beta_end, T)
    elif schedule == 'cosine':
        # Alternative cosine schedule
        s = 0.008
        steps = T + 1
        x = torch.linspace(0, T, steps)
        alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, beta_start, beta_end)

    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bar

def power_flow_residual(x_phys, G, B):
    """
    Compute power flow residual based on Equations (8a) and (8b) from the paper.
    This is the correct formulation using power balance equations.
    """
    Bdim = G.shape[0]
    p, q = x_phys[:, :Bdim], x_phys[:, Bdim:2*Bdim]
    v, theta = x_phys[:, 2*Bdim:3*Bdim], x_phys[:, 3*Bdim:4*Bdim]

    batch_size = x_phys.shape[0]
    device = x_phys.device

    # Ensure G and B are tensors on the correct device
    if not isinstance(G, torch.Tensor):
        G = torch.tensor(G, dtype=torch.float32, device=device)
    if not isinstance(B, torch.Tensor):
        B = torch.tensor(B, dtype=torch.float32, device=device)

    # Expand dimensions for broadcasting
    v_i = v.unsqueeze(2)  # [batch, Bdim, 1]
    v_j = v.unsqueeze(1)  # [batch, 1, Bdim]
    theta_i = theta.unsqueeze(2)  # [batch, Bdim, 1]
    theta_j = theta.unsqueeze(1)  # [batch, 1, Bdim]

    # Calculate angle differences
    theta_diff = theta_i - theta_j  # [batch, Bdim, Bdim]

    # Power flow equations from the paper
    # P_i = Σ_j v_i * v_j * (G_ij * cos(θ_i - θ_j) + B_ij * sin(θ_i - θ_j))
    P_calc = torch.sum(v_i * v_j * (G * torch.cos(theta_diff) + B * torch.sin(theta_diff)), dim=2)

    # Q_i = Σ_j v_i * v_j * (G_ij * sin(θ_i - θ_j) - B_ij * cos(θ_i - θ_j))
    Q_calc = torch.sum(v_i * v_j * (G * torch.sin(theta_diff) - B * torch.cos(theta_diff)), dim=2)

    # Residuals (power balance mismatch)
    p_residual = p - P_calc
    q_residual = q - Q_calc

    return torch.cat([p_residual, q_residual], dim=-1)

def compute_inequality_violations(x_phys, net):
    """Compute violations of inequality constraints (10a-10d)"""
    Bdim = len(net.bus)
    p, q = x_phys[:, :Bdim], x_phys[:, Bdim:2*Bdim]
    v, theta = x_phys[:, 2*Bdim:3*Bdim], x_phys[:, 3*Bdim:4*Bdim]

    violations = []

    # Voltage limits (10c)
    v_min = torch.tensor(net.bus.min_vm_pu.values, device=x_phys.device)
    v_max = torch.tensor(net.bus.max_vm_pu.values, device=x_phys.device)
    violations.append(torch.relu(v_min - v))  # v >= v_min
    violations.append(torch.relu(v - v_max))  # v <= v_max

    return torch.cat(violations, dim=-1)

def make_dataset(net_fn, N=500, scale_range=(0.8, 1.0), seed=0):
    """Generate ground truth dataset by solving AC-OPF as described in Section VI-A"""
    np.random.seed(seed)
    torch.manual_seed(seed)

    net = net_fn()
    base_loads_p = net.load["p_mw"].copy()
    base_loads_q = net.load["q_mvar"].copy()

    samples = []
    print(f"Generating {N} samples...")

    pbar = tqdm(range(N), desc="Generating Data")
    for _ in pbar:
        net_temp = copy.deepcopy(net)

        # Sample load scaling factors uniformly as in the paper
        scale_p = np.random.uniform(scale_range[0], scale_range[1], size=len(net.load))
        scale_q = np.random.uniform(scale_range[0], scale_range[1], size=len(net.load))

        net_temp.load["p_mw"] = base_loads_p * scale_p
        net_temp.load["q_mvar"] = base_loads_q * scale_q

        try:
            # Run power flow (not OPF for simplicity in this implementation)
            pp.runpp(net_temp, algorithm='nr', init='dc', numba=False, max_iteration=50)

            if net_temp.converged:
                # Extract power injections and voltages
                p = net_temp.res_bus.p_mw.values
                q = net_temp.res_bus.q_mvar.values
                v = net_temp.res_bus.vm_pu.values
                theta = np.deg2rad(net_temp.res_bus.va_degree.values)

                # Stack into single vector
                sample = np.concatenate([p, q, v, theta])
                samples.append(sample)
                pbar.set_postfix({"collected": len(samples)})

        except Exception as e:
            continue

    print(f"Successfully generated {len(samples)} samples.")
    return torch.tensor(np.array(samples), dtype=torch.float32)

def train_model(data, B, config):
    """Train diffusion model with variable decoupling (Algorithm 4)"""
    device = config["device"]
    T = config["diffusion_steps"]

    # Create noise schedule
    betas, alphas, alpha_bar = make_beta_schedule(T, config["beta_start"], config["beta_end"])
    betas = betas.to(device)
    alphas = alphas.to(device)
    alpha_bar = alpha_bar.to(device)

    # Initialize denoiser networks for variable decoupling
    den1 = DenoiserMLP(2*B, config["hidden_dim"], config["depth"]).to(device)
    den2 = DenoiserMLP(2*B, config["hidden_dim"], config["depth"]).to(device)

    # Normalization
    data_min = data.min(0).values
    data_max = data.max(0).values

    # Add small noise to constant columns
    data_range = data_max - data_min
    constant_mask = data_range < 1e-6
    if constant_mask.any():
        print(f"Warning: Found {constant_mask.sum().item()} constant columns. Adding small noise.")
        data_min[constant_mask] -= 1e-4
        data_max[constant_mask] += 1e-4

    normer = MinMaxNorm(data_min, data_max)
    data_norm = normer.norm(data)

    # Check for NaN/Inf
    if torch.isnan(data_norm).any() or torch.isinf(data_norm).any():
        print("ERROR: NaN or Inf detected after normalization!")
        return None, None, None, None

    # Create dataloader
    loader = DataLoader(TensorDataset(data_norm), batch_size=config["batch_size"], shuffle=True)

    # Optimizer
    opt = torch.optim.AdamW(
        list(den1.parameters()) + list(den2.parameters()),
        lr=config["lr"],
        weight_decay=1e-5
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config["epochs"])

    loss_history = []

    print("Starting training...")
    for ep in range(config["epochs"]):
        total_loss = 0
        num_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {ep+1}/{config['epochs']}")

        for (xb,) in pbar:
            xb = xb.to(device)

            # Variable decoupling: split into (p,θ) and (q,v)
            x1 = torch.cat([xb[:, :B], xb[:, 3*B:4*B]], dim=-1)  # (p, θ)
            x2 = torch.cat([xb[:, B:2*B], xb[:, 2*B:3*B]], dim=-1)  # (q, v)

            # Sample random timesteps
            t = torch.randint(0, T, (xb.size(0),), device=device)

            # Sample noise
            eps1 = torch.randn_like(x1)
            eps2 = torch.randn_like(x2)

            # Forward diffusion (equation 3)
            sqrt_ab_t = torch.sqrt(alpha_bar[t]).unsqueeze(-1)
            sqrt_1_minus_ab_t = torch.sqrt(1 - alpha_bar[t]).unsqueeze(-1)

            x1_t = sqrt_ab_t * x1 + sqrt_1_minus_ab_t * eps1
            x2_t = sqrt_ab_t * x2 + sqrt_1_minus_ab_t * eps2

            # Predict noise
            pred1 = den1(x1_t, t)
            pred2 = den2(x2_t, t)

            # MSE loss (equation 5)
            loss1 = ((eps1 - pred1)**2).mean()
            loss2 = ((eps2 - pred2)**2).mean()
            loss = loss1 + loss2

            # Check for NaN
            if torch.isnan(loss):
                print(f"\nNaN loss detected at epoch {ep+1}!")
                break

            opt.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(
                list(den1.parameters()) + list(den2.parameters()),
                max_norm=1.0
            )

            opt.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        avg_loss = total_loss / num_batches
        loss_history.append(avg_loss)
        print(f"Epoch {ep+1} - Average Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    # Save checkpoint
    checkpoint = {
        "den1": den1.state_dict(),
        "den2": den2.state_dict(),
        "xmin": normer.xmin,
        "xmax": normer.xmax,
        "range": normer.range,
        "B": B
    }
    torch.save(checkpoint, "checkpoint.pth")
    print("Model saved to checkpoint.pth")

    return den1, den2, normer, loss_history

def sample_with_guidance(G, Bmat, config, ckpt_path="checkpoint.pth", track_residuals=False):
    """Sampling with manifold-constrained gradient guidance (Algorithm 5)"""
    device = config["device"]

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    Bdim = ckpt["B"]

    # Initialize denoisers
    den1 = DenoiserMLP(2*Bdim, config["hidden_dim"], config["depth"]).to(device)
    den2 = DenoiserMLP(2*Bdim, config["hidden_dim"], config["depth"]).to(device)
    den1.load_state_dict(ckpt["den1"])
    den2.load_state_dict(ckpt["den2"])
    den1.eval()
    den2.eval()

    # Normalizer
    normer = MinMaxNorm(ckpt["xmin"], ckpt["xmax"])
    if "range" in ckpt:
        normer.range = ckpt["range"]

    # Noise schedule
    T = config["diffusion_steps"]
    betas, alphas, alpha_bar = make_beta_schedule(T, config["beta_start"], config["beta_end"])
    betas = betas.to(device)
    alphas = alphas.to(device)
    alpha_bar = alpha_bar.to(device)

    # Guidance strength
    lam = config["lambda_guidance"]

    # Initialize from pure noise
    x_t = torch.randn(1, 4 * Bdim, device=device)

    # Track residuals if requested
    residual_history = [] if track_residuals else None

    print("Starting guided sampling..." if track_residuals else "Sampling...")
    pbar = tqdm(reversed(range(T)), desc="Sampling", total=T, disable=not track_residuals)

    for t in pbar:
        # Split for variable decoupling
        x1_t = torch.cat([x_t[:, :Bdim], x_t[:, 3*Bdim:4*Bdim]], dim=-1)  # (p, θ)
        x2_t = torch.cat([x_t[:, Bdim:2*Bdim], x_t[:, 2*Bdim:3*Bdim]], dim=-1)  # (q, v)

        with torch.no_grad():
            t_tensor = torch.tensor([t], device=device)

            # Predict noise (denoisers)
            eps1 = den1(x1_t, t_tensor)
            eps2 = den2(x2_t, t_tensor)

        # Estimate clean data using Tweedie's formula (equation 6)
        sqrt_ab_t = torch.sqrt(alpha_bar[t])
        sqrt_1_minus_ab_t = torch.sqrt(1 - alpha_bar[t])

        hat_x0_p1 = (x1_t - sqrt_1_minus_ab_t * eps1) / (sqrt_ab_t + 1e-8)
        hat_x0_p2 = (x2_t - sqrt_1_minus_ab_t * eps2) / (sqrt_ab_t + 1e-8)

        # Reconstruct full vector
        hat_x0_norm = torch.zeros(1, 4*Bdim, device=device)
        hat_x0_norm[:, :Bdim] = hat_x0_p1[:, :Bdim]  # p
        hat_x0_norm[:, Bdim:2*Bdim] = hat_x0_p2[:, :Bdim]  # q
        hat_x0_norm[:, 2*Bdim:3*Bdim] = hat_x0_p2[:, Bdim:2*Bdim]  # v
        hat_x0_norm[:, 3*Bdim:4*Bdim] = hat_x0_p1[:, Bdim:2*Bdim]  # θ

        # Clamp for stability
        hat_x0_norm = torch.clamp(hat_x0_norm, -3, 3)

        # Apply gradient guidance if lambda > 0
        if lam > 0 and t > 0:  # No guidance at t=0
            hat_x0_norm.requires_grad_(True)

            # Denormalize for physics evaluation
            x_phys = normer.denorm(hat_x0_norm)

            # Compute power flow residual
            residual = power_flow_residual(x_phys, G, Bmat)
            residual_loss = (residual**2).sum()

            if track_residuals:
                residual_history.append(torch.sqrt(residual_loss).item())

            if not torch.isnan(residual_loss):
                # Compute gradient with respect to normalized variables
                grad_norm = torch.autograd.grad(residual_loss, hat_x0_norm, create_graph=False)[0]

                # Scale gradient properly
                grad_norm = normer.scale_gradient(grad_norm)

                # Clip gradient for stability
                grad_norm = torch.clamp(grad_norm, -1.0, 1.0)

                # Apply guidance (equation 23)
                guidance_strength = lam * (1 - t/T)  # Decay guidance over time
                hat_x0_guided = hat_x0_norm - guidance_strength * grad_norm
                hat_x0_guided = hat_x0_guided.detach()
            else:
                hat_x0_guided = hat_x0_norm.detach()
        else:
            hat_x0_guided = hat_x0_norm.detach()

        # Split guided estimate for variable decoupling
        hat_x0_guided_p1 = torch.cat([hat_x0_guided[:, :Bdim], hat_x0_guided[:, 3*Bdim:4*Bdim]], dim=-1)
        hat_x0_guided_p2 = torch.cat([hat_x0_guided[:, Bdim:2*Bdim], hat_x0_guided[:, 2*Bdim:3*Bdim]], dim=-1)

        # Reverse diffusion step (equation 7)
        if t > 0:
            z = torch.randn_like(x_t)
            z1 = torch.cat([z[:, :Bdim], z[:, 3*Bdim:4*Bdim]], dim=-1)
            z2 = torch.cat([z[:, Bdim:2*Bdim], z[:, 2*Bdim:3*Bdim]], dim=-1)
        else:
            z1 = z2 = 0

        beta_t = betas[t]
        alpha_t = alphas[t]
        alpha_bar_prev = alpha_bar[t-1] if t > 0 else torch.tensor(1.0, device=device)

        # DDPM sampling coefficients
        coef1 = torch.sqrt(alpha_bar_prev) * beta_t / (1 - alpha_bar[t] + 1e-8)
        coef2 = torch.sqrt(alpha_t) * (1 - alpha_bar_prev) / (1 - alpha_bar[t] + 1e-8)
        sigma_t = torch.sqrt(beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar[t] + 1e-8))  # Fixed by adding closing parenthesis

        # Update x1_t and x2_t
        x1_t = coef1 * hat_x0_guided_p1 + coef2 * x1_t + sigma_t * z1
        x2_t = coef1 * hat_x0_guided_p2 + coef2 * x2_t + sigma_t * z2

        # Reconstruct full x_t
        x_t[:, :Bdim] = x1_t[:, :Bdim]
        x_t[:, Bdim:2*Bdim] = x2_t[:, :Bdim]
        x_t[:, 2*Bdim:3*Bdim] = x2_t[:, Bdim:2*Bdim]
        x_t[:, 3*Bdim:4*Bdim] = x1_t[:, Bdim:2*Bdim]

    # Final denormalization
    final_sample = normer.denorm(x_t).cpu().numpy()

    if track_residuals:
        return final_sample, residual_history
    return final_sample
def compute_wasserstein_distance_multidim(data1, data2, seed=None):
    """ Compute multidimensional 1-Wasserstein distance (W1) between two datasets using L2 cost, as in Equation (33) of the paper.
    Assumes equal sample sizes; subsample if needed. """
    # Check dimensionality compatibility
    if data1.shape[1] != data2.shape[1]:
        raise ValueError("Data dimensions must match")

    # Subsample to min size if unequal, with optional seed for reproducibility
    min_size = min(data1.shape[0], data2.shape[0])
    if seed is not None:
        np.random.seed(seed)
    if data1.shape[0] > min_size:
        data1 = data1[np.random.choice(data1.shape[0], min_size, replace=False)]
    if data2.shape[0] > min_size:
        data2 = data2[np.random.choice(data2.shape[0], min_size, replace=False)]

    # Compute L2 cost matrix: C_ij = ||data1[i] - data2[j]||_2
    cost_matrix = euclidean_distances(data1, data2)

    # Solve optimal assignment (Hungarian algorithm)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # W1 = (1 / N) * sum of assigned costs
    w1_distance = cost_matrix[row_ind, col_ind].sum() / min_size

    return w1_distance

# 5. Main Execution with Verification
if __name__ == "__main__":
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])

    # Load network
    network_function = getattr(pn, CONFIG['system'])
    net = network_function()
    num_buses = len(net.bus)
    print(f"Using {CONFIG['system']} with {num_buses} buses")

    # Get admittance matrix
    pp.runpp(net)
    Ybus = net._ppc['internal']['Ybus']
    G_matrix, B_matrix = Ybus.real.toarray(), Ybus.imag.toarray()

    # Generate ground truth dataset
    dataset = make_dataset(network_function, N=CONFIG['num_samples'], seed=CONFIG['seed'])

    if len(dataset) > 0:
        print(f"\nDataset shape: {dataset.shape}")
        print(f"Dataset stats:")
        print(f"  P range: [{dataset[:, :num_buses].min():.2f}, {dataset[:, :num_buses].max():.2f}] MW")
        print(f"  Q range: [{dataset[:, num_buses:2*num_buses].min():.2f}, {dataset[:, num_buses:2*num_buses].max():.2f}] MVAr")
        print(f"  V range: [{dataset[:, 2*num_buses:3*num_buses].min():.3f}, {dataset[:, 2*num_buses:3*num_buses].max():.3f}] p.u.")
        print(f"  θ range: [{np.rad2deg(dataset[:, 3*num_buses:].min()):.1f}, {np.rad2deg(dataset[:, 3*num_buses:].max()):.1f}] degrees")

        # Check ground truth feasibility
        print("\nChecking ground truth feasibility...")
        gt_tensor = torch.tensor(dataset[:10], dtype=torch.float32)
        gt_residuals = power_flow_residual(gt_tensor, G_matrix, B_matrix)
        gt_mismatch = torch.sqrt((gt_residuals**2).mean(dim=1))
        print(f"Ground truth power mismatch (first 10): {gt_mismatch.mean():.6f} ± {gt_mismatch.std():.6f}")

        # Train the diffusion model
        den1, den2, normer, loss_history = train_model(dataset, num_buses, CONFIG)

        if den1 is not None:
            # Plot training loss
            plt.figure(figsize=(6, 4))
            plt.plot(range(1, len(loss_history) + 1), loss_history, linewidth=2)
            plt.xlabel("Epoch")
            plt.ylabel("Average Loss")
            plt.title("Training Loss over Epochs")
            plt.grid(True, alpha=0.4)
            plt.tight_layout()
            plt.show()

            # Generate synthetic samples
            print(f"\nGenerating {CONFIG['num_synthetic_samples']} synthetic samples...")
            synthetic_samples = []

            # First sample with residual tracking
            first_sample, residual_history = sample_with_guidance(
                G_matrix, B_matrix, CONFIG, track_residuals=True
            )
            synthetic_samples.append(first_sample[0])

            # Generate remaining samples
            for i in tqdm(range(1, CONFIG['num_synthetic_samples']), desc="Generating samples"):
                sample = sample_with_guidance(G_matrix, B_matrix, CONFIG, track_residuals=False)
                synthetic_samples.append(sample[0])

            synthetic_data = np.array(synthetic_samples)
            ground_truth_data = dataset.numpy()

            # Verification Results
            print("\n" + "="*60)
            print("VERIFICATION RESULTS (As per Table I and II in the paper)")
            print("="*60)

            # 1. Wasserstein Distance (Table I)
            print("\n1. Statistical Similarity (Wasserstein Distance):")
            subset_size = min(1000, len(ground_truth_data), len(synthetic_data))
            w_distance = compute_wasserstein_distance_multidim(
                ground_truth_data[:subset_size],
                synthetic_data[:subset_size]
            )
            print(f"   Wasserstein distance: {w_distance:.4f}")

            # ==============================================================
            # 2. Constraint Satisfaction (Section VI-C)
            # ==============================================================
            print("\n2. Constraint Satisfaction (Power Flow Residuals):")
            synth_tensor = torch.tensor(synthetic_data, dtype=torch.float32)
            synth_residuals = power_flow_residual(synth_tensor, G_matrix, B_matrix)
            synth_mismatch = torch.sqrt((synth_residuals ** 2).mean(dim=1))
            print(f"   Synthetic mean mismatch: {synth_mismatch.mean():.6f} ± {synth_mismatch.std():.6f}")
            print(f"   Ground truth mean mismatch: {gt_mismatch.mean():.6f} ± {gt_mismatch.std():.6f}")

            # Plot residual convergence history
            plt.figure(figsize=(6, 4))
            plt.plot(residual_history, linewidth=2)
            plt.xlabel("Diffusion step (t)")
            plt.ylabel("Residual norm ||H(x)||₂")
            plt.title("Residual Convergence During Guided Sampling")
            plt.grid(True, alpha=0.4)
            plt.tight_layout()
            plt.show()

            # ==============================================================
            # 3. Visual Comparison (Section VI-B figures)
            # ==============================================================
            print("\n3. Visualizing statistical similarity...")

            var_names = ['P (MW)', 'Q (MVAr)', 'V (p.u.)', 'θ (rad)']
            num_buses = G_matrix.shape[0]
            fig, axs = plt.subplots(4, 1, figsize=(8, 10))
            for i in range(4):
                start = i * num_buses
                end = (i + 1) * num_buses
                sns.kdeplot(dataset[:, start:end].flatten(), label="Ground Truth", ax=axs[i])
                sns.kdeplot(synthetic_data[:, start:end].flatten(), label="Synthetic", ax=axs[i])
                axs[i].set_title(var_names[i])
                axs[i].legend()
            plt.tight_layout()
            plt.show()

            # Additional plots for P, Q, V, θ for five randomly selected buses
            print("   Creating bus-level distribution plots for five buses...")
            np.random.seed(CONFIG['seed'])  # Ensure reproducibility
            selected_buses = np.random.choice(num_buses, 5, replace=False)
            fig, axs = plt.subplots(5, 4, figsize=(16, 12), sharex='col')
            for row, bus_idx in enumerate(selected_buses):
                # Plot P, Q, V, θ for each selected bus
                for col, (var_name, offset) in enumerate(zip(var_names, [0, num_buses, 2*num_buses, 3*num_buses])):
                    idx = offset + bus_idx
                    sns.kdeplot(dataset[:, idx].numpy(), label="Ground Truth", ax=axs[row, col])
                    sns.kdeplot(synthetic_data[:, idx], label="Synthetic", ax=axs[row, col])
                    if row == 0:
                        axs[row, col].set_title(f"{var_name}")
                    if col == 0:
                        axs[row, col].set_ylabel(f"Bus {bus_idx}")
                    if row == 4:
                        axs[row, col].set_xlabel(var_name.split(' ')[0])
                    axs[row, col].legend()
            plt.suptitle("Bus-Level Distributions for P, Q, V, θ", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()

            # Joint scatter plots like in the image for 5 random buses
            print("   Creating joint density plots for 5 random buses...")
            base_kv = net.bus.vn_kv.values  # Base voltages for each bus
            np.random.seed(CONFIG['seed'])  # Ensure reproducibility
            selected_buses = np.random.choice(num_buses, 5, replace=False)
            fig, axs = plt.subplots(2, 5, figsize=(15, 6), sharey='row')
            for col, bus in enumerate(selected_buses):
                # Top row: q vs p (MVAr vs MW)
                sns.kdeplot(x=ground_truth_data[:, bus], y=ground_truth_data[:, num_buses + bus],
                            fill=True, cmap="Blues", alpha=0.5, ax=axs[0, col])
                sns.scatterplot(x=synthetic_data[:, bus], y=synthetic_data[:, num_buses + bus],
                                color="magenta", alpha=0.3, label="Syn.: constrained", ax=axs[0, col])
                axs[0, col].set_xlabel(f"p_{bus+1} (MW)")
                if col == 0:
                    axs[0, col].set_ylabel("q (MVAr)")
                axs[0, col].legend(loc='upper left')

                # Bottom row: theta (rad) vs v (kV)
                v_gt_kv = ground_truth_data[:, 2*num_buses + bus] * base_kv[bus]
                theta_gt = ground_truth_data[:, 3*num_buses + bus]
                v_syn_kv = synthetic_data[:, 2*num_buses + bus] * base_kv[bus]
                theta_syn = synthetic_data[:, 3*num_buses + bus]
                sns.kdeplot(x=v_gt_kv, y=theta_gt,
                            fill=True, cmap="Blues", alpha=0.5, ax=axs[1, col])
                sns.scatterplot(x=v_syn_kv, y=theta_syn,
                                color="magenta", alpha=0.3, label="Syn.: constrained", ax=axs[1, col])
                axs[1, col].set_xlabel(f"v_{bus+1} (kV)")
                if col == 0:
                    axs[1, col].set_ylabel("θ (rad)")
                axs[1, col].legend(loc='upper left')

            # Add colorbar for density
            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
            plt.colorbar(axs[0,0].collections[0], cax=cbar_ax, label="Density Estimate")
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            plt.show()

            # Joint scatter plots (P vs Q, V vs θ)
            print("   Creating joint density plots...")
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            sns.kdeplot(
                x=dataset[:, 0], y=dataset[:, num_buses],
                fill=True, cmap="Blues", label="Ground Truth", alpha=0.5
            )
            sns.scatterplot(
                x=synthetic_data[:, 0], y=synthetic_data[:, num_buses],
                color="red", alpha=0.3, label="Synthetic"
            )
            plt.title("Joint Distribution: P vs Q")

            plt.subplot(1, 2, 2)
            sns.kdeplot(
                x=dataset[:, 2*num_buses], y=dataset[:, 3*num_buses],
                fill=True, cmap="Greens", label="Ground Truth", alpha=0.5
            )
            sns.scatterplot(
                x=synthetic_data[:, 2*num_buses], y=synthetic_data[:, 3*num_buses],
                color="purple", alpha=0.3, label="Synthetic"
            )
            plt.title("Joint Distribution: V vs θ")
            plt.tight_layout()
            plt.show()

            # ==============================================================
            # 4. Utility in Downstream ML Task (Section VI-D)
            # ==============================================================
            print("\n4. Downstream ML Task: Warm-Start Prediction Test")

            import sklearn.model_selection as skms
            import sklearn.neural_network as sknn
            from sklearn.metrics import mean_squared_error

            # Create simple supervised learning task
            # Here we try to learn mapping from (p,q) → (v,θ)
            X_gt = dataset[:, :2*num_buses].numpy()
            y_gt = dataset[:, 2*num_buses:].numpy()

            X_syn = synthetic_data[:, :2*num_buses]
            y_syn = synthetic_data[:, 2*num_buses:]

            # Train-test split
            X_train_gt, X_test_gt, y_train_gt, y_test_gt = skms.train_test_split(X_gt, y_gt, test_size=0.3, random_state=CONFIG['seed'])
            X_train_syn, X_test_syn, y_train_syn, y_test_syn = skms.train_test_split(X_syn, y_syn, test_size=0.3, random_state=CONFIG['seed'])

            # MLP regressor
            def train_mlp(X_train, y_train):
                mlp = sknn.MLPRegressor(hidden_layer_sizes=(128, 128),
                                         activation='relu',
                                         solver='adam',
                                         max_iter=500,
                                         random_state=CONFIG['seed'])
                mlp.fit(X_train, y_train)
                return mlp

            print("   Training baseline (Ground Truth)...")
            mlp_gt = train_mlp(X_train_gt, y_train_gt)
            print("   Training on constrained synthetic data...")
            mlp_syn = train_mlp(X_train_syn, y_train_syn)

            # Evaluate
            y_pred_gt = mlp_gt.predict(X_test_gt)
            y_pred_syn = mlp_syn.predict(X_test_gt)

            mse_gt = mean_squared_error(y_test_gt, y_pred_gt)
            mse_syn = mean_squared_error(y_test_gt, y_pred_syn)

            print(f"   MSE using Ground Truth: {mse_gt:.6f}")
            print(f"   MSE using Synthetic Data: {mse_syn:.6f}")
            print("\n===========================================================")
            print("EXPERIMENTAL VERIFICATION COMPLETED SUCCESSFULLY")
            print("===========================================================")

        else:
            print("Dataset generation failed or returned no samples.")
    else:
        print("Dataset generation failed. Check pandapower installation or seed.")
