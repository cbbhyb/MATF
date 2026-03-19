"""
yancy F. 2020/10/31
For revised DASMN.
"""

from abc import ABC

import torch
import torch.nn.modules as nn
import torch.nn.functional as F

from my_utils.metric_utils import Euclidean_Distance


device = torch.device('cuda:0')
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

Conv_CHN = 64
K_SIZE = 3
PADDING = (K_SIZE - 1) // 2

# ========================================================================
# Temperature-Free Contrastive Learning Configuration
# Based on: "Temperature-Free Loss Function for Contrastive Learning"
# ========================================================================
USE_TEMPERATURE_FREE = True  # True: use arctanh method; False: use original temperature division
SIMCLR_T_BASE = 1.0           # Base temperature for arctanh method (recommended: 1.0)
                              # - 0.5: more aggressive, larger gradients
                              # - 1.0: balanced (paper's recommendation)
                              # - 2.0: smoother, smaller gradients
SIMCLR_CLAMP_VALUE = 0.99     # Clamp cosine similarity to avoid ±1 (recommended: 0.99)
                              # - 0.9: conservative
                              # - 0.99: balanced (recommended)
                              # - 0.999: aggressive
SIMCLR_TEMPERATURE = 6        # Temperature for original method (only used when USE_TEMPERATURE_FREE=False)
# ========================================================================

# ========================================================================
# Dynamic Prototype Adaptation (DPA) Configuration
# Based on: "Dynamic Prototype Adaptation with Distillation for Few-shot Point Cloud Segmentation"
# ========================================================================
USE_DPA = False                # True: use Dynamic Prototype Adaptation; False: use baseline prototypical network
DPA_NUM_LAYERS = 2            # Number of decoder layers for iterative refinement (paper uses 2)2---94   1----99
DPA_GAMMA = 0.1               # Weight for prototype distillation loss (paper uses 0.1)
# ========================================================================

# ========================================================================
# Manifold-Aware Dynamic Prototype Adaptation (MA-DPA) Configuration
# Based on: "Manifold-Aware Temperature-Free Prototypical Networks for Open-Set Cross-Domain Fault Diagnosis"
# ========================================================================
USE_MA_DPA = True             # True: use MA-DPA with geometric transferability gating; False: use original DPA
MA_DPA_GATE_LAMBDA = 10.0     # Sharpness of the transferability gate (λ in paper, default: 10.0)
MA_DPA_GATE_TAU = 0.7         # Manifold boundary threshold (τ in paper, default: 0.5)
MA_DPA_TF_TAU_BASE = 0.5      # Temperature-Free metric base temperature (default: 0.5)
# ========================================================================

# ========================================================================
# Domain-Specific Batch Normalization (DSBN) Configuration
# Based on: "Manifold-Aware Temperature-Free Prototypical Networks for Open-Set Cross-Domain Fault Diagnosis"
# ========================================================================
USE_DSBN = True               # True: use DSBN to decouple domain statistics; False: use standard BN
NUM_DOMAINS = 2               # Number of domains (typically 2: source and target)
# ========================================================================

class SimCLRLoss(nn.Module):
    # Based on the implementation of SupContrast
    # Extended with Temperature-Free option
    def __init__(self, temperature):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature
        self.use_temp_free = USE_TEMPERATURE_FREE
        self.t_base = SIMCLR_T_BASE
        self.clamp_value = SIMCLR_CLAMP_VALUE

        # Print configuration
        if self.use_temp_free:
            print(f'[SimCLR] Using Temperature-Free method with t_base={self.t_base}, clamp={self.clamp_value}')
        else:
            print(f'[SimCLR] Using original temperature division with τ={self.temperature}')

    def forward(self, features):
        """
        input:
            - features: hidden feature representation of shape [b, 2, dim]
        output:
            - loss: loss computed according to SimCLR
        """

        b, n, dim = features.size()
        assert (n == 2)
        mask = torch.eye(b, dtype=torch.float32).cuda() #返回一个对角线全是1的数组64，
        features = F.normalize(features, dim=2)
        contrast_features = torch.cat(torch.unbind(features, dim=1),dim=0)
        anchor = features[:, 0]

        # Compute cosine similarity (dot product of normalized vectors)
        cos_sim = torch.matmul(anchor, contrast_features.T)  # [b, 2b]

        # ========== Choose method based on configuration ==========
        if self.use_temp_free:
            # Temperature-Free method: 2 * arctanh(cos_sim) / t_base
            # Step 1: Clamp to avoid ±1 (which would cause inf)
            cos_sim_safe = torch.clamp(cos_sim, -self.clamp_value, self.clamp_value)

            # Step 2: Apply scaled log-odds transformation
            # logits = log((1 + cos)/(1 - cos)) = 2 * arctanh(cos)
            logits = 2 * torch.atanh(cos_sim_safe) / self.t_base
        else:
            # Original method: divide by temperature
            logits = cos_sim / self.temperature
        # ==========================================================

        # Log-sum trick for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        mask = mask.repeat(1, 2)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(b).view(-1, 1).cuda(), 0)
        mask = mask * logits_mask

        # Log-softmax
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # Mean log-likelihood for positive
        sim_loss = - ((mask * log_prob).sum(1) / mask.sum(1)).mean() + 1e-6

        return sim_loss


def compute_temperature_free_similarity(z_i, z_j, tau_base=0.5, epsilon=1e-6):
    """
    Compute Temperature-Free similarity using arctanh transformation.
    Based on paper equation (12): sim_TF(z_i, z_j) = arctanh(clip(cos(z_i, z_j), -1+ε, 1-ε)) / τ_base

    Args:
        z_i: [*, d] - First feature vector(s)
        z_j: [*, d] - Second feature vector(s)
        tau_base: Manifold scaling constant (default: 0.5 as in paper)
        epsilon: Numerical stability term (default: 1e-6)

    Returns:
        sim_TF: Temperature-Free similarity with infinite range (-∞, ∞)
    """
    # Normalize features to unit sphere
    z_i_norm = F.normalize(z_i, p=2, dim=-1)
    z_j_norm = F.normalize(z_j, p=2, dim=-1)

    # Compute cosine similarity
    cos_sim = (z_i_norm * z_j_norm).sum(dim=-1)

    # Clip to avoid numerical instability at ±1
    cos_sim_clipped = torch.clamp(cos_sim, -1 + epsilon, 1 - epsilon)

    # Apply arctanh transformation to create unbounded metric
    sim_TF = torch.atanh(cos_sim_clipped) / tau_base

    return sim_TF


class DomainSpecificBatchNorm1d(nn.Module):
    """
    Domain-Specific Batch Normalization for 1D data.
    Maintains separate BN parameters for each domain to decouple domain-private statistics.

    Based on paper equation (1) and (2):
    DSBN_s(x) = γ_s * (x - μ_s) / √(σ_s² + ε) + β_s
    DSBN_t(x) = γ_t * (x - μ_t) / √(σ_t² + ε) + β_t
    """
    def __init__(self, num_features, num_domains=2, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.num_domains = num_domains
        self.eps = eps
        self.momentum = momentum

        # Create separate BN layers for each domain
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(num_features, eps=eps, momentum=momentum, affine=True)
            for _ in range(num_domains)
        ])

    def forward(self, x, domain_label=0):
        """
        Args:
            x: Input tensor [batch, channels, length]
            domain_label: Domain index (0 for source, 1 for target, etc.)
        Returns:
            Normalized output using domain-specific statistics
        """
        if domain_label >= self.num_domains:
            raise ValueError(f"domain_label {domain_label} exceeds num_domains {self.num_domains}")

        return self.bn_layers[domain_label](x)


class PrototypeDecoderLayer(nn.Module):
    """
    Single layer of Manifold-Aware Dynamic Prototype Adaptation (MA-DPA) implementing:
    1. Geometric Transferability Estimation: Distinguishes domain-shifted samples from open-set outliers
    2. Gated Prototype Rectification (PR): Aligns prototypes with transferability gating
    3. Outlier-Suppressed Prototype-to-Query Attention (P2QA): Aggregates query context with outlier suppression
    """
    def __init__(self, z_dim, gate_lambda=10.0, gate_tau=0.5):
        super().__init__()
        self.z_dim = z_dim

        # MA-DPA gating parameters
        self.gate_lambda = gate_lambda  # Sharpness of the gate (λ in paper)
        self.gate_tau = gate_tau        # Manifold boundary threshold (τ in paper)

        # For Prototype Rectification
        self.pr_query = nn.Linear(z_dim, z_dim)
        self.pr_key = nn.Linear(z_dim, z_dim)
        self.pr_value = nn.Linear(z_dim, z_dim)

        # For Prototype-to-Query Attention
        self.p2q_query = nn.Linear(z_dim, z_dim)
        self.p2q_key = nn.Linear(z_dim, z_dim)
        self.p2q_value = nn.Linear(z_dim, z_dim)

        self.norm = nn.LayerNorm(z_dim)

        # Learnable step size for prototype rectification (η in paper)
        self.eta = nn.Parameter(torch.tensor(0.1))

    def forward(self, z_proto, zs_reshaped, zq, way, ns, nq):
        """
        Args:
            z_proto: [way, z_dim] - Current prototypes
            zs_reshaped: [way, ns, z_dim] - Support features
            zq: [way*nq, z_dim] - Query features
            way, ns, nq: Integers for reshaping
        Returns:
            z_proto_refined: [way, z_dim] - Refined prototypes
        """
        if USE_MA_DPA:
            # ========== MA-DPA: Manifold-Aware Dynamic Prototype Adaptation ==========

            # Step 1: Geometric Transferability Estimation
            # Compute manifold affinity score: s_j = max_k M(x_qj, c_k) using Temperature-Free metric
            affinity_scores = []
            for k in range(way):
                sim_k = compute_temperature_free_similarity(
                    zq, z_proto[k].unsqueeze(0), tau_base=MA_DPA_TF_TAU_BASE
                )  # [way*nq]
                affinity_scores.append(sim_k)

            affinity_scores = torch.stack(affinity_scores, dim=1)  # [way*nq, way]
            s_j, _ = torch.max(affinity_scores, dim=1)  # [way*nq]

            # Transferability weight: w_j = 1/(1 + exp(-λ(s_j - τ)))
            w_j = 1.0 / (1.0 + torch.exp(-MA_DPA_GATE_LAMBDA * (s_j - MA_DPA_GATE_TAU)))  # [way*nq]

            # Step 2: Gated Prototype Rectification
            # Global statistical representation of support set
            F_s = zs_reshaped.mean(dim=1)  # [way, z_dim]

            # Rectification coefficient: α_j = w_j · Softmax((x_qj)^T F_s)
            rectification_logits = torch.matmul(zq, F_s.transpose(0, 1))  # [way*nq, way]
            rectification_weights = F.softmax(rectification_logits, dim=-1)  # [way*nq, way]
            alpha_j = rectification_weights * w_j.unsqueeze(1)  # [way*nq, way]

            # Rectified prototype: ĉ_k = c_k + η * Σ(α_j · (x_qj - c_k))
            z_proto_rectified = []
            for k in range(way):
                residuals = zq - z_proto[k].unsqueeze(0)  # [way*nq, z_dim]
                weighted_residuals = alpha_j[:, k].unsqueeze(1) * residuals  # [way*nq, z_dim]
                delta = weighted_residuals.sum(dim=0)  # [z_dim]
                z_proto_k_rectified = z_proto[k] + self.eta * delta
                z_proto_rectified.append(z_proto_k_rectified)

            z_proto_rectified = torch.stack(z_proto_rectified, dim=0)  # [way, z_dim]

            # Step 3: Outlier-Suppressed Prototype-to-Query Attention
            Q = self.p2q_query(z_proto_rectified)  # [way, z_dim]
            K = self.p2q_key(zq)  # [way*nq, z_dim]
            V = self.p2q_value(zq)  # [way*nq, z_dim]

            attn_scores = torch.matmul(Q, K.transpose(0, 1)) / (self.z_dim ** 0.5)  # [way, way*nq]
            attn_weights = F.softmax(attn_scores, dim=-1)  # [way, way*nq]

            # Modulate values with transferability weights: V_j ⊙ w_j
            V_gated = V * w_j.unsqueeze(1)  # [way*nq, z_dim]
            attended = torch.matmul(attn_weights, V_gated)  # [way, z_dim]

            # Residual connection + Layer normalization
            z_proto_final = self.norm(z_proto_rectified + attended)

        else:
            # ========== Original DPA: Dynamic Prototype Adaptation ==========

            # Reshape query for per-class processing
            zq_reshaped = zq.reshape(way, nq, -1)  # [way, nq, z_dim]

            # Prototype Rectification (PR)
            z_proto_pr = []
            for i in range(way):
                q_i = zq_reshaped[i]  # [nq, z_dim]
                s_all = zs_reshaped.reshape(way * ns, -1)  # [way*ns, z_dim]

                Q = self.pr_query(q_i)  # [nq, z_dim]
                K = self.pr_key(s_all)  # [way*ns, z_dim]
                V = self.pr_value(s_all)  # [way*ns, z_dim]

                attn_scores = torch.matmul(Q, K.transpose(0, 1)) / (self.z_dim ** 0.5)
                attn_weights = F.softmax(attn_scores, dim=-1)
                aggregated = torch.matmul(attn_weights, V)

                delta = aggregated.mean(dim=0)
                z_proto_pr.append(z_proto[i] + delta)

            z_proto_pr = torch.stack(z_proto_pr, dim=0)  # [way, z_dim]

            # Prototype-to-Query Attention (P2QA)
            Q = self.p2q_query(zq)  # [way*nq, z_dim]
            K = self.p2q_key(z_proto_pr)  # [way, z_dim]
            V = self.p2q_value(z_proto_pr)  # [way, z_dim]

            attn_scores = torch.matmul(Q, K.transpose(0, 1)) / (self.z_dim ** 0.5)
            attn_weights = F.softmax(attn_scores, dim=-1)
            attended = torch.matmul(attn_weights, V)

            z_proto_p2qa = []
            for i in range(way):
                z_proto_p2qa.append(attended[i::way].mean(dim=0))
            z_proto_p2qa = torch.stack(z_proto_p2qa, dim=0)

            # Residual connection + Layer normalization
            z_proto_final = self.norm(z_proto_pr + z_proto_p2qa)

        return z_proto_final


class PrototypeDecoder(nn.Module):
    """
    Multi-layer Prototype Decoder with iterative refinement.
    Implements DPA or MA-DPA approach based on USE_MA_DPA flag.
    """
    def __init__(self, z_dim, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            PrototypeDecoderLayer(z_dim) for _ in range(num_layers)
        ])

        if USE_MA_DPA:
            print(f'[MA-DPA] Using {num_layers}-layer Manifold-Aware Dynamic Prototype Adaptation')
            print(f'[MA-DPA] Gate parameters: λ={MA_DPA_GATE_LAMBDA}, τ={MA_DPA_GATE_TAU}')
        else:
            print(f'[DPA] Using {num_layers}-layer Dynamic Prototype Adaptation')

    def forward(self, z_proto_init, zs_reshaped, zq, way, ns, nq):
        """
        Args:
            z_proto_init: [way, z_dim] - Initial prototypes (mean of support features)
            zs_reshaped: [way, ns, z_dim] - Support features
            zq: [way*nq, z_dim] - Query features
            way, ns, nq: Integers
        Returns:
            z_proto: [way, z_dim] - Final refined prototypes
        """
        z_proto = z_proto_init

        # Iteratively refine prototypes through multiple layers
        for layer in self.layers:
            z_proto = layer(z_proto, zs_reshaped, zq, way, ns, nq)

        return z_proto


def conv_block(in_channels, out_channels):
    """Legacy conv_block for backward compatibility (uses standard BN)"""
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=K_SIZE, padding=PADDING),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2),
    )


class ConvBlock(nn.Module):
    """
    Convolutional block with optional Domain-Specific Batch Normalization.
    Supports both standard BN and DSBN based on USE_DSBN flag.
    """
    def __init__(self, in_channels, out_channels, use_dsbn=USE_DSBN, num_domains=NUM_DOMAINS):
        super().__init__()
        self.use_dsbn = use_dsbn

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=K_SIZE, padding=PADDING)

        if self.use_dsbn:
            self.bn = DomainSpecificBatchNorm1d(out_channels, num_domains=num_domains)
        else:
            self.bn = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, x, domain_label=0):
        """
        Args:
            x: Input tensor [batch, channels, length]
            domain_label: Domain index (only used if use_dsbn=True)
        Returns:
            Output tensor after conv -> bn -> relu -> pool
        """
        x = self.conv(x)

        if self.use_dsbn:
            x = self.bn(x, domain_label)
        else:
            x = self.bn(x)

        x = self.relu(x)
        x = self.pool(x)

        return x


class Encoder(nn.Module, ABC):
    def __init__(self, in_chn=1, cb_num=8, use_dsbn=USE_DSBN):
        super().__init__()
        self.use_dsbn = use_dsbn
        print('The Convolution Channel: {}'.format(Conv_CHN))
        print('The Convolution Block: {}'.format(cb_num))

        if self.use_dsbn:
            print(f'[DSBN] Using Domain-Specific Batch Normalization with {NUM_DOMAINS} domains')
            # Use ConvBlock with DSBN support
            self.conv_blocks = nn.ModuleList([
                ConvBlock(in_chn if i == 0 else Conv_CHN, Conv_CHN, use_dsbn=True)
                for i in range(cb_num)
            ])
        else:
            print('[DSBN] Using standard Batch Normalization')
            # Use legacy conv_block with standard BN
            conv1 = conv_block(in_chn, Conv_CHN)
            conv_more = [conv_block(Conv_CHN, Conv_CHN) for i in range(cb_num - 1)]
            self.conv_blocks = nn.Sequential(conv1, *conv_more)

    def forward(self, x, domain_label=0):
        """
        Args:
            x: Input tensor [batch, channels, length]
            domain_label: Domain index (0 for source, 1 for target)
        Returns:
            Flattened feature tensor
        """
        if self.use_dsbn:
            # Pass domain_label through each ConvBlock
            for block in self.conv_blocks:
                x = block(x, domain_label)
        else:
            # Use sequential blocks (no domain_label needed)
            x = self.conv_blocks(x)

        feat = x.reshape(x.shape[0], -1)
        return feat


class MetricNet(nn.Module, ABC):
    def __init__(self, way, ns, nq, vis, cb_num=8):
        super().__init__()
        self.chn = 1
        self.way = way
        self.ns = ns
        self.nq = nq
        self.vis = vis
        self.encoder = Encoder(cb_num=cb_num)
        self.sim_loss = SimCLRLoss(SIMCLR_TEMPERATURE)  # Use global config

        # Dynamic Prototype Adaptation
        self.use_dpa = USE_DPA
        if self.use_dpa:
            # Get z_dim from encoder output
            # The encoder output is reshaped to [batch, -1], need to infer dimension
            # For cb_num=8 with Conv_CHN=64 and input 1024, output dim is 64 * (1024 / 2^8) = 64 * 4 = 256
            self.z_dim = None  # Will be inferred on first forward pass
            self.proto_decoder = None  # Will be initialized after first forward
            print(f'[DPA] Dynamic Prototype Adaptation ENABLED (num_layers={DPA_NUM_LAYERS}, gamma={DPA_GAMMA})')
        else:
            print('[DPA] Using baseline prototypical network (DPA DISABLED)')

        # self.criterion = nn.CrossEntropyLoss()  # ([n, nc], n)

    def get_loss(self, net_out, target_id, conloss, distill_loss=0.0):
        # method 1:
        # log_p_y = torch.log_softmax(net_out, dim=-1).reshape(self.way, self.nq, -1)  # [nc, nq, nc]
        # loss_val = -log_p_y.gather(dim=2, index=target_ids).squeeze(dim=-1).reshape(-1).mean()
        # y_hat = log_p_y.max(dim=2)[1]  # [nc, nq]

        # method 2:

        log_p_y = torch.log_softmax(net_out, dim=-1)  # [nc*nq, nc], probability.
        loss = F.nll_loss(log_p_y, target_id.reshape(-1))  # (N, nc), (N,)
        loss += conloss
        loss += distill_loss  # Add prototype distillation loss
        y_hat = torch.max(log_p_y, dim=1)[1]  # [nc*nq]
        acc = torch.eq(y_hat, target_id).float().mean()

        return loss, acc, y_hat, -log_p_y.reshape(self.way, self.nq, -1)

    def get_features(self, x, domain_label=0):
        return self.encoder.forward(x, domain_label)

    def forward(self, xs, xq, sne_state=False):
        # target_ids [nc, nq]
        target_id = torch.arange(self.way).unsqueeze(1).repeat([1, self.nq])
        target_id = target_id.long().to(device)
        # ================
        # x = torch.cat([xs.reshape(self.way * self.ns, self.chn, -1),
        #                xq.reshape(self.way * self.nq, self.chn, -1)], dim=0)
        # z = self.get_features(x)
        # z_proto = z[:self.way * self.ns].reshape(self.way, self.ns, z.shape[-1]).mean(dim=1)
        # zq = z[self.way * self.ns:]
        # =================
        xs = xs.reshape(self.way * self.ns, self.chn, -1)
        xq = xq.reshape(self.way * self.nq, self.chn, -1)

        # Extract features with domain-specific normalization
        zs = self.get_features(xs, domain_label=0)  # Support: source domain
        zq = self.get_features(xq, domain_label=1)  # Query: target domain

        # Initialize DPA decoder on first forward pass
        if self.use_dpa and self.proto_decoder is None:
            self.z_dim = zs.shape[-1]
            self.proto_decoder = PrototypeDecoder(self.z_dim, num_layers=DPA_NUM_LAYERS).to(device)

        # Reshape support features for prototype computation
        zs_reshaped = zs.reshape(self.way, self.ns, -1)  # [way, ns, z_dim]
        z_proto_init = zs_reshaped.mean(dim=1)  # [way, z_dim] - Initial vanilla prototypes

        # ========== Conditional DPA ==========
        distill_loss = 0.0  # No longer used (distillation removed from MA-DPA)
        if self.use_dpa:
            # Use Dynamic Prototype Adaptation (or MA-DPA if USE_MA_DPA=True)
            z_proto = self.proto_decoder(z_proto_init, zs_reshaped, zq, self.way, self.ns, self.nq)
        else:
            # Use baseline prototypical network
            z_proto = z_proto_init
        # =====================================

        dist = Euclidean_Distance(zq, z_proto)  # [nc*nq, nc]

        contra_feature = torch.stack((zs, zq), 1)
        simclr_loss = self.sim_loss(contra_feature)

        loss_val, acc_val, y_hat, label_distribution = self.get_loss(-dist, target_id.reshape(-1), simclr_loss, distill_loss)


        # if sne_state and self.ns > 1:
        #     self.draw_feature(zq, target_id, y_hat)
        #     self.draw_label(label_distribution, target_id)

        return loss_val, acc_val, zq, y_hat, target_id.reshape(-1)  # 添加预测标签和真实标签


if __name__ == "__main__":
    e = Encoder(cb_num=8)
    data = torch.ones([12, 1, 1024], dtype=torch.float)
    print(e)
    print(e.forward(data).shape)

