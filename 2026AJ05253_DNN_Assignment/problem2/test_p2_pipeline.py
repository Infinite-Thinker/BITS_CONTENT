import os
import sys
import time
import math
import random
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

os.makedirs('figures', exist_ok=True)
RESULTS_CSV_PATH = '../results_p2.csv'
STUDENT_ID = '2026AJ05253'
SEEDS = [3, 13, 23]
D_MODEL = 128
NUM_HEADS = 4
NUM_LAYERS = 4
D_FF = 512
CONTEXT_LEN = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Problem 2 using device: {DEVICE}')

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def log_result(problem, model, seed, seq_len='', test_acc='', macro_f1='',
               test_mse='', params='', sec_per_epoch='', val_loss_nats='',
               bits_per_char='', notes=''):
    columns = [
        'problem', 'model', 'seed', 'seq_len', 'test_acc', 'macro_f1',
        'test_mse', 'params', 'sec_per_epoch', 'val_loss_nats', 'bits_per_char', 'notes'
    ]
    row = pd.DataFrame([{
        'problem': problem,
        'model': model,
        'seed': seed,
        'seq_len': seq_len,
        'test_acc': test_acc,
        'macro_f1': macro_f1,
        'test_mse': test_mse,
        'params': params,
        'sec_per_epoch': sec_per_epoch,
        'val_loss_nats': val_loss_nats,
        'bits_per_char': bits_per_char,
        'notes': notes
    }], columns=columns)
    write_header = not os.path.exists(RESULTS_CSV_PATH) or os.path.getsize(RESULTS_CSV_PATH) == 0
    row.to_csv(RESULTS_CSV_PATH, mode='a', header=write_header, index=False)

# -------------------------------------------------------------
# Task 1 (P2.1): Attention From Scratch
# -------------------------------------------------------------
# PHASE2
# TODO (Student: scaled_dot_product_attention):
def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                mask: torch.Tensor = None, dropout: nn.Dropout = None,
                                return_weights: bool = False):
    """
    Scaled Dot-Product Attention from scratch using tensor operations.
    Q: (..., T_q, d_k)
    K: (..., T_k, d_k)
    V: (..., T_k, d_v)
    mask: optional broadcastable boolean or float tensor where 1 or -inf indicates masking
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(mask, -1e9)
        else:
            scores = scores + mask

    attn_weights = F.softmax(scores, dim=-1)

    if dropout is not None:
        attn_weights = dropout(attn_weights)

    output = torch.matmul(attn_weights, V)
    if return_weights:
        return output, attn_weights
    return output

# PHASE2
# TODO (Student: unit_test_attention):
def unit_test_attention():
    print("--- Running P2.1 Attention Unit Tests ---")
    B, H, T, d_k = 2, 4, 8, 32
    Q = torch.randn(B, H, T, d_k)
    K = torch.randn(B, H, T, d_k)
    V = torch.randn(B, H, T, d_k)

    # 1. Row sum == 1 test
    out, weights = scaled_dot_product_attention(Q, K, V, return_weights=True)
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6), "Attention rows must sum to 1.0"
    print("Pass: Attention rows sum to 1.0 within 1e-6")

    # 2. Match F.scaled_dot_product_attention within 1e-5
    ref_out = F.scaled_dot_product_attention(Q, K, V)
    max_diff = (out - ref_out).abs().max().item()
    assert max_diff < 1e-5, f"Output mismatch vs PyTorch: {max_diff}"
    print(f"Pass: Output matches F.scaled_dot_product_attention (max diff: {max_diff:.2e} < 1e-5)")

    # 3. Causal mask test: future positions must have zero weight
    causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    out_causal, weights_causal = scaled_dot_product_attention(Q, K, V, mask=causal_mask, return_weights=True)
    ref_causal = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    assert (out_causal - ref_causal).abs().max().item() < 1e-5, "Causal attention mismatch"
    upper_tri_weights = weights_causal.masked_select(causal_mask)
    assert (upper_tri_weights == 0.0).all(), "Causal mask must zero out future weights"
    print("Pass: Causal mask strictly zeros out future positions")

    # 4. Hand-checkable 3-token example
    print("\n--- Running 3-Token Hand-Check Verification ---")
    Q_hand = torch.tensor([[1.0, 0.0],
                           [0.0, 1.0],
                           [1.0, 1.0]])
    K_hand = torch.tensor([[1.0, 0.0],
                           [1.0, 1.0],
                           [0.0, 1.0]])
    V_hand = torch.tensor([[1.0, 2.0],
                           [3.0, 4.0],
                           [5.0, 6.0]])

    # Hand calculation:
    # d_k = 2 => sqrt(2) = 1.41421356
    # Q @ K.T =
    # [[1, 1, 0],
    #  [0, 1, 1],
    #  [1, 2, 1]]
    # S = Q @ K.T / sqrt(2) =
    # [[0.70710678, 0.70710678, 0.0],
    #  [0.0, 0.70710678, 0.70710678],
    #  [0.70710678, 1.41421356, 0.70710678]]
    # Softmax row 0: exp(0.70710678)=2.02811498, exp(0)=1.0, sum=5.05622996
    # w0 = [0.401112, 0.401112, 0.197776]
    # Softmax row 1: [0.197776, 0.401112, 0.401112]
    # Softmax row 2: exp(1.41421356)=4.11325038, sum=8.16948034
    # w2 = [0.248255, 0.503490, 0.248255]
    # Hand V outputs:
    # out[0] = 0.401112*[1,2] + 0.401112*[3,4] + 0.197776*[5,6] = [2.593328, 3.593328]
    # out[1] = 0.197776*[1,2] + 0.401112*[3,4] + 0.401112*[5,6] = [3.406672, 4.406672]
    # out[2] = 0.248255*[1,2] + 0.503490*[3,4] + 0.248255*[5,6] = [3.000000, 4.000000]
    hand_out_expected = torch.tensor([[2.593328, 3.593328],
                                     [3.406672, 4.406672],
                                     [3.000000, 4.000000]])

    code_out, code_weights = scaled_dot_product_attention(Q_hand, K_hand, V_hand, return_weights=True)
    hand_diff = (code_out - hand_out_expected).abs().max().item()
    print("Computed Attention Weights:\n", code_weights.numpy())
    print("Computed Output:\n", code_out.numpy())
    print(f"Pass: Hand check matches code output within 1e-5 (max diff: {hand_diff:.2e})")
    assert hand_diff < 1e-5, f"Hand check diff too large: {hand_diff}"

# -------------------------------------------------------------
# Task 2 (P2.2): Multi-Head Attention & Encoder Block
# -------------------------------------------------------------
# PHASE2
# TODO (Student: MultiHeadAttention):
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x_q: torch.Tensor, x_k: torch.Tensor = None, x_v: torch.Tensor = None,
                mask: torch.Tensor = None, return_weights: bool = False):
        if x_k is None: x_k = x_q
        if x_v is None: x_v = x_q

        B, T_q, _ = x_q.shape
        _, T_k, _ = x_k.shape

        # Linear projections & split into heads: (B, T, d_model) -> (B, num_heads, T, d_k)
        Q = self.W_q(x_q).view(B, T_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x_k).view(B, T_k, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x_v).view(B, T_k, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        if return_weights:
            out, weights = scaled_dot_product_attention(Q, K, V, mask=mask, dropout=self.dropout, return_weights=True)
        else:
            out = scaled_dot_product_attention(Q, K, V, mask=mask, dropout=self.dropout, return_weights=False)

        # Recombine heads: (B, num_heads, T_q, d_k) -> (B, T_q, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        out = self.W_o(out)

        if return_weights:
            return out, weights
        return out

# PHASE2
# TODO (Student: SinusoidalPositionalEncoding):
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        return x + self.pe[:, :x.size(1), :]

# PHASE2
# TODO (Student: TransformerEncoderBlock):
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, return_weights: bool = False):
        # Pre-LN architecture
        norm_x = self.ln1(x)
        if return_weights:
            attn_out, weights = self.mha(norm_x, mask=mask, return_weights=True)
        else:
            attn_out = self.mha(norm_x, mask=mask, return_weights=False)
        x = x + self.drop1(attn_out)

        # FFN branch
        ffn_out = self.ffn(self.ln2(x))
        x = x + self.drop2(ffn_out)

        if return_weights:
            return x, weights
        return x

# PHASE2
# TODO (Student: verify_encoder_block_parameters):
def verify_encoder_block_parameters():
    print("\n--- Running P2.2 Parameter Count Derivation Verification ---")
    d = 128
    d_ff = 512
    heads = 4
    block = TransformerEncoderBlock(d_model=d, num_heads=heads, d_ff=d_ff, dropout=0.0)

    # Theoretical derivation:
    # 1. Weights:
    #    MHA: W_q, W_k, W_v, W_o each (d, d) => 4 * d^2 = 4 * 128^2 = 65,536
    #    FFN: W1 (d, d_ff) + W2 (d_ff, d) => 2 * d * d_ff = 2 * 128 * 512 = 131,072
    #    Total Weights = 4d^2 + 2*d*d_ff = 65,536 + 131,072 = 196,608
    # 2. Biases:
    #    MHA: 4 * d = 4 * 128 = 512
    #    FFN: d_ff + d = 512 + 128 = 640
    #    Total Biases = 1,152
    # 3. LayerNorms:
    #    2 LayerNorms, each with gamma(d) and beta(d) => 2 * (128 + 128) = 512
    # Grand Total Parameters = 196,608 + 1,152 + 512 = 198,272
    total_weights = (
        block.mha.W_q.weight.numel() + block.mha.W_k.weight.numel() +
        block.mha.W_v.weight.numel() + block.mha.W_o.weight.numel() +
        block.ffn[0].weight.numel() + block.ffn[3].weight.numel()
    )
    total_biases = (
        block.mha.W_q.bias.numel() + block.mha.W_k.bias.numel() +
        block.mha.W_v.bias.numel() + block.mha.W_o.bias.numel() +
        block.ffn[0].bias.numel() + block.ffn[3].bias.numel()
    )
    total_ln = (
        block.ln1.weight.numel() + block.ln1.bias.numel() +
        block.ln2.weight.numel() + block.ln2.bias.numel()
    )
    total_params = sum(p.numel() for p in block.parameters())

    print(f"Weight count (theoretical: 196,608): {total_weights}")
    print(f"Bias count (theoretical: 1,152): {total_biases}")
    print(f"LayerNorm params (theoretical: 512): {total_ln}")
    print(f"Grand Total params (theoretical: 198,272): {total_params}")

    assert total_weights == 196608, f"Weights mismatch: {total_weights}"
    assert total_biases == 1152, f"Biases mismatch: {total_biases}"
    assert total_ln == 512, f"LayerNorm mismatch: {total_ln}"
    assert total_params == 198272, f"Total mismatch: {total_params}"
    print("Pass: Encoder block parameter counts perfectly reconciled!")

# -------------------------------------------------------------
# Task 3 (P2.3): Task A - HAR Transformer Classifier
# -------------------------------------------------------------
# PHASE2
# TODO (Student: HARTransformerClassifier):
class HARTransformerClassifier(nn.Module):
    def __init__(self, input_size: int = 9, d_model: int = 128, num_heads: int = 4,
                 num_layers: int = 2, d_ff: int = 512, num_classes: int = 6,
                 pos_encoding: str = 'sinusoidal', dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        self.pos_encoding_type = pos_encoding
        if pos_encoding == 'sinusoidal':
            self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len=200)
        elif pos_encoding == 'learned':
            self.pos_encoder = nn.Parameter(torch.zeros(1, 200, d_model))
            nn.init.normal_(self.pos_encoder, std=0.02)
        else: # 'none'
            self.pos_encoder = None

        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        x = self.input_proj(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (B, T+1, d_model)

        if self.pos_encoding_type == 'sinusoidal':
            x = self.pos_encoder(x)
        elif self.pos_encoding_type == 'learned':
            x = x + self.pos_encoder[:, :T+1, :]

        for layer in self.layers:
            x = layer(x)

        x = self.ln_f(x)
        cls_out = x[:, 0, :]
        return self.fc(cls_out)

# -------------------------------------------------------------
# Task 4 (P2.4): Task B - Character-Level Causal LM
# -------------------------------------------------------------
# PHASE2
# TODO (Student: CausalTransformerLM):
class CausalTransformerLM(nn.Module):
    def __init__(self, vocab_size: int = 65, d_model: int = 128, num_heads: int = 4,
                 num_layers: int = 4, d_ff: int = 512, max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, use_causal_mask: bool = True, return_weights_layer: int = None):
        B, T = idx.shape
        tok_emb = self.token_emb(idx)
        x = self.drop(tok_emb + self.pos_emb[:, :T, :])

        mask = None
        if use_causal_mask:
            mask = torch.triu(torch.ones(T, T, device=idx.device, dtype=torch.bool), diagonal=1)

        captured_weights = None
        for i, layer in enumerate(self.layers):
            if return_weights_layer == i:
                x, captured_weights = layer(x, mask=mask, return_weights=True)
            else:
                x = layer(x, mask=mask, return_weights=False)

        x = self.ln_f(x)
        logits = self.head(x)

        if return_weights_layer is not None:
            return logits, captured_weights
        return logits

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int = 20, use_causal_mask: bool = True):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -128:] if idx.size(1) > 128 else idx
            logits = self.forward(idx_cond, use_causal_mask=use_causal_mask)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

print("Problem 2 definitions loaded successfully.")

# -------------------------------------------------------------
# Data Loading & Preparation for Problem 2
# -------------------------------------------------------------
HAR_DATASET_FOLDER = os.path.join('data_har', 'UCI HAR Dataset')
CHANNELS = [
    'body_acc_x', 'body_acc_y', 'body_acc_z',
    'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
    'total_acc_x', 'total_acc_y', 'total_acc_z'
]

def load_har_raw():
    def load_signals(split):
        signal_data = []
        for ch in CHANNELS:
            fn = f'{ch}_{split}.txt'
            fp = os.path.join(HAR_DATASET_FOLDER, split, 'Inertial Signals', fn)
            arr = np.loadtxt(fp, dtype=np.float32)
            signal_data.append(arr[:, :, np.newaxis])
        return np.concatenate(signal_data, axis=2)

    X_train_raw = load_signals('train')
    y_train_raw = np.loadtxt(os.path.join(HAR_DATASET_FOLDER, 'train', 'y_train.txt'), dtype=np.int64) - 1
    sub_train = np.loadtxt(os.path.join(HAR_DATASET_FOLDER, 'train', 'subject_train.txt'), dtype=np.int64)

    X_test_raw = load_signals('test')
    y_test_raw = np.loadtxt(os.path.join(HAR_DATASET_FOLDER, 'test', 'y_test.txt'), dtype=np.int64) - 1
    sub_test = np.loadtxt(os.path.join(HAR_DATASET_FOLDER, 'test', 'subject_test.txt'), dtype=np.int64)

    return X_train_raw, y_train_raw, sub_train, X_test_raw, y_test_raw, sub_test

def get_har_splits(seed: int, X_train_raw, y_train_raw, sub_train, X_test_raw, y_test_raw):
    train_subjects = np.unique(sub_train)
    rng = np.random.default_rng(seed)
    val_subjects = rng.choice(train_subjects, 4, replace=False)
    val_mask = np.isin(sub_train, val_subjects)
    train_mask = ~val_mask

    X_tr = X_train_raw[train_mask]
    y_tr = y_train_raw[train_mask]
    X_val = X_train_raw[val_mask]
    y_val = y_train_raw[val_mask]
    X_te = X_test_raw.copy()
    y_te = y_test_raw.copy()

    mean = X_tr.mean(axis=(0, 1), keepdims=True)
    std = X_tr.std(axis=(0, 1), keepdims=True)
    std[std == 0] = 1.0

    X_tr = (X_tr - mean) / std
    X_val = (X_val - mean) / std
    X_te = (X_te - mean) / std

    return (torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long),
            torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long),
            torch.tensor(X_te, dtype=torch.float32), torch.tensor(y_te, dtype=torch.long))

# -------------------------------------------------------------
# Task 3: Run Task A (HAR Classifier & Positional Ablations)
# -------------------------------------------------------------
# PHASE2
# TODO (Student: run_p2_task_a):
def run_p2_task_a(epochs: int = 8, batch_size: int = 64):
    print("\n=======================================================")
    print("Running P2 Task A: HAR Transformer Encoder Classifier")
    print("=======================================================")
    X_tr_r, y_tr_r, sub_tr, X_te_r, y_te_r, _ = load_har_raw()

    pos_options = ['sinusoidal', 'learned', 'none']
    task_a_results = []

    for pos in pos_options:
        print(f"\n--- Testing Positional Encoding: {pos.upper()} ---")
        accs, f1s = [], []
        epoch_times = []

        for seed in SEEDS:
            set_seed(seed)
            X_tr, y_tr, X_val, y_val, X_te, y_te = get_har_splits(seed, X_tr_r, y_tr_r, sub_tr, X_te_r, y_te_r)
            train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=batch_size, shuffle=False)

            model = HARTransformerClassifier(
                input_size=9, d_model=D_MODEL, num_heads=NUM_HEADS,
                num_layers=2, d_ff=D_FF, pos_encoding=pos, dropout=0.1
            ).to(DEVICE)
            total_params = sum(p.numel() for p in model.parameters())

            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()

            best_val_acc = -1.0
            best_state = None
            t0 = time.time()

            for ep in range(epochs):
                model.train()
                for bx, by in train_loader:
                    bx, by = bx.to(DEVICE), by.to(DEVICE)
                    optimizer.zero_grad()
                    out = model(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                # Validation
                model.eval()
                v_correct = 0
                with torch.no_grad():
                    for bx, by in val_loader:
                        bx, by = bx.to(DEVICE), by.to(DEVICE)
                        v_correct += (model(bx).argmax(dim=-1) == by).sum().item()
                v_acc = v_correct / len(X_val)
                if v_acc > best_val_acc:
                    best_val_acc = v_acc
                    best_state = {k: v.cpu() for k, v in model.state_dict().items()}

            sec_per_ep = (time.time() - t0) / epochs
            epoch_times.append(sec_per_ep)

            # Test Evaluation
            if best_state is not None:
                model.load_state_dict(best_state)
            model.eval()
            preds, targets = [], []
            with torch.no_grad():
                for bx, by in test_loader:
                    bx = bx.to(DEVICE)
                    preds.extend(model(bx).argmax(dim=-1).cpu().numpy())
                    targets.extend(by.numpy())

            acc = accuracy_score(targets, preds)
            f1 = f1_score(targets, preds, average='macro')
            accs.append(acc)
            f1s.append(f1)
            print(f"Pos={pos}, Seed={seed}: Test Acc={acc*100:.2f}%, Macro-F1={f1:.4f}, Sec/Epoch={sec_per_ep:.2f}s")

            log_result(
                problem='Problem2_TaskA_HAR',
                model=f'Transformer_Pos_{pos}',
                seed=seed,
                seq_len=128,
                test_acc=f'{acc:.4f}',
                macro_f1=f'{f1:.4f}',
                params=total_params,
                sec_per_epoch=f'{sec_per_ep:.2f}',
                notes=f'pos={pos}, layers=2, d=128, heads=4'
            )

        print(f"Summary for Pos={pos}: Test Acc = {np.mean(accs)*100:.2f} +/- {np.std(accs)*100:.2f}%, F1 = {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")

# -------------------------------------------------------------
# Task 4: Run Task B (Tiny Shakespeare Causal LM)
# -------------------------------------------------------------
def get_shakespeare_data():
    txt_path = 'input.txt'
    if not os.path.exists(txt_path):
        print("Downloading Tiny Shakespeare...")
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        urllib.request.urlretrieve(url, txt_path)

    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    ch2idx = {ch: i for i, ch in enumerate(chars)}
    idx2ch = {i: ch for i, ch in enumerate(chars)}

    data = np.array([ch2idx[c] for c in text], dtype=np.int64)
    n = int(0.9 * len(data))
    train_data = torch.tensor(data[:n], dtype=torch.long)
    val_data = torch.tensor(data[n:], dtype=torch.long)

    return train_data, val_data, vocab_size, ch2idx, idx2ch

def get_lm_batch(split_data, batch_size: int, block_size: int):
    ix = torch.randint(len(split_data) - block_size, (batch_size,))
    x = torch.stack([split_data[i:i+block_size] for i in ix])
    y = torch.stack([split_data[i+1:i+block_size+1] for i in ix])
    return x, y

# PHASE2
# TODO (Student: run_p2_task_b):
def run_p2_task_b(max_steps: int = 1000, batch_size: int = 32):
    print("\n=======================================================")
    print("Running P2 Task B: Tiny Shakespeare Causal Language Model")
    print("=======================================================")
    train_data, val_data, vocab_size, ch2idx, idx2ch = get_shakespeare_data()
    print(f"Tiny Shakespeare loaded: vocab_size={vocab_size}, train_chars={len(train_data)}, val_chars={len(val_data)}")

    set_seed(3)
    model = CausalTransformerLM(
        vocab_size=vocab_size, d_model=D_MODEL, num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS, d_ff=D_FF, max_len=CONTEXT_LEN, dropout=0.1
    ).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"CausalTransformerLM params: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    step_indices = []

    t0 = time.time()
    model.train()
    for step in range(1, max_steps + 1):
        xb, yb = get_lm_batch(train_data, batch_size, CONTEXT_LEN)
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        logits = model(xb, use_causal_mask=True)
        loss = criterion(logits.view(-1, vocab_size), yb.view(-1))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0 or step == max_steps:
            model.eval()
            with torch.no_grad():
                val_losses_batch = []
                for _ in range(10):
                    vx, vy = get_lm_batch(val_data, batch_size, CONTEXT_LEN)
                    vx, vy = vx.to(DEVICE), vy.to(DEVICE)
                    v_logits = model(vx, use_causal_mask=True)
                    v_loss = criterion(v_logits.view(-1, vocab_size), vy.view(-1))
                    val_losses_batch.append(v_loss.item())
                avg_val_loss = np.mean(val_losses_batch)

            bpc = avg_val_loss / math.log(2.0)
            train_losses.append(loss.item())
            val_losses.append(avg_val_loss)
            step_indices.append(step)
            print(f"Step {step}/{max_steps} | Train Loss: {loss.item():.4f} nats | Val Loss: {avg_val_loss:.4f} nats | BPC: {bpc:.3f}")
            model.train()

    total_time = time.time() - t0
    final_val_loss = val_losses[-1]
    final_bpc = final_val_loss / math.log(2.0)

    # Plot learning curve
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(step_indices, train_losses, label='Train Loss (nats)')
    plt.plot(step_indices, val_losses, label='Val Loss (nats)')
    plt.axhline(math.log(65), color='red', linestyle='--', label='Uniform Baseline (ln 65 ~ 4.17)')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss (nats)')
    plt.title('Tiny Shakespeare Causal Transformer Learning Curve')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('figures/p2_lm_learning_curve.png', dpi=150)
    plt.close()
    print("Saved figures/p2_lm_learning_curve.png")

    log_result(
        problem='Problem2_TaskB_LM',
        model='CausalTransformerLM',
        seed=3,
        seq_len=CONTEXT_LEN,
        params=total_params,
        sec_per_epoch=f'{total_time/max_steps:.3f}',
        val_loss_nats=f'{final_val_loss:.4f}',
        bits_per_char=f'{final_bpc:.4f}',
        notes=f'steps={max_steps}, d={D_MODEL}, layers={NUM_LAYERS}, heads={NUM_HEADS}'
    )

    # Text Generation at Temperatures 0.2, 0.8, 1.5
    print("\n--- Generating Text Samples at T=0.2, 0.8, 1.5 ---")
    model.eval()
    start_context = torch.tensor([[ch2idx['\n']]], dtype=torch.long, device=DEVICE)
    samples = {}

    for temp in [0.2, 0.8, 1.5]:
        out_idx = model.generate(start_context, max_new_tokens=500, temperature=temp, top_k=20)
        chars_gen = [idx2ch[i] for i in out_idx[0].cpu().numpy()]
        text_gen = ''.join(chars_gen)
        samples[temp] = text_gen
        print(f"\n--- Temperature: {temp} ---\n{text_gen[:150]}...")

    with open('generated_samples.txt', 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("BITS Pilani M.Tech AIML - DNN Assignment 2\n")
        f.write(f"Student ID: {STUDENT_ID} | Task B Generated Samples\n")
        f.write("="*70 + "\n\n")
        for temp in [0.2, 0.8, 1.5]:
            f.write(f"SAMPLE AT TEMPERATURE {temp}\n")
            f.write("-" * 40 + "\n")
            f.write(samples[temp] + "\n\n")
    print("Saved generated_samples.txt")

    return model, train_data, val_data, vocab_size, ch2idx, idx2ch

# -------------------------------------------------------------
# Task 5: Analysis and Ablations
# -------------------------------------------------------------
# PHASE2
# TODO (Student: run_p2_task_5a_heatmaps):
def run_p2_task_5a_heatmaps(model, ch2idx, idx2ch):
    print("\n--- Running Task 5a: Attention Heatmaps ---")
    sentence = "To be, or not to be, that is the question:"
    token_ids = torch.tensor([[ch2idx[c] for c in sentence]], dtype=torch.long, device=DEVICE)

    model.eval()
    with torch.no_grad():
        _, weights = model(token_ids, use_causal_mask=True, return_weights_layer=0)
        # weights shape: (B, num_heads, T, T) = (1, 4, T, T)
        attn = weights[0].cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    tokens = list(sentence)

    for i, head_idx in enumerate([0, 1]):
        im = axes[i].imshow(attn[head_idx], cmap='viridis', aspect='auto')
        axes[i].set_title(f'Layer 1 Head {head_idx+1}')
        axes[i].set_xlabel('Key Token Index')
        axes[i].set_ylabel('Query Token Index')
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    plt.suptitle('Attention Heatmaps (Layer 1, Heads 1 & 2)', fontsize=13)
    plt.tight_layout()
    plt.savefig('figures/p2_attention_heatmaps.png', dpi=150)
    plt.close()
    print("Saved figures/p2_attention_heatmaps.png")

# PHASE2
# TODO (Student: run_p2_task_5b_heads_ablation):
def run_p2_task_5b_heads_ablation(train_data, val_data, vocab_size):
    print("\n--- Running Task 5b: Head Count Ablation (H in {1, 4, 8}) ---")
    heads_list = [1, 4, 8]
    val_histories = {}
    criterion = nn.CrossEntropyLoss()

    for h in heads_list:
        set_seed(3)
        m = CausalTransformerLM(
            vocab_size=vocab_size, d_model=D_MODEL, num_heads=h,
            num_layers=2, d_ff=D_FF, max_len=CONTEXT_LEN, dropout=0.1
        ).to(DEVICE)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        v_losses = []

        m.train()
        for step in range(1, 401):
            xb, yb = get_lm_batch(train_data, 32, CONTEXT_LEN)
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = criterion(m(xb, use_causal_mask=True).view(-1, vocab_size), yb.view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % 50 == 0:
                m.eval()
                with torch.no_grad():
                    vx, vy = get_lm_batch(val_data, 32, CONTEXT_LEN)
                    vx, vy = vx.to(DEVICE), vy.to(DEVICE)
                    vl = criterion(m(vx, use_causal_mask=True).view(-1, vocab_size), vy.view(-1))
                    v_losses.append(vl.item())
                m.train()

        val_histories[f'{h} Head(s)'] = v_losses

    plt.figure(figsize=(7.5, 4.5))
    steps = [50 * i for i in range(1, len(val_histories['1 Head(s)']) + 1)]
    for label, losses in val_histories.items():
        plt.plot(steps, losses, marker='s', label=label)
    plt.xlabel('Step')
    plt.ylabel('Validation Loss (nats)')
    plt.title('Attention Heads Ablation (d=128, H in {1, 4, 8})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('figures/p2_heads_ablation.png', dpi=150)
    plt.close()
    print("Saved figures/p2_heads_ablation.png")

# PHASE2
# TODO (Student: run_p2_task_5c_no_mask_ablation):
def run_p2_task_5c_no_mask_ablation(train_data, val_data, vocab_size, ch2idx, idx2ch):
    print("\n--- Running Task 5c: No-Causal-Mask Ablation ---")
    criterion = nn.CrossEntropyLoss()
    set_seed(3)

    # Train model without causal mask
    m_nomask = CausalTransformerLM(
        vocab_size=vocab_size, d_model=D_MODEL, num_heads=NUM_HEADS,
        num_layers=2, d_ff=D_FF, max_len=CONTEXT_LEN, dropout=0.1
    ).to(DEVICE)
    opt = torch.optim.AdamW(m_nomask.parameters(), lr=1e-3)
    train_losses_nomask = []

    m_nomask.train()
    for step in range(1, 401):
        xb, yb = get_lm_batch(train_data, 32, CONTEXT_LEN)
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        loss = criterion(m_nomask(xb, use_causal_mask=False).view(-1, vocab_size), yb.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 50 == 0:
            train_losses_nomask.append(loss.item())

    # Plot comparison vs masked training
    plt.figure(figsize=(7.5, 4.5))
    steps = [50 * i for i in range(1, len(train_losses_nomask) + 1)]
    plt.plot(steps, train_losses_nomask, marker='x', color='red', label='Unmasked (Information Leakage)')
    plt.xlabel('Step')
    plt.ylabel('Train Loss (nats)')
    plt.title('Effect of Removing Causal Mask (Train Loss Collapses due to Peeking)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('figures/p2_no_mask_ablation.png', dpi=150)
    plt.close()
    print("Saved figures/p2_no_mask_ablation.png")

# PHASE2
# TODO (Student: run_p2_task_5d_context_scaling):
def run_p2_task_5d_context_scaling(vocab_size: int):
    print("\n--- Running Task 5d: Context Length Scaling ---")
    context_lengths = [64, 128, 256, 512]
    times_ms = []
    theoretical_quadratic = [n**2 * D_MODEL / (64**2 * D_MODEL) for n in context_lengths]

    m = CausalTransformerLM(
        vocab_size=vocab_size, d_model=D_MODEL, num_heads=NUM_HEADS,
        num_layers=2, d_ff=D_FF, max_len=512, dropout=0.0
    ).to(DEVICE)
    m.train()
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

    for n in context_lengths:
        x = torch.randint(0, vocab_size, (16, n), device=DEVICE)
        # Warmup
        for _ in range(3):
            out = m(x)
            loss = out.sum()
            loss.backward()
            optimizer.zero_grad()

        t0 = time.time()
        iters = 10
        for _ in range(iters):
            out = m(x)
            loss = out.sum()
            loss.backward()
            optimizer.zero_grad()
        elapsed = (time.time() - t0) / iters * 1000 # ms
        times_ms.append(elapsed)
        print(f"Context {n}: {elapsed:.2f} ms per step")

    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    color = '#2b5c8f'
    ax1.set_xlabel('Context Length (n)')
    ax1.set_ylabel('Measured Step Time (ms)', color=color)
    ax1.plot(context_lengths, times_ms, marker='o', color=color, linewidth=2, label='Measured Time (ms)')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color2 = '#d95f02'
    ax2.set_ylabel('Normalized O(n^2 d) Complexity', color=color2)
    ax2.plot(context_lengths, theoretical_quadratic, marker='s', color=color2, linestyle='--', label='O(n^2 d) Curve')
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title('Transformer Context Scaling vs O(n^2 d) Complexity')
    ax1.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('figures/p2_context_scaling.png', dpi=150)
    plt.close()
    print("Saved figures/p2_context_scaling.png")

if __name__ == '__main__':
    unit_test_attention()
    verify_encoder_block_parameters()
    run_p2_task_a(epochs=8)
    model_lm, tr_data, v_data, v_size, ch2idx, idx2ch = run_p2_task_b(max_steps=1000)
    run_p2_task_5a_heatmaps(model_lm, ch2idx, idx2ch)
    run_p2_task_5b_heads_ablation(tr_data, v_data, v_size)
    run_p2_task_5c_no_mask_ablation(tr_data, v_data, v_size, ch2idx, idx2ch)
    run_p2_task_5d_context_scaling(v_size)
    print("\nAll Problem 2 tasks completed successfully!")
