import os
import sys
import time
import math
import random
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

os.makedirs('figures', exist_ok=True)
RESULTS_CSV_PATH = '../results.csv'
STUDENT_ID = '2026AJ05253'
SEEDS = [3, 13, 23]
HIDDEN_SIZE = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

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
# Task 1 (P1.1): Data Pipelines
# -------------------------------------------------------------
HAR_DIR = 'data_har'
HAR_DATASET_FOLDER = os.path.join(HAR_DIR, 'UCI HAR Dataset')
CHANNELS = [
    'body_acc_x', 'body_acc_y', 'body_acc_z',
    'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
    'total_acc_x', 'total_acc_y', 'total_acc_z'
]
ACTIVITY_NAMES = ['Walking', 'Walking Upstairs', 'Walking Downstairs', 'Sitting', 'Standing', 'Laying']

def download_and_extract_har():
    if os.path.exists(HAR_DATASET_FOLDER):
        return
    os.makedirs(HAR_DIR, exist_ok=True)
    zip_path = os.path.join(HAR_DIR, 'har.zip')
    url = 'https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp, open(zip_path, 'wb') as f:
        f.write(resp.read())
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(HAR_DIR)
    inner_zip = os.path.join(HAR_DIR, 'UCI HAR Dataset.zip')
    if os.path.exists(inner_zip):
        with zipfile.ZipFile(inner_zip, 'r') as zf:
            zf.extractall(HAR_DIR)

# PHASE2
# TODO (Student: load_har_raw):
def load_har_raw():
    download_and_extract_har()
    def load_signals(split):
        signal_data = []
        for ch in CHANNELS:
            fn = f'{ch}_{split}.txt'
            fp = os.path.join(HAR_DATASET_FOLDER, split, 'Inertial Signals', fn)
            arr = np.loadtxt(fp, dtype=np.float32) # (N, 128)
            signal_data.append(arr[:, :, np.newaxis])
        return np.concatenate(signal_data, axis=2) # (N, 128, 9)

    X_train_raw = load_signals('train')
    y_train_raw = np.loadtxt(os.path.join(HAR_DATASET_FOLDER, 'train', 'y_train.txt'), dtype=np.int64) - 1
    sub_train = np.loadtxt(os.path.join(HAR_DATASET_FOLDER, 'train', 'subject_train.txt'), dtype=np.int64)

    X_test_raw = load_signals('test')
    y_test_raw = np.loadtxt(os.path.join(HAR_DATASET_FOLDER, 'test', 'y_test.txt'), dtype=np.int64) - 1
    sub_test = np.loadtxt(os.path.join(HAR_DATASET_FOLDER, 'test', 'subject_test.txt'), dtype=np.int64)

    return X_train_raw, y_train_raw, sub_train, X_test_raw, y_test_raw, sub_test

# PHASE2
# TODO (Student: get_har_splits):
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

    # Per-channel mean & std from training subset ONLY
    mean = X_tr.mean(axis=(0, 1), keepdims=True)
    std = X_tr.std(axis=(0, 1), keepdims=True)
    std[std == 0] = 1.0

    X_tr = (X_tr - mean) / std
    X_val = (X_val - mean) / std
    X_te = (X_te - mean) / std

    return (torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long),
            torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long),
            torch.tensor(X_te, dtype=torch.float32), torch.tensor(y_te, dtype=torch.long))

# PHASE2
# TODO (Student: plot_har_class_balance):
def plot_har_class_balance(y_train_raw, y_test_raw):
    train_counts = np.bincount(y_train_raw, minlength=6)
    test_counts = np.bincount(y_test_raw, minlength=6)
    x = np.arange(6)
    width = 0.35

    plt.figure(figsize=(9, 4.5))
    plt.bar(x - width/2, train_counts, width, label='Train (7352)', color='#2b5c8f')
    plt.bar(x + width/2, test_counts, width, label='Test (2947)', color='#d95f02')
    plt.xticks(x, ACTIVITY_NAMES, rotation=20, ha='right')
    plt.ylabel('Window Count')
    plt.title('UCI HAR Class Balance Distribution')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('figures/p1_har_class_balance.png', dpi=150)
    plt.close()
    print('Saved figures/p1_har_class_balance.png')

# Adding problem generator
def make_adding(n, T, seed):
    rng = np.random.default_rng(seed)
    x = np.zeros((n, T, 2), dtype=np.float32)
    x[:, :, 0] = rng.random((n, T))
    i1 = rng.integers(0, T // 2, n)
    i2 = rng.integers(T // 2, T, n)
    x[np.arange(n), i1, 1] = 1
    x[np.arange(n), i2, 1] = 1
    y = x[np.arange(n), i1, 0] + x[np.arange(n), i2, 0]
    return x, y.astype(np.float32)

# PHASE2
# TODO (Student: test_make_adding):
def test_make_adding():
    n_test, T_test = 500, 100
    x, y = make_adding(n_test, T_test, seed=42)
    # 1. Exactly 2 markers
    marker_sums = x[:, :, 1].sum(axis=1)
    assert np.allclose(marker_sums, 2.0), 'Marker sum != 2.0'
    # 2. One in each half
    first_half_sum = x[:, :T_test // 2, 1].sum(axis=1)
    second_half_sum = x[:, T_test // 2:, 1].sum(axis=1)
    assert np.allclose(first_half_sum, 1.0), 'First half marker count != 1.0'
    assert np.allclose(second_half_sum, 1.0), 'Second half marker count != 1.0'
    # 3. y equals sum of marked channel 0 values
    computed_y = (x[:, :, 0] * x[:, :, 1]).sum(axis=1)
    assert np.allclose(y, computed_y, atol=1e-6), 'Target does not match sum of marked values'
    print('make_adding unit test passed successfully!')

# -------------------------------------------------------------
# Task 2 (P1.2): Cells from Scratch
# -------------------------------------------------------------
# PHASE2
# TODO (Phase 2 Step 1 - Student to write):
class MyRNNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.empty(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        k = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            nn.init.uniform_(p, -k, k)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor = None) -> torch.Tensor:
        if h_prev is None:
            h_prev = torch.zeros(x_t.size(0), self.hidden_size, device=x_t.device, dtype=x_t.dtype)
        return torch.tanh(x_t @ self.weight_ih.T + self.bias_ih + h_prev @ self.weight_hh.T + self.bias_hh)

# PHASE2
# TODO (Phase 2 Step 2 - Student to write):
class MyLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(4 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        k = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            nn.init.uniform_(p, -k, k)

    def forward(self, x_t: torch.Tensor, state: tuple = None) -> tuple:
        if state is None:
            h_prev = torch.zeros(x_t.size(0), self.hidden_size, device=x_t.device, dtype=x_t.dtype)
            c_prev = torch.zeros(x_t.size(0), self.hidden_size, device=x_t.device, dtype=x_t.dtype)
        else:
            h_prev, c_prev = state

        gates = x_t @ self.weight_ih.T + self.bias_ih + h_prev @ self.weight_hh.T + self.bias_hh
        i, f, g, o = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)
        return h_t, c_t

# PHASE2
# TODO (Phase 2 Step 3 - Student to write):
class MyGRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(3 * hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(3 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        k = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            nn.init.uniform_(p, -k, k)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor = None) -> torch.Tensor:
        if h_prev is None:
            h_prev = torch.zeros(x_t.size(0), self.hidden_size, device=x_t.device, dtype=x_t.dtype)

        gi = x_t @ self.weight_ih.T + self.bias_ih
        gh = h_prev @ self.weight_hh.T + self.bias_hh
        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)

        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        h_t = (1 - z) * n + z * h_prev
        return h_t

# PHASE2
# TODO (Student: verify_recurrent_cells):
def verify_recurrent_cells():
    d, h = 9, 64
    B, T = 4, 10
    x = torch.randn(B, T, d)

    # 1. RNN
    rnn = nn.RNN(d, h, batch_first=True)
    my_rnn = MyRNNCell(d, h)
    with torch.no_grad():
        my_rnn.weight_ih.copy_(rnn.weight_ih_l0)
        my_rnn.weight_hh.copy_(rnn.weight_hh_l0)
        my_rnn.bias_ih.copy_(rnn.bias_ih_l0)
        my_rnn.bias_hh.copy_(rnn.bias_hh_l0)
    torch_rnn_out, _ = rnn(x)
    outs = []
    ht = None
    for t in range(T):
        ht = my_rnn(x[:, t, :], ht)
        outs.append(ht.unsqueeze(1))
    diff_rnn = (torch.cat(outs, dim=1) - torch_rnn_out).abs().max().item()
    assert diff_rnn < 1e-5, f'RNN diff: {diff_rnn}'

    # 2. LSTM
    lstm = nn.LSTM(d, h, batch_first=True)
    my_lstm = MyLSTMCell(d, h)
    with torch.no_grad():
        my_lstm.weight_ih.copy_(lstm.weight_ih_l0)
        my_lstm.weight_hh.copy_(lstm.weight_hh_l0)
        my_lstm.bias_ih.copy_(lstm.bias_ih_l0)
        my_lstm.bias_hh.copy_(lstm.bias_hh_l0)
    torch_lstm_out, _ = lstm(x)
    outs = []
    state = None
    for t in range(T):
        state = my_lstm(x[:, t, :], state)
        outs.append(state[0].unsqueeze(1))
    diff_lstm = (torch.cat(outs, dim=1) - torch_lstm_out).abs().max().item()
    assert diff_lstm < 1e-5, f'LSTM diff: {diff_lstm}'

    # 3. GRU
    gru = nn.GRU(d, h, batch_first=True)
    my_gru = MyGRUCell(d, h)
    with torch.no_grad():
        my_gru.weight_ih.copy_(gru.weight_ih_l0)
        my_gru.weight_hh.copy_(gru.weight_hh_l0)
        my_gru.bias_ih.copy_(gru.bias_ih_l0)
        my_gru.bias_hh.copy_(gru.bias_hh_l0)
    torch_gru_out, _ = gru(x)
    outs = []
    ht = None
    for t in range(T):
        ht = my_gru(x[:, t, :], ht)
        outs.append(ht.unsqueeze(1))
    diff_gru = (torch.cat(outs, dim=1) - torch_gru_out).abs().max().item()
    assert diff_gru < 1e-5, f'GRU diff: {diff_gru}'

    print(f'Verification passed! Diffs -> RNN: {diff_rnn:.2e}, LSTM: {diff_lstm:.2e}, GRU: {diff_gru:.2e}')
    # Parameter counts check
    print('Hand derived (1 bias): RNN=4,736, LSTM=18,944, GRU=14,208')
    print(f'PyTorch numel (2 bias): RNN={sum(p.numel() for p in rnn.parameters())}, '
          f'LSTM={sum(p.numel() for p in lstm.parameters())}, '
          f'GRU={sum(p.numel() for p in gru.parameters())}')

if __name__ == '__main__':
    print('Testing P1 data pipeline & cell verification...')
    X_tr_r, y_tr_r, sub_tr, X_te_r, y_te_r, sub_te = load_har_raw()
    print('HAR Raw Shapes:', X_tr_r.shape, X_te_r.shape)
    plot_har_class_balance(y_tr_r, y_te_r)
    test_make_adding()
    verify_recurrent_cells()

# -------------------------------------------------------------
# Task 3 (P1.3): HAR Classification
# -------------------------------------------------------------
# PHASE2
# TODO (Student: HARClassifier):
class HARClassifier(nn.Module):
    def __init__(self, cell_type: str, input_size: int = 9, hidden_size: int = 64, num_classes: int = 6):
        super().__init__()
        self.cell_type = cell_type.upper()
        if self.cell_type == 'RNN':
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        elif self.cell_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        elif self.cell_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        else:
            raise ValueError(f'Unknown cell type: {cell_type}')
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        # Classify from last hidden state h_T
        h_T = out[:, -1, :]
        return self.fc(h_T)

# PHASE2
# TODO (Student: run_p1_har_experiments):
def run_p1_har_experiments(X_tr_r, y_tr_r, sub_tr, X_te_r, y_te_r, epochs: int = 10, batch_size: int = 64):
    models = ['RNN', 'LSTM', 'GRU']
    har_results = []
    best_overall_acc = -1.0
    best_cm = None
    best_model_name = ''

    for m_name in models:
        accs, f1s, times_per_epoch = [], [], []
        model_param_count = 0

        for seed in SEEDS:
            set_seed(seed)
            X_tr, y_tr, X_val, y_val, X_te, y_te = get_har_splits(seed, X_tr_r, y_tr_r, sub_tr, X_te_r, y_te_r)
            train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=batch_size, shuffle=False)

            model = HARClassifier(m_name, input_size=9, hidden_size=HIDDEN_SIZE).to(DEVICE)
            model_param_count = sum(p.numel() for p in model.parameters())
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            epoch_durations = []
            for ep in range(epochs):
                t0 = time.time()
                model.train()
                for bx, by in train_loader:
                    bx, by = bx.to(DEVICE), by.to(DEVICE)
                    optimizer.zero_grad()
                    out = model(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    optimizer.step()
                epoch_durations.append(time.time() - t0)

            # Evaluate on test set
            model.eval()
            all_preds, all_targets = [], []
            with torch.no_grad():
                for bx, by in test_loader:
                    bx = bx.to(DEVICE)
                    preds = model(bx).argmax(dim=-1).cpu().numpy()
                    all_preds.extend(preds)
                    all_targets.extend(by.numpy())

            all_preds = np.array(all_preds)
            all_targets = np.array(all_targets)
            acc = accuracy_score(all_targets, all_preds)
            macro_f1 = f1_score(all_targets, all_preds, average='macro')
            avg_sec = np.mean(epoch_durations)

            accs.append(acc)
            f1s.append(macro_f1)
            times_per_epoch.append(avg_sec)

            if acc > best_overall_acc:
                best_overall_acc = acc
                best_cm = confusion_matrix(all_targets, all_preds)
                best_model_name = f'{m_name} (seed {seed})'

            log_result(
                problem='Problem1_HAR',
                model=m_name,
                seed=seed,
                seq_len=128,
                test_acc=f'{acc:.4f}',
                macro_f1=f'{macro_f1:.4f}',
                params=model_param_count,
                sec_per_epoch=f'{avg_sec:.2f}',
                notes=f'epochs={epochs}, lr=1e-3'
            )

        mean_acc, std_acc = np.mean(accs), np.std(accs, ddof=1)
        mean_f1, std_f1 = np.mean(f1s), np.std(f1s, ddof=1)
        mean_sec = np.mean(times_per_epoch)
        har_results.append({
            'Model': m_name,
            'Params': model_param_count,
            'Test Acc (mean ± std)': f'{mean_acc*100:.2f}% ± {std_acc*100:.2f}%',
            'Macro F1 (mean ± std)': f'{mean_f1:.4f} ± {std_f1:.4f}',
            'Sec/Epoch': f'{mean_sec:.2f}s'
        })

    # Plot & Save Confusion Matrix of Best Model
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=best_cm, display_labels=ACTIVITY_NAMES)
    disp.plot(ax=ax, cmap='Blues', colorbar=False, xticks_rotation=35)
    plt.title(f'Confusion Matrix: Best HAR Model ({best_model_name}, Acc: {best_overall_acc*100:.2f}%)')
    plt.tight_layout()
    plt.savefig('figures/p1_har_confusion_matrix.png', dpi=150)
    plt.close()
    print('Saved figures/p1_har_confusion_matrix.png')

    har_df = pd.DataFrame(har_results)
    print('\n=== HAR Classification Results ===')
    print(har_df.to_string(index=False))
    return har_df

# -------------------------------------------------------------
# Task 4 (P1.4): Adding Problem
# -------------------------------------------------------------
# PHASE2
# TODO (Student: AddingRegressor):
class AddingRegressor(nn.Module):
    def __init__(self, cell_type: str, input_size: int = 2, hidden_size: int = 64):
        super().__init__()
        self.cell_type = cell_type.upper()
        if self.cell_type == 'RNN':
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        elif self.cell_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        elif self.cell_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        else:
            raise ValueError(f'Unknown cell type: {cell_type}')
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        h_T = out[:, -1, :]
        return self.fc(h_T).squeeze(-1)

# PHASE2
# TODO (Student: run_p1_adding_experiments):
def run_p1_adding_experiments(T_list=(50, 100, 200, 400), epochs: int = 10, batch_size: int = 128):
    models = ['RNN', 'LSTM', 'GRU']
    adding_summary = []
    curves_T200 = {m: [] for m in models}
    mse_vs_T = {m: [] for m in models}

    for T in T_list:
        # Exact handout seeds: 0 train, 1 val, 2 test
        X_tr, y_tr = make_adding(10000, T, seed=0)
        X_val, y_val = make_adding(1000, T, seed=1)
        X_te, y_te = make_adding(1000, T, seed=2)

        train_loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(TensorDataset(torch.tensor(X_te), torch.tensor(y_te)), batch_size=batch_size, shuffle=False)

        for m_name in models:
            mses, solved_count = [], 0
            loss_histories = []

            for seed in SEEDS:
                set_seed(seed)
                model = AddingRegressor(m_name, input_size=2, hidden_size=HIDDEN_SIZE).to(DEVICE)
                params = sum(p.numel() for p in model.parameters())
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

                val_losses = []
                durations = []
                for ep in range(epochs):
                    t0 = time.time()
                    model.train()
                    for bx, by in train_loader:
                        bx, by = bx.to(DEVICE), by.to(DEVICE)
                        optimizer.zero_grad()
                        pred = model(bx)
                        loss = criterion(pred, by)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    durations.append(time.time() - t0)

                    # Validation loss
                    model.eval()
                    v_loss = 0.0
                    with torch.no_grad():
                        for bx, by in val_loader:
                            bx, by = bx.to(DEVICE), by.to(DEVICE)
                            v_loss += criterion(model(bx), by).item() * len(by)
                    val_losses.append(v_loss / len(X_val))

                loss_histories.append(val_losses)

                # Test evaluation
                model.eval()
                test_loss = 0.0
                with torch.no_grad():
                    for bx, by in test_loader:
                        bx, by = bx.to(DEVICE), by.to(DEVICE)
                        test_loss += criterion(model(bx), by).item() * len(by)
                test_mse = test_loss / len(X_te)
                mses.append(test_mse)
                if test_mse < 0.05:
                    solved_count += 1

                log_result(
                    problem='Problem1_Adding',
                    model=m_name,
                    seed=seed,
                    seq_len=T,
                    test_mse=f'{test_mse:.5f}',
                    params=params,
                    sec_per_epoch=f'{np.mean(durations):.2f}',
                    notes=f'baseline=0.167, solved={test_mse < 0.05}'
                )

            mean_mse = np.mean(mses)
            std_mse = np.std(mses, ddof=1) if len(mses) > 1 else 0.0
            mse_vs_T[m_name].append(mean_mse)

            if T == 200:
                curves_T200[m_name] = np.mean(loss_histories, axis=0)

            adding_summary.append({
                'T': T,
                'Model': m_name,
                'Test MSE (mean ± std)': f'{mean_mse:.4f} ± {std_mse:.4f}',
                'Solved Runs (<0.05)': f'{solved_count}/3'
            })

    # Plot 1: Learning curves at T = 200
    plt.figure(figsize=(7.5, 4.5))
    for m_name in models:
        plt.plot(range(1, epochs + 1), curves_T200[m_name], marker='o', label=m_name)
    plt.axhline(0.167, color='red', linestyle='--', label='Baseline (~0.167)')
    plt.axhline(0.05, color='green', linestyle=':', label='Solved Threshold (0.05)')
    plt.xlabel('Epoch')
    plt.ylabel('Validation MSE')
    plt.title('Adding Problem: Learning Curves at T = 200')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('figures/p1_adding_curves_T200.png', dpi=150)
    plt.close()
    print('Saved figures/p1_adding_curves_T200.png')

    # Plot 2: Test MSE vs T
    plt.figure(figsize=(7.5, 4.5))
    for m_name in models:
        plt.plot(T_list, mse_vs_T[m_name], marker='s', label=m_name)
    plt.axhline(0.167, color='red', linestyle='--', label='Baseline (~0.167)')
    plt.axhline(0.05, color='green', linestyle=':', label='Solved Threshold (0.05)')
    plt.xlabel('Sequence Length (T)')
    plt.ylabel('Test MSE')
    plt.title('Adding Problem: Test MSE vs. Sequence Length T')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('figures/p1_adding_mse_vs_T.png', dpi=150)
    plt.close()
    print('Saved figures/p1_adding_mse_vs_T.png')

    df_add = pd.DataFrame(adding_summary)
    print('\n=== Adding Problem Summary ===')
    print(df_add.to_string(index=False))
    return df_add

# -------------------------------------------------------------
# Task 5 (P1.5): Gradients, Ablations, and Analysis
# -------------------------------------------------------------
# PHASE2
# TODO (Student: Task 5a - Gradient Norms at T=100):
def run_gradient_norm_analysis():
    T = 100
    X_tr, y_tr = make_adding(100, T, seed=0)
    bx = torch.tensor(X_tr, dtype=torch.float32).to(DEVICE)
    by = torch.tensor(y_tr, dtype=torch.float32).to(DEVICE)
    models = ['RNN', 'LSTM', 'GRU']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for idx, trained_status in enumerate(['Before Training', 'After Training']):
        ax = axes[idx]
        for m_name in models:
            set_seed(42)
            model = AddingRegressor(m_name, input_size=2, hidden_size=HIDDEN_SIZE).to(DEVICE)

            if trained_status == 'After Training':
                opt = torch.optim.Adam(model.parameters(), lr=1e-3)
                crit = nn.MSELoss()
                for _ in range(5):
                    pred = model(bx)
                    loss = crit(pred, by)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

            # Forward pass through recurrent cell with hidden state tracking
            model.eval()
            out, _ = model.rnn(bx) # (B, T, h)
            out.retain_grad()
            h_T = out[:, -1, :]
            pred = model.fc(h_T).squeeze(-1)
            loss = nn.MSELoss()(pred, by)
            model.zero_grad()
            loss.backward()

            # out.grad shape: (B, T, h)
            grad_norms = out.grad.norm(dim=-1).mean(dim=0).cpu().numpy() # (T,)
            ax.plot(range(1, T + 1), grad_norms, label=m_name)

        ax.set_yscale('log')
        ax.set_xlabel('Time Step t')
        ax.set_ylabel('||∂L / ∂h_t||')
        ax.set_title(f'Gradient Norms ({trained_status})')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

    plt.tight_layout()
    plt.savefig('figures/p1_gradient_norms_T100.png', dpi=150)
    plt.close()
    print('Saved figures/p1_gradient_norms_T100.png')

# PHASE2
# TODO (Student: Task 5b - Hidden-Size Sweep):
def run_hidden_size_sweep(X_tr_r, y_tr_r, sub_tr, X_te_r, y_te_r):
    h_values = [16, 32, 64, 128]
    sweep_results = {'LSTM': [], 'GRU': []}

    set_seed(3)
    X_tr, y_tr, _, _, X_te, y_te = get_har_splits(3, X_tr_r, y_tr_r, sub_tr, X_te_r, y_te_r)
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=64, shuffle=False)

    for m_name in ['LSTM', 'GRU']:
        for h in h_values:
            model = HARClassifier(m_name, input_size=9, hidden_size=h).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            crit = nn.CrossEntropyLoss()

            for _ in range(8):
                model.train()
                for bx, by in train_loader:
                    bx, by = bx.to(DEVICE), by.to(DEVICE)
                    optimizer.zero_grad()
                    crit(model(bx), by).backward()
                    optimizer.step()

            model.eval()
            correct = 0
            with torch.no_grad():
                for bx, by in test_loader:
                    bx, by = bx.to(DEVICE), by.to(DEVICE)
                    correct += (model(bx).argmax(dim=-1) == by).sum().item()
            acc = correct / len(X_te)
            params = sum(p.numel() for p in model.parameters())
            sweep_results[m_name].append((h, acc * 100, params))

    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    for m_name in ['LSTM', 'GRU']:
        hs = [x[0] for x in sweep_results[m_name]]
        accs = [x[1] for x in sweep_results[m_name]]
        ax1.plot(hs, accs, marker='o', label=f'{m_name} Accuracy')
    ax1.set_xlabel('Hidden Size (h)')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('UCI HAR Hidden-Size Sweep {16, 32, 64, 128}')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('figures/p1_hidden_size_sweep.png', dpi=150)
    plt.close()
    print('Saved figures/p1_hidden_size_sweep.png')

# PHASE2
# TODO (Student: Task 5c - Forget-Gate Bias Ablation for Odd ID 2026AJ05253):
def run_forget_bias_ablation(T: int = 200, epochs: int = 10, batch_size: int = 128):
    # Odd Student ID: LSTM forget-gate bias 0 vs 1 at T=200
    X_tr, y_tr = make_adding(10000, T, seed=0)
    X_val, y_val = make_adding(1000, T, seed=1)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=batch_size, shuffle=False)

    bias_settings = [0.0, 1.0]
    histories = {}

    for b_val in bias_settings:
        set_seed(3)
        model = AddingRegressor('LSTM', input_size=2, hidden_size=HIDDEN_SIZE).to(DEVICE)
        # Initialize forget-gate bias
        # PyTorch LSTM bias order: [i, f, g, o]
        with torch.no_grad():
            h = HIDDEN_SIZE
            # bias_ih: [0:h] = i, [h:2h] = f, [2h:3h] = g, [3h:4h] = o
            model.rnn.bias_ih_l0.data.zero_()
            model.rnn.bias_hh_l0.data.zero_()
            model.rnn.bias_ih_l0.data[h:2*h].fill_(b_val)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = nn.MSELoss()
        v_losses = []

        for ep in range(epochs):
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                pred = model(bx)
                loss = crit(pred, by)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            v_loss = 0.0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(DEVICE), by.to(DEVICE)
                    v_loss += crit(model(bx), by).item() * len(by)
            v_losses.append(v_loss / len(X_val))

        histories[f'Forget Bias = {int(b_val)}'] = v_losses

    plt.figure(figsize=(7.5, 4.5))
    for label, losses in histories.items():
        plt.plot(range(1, epochs + 1), losses, marker='o', label=label)
    plt.axhline(0.167, color='red', linestyle='--', label='Baseline (~0.167)')
    plt.axhline(0.05, color='green', linestyle=':', label='Solved Threshold (0.05)')
    plt.xlabel('Epoch')
    plt.ylabel('Validation MSE')
    plt.title('Odd Student ID Ablation: LSTM Forget-Gate Bias (0 vs. 1) at T = 200')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('figures/p1_ablation_forget_bias.png', dpi=150)
    plt.close()
    print('Saved figures/p1_ablation_forget_bias.png')

# Run remaining tasks
if __name__ == '__main__':
    print('\nRunning Task 3: HAR classification...')
    run_p1_har_experiments(X_tr_r, y_tr_r, sub_tr, X_te_r, y_te_r, epochs=8)
    print('\nRunning Task 4: Adding problem...')
    run_p1_adding_experiments(T_list=(50, 100, 200, 400), epochs=8)
    print('\nRunning Task 5a: Gradient norms...')
    run_gradient_norm_analysis()
    print('\nRunning Task 5b: Hidden-size sweep...')
    run_hidden_size_sweep(X_tr_r, y_tr_r, sub_tr, X_te_r, y_te_r)
    print('\nRunning Task 5c: Forget-bias ablation...')
    run_forget_bias_ablation(T=200, epochs=8)
    print('\nAll Problem 1 tasks finished successfully!')
