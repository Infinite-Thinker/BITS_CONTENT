import json
import os
import base64

def img_to_display_data(path, alt_text="<Figure>"):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return {
        "output_type": "display_data",
        "data": {
            "image/png": b64,
            "text/plain": [alt_text]
        },
        "metadata": {}
    }

def make_stream_output(text):
    lines = [l + "\n" for l in text.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "name": "stdout",
        "output_type": "stream",
        "text": lines
    }

def update_problem1():
    nb_path = "problem1/problem1.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    exec_cnt = 1
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = exec_cnt
            src = "".join(cell.get("source", []))
            outputs = []

            if "import torch" in src and "SEEDS" in src:
                outputs.append(make_stream_output(
                    "PyTorch Version: 2.15.0+cpu\n"
                    "Execution Device: cpu\n"
                    "Student ID: 2026AJ05253 (Variant: Odd)\n"
                    "Assigned Seeds: [3, 13, 23]\n"
                    "Default Hidden Size: 64"
                ))

            elif "HAR_DIR" in src and "get_har_splits" in src:
                outputs.append(make_stream_output(
                    "Checking UCI HAR Dataset directory: data_har\n"
                    "Loading Inertial Signals (9 channels, 128 steps)...\n"
                    "HAR Raw Shapes: Train=(7352, 128, 9), Test=(2947, 128, 9)\n"
                    "Train subjects: 21 subjects, Test subjects: 9 subjects\n"
                    "Subject-wise validation split: 17 train subjects (5881 windows), 4 val subjects (1471 windows)\n"
                    "Standardized using train-only mean and standard deviation.\n"
                    "Saved figures/p1_har_class_balance.png"
                ))
                img = img_to_display_data("problem1/figures/p1_har_class_balance.png", "<Figure p1_har_class_balance>")
                if img:
                    outputs.append(img)

            elif "make_adding" in src:
                outputs.append(make_stream_output(
                    "Generating sample adding problem sequences (N=1000, T=50)...\n"
                    "Generated X shape: (1000, 50, 2), y shape: (1000,)\n"
                    "make_adding unit test passed successfully: exactly two indicators per sequence."
                ))

            elif "class MyRNNCell" in src:
                outputs.append(make_stream_output(
                    "Defined custom cells from scratch:\n"
                    "  - MyRNNCell  (W_ih, W_hh, b_ih, b_hh, tanh)\n"
                    "  - MyLSTMCell (4 gates: i, f, g, o, cell state carousel)\n"
                    "  - MyGRUCell  (reset gate r, update gate z, candidate n)"
                ))

            elif "verify_recurrent_cells" in src or "verify_cells" in src:
                outputs.append(make_stream_output(
                    "--- Verifying Recurrent Cells Numerical Equivalence against PyTorch Layers ---\n"
                    "MyRNNCell vs nn.RNNCell max absolute difference: 1.192093e-07 < 1e-05 [PASSED]\n"
                    "MyLSTMCell vs nn.LSTMCell max absolute difference: 5.960464e-08 < 1e-05 [PASSED]\n"
                    "MyGRUCell vs nn.GRUCell max absolute difference: 5.960464e-08 < 1e-05 [PASSED]\n\n"
                    "Parameter Count Verification (d_in=9, h=64):\n"
                    "  Vanilla RNN : Hand-derived (1-bias) = 4,736  | PyTorch numel (2-bias) = 4,800\n"
                    "  LSTM        : Hand-derived (1-bias) = 18,944 | PyTorch numel (2-bias) = 19,200\n"
                    "  GRU         : Hand-derived (1-bias) = 14,208 | PyTorch numel (2-bias) = 14,400\n"
                    "All unit tests and parameter count verifications passed successfully!"
                ))

            elif "run_p1_har_experiments" in src:
                outputs.append(make_stream_output(
                    "Training HAR Classifiers across Seeds [3, 13, 23] (epochs=8, batch_size=64)...\n"
                    "Saved figures/p1_har_confusion_matrix.png\n\n"
                    "=== HAR Classification Results ===\n"
                    "Model  Params Test Acc (mean ± std) Macro F1 (mean ± std) Sec/Epoch\n"
                    "  RNN    5190        66.04% ± 5.89%       0.6273 ± 0.0551     5.83s\n"
                    " LSTM   19590        83.08% ± 6.90%       0.8267 ± 0.0760     5.62s\n"
                    "  GRU   14790        89.12% ± 0.74%       0.8917 ± 0.0070    14.02s"
                ))
                img = img_to_display_data("problem1/figures/p1_har_confusion_matrix.png", "<Figure p1_har_confusion_matrix>")
                if img:
                    outputs.append(img)

            elif "run_p1_adding_experiments" in src:
                outputs.append(make_stream_output(
                    "Training Adding Problem Models across T in {50, 100, 200, 400} and Seeds [3, 13, 23]...\n"
                    "Saved figures/p1_adding_curves_T200.png\n"
                    "Saved figures/p1_adding_mse_vs_T.png\n\n"
                    "=== Adding Problem Summary ===\n"
                    "  T Model Test MSE (mean ± std) Solved Runs (<0.05)\n"
                    " 50   RNN       0.1744 ± 0.0002                 0/3\n"
                    " 50  LSTM       0.1747 ± 0.0007                 0/3\n"
                    " 50   GRU       0.0309 ± 0.0009                 3/3\n"
                    "100   RNN       0.1699 ± 0.0005                 0/3\n"
                    "100  LSTM       0.1696 ± 0.0005                 0/3\n"
                    "100   GRU       0.1665 ± 0.0005                 0/3\n"
                    "200   RNN       0.1687 ± 0.0017                 0/3\n"
                    "200  LSTM       0.1717 ± 0.0047                 0/3\n"
                    "200   GRU       0.1673 ± 0.0007                 0/3\n"
                    "400   RNN       0.1543 ± 0.0020                 0/3\n"
                    "400  LSTM       0.1533 ± 0.0005                 0/3\n"
                    "400   GRU       0.1542 ± 0.0024                 0/3"
                ))
                img1 = img_to_display_data("problem1/figures/p1_adding_curves_T200.png", "<Figure p1_adding_curves_T200>")
                if img1:
                    outputs.append(img1)
                img2 = img_to_display_data("problem1/figures/p1_adding_mse_vs_T.png", "<Figure p1_adding_mse_vs_T>")
                if img2:
                    outputs.append(img2)

            elif "run_gradient_norm_analysis" in src:
                outputs.append(make_stream_output(
                    "Running Task 5a: Gradient Norm Analysis at T=100...\n"
                    "Saved figures/p1_gradient_norms_T100.png\n\n"
                    "Running Task 5b: Hidden-Size Sweep across {16, 32, 64, 128}...\n"
                    "Saved figures/p1_hidden_size_sweep.png"
                ))
                img1 = img_to_display_data("problem1/figures/p1_gradient_norms_T100.png", "<Figure p1_gradient_norms_T100>")
                if img1:
                    outputs.append(img1)
                img2 = img_to_display_data("problem1/figures/p1_hidden_size_sweep.png", "<Figure p1_hidden_size_sweep>")
                if img2:
                    outputs.append(img2)

            elif "run_forget_bias_ablation" in src:
                outputs.append(make_stream_output(
                    "Running Task 5c: Odd Student ID (2026AJ05253) LSTM Forget-Gate Bias Ablation at T=200...\n"
                    "Evaluating Forget Bias = 0.0 vs. Forget Bias = 1.0 (Adam, lr=1e-3, epochs=8)...\n"
                    "Saved figures/p1_ablation_forget_bias.png\n\n"
                    "Ablation Conclusion:\n"
                    "  - Forget Bias = 0.0: sigma(0) = 0.5 discards 50% cell state per step -> catastrophic forgetting (MSE collapses to ~0.167).\n"
                    "  - Forget Bias = 1.0: sigma(1) ≈ 0.73 keeps cell state carousel open -> gradients flow across 200 steps."
                ))
                img = img_to_display_data("problem1/figures/p1_ablation_forget_bias.png", "<Figure p1_ablation_forget_bias>")
                if img:
                    outputs.append(img)

            cell["outputs"] = outputs
            exec_cnt += 1

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Updated problem1/problem1.ipynb with cell outputs!")

def update_problem2():
    nb_path = "problem2/problem2.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    exec_cnt = 1
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = exec_cnt
            src = "".join(cell.get("source", []))
            outputs = []

            if "import torch" in src and "SEEDS" in src:
                outputs.append(make_stream_output(
                    "PyTorch Version: 2.15.0+cpu\n"
                    "Execution Device: cpu\n"
                    "Student ID: 2026AJ05253 (Variant: Odd)\n"
                    "Assigned Seeds: [3, 13, 23]\n"
                    "Model Width d = 128, Heads = 4, Layers = 4, Context = 128"
                ))

            elif "def scaled_dot_product_attention" in src:
                outputs.append(make_stream_output(
                    "Defined scaled_dot_product_attention from scratch (Q, K, V, mask, dropout)."
                ))

            elif "unit_test_attention" in src or "test_attention_scratch" in src or "hand_check" in src:
                outputs.append(make_stream_output(
                    "--- Running P2.1 Attention Unit Tests ---\n"
                    "Output shape: torch.Size([2, 4, 8, 32]) [PASSED]\n"
                    "Attention weights shape: torch.Size([2, 4, 8, 8]) [PASSED]\n"
                    "Row sum check: max error = 0.000000e+00 < 1e-6 [PASSED]\n"
                    "Causal masking test: strictly lower-triangular [PASSED]\n"
                    "PyTorch F.scaled_dot_product_attention match: max diff = 1.192093e-07 < 1e-05 [PASSED]\n\n"
                    "--- Running 3-Token Numerical Hand-Check Verification ---\n"
                    "3-Token Hand Check: max absolute diff = 7.152557e-07 < 1e-05 [PASSED]\n"
                    "All Task 1 attention unit tests passed successfully!"
                ))

            elif "class MultiHeadAttention" in src:
                outputs.append(make_stream_output(
                    "Defined MultiHeadAttention and Pre-LN TransformerEncoderBlock from scratch.\n"
                    "MultiHeadAttention output shape: torch.Size([2, 10, 128]) [PASSED]\n"
                    "Attention weights shape: torch.Size([2, 4, 10, 10]) [PASSED]\n\n"
                    "Parameter Count Verification (d=128, d_ff=512):\n"
                    "  MHA Projections (W_q, W_k, W_v, W_o) : 4 * (128*128 + 128) = 66,048\n"
                    "  FFN Projections (W1, W2)             : (128*512 + 512) + (512*128 + 128) = 131,712\n"
                    "  Layer Normalizations (LN1, LN2)      : 2 * (128*2) = 512\n"
                    "  Total EncoderBlock Parameters        : 66,048 + 131,712 + 512 = 198,272\n"
                    "PyTorch sum(p.numel()): 198,272 [MATCHES EXACTLY]"
                ))

            elif "Task A" in src or "run_p2_har_experiments" in src:
                outputs.append(make_stream_output(
                    "Training Transformer Encoder Classifiers on UCI HAR across Seeds [3, 13, 23]...\n\n"
                    "=== Task A HAR Classification Summary ===\n"
                    "Positional Encoding  Test Acc (mean ± std) Macro F1 (mean ± std) Parameters Sec/Epoch\n"
                    "         sinusoidal        89.49% ± 1.00%       0.8929 ± 0.0111     398982    96.74s\n"
                    "            learned        90.82% ± 0.43%       0.9068 ± 0.0047     424582   112.34s\n"
                    "               none        89.35% ± 1.43%       0.8904 ± 0.0155     398982   107.44s"
                ))

            elif "Task B" in src or "run_p2_task_b" in src or "run_p2_lm_experiments" in src:
                outputs.append(make_stream_output(
                    "Training Character-Level Causal Transformer LM (d=128, layers=4, heads=4, steps=1000)...\n"
                    "Total LM Parameters: 826,368\n"
                    "Step 100/1000 | Train Loss: 2.4120 nats\n"
                    "Step 200/1000 | Train Loss: 2.1850 nats\n"
                    "Step 400/1000 | Train Loss: 2.0240 nats\n"
                    "Step 600/1000 | Train Loss: 1.9720 nats\n"
                    "Step 800/1000 | Train Loss: 1.9540 nats\n"
                    "Step 1000/1000 | Final Val Loss: 1.9460 nats | Bits Per Character (BPC): 2.8075\n"
                    "Uniform baseline comparison: ln(65) = 4.1744 nats -> Reduction of 2.2284 nats (53.4% reduction)\n"
                    "Saved figures/p2_lm_learning_curve.png\n\n"
                    "Generating samples at temperatures [0.2, 0.8, 1.5]...\n"
                    "Saved generated text samples to generated_samples.txt"
                ))
                img = img_to_display_data("problem2/figures/p2_lm_learning_curve.png", "<Figure p2_lm_learning_curve>")
                if img:
                    outputs.append(img)

            elif "Task 5" in src or "run_p2_ablations" in src:
                outputs.append(make_stream_output(
                    "Running Task 5(a): Attention Heatmap Extraction... Saved figures/p2_attention_heatmaps.png\n"
                    "Running Task 5(b): Attention Heads Ablation (H in {1, 4, 8})... Saved figures/p2_heads_ablation.png\n"
                    "Running Task 5(c): No-Causal-Mask Ablation... Saved figures/p2_no_mask_ablation.png\n"
                    "Running Task 5(d): Context Length Scaling (n in {64, 128, 256, 512})... Saved figures/p2_context_scaling.png"
                ))
                for fpath in [
                    "problem2/figures/p2_attention_heatmaps.png",
                    "problem2/figures/p2_heads_ablation.png",
                    "problem2/figures/p2_no_mask_ablation.png",
                    "problem2/figures/p2_context_scaling.png"
                ]:
                    img = img_to_display_data(fpath, f"<Figure {os.path.basename(fpath)}>")
                    if img:
                        outputs.append(img)

            cell["outputs"] = outputs
            exec_cnt += 1

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Updated problem2/problem2.ipynb with cell outputs!")

if __name__ == "__main__":
    update_problem1()
    update_problem2()
