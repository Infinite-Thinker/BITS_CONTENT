# DNN Programming Assignment 2 — RNN, LSTM, GRU and Transformer

**Student ID:** `2026AJ05253`  
**Course:** Deep Neural Networks (M.Tech AIML, BITS Pilani WILP)

---

## 1. Environment Setup

This project requires Python 3.10+ and PyTorch 2.x. Install the exact dependencies using:

```bash
pip install -r requirements.txt
```

On Google Colab, select **Runtime → Change runtime type → T4 GPU** and verify `torch.cuda.is_available()` returns `True`.

---

## 2. Student Variant & Documented Assumptions

Per the assignment instructions (*"In case of any ambiguity or missing information, you may make reasonable assumptions and proceed accordingly. Clearly mention any assumptions you make"*), the following variant values are used for Student ID `2026AJ05253` (last digit `3`, odd):

- **Three Random Seeds:** `[3, 13, 23]` (derived from the last digit `3` of Student ID `2026AJ05253`).
  - *Note:* For the Adding Problem dataset splits (Problem 1), the fixed data generation seeds specified in the handout (`seed=0` for 10,000 train, `seed=1` for 1,000 val, `seed=2` for 1,000 test) are used so the dataset is identical across students, while `[3, 13, 23]` are used for model initialization and training runs.
- **Problem 1 Hidden Size (`h`):** `h = 64` (matches Problem 1 Task 2 and the midpoint of the Task 5b sweep `{16, 32, 64, 128}`).
- **Problem 1 Individual Ablation (Task 5c):** Since the last digit `3` is **odd**, the ablation compares **LSTM forget-gate bias initialized to `0` vs. `1` at `T = 200`**.
- **Problem 2 Model Dimensions:** `d = 128`, `d_ff = 4d = 512`, `heads = 4`, `layers = 4`, `context_len = 128` (matches Problem 2 Task 2 and Task 4).
- **Problem 2 Task 1 (3-Token Example):** Uses a hand-checkable 3-token, 2-dimensional example (`n = 3, d_k = 2`) where intermediate attention scores, softmax weights, and outputs are computed both by hand and in code.

---

## 3. Repository Structure

```text
2026AJ05253_DNN_Assignment/
  README.md
  requirements.txt
  report.pdf
  results.csv
  problem1/
    problem1.ipynb
    figures/
  problem2/
    problem2.ipynb
    generated_samples.txt
    figures/
```

---

## 4. How to Run the Notebooks

1. **Problem 1 (`problem1/problem1.ipynb`):**
   - Open `problem1/problem1.ipynb` and select **Restart and Run All**.
   - Automatically downloads the UCI HAR dataset into a local temporary folder, generates the synthetic Adding Problem sequences, runs unit tests (`assert` checks) for `MyRNNCell`, `MyLSTMCell`, and `MyGRUCell`, trains all comparison and ablation models, saves plots to `problem1/figures/`, and appends metrics to `../results.csv`.
2. **Problem 2 (`problem2/problem2.ipynb`):**
   - Open `problem2/problem2.ipynb` and select **Restart and Run All**.
   - Runs unit tests (`assert` checks) for Scaled Dot-Product Attention, Multi-Head Attention, Positional Encoding, and the Transformer Encoder Block, trains the HAR Transformer Classifier (Task A) and Tiny Shakespeare Causal Language Model (Task B), writes generated text samples to `problem2/generated_samples.txt`, saves plots to `problem2/figures/`, and appends metrics to `../results.csv`.

---

## 5. Approximate Run Time & Known Notes

- **Total Run Time (Colab T4 GPU):** ~15–20 minutes total (~5–7 minutes for `problem1.ipynb` and ~10–12 minutes for `problem2.ipynb`).
- **Known Limitations:** None. All unit tests pass within `1e-5` tolerance and all figures and CSV entries regenerate automatically from the fixed seeds.
