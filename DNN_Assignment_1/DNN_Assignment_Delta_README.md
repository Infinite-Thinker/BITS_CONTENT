# DNN Assignment — Delta from Template to Solution

## Purpose

This README documents the changes present in `DNN_Assignment_Solution.ipynb` compared with `DNN_Assignment_Template.ipynb`.

The comparison was performed cell-by-cell on the notebook source. It also checks notebook outputs and metadata separately so that **intentional solution changes** are not confused with **Jupyter execution artifacts**.

---

## 1. Overall delta

- Template cells: **24**
- Solution cells: **24**
- Cell order and cell types are unchanged.
- Intentional source changes occur in cells:
  **0, 3, 5, 7, 8, 10, 11, 13, 15, 16, 18, 20**
- Source of the following cells is unchanged:
  **1, 2, 4, 6, 9, 12, 14, 17, 19, 21, 22, 23**
- The solution contains executed outputs that are not present in the template. These are execution artifacts, not additional source-code changes.
- Notebook top-level metadata differs only in the recorded Python language version: template reports **Python 3.8.10**, while the solution reports **Python 3.14.0**. Kernel name/display name remain `python3` / `Python 3 (ipykernel)`.

---

# 2. Cell-by-cell changes

## Cell 0 — Student information

### Template
The template contained four blank Student Name / Student ID pairs and a blank date.

### Solution
The solution:
- Filled the first Student Name.
- Filled the first Student ID.
- Removed the other three unused Student Name / Student ID pairs.
- Filled the date.

This is a submission-header change only; no assignment logic was changed here.

---

## Cell 3 — Dataset definition and problem description

The template contained TODO placeholders for dataset loading and dataset information.

The solution adds the complete dataset setup:

- Loads the Student Dropout dataset from a CSV URL.
- Uses `sep=';'` when reading the CSV.
- Adds a missing-value check:
  `data.isnull().sum().sum()`
- Sets the dataset name to:
  **Predict Students' Dropout and Academic Success**
- Sets the source to:
  **UCI Machine Learning Repository**
- Derives the number of samples from `data.shape[0]`.
- Derives the number of features from `data.shape[1] - 1`.
- Sets the problem type to:
  **multiclass_classification**
- Adds the complete problem statement.
- Sets the primary metric to:
  **recall**
- Adds the metric justification explaining macro-averaged recall for the imbalanced multiclass problem.

### Important execution note
The **source code contains the new missing-value check, but the stored output does not show its result**. This means this cell appears to have been edited after its last execution.

The source is correct, but the notebook should be re-run before submission so the visible output also contains the missing-value result.

---

## Cell 5 — Preprocessing

The template contained preprocessing TODOs.

The solution implements:

1. Feature/target separation:
   - `X = data.drop('Target', axis=1).values`
   - `y_raw = data['Target'].values`

2. Target encoding:
   - Uses `LabelEncoder`.
   - Converts the three target classes into numeric labels.

3. Train/test split:
   - `test_size=0.2`
   - `random_state=42`

4. Feature scaling:
   - Uses `StandardScaler`.
   - Fits the scaler only on training data.
   - Applies the fitted scaler to both train and test data.

5. Stores actual dataset split values:
   - `train_samples = X_train.shape[0]`
   - `test_samples = X_test.shape[0]`
   - `train_test_ratio = 0.8`

---

## Cell 7 — Baseline model

The template provided a generic linear-model skeleton.

The solution specializes it into **Softmax Regression for multiclass classification**.

Changes include:

- Adds a numerically stable `softmax()` implementation.
- Changes the parameter shapes from a single-output linear model to:
  - weights: `(n_features, num_classes)`
  - bias: `(num_classes,)`
- One-hot encodes the target labels.
- Implements the complete forward pass.
- Implements categorical cross-entropy loss.
- Clips probabilities using epsilon for numerical stability.
- Implements the gradient:
  `dZ = y_pred - y_one_hot`
- Implements vectorized weight and bias gradients.
- Implements gradient-descent parameter updates.
- Stores loss at every iteration in `loss_history`.
- Implements prediction using `argmax()` over class probabilities.

This is the main from-scratch baseline implementation added on top of the template.

---

## Cell 8 — Baseline training

The template contained commented-out training code.

The solution:

- Creates the baseline model with learning rate **0.1**.
- Trains for **1000 iterations**.
- Calls `fit(X_train_scaled, y_train)`.
- Generates test predictions.
- Measures training time.
- Stores the actual initial loss.
- Stores the actual final loss.

The stored execution output shows:

- Training time: **0.83 s**
- Initial loss: **1.0986**
- Final loss: **0.5526**

---

## Cell 10 — MLP implementation

The template provided the MLP class structure but left the core implementation as TODOs.

The solution implements the complete MLP:

### Parameter initialization
- Adds parameter initialization for every layer.
- Uses **He initialization** for weights.
- Initializes biases to zero.
- Uses a fixed seed of `42`.

### Activation
- Keeps ReLU for hidden layers.
- Replaces the generic sigmoid output helper with a multiclass **softmax** implementation.

### Forward propagation
- Uses the samples-as-columns convention internally.
- Stores intermediate `Z` and `A` values in `cache`.
- Applies ReLU to hidden layers.
- Applies Softmax to the output layer.

### Backpropagation
- One-hot encodes targets.
- Uses the simplified Softmax + categorical-cross-entropy gradient.
- Computes `dW` and `db` for every layer.
- Propagates gradients backwards through the hidden layers using the ReLU derivative.

### Training
- Implements the full loop:
  1. Forward propagation
  2. Loss calculation
  3. Loss-history update
  4. Backpropagation
  5. Parameter update

### Prediction
- Runs forward propagation.
- Uses `argmax()` to select the predicted class.

This is the largest implementation change in the notebook.

---

## Cell 11 — MLP training

The solution replaces the MLP training TODOs with:

- Architecture:
  **`[36, 16, 8, 3]`**
- Learning rate:
  **0.05**
- Iterations:
  **1000**
- Actual training call.
- Test prediction generation.
- Actual initial/final loss extraction.

Stored execution output:

- Training time: **1.85 s**
- Initial loss: **1.6327**
- Final loss: **0.5309**

---

## Cell 13 — Metrics

The template provided placeholder metric values.

The solution implements metrics from scratch.

### Regression branch
Although this assignment is classification, the generic function retains implementations for:
- MSE
- RMSE
- MAE
- R²

### Classification branch
For each class, the solution calculates:

- True positives
- False positives
- False negatives
- Precision
- Recall
- F1

Then macro-averages precision, recall, and F1 across classes.

Accuracy is calculated from total correct predictions.

No `sklearn.metrics` functions are used.

The solution also:
- Calculates metrics for both models.
- Unpacks them into global variables.
- Initializes unused regression result variables to `0.0`.
- Prints the final metrics.

Stored results:

### Baseline
- Accuracy: **0.7503**
- Precision: **0.6739**
- Recall: **0.6456**
- F1: **0.6492**

### MLP
- Accuracy: **0.7401**
- Precision: **0.6630**
- Recall: **0.6384**
- F1: **0.6427**

---

## Cell 15 — Training-loss visualization

The template contained commented plotting calls.

The solution activates the plots:

- Baseline loss history is plotted.
- MLP loss history is plotted.
- Baseline legend is labelled **Baseline (Softmax)**.
- MLP legend is labelled **MLP**.

The overall plot layout from the template is retained.

---

## Cell 16 — Performance comparison visualization

The template contained only an example/commented implementation.

The solution adds the complete comparison chart:

- Defines metrics:
  - Accuracy
  - Precision
  - Recall
  - F1
- Reads the actual metric variables from both models.
- Creates side-by-side bars.
- Adds model labels.
- Adds axis labels and title.
- Sets the y-axis to `0`–`1.1`.
- Adds legend and grid.

---

## Cell 18 — Analysis

The template contained the analysis questions and TODO placeholder.

The solution replaces that placeholder with the actual analysis discussing:

1. Which model performed better.
2. Accuracy, recall and F1 comparison.
3. Why the simpler model may have generalized better.
4. Training-time comparison.
5. Learning-rate observations.
6. He initialization.
7. General observations about neural networks versus simpler models.

The source currently states:

- Baseline accuracy: **75.03%**
- MLP accuracy: **74.01%**
- Baseline macro recall: **64.56%**
- MLP macro recall: **63.84%**
- Baseline F1: **64.92%**
- MLP F1: **64.27%**
- Baseline training time: **0.83 s**
- MLP training time: **1.85 s**
- Baseline is described as roughly **2.2× faster**.

### Important execution note
The source analysis has the corrected timing values **0.83 s / 1.85 s**, but the stored output of the results cells still contains the **older 0.70 s / 1.92 s** analysis text.

Therefore, the source and stored output are currently inconsistent.

The analysis cell should be executed again, followed by the result cells.

---

## Cell 20 — Assignment-results serialization

The template already provided the overall `get_assignment_results()` structure, but many fields were placeholders.

The solution fills and strengthens the serialization:

### Dataset values
Adds explicit type conversion:
- `str(...)`
- `int(...)`
- `float(...)`

### Baseline
Sets:
- model type = `softmax_regression`
- learning rate = `0.1`
- iterations = `1000`
- actual initial/final loss
- actual training time
- actual classification metrics

### MLP
Sets:
- architecture = `[36, 16, 8, 3]`
- hidden layers = `2`
- learning rate = `0.05`
- iterations = `1000`
- actual initial/final loss
- actual training time
- actual classification metrics

### Analysis
Adds explicit:
- `str(analysis_text)`
- integer analysis word count

### Why the explicit conversions were added
The conversions make the generated results JSON-safe and ensure values such as NumPy integers/floats/booleans are represented as standard Python types.

---

# 3. Cells with no source-code changes

The following cells retain their template source:

- Cell 1 — imports
- Cell 2 — dataset instructions
- Cell 4 — preprocessing instructions
- Cell 6 — baseline-model instructions
- Cell 9 — MLP instructions
- Cell 12 — metrics instructions
- Cell 14 — visualization instructions
- Cell 17 — analysis instructions
- Cell 19 — results instructions
- Cell 21 — final-summary heading/instructions
- Cell 22 — final summary code
- Cell 23 — submission instructions

These cells may have stored execution outputs in the solution, but their source code is unchanged.

---

# 4. Execution/output delta

The solution notebook contains execution outputs that the blank template does not contain.

The important stored results are:

| Item | Solution output |
|---|---:|
| Samples | 4,424 |
| Features | 36 |
| Training samples | 3,539 |
| Test samples | 885 |
| Baseline initial loss | 1.0986 |
| Baseline final loss | 0.5526 |
| Baseline training time | 0.83 s |
| MLP initial loss | 1.6327 |
| MLP final loss | 0.5309 |
| MLP training time | 1.85 s |
| Baseline accuracy | 75.03% |
| Baseline macro precision | 67.39% |
| Baseline macro recall | 64.56% |
| Baseline macro F1 | 64.92% |
| MLP accuracy | 74.01% |
| MLP macro precision | 66.30% |
| MLP macro recall | 63.84% |
| MLP macro F1 | 64.27% |

---

# 5. Critical stale-output issue found

This is important for the final submission.

The **source code** in Cell 18 contains the corrected analysis:

> baseline 0.83 seconds  
> MLP 1.85 seconds  
> roughly 2.2 times faster

However, the stored JSON output in Cells 20 and 22 still contains the older analysis:

> baseline 0.70 seconds  
> MLP 1.92 seconds  
> roughly 2.7 times faster

Also, Cell 3 source contains the new missing-value check, but its stored output does **not** show the missing-value result.

This means the notebook was **not fully re-executed after the latest edits**.

### Required final action

Before submission:

1. Restart the kernel.
2. Run all cells from the beginning.
3. Confirm Cell 3 output shows the missing-value check result.
4. Confirm Cell 18 output is generated from the current analysis source.
5. Confirm Cells 20 and 22 contain the same current analysis text and current training-time values.
6. Confirm there are no execution errors.
7. Save the notebook.

This is more important than any cosmetic difference because the notebook currently contains both old and new information.

---

# 6. Final delta summary

The actual assignment work added on top of the template consists of:

- Student information.
- Dataset loading.
- Dataset identification.
- Problem formulation.
- Primary metric selection and justification.
- Missing-value check.
- Target encoding.
- Train/test split.
- Feature scaling.
- Complete Softmax Regression implementation.
- Softmax and categorical cross-entropy.
- Baseline gradient descent.
- Baseline prediction.
- Complete MLP implementation.
- He initialization.
- ReLU hidden layers.
- Softmax output layer.
- Forward propagation.
- Backpropagation.
- Parameter updates.
- MLP training.
- MLP prediction.
- Manual classification metrics.
- Macro-averaging.
- Training-loss visualization.
- Model-performance comparison visualization.
- Written analysis.
- Population of assignment-result fields.
- Explicit JSON-compatible type conversions.
- Actual result serialization.

The **template structure itself has been preserved**: the notebook still contains the same 24 cells in the same order, with the solution filling the intended TODO sections rather than restructuring the notebook.

---

## Submission-status note

The source-code delta is complete, but the current notebook should **not yet be treated as the final executed artifact** because the stored outputs in Cells 3, 18, 20 and 22 are stale relative to the latest source edits.

A final **Restart → Run All → Save** will synchronize the source, outputs, analysis text, timing values, and missing-value check.
