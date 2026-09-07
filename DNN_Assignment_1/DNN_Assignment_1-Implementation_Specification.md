# Deep Neural Networks – Programming Assignment 1

## Comparing Linear Models and Multi-Layer Perceptrons

**Course:** Deep Neural Networks  
**Assignment Type:** Programming Assignment  
**Submission Format:** Jupyter Notebook (`.ipynb`)  
**Maximum Marks:** 10 (Scaled to 7.5)

---

# 1. Assignment Objective

Implement and compare two machine learning models **from scratch using NumPy**:

1. A baseline linear model
2. A Multi-Layer Perceptron (MLP)

Use a real-world tabular dataset and compare the models on test data.

---

# 2. Required Tasks

The assignment requires the following:

1. Select a valid real-world dataset.
2. Implement a baseline model from scratch.
3. Implement an MLP from scratch.
4. Train both models using gradient descent.
5. Track training loss for both models.
6. Evaluate both models on test data.
7. Create required visualizations.
8. Write an analysis of fewer than 200 words.
9. Complete the provided `get_assignment_results()` function.
10. Ensure the entire notebook runs successfully from start to finish.

---

# 3. Dataset Requirements

## Mandatory Dataset Conditions

The selected dataset must satisfy ALL of the following requirements:

- At least **500 samples**
- At least **5 input features**, excluding the target column
- Must be a tabular dataset
- Features can be:
  - Numeric
  - Categorical
  - Or a combination of both
- Problem type must be either:
  - Regression
  - Binary Classification
  - Multi-class Classification

## Allowed Dataset Sources

Examples:

- UCI Machine Learning Repository
- Kaggle
- OpenML
- Data.gov
- Other public datasets

## Example Datasets Mentioned

- Breast Cancer Wisconsin
- Adult Income
- Abalone
- House Prices
- Credit Card Fraud

## Prohibited Datasets

DO NOT use:

- Image datasets
  - MNIST
  - CIFAR
  - ImageNet
- Text or NLP datasets
- Audio datasets
- Preprocessed datasets from Papers with Code
- Toy datasets with fewer than 500 samples
  - Iris
  - Wine
  - Diabetes

---

# 4. Choose the Problem Type

The implementation depends on the dataset's problem type.

## Option A: Regression

Implement:

### Baseline Model

- Linear Regression
- Gradient Descent
- Implement from scratch

### MLP

- Multi-Layer Perceptron
- Backpropagation
- Gradient Descent
- Implement from scratch

---

## Option B: Binary Classification

Implement:

### Baseline Model

- Logistic Regression
- Gradient Descent
- Implement from scratch

### MLP

- Multi-Layer Perceptron
- Appropriate output activation for binary classification
- Backpropagation
- Gradient Descent

---

## Option C: Multi-class Classification

Implement:

### Baseline Model

- Softmax Regression
- Gradient Descent
- Implement from scratch

### MLP

- Multi-Layer Perceptron
- Softmax output activation
- Backpropagation
- Gradient Descent

---

# 5. Library Restrictions

## Allowed Libraries

You MAY use:

```python
numpy
pandas
matplotlib
seaborn
```

You MAY use specific utilities from `sklearn` ONLY for:

```python
train_test_split
StandardScaler
MinMaxScaler
LabelEncoder
OneHotEncoder
```

---

## Prohibited Libraries

DO NOT use:

```python
sklearn.linear_model
sklearn.neural_network
tensorflow
keras
pytorch
jax
```

DO NOT use:

```python
MLPClassifier
MLPRegressor
LogisticRegression
LinearRegression
RandomForest
XGBoost
```

or any other high-level ML model implementation.

### Important

All model implementations must be written manually using NumPy.

Using prohibited model libraries can result in a **-5 mark penalty**.

---

# 6. Dataset Selection Section

The notebook must include:

## Dataset Description

Describe:

- Dataset name
- Dataset source
- Number of samples
- Number of features
- Target variable
- Feature types
- Problem type

## Problem Type

Explicitly state whether the task is:

```text
Regression
```

or:

```text
Binary Classification
```

or:

```text
Multi-class Classification
```

## Evaluation Metric Justification

Choose an appropriate primary metric and explain why it is suitable for the problem domain.

Examples:

### Regression

Possible metrics:

- MSE
- RMSE
- MAE
- R²

### Classification

Possible metrics:

- Accuracy
- Precision
- Recall
- F1 Score

Example reasoning:

> Recall may be prioritized in medical diagnosis because false negatives can be more costly than false positives.

---

# 7. Data Preprocessing

Perform the following preprocessing steps as required by the selected dataset.

## 7.1 Train/Test Split

Use either:

```text
80% Training / 20% Testing
```

or:

```text
90% Training / 10% Testing
```

Document the selected split ratio.

Use:

```python
train_test_split
```

from sklearn if required.

---

## 7.2 Missing Values

If missing values exist:

- Drop rows, OR
- Fill using mean, median, or another appropriate method

Document the approach used.

---

## 7.3 Categorical Features

If categorical features exist:

- Encode them appropriately.

Allowed utilities include:

```python
LabelEncoder
OneHotEncoder
```

---

## 7.4 Feature Scaling

Scale the input features.

Allowed options:

```python
StandardScaler
```

or:

```python
MinMaxScaler
```

Scaling should be performed correctly to avoid data leakage.

---

# 8. Baseline Model Implementation

## Marks: 3

Implement the baseline model completely from scratch using NumPy.

The exact baseline depends on the problem type:

| Problem Type | Baseline Model |
|---|---|
| Regression | Linear Regression |
| Binary Classification | Logistic Regression |
| Multi-class Classification | Softmax Regression |

---

## Required Class Structure

The provided notebook may contain a template similar to:

```python
class BaselineModel:

    def __init__(
        self,
        learning_rate=0.01,
        n_iterations=1000
    ):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.loss_history = []

    def fit(self, X, y):

        # Initialize weights

        # Gradient descent loop:
        # 1. Forward pass
        # 2. Compute loss
        # 3. Compute gradients
        # 4. Update parameters
        # 5. Store loss

        return self

    def predict(self, X):
        pass
```

---

## Mandatory Baseline Requirements

The implementation must include:

### 1. Parameter Initialization

Initialize:

- Weights
- Bias

with appropriate dimensions.

---

### 2. Forward Pass

Compute predictions.

The exact operation depends on the selected problem type.

---

### 3. Loss Calculation

Use the appropriate loss function.

Examples:

#### Regression

Mean Squared Error.

#### Binary Classification

Binary Cross Entropy.

#### Multi-class Classification

Categorical Cross Entropy.

---

### 4. Gradient Calculation

Compute gradients manually.

Use vectorized NumPy operations wherever possible.

---

### 5. Gradient Descent

Update parameters using:

```text
parameter = parameter - learning_rate * gradient
```

---

### 6. Loss Tracking

This is mandatory.

The model MUST store loss values at every training iteration:

```python
self.loss_history
```

The loss history must be appended during training.

Example expectation:

```python
self.loss_history.append(loss)
```

---

### 7. Prediction Function

Implement:

```python
predict(X)
```

The function should return predictions appropriate for the selected problem type.

---

# 9. Multi-Layer Perceptron Implementation

## Marks: 4

Implement an MLP completely from scratch using NumPy.

---

## Mandatory Requirements

The MLP must have:

- At least one hidden layer
- ReLU activation in hidden layers
- Appropriate output activation
- Forward propagation
- Backpropagation
- Gradient descent optimization
- Loss tracking

---

# 10. MLP Architecture

The architecture should follow:

```text
[input_layer, hidden_layer_1, hidden_layer_2, ..., output_layer]
```

Example:

```python
architecture = [
    input_size,
    hidden_size,
    output_size
]
```

Multiple hidden layers are allowed.

Example:

```python
architecture = [
    input_size,
    32,
    16,
    output_size
]
```

Minimum requirement:

```text
Input Layer
    ↓
At least 1 Hidden Layer
    ↓
Output Layer
```

---

# 11. MLP Class Structure

The notebook may contain a template similar to:

```python
class MLP:

    def __init__(
        self,
        architecture,
        learning_rate=0.01,
        n_iterations=1000
    ):

        self.architecture = architecture
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

        self.loss_history = []

    def initialize_parameters(self):

        # Initialize W and b
        # for every layer

        pass

    def forward_propagation(self, X):

        # Compute activations
        # Store intermediate values
        # required for backpropagation

        pass

    def backward_propagation(self, X, y):

        # Compute gradients
        # using chain rule

        pass

    def fit(self, X, y):

        # Training loop
        # Forward pass
        # Loss calculation
        # Backward pass
        # Parameter update
        # Store loss

        return self

    def predict(self, X):

        pass
```

---

# 12. MLP Forward Propagation Requirements

The forward pass must:

1. Accept input `X`
2. Compute weighted sums for each layer
3. Apply activation functions
4. Store intermediate values
5. Store activations and pre-activations required for backpropagation

Hidden layers must use:

```text
ReLU
```

Conceptually:

```text
Z = A_previous @ W + b
A = ReLU(Z)
```

The output layer activation depends on the problem type.

---

## Regression Output Layer

Use an appropriate regression output.

Typically:

```text
Linear output
```

---

## Binary Classification Output Layer

Use:

```text
Sigmoid
```

---

## Multi-class Classification Output Layer

Use:

```text
Softmax
```

---

# 13. MLP Backpropagation Requirements

Implement backpropagation manually.

The implementation must:

1. Compute output layer gradients.
2. Propagate gradients backward through each hidden layer.
3. Apply the chain rule.
4. Compute gradients for:
   - Weights
   - Biases
5. Use the derivative of ReLU in hidden layers.
6. Update parameters using gradient descent.

Do NOT use automatic differentiation libraries.

---

# 14. Training Loop Requirements

For both models:

1. Initialize parameters.
2. Perform forward propagation.
3. Compute loss.
4. Compute gradients.
5. Update parameters.
6. Store training loss.
7. Repeat for the configured number of iterations.

Both models should show a generally decreasing loss curve.

---

# 15. Evaluation and Comparison

## Marks: 2

Evaluate both models using the appropriate metrics.

---

## Regression Metrics

Calculate:

```text
MSE
RMSE
MAE
R²
```

for:

- Baseline Model
- MLP

---

## Classification Metrics

Calculate:

```text
Accuracy
Precision
Recall
F1 Score
```

for:

- Baseline Model
- MLP

---

# 16. Required Visualizations

Create the following plots.

## Visualization 1: Training Loss Curves

Plot:

```text
Baseline Model Loss
vs
MLP Loss
```

Requirements:

- Clearly labeled axes
- Legend
- Title
- Show training progression

The plot may contain both loss curves for comparison.

---

## Visualization 2: Performance Comparison Bar Chart

Create a bar chart comparing the models.

Example:

```text
                Metric Value
Baseline Model       ███████
MLP                  █████████
```

For classification, compare one or more important metrics.

For regression, compare one or more error metrics and/or R².

---

## Visualization 3: Domain-Specific Plot (Optional)

Additional visualizations are optional.

---

# 17. Required Analysis

Write an analysis of:

```text
LESS THAN 200 WORDS
```

The analysis must answer ALL of the following:

### Question 1

Which model performed better?

### Question 2

By how much did it perform better?

Use actual metric values.

### Question 3

Why did one model outperform the other?

Discuss concepts such as:

- Linear relationships
- Non-linear relationships
- Model capacity
- Hidden layers
- Feature interactions

### Question 4

What was the computational cost difference?

Discuss:

- Training time
- Number of parameters
- Number of iterations
- Complexity of forward propagation
- Complexity of backpropagation

### Question 5

Were there any surprising findings or challenges?

Examples:

- Loss instability
- Learning rate tuning
- Overfitting
- Shape mismatch errors
- Slow training
- Numerical instability

---

# 18. Mandatory `get_assignment_results()` Function

This function is REQUIRED.

Missing this function results in:

```text
-2 Marks
```

The existing function in the provided notebook template must be completed accurately.

## Important Instructions

Do NOT:

- Rename the function
- Delete the function
- Change its expected structure unless the template explicitly allows it

Fill the required results based on the actual model outputs.

Before completing this function:

1. Run the entire notebook.
2. Train both models.
3. Calculate final metrics.
4. Insert the actual values/results expected by the template.

---

# 19. Grading Breakdown

| Component | Marks | Requirements |
|---|---:|---|
| Dataset | 1 | ≥ 500 samples, ≥ 5 features, valid problem |
| Baseline Model | 3 | Correct implementation, decreasing loss, reasonable performance |
| MLP | 4 | Valid architecture, forward pass, backpropagation, decreasing loss |
| Comparison | 2 | Metrics, visualizations, analysis under 200 words |

Total:

```text
10 Marks
```

Scaled to:

```text
7.5 Marks
```

---

# 20. Important Deductions

| Issue | Penalty |
|---|---:|
| Missing `get_assignment_results()` | -2 marks |
| Dataset too small | -1 mark |
| Baseline loss does not decrease | -1 mark |
| MLP loss does not decrease | -1 mark |
| Analysis is 200 words or more | -0.5 mark |
| Prohibited libraries/models | -5 marks |

---

# 21. Implementation Strategy

Follow this implementation order.

## Step 1: Inspect the Provided Notebook Template

Before writing code:

1. Open the provided `.ipynb`.
2. Identify every cell marked:
   ```text
   TODO
   ```
3. Do not remove required sections.
4. Preserve required function names and output formats.

---

## Step 2: Select Dataset

Choose a dataset that:

- Has more than 500 samples
- Has at least 5 features
- Is tabular
- Is reasonably easy to preprocess

Binary classification may be simpler to implement than multi-class classification.

---

## Step 3: Load and Explore Data

Perform:

```python
df.head()
df.shape
df.info()
df.describe()
df.isnull().sum()
```

Document:

- Number of samples
- Number of features
- Missing values
- Target distribution, if classification

---

## Step 4: Preprocess Data

Perform:

1. Handle missing values.
2. Encode categorical columns.
3. Separate features and target.
4. Split into training and testing sets.
5. Scale features.

Important:

```text
Fit scalers only on training data.
Transform both training and testing data.
```

---

## Step 5: Implement Baseline Model

Implement the correct baseline model from scratch.

Verify:

```text
✓ Correct shapes
✓ Loss calculation works
✓ Gradients are correct
✓ Loss decreases
✓ Predictions have correct format
```

---

## Step 6: Implement MLP

Implement:

1. Parameter initialization.
2. Forward propagation.
3. Activation functions.
4. Loss calculation.
5. Backpropagation.
6. Parameter updates.
7. Prediction method.

Verify:

```text
✓ Weight shapes
✓ Bias shapes
✓ Forward pass shapes
✓ Backpropagation shapes
✓ Loss decreases
```

---

## Step 7: Train Both Models

Use suitable:

- Learning rate
- Number of iterations
- MLP architecture

Recommended learning rate starting point:

```text
0.01
```

If training is unstable:

```text
Try 0.001
```

Other suggested values:

```text
0.1
```

Only use a learning rate if training remains stable.

---

## Step 8: Evaluate Models

Calculate all required metrics.

Store results clearly.

Example conceptual structure:

```python
baseline_results = {
    "accuracy": ...,
    "precision": ...,
    "recall": ...,
    "f1": ...
}

mlp_results = {
    "accuracy": ...,
    "precision": ...,
    "recall": ...,
    "f1": ...
}
```

Use the metric structure appropriate to the selected problem type.

---

## Step 9: Create Visualizations

Create:

1. Loss comparison plot.
2. Performance comparison bar chart.

Ensure:

```text
✓ Titles
✓ Axis labels
✓ Legends
✓ Readable output
```

---

## Step 10: Write Final Analysis

Keep the analysis:

```text
< 200 words
```

Use actual results from the experiment.

Do not write conclusions before running the models.

---

## Step 11: Complete `get_assignment_results()`

Use the actual values generated by the notebook.

Verify that:

```text
✓ Function exists
✓ Required keys are correct
✓ Results are accurate
✓ No placeholder values remain
```

---

# 22. Debugging Checklist

If the models do not work:

## Check Data Shapes

Print:

```python
print(X.shape)
print(y.shape)
```

Check shapes during:

- Forward propagation
- Gradient calculation
- Weight updates

---

## Check for NaN or Infinity

Verify:

```python
np.isnan(...)
np.isinf(...)
```

No unexpected NaN or Inf values should exist.

---

## Check Learning Rate

Try:

```text
0.001
0.01
0.1
```

Monitor the loss.

---

## Verify Gradient Descent Update

The update must follow:

```text
parameter = parameter - learning_rate × gradient
```

---

## Verify Activations

Ensure:

- Hidden layers use ReLU.
- Output layer uses the correct activation.
- Activation derivatives are correctly implemented.

---

## Check Loss

Verify:

```text
Loss is computed correctly.
Loss values are being stored.
Loss generally decreases.
```

---

# 23. Efficiency Requirements

Prefer:

```text
Vectorized NumPy operations
```

Avoid unnecessary loops over individual samples.

Prefer operations such as:

```python
X @ W
np.dot(...)
```

instead of manually iterating through every sample.

---

# 24. Random Seeds

Use random seeds for reproducibility.

Example:

```python
np.random.seed(...)
```

If using `train_test_split`, set:

```python
random_state=...
```

Keep the experiment reproducible.

---

# 25. Final Notebook Checklist

Before submission, verify:

## Dataset

- [ ] At least 500 samples
- [ ] At least 5 features
- [ ] Valid tabular dataset
- [ ] Correct problem type documented

## Preprocessing

- [ ] Missing values handled
- [ ] Categorical features encoded
- [ ] Data split performed
- [ ] Features scaled
- [ ] Split ratio documented

## Baseline Model

- [ ] Implemented from scratch
- [ ] No sklearn model used
- [ ] Gradient descent implemented
- [ ] Loss tracked
- [ ] Loss decreases
- [ ] Predict method implemented

## MLP

- [ ] At least one hidden layer
- [ ] ReLU hidden activation
- [ ] Appropriate output activation
- [ ] Forward propagation implemented
- [ ] Backpropagation implemented
- [ ] Gradient descent implemented
- [ ] Loss tracked
- [ ] Loss decreases
- [ ] Predict method implemented

## Evaluation

- [ ] Required metrics calculated
- [ ] Baseline evaluated
- [ ] MLP evaluated
- [ ] Metrics compared

## Visualization

- [ ] Training loss curves
- [ ] Performance comparison bar chart

## Analysis

- [ ] Less than 200 words
- [ ] Which model performed better?
- [ ] How much better?
- [ ] Why did it perform better?
- [ ] Computational cost difference discussed
- [ ] Challenges or surprising findings discussed

## Results Function

- [ ] `get_assignment_results()` exists
- [ ] Filled accurately
- [ ] No placeholder values remain

---

# 26. Final Submission Process

Before submitting:

1. Complete every `TODO` in the provided notebook.
2. Run:
   ```text
   Kernel → Restart & Run All
   ```
3. Verify there are no errors.
4. Review all outputs.
5. Run:
   ```text
   Kernel → Restart & Clear Output
   ```
6. Then run:
   ```text
   Kernel → Restart & Run All
   ```
7. Verify the notebook again.
8. Save the notebook.

---

# 27. File Naming

Use the required format:

```text
YourGroupID_assignment.ipynb
```

Use the exact naming convention required by the assignment instructions/template.

---

# 28. Submission

Submit:

```text
1. Completed .ipynb file
2. PDF file, if required by the assignment portal
```

Verify that the upload is successful.

---

# 29. Critical Constraints for Implementation

The implementation MUST NOT:

```text
❌ Use sklearn model implementations
❌ Use TensorFlow
❌ Use Keras
❌ Use PyTorch
❌ Use JAX
❌ Use high-level neural network libraries
```

The implementation MUST:

```text
✓ Use NumPy for model calculations
✓ Implement gradient descent manually
✓ Implement forward propagation manually
✓ Implement backpropagation manually
✓ Track loss at every iteration
✓ Compare baseline and MLP
✓ Complete get_assignment_results()
✓ Keep analysis below 200 words
```

---

# 30. Code Assistant Instructions

When working on the provided Jupyter Notebook:

1. First inspect the entire notebook structure.
2. Identify all `TODO` sections.
3. Do not change required function names.
4. Do not remove or rename `get_assignment_results()`.
5. Preserve the notebook structure unless modification is necessary.
6. Use only permitted libraries.
7. Do not use prebuilt ML or neural network models.
8. Implement all mathematical operations using NumPy.
9. Use vectorized operations where possible.
10. Ensure all code runs from a clean kernel.
11. Verify that both baseline and MLP losses decrease.
12. Use actual experimental results for evaluation and analysis.
13. Keep the final analysis below 200 words.
14. Ensure the final notebook has no errors when using:
    ```text
    Restart Kernel → Run All
    ```

---

# SUCCESS CRITERIA

The assignment is complete only when:

```text
✓ Valid dataset selected
✓ Data preprocessing completed
✓ Baseline model implemented from scratch
✓ MLP implemented from scratch
✓ Both models trained successfully
✓ Both losses tracked and generally decreasing
✓ Test metrics calculated
✓ Required plots created
✓ Comparison analysis written (<200 words)
✓ get_assignment_results() completed
✓ Notebook runs successfully from start to finish
✓ No prohibited ML libraries used
```