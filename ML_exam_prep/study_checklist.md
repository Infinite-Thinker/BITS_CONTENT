# Machine Learning (AIMLZG565) Mid-Sem Exam Preparation Checklist

An interactive, checkbox-based study tracker optimized for a **2-day sprint** leading to the exam. Each module is structured into a highly efficient **60-minute revision block** to maximize focus and retention.

---

## Overall Progress

- [ ] **Module 1** — Introduction to ML (Tom Mitchell <T,P,E>, Generative vs Discriminative) — ~60 min
- [ ] **Module 2** — ML Workflow & Preprocessing (Stevens' scales, imputation, outlier detection, scaling) — ~60 min
- [ ] **Module 3** — Linear Models for Regression (OLS, GD variants, complexity, scaling) — ~60 min
- [ ] **Module 4** — Linear Models for Classification (Logistic regression, log-odds, non-convexity, XOR, multiclass) — ~60 min
- [ ] **Module 5** — Decision Trees (ID3 split, entropy, Gini, continuous split, pruning) — ~60 min
- [ ] **Module 6** — Model Evaluation Metrics (Confusion matrix, precision, recall, F1, error cost) — ~60 min
- [ ] **Formula Sheet Revision** (Section 2 of Master Guide) — ~30 min
- [ ] **High-Yield Repeated Questions Review** (Section 4 of Master Guide) — ~30 min
- [ ] **Commonly Confused Concepts Review** (Section 5 of Master Guide) — ~15 min
- [ ] **Bounce-Off / Unexpected Topics Prep** (`bounce_off.md` review) — ~45 min
- [ ] **Final Last-Minute revision run** (Section 6 of Master Guide) — ~20 min

---

## Module 1 — Introduction to Machine Learning — ~60 min

### 0–15 min — Core Concepts
- [ ] Tom Mitchell's Machine Learning definition: Experience $E$, Task $T$, Performance $P$
- [ ] Supervised vs. Unsupervised vs. Reinforcement Learning
- [ ] Generative vs. Discriminative model definitions and capabilities

### 15–30 min — Important Formulas
- [ ] Generative: Joint probability modeling $P(X, Y) = P(X|Y)P(Y)$
- [ ] Discriminative: Direct posterior modeling $P(Y|X)$

### 30–50 min — Important Questions
- [ ] **Q1** — Discriminative vs. Generative Classifier Selection for density estimation (`Sample QP, Q1b`)

### 50–60 min — Quick Revision
- [ ] Self-test: State Tom Mitchell's `<T, P, E>` for a self-driving car system
- [ ] Self-test: Explain why a discriminative classifier cannot perform anomaly/outlier detection

---

## Module 2 — ML Workflow & Preprocessing — ~60 min

### 0–15 min — Core Concepts
- [ ] Stevens' Scales of measurement: Nominal, Ordinal, Interval, Ratio
- [ ] Data preprocessing issues: Inconsistent data types, missing values, unit mismatches, outliers
- [ ] Imputation strategies: Mean, Median, and Mode imputation rules

### 15–30 min — Important Formulas
- [ ] Interquartile Range outlier detection: $\text{IQR} = Q_3 - Q_1$; Bounds = $[Q_1 - 1.5\text{IQR}, Q_3 + 1.5\text{IQR}]$
- [ ] 3-Sigma Rule bounds: $[\mu - 3\sigma, \mu + 3\sigma]$
- [ ] Min-Max Normalization: $v' = \frac{v - \min}{\max - \min} (\text{new\_max} - \text{new\_min}) + \text{new\_min}$
- [ ] Z-score Standardization: $v' = \frac{v - \mu}{\sigma}$

### 30–50 min — Important Questions
- [ ] **Q2** — Stevens' Attribute Scales Classification (`March 2026 Regular, Q2a, Q2b`)
- [ ] **Q3** — Data Quality & Preprocessing Case Study (`Dec 2025 Midsem Regular, Q1a, Q1b`)

### 50–60 min — Quick Revision
- [ ] Self-test: Why is Z-score standardization preferred over Min-Max normalization in the presence of extreme outliers?
- [ ] Recall: What arithmetic operations are mathematically meaningful on an Interval attribute?

---

## Module 3 — Linear Models for Regression — ~60 min

### 0–15 min — Core Concepts
- [ ] Notion of regression cost function (Mean Squared Error)
- [ ] Difference between Closed-form OLS (Normal Equation) and Iterative (Gradient Descent)
- [ ] The bias-variance trade-off in polynomial regressions (Overfitting vs Underfitting)
- [ ] Role and contours of Feature Scaling in Gradient Descent

### 15–30 min — Important Formulas
- [ ] OLS Closed Form: $\theta = (X^T X)^{-1} X^T Y$
- [ ] MAE, MSE, and RMSE error metrics
- [ ] Gradient descent parameter update rules (with and without L2 Ridge / L1 Lasso regularization)

### 30–50 min — Important Questions
- [ ] **Q4** — OLS Closed-Form Regression Line Matrix Calculation (`March 2026 Regular, Q1a, Q1b`)
- [ ] **Q5** — 1D Gradient Descent Minimization Iterations (`March 2026 Regular, Q3a`)

### 50–60 min — Quick Revision
- [ ] Write out the closed-form normal equation by hand from memory
- [ ] Draw the optimization paths of Gradient Descent with and without feature scaling

---

## Module 4 — Linear Models for Classification — ~60 min

### 0–15 min — Core Concepts
- [ ] The Sigmoid function and squashing properties
- [ ] Notion of Log-odds and how they relate linearly to inputs
- [ ] Why MSE cost function is non-convex for Logistic Regression
- [ ] Limits of linear boundaries (Why XOR is not linearly separable)
- [ ] Multiclass Logistic Regression strategies: One-vs-One and One-vs-All

### 15–30 min — Important Formulas
- [ ] Sigmoid: $\sigma(z) = \frac{1}{1 + e^{-z}}$
- [ ] Log-odds (logit): $\ln(\frac{p}{1-p}) = \theta^T x$
- [ ] Binary Cross-Entropy (Log Loss) cost function formula
- [ ] One-vs-One classifier count formula: $\frac{K(K-1)}{2}$

### 30–50 min — Important Questions
- [ ] **Q6** — Logistic Regression GD Weight Update and log loss calculation (`Dec 2025 Midsem Regular, Q1c, Q1d, Q1e`)

### 50–60 min — Quick Revision
- [ ] Self-test: If a class has 5 distinct labels, how many binary classifiers are needed under One-vs-One? How many under One-vs-All?
- [ ] Explain how a large positive weight $\theta_j$ affects the log-odds of a positive prediction.

---

## Module 5 — Decision Tree Classifier — ~60 min

### 0–15 min — Core Concepts
- [ ] Node purity, binary vs. multiclass splitting
- [ ] ID3, C4.5, and CART decision tree algorithms
- [ ] High-cardinality attribute split bias and solutions
- [ ] Model overfitting in Decision Trees and mitigation using Pruning (Pre-pruning vs Post-pruning)

### 15–30 min — Important Formulas
- [ ] Entropy of a node: $-\sum p_c \log_2(p_c)$
- [ ] Gini Index of a node: $1 - \sum p_c^2$
- [ ] Information Gain: $\text{Entropy}(S) - \sum \frac{|S_v|}{|S|}\text{Entropy}(S_v)$
- [ ] Split Information and Gain Ratio calculations

### 30–50 min — Important Questions
- [ ] **Q7** — Entropy & Information Gain split calculation (`Jan 2026 Midsem Makeup, Q2`)

### 50–60 min — Quick Revision
- [ ] What is the maximum value of Entropy for a 4-class classification problem?
- [ ] Self-test: Differentiate between the stoppage condition of Pre-pruning and the trimming logic of Post-pruning.

---

## Module 6 — Evaluation Measures — ~60 min

### 0–15 min — Core Concepts
- [ ] Understanding the binary Confusion Matrix: TP, FP, FN, TN
- [ ] Precision vs. Recall/Sensitivity trade-offs across applications
- [ ] Notion of cost-sensitive error evaluations
- [ ] Multiclass confusion matrices

### 15–30 min — Important Formulas
- [ ] Accuracy = $\frac{TP + TN}{\text{Total}}$
- [ ] Precision = $\frac{TP}{TP + FP}$
- [ ] Recall = $\frac{TP}{TP + FN}$
- [ ] F1-Score = $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
- [ ] Financial Error Cost: $(\text{FN} \times \text{Cost}_{\text{FN}}) + (\text{FP} \times \text{Cost}_{\text{FP}})$

### 30–50 min — Important Questions
- [ ] **Q8** — Diagnostic System Error-Cost and Threshold Recommendation (`Jan 2026 Midsem Makeup, Q4`)

### 50–60 min — Quick Revision
- [ ] Write down the F1-Score formula in terms of True Positives, False Positives, and False Negatives
- [ ] Self-test: In a high-security airport weapon detection system, would you prioritize Precision or Recall? Why?
