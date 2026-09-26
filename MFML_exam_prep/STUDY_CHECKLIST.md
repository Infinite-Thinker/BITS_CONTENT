# ⏱️ BITS Pilani M.Tech AIML: 8-Hour MFML Revision Checklist
*Your Step-by-Step 8-Hour Marathon Study Roadmap to Master Mathematical Foundations for Machine Learning.*

Dedicate **exactly 1 hour** to each lecture note. Each hour consists of:
1.  **Reviewing the slides/notes** (25 minutes)
2.  **Mastering the corresponding crucial proofs** (15 minutes)
3.  **Solving the corresponding Previous Year Questions (PYQs)** (20 minutes)

Use the checkboxes below (`- [ ]`) to track your progress as you work through each lecture.

---

## 🕒 HOUR 1: Lecture 1 - Systems of Linear Equations & Matrices
*Central Goal: Understand how systems of equations are written in matrix form and how they are solved systematically via row operations.*

### 📋 Checklist
- [ ] **Core Topics to Study:**
  - [ ] Definition of closure in mathematical operations.
  - [ ] Setting up augmented matrices $[A \mid b]$.
  - [ ] Matrix algebraic properties: Associativity $(AB)C = A(BC)$ and Distributivity $A(B+C) = AB + AC$.
  - [ ] Transpose properties: $(AB)^T = B^T A^T$ and inverse properties $(AB)^{-1} = B^{-1} A^{-1}$.
  - [ ] General solutions to $Ax = b$ as a sum of particular and homogeneous solutions: $x = x_p + x_h$.
- [ ] **Crucial Proof to Memorize:**
  - [ ] **Proof of Transpose Product Identity:** Show that $(AB)^T = B^T A^T$.
    *   *Hint:* Compare the $(i,j)$-th entry of both sides.
        $$\text{Entry}(AB)_{ji} = \sum_k A_{jk} B_{ki} \implies \text{Entry}(AB)^T_{ij} = \sum_k B^T_{ik} A^T_{kj} = (B^T A^T)_{ij}$$
- [ ] **PYQs to Solve (Ref: README.md):**
  - [ ] **Question 1:** Form the REF of a matrix, identifying pivot columns and writing row operations explicitly.
  - [ ] **Question 10 (Part 1):** Proving $(A+B)^{-1} \neq A^{-1} + B^{-1}$ using a clean $1 \times 1$ or $2 \times 2$ counter-example.

---

## 🕒 HOUR 2: Lecture 2 - Vector Spaces, Subspaces, & Groups
*Central Goal: Transition from raw matrices to abstract algebraic structures (groups, fields, vector spaces, and spans).*

### 📋 Checklist
- [ ] **Core Topics to Study:**
  - [ ] Group definitions and Abelian groups (e.g., verifying if $(\mathbb{Z}, +)$ or $(\mathbb{R}, \cdot)$ form groups).
  - [ ] The **10 Vector Space Axioms** (closure, associativity, commutativity, distributivity, identities).
  - [ ] Vector Subspace theorem: Show $U \subseteq V$ is a subspace by proving **closure under addition** ($u_1 + u_2 \in U$) and **scalar multiplication** ($k u \in U$).
  - [ ] Linear independence and checking it via Gaussian elimination (testing if $c_1 x_1 + \dots + c_k x_k = 0 \implies c_i = 0$).
  - [ ] Basis as the smallest generating set, and Dimension as the size of the basis.
- [ ] **Crucial Derivation to Memorize:**
  - [ ] Proving whether a custom subset (e.g., the y-axis, a square boundary, or upper triangular matrices) constitutes a valid vector subspace.
- [ ] **PYQs to Solve (Ref: README.md):**
  - [ ] **Question 2:** Disproving a custom vector space $x \oplus y = [x_1 + y_1 + 1, x_2 + y_2 + 1]^T$ by showing the failure of scalar distributivity.
  - [ ] **Question 1 (Null space basis):** Solving $Ax=0$ to find the basis and dimension (nullity) of the null space.

---

## 🕒 HOUR 3: Lecture 3 - Bilinear Forms, SPD Matrices, Norms, & Orthogonality
*Central Goal: Learn how to measure length, distances, angles, and define custom inner products using matrices.*

### 📋 Checklist
- [ ] **Core Topics to Study:**
  - [ ] Bilinear forms and extracting their matrix representation: $f(x, y) = x^T A y$.
  - [ ] Inner products induced by a symmetric positive-definite (SPD) matrix: $\langle x, y \rangle_A = x^T A y$.
  - [ ] Induced norm properties: $\|x\|_A = \sqrt{x^T A x}$ and Cauchy-Schwarz Inequality: $|\langle x, y \rangle| \leq \|x\| \|y\|$.
  - [ ] Metric spaces and defining distances: $d(x, y) = \|x - y\|$.
  - [ ] Angle definition: $\cos \theta = \frac{\langle x, y \rangle}{\|x\| \|y\|}$ and Orthogonality condition: $\langle x, y \rangle = 0$.
  - [ ] Gram-Schmidt Orthonormalization Process.
- [ ] **Crucial Proof to Memorize:**
  - [ ] Prove that $\langle x, y \rangle = x^T A y$ defines a valid inner product if and only if $A$ is Symmetric and Positive Definite.
- [ ] **PYQs to Solve (Ref: README.md):**
  - [ ] **Question 1 (Gram-Schmidt):** Converting $\{b_1, b_2\}$ to orthonormal $\{c_1, c_2\}$ step-by-step.
  - [ ] **Question 3:** Calculating custom distances $d(a, b)$ and angles $\theta$ under a bilinear matrix $A$.

---

## 🕒 HOUR 4: Lecture 4 - Determinants, Trace, EVD, & Cholesky
*Central Goal: Understand core matrix operators (determinant, trace) and how to decompose square symmetric matrices.*

### 📋 Checklist
- [ ] **Core Topics to Study:**
  - [ ] Determinants via cofactor formulas and behavior under elementary row operations.
  - [ ] Trace of a square matrix and trace-eigenvalue relation: $\text{tr}(A) = \sum \lambda_i$.
  - [ ] Eigenvalues and Eigenvectors: Characteristic Polynomial $p_A(\lambda) = \det(A - \lambda I) = 0$.
  - [ ] Spectral Theorem: Symmetric real matrices have real eigenvalues and orthogonal eigenvectors.
  - [ ] Cholesky Decomposition of SPD matrices: $A = L L^T$ where $L$ is lower-triangular.
- [ ] **Crucial Proofs to Memorize:**
  - [ ] Prove that eigenvectors corresponding to distinct eigenvalues of a symmetric matrix are strictly orthogonal.
  - [ ] Prove that a matrix and its transpose share the exact same eigenvalues.
- [ ] **PYQs to Solve (Ref: README.md):**
  - [ ] **Question 4:** Verifying positive definiteness via Leading Principal Minors and computing Cholesky factors $L$.
  - [ ] **Conceptual Cheat Sheet:** Distinct eigenvalues guarantee diagonalizability but do *not* guarantee invertibility (since an eigenvalue can be 0).

---

## 🕒 HOUR 5: Lecture 5 - Singular Value Decomposition (SVD)
*Central Goal: Master SVD—the ultimate decomposition for any rectangular matrix—and low-rank approximations.*

### 📋 Checklist
- [ ] **Core Topics to Study:**
  - [ ] SVD Theorem: $A = U \Sigma V^T$ for any $m \times n$ matrix of rank $r$.
  - [ ] Finding right singular vectors $V$ (eigenvectors of $A^T A$) and left singular vectors $U$ (eigenvectors of $A A^T$).
  - [ ] Singular values relation: $\sigma_i = \sqrt{\lambda_i(A^T A)}$.
  - [ ] Spectral Norm of a matrix: $\|A\|_2 = \sigma_{\max}$.
  - [ ] Low-Rank Approximations: Best rank-k approximation $A_k = \sum_{i=1}^k \sigma_i u_i v_i^T$.
- [ ] **Crucial Derivation to Memorize:**
  - [ ] Derivation of left singular vectors as the normalized image of right singular vectors: $u_i = \frac{1}{\sigma_i} A v_i$.
- [ ] **PYQs to Solve (Ref: README.md):**
  - [ ] **Question 5:** Computing the complete SVD of a rectangular matrix $A$ and identifying its spectral norm.
  - [ ] **Question 6:** Finding the best rank-1 approximation of $A^2$ under EVD parameters.
  - [ ] **Pseudocode Section:** Algorithm for SVD verification without direct matrix multiplication.

---

## 🕒 HOUR 6: Lecture 6 - Differentiation, Jacobians, & Vector Calculus
*Central Goal: Understand how univariate derivatives scale to multivariable functions, gradients, and Jacobian matrices.*

### 📋 Checklist
- [ ] **Core Topics to Study:**
  - [ ] Univariate Taylor and Maclaurin expansions.
  - [ ] Partial derivatives and collecting them into a row vector called the **Gradient**: $\nabla f = \left[ \frac{\partial f}{\partial x_1}, \dots, \frac{\partial f}{\partial x_n} \right]$.
  - [ ] Vector-valued functions $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$ and the **Jacobian Matrix** $J$.
  - [ ] Matrix differentiation properties:
    *   $\frac{\partial (x^T a)}{\partial x} = a^T$
    *   $\frac{\partial (Ax)}{\partial x} = A^T$ (for $f(x) = Ax$)
  - [ ] Multivariable chain rules and matrix multiplications.
- [ ] **Crucial Derivation to Memorize:**
  - [ ] Deriving the Jacobian matrix of a linear system $f(x) = Ax$.
- [ ] **PYQs to Solve (Ref: README.md):**
  - [ ] **Question 7 (Part 1):** Finding the 1st degree Taylor Polynomial linear model $L(x, y) = f(0,0) + f_x x + f_y y$.
  - [ ] **Question 9:** Deriving the gradient matrix $\frac{d(A^T A)}{dA}$ of size $9 \times 6$ step-by-step.

---

## 🕒 HOUR 7: Lecture 7 - Backpropagation & Automatic Differentiation
*Central Goal: Understand the exact mathematical machinery that updates weights in multi-layered neural networks.*

### 📋 Checklist
- [ ] **Core Topics to Study:**
  - [ ] Forward and Reverse modes of Automatic Differentiation.
  - [ ] Definition of backpropagated error signals: $\delta = \frac{\partial Loss}{\partial Z}$.
  - [ ] Applying the multi-variable chain rule across multiple layers.
  - [ ] Gradients with respect to weights ($\frac{\partial Loss}{\partial W}$) and biases ($\frac{\partial Loss}{\partial b}$).
- [ ] **Crucial Proof to Memorize (Extremely Common Exam Question!):**
  - [ ] **Sigmoid Layer Derivation:** Prove that for $a = \sigma(z)$, $\sigma'(z) = a \odot (1 - a)$, and derive:
    $$\delta(Z) = \delta(a) \odot a \odot (1 - a)$$
    $$\frac{\partial Loss}{\partial W} = \delta(Z) x^T, \quad \frac{\partial Loss}{\partial b} = \delta(Z)$$
- [ ] **PYQs to Solve (Ref: README.md):**
  - [ ] Trace through the step-by-step mathematical derivation of backpropagation through a Sigmoid layer listed in `README.md`.

---

## 🕒 HOUR 8: Lecture 8 - Multivariable Taylor Series, Hessians, & Optimization
*Central Goal: Learn how to perform local optimization (finding maximums, minimums, and saddle points) using multi-variable Taylor series.*

### 📋 Checklist
- [ ] **Core Topics to Study:**
  - [ ] Mean Value Theorem and Rolle's Theorem definitions.
  - [ ] Second-order Multivariable Taylor expansion:
    $$f(a+h) \approx f(a) + \nabla f(a)^T h + \frac{1}{2} h^T H_f(a) h$$
  - [ ] The **Hessian Matrix** of second-order partial derivatives.
  - [ ] Classifying critical points ($\nabla f = 0$) using the definiteness of the Hessian:
    *   **Positive Definite Hessian** ($H > 0$) $\implies$ **Local Minimum**
    *   **Negative Definite Hessian** ($H < 0$) $\implies$ **Local Maximum**
    *   **Indefinite Hessian** (mixed sign eigenvalues) $\implies$ **Saddle Point**
- [ ] **Crucial Proof to Memorize:**
  - [ ] Prove that if a function's Hessian is negative-definite at a stationary point, it represents a local maximum.
- [ ] **PYQs to Solve (Ref: README.md):**
  - [ ] **Question 7 (Parts 2 & 3):** Computing gradient/Hessian for a loss $h(x, y)$ and classifying its 4 critical points.
  - [ ] **Question 8:** Maximizing corporate profit by finding stationary points, evaluating Hessian definiteness, and identifying the relative maximum.

---

### 🏆 Final Sprint: 15 Mins Before Exam
*   Take a quick glance at the **Conceptual Proofs & Traps (Section D of README.md)**.
*   Make sure you know the eigenvalues of $rr^T$ ($\|r\|_2^2$ and $0$) and the Rank of $I - vv^T$ ($\leq n-1$).
*   Take a deep breath. You are fully prepared to secure an outstanding grade tomorrow! 🎓🔥
