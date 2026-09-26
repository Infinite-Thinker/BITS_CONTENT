# 🎯 BITS Pilani M.Tech AIML: MFML (AIML ZC416) Master Solved Question Bank
*The Ultimate Revision Playbook Containing Every Question from the 2025 and 2026 Mid-Sem Exams with Detailed step-by-step Solutions.*

---

## 📑 SECTION A: LINEAR ALGEBRA, VECTOR SPACES & INNER PRODUCTS

### ❓ Question 1: REF, Column Space, Null Space, & Rank-Nullity
*(Source: Regular Mid-Sem Exam)*
**Question:** Consider the matrix:
$$A = \begin{bmatrix} -3 & 6 & -1 & 1 & -7 \\ 1 & -2 & 2 & 3 & -1 \\ 2 & -4 & 5 & 8 & -4 \end{bmatrix}$$
1.  Using elementary row operations, write the matrix in Row Echelon Form (REF).
2.  Find the basis and dimension of the Column Space $Col(A)$.
3.  Find the basis and dimension of the Null Space $Null(A)$ / Kernel.
4.  Find the basis of the Row Space of $A$. What is its rank?

#### 💡 Detailed Step-by-Step Solution:
1.  **Row Echelon Form (REF):**
    *   Swap rows 1 and 2 ($R_1 \leftrightarrow R_2$) to get a pivot of 1 in the top-left:
        $$A \sim \begin{bmatrix} 1 & -2 & 2 & 3 & -1 \\ -3 & 6 & -1 & 1 & -7 \\ 2 & -4 & 5 & 8 & -4 \end{bmatrix}$$
    *   Perform $R_2 \leftarrow R_2 + 3R_1$ and $R_3 \leftarrow R_3 - 2R_1$:
        $$R_2: \begin{bmatrix} -3 + 3(1) & 6 - 2(3) & -1 + 2(3) & 1 + 3(3) & -7 + 3(-1) \end{bmatrix} = \begin{bmatrix} 0 & 0 & 5 & 10 & -10 \end{bmatrix}$$
        $$R_3: \begin{bmatrix} 2 - 2(1) & -4 - 2(-2) & 5 - 2(2) & 8 - 2(3) & -4 - 2(-1) \end{bmatrix} = \begin{bmatrix} 0 & 0 & 1 & 2 & -2 \end{bmatrix}$$
        $$A \sim \begin{bmatrix} 1 & -2 & 2 & 3 & -1 \\ 0 & 0 & 5 & 10 & -10 \\ 0 & 0 & 1 & 2 & -2 \end{bmatrix}$$
    *   Scale row 2 ($R_2 \leftarrow \frac{1}{5}R_2$):
        $$A \sim \begin{bmatrix} 1 & -2 & 2 & 3 & -1 \\ 0 & 0 & 1 & 2 & -2 \\ 0 & 0 & 1 & 2 & -2 \end{bmatrix}$$
    *   Subtract row 2 from row 3 ($R_3 \leftarrow R_3 - R_2$):
        $$A \sim \begin{bmatrix} 1 & -2 & 2 & 3 & -1 \\ 0 & 0 & 1 & 2 & -2 \\ 0 & 0 & 0 & 0 & 0 \end{bmatrix} \quad \text{(This is a valid REF!)}$$
2.  **Basis and Dimension of Column Space ($Col(A)$):**
    *   The pivots are in **Column 1** and **Column 3**.
    *   The basis is formed by the corresponding **original columns** of $A$:
        $$\text{Basis}(Col(A)) = \left\{ \begin{bmatrix} -3 \\ 1 \\ 2 \end{bmatrix}, \begin{bmatrix} -1 \\ 2 \\ 5 \end{bmatrix} \right\}, \quad \text{Dimension} = \text{Rank}(A) = 2$$
3.  **Basis and Dimension of Null Space ($Null(A)$):**
    *   Convert REF to RREF by performing $R_1 \leftarrow R_1 - 2R_2$:
        $$\text{RREF}(A) = \begin{bmatrix} 1 & -2 & 0 & -1 & 3 \\ 0 & 0 & 1 & 2 & -2 \\ 0 & 0 & 0 & 0 & 0 \end{bmatrix}$$
    *   Express the equations: $x_1 - 2x_2 - x_4 + 3x_5 = 0 \implies x_1 = 2x_2 + x_4 - 3x_5$
    *   $x_3 + 2x_4 - 2x_5 = 0 \implies x_3 = -2x_4 + 2x_5$
    *   Let the free variables be $x_2 = r$, $x_4 = s$, $x_5 = t$:
        $$x = \begin{bmatrix} 2r + s - 3t \\ r \\ -2s + 2t \\ s \\ t \end{bmatrix} = r \begin{bmatrix} 2 \\ 1 \\ 0 \\ 0 \\ 0 \end{bmatrix} + s \begin{bmatrix} 1 \\ 0 \\ -2 \\ 1 \\ 0 \end{bmatrix} + t \begin{bmatrix} -3 \\ 0 \\ 2 \\ 0 \\ 1 \end{bmatrix}$$
        $$\text{Basis}(Null(A)) = \left\{ \begin{bmatrix} 2 \\ 1 \\ 0 \\ 0 \\ 0 \end{bmatrix}, \begin{bmatrix} 1 \\ 0 \\ -2 \\ 1 \\ 0 \end{bmatrix}, \begin{bmatrix} -3 \\ 0 \\ 2 \\ 0 \\ 1 \end{bmatrix} \right\}, \quad \text{Dimension (Nullity)} = 3$$
4.  **Row Space Basis and Rank:**
    *   The non-zero rows of the REF form a basis of the Row Space:
        $$\text{Basis}(Row(A)) = \left\{ \begin{bmatrix} 1 & -2 & 2 & 3 & -1 \end{bmatrix}, \begin{bmatrix} 0 & 0 & 1 & 2 & -2 \end{bmatrix} \right\}, \quad \text{Rank} = 2$$

---

### ❓ Question 2: Custom Vector Space Axioms Proof/Disproof
*(Source: 2025-2026 SEM 2 Q2b)*
**Question:** Let $V = \mathbb{R}^2$. Define vector addition as $x \oplus y = [x_1 + y_1 + 1, x_2 + y_2 + 1]^T$, and scalar multiplication as $k x = [k x_1, k x_2]^T$ where $k \in \mathbb{R}$. Prove or disprove that $V$ is a vector space over $\mathbb{R}$.

#### 💡 Detailed Step-by-Step Solution:
Let's check the vector space axioms:
1.  **Additive Identity Axiom:** We seek an identity vector $e = [e_1, e_2]^T$ such that $x \oplus e = x$:
    $$\begin{bmatrix} x_1 + e_1 + 1 \\ x_2 + e_2 + 1 \end{bmatrix} = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} \implies \begin{cases} e_1 + 1 = 0 \implies e_1 = -1 \\ e_2 + 1 = 0 \implies e_2 = -1 \end{cases} \implies e = \begin{bmatrix} -1 \\ -1 \end{bmatrix}$$
    This identity exists.
2.  **Distributive Axiom ($k(x \oplus y) = kx \oplus ky$):**
    *   **Left Hand Side (LHS):**
        $$k(x \oplus y) = k \begin{bmatrix} x_1 + y_1 + 1 \\ x_2 + y_2 + 1 \end{bmatrix} = \begin{bmatrix} kx_1 + ky_1 + k \\ kx_2 + ky_2 + k \end{bmatrix}$$
    *   **Right Hand Side (RHS):**
        $$kx \oplus ky = \begin{bmatrix} kx_1 \\ kx_2 \end{bmatrix} \oplus \begin{bmatrix} ky_1 \\ ky_2 \end{bmatrix} = \begin{bmatrix} kx_1 + ky_1 + 1 \\ kx_2 + ky_2 + 1 \end{bmatrix}$$
    *   **Comparison:** Comparing LHS and RHS:
        $$\begin{bmatrix} kx_1 + ky_1 + k \\ kx_2 + ky_2 + k \end{bmatrix} \neq \begin{bmatrix} kx_1 + ky_1 + 1 \\ kx_2 + ky_2 + 1 \end{bmatrix} \quad (\text{Fails for any } k \neq 1)$$
    *   *Conclusion:* Since the distributive axiom fails, **$V$ is NOT a vector space**.

---

### ❓ Question 3: Coordinate Distances & Angles in General Inner Product Spaces
*(Source: Regular Mid-Sem Exam)*
**Question:** Consider an inner product space with $\langle x, y \rangle = x^T A y$ where:
$$A = \begin{bmatrix} 5.5 & -1.5 \\ -1.5 & 5.5 \end{bmatrix}$$
For vectors $a = [1, 5]^T$ and $b = [2, 7]^T$:
1.  Find the distance $d(a, b) = \|a - b\|_A$.
2.  Find the angle $\theta$ between $a$ and $b$ under this inner product.

#### 💡 Detailed Step-by-Step Solution:
1.  **Distance $d(a, b)$:**
    *   Let $v = a - b = \begin{bmatrix} 1 - 2 \\ 5 - 7 \end{bmatrix} = \begin{bmatrix} -1 \\ -2 \end{bmatrix}$.
    *   Compute $\|v\|_A^2 = v^T A v$:
        $$v^T A v = \begin{bmatrix} -1 & -2 \end{bmatrix} \begin{bmatrix} 5.5 & -1.5 \\ -1.5 & 5.5 \end{bmatrix} \begin{bmatrix} -1 \\ -2 \end{bmatrix}$$
        $$A v = \begin{bmatrix} 5.5(-1) - 1.5(-2) \\ -1.5(-1) + 5.5(-2) \end{bmatrix} = \begin{bmatrix} -5.5 + 3.0 \\ 1.5 - 11.0 \end{bmatrix} = \begin{bmatrix} -2.5 \\ -9.5 \end{bmatrix}$$
        $$v^T A v = \begin{bmatrix} -1 & -2 \end{bmatrix} \begin{bmatrix} -2.5 \\ -9.5 \end{bmatrix} = (-1)(-2.5) + (-2)(-9.5) = 2.5 + 19 = 21.5$$
    *   $$d(a, b) = \sqrt{21.5} \approx 4.637$$
2.  **Angle $\theta$:**
    *   $$\cos \theta = \frac{a^T A b}{\|a\|_A \|b\|_A}$$
    *   Compute $\|a\|_A^2 = a^T A a$:
        $$a^T A a = \begin{bmatrix} 1 & 5 \end{bmatrix} \begin{bmatrix} 5.5 & -1.5 \\ -1.5 & 5.5 \end{bmatrix} \begin{bmatrix} 1 \\ 5 \end{bmatrix} = \begin{bmatrix} 1 & 5 \end{bmatrix} \begin{bmatrix} -2 \\ 26 \end{bmatrix} = -2 + 130 = 128 \implies \|a\|_A = \sqrt{128}$$
    *   Compute $\|b\|_A^2 = b^T A b$:
        $$b^T A b = \begin{bmatrix} 2 & 7 \end{bmatrix} \begin{bmatrix} 5.5 & -1.5 \\ -1.5 & 5.5 \end{bmatrix} \begin{bmatrix} 2 \\ 7 \end{bmatrix} = \begin{bmatrix} 2 & 7 \end{bmatrix} \begin{bmatrix} 0.5 \\ 35.5 \end{bmatrix} = 1 + 248.5 = 249.5 \implies \|b\|_A = \sqrt{249.5}$$
    *   Compute $a^T A b$:
        $$a^T A b = \begin{bmatrix} 1 & 5 \end{bmatrix} \begin{bmatrix} 5.5 & -1.5 \\ -1.5 & 5.5 \end{bmatrix} \begin{bmatrix} 2 \\ 7 \end{bmatrix} = \begin{bmatrix} 1 & 5 \end{bmatrix} \begin{bmatrix} 0.5 \\ 35.5 \end{bmatrix} = 0.5 + 177.5 = 178$$
    *   Compute Cosine:
        $$\cos \theta = \frac{178}{\sqrt{128 \times 249.5}} = \frac{178}{\sqrt{31936}} = \frac{178}{178.706} \approx 0.996 \implies \theta \approx 5.12^\circ$$
    *(Note: The BITS Pilani printed key has an arithmetic error showing $0.005573 / 89.6^\circ$, but full marks are awarded for writing the correct formula and substituting the correct terms!)*

---

## 💾 SECTION B: MATRIX DECOMPOSITIONS (CHOLESKY & SVD)

### ❓ Question 4: Proving Positive Definiteness & Cholesky Decomposition
*(Source: 2025-2026 SEM 2 Q3)*
**Question:** A machine learning engineer has covariance matrix $\Sigma = \begin{bmatrix} 4 & 2 & 2 \\ 2 & 5 & 1 \\ 2 & 1 & 3 \end{bmatrix}$:
1.  Prove that $\Sigma$ is symmetric and positive-definite.
2.  Compute the Cholesky decomposition $\Sigma = L L^T$.
3.  Explain why positive definiteness is a necessary condition for Cholesky.

#### 💡 Detailed Step-by-Step Solution:
1.  **Prove Symmetry & Positive-Definiteness:**
    *   **Symmetry:** $\Sigma^T = \Sigma$ because $\Sigma_{ij} = \Sigma_{ji}$ for all $i, j$.
    *   **Positive-Definiteness (Leading Principal Minors):**
        *   Minor 1: $M_1 = 4 > 0$
        *   Minor 2: $M_2 = \det\begin{bmatrix} 4 & 2 \\ 2 & 5 \end{bmatrix} = (4 \times 5) - (2 \times 2) = 16 > 0$
        *   Minor 3 (Full $\det(\Sigma)$):
            $$\det(\Sigma) = 4(15 - 1) - 2(6 - 2) + 2(2 - 10) = 56 - 8 - 16 = 32 > 0$$
        Since all leading principal minors are positive, $\Sigma$ is positive-definite.
2.  **Compute Cholesky Decomposition $\Sigma = L L^T$:**
    Let $L = \begin{bmatrix} l_{11} & 0 & 0 \\ l_{21} & l_{22} & 0 \\ l_{31} & l_{32} & l_{33} \end{bmatrix}$. Set $L L^T = \Sigma$:
    *   $l_{11}^2 = 4 \implies l_{11} = 2$
    *   $l_{11}l_{21} = 2 \implies 2l_{21} = 2 \implies l_{21} = 1$
    *   $l_{11}l_{31} = 2 \implies 2l_{31} = 2 \implies l_{31} = 1$
    *   $l_{21}^2 + l_{22}^2 = 5 \implies 1^2 + l_{22}^2 = 5 \implies l_{22}^2 = 4 \implies l_{22} = 2$
    *   $l_{21}l_{31} + l_{22}l_{32} = 1 \implies (1)(1) + 2l_{32} = 1 \implies 2l_{32} = 0 \implies l_{32} = 0$
    *   $l_{31}^2 + l_{32}^2 + l_{33}^2 = 3 \implies 1^2 + 0^2 + l_{33}^2 = 3 \implies l_{33}^2 = 2 \implies l_{33} = \sqrt{2}$
    $$L = \begin{bmatrix} 2 & 0 & 0 \\ 1 & 2 & 0 \\ 1 & 0 & \sqrt{2} \end{bmatrix}$$
3.  **Why Positive Definiteness is Necessary:**
    The diagonal values of $L$ are solved via $l_{ii} = \sqrt{\Sigma_{ii} - \sum_{k=1}^{i-1} l_{ik}^2}$. If a matrix is not positive-definite, the term inside the square root can become negative or zero, yielding complex numbers or a non-invertible matrix, making real Cholesky decomposition impossible.

---

### ❓ Question 5: EVD of $A^T A$ / $A A^T$ & SVD Derivation
*(Source: Regular Mid-Sem Exam)*
**Question:** Obtain the complete Singular Value Decomposition ($A = U \Sigma V^T$) of the matrix $A = \begin{bmatrix} 2 & 1 \\ 1 & 0 \\ 0 & 1 \end{bmatrix}$. What is the spectral norm of $A$?

#### 💡 Detailed Step-by-Step Solution:
1.  **Form $A^T A$:**
    $$A^T A = \begin{bmatrix} 2 & 1 & 0 \\ 1 & 0 & 1 \end{bmatrix} \begin{bmatrix} 2 & 1 \\ 1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 5 & 2 \\ 2 & 2 \end{bmatrix}$$
2.  **Find Eigenvalues of $A^T A$:**
    $$\det \begin{bmatrix} 5-\lambda & 2 \\ 2 & 2-\lambda \end{bmatrix} = \lambda^2 - 7\lambda + 6 = 0 \implies (\lambda-6)(\lambda-1)=0 \implies \lambda_1=6, \lambda_2=1$$
    *   **Singular Values:** $\sigma_1 = \sqrt{6}$, $\sigma_2 = 1$.
    *   **Spectral Norm:** $\|A\|_2 = \sigma_{\max} = \sqrt{6}$.
3.  **Find Right Singular Vectors $V$ (eigenvectors of $A^T A$):**
    *   **For $\lambda_1 = 6$:** $\begin{bmatrix} -1 & 2 \\ 2 & -4 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = 0 \implies x = 2y \implies v_1 = \frac{1}{\sqrt{5}} \begin{bmatrix} 2 \\ 1 \end{bmatrix}$
    *   **For $\lambda_2 = 1$:** $\begin{bmatrix} 4 & 2 \\ 2 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = 0 \implies y = -2x \implies v_2 = \frac{1}{\sqrt{5}} \begin{bmatrix} -1 \\ 2 \end{bmatrix}$
    $$V = \frac{1}{\sqrt{5}} \begin{bmatrix} 2 & -1 \\ 1 & 2 \end{bmatrix}$$
4.  **Compute Left Singular Vectors $U$ ($u_i = \frac{1}{\sigma_i} A v_i$):**
    *   $$u_1 = \frac{1}{\sqrt{6}} \begin{bmatrix} 2 & 1 \\ 1 & 0 \\ 0 & 1 \end{bmatrix} \left( \frac{1}{\sqrt{5}} \begin{bmatrix} 2 \\ 1 \end{bmatrix} \right) = \frac{1}{\sqrt{30}} \begin{bmatrix} 5 \\ 2 \\ 1 \end{bmatrix}$$
    *   $$u_2 = \frac{1}{1} \begin{bmatrix} 2 & 1 \\ 1 & 0 \\ 0 & 1 \end{bmatrix} \left( \frac{1}{\sqrt{5}} \begin{bmatrix} -1 \\ 2 \end{bmatrix} \right) = \frac{1}{\sqrt{5}} \begin{bmatrix} 0 \\ -1 \\ 2 \end{bmatrix}$$
    *   **Find $u_3$:** Must be orthogonal to $u_1$ and $u_2$. Solving $u_1^T u_3 = 0, u_2^T u_3 = 0$:
        $$5x + 2y + z = 0 \quad \text{and} \quad -y + 2z = 0 \implies y = 2z \implies x = -z \implies u_3 = \frac{1}{\sqrt{6}} \begin{bmatrix} -1 \\ 2 \\ 1 \end{bmatrix}$$
5.  **Compile the final SVD Matrices:**
    $$U = \begin{bmatrix} 5/\sqrt{30} & 0 & -1/\sqrt{6} \\ 2/\sqrt{30} & -1/\sqrt{5} & 2/\sqrt{6} \\ 1/\sqrt{30} & 2/\sqrt{5} & 1/\sqrt{6} \end{bmatrix}, \quad \Sigma = \begin{bmatrix} \sqrt{6} & 0 \\ 0 & 1 \\ 0 & 0 \end{bmatrix}, \quad V^T = \frac{1}{\sqrt{5}} \begin{bmatrix} 2 & 1 \\ -1 & 2 \end{bmatrix}$$

---

### ❓ Question 6: EVD Spectral Decomposition & Best Rank-1 Approximation
*(Source: Jan 2025 AIML Solution Key Q3i)*
**Question:** Suppose we arrive at an $m \times m$ matrix $A = P D P^T$ where $P^{-1} = P^T$ and $D = \text{diag}(\ell, \ell^2, \dots, \ell^m)$ where $|\ell| \neq 0, 1$. Find the best rank-1 approximation of $A^2$ with justification.

#### 💡 Detailed Step-by-Step Solution:
1.  **Analyze the Eigenvalues of $A^2$:**
    Since $A = P D P^T$ is symmetric ($P^T = P^{-1}$), we have:
    $$A^2 = (P D P^T)(P D P^T) = P D^2 P^T$$
    Therefore, the eigenvalues of $A^2$ are the diagonal elements of $D^2$, which are:
    $$\lambda_i = (\ell^i)^2 = \ell^{2i} \quad \text{for } i = 1, 2, \dots, m$$
2.  **Determine the Singular Values:**
    Since $A^2$ is symmetric, its singular values are the absolute values of its eigenvalues:
    $$\sigma_i = |\ell^{2i}| = |\ell|^{2i}$$
3.  **Identify the Largest Singular Value:**
    *   **Case 1: $|\ell| > 1$:** The terms $|\ell|^{2i}$ increase as $i$ increases. The largest singular value is the last one: $\sigma_{\max} = |\ell|^{2m}$, corresponding to eigenvector $P_m$ (the $m$-th column of $P$).
    *   **Case 2: $|\ell| < 1$:** The terms $|\ell|^{2i}$ decrease as $i$ increases. The largest singular value is the first one: $\sigma_{\max} = |\ell|^2$, corresponding to eigenvector $P_1$ (the 1st column of $P$).
4.  **Construct the Best Rank-1 Approximation:**
    The best rank-1 approximation of a symmetric matrix $M$ is $\sigma_{\max} u_1 u_1^T$.
    *   If $|\ell| > 1$: Best Rank-1 Approximation of $A^2$ is $\ell^{2m} P_m P_m^T$.
    *   If $|\ell| < 1$: Best Rank-1 Approximation of $A^2$ is $\ell^2 P_1 P_1^T$.

---

## 📈 SECTION C: CALCULUS, OPTIMIZATION & AUTOMATIC DIFFERENTIATION

### ❓ Question 7: Gradient and Hessian of a Quadratic Loss Process
*(Source: 2025-2026 SEM 2 Q1)*
**Question:** Given $f(x, y) = x + y + \cos(x) + \cos(y)$ is generated by a process:
1.  Find a linear model $L(x, y)$ to predict the output about $(0,0)$.
2.  If the loss function is $h(x, y) = (f(x, y) - L(x, y))^2$, find the gradient and Hessian of $h(x, y)$.
3.  Find all critical points of $h(x, y)$ in $[0, \pi] \times [0, \pi]$ and classify them.

#### 💡 Detailed Step-by-Step Solution:
1.  **Linear Model $L(x, y)$:**
    $$L(x, y) = f(0,0) + \frac{\partial f}{\partial x}\bigg|_{(0,0)}x + \frac{\partial f}{\partial y}\bigg|_{(0,0)}y$$
    $$f(0,0) = 0 + 0 + \cos(0) + \cos(0) = 2$$
    $$\frac{\partial f}{\partial x} = 1 - \sin(x) \implies f_x(0,0) = 1, \quad \frac{\partial f}{\partial y} = 1 - \sin(y) \implies f_y(0,0) = 1$$
    $$L(x, y) = 2 + x + y$$
2.  **Gradient and Hessian of Loss $h(x, y) = (\cos(x) + \cos(y) - 2)^2$:**
    Let $u = \cos(x) + \cos(y) - 2$.
    *   **Gradient:**
        $$\frac{\partial h}{\partial x} = 2u (-\sin(x)) = -2\sin(x)(\cos(x) + \cos(y) - 2)$$
        $$\frac{\partial h}{\partial y} = 2u (-\sin(y)) = -2\sin(y)(\cos(x) + \cos(y) - 2)$$
        $$\nabla h(x, y) = \begin{bmatrix} -2\sin(x)(\cos(x) + \cos(y) - 2) \\ -2\sin(y)(\cos(x) + \cos(y) - 2) \end{bmatrix}^T$$
    *   **Hessian:**
        $$H_{11} = -2\cos(x)(\cos(x) + \cos(y) - 2) + 2\sin^2(x)$$
        $$H_{12} = H_{21} = 2\sin(x)\sin(y)$$
        $$H_{22} = -2\cos(y)(\cos(x) + \cos(y) - 2) + 2\sin^2(y)$$
3.  **Critical Points and Classification in $[0, \pi] \times [0, \pi]$:**
    Setting $\nabla h = 0$:
    *   **Point $(0,0)$:** Here, $\cos(0) + \cos(0) - 2 = 0 \implies h(0,0) = 0$. Since $h \geq 0$ everywhere, **$(0,0)$ is a Global Minimum**.
    *   **Point $(\pi, \pi)$:** Here, $\sin(\pi) = 0, \cos(\pi) = -1 \implies u = -4$.
        $$H_{11} = -2(-1)(-4) + 0 = -8, \quad H_{12} = 0, \quad H_{22} = -8 \implies H = \begin{bmatrix} -8 & 0 \\ 0 & -8 \end{bmatrix}$$
        The Hessian is Negative Definite $\implies$ **$(\pi, \pi)$ is a Local Maximum**.
    *   **Points $(0, \pi)$ and $(\pi, 0)$:** Let's test $(0, \pi)$:
        $$\cos(0)=1, \cos(\pi)=-1 \implies u = -2$$
        $$H_{11} = -2(1)(-2) + 0 = 4, \quad H_{12} = 0, \quad H_{22} = -2(-1)(-2) + 0 = -4 \implies H = \begin{bmatrix} 4 & 0 \\ 0 & -4 \end{bmatrix}$$
        The eigenvalues are $+4, -4$ (Indefinite) $\implies$ **$(0, \pi)$ and $(\pi, 0)$ are Saddle Points**.

---

### ❓ Question 8: Optimization of Corporate Profit Function
*(Source: Jan 2025 AIML Solution Key Q5)*
**Question:** The profit of a software company on a project is given by:
$$f(x_1, x_2) = x_1^3 + x_2^3 + 2x_1^2 + 4x_2^2 + 6$$
Find the Hessian matrix and check whether it is positive-definite or negative-definite at each stationary point. Find the values of $x_1$ and $x_2$ that maximize the profit.

#### 💡 Detailed Step-by-Step Solution:
1.  **Find Stationary Points ($\nabla f = 0$):**
    $$\frac{\partial f}{\partial x_1} = 3x_1^2 + 4x_1 = 0 \implies x_1(3x_1 + 4) = 0 \implies x_1 = 0, \quad x_1 = -\frac{4}{3}$$
    $$\frac{\partial f}{\partial x_2} = 3x_2^2 + 8x_2 = 0 \implies x_2(3x_2 + 8) = 0 \implies x_2 = 0, \quad x_2 = -\frac{8}{3}$$
    This yields 4 stationary points: $(0,0)$, $\left(0, -\frac{8}{3}\right)$, $\left(-\frac{4}{3}, 0\right)$, and $\left(-\frac{4}{3}, -\frac{8}{3}\right)$.
2.  **Form the Hessian Matrix:**
    $$\frac{\partial^2 f}{\partial x_1^2} = 6x_1 + 4, \quad \frac{\partial^2 f}{\partial x_2^2} = 6x_2 + 8, \quad \frac{\partial^2 f}{\partial x_1 \partial x_2} = 0 \implies H = \begin{bmatrix} 6x_1 + 4 & 0 \\ 0 & 6x_2 + 8 \end{bmatrix}$$
3.  **Evaluate at each Point:**
    *   **At $(0,0)$:**
        $$H = \begin{bmatrix} 4 & 0 \\ 0 & 8 \end{bmatrix} \implies \text{Positive Definite} \implies \text{Relative Minimum}$$
    *   **At $\left(0, -\frac{8}{3}\right)$:**
        $$H = \begin{bmatrix} 4 & 0 \\ 0 & -8 \end{bmatrix} \implies \text{Indefinite} \implies \text{Saddle Point}$$
    *   **At $\left(-\frac{4}{3}, 0\right)$:**
        $$H = \begin{bmatrix} -4 & 0 \\ 0 & 8 \end{bmatrix} \implies \text{Indefinite} \implies \text{Saddle Point}$$
    *   **At $\left(-\frac{4}{3}, -\frac{8}{3}\right)$:**
        $$H = \begin{bmatrix} -4 & 0 \\ 0 & -8 \end{bmatrix} \implies \text{Negative Definite} \implies \text{Relative Maximum}$$
4.  **Maximize Profit:**
    Profit is maximized at the relative maximum:
    $$x_1 = -\frac{4}{3}, \quad x_2 = -\frac{8}{3}$$

---

### ❓ Question 9: Multi-variable Matrix Gradient Derivation
*(Source: Jan 2025 AIML Solution Key Q6)*
**Question:** Let $A = \begin{bmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \end{bmatrix} \in \mathbb{R}^{2 \times 3}$. Consider the function $f(A) = A^T A = K \in \mathbb{R}^{3 \times 3}$. Calculate the gradient of $K$ with respect to $A$ in matrix form.

#### 💡 Detailed Step-by-Step Solution:
1.  **Write out the elements of $K = A^T A$:**
    $$K = \begin{bmatrix} a_{11} & a_{21} \\ a_{12} & a_{22} \\ a_{13} & a_{23} \end{bmatrix} \begin{bmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \end{bmatrix} = \begin{bmatrix} a_{11}^2 + a_{21}^2 & a_{11}a_{12} + a_{21}a_{22} & a_{11}a_{13} + a_{21}a_{23} \\ a_{12}a_{11} + a_{22}a_{21} & a_{12}^2 + a_{22}^2 & a_{12}a_{13} + a_{22}a_{23} \\ a_{13}a_{11} + a_{23}a_{21} & a_{13}a_{12} + a_{23}a_{22} & a_{13}^2 + a_{23}^2 \end{bmatrix}$$
2.  **Define the Gradient Matrix $\frac{dK}{dA}$:**
    The gradient of a $3 \times 3$ matrix with respect to a $2 \times 3$ matrix is represented as a $9 \times 6$ (or stacked $9 \times 6$) Jacobian of partial derivatives:
    $$\frac{dK}{dA} = \left[ \frac{\partial K_{pq}}{\partial A_{ij}} \right]$$
3.  **Differentiate elements using the piecewise rule:**
    $$\frac{\partial K_{pq}}{\partial A_{ij}} = \begin{cases} A_{iq} & \text{if } j = p, p \neq q \\ A_{ip} & \text{if } j = q, p \neq q \\ 2A_{iq} & \text{if } j = p, p = q \\ 0 & \text{otherwise} \end{cases}$$
4.  **Assemble the final $9 \times 6$ matrix:**
    $$\frac{dK}{dA} = \begin{bmatrix} 2a_{11} & 0 & 0 & 2a_{21} & 0 & 0 \\ a_{12} & a_{11} & 0 & a_{22} & a_{21} & 0 \\ a_{13} & 0 & a_{11} & a_{23} & 0 & a_{21} \\ a_{12} & a_{11} & 0 & a_{22} & a_{21} & 0 \\ 0 & 2a_{12} & 0 & 0 & 2a_{22} & 0 \\ 0 & a_{13} & a_{12} & 0 & a_{23} & a_{22} \\ a_{13} & 0 & a_{11} & a_{23} & 0 & a_{21} \\ 0 & a_{13} & a_{12} & 0 & a_{23} & a_{22} \\ 0 & 0 & 2a_{13} & 0 & 0 & 2a_{23} \end{bmatrix}$$

---

## 🔐 SECTION D: HIGHLY PROBABLE CONCEPTUAL PROOFS (JUSTIFICATIONS)

### ❓ Question 10: Matrix Properties & Identity Verification
*(Source: Jan 2025 AIML Solution Key Q4)*
**Question:** Let $v, w \in \mathbb{R}^{n \times 1}$ be non-zero vectors. Consider the identity matrix $I_n$:
1.  Consider $B = I_n - 2v v^T$. Is $B$ orthogonal for *any* $v$? Is $B$ symmetric?
2.  Assume $\|w\|_2 = 1$. Prove that $C = I_n - 2 w w^T$ (Householder Reflector) is orthogonal.

#### 💡 Detailed Step-by-Step Solution:
1.  **Properties of $B = I_n - 2v v^T$:**
    *   **Symmetry:**
        $$B^T = (I_n - 2v v^T)^T = I_n^T - 2 (v v^T)^T = I_n - 2 (v^{T^T} v^T) = I_n - 2v v^T = B$$
        Yes, $B$ is symmetric for any $v$.
    *   **Orthogonality:**
        $$B^T B = B B = (I_n - 2 v v^T)(I_n - 2 v v^T) = I_n - 4 v v^T + 4 v (v^T v) v^T = I_n - 4(1 - v^T v) v v^T$$
        For $B^T B = I_n$, we need $4(1 - v^T v) v v^T = 0 \implies v^T v = \|v\|_2^2 = 1$.
        Therefore, $B$ is orthogonal **only** when $v$ is a unit vector. It is **not** orthogonal for *any* general vector $v$.
2.  **Orthogonality of $C = I_n - 2 w w^T$ when $\|w\|_2 = 1$:**
    *   Since $\|w\|_2 = 1 \implies w^T w = 1$.
        $$C^T C = (I_n - 2 w w^T)(I_n - 2 w w^T) = I_n - 4 w w^T + 4 w (w^T w) w^T$$
        Substitute $w^T w = 1$:
        $$C^T C = I_n - 4 w w^T + 4 w (1) w^T = I_n - 4 w w^T + 4 w w^T = I_n \quad \text{(Orthogonal!)}$$

---

*Keep this master key open on your machine. Every single numerical and theoretical pattern from the official previous years' papers is fully detailed here. Re-reading this tonight ensures you are prepared for whatever gets thrown at you tomorrow at 9:00 AM! Good luck!*
