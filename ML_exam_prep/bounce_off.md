# Machine Learning (AIMLZG565) — Mid-Sem Bounce-Off & Less Predictable Topics

This guide is specifically compiled to prepare you for **less predictable but highly syllabus-relevant exam questions**. These topics are clearly present in the provided lecture slides and student practice materials, but have appeared rarely or never in the recent PQPs. 

Reviewing these topics ensures you are not caught unprepared if the examiner shifts patterns.

---

## 1. Normalization by Decimal Scaling
* **Module:** Module 2 — ML Workflow & Preprocessing
* **Where it appears in notes:** `ML_CS 2_1Aug2026 .pdf`, slide/page 48.
* **PQP Frequency:** **Never seen in PQPs.**
* **Why it is a potential unexpected question:** It is the third feature scaling technique alongside Min-Max and Z-score standardization. Since those two are heavily tested, decimal scaling is a natural candidate for an alternative question.
* **What I should know:** It normalizes by moving the decimal point of values of attribute $A$. The number of decimal places moved depends on the maximum absolute value in the dataset.
* **Important Formula:** 
  $$v' = \frac{v}{10^j}$$
  where $j$ is the smallest integer such that $\max(|v'|) < 1$.
* **Possible Exam Question:**
  Normalize the following set of values of a numerical attribute $A$ using normalization by decimal scaling:
  $$\text{Data} = \{-830, -12, 45, 120, 750, 910\}$$
* **Short Answer/Solution:**
  - **Step 1: Identify the maximum absolute value in the dataset:**
    $$\max(|v|) = |-830| \text{ or } |910| = 910$$
  - **Step 2: Determine the value of $j$:**
    We need to find the smallest integer $j$ such that $\frac{910}{10^j} < 1$.
    - If $j = 1 \implies \frac{910}{10} = 91 \not< 1$
    - If $j = 2 \implies \frac{910}{100} = 9.1 \not< 1$
    - If $j = 3 \implies \frac{910}{1000} = 0.91 < 1$
    - Therefore, $j = 3$.
  - **Step 3: Normalize each value using $v' = \frac{v}{10^3} = \frac{v}{1000}$:**
    - $-830 \rightarrow -0.830$
    - $-12 \rightarrow -0.012$
    - $45 \rightarrow 0.045$
    - $120 \rightarrow 0.120$
    - $750 \rightarrow 0.750$
    - $910 \rightarrow 0.910$

---

## 2. Gain Ratio & Split Information (Tackling ID3 High-Cardinality Bias)
* **Module:** Module 5 — Decision Tree
* **Where it appears in notes:** `ML_ CS 78_6th12th Sep2026.pdf`, slides/pages 591 to 630.
* **PQP Frequency:** **Never seen in PQPs.** (PQPs only ask for standard Information Gain or parent/child entropy).
* **Why it is a potential unexpected question:** Information Gain is heavily criticized for favoring high-cardinality nominal attributes (e.g., Customer ID, transactional keys) which split data into meaningless pure leaf nodes. Gain Ratio is the standard algorithmic remedy (used in C4.5).
* **What I should know:** Gain Ratio balances Information Gain by dividing it by **Split Information**, which measures the entropy of the attribute values itself (how broadly and evenly the attribute splits the instances).
* **Important Formulas:**
  $$\text{SplitInfo}(S, A) = -\sum_{i=1}^d \frac{|S_i|}{|S|} \log_2 \left(\frac{|S_i|}{|S|}\right)$$
  $$\text{GainRatio}(S, A) = \frac{\text{Gain}(S, A)}{\text{SplitInfo}(S, A)}$$
* **Possible Exam Question:**
  A dataset has 10 customer records. You want to evaluate if splitting on a unique "Customer ID" feature (where each of the 10 rows has a distinct numeric ID) is appropriate. Compute the Split Information and explain why the Gain Ratio is a better metric than standard Information Gain.
* **Short Answer/Solution:**
  - **Step 1: Understand the split under Customer ID:**
    Splitting on "Customer ID" creates 10 branches, each with exactly $1$ instance ($|S_i| = 1$ for all $10$ branches, and total instances $|S| = 10$).
  - **Step 2: Calculate Split Information:**
    $$\text{SplitInfo}(S, \text{CustomerID}) = -\sum_{i=1}^{10} \frac{1}{10} \log_2 \left(\frac{1}{10}\right)$$
    $$\text{SplitInfo}(S, \text{CustomerID}) = -10 \times \left[ 0.1 \times \log_2(0.1) \right] = -\log_2(0.1) = \log_2(10) \approx \mathbf{3.3219}$$
  - **Step 3: Analyze the Gain Ratio impact:**
    Since each leaf node has exactly $1$ instance, every leaf is perfectly pure ($\text{Entropy}(S_i) = 0$). Thus, the weighted entropy after split is $0$, leading to a perfect standard Information Gain ($\text{Gain}(S, A) = \text{Entropy}(S)$).
    However, the **Split Information** is extremely high ($3.3219$).
    Dividing the Information Gain by this high SplitInfo heavily penalizes this split, making the **Gain Ratio** extremely low. This prevents the decision tree from selecting this overfitting, non-generalizable feature as the root node.

---

## 3. Outlier Detection using Inter-Quartile Range (IQR)
* **Module:** Module 2 — ML Workflow & Preprocessing
* **Where it appears in notes:** `ML_CS 2_1Aug2026 .pdf`, slide/page 39.
* **PQP Frequency:** **Never seen in PQPs.** (PQPs only ask about NaN missing value imputations or Z-score normalization).
* **Why it is a potential unexpected question:** Outlier detection is a major part of data cleansing in the ML workflow, and IQR is the standard non-parametric method. It is highly testable as a short numerical problem.
* **What I should know:** You must first sort the data, find the median ($Q_2$), then the lower quartile ($Q_1$, median of lower half) and upper quartile ($Q_3$, median of upper half).
* **Important Formulas:**
  $$\text{IQR} = Q_3 - Q_1$$
  $$\text{Lower Bound} = Q_1 - 1.5 \times \text{IQR}$$
  $$\text{Upper Bound} = Q_3 + 1.5 \times \text{IQR}$$
* **Possible Exam Question:**
  Find any outliers in the following dataset using the Inter-Quartile Range (IQR) method:
  $$\text{Raw Data} = [14, 12, 11, 22, 15, 11, 10, 17, 12, 14, 13, 11]$$
* **Short Answer/Solution:**
  - **Step 1: Sort the data in ascending order:**
    $$\text{Sorted Data} = [10, 11, 11, 11, 12, 12, 13, 14, 14, 15, 17, 22]$$
  - **Step 2: Find the Median ($Q_2$):**
    With $n = 12$ instances, the median is the average of the 6th and 7th elements:
    $$\text{Median } (Q_2) = \frac{12 + 13}{2} = 12.5$$
  - **Step 3: Find Lower Quartile ($Q_1$) and Upper Quartile ($Q_3$):**
    - The lower half consists of 6 elements: `[10, 11, 11, 11, 12, 12]`. Median of this lower half is $Q_1 = \frac{11 + 11}{2} = \mathbf{11}$.
    - The upper half consists of 6 elements: `[13, 14, 14, 15, 17, 22]`. Median of this upper half is $Q_3 = \frac{14 + 15}{2} = \mathbf{14.5}$.
  - **Step 4: Calculate IQR:**
    $$\text{IQR} = Q_3 - Q_1 = 14.5 - 11 = \mathbf{3.5}$$
  - **Step 5: Compute Outlier Detection Bounds:**
    - $\text{Lower Bound} = Q_1 - 1.5 \times \text{IQR} = 11 - 1.5(3.5) = 11 - 5.25 = \mathbf{5.75}$
    - $\text{Upper Bound} = Q_3 + 1.5 \times \text{IQR} = 14.5 + 1.5(3.5) = 14.5 + 5.25 = \mathbf{19.75}$
  - **Step 6: Identify Outliers:**
    Any data point $< 5.75$ or $> 19.75$ is classified as an outlier.
    - $22$ is greater than $19.75$, so **22 is an outlier**. All other data points are within the bounds.

---

## 4. Elastic Net Regularization
* **Module:** Module 3 — Linear Models for Regression
* **Where it appears in notes:** `ML_CS 34_916 Aug 2026.pdf`, slides/pages 664 to 680; `Regularization practice numbericals.pdf` Q10.
* **PQP Frequency:** **Never seen in PQPs.** (PQPs only ask about Ridge/Lasso bias-variance trade-offs).
* **Why it is a potential unexpected question:** Elastic Net is a major regularization method that linearly combines both L1 (Lasso) and L2 (Ridge) penalties. It has a specific "mix ratio" $r$ parameter that is highly testable numerically.
* **What I should know:** It balances between Ridge and Lasso. If $r = 0$, it is equivalent to Ridge. If $r = 1$, it is equivalent to Lasso. It is highly preferred when dealing with multiple highly correlated features.
* **Important Formula:**
  $$\mathcal{L}_{\text{ElasticNet}} = \text{Base Loss} + \lambda \left[ r \sum_{j=1}^m |\theta_j| + (1 - r) \sum_{j=1}^m \theta_j^2 \right]$$
* **Possible Exam Question:**
  A regression model has weights $\theta_1 = 2$ and $\theta_2 = -3$. The base mean squared error loss is $7$. Calculate the total Elastic Net regularized loss if the regularization constant $\lambda = 0.5$ and the L1 mix ratio $r = 0.6$.
* **Short Answer/Solution:**
  - **Step 1: Compute absolute sum of weights (L1 penalty):**
    $$\sum |\theta_j| = |2| + |-3| = 5$$
  - **Step 2: Compute squared sum of weights (L2 penalty):**
    $$\sum \theta_j^2 = (2)^2 + (-3)^2 = 4 + 9 = 13$$
  - **Step 3: Substitute values into Elastic Net penalty formula:**
    $$\text{Penalty} = \lambda \left[ r \sum |\theta_j| + (1 - r) \sum \theta_j^2 \right]$$
    $$\text{Penalty} = 0.5 \left[ 0.6(5) + (1 - 0.6)(13) \right]$$
    $$\text{Penalty} = 0.5 \left[ 3.0 + 0.4(13) \right] = 0.5 \left[ 3.0 + 5.2 \right] = 0.5 \times 8.2 = \mathbf{4.1}$$
  - **Step 4: Compute Total Loss:**
    $$\mathcal{L}_{\text{ElasticNet}} = \text{Base Loss} + \text{Penalty} = 7 + 4.1 = \mathbf{11.1}$$

---

## 5. Weight Updates under Ridge and Lasso in Gradient Descent
* **Module:** Module 3 — Linear Models for Regression
* **Where it appears in notes:** `ML_CS 34_916 Aug 2026.pdf` page/slide 100-101; `regularization_15_solutions_line_by_line.pdf` Q6, Q7.
* **PQP Frequency:** **Never seen in PQPs.** (PQPs only ask for standard gradient updates on unregularized models).
* **Why it is a potential unexpected question:** BITS Pilani exams heavily emphasize weight updates, and sliding in L1/L2 penalties is a common way to test advanced learners.
* **What I should know:** How the penalty terms mathematically shift the gradient update formula.
* **Important Formulas:**
  - **Ridge (L2) Weight Update:**
    $$\theta_{\text{new}} = \theta_{\text{old}} (1 - \alpha \lambda) - \alpha \frac{\partial \mathcal{L}_{\text{base}}}{\partial \theta}$$
  - **Lasso (L1) Weight Update (for positive $\theta$):**
    $$\theta_{\text{new}} = \theta_{\text{old}} - \alpha \left( \frac{\partial \mathcal{L}_{\text{base}}}{\partial \theta} + \lambda \right)$$
* **Possible Exam Question:**
  Given a model parameter weight $w = 5$, learning rate $\eta = 0.1$, unregularized base loss partial derivative $\frac{\partial \mathcal{L}}{\partial w} = 2$, and L2 regularization constant $\lambda = 0.5$. Calculate the new weight after one gradient descent update under L2 regularization.
* **Short Answer/Solution:**
  - **Step 1: Recall the L2 Gradient update formula:**
    $$w_{\text{new}} = w - \eta \left( \frac{\partial \mathcal{L}}{\partial w} + 2 \lambda w \right)$$
    *(Note: Using standard Ridge formulation $\mathcal{L}_{\text{Ridge}} = \mathcal{L}_{\text{base}} + \lambda w^2$, the derivative is $\frac{\partial \mathcal{L}}{\partial w} + 2\lambda w$).*
  - **Step 2: Substitute values:**
    $$2 \lambda w = 2 \times 0.5 \times 5 = 5$$
    $$\text{Total Gradient} = \frac{\partial \mathcal{L}}{\partial w} + 2 \lambda w = 2 + 5 = 7$$
    $$w_{\text{new}} = 5 - (0.1 \times 7) = 5 - 0.7 = \mathbf{4.3}$$

---

## 6. Continuous Feature Splitting in CART using Mean Threshold
* **Module:** Module 5 — Decision Tree
* **Where it appears in notes:** `ML_ CS 78_6th12th Sep2026.pdf` slide 517-526.
* **PQP Frequency:** **Seen once** in `March 2026 Regular, Q6c`.
* **Why it is a potential unexpected question:** Most decision tree exercises in classrooms utilize simple nominal binary splits. Testing split calculations on a continuous variable (like house area or age) using a mean threshold is much less predictable.
* **What I should know:** Calculate the mean of the continuous column first. This splits instances into two binary branches: $\le \text{mean}$ and $> \text{mean}$. Then compute the Gini index for each branch and find the weighted Gini.
* **Possible Exam Question:**
  A continuous attribute $a_3$ has values `[1, 6, 5, 4, 7, 3, 8, 7, 5]` for 9 instances. Class labels are `[+, +, -, +, -, -, -, +, -]` (3 positives, 6 negatives). Calculate the weighted Gini Index split of $a_3$ using a mean threshold.
* **Short Answer/Solution:**
  - **Step 1: Compute Mean of $a_3$:**
    $$\mu = \frac{1 + 6 + 5 + 4 + 7 + 3 + 8 + 7 + 5}{9} = \frac{46}{9} \approx 5.11$$
  - **Step 2: Split data into two subsets based on threshold $\le 5.11$ vs. $> 5.11$:**
    - **Branch 1 ($a_3 \le 5.11$):** Instances `[1, 5, 4, 3, 5]` (Instance IDs: 1, 3, 4, 6, 9)
      - Labels: `[+, -, +, -, -]` (2 Positives, 3 Negatives, Total $n_1 = 5$)
      - $p_+ = 2/5 = 0.4, \quad p_- = 3/5 = 0.6$
      - $\text{Gini}(\text{Branch 1}) = 1 - (0.4^2 + 0.6^2) = 1 - (0.16 + 0.36) = 1 - 0.52 = \mathbf{0.48}$
    - **Branch 2 ($a_3 > 5.11$):** Instances `[6, 7, 8, 7]` (Instance IDs: 2, 5, 7, 8)
      - Labels: `[+, -, -, +]` (2 Positives, 2 Negatives, Total $n_2 = 4$)
      - $p_+ = 2/4 = 0.5, \quad p_- = 2/4 = 0.5$
      - $\text{Gini}(\text{Branch 2}) = 1 - (0.5^2 + 0.5^2) = 1 - (0.25 + 0.25) = 1 - 0.50 = \mathbf{0.50}$
  - **Step 3: Compute Weighted Gini Index after Split:**
    $$\text{Gini}_{\text{split}}(S, a_3) = \frac{5}{9} \text{Gini}(\text{Branch 1}) + \frac{4}{9} \text{Gini}(\text{Branch 2})$$
    $$\text{Gini}_{\text{split}}(S, a_3) = \left(\frac{5}{9} \times 0.48\right) + \left(\frac{4}{9} \times 0.50\right) = 0.2667 + 0.2222 = \mathbf{0.4889}$$
