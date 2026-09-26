# ISM Exam Preparation Practice Set & Mid-Sem Solutions

This document contains a comprehensive set of practice questions compiled from the 7 lecture notes PDFs (found at the end of each slide deck) along with their complete step-by-step solutions, and detailed step-by-step solutions to the Mid-Semester Exam Questions (19 Sep 2026).

---

## Part 1: Solutions to Mid-Semester Exam Questions (19 Sep 2026)

### Question 1

#### **a) ER Waiting Times**
A hospital records ER waiting times (minutes) for 15 patients:
$$5, 8, 9, 12, 14, 15, 16, 18, 20, 22, 25, 28, 30, 45, 95$$

**(i) Compute the five-number summary and IQR.**
* **Sorted Data ($n = 15$):**
  $$5, 8, 9, 12, 14, 15, 16, 18, 20, 22, 25, 28, 30, 45, 95$$
* **Min:** $5$ minutes.
* **Q1 (First Quartile):** Position of Q1 $= \frac{15+1}{4} = 4^{\text{th}}$ element.
  $$\text{Q1} = 12 \text{ minutes}$$
* **Median (Q2):** Position of Median $= \frac{15+1}{2} = 8^{\text{th}}$ element.
  $$\text{Median} = 18 \text{ minutes}$$
* **Q3 (Third Quartile):** Position of Q3 $= \frac{3(15+1)}{4} = 12^{\text{th}}$ element.
  $$\text{Q3} = 28 \text{ minutes}$$
* **Max:** $95$ minutes.
* **Interquartile Range (IQR):**
  $$\text{IQR} = \text{Q3} - \text{Q1} = 28 - 12 = 16 \text{ minutes}$$

**(ii) Identify any outliers using the $1.5 \times \text{IQR}$ rule.**
* **Lower Limit (Fence):**
  $$\text{Q1} - 1.5 \times \text{IQR} = 12 - 1.5 \times 16 = 12 - 24 = -12$$
* **Upper Limit (Fence):**
  $$\text{Q3} + 1.5 \times \text{IQR} = 28 + 1.5 \times 16 = 28 + 24 = 52$$
* **Outlier Analysis:**
  * Since waiting times cannot be negative, there are no lower outliers.
  * Any observation greater than $52$ is an outlier.
  * Looking at the data, **$95$ is an outlier** because $95 > 52$.

**(iii) Discuss whether the mean or median is a better measure of central tendency here, and why.**
* **Median** is a much better measure of central tendency here.
* **Reasoning:** The dataset contains an extreme outlier ($95$ minutes). The **mean** is highly sensitive to outliers—it is calculated as:
  $$\text{Mean} = \frac{5+8+9+12+14+15+16+18+20+22+25+28+30+45+95}{15} = \frac{362}{15} \approx 24.13 \text{ minutes}$$
  This mean of $24.13$ minutes is larger than $10$ out of the $15$ observations, making it non-representative of the typical patient's waiting time.
  The **median** ($18$ minutes) is resistant (robust) to outliers and accurately represents the central/typical waiting time.

---

#### **b) Smartphone App Usage**
Among smartphone users, $65\%$ use App A, $40\%$ use App B, and $25\%$ use neither app.
* **Given Probabilities:**
  * $P(A) = 0.65$
  * $P(B) = 0.40$
  * $P(\text{neither}) = P(A' \cap B') = 0.25$

**(i) Find $P(\text{use at least one app})$.**
* The event "use at least one app" is the union $A \cup B$.
  $$P(A \cup B) = 1 - P(A' \cap B') = 1 - 0.25 = 0.75 \text{ (or } 75\% \text{)}$$

**(ii) Find $P(\text{use App B} \mid \text{uses App A})$.**
* We first need the joint probability $P(A \cap B)$. From the union formula:
  $$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$
  $$0.75 = 0.65 + 0.40 - P(A \cap B) \implies P(A \cap B) = 1.05 - 0.75 = 0.30 \text{ (or } 30\% \text{)}$$
* Now, compute the conditional probability:
  $$P(B \mid A) = \frac{P(A \cap B)}{P(A)} = \frac{0.30}{0.65} = \frac{30}{65} = \frac{6}{13} \approx 0.4615 \text{ (or } 46.15\% \text{)}$$

**(iii) Are using App A and App B independent events? Justify with a calculation.**
* For events A and B to be independent, they must satisfy:
  $$P(A \cap B) = P(A) \times P(B)$$
* Let's calculate the product:
  $$P(A) \times P(B) = 0.65 \times 0.40 = 0.26$$
* Since $P(A \cap B) = 0.30 \neq 0.26$, the events App A usage and App B usage are **dependent (not independent)**. Since $P(A \cap B) > P(A) P(B)$, using App A makes a person more likely to use App B.

---

### Question 2

#### **a) University Scholarship (Bayes' Theorem)**
A university has three departments $D_1$, $D_2$, and $D_3$, containing $40\%$, $35\%$, and $25\%$ of the total students. The probabilities of receiving a scholarship are $0.10$, $0.20$, and $0.30$, respectively.
* **Prior Probabilities:**
  * $P(D_1) = 0.40$, $P(D_2) = 0.35$, $P(D_3) = 0.25$
* **Conditional Probabilities of Scholarship ($S$):**
  * $P(S \mid D_1) = 0.10$, $P(S \mid D_2) = 0.20$, $P(S \mid D_3) = 0.30$

Determine the probability that a student who received a scholarship belongs to $D_2$ (i.e., $P(D_2 \mid S)$).

* **Step 1: Total Probability of receiving a scholarship, $P(S)$:**
  $$P(S) = P(S \mid D_1)P(D_1) + P(S \mid D_2)P(D_2) + P(S \mid D_3)P(D_3)$$
  $$P(S) = (0.10 \times 0.40) + (0.20 \times 0.35) + (0.30 \times 0.25)$$
  $$P(S) = 0.040 + 0.070 + 0.075 = 0.185 \text{ (or } 18.5\% \text{)}$$
* **Step 2: Apply Bayes' Theorem:**
  $$P(D_2 \mid S) = \frac{P(S \mid D_2) P(D_2)}{P(S)} = \frac{0.20 \times 0.35}{0.185} = \frac{0.070}{0.185} = \frac{70}{185} = \frac{14}{37} \approx 0.3784 \text{ (or } 37.84\% \text{)}$$

---

#### **b) Discrete Probability Distribution**
A discrete random variable $X$ has the following probability distribution:
| $X$ | 0 | 1 | 2 | 3 | 4 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $P(X = x)$ | 0.10 | 0.25 | $K$ | 0.20 | 0.15 |

**(i) Find the value of $K$ that makes this a valid probability distribution.**
* The sum of all probabilities in a valid PMF must equal $1$:
  $$\sum P(X = x) = 0.10 + 0.25 + K + 0.20 + 0.15 = 1$$
  $$0.70 + K = 1 \implies K = 0.30$$

**(ii) Find $P(X \le 2)$ and $P(X > 2)$.**
* **$P(X \le 2)$:**
  $$P(X \le 2) = P(X = 0) + P(X = 1) + P(X = 2) = 0.10 + 0.25 + 0.30 = 0.65$$
* **$P(X > 2)$:**
  $$P(X > 2) = 1 - P(X \le 2) = 1 - 0.65 = 0.35$$
  *(Or: $P(X = 3) + P(X = 4) = 0.20 + 0.15 = 0.35$)*

**(iii) Find the expected value $E(X)$ and variance $\text{Var}(X)$.**
* **Expected Value $E(X)$:**
  $$E(X) = \sum x P(X = x)$$
  $$E(X) = 0(0.10) + 1(0.25) + 2(0.30) + 3(0.20) + 4(0.15)$$
  $$E(X) = 0 + 0.25 + 0.60 + 0.60 + 0.60 = 2.05$$
* **Second Moment $E(X^2)$:**
  $$E(X^2) = \sum x^2 P(X = x)$$
  $$E(X^2) = 0^2(0.10) + 1^2(0.25) + 2^2(0.30) + 3^2(0.20) + 4^2(0.15)$$
  $$E(X^2) = 0 + 0.25 + 1.20 + 1.80 + 2.40 = 5.65$$
* **Variance $\text{Var}(X)$:**
  $$\text{Var}(X) = E(X^2) - [E(X)]^2 = 5.65 - (2.05)^2 = 5.65 - 4.2025 = 1.4475$$

---

### Question 3

#### **a) Plant Growth (Normal Distribution)**
The growth of young plants is normally distributed with mean $\mu = 100\text{ cm}$ and standard deviation $\sigma = 15\text{ cm}$.
$$X \sim N(100, 15^2)$$

**(i) What is the probability that a randomly selected plant has a growth above $130\text{ cm}$?**
* **Convert to Z-score:**
  $$Z = \frac{X - \mu}{\sigma} = \frac{130 - 100}{15} = \frac{30}{15} = 2.00$$
* **Find Probability:**
  $$P(X > 130) = P(Z > 2.00) = 1 - \Phi(2.00) = 1 - 0.9772 = 0.0228 \text{ (or } 2.28\% \text{)}$$

**(ii) What is the probability that its growth falls between $85\text{ cm}$ and $115\text{ cm}$?**
* **Convert both limits to Z-scores:**
  $$Z_{\text{lower}} = \frac{85 - 100}{15} = -1.00$$
  $$Z_{\text{upper}} = \frac{115 - 100}{15} = +1.00$$
* **Find Probability:**
  $$P(85 < X < 115) = P(-1.00 < Z < 1.00) = \Phi(1.00) - \Phi(-1.00) = 0.8413 - 0.1587 = 0.6826 \text{ (or } 68.26\% \text{)}$$

**(iii) What is the probability that its growth falls below $120\text{ cm}$?**
* **Convert to Z-score:**
  $$Z = \frac{120 - 100}{15} = \frac{20}{15} \approx 1.33$$
* **Find Probability:**
  $$P(X < 120) = P(Z < 1.33) = \Phi(1.33) \approx 0.9082 \text{ (or } 90.82\% \text{)}$$

---

#### **b) Call Center Complaint Escalations (Binomial Distribution)**
$8\%$ of incoming calls result in a customer complaint escalation ($p = 0.08$). A shift handles 25 calls ($n = 25$).
$$X \sim \text{Binomial}(n = 25, p = 0.08)$$

**(i) Find the probability that exactly 3 calls are escalated.**
* **Formula:** $P(X = x) = \binom{n}{x} p^x (1-p)^{n-x}$
  $$P(X = 3) = \binom{25}{3} (0.08)^3 (0.92)^{22}$$
  $$\binom{25}{3} = \frac{25 \times 24 \times 23}{3 \times 2 \times 1} = 2300$$
  $$P(X = 3) = 2300 \times 0.000512 \times 0.1597 \approx 0.1881 \text{ (or } 18.81\% \text{)}$$

**(ii) Find the probability that at most 2 calls are escalated.**
* $$P(X \le 2) = P(X = 0) + P(X = 1) + P(X = 2)$$
  * $P(X = 0) = (0.92)^{25} \approx 0.1244$
  * $P(X = 1) = \binom{25}{1} (0.08)^1 (0.92)^{24} = 25 \times 0.08 \times 0.1352 \approx 0.2704$
  * $P(X = 2) = \binom{25}{2} (0.08)^2 (0.92)^{23} = 300 \times 0.0064 \times 0.1470 \approx 0.2821$
* **Sum:**
  $$P(X \le 2) = 0.1244 + 0.2704 + 0.2821 \approx 0.6768 \text{ (or } 67.68\% \text{)}$$

**(iii) Find the probability that at least 1 call is escalated.**
* $$P(X \ge 1) = 1 - P(X = 0) = 1 - 0.1244 = 0.8756 \text{ (or } 87.56\% \text{)}$$

---

### Question 4

#### **a) Naive Bayes Classification**
Given the training dataset:
| Age Group | Income | Purchase |
| :---: | :---: | :---: |
| Young | High | Yes |
| Middle | Low | Yes |
| Senior | High | No |
| Young | Low | No |
| Middle | Medium | Yes |
| Senior | High | No |
| Young | Medium | Yes |
| Middle | Medium | Yes |
| Senior | Medium | No |
| Young | High | No |

Predict whether a new customer with **`Age Group = Young`** and **`Income = Medium`** will purchase the product using Naive Bayes.

* **Step 1: Calculate Prior Probabilities of class `Purchase`:**
  * Total training records $= 10$
  * $P(\text{Purchase} = \text{Yes}) = \frac{5}{10} = 0.50$
  * $P(\text{Purchase} = \text{No}) = \frac{5}{10} = 0.50$

* **Step 2: Calculate Conditional Probabilities for class `Yes`:**
  * Out of the 5 `Yes` records:
    * `Age Group = Young` appears 2 times $\implies P(\text{Young} \mid \text{Yes}) = \frac{2}{5} = 0.40$
    * `Income = Medium` appears 3 times $\implies P(\text{Medium} \mid \text{Yes}) = \frac{3}{5} = 0.60$

* **Step 3: Calculate Conditional Probabilities for class `No`:**
  * Out of the 5 `No` records:
    * `Age Group = Young` appears 2 times $\implies P(\text{Young} \mid \text{No}) = \frac{2}{5} = 0.40$
    * `Income = Medium` appears 1 time $\implies P(\text{Medium} \mid \text{No}) = \frac{1}{5} = 0.20$

* **Step 4: Compute Posterior Numerators (Priors $\times$ Likelihoods):**
  * **For `Yes`:**
    $$\text{Posterior}_{\text{Yes}} \propto P(\text{Yes}) \times P(\text{Young} \mid \text{Yes}) \times P(\text{Medium} \mid \text{Yes}) = 0.50 \times 0.40 \times 0.60 = 0.12$$
  * **For `No`:**
    $$\text{Posterior}_{\text{No}} \propto P(\text{No}) \times P(\text{Young} \mid \text{No}) \times P(\text{Medium} \mid \text{No}) = 0.50 \times 0.40 \times 0.20 = 0.04$$

* **Step 5: Normalize and Predict:**
  * Sum of numerators $= 0.12 + 0.04 = 0.16$
  * Probability of `Yes`:
    $$P(\text{Yes} \mid \text{Young, Medium}) = \frac{0.12}{0.16} = 0.75 \text{ (or } 75\% \text{)}$$
  * Probability of `No`:
    $$P(\text{No} \mid \text{Young, Medium}) = \frac{0.04}{0.16} = 0.25 \text{ (or } 25\% \text{)}$$
  * **Final Prediction:** Since $0.75 > 0.25$, the customer is predicted to **Purchase (Yes)**.

---

#### **b) Class Exam Pass Rate (Binomial Justification)**
A class has 50 students where each student has a probability $p$ of passing an exam. The exam is proctored to prevent copying.

**What is the expected number of students who pass?**
* By linearity of expectation, the expected number of students who pass is:
  $$E(X) = n p = 50p$$

**Can we use the Binomial distribution to model this? If so, find $P(X = 10)$. Otherwise, explain why not.**
* **Yes, we can use the Binomial distribution here.**
* **Justification (4 Core Conditions Met):**
  1. **Fixed Number of Trials ($n$):** There are exactly $50$ independent trials (students), so $n = 50$.
  2. **Two Mutually Exclusive Outcomes:** Each trial results in either "Pass" (Success) or "Fail" (Failure).
  3. **Constant Probability of Success ($p$):** Each student has the same constant probability $p$ of passing.
  4. **Independence of Trials:** Because the exam is proctored, students cannot copy from one another. This eliminates copying-based dependencies. Thus, one student's success is independent of another's.
* **Probability that exactly 10 pass:**
  $$P(X = 10) = \binom{50}{10} p^{10} (1-p)^{40}$$

---

### Question 5

#### **a) Sensor Lifetime (Continuous PDF)**
The continuous lifetime (years) follows the PDF:
$$f(x) = \begin{cases} 2kx & \text{for } 0 \le x \le 5 \\ 0 & \text{otherwise} \end{cases}$$

**(i) Find the value of $k$ that makes this a valid PDF.**
* For $f(x)$ to be a valid PDF, the total area under the curve must equal $1$:
  $$\int_{0}^{5} f(x) \, dx = 1 \implies \int_{0}^{5} 2kx \, dx = 1$$
  $$\left[ kx^2 \right]_{0}^{5} = 1 \implies k(5^2) - k(0^2) = 1 \implies 25k = 1 \implies k = \frac{1}{25} = 0.04$$

**(ii) Find $P(2 < X < 4)$.**
* Integrate $f(x)$ from $2$ to $4$ using $k = 0.04$:
  $$P(2 < X < 4) = \int_{2}^{4} 2(0.04)x \, dx = \left[ 0.04x^2 \right]_{2}^{4} = 0.04(4^2 - 2^2) = 0.04(16 - 4) = 0.04(12) = 0.48$$

**(iii) Find $E(X)$ and $\text{Var}(X)$.**
* **Expected Value $E(X)$:**
  $$E(X) = \int_{0}^{5} x f(x) \, dx = \int_{0}^{5} 2kx^2 \, dx = \left[ \frac{2k}{3} x^3 \right]_{0}^{5} = \frac{2(0.04)}{3} (125) = \frac{10}{3} \approx 3.333 \text{ years}$$
* **Second Moment $E(X^2)$:**
  $$E(X^2) = \int_{0}^{5} x^2 f(x) \, dx = \int_{0}^{5} 2kx^3 \, dx = \left[ \frac{2k}{4} x^4 \right]_{0}^{5} = \frac{0.08}{4} (625) = 0.02 \times 625 = 12.5$$
* **Variance $\text{Var}(X)$:**
  $$\text{Var}(X) = E(X^2) - [E(X)]^2 = 12.5 - \left(\frac{10}{3}\right)^2 = 12.5 - \frac{100}{9} = \frac{12.5 \times 9 - 100}{9} = \frac{112.5 - 100}{9} = \frac{12.5}{9} = \frac{25}{18} \approx 1.389 \text{ years}^2$$

---

#### **b) Telecom Call Times (Central Limit Theorem)**
The time customers spend on a call has a right-skewed (non-normal) distribution with mean $\mu = 8\text{ minutes}$ and standard deviation $\sigma = 3\text{ minutes}$. A random sample of $n = 100$ calls is taken.

**(i) Can you still apply the Central Limit Theorem here? Justify why or why not, given the population isn't normal.**
* **Yes, we can definitely apply the Central Limit Theorem (CLT).**
* **Justification:** The CLT states that as long as the sample size $n$ is sufficiently large, the sampling distribution of the sample mean $\bar{X}$ will be approximately normally distributed, regardless of the shape of the underlying population distribution. The standard rule of thumb is that **$n \ge 30$** is large enough. Since $n = 100 > 30$, the CLT holds, and $\bar{X}$ can be treated as approximately normal:
  $$\bar{X} \sim N\left(\mu, \frac{\sigma^2}{n}\right) \implies \bar{X} \sim N\left(8, \frac{3^2}{100}\right) \implies \bar{X} \sim N(8, 0.3^2)$$

**(ii) Find the probability that the sample mean call time exceeds $8.5\text{ minutes}$.**
* **Standard Error ($\sigma_{\bar{X}}$):**
  $$\sigma_{\bar{X}} = \frac{\sigma}{\sqrt{n}} = \frac{3}{\sqrt{100}} = 0.3 \text{ minutes}$$
* **Convert $\bar{X} = 8.5$ to a Z-score:**
  $$Z = \frac{\bar{X} - \mu}{\sigma_{\bar{X}}} = \frac{8.5 - 8}{0.3} = \frac{0.5}{0.3} = \frac{5}{3} \approx 1.67$$
* **Find Probability:**
  $$P(\bar{X} > 8.5) = P(Z > 1.67) = 1 - \Phi(1.67) = 1 - 0.9525 = 0.0475 \text{ (or } 4.75\% \text{)}$$
  *(Using more precise decimal places: $P(Z > 1.6667) \approx 0.0478$)*

**(iii) How would your answer to (i) change if the sample size were only $n = 15$ instead of 100?**
* If $n = 15$, **we cannot apply the Central Limit Theorem.**
* **Explanation:** Since $n = 15 < 30$ (which is a small sample) and the underlying population is right-skewed and non-normal, the sampling distribution of the sample mean will remain skewed and will **not** be approximately normal. Thus, normal distribution probability calculations would be invalid.

---
---

## Part 2: Lecture-Wise Practice Problems with Solutions

### Lecture 1: Descriptive Statistics (S1-26_ISM_CS-1_PPT.pdf)

#### **Question 1 (Noise Levels in an Office)**
A sample of 77 office workers experiences the following noise levels (dBA):
*Data (Sorted):*
$$55.3, 55.3, 55.3, 55.9, 55.9, 55.9, 55.9, 56.1, 56.1, 56.1, 56.1, 56.1, 56.1, 56.8, 56.8, 57.0, 57.0, 57.0, 57.8, 57.8, 57.8, 57.9, 57.9, 57.9, 58.8, 58.8, 58.8, 59.8, 59.8, 59.8, 62.2, 62.2, 63.8, 63.8, 63.8, 63.9, 63.9, 63.9, 64.7, 64.7, 64.7, 65.1, 65.1, 65.1, 65.3, 65.3, 65.3, 65.3, 67.4, 67.4, 67.4, 67.4, 68.7, 68.7, 68.7, 68.7, 69.0, 70.4, 70.4, 71.2, 71.2, 71.2, 73.0, 73.0, 73.1, 73.1, 74.6, 74.6, 74.6, 74.6, 79.3, 79.3, 79.3, 79.3, 83.0, 83.0, 83.0$$

**a) Find the Arithmetic Mean, Variance, SD, and IQR.**
* **Mean ($\bar{X}$):** Sum of all 77 elements $= 4996.3$ dBA.
  $$\bar{X} = \frac{4996.3}{77} \approx 64.89 \text{ dBA}$$
* **Sample Variance ($s^2$):**
  $$s^2 = \frac{\sum (X_i - \bar{X})^2}{n-1} \approx 60.88 \text{ dBA}^2$$
* **Sample Standard Deviation ($s$):**
  $$s = \sqrt{s^2} = \sqrt{60.88} \approx 7.80 \text{ dBA}$$
* **Quartiles (Position of Quartile $= \frac{p \cdot (n+1)}{100}$):**
  * Position of Q1 $= \frac{1(77+1)}{4} = 19.5 \implies$ Average of $19^{\text{th}}$ and $20^{\text{th}}$ sorted values.
    $$19^{\text{th}} \text{ value} = 57.8, \quad 20^{\text{th}} \text{ value} = 57.8 \implies \text{Q1} = 57.80 \text{ dBA}$$
  * Median (Q2): Position $= \frac{77+1}{2} = 39^{\text{th}}$ sorted value.
    $$\text{Median} = 64.70 \text{ dBA}$$
  * Position of Q3 $= \frac{3(77+1)}{4} = 58.5 \implies$ Average of $58^{\text{th}}$ and $59^{\text{th}}$ sorted values.
    $$58^{\text{th}} \text{ value} = 70.4, \quad 59^{\text{th}} \text{ value} = 70.4 \implies \text{Q3} = 70.40 \text{ dBA}$$
* **Interquartile Range (IQR):**
  $$\text{IQR} = \text{Q3} - \text{Q1} = 70.40 - 57.80 = 12.60 \text{ dBA}$$

**b) Draw Box and Whisker Plot (Description).**
* To construct the boxplot, we map: Min $= 55.3$, Q1 $= 57.8$, Median $= 64.7$, Q3 $= 70.4$, Max $= 83.0$.
* The central box spans from $57.8$ to $70.4$, with a vertical line at $64.7$. Whiskers extend outwards to the non-outlier limits.

**c) Comment on outliers.**
* Lower Fence $= \text{Q1} - 1.5 \times \text{IQR} = 57.80 - 1.5(12.60) = 38.90$
* Upper Fence $= \text{Q3} + 1.5 \times \text{IQR} = 70.40 + 1.5(12.60) = 89.30$
* Since the minimum data value ($55.3$) is greater than $38.90$, and the maximum data value ($83.0$) is less than $89.30$, **there are no outliers in this dataset**.

---

#### **Question 2 (Chicken Sandwiches Fat Content)**
Total fat (grams per serving) for 20 chicken sandwiches:
$$7, 8, 4, 5, 16, 20, 20, 24, 19, 30, 23, 30, 25, 19, 29, 29, 30, 30, 40, 56$$

**a) Compute Mean, Median, Q1, and Q3.**
* **Sorted Data ($n = 20$):**
  $$4, 5, 7, 8, 16, 19, 19, 20, 20, 23, 24, 25, 29, 29, 30, 30, 30, 30, 40, 56$$
* **Mean ($\bar{X}$):** Sum $= 464$.
  $$\bar{X} = \frac{464}{20} = 23.20 \text{ grams}$$
* **Median:** Average of $10^{\text{th}}$ ($23$) and $11^{\text{th}}$ ($24$) values.
  $$\text{Median} = \frac{23+24}{2} = 23.50 \text{ grams}$$
* **Quartiles (using standard positional method $\frac{p \cdot (n+1)}{4}$):**
  * Position of Q1 $= \frac{20+1}{4} = 5.25 \implies$ Weighted average of $5^{\text{th}}$ ($16$) and $6^{\text{th}}$ ($19$) values.
    $$\text{Q1} = 16 + 0.25(19 - 16) = 16.75 \text{ grams}$$
  * Position of Q3 $= \frac{3(20+1)}{4} = 15.75 \implies$ Weighted average of $15^{\text{th}}$ ($30$) and $16^{\text{th}}$ ($30$) values.
    $$\text{Q3} = 30 + 0.75(30 - 30) = 30.00 \text{ grams}$$
  *(Note: If splitting in half, the lower 10 values have median $\text{Q1} = \frac{16+19}{2} = 17.50\text{g}$, and upper 10 values have median $\text{Q3} = \frac{30+30}{2} = 30.00\text{g}$)*

**b) Compute Variance, SD, Range, IQR, and Outliers.**
* **Sample Variance ($s^2$):** $s^2 \approx 188.06 \text{ grams}^2$
* **Sample SD ($s$):** $s \approx 13.71 \text{ grams}$
* **Range:** $56 - 4 = 52 \text{ grams}$
* **IQR:** $\text{Q3} - \text{Q1} = 30.00 - 16.75 = 13.25 \text{ grams}$ (or $30 - 17.5 = 12.50$ depending on method).
* **Outliers:**
  * Lower Fence $= 16.75 - 1.5(13.25) = -3.125$ (No lower outliers)
  * Upper Fence $= 30.00 + 1.5(13.25) = 49.875$
  * Since Max $= 56 > 49.875$, **$56$ is an outlier**.

**c) Are the data skewed?**
* Yes. The mean ($23.20$) is slightly less than the median ($23.50$), but the data has a massive outlier on the right ($56$). This extreme value pulls the right tail out, meaning the data is **right-skewed (positively skewed)**.

**d) Conclusions:**
* Chicken sandwiches from fast-food chains typically contain a significant amount of fat, with a median of $23.5$ grams. The fat content is highly variable (ranging from $4$g to $56$g) and includes extremely fatty options.

---

#### **Question 3 (Battery Life)**
Battery life (in shots) of 12 digital cameras:
$$300, 180, 85, 170, 380, 460, 260, 35, 380, 120, 110, 240$$

**List the Five-point Summary.**
* **Sorted Data ($n = 12$):**
  $$35, 85, 110, 120, 170, 180, 240, 260, 300, 380, 380, 460$$
* **Min:** $35$
* **Q1:** Position $= \frac{12+1}{4} = 3.25 \implies 110 + 0.25(120-110) = 112.5$
* **Median:** Position $= 6.5 \implies \frac{180+240}{2} = 210$
* **Q3:** Position $= \frac{3(12+1)}{4} = 9.75 \implies 300 + 0.75(380-300) = 360$
* **Max:** $460$
* **Summary:** $[35, 112.5, 210, 360, 460]$

---

#### **Question 4 (Descriptive Practice)**
Data set ($n = 20$):
$$82, 45, 64, 80, 82, 74, 79, 80, 80, 78, 80, 80, 48, 73, 80, 79, 81, 70, 78, 73$$

* **Sorted Data:**
  $$45, 48, 64, 70, 73, 73, 74, 78, 78, 79, 79, 80, 80, 80, 80, 80, 80, 81, 82, 82$$
* **Five-Point Summary:**
  * **Min:** $45$
  * **Q1:** Position $= 5.25 \implies 73 + 0.25(73-73) = 73$
  * **Median:** Position $= 10.5 \implies \frac{79+79}{2} = 79$
  * **Q3:** Position $= 15.75 \implies 80 + 0.75(80-80) = 80$
  * **Max:** $82$
* **IQR:** $\text{Q3} - \text{Q1} = 80 - 73 = 7$
* **Outliers:**
  * Lower Fence $= 73 - 1.5(7) = 62.5$
  * Upper Fence $= 80 + 1.5(7) = 90.5$
  * Outliers are values $< 62.5$. Therefore, **$45$ and $48$ are outliers**.

---

#### **Question 5 (Bank Waiting Times comparison)**
* **Branch 1 (Commercial) Waiting Times ($n = 15$):**
  $$0.38, 2.34, 3.02, 3.20, 3.54, 3.79, 4.21, 4.50, 4.77, 5.12, 5.13, 5.55, 6.10, 6.19, 6.46$$
* **Branch 2 (Residential) Waiting Times ($n = 15$):**
  $$3.82, 4.08, 5.47, 5.64, 5.79, 5.90, 6.17, 6.68, 8.01, 8.02, 8.35, 8.73, 9.66, 9.91, 10.49$$

**a) List five-number summaries:**
* **Branch 1:** $\text{Min} = 0.38$, $\text{Q1} = 3.20$, $\text{Median} = 4.50$, $\text{Q3} = 5.55$, $\text{Max} = 6.46$
* **Branch 2:** $\text{Min} = 3.82$, $\text{Q1} = 5.64$, $\text{Median} = 6.68$, $\text{Q3} = 8.73$, $\text{Max} = 10.49$

**b) Construct box-and-whisker plots & describe shape:**
* **Branch 1:** Mean $= 4.29$ minutes. Since Mean ($4.29$) < Median ($4.50$), it is slightly **left-skewed**.
* **Branch 2:** Mean $= 7.11$ minutes. Since Mean ($7.11$) > Median ($6.68$), it is **right-skewed**.

**c) Similarities and Differences:**
* **Similarities:** Both branches have comparable waiting time ranges (around $6.1$ to $6.7$ minutes) and neither contains any outliers.
* **Differences:** Waiting times at the Residential branch are significantly longer (Median $= 6.68$ min) and exhibit more variability ($\text{IQR} = 3.09$ min) than the Commercial branch ($\text{Median} = 4.50$ min, $\text{IQR} = 2.35$ min).

---

### Lecture 2: Probability Concepts (S1-26_ISM_CS-2_PPT.pdf)

#### **Question 1 (Bank Account Survey)**
A survey reveals $40\%$ are savings accounts, $35\%$ current accounts, and the balance are loan accounts.
* **Prior Probabilities:** $P(\text{Savings}) = 0.40$, $P(\text{Current}) = 0.35$, $P(\text{Loan}) = 1 - 0.40 - 0.35 = 0.25$.
* **Solutions:**
  * $P(\text{Loan}) = \mathbf{0.25}$
  * $P(\text{Not Savings}) = 1 - 0.40 = \mathbf{0.60}$
  * $P(\text{Not Current}) = 1 - 0.35 = \mathbf{0.65}$
  * $P(\text{Current or Loan}) = P(\text{Current}) + P(\text{Loan}) = 0.35 + 0.25 = \mathbf{0.60}$

---

#### **Question 2 (A and B speaking the truth)**
A speaks truth in $80\%$ of cases ($P(A) = 0.80$), B in $60\%$ of cases ($P(B) = 0.60$).
What percentage of cases are they likely to contradict each other?
* **Solution:**
  They contradict when one lies and the other speaks the truth:
  $$P(\text{Contradiction}) = P(A \cap B') + P(A' \cap B) = P(A)P(B') + P(A')P(B)$$
  $$P(\text{Contradiction}) = 0.80(0.40) + 0.20(0.60) = 0.32 + 0.12 = 0.44 \implies \mathbf{44\%}$$

---

#### **Question 3 (Cable Internet & TV)**
$60\%$ get Internet ($P(I) = 0.60$), $80\%$ get TV ($P(T) = 0.80$), $50\%$ get both ($P(I \cap T) = 0.50$).
* **Solutions:**
  * **i. Get at least one service:**
    $$P(I \cup T) = P(I) + P(T) - P(I \cap T) = 0.60 + 0.80 - 0.50 = \mathbf{0.90 \text{ (90\%)}}$$
  * **ii. Get exactly one service:**
    $$P(\text{Exactly one}) = P(I \cup T) - P(I \cap T) = 0.90 - 0.50 = \mathbf{0.40 \text{ (40\%)}}$$

---

#### **Question 4 (Electric Motors in Capsule)**
10 motors fabricated, 2 are defective. 2 motors randomly selected without replacement.
* **Solutions:**
  * **a) Both operate satisfactorily:**
    $$P(\text{Both sat}) = \frac{8}{10} \times \frac{7}{9} = \frac{56}{90} = \frac{28}{45} \approx \mathbf{0.6222}$$
  * **b) One operates satisfactorily and other does not:**
    $$P(\text{One sat, one def}) = P(\text{Sat, Def}) + P(\text{Def, Sat}) = \left(\frac{8}{10} \times \frac{2}{9}\right) + \left(\frac{2}{10} \times \frac{8}{9}\right) = \frac{32}{90} \approx \mathbf{0.3556}$$

---

#### **Question 5 (Coffee and Soda Consumption)**
$55\%$ regularly consume coffee ($P(C) = 0.55$), $45\%$ regularly consume carbonated soda ($P(S) = 0.45$), $70\%$ regularly consume at least one ($P(C \cup S) = 0.70$).
* **Solutions:**
  * **i) Probability they consume both:**
    $$P(C \cap S) = P(C) + P(S) - P(C \cup S) = 0.55 + 0.45 - 0.70 = \mathbf{0.30}$$
  * **ii) Probability they don't regularly consume at least one of these two:**
    * *Interpretation A:* "Consumes neither" (complement of "at least one"):
      $$P((C \cup S)') = 1 - P(C \cup S) = 1 - 0.70 = \mathbf{0.30}$$
    * *Interpretation B:* "Does not consume both" (complement of "both"):
      $$P((C \cap S)') = 1 - P(C \cap S) = 1 - 0.30 = \mathbf{0.70}$$

---

#### **Question 6 (Mathematics and Chemistry Students)**
Out of 80 students, 30 take Math ($P(M) = 30/80$), 20 take Chemistry ($P(C) = 20/80$), and 10 take both ($P(M \cap C) = 10/80$).
* **Solution:**
  $$P(M \cup C) = P(M) + P(C) - P(M \cap C) = \frac{30}{80} + \frac{20}{80} - \frac{10}{80} = \frac{40}{80} = \mathbf{0.50 \text{ (50\%)}}$$

---

#### **Question 7 (Abstract Probability Laws)**
Given $P(A \cup B) = 7/8$, $P(A \cap B) = 1/4$, $P(A') = 5/8$.
* **Solutions:**
  * **Find $P(A)$:**
    $$P(A) = 1 - P(A') = 1 - 5/8 = \mathbf{3/8}$$
  * **Find $P(B)$:**
    $$P(A \cup B) = P(A) + P(B) - P(A \cap B) \implies 7/8 = 3/8 + P(B) - 2/8 \implies P(B) = \mathbf{6/8 = 3/4}$$
  * **Find $P(A \cap B')$:**
    $$P(A \cap B') = P(A) - P(A \cap B) = 3/8 - 2/8 = \mathbf{1/8}$$

---

#### **Question 8 (Probability Relations)**
In a sample space, $P(A) = P(B)$, $P(A' \cap B') = P(A' \cap B) = 1/6$.
* **Solutions:**
  * Let $P(A) = P(B) = x$.
  * $P(A' \cap B') = P((A \cup B)') = 1 - P(A \cup B) = 1/6 \implies P(A \cup B) = 5/6$.
  * $P(A' \cap B) = P(B) - P(A \cap B) = 1/6 \implies P(A \cap B) = x - 1/6$.
  * Now use union formula:
    $$P(A \cup B) = P(A) + P(B) - P(A \cap B) \implies 5/6 = x + x - (x - 1/6) \implies x + 1/6 = 5/6 \implies x = \mathbf{2/3}$$
  * Joint intersection $P(A \cap B) = 2/3 - 1/6 = 1/2$.
  * **a) $P(A) = \mathbf{2/3}$**
  * **b) $P(A' \cup B') = 1 - P(A \cap B) = 1 - 1/2 = \mathbf{1/2}$**
  * **c) $P(\text{Exactly one}) = P(A \cup B) - P(A \cap B) = 5/6 - 3/6 = 2/6 = \mathbf{1/3}$**

---

#### **Question 9 (Homeowner & Auto Deductibles)**
Proportions Table:
| Auto \ Home | None (N) | Low (L) | Medium (M) | High (H) |
| :---: | :---: | :---: | :---: | :---: |
| **L** | 0.04 | 0.06 | 0.05 | 0.03 |
| **M** | 0.07 | 0.10 | 0.20 | 0.10 |
| **H** | 0.02 | 0.03 | 0.15 | 0.10 |
*(Note: To make a standard textbook table sum to 1.0, Auto-H / Homeowner-H is typically 0.15. We will provide solutions based on the slide's printed values where Auto-H / Homeowner-H is 0.10, making the table sum to 0.95)*

* **Solutions:**
  * **a) P(Medium Auto and High Homeowner):**
    $$P(M \cap H) = \mathbf{0.10}$$
  * **b) P(Low Auto) & P(Low Homeowner):**
    $$P(\text{Low Auto}) = 0.04 + 0.06 + 0.05 + 0.03 = \mathbf{0.18}$$
    $$P(\text{Low Home}) = 0.06 + 0.10 + 0.03 = \mathbf{0.19}$$
  * **c) P(Same category for both):**
    $$P(\text{Same}) = P(L \cap L) + P(M \cap M) + P(H \cap H) = 0.06 + 0.20 + 0.10 = \mathbf{0.36}$$
  * **d) P(Different categories):**
    * Assuming table is the entire sample space:
      $$P(\text{Diff}) = \text{Total} - P(\text{Same}) = 0.95 - 0.36 = \mathbf{0.59}$$
      *(Or $1 - 0.36 = 0.64$ if treating remaining 0.05 as having different categories/empty)*
  * **e) P(At least one low-deductible level):**
    $$P(\text{At least one Low}) = P(\text{Low Auto}) + P(\text{Low Home}) - P(L \cap L) = 0.18 + 0.19 - 0.06 = \mathbf{0.31}$$
  * **f) P(Neither deductible level is low):**
    $$P(\text{Neither Low}) = \text{Total} - P(\text{At least one Low}) = 0.95 - 0.31 = \mathbf{0.64} \quad (\text{or } 1 - 0.31 = 0.69)$$

---

### Lecture 3: Conditional Probability (S1-26_ISM_CS-3_PPT.pdf)

#### **Example 1 (Government Secretaries)**
Secretaries $S_1, S_2, S_3, S_4$ handle $20\%, 60\%, 15\%, 5\%$ of files. Misfiles occur with probabilities $0.05, 0.10, 0.10, 0.05$ respectively. What is the probability that a misfiled report was handled by $S_1$?
* **Solution:**
  * $P(S_1)=0.20, P(S_2)=0.60, P(S_3)=0.15, P(S_4)=0.05$
  * $P(M \mid S_1)=0.05, P(M \mid S_2)=0.10, P(M \mid S_3)=0.10, P(M \mid S_4)=0.05$
  * **Total misfile probability $P(M)$:**
    $$P(M) = 0.20(0.05) + 0.60(0.10) + 0.15(0.10) + 0.05(0.05) = 0.0100 + 0.0600 + 0.0150 + 0.0025 = 0.0875$$
  * **Posterior $P(S_1 \mid M)$:**
    $$P(S_1 \mid M) = \frac{P(M \mid S_1)P(S_1)}{P(M)} = \frac{0.0100}{0.0875} = \frac{4}{35} \approx \mathbf{0.1143 \text{ (11.43\%)}}$$

---

#### **Example 2 (Class Attendance)**
Class: $70\%$ boys ($P(B) = 0.70$), $30\%$ girls ($P(G) = 0.30$). $5\%$ of boys and $3\%$ of girls are irregular.
* **Solutions:**
  * **P(Selected student is irregular):**
    $$P(I) = P(I \mid B)P(B) + P(I \mid G)P(G) = 0.05(0.70) + 0.03(0.30) = 0.035 + 0.009 = \mathbf{0.044 \text{ (4.4\%)}}$$
  * **P(Irregular student is a girl):**
    $$P(G \mid I) = \frac{P(I \mid G) P(G)}{P(I)} = \frac{0.009}{0.044} = \frac{9}{44} \approx \mathbf{0.2045 \text{ (20.45\%)}}$$
    *(Note: Slide states $P(\text{Irregular and Girl}) = \mathbf{0.009}$)*

---

#### **Example 3 (Factory Machines)**
Machines A, B, C produce $60\%, 30\%, 10\%$ of items. Defect rates are $2\%, 3\%, 4\%$.
* **Solutions:**
  * **(i) P(Selected item was not defective and produced by C):**
    $$P(C \cap D') = P(C) P(D' \mid C) = 0.10 \times (1 - 0.04) = 0.10 \times 0.96 = \mathbf{0.096 \text{ (9.6\%)}}$$
  * **(ii) P(Item is produced by C or B given it is defective):**
    * Total defective rate $P(D) = 0.60(0.02) + 0.30(0.03) + 0.10(0.04) = 0.012 + 0.009 + 0.004 = 0.025$.
    * $$P(C \cup B \mid D) = P(C \mid D) + P(B \mid D) = \frac{0.004}{0.025} + \frac{0.009}{0.025} = \frac{0.013}{0.025} = \mathbf{0.52 \text{ (52\%)}}$$
    *(Note: The slide answer sheets list a simplified value of **0.4**, but 0.52 is the mathematically precise answer)*

---

#### **Example 4 (Incomplete Card Deck)**
Deck is missing the Ace of Diamonds ($n = 51$).
* **Solutions:**
  * **1. P(clubs):** Clubs has all 13 cards remaining.
    $$P(\text{Clubs}) = \mathbf{13/51 \approx 0.2549}$$
  * **2. P(queen):** There are still 4 queens in the deck.
    $$P(\text{Queen}) = \mathbf{4/51 \approx 0.0784}$$
  * **3. Are Clubs and Queen independent?**
    $$P(\text{Clubs} \cap \text{Queen}) = P(\text{Queen of Clubs}) = 1/51 \approx 0.0196$$
    $$P(\text{Clubs}) \times P(\text{Queen}) = \frac{13}{51} \times \frac{4}{51} = \frac{52}{2601} \approx 0.0200 \neq 0.0196 \implies \mathbf{\text{No}}$$

---

#### **Example 5 (Company Tool Supplies)**
Company A produces $30\%$ ($P(A) = 0.30$). Companies B and C produce remaining $70\%$ ($P(B) + P(C) = 0.70$). Non-defective rates are $98\% (A), 95\% (B), 97\% (C)$. Target non-defective rate $= 96\%$.
* **Solution:**
  Let $P(B) = x$, so $P(C) = 0.70 - x$.
  $$0.96 = 0.98(0.30) + 0.95(x) + 0.97(0.70 - x)$$
  $$0.96 = 0.294 + 0.95x + 0.679 - 0.97x \implies 0.02x = 0.013 \implies x = 0.65$$
  * **Company B Market Share:** **$65\%$**
  * **Company C Market Share:** **$5\%$**
  *(Note: Relative to the remaining 70% of production, Company B produces $\frac{0.65}{0.70} \approx \mathbf{92.86\%}$, and Company C produces $\frac{0.05}{0.70} \approx \mathbf{7.14\%}$. The slide answer lists B's relative share as **16.4%** under alternative non-defective rate assumptions)*

---

#### **Example 6 (Consumer Behavior)**
Contingency table of 500 respondents:
* **Males ($240$ total):** Enjoys $= 136$, Does Not Enjoy $= 104$.
* **Females ($260$ total):** Enjoys $= 224$, Does Not Enjoy $= 36$.
* **Total ($500$ total):** Enjoys $= 360$, Does Not Enjoy $= 140$.
* **Solutions:**
  * **a. Given female, probability she does not enjoy clothing shopping:**
    $$P(\text{No} \mid \text{Female}) = \frac{36}{260} \approx \mathbf{0.1384}$$
  * **b. Probability that a respondent who enjoys shopping is a male:**
    * In standard statistics:
      $$P(\text{Male} \mid \text{Enjoys}) = \frac{136}{360} \approx \mathbf{0.3778}$$
    * Under the slide's printed answer key interpretation of $P(\text{Enjoys} \mid \text{Male})$:
      $$P(\text{Enjoys} \mid \text{Male}) = \frac{136}{240} \approx \mathbf{0.56}$$

---

#### **Example 7 (Card Game without Replacement)**
* **Solutions:**
  * **a. P(Both cards are queens - without replacement):**
    $$P(\text{Queen}_1 \cap \text{Queen}_2) = \frac{4}{52} \times \frac{3}{51} = \frac{1}{221} \approx \mathbf{0.0045}$$
  * **b. P(1st card is a 10, 2nd is a 5 or 6):**
    $$P(10_1 \cap (5 \cup 6)_2) = \frac{4}{52} \times \frac{8}{51} = \frac{32}{2652} \approx \mathbf{0.012}$$
  * **c. Same as (a), but with replacement:**
    $$P(\text{Queen}_1 \cap \text{Queen}_2) = \frac{4}{52} \times \frac{4}{52} = \frac{1}{169} \approx \mathbf{0.0059}$$

---

### Lecture 4: Bayesian Learning (S1-26_ISM_CS-4_PPT.pdf)

#### **Practice Problems 1 (Board Candidates)**
Two sets of candidates compete. $P(W_1) = 0.60, P(W_2) = 0.40$. Probability of introducing a new product is $0.80$ if set 1 wins, and $0.30$ if set 2 wins.
* **Solution:**
  $$P(\text{Product}) = P(\text{Product} \mid W_1)P(W_1) + P(\text{Product} \mid W_2)P(W_2) = 0.80(0.60) + 0.30(0.40) = 0.48 + 0.12 = \mathbf{0.60}$$

---

#### **Practice Problems 2 (Ad Remembrances)**
Audience: $80\%$ remember audio ($P(A) = 0.80$), $60\%$ remember video ($P(V) = 0.60$). Independent events.
* **Solutions:**
  * **(i) Both:** $P(A \cap V) = P(A) \times P(V) = 0.80 \times 0.60 = \mathbf{0.48}$
  * **(ii) At least one:** $P(A \cup V) = 0.80 + 0.60 - 0.48 = \mathbf{0.92}$
  * **(iii) Video given audio:** $P(V \mid A) = P(V) = \mathbf{0.60}$ (due to independence)
  * **(iv) Neither:** $P(A' \cap V') = 1 - P(A \cup V) = 1 - 0.92 = \mathbf{0.08}$

---

#### **Practice Problems 3 (Assembly Laser Scanner)**
Scanner detects $90\%$ of defective parts ($P(F \mid D) = 0.90$). Incorrectly flags $2\%$ of non-defective parts ($P(F \mid D') = 0.02$). $5\%$ of parts are actually defective ($P(D) = 0.05$). Find $P(D' \mid F)$.
* **Solution:**
  * **Total flagged $P(F)$:**
    $$P(F) = P(F \mid D)P(D) + P(F \mid D')P(D') = 0.90(0.05) + 0.02(0.95) = 0.045 + 0.019 = 0.064$$
  * **Posterior $P(D' \mid F)$:**
    $$P(D' \mid F) = \frac{P(F \mid D') P(D')}{P(F)} = \frac{0.019}{0.064} = \frac{19}{64} = \mathbf{0.2969 \text{ (29.69\%)}}$$

---

#### **Practice Problems 4 (Vice Chancellor Selection)**
VC Probabilities: Academician $A = 0.50$, Business Man $B = 0.30$, Politician $P = 0.20$.
Research promotion rates: $P(R \mid A) = 0.80, P(R \mid B) = 0.60, P(R \mid P) = 0.40$.
If research is promoted, find $P(A \mid R)$.
* **Solution:**
  * **Total research promotion $P(R)$:**
    $$P(R) = 0.80(0.50) + 0.60(0.30) + 0.40(0.20) = 0.40 + 0.18 + 0.08 = 0.66$$
  * **Posterior $P(A \mid R)$:**
    $$P(A \mid R) = \frac{P(R \mid A)P(A)}{P(R)} = \frac{0.40}{0.66} = \frac{20}{33} \approx \mathbf{0.6061 \text{ (60.61\%)}}$$

---

#### **Practice Problems 5 (Production Lines)**
Lines $L_1, L_2, L_3$ output $40\%, 35\%, 25\%$. Defective rates are $1\%, 2\%, 3\%$.
* **Solutions:**
  * Total defective rate $P(D) = 0.40(0.01) + 0.35(0.02) + 0.25(0.03) = 0.0040 + 0.0070 + 0.0075 = 0.0185$.
  * **(i) P(Produced by L1 given defective):**
    $$P(L_1 \mid D) = \frac{0.0040}{0.0185} \approx \mathbf{0.2162 \text{ (21.62\%)}}$$
  * **(ii) P(Produced by L2 given defective):**
    $$P(L_2 \mid D) = \frac{0.0070}{0.0185} \approx \mathbf{0.3784 \text{ (37.84\%)}}$$

---

#### **Practice Problems 6 (Disappearing Aircraft)**
$70\%$ of crashed light aircraft are discovered ($P(\text{Dis}) = 0.70$). $60\%$ of discovered have locator beacons ($P(\text{Loc} \mid \text{Dis}) = 0.60$). $90\%$ of undiscovered do not have locator beacons ($P(\text{Loc}' \mid \text{Dis}') = 0.90 \implies P(\text{Loc} \mid \text{Dis}') = 0.10$).
* **Solutions:**
  * **(a) If it has locator, find probability it is not discovered:**
    $$P(\text{Loc}) = P(\text{Loc} \mid \text{Dis})P(\text{Dis}) + P(\text{Loc} \mid \text{Dis}')P(\text{Dis}') = 0.60(0.70) + 0.10(0.30) = 0.42 + 0.03 = 0.45$$
    $$P(\text{Dis}' \mid \text{Loc}) = \frac{P(\text{Loc} \mid \text{Dis}')P(\text{Dis}')}{P(\text{Loc})} = \frac{0.03}{0.45} = \frac{1}{15} \approx \mathbf{0.0667 \text{ (6.67\%)}}$$
  * **(b) If it doesn't have locator, find probability it is discovered:**
    $$P(\text{Loc}') = 1 - P(\text{Loc}) = 0.55$$
    $$P(\text{Dis} \mid \text{Loc}') = \frac{P(\text{Loc}' \mid \text{Dis})P(\text{Dis})}{P(\text{Loc}')} = \frac{(1 - 0.60)(0.70)}{0.55} = \frac{0.28}{0.55} \approx \mathbf{0.5091 \text{ (50.91\%)}}$$

---

#### **Practice Problems 7 (Contract Bidding)**
$P(\text{W bids}) = 3/4$. If W does not bid, $P(\text{V gets job} \mid \text{W}') = 3/4$. If W bids, $P(\text{V gets job} \mid \text{W}) = 1/3$.
* **Solutions:**
  * **(a) P(V gets job):**
    $$P(V) = P(V \mid W)P(W) + P(V \mid W')P(W') = \left(\frac{1}{3} \times \frac{3}{4}\right) + \left(\frac{3}{4} \times \frac{1}{4}\right) = \frac{1}{4} + \frac{3}{16} = \frac{7}{16} = \mathbf{0.4375}$$
  * **(b) If V gets job, probability W did not bid:**
    $$P(W' \mid V) = \frac{P(V \mid W')P(W')}{P(V)} = \frac{3/16}{7/16} = \frac{3}{7} \approx \mathbf{0.4286}$$

---

### Lecture 5: Joint Probability Distributions (S1-26_ISM_CS-5_PPT.pdf)

#### **Practice Problems-Set 2**
Joint PMF Table of $X$ and $Y$:
| $Y$ \ $X$ | **2** | **4** |
| :---: | :---: | :---: |
| **1** | 0.10 | 0.15 |
| **3** | 0.20 | 0.30 |
| **5** | 0.10 | 0.15 |

* **Solutions:**
  * **(a) Find Marginal Probability Distributions:**
    * **Marginal PMF of $X$ (Sum columns):**
      $$P(X = 2) = 0.10 + 0.20 + 0.10 = \mathbf{0.40}$$
      $$P(X = 4) = 0.15 + 0.30 + 0.15 = \mathbf{0.60}$$
    * **Marginal PMF of $Y$ (Sum rows):**
      $$P(Y = 1) = 0.10 + 0.15 = \mathbf{0.25}$$
      $$P(Y = 3) = 0.20 + 0.30 = \mathbf{0.50}$$
      $$P(Y = 5) = 0.10 + 0.15 = \mathbf{0.25}$$
  * **(b) Find specific values:**
    * $P(X = 2, Y = 3) = \mathbf{0.20}$
    * $P(X = 4, Y = 1) = \mathbf{0.15}$
    * $P(Y = 5) = \mathbf{0.25}$
    * $P(X = 2) = \mathbf{0.40}$
  * **(c) Are $X$ and $Y$ independent? Why?**
    * **Yes.** Two random variables are independent if for all $x, y$:
      $$P(X=x, Y=y) = P(X=x) P(Y=y)$$
    * Testing cell $(2, 1)$: $P(X=2, Y=1) = 0.10$. Product $= P(X=2) P(Y=1) = 0.40 \times 0.25 = 0.10$.
    * Testing cell $(2, 3)$: $P(X=2, Y=3) = 0.20$. Product $= 0.40 \times 0.50 = 0.20$.
    * Testing cell $(4, 5)$: $P(X=4, Y=5) = 0.15$. Product $= 0.60 \times 0.25 = 0.15$.
    * Since this product relation holds for every single joint state, the variables are **independent**.

---

### Lecture 6: Continuous Distributions (ISM_CS-6_PPT.pdf)

#### **Practice Problem 4 (Standard Normal Variate Z)**
* **Solutions:**
  * **(i) $P(Z > 1.09)$:**
    $$P(Z > 1.09) = 1 - \Phi(1.09) = 1 - 0.8621 = \mathbf{0.1379}$$
  * **(ii) $P(Z < -1.65)$:**
    $$P(Z < -1.65) = \Phi(-1.65) = \mathbf{0.0495}$$
  * **(iii) $P(-1.00 < Z < 1.96)$:**
    $$P(-1.00 < Z < 1.96) = \Phi(1.96) - \Phi(-1.00) = 0.9750 - 0.1587 = \mathbf{0.8163}$$
  * **(iv) $P(1.25 < Z < 2.75)$:**
    $$P(1.25 < Z < 2.75) = \Phi(2.75) - \Phi(1.25) = 0.9970 - 0.8944 = \mathbf{0.1026}$$

---

#### **Practice Problem 5 (Normal Instrument Lifetime)**
Continuous life of instrument follows $X \sim N(12, 2^2)$.
* **Solutions:**
  * **a) P(lasts less than 7 months):**
    $$Z = \frac{7 - 12}{2} = -2.50 \implies P(X < 7) = P(Z < -2.50) = \Phi(-2.50) = \mathbf{0.0062}$$
  * **b) P(lasts between 7 and 12 months):**
    $$Z_{\text{lower}} = -2.50, \quad Z_{\text{upper}} = \frac{12 - 12}{2} = 0$$
    $$P(7 < X < 12) = P(-2.50 < Z < 0) = \Phi(0) - \Phi(-2.50) = 0.5000 - 0.0062 = \mathbf{0.4938}$$

---

#### **Practice Problem 6 (Computer Crashes - Poisson)**
A computer crashes once every 4 months on average. Let $X$ be the number of crashes in a 4-month period.
$$X \sim \text{Poisson}(\lambda = 1)$$
$$P(X = x) = \frac{e^{-1} 1^x}{x!} = \frac{e^{-1}}{x!}$$

* **Solutions ($e^{-1} \approx 0.3679$):**
  * **a) P(no crash in 4 months):** $P(X = 0) = e^{-1} = \mathbf{0.3679}$
  * **b) P(1 crash in 4 months):** $P(X = 1) = \frac{e^{-1}}{1} = \mathbf{0.3679}$
  * **c) P(2 crashes in 4 months):** $P(X = 2) = \frac{e^{-1}}{2} \approx \mathbf{0.1839}$
  * **d) P(3 crashes in 4 months):** $P(X = 3) = \frac{e^{-1}}{6} \approx \mathbf{0.0613}$

---

#### **Practice Problems 7, 8, 9 (Discrete Distributions)**
* **7) Binomial distribution with mean np = 20 and n = 30. Find Variance & SD:**
  * $p = \frac{20}{30} = \frac{2}{3} \implies q = 1 - p = \frac{1}{3}$
  * **Variance:** $\text{Var} = n p q = 30 \times \frac{2}{3} \times \frac{1}{3} = \frac{20}{3} \approx \mathbf{6.67}$
  * **Standard Deviation:** $\text{SD} = \sqrt{6.67} \approx \mathbf{2.58}$

* **8) Boy hitting target ($p = 0.2$). How many times must he fire so that $P(X \ge 1) > 2/3$?**
  * $$P(X \ge 1) = 1 - P(X = 0) = 1 - (0.8)^n > \frac{2}{3} \implies (0.8)^n < \frac{1}{3} \approx 0.3333$$
  * For $n=4$: $0.8^4 = 0.4096 > 0.3333$
  * For $n=5$: $0.8^5 = 0.3277 < 0.3333$
  * **Solution:** He must fire at least **$5$ times**.

* **9) Candy box: $15\%$ pink ($p = 0.15$). Find $P(\text{exactly 4 pink out of 10})$:**
  * $$P(X = 4) = \binom{10}{4} (0.15)^4 (0.85)^6 = 210 \times 0.000506 \times 0.3771 \approx \mathbf{0.0401 \text{ (or 4.01\%)}}$$

---

#### **Example 10 & 11 (Discrete Application)**
* **10) Find Binomial Distribution if Mean is 6 and Variance is 4:**
  * $np = 6$, $npq = 4 \implies q = \frac{npq}{np} = \frac{4}{6} = \frac{2}{3} \implies p = \frac{1}{3}$
  * $n = \frac{6}{p} = \frac{6}{1/3} = 18 \implies \mathbf{X \sim \text{Binomial}(18, 1/3)}$

* **11) Mean defectives in sample of 20 is 2. Out of 1000 samples, how many contain at least 3 defectives?**
  * $n = 20, p = 2/20 = 0.10 \implies X \sim \text{Binomial}(20, 0.10)$.
  * $$P(X \ge 3) = 1 - [P(X=0) + P(X=1) + P(X=2)]$$
    * $P(X=0) = (0.90)^{20} \approx 0.1216$
    * $P(X=1) = \binom{20}{1} (0.10)^1 (0.90)^{19} \approx 0.2702$
    * $P(X=2) = \binom{20}{2} (0.10)^2 (0.90)^{18} \approx 0.2852$
  * Sum of at most 2 defectives $= 0.1216 + 0.2702 + 0.2852 = 0.6770$.
  * $P(X \ge 3) = 1 - 0.6770 = 0.3230$ (or $32.3\%$).
  * Out of 1000 samples: $1000 \times 0.3230 = \mathbf{323 \text{ samples}}$.

---

#### **Example 12 (Poisson Cafeteria Arrivals)**
Arrivals occur at average rate of $\alpha = 0.3$ per minute.
* **Solutions:**
  * **(a) Exactly 2 arrivals in a 10-minute span ($\lambda = 0.3 \times 10 = 3$):**
    $$P(X = 2) = \frac{e^{-3} 3^2}{2!} = \frac{9 e^{-3}}{2} \approx \mathbf{0.2240 \text{ (22.40\%)}}$$
  * **(b) 2 or more arrivals in 10 minutes:**
    $$P(X \ge 2) = 1 - [P(X=0) + P(X=1)] = 1 - [e^{-3} + 3e^{-3}] = 1 - 4e^{-3} \approx 1 - 0.1991 = \mathbf{0.8009 \text{ (80.09\%)}}$$
  * **(c) Exactly one customer in first 5 mins and one in next 5 mins ($\lambda = 1.5$ per 5-min interval):**
    * By independence of disjoint Poisson intervals:
      $$P(X_{\text{first}} = 1) \times P(X_{\text{second}} = 1) = \left( \frac{e^{-1.5} 1.5^1}{1!} \right)^2 = (1.5 e^{-1.5})^2 \approx 0.3347^2 \approx \mathbf{0.1120 \text{ (11.20\%)}}$$

---

#### **Example 13 & 14 (Normal Distribution Applications)**
* **13) Valve flow coefficient $X \sim N(496, 25^2)$:**
  * **a) P(coefficient is at least 450 Cv):**
    $$Z = \frac{450 - 496}{25} = -1.84 \implies P(X > 450) = P(Z > -1.84) = \Phi(1.84) = \mathbf{0.9671}$$
  * **b) P(coefficient is between 445.5 and 522 Cv):**
    $$Z_{\text{lower}} = \frac{445.5 - 496}{25} = -2.02, \quad Z_{\text{upper}} = \frac{522 - 496}{25} = 1.04$$
    $$P(445.5 < X < 522) = \Phi(1.04) - \Phi(-2.02) = 0.8508 - 0.0217 = \mathbf{0.8291}$$

* **14) silver iodide seed cloud growth ($n = 40, p = 0.62$). Find probability of at most 20 clouds showing growth by normal approximation (with continuity correction):**
  * $\mu = np = 24.8$, $\sigma = \sqrt{40 \times 0.62 \times 0.38} = \sqrt{9.424} \approx 3.0699$.
  * We want $P(X \le 20.5)$ (continuity correction):
    $$Z = \frac{20.5 - 24.8}{3.0699} = -1.40 \implies P(Z \le -1.40) = \Phi(-1.40) = \mathbf{0.0808}$$
  *(Note: The slide has a typo printing this probability as 0.808 instead of 0.0808)*

---

### Lecture 7: Estimation & Confidence Intervals (ISM_CS-7_PPT.pdf)

#### **Example 1 (Cellular Phone Bills)**
$n = 85$ bills, sample mean $\bar{X} = 510$ minutes, population standard deviation $\sigma = 46$ minutes. Determine a $95\%$ confidence interval.
* **Solution:**
  Using the Z-distribution ($Z_{\alpha/2} = 1.96$):
  $$CI = \bar{X} \pm Z_{\alpha/2} \left( \frac{\sigma}{\sqrt{n}} \right) = 510 \pm 1.96 \left( \frac{46}{\sqrt{85}} \right) = 510 \pm 1.96(4.989) = 510 \pm 9.78$$
  $$CI = \mathbf{(500.22, 519.78) \text{ minutes}}$$

---

#### **Example 2 (Company Engineers Age)**
$N = 800$ total engineers (finite population), sample of $n = 50$ reveals average age $\bar{X} = 34.3$ years, population standard deviation $\sigma = 8$ years. Construct a $98\%$ confidence interval.
* **Solution:**
  * Since sample fraction $\frac{n}{N} = \frac{50}{800} = 0.0625 \ge 0.05$, we apply the Finite Population Correction (FPC) factor:
    $$\text{FPC} = \sqrt{\frac{N-n}{N-1}} = \sqrt{\frac{800-50}{800-1}} = \sqrt{\frac{750}{799}} \approx 0.9688$$
    $$\sigma_{\bar{X}} = \frac{\sigma}{\sqrt{n}} \times \text{FPC} = \frac{8}{\sqrt{50}} \times 0.9688 \approx 1.1314 \times 0.9688 \approx 1.096 \text{ years}$$
  * For $98\%$ confidence, $Z_{\alpha/2} = 2.33$:
    $$CI = 34.3 \pm 2.33(1.096) = 34.3 \pm 2.55 = \mathbf{(31.75, 36.85) \text{ years}}$$
  * *(Without FPC: $CI = 34.3 \pm 2.33(1.131) = 34.3 \pm 2.64 = \mathbf{(31.66, 36.94) \text{ years}}$)*

---

#### **Example 3 (Trading with India)**
$n = 44$, sample mean $\bar{X} = 10.455$ years, population standard deviation $\sigma = 7.7$ years. Construct a $90\%$ confidence interval.
* **Solution:**
  For $90\%$ confidence, $Z_{\alpha/2} = 1.645$:
  $$CI = 10.455 \pm 1.645 \left( \frac{7.7}{\sqrt{44}} \right) = 10.455 \pm 1.645(1.161) = 10.455 \pm 1.91 \approx \mathbf{(8.55, 12.37) \text{ years}}$$

---

#### **Practice Problem-1 (Atlanta Restaurant)**
$n = 49$ customers, population standard deviation $\sigma = \$5$.
* **Solutions:**
  * **a. At 95% confidence, what is the margin of error?**
    $$ME = Z_{\alpha/2} \left( \frac{\sigma}{\sqrt{n}} \right) = 1.96 \left( \frac{5}{\sqrt{49}} \right) = 1.96 \left( \frac{5}{7} \right) = \mathbf{\$1.40}$$
  * **b. If sample mean is \$24.80, find confidence interval:**
    $$CI = \bar{X} \pm ME = 24.80 \pm 1.40 = \mathbf{(\$23.40, \$26.20)}$$

---

#### **Practice Problem-2 (Light Bulb Lifetime)**
$\sigma = 100$ hours, $n = 64$ light bulbs, sample mean $\bar{X} = 350$ hours.
* **Solutions:**
  * **a. Construct a 95% confidence interval:**
    $$ME = 1.96 \left( \frac{100}{\sqrt{64}} \right) = 1.96 \left( \frac{100}{8} \right) = 1.96(12.5) = \$24.50 \text{ hours}$$
    $$CI = 350 \pm 24.50 = \mathbf{(325.50, 374.50) \text{ hours}}$$
  * **b. Do you think that the manufacturer has the right to state that the light bulbs have a mean life of 400 hours? Explain.**
    * **No.** Since the claimed mean life of $400$ hours lies completely outside (and well above) the 95% confidence interval of $(325.5, 374.5)$ hours, we can be $95\%$ confident that the true population mean is less than $374.5$ hours. Therefore, the manufacturer's claim of $400$ hours is statistically unsupported by the sample.

---

#### **Practice Problem-3 (Election Night Pollster)**
$n = 100$. Forecast winner if sample proportion $p \ge 0.55$.
* **Solutions:**
  * **a. True percentage $\pi = 50.1\%$ ($0.501$):**
    $$\sigma_p = \sqrt{\frac{0.501(0.499)}{100}} \approx 0.05 \implies Z = \frac{0.55 - 0.501}{0.05} = 0.98$$
    $$P(p \ge 0.55) = P(Z \ge 0.98) = 1 - 0.8365 = \mathbf{0.1635 \text{ (16.35\%)}}$$
  * **b. True percentage $\pi = 60.0\%$ ($0.60$):**
    $$\sigma_p = \sqrt{\frac{0.60(0.40)}{100}} \approx 0.049 \implies Z = \frac{0.55 - 0.60}{0.049} = -1.02$$
    $$P(p \ge 0.55) = P(Z \ge -1.02) = 1 - 0.1539 = \mathbf{0.8461 \text{ (84.61\%)}}$$
  * **c. True percentage $\pi = 49.0\%$ ($0.49$):**
    $$\sigma_p = \sqrt{\frac{0.49(0.51)}{100}} \approx 0.05 \implies Z = \frac{0.55 - 0.49}{0.05} = 1.20$$
    $$P(p \ge 0.55) = P(Z \ge 1.20) = 1 - 0.8849 = \mathbf{0.1151 \text{ (11.51\%)}}$$
  * **d. If $n$ is increased to 400:**
    * **For $\pi = 50.1\%$:** $\sigma_p \approx 0.025 \implies Z = \frac{0.55 - 0.501}{0.025} = 1.96 \implies P(Z \ge 1.96) \approx \mathbf{0.0250 \text{ (2.50\%)}}$
    * **For $\pi = 60.0\%$:** $\sigma_p = 0.0245 \implies Z = \frac{0.55 - 0.60}{0.0245} = -2.04 \implies P(Z \ge -2.04) = 1 - 0.0207 = \mathbf{0.9793 \text{ (97.93\%)}}$
    * **For $\pi = 49.0\%$:** $\sigma_p \approx 0.025 \implies Z = \frac{0.55 - 0.49}{0.025} = 2.40 \implies P(Z \ge 2.40) = 1 - 0.9918 = \mathbf{0.0082 \text{ (0.82\%)}}$

---

#### **Practice Problem-4 (Bounced Cheque Fees)**
Sample of 23 banks ($n = 23$):
$$26, 28, 20, 20, 21, 22, 25, 25, 18, 25, 15, 20, 18, 20, 25, 25, 22, 30, 30, 30, 15, 20, 29$$

* **Solutions:**
  * Since population standard deviation $\sigma$ is unknown, we must use the **sample standard deviation $s$ and t-distribution (with $df = n - 1 = 22$)**.
  * **Sample Mean ($\bar{X}$):** $\bar{X} = \frac{529}{23} = \mathbf{23.00 \text{ dollars}}$
  * **Sample Standard Deviation ($s$):** $s \approx \mathbf{4.602 \text{ dollars}}$
  * **Standard Error (SE):** $\text{SE} = \frac{s}{\sqrt{n}} = \frac{4.602}{\sqrt{23}} \approx 0.9597$
  * **t-value (for df=22, 95% confidence):** $t_{0.025, 22} = \mathbf{2.0739}$
  * **Margin of Error:** $ME = 2.0739 \times 0.9597 = \mathbf{1.99 \text{ dollars}}$
  * **a. 95% Confidence Interval:**
    $$CI = \bar{X} \pm ME = 23.00 \pm 1.99 = \mathbf{(21.01, 24.99) \text{ dollars}}$$
  * **b. Interpret the interval:**
    * We can be $95\%$ confident that the true population mean bounced cheque fee charged by all banks for direct-deposit customers who maintain a \$100 balance is between **\$21.01** and **\$24.99**.

---

#### **Practice Problem-5 (Medical Expenses Sample Size)**
Target: 95% confidence ($Z_{\alpha/2} = 1.96$), standard deviation estimate $\sigma \approx \$400$.
* **Solutions:**
  * **a. Correct within \$50 ($e = 50$):**
    $$n = \left( \frac{Z_{\alpha/2} \times \sigma}{e} \right)^2 = \left( \frac{1.96 \times 400}{50} \right)^2 = (15.68)^2 = 245.86 \implies \mathbf{246 \text{ families}}$$
    *(Always round sample sizes up to the next integer)*
  * **b. Correct within \$25 ($e = 25$):**
    $$n = \left( \frac{1.96 \times 400}{25} \right)^2 = (31.36)^2 = 983.45 \implies \mathbf{984 \text{ families}}$$
