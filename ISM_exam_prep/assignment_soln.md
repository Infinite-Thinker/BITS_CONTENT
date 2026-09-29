# BITS Pilani WILP -- Introduction to Statistical Methods (S1-26 AIMLCZC418)
## Assignment 1 Solution Manual (Handwriting-Ready Guide)

**Section:** Prof. G Krishna Kumari  
**Course Code:** S1-26 AIMLCZC418  
**Max Marks:** 10  
**Deadline:** September 30, 2026  

---

### 📝 HANDWRITING & PRESENTATION TIPS FOR MAXIMUM MARKS:
1. **Header on Every Page:** Write your **Name**, **BITS Student ID**, and **Section (Prof. G Krishna Kumari)** at the very top of each page.
2. **Step-by-Step Working:** Show all formulas and intermediate arithmetic steps. Do not skip any steps; professors award partial credit based on steps.
3. **Box Final Answers:** Draw a neat box around each final answer (e.g., using a blue or black pen) so that it is easily identifiable.
4. **Venn Diagrams & Tables:** Draw the tables and Venn diagrams neatly with a ruler. 

---

## Page 1: Student Information Block (Copy at top of each page)
```text
Name: [Your Name]             BITS Student ID: [Your BITS ID]
Course: Introduction to Statistical Methods (S1-26 AIMLCZC418)
Section: Prof. G Krishna Kumari                 Assignment 1
```

---

## 🟥 QUESTION 1: SOFTWARE MODULE COMPLETION TIMES (2.5 Marks)

**Problem Statement:**  
The completion times (in days) for 12 recent software modules are:  
`14, 16, 15, 18, 17, 16, 15, 14, 17, 16, 15, 72`

---

### **a) Compute the Mean, Median, Sample Variance, Range, and $Q_1, Q_3$**

#### **Step 1: Sort the Dataset in Ascending Order**
Let the sorted dataset be denoted as $X_{(1)}, X_{(2)}, \dots, X_{(n)}$ where $n = 12$:
$$\mathbf{14, 14, 15, 15, 15, 16, 16, 16, 17, 17, 18, 72}$$

---

#### **Step 2: Compute the Mean ($\bar{x}$)**
* **Formula:**
  $$\bar{x} = \frac{\sum_{i=1}^{n} x_i}{n}$$
* **Calculation:**
  $$\sum_{i=1}^{12} x_i = 14 + 14 + 15 + 15 + 15 + 16 + 16 + 16 + 17 + 17 + 18 + 72 = 245$$
  $$\bar{x} = \frac{245}{12} \approx 20.4167 \text{ days}$$

* **Final Answer:** 
  $$\boxed{\bar{x} \approx 20.42\text{ days}}$$

---

#### **Step 3: Compute the Median (M)**
* **Formula (for even $n = 12$):**
  $$\text{Median Position} = \frac{n + 1}{2} = \frac{12 + 1}{2} = 6.5^{\text{th}}\text{ position}$$
  $$\text{Median} = \frac{X_{(6)} + X_{(7)}}{2}$$
* **Calculation:**
  From our sorted dataset:
  * $6^{\text{th}}$ element ($X_{(6)}$) = $16$
  * $7^{\text{th}}$ element ($X_{(7)}$) = $16$
  $$\text{Median} = \frac{16 + 16}{2} = 16.00\text{ days}$$

* **Final Answer:** 
  $$\boxed{\text{Median} = 16\text{ days}}$$

---

#### **Step 4: Compute the Range**
* **Formula:**
  $$\text{Range} = \text{Maximum value} - \text{Minimum value}$$
* **Calculation:**
  $$\text{Maximum} = 72, \quad \text{Minimum} = 14$$
  $$\text{Range} = 72 - 14 = 58\text{ days}$$

* **Final Answer:** 
  $$\boxed{\text{Range} = 58\text{ days}}$$

---

#### **Step 5: Compute First Quartile ($Q_1$) and Third Quartile ($Q_3$)**
We can calculate these using the standard **positional interpolation method**:

* **For First Quartile ($Q_1$):**
  $$\text{Position of } Q_1 = \frac{n + 1}{4} = \frac{12 + 1}{4} = 3.25^{\text{th}}\text{ position}$$
  $$\text{Since the position is } 3.25\text{, we interpolate between } X_{(3)}\text{ and } X_{(4)}\text{:}$$
  $$Q_1 = X_{(3)} + 0.25 \times (X_{(4)} - X_{(3)})$$
  From the sorted data, $X_{(3)} = 15$ and $X_{(4)} = 15$:
  $$Q_1 = 15 + 0.25 \times (15 - 15) = 15\text{ days}$$

* **For Third Quartile ($Q_3$):**
  $$\text{Position of } Q_3 = \frac{3(n + 1)}{4} = \frac{3(12 + 1)}{4} = 9.75^{\text{th}}\text{ position}$$
  $$\text{Since the position is } 9.75\text{, we interpolate between } X_{(9)}\text{ and } X_{(10)}\text{:}$$
  $$Q_3 = X_{(9)} + 0.75 \times (X_{(10)} - X_{(9)})$$
  From the sorted data, $X_{(9)} = 17$ and $X_{(10)} = 17$:
  $$Q_3 = 17 + 0.75 \times (17 - 17) = 17\text{ days}$$

*(Note: If using the split-median/Tukey's method, the lower half of 6 values $[14, 14, 15, 15, 15, 16]$ has a median of $15$, and the upper half of 6 values $[16, 16, 17, 17, 18, 72]$ has a median of $17$. Both methods yield identical results!)*

* **Final Answers:**
  $$\boxed{Q_1 = 15\text{ days}}$$
  $$\boxed{Q_3 = 17\text{ days}}$$

---

#### **Step 6: Compute the Sample Variance ($s^2$)**
* **Formula:**
  $$s^2 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})^2}{n - 1} = \frac{\sum x_i^2 - \frac{(\sum x_i)^2}{n}}{n - 1}$$
  
Let's construct a step-by-step arithmetic table of squared deviations to show full work:

| Sorted $x_i$ | Deviation $(x_i - \bar{x})$ where $\bar{x} = 20.4167$ | Squared Deviation $(x_i - \bar{x})^2$ |
| :---: | :---: | :---: |
| 14 | $14 - 20.4167 = -6.4167$ | 41.1736 |
| 14 | $14 - 20.4167 = -6.4167$ | 41.1736 |
| 15 | $15 - 20.4167 = -5.4167$ | 29.3403 |
| 15 | $15 - 20.4167 = -5.4167$ | 29.3403 |
| 15 | $15 - 20.4167 = -5.4167$ | 29.3403 |
| 16 | $16 - 20.4167 = -4.4167$ | 19.5069 |
| 16 | $16 - 20.4167 = -4.4167$ | 19.5069 |
| 16 | $16 - 20.4167 = -4.4167$ | 19.5069 |
| 17 | $17 - 20.4167 = -3.4167$ | 11.6736 |
| 17 | $17 - 20.4167 = -3.4167$ | 11.6736 |
| 18 | $18 - 20.4167 = -2.4167$ | 5.8403 |
| 72 | $72 - 20.4167 = 51.5833$ | 2660.8403 |
| **Sum:** | $\mathbf{\sum x_i = 245}$ | $\mathbf{\sum(x_i - \bar{x})^2 = 2918.9167}$ |

* **Calculation:**
  $$s^2 = \frac{2918.9167}{12 - 1} = \frac{2918.9167}{11} \approx 265.3561\text{ days}^2$$
  *(For completeness, the sample standard deviation is $s = \sqrt{265.3561} \approx 16.29\text{ days}$)*

* **Final Answer:** 
  $$\boxed{s^2 \approx 265.36\text{ days}^2}$$

---

### **b) Determine the Skewness (Left/Right) Based on Central Tendencies**

To determine the skewness, we compare our calculated measures of central tendency:
* **Mean:** $\approx 20.42\text{ days}$
* **Median:** $16.00\text{ days}$
* **Mode:** The values $15$ and $16$ each appear 3 times (the distribution is bimodal).

* **Analysis:**
  Since the **Mean ($20.42\text{ days}$)** is significantly greater than the **Median ($16.00\text{ days}$)**, the distribution is **positively skewed** (also known as **skewed to the right**).
  $$\text{Mean } (20.42) > \text{Median } (16)$$

* **Justification:**
  An extremely large value on the right side ($72$ days) acts as an outlier. This extreme value pulls the mean upward (to the right) while having almost no effect on the median, resulting in a long tail stretching to the right.

---

### **c) Recommendation and Justification for the Project Manager**

* **Recommendation:**
  The project manager should report the **Median** ($16$ days) as the measure of central tendency to the clients.

* **Justification:**
  1. **Resistance to Outliers:** The dataset contains a severe extreme outlier ($72$ days). The **Mean** is highly sensitive to outliers, rising to $20.42$ days, which is larger than $11$ out of the $12$ (or $91.7\%$) modules in the dataset! Reporting the mean would falsely represent the team's typical capability as being much slower than it actually is.
  2. **Representativeness:** The **Median** is robust (resistant) to outliers. At $16$ days, it perfectly represents the center of the distribution and gives clients an accurate, honest picture of how long a typical module takes to complete.
  3. **Alternative/Contextual Reporting:** If the project manager is discussing total scheduling or resource allocation budgets for future modules, they can mention the total sum or mean. However, to convey the typical delivery cycle time, the **Median** is mathematically the most appropriate measure.

---

## Page 2: Student Information Block
```text
Name: [Your Name]             BITS Student ID: [Your BITS ID]
Course: Introduction to Statistical Methods (S1-26 AIMLCZC418)
Section: Prof. G Krishna Kumari                 Assignment 1
```

---

## 🟦 QUESTION 2: RESEARCH SCHOLARS PROBABILITY (2.5 Marks)

**Problem Statement:**  
Among 48 research scholars, 20 collaborated with other institutes, 18 participated in international conferences, and 8 did both. If a scholar is picked at random, what is the probability they collaborated with other institutes or participated in a conference?

---

### **Step-by-Step Solution:**

#### **Step 1: Define the Events & Sample Space**
Let the total number of scholars in the sample space be $S$:
$$N(S) = 48$$

Define the events:
* **Event $C$:** The selected scholar collaborated with other institutes.
* **Event $I$:** The selected scholar participated in international conferences.
* **Event $C \cap I$:** The selected scholar did both (collaborated AND participated).
* **Event $C \cup I$:** The selected scholar collaborated OR participated in a conference.

---

#### **Step 2: List the Given Quantities**
* $N(C) = 20 \implies P(C) = \frac{N(C)}{N(S)} = \frac{20}{48}$
* $N(I) = 18 \implies P(I) = \frac{N(I)}{N(S)} = \frac{18}{48}$
* $N(C \cap I) = 8 \implies P(C \cap I) = \frac{N(C \cap I)}{N(S)} = \frac{8}{48}$

---

#### **Step 3: Apply the Addition Rule of Probability**
The probability of the union of two events is given by:
$$P(C \cup I) = P(C) + P(I) - P(C \cap I)$$

Substitute the known fractions:
$$P(C \cup I) = \frac{20}{48} + \frac{18}{48} - \frac{8}{48}$$
$$P(C \cup I) = \frac{20 + 18 - 8}{48}$$
$$P(C \cup I) = \frac{30}{48}$$

---

#### **Step 4: Simplify the Fraction**
Divide numerator and denominator by their greatest common divisor ($6$):
$$\frac{30}{48} = \frac{30 \div 6}{48 \div 6} = \frac{5}{8} = 0.625$$

* **Final Answer:** 
  $$\boxed{P(C \cup I) = 0.625\text{ (or } 62.5\%\text{)}}$$

---

### **Visual Aid: Venn Diagram (Highly Recommended to Draw on Paper)**
Draw a rectangular box representing the Sample Space $S = 48$ containing two intersecting circles $C$ and $I$:
* **Intersection ($C \cap I$ - center):** $8$
* **Collaborated Only ($C \setminus I$ - left circle):** $20 - 8 = 12$
* **Conferences Only ($I \setminus C$ - right circle):** $18 - 8 = 10$
* **Neither ($S \setminus (C \cup I)$ - outside circles):** $48 - (12 + 8 + 10) = 48 - 30 = 18$
* **Total in Union ($C \cup I$):** $12 + 8 + 10 = 30$ scholars.
* **Probability:** $30 / 48 = 0.625$ (or $62.5\%$).

---

## Page 3: Student Information Block
```text
Name: [Your Name]             BITS Student ID: [Your BITS ID]
Course: Introduction to Statistical Methods (S1-26 AIMLCZC418)
Section: Prof. G Krishna Kumari                 Assignment 1
```

---

## 🟨 QUESTION 3: NAIVE BAYES FOR FLU PREDICTION (2.5 Marks)

**Problem Statement:**  
You have the following dataset:

| Patient | Fatigue | Loss of Smell | Sore Eyes | Has Flu |
| :---: | :---: | :---: | :---: | :---: |
| P1 | Yes | Yes | No | **Yes** |
| P2 | No | No | Yes | **No** |
| P3 | Yes | No | No | **Yes** |
| P4 | No | Yes | Yes | **No** |

A new patient exhibits: **Fatigue = No, Loss of Smell = Yes, Sore Eyes = No**.  
Apply Naive Bayes to predict if the patient has flu.

---

### **Step-by-Step Solution:**

Let the features of the new patient be denoted as evidence $E$:
$$E = \{\text{Fatigue} = \text{No}, \text{Loss of Smell} = \text{Yes}, \text{Sore Eyes} = \text{No}\}$$

We want to find and compare the posterior probabilities of the target class $Y \in \{\text{Yes}, \text{No}\}$ given $E$:
$$P(\text{Flu} = Y \mid E) \propto P(\text{Flu} = Y) \times P(\text{Fatigue} = \text{No} \mid Y) \times P(\text{Loss of Smell} = \text{Yes} \mid Y) \times P(\text{Sore Eyes} = \text{No} \mid Y)$$

---

#### **Step 1: Compute Prior Class Probabilities**
Total patients $n = 4$:
* $P(\text{Flu} = \text{Yes}) = \frac{2}{4} = 0.5$ (Patients P1, P3)
* $P(\text{Flu} = \text{No}) = \frac{2}{4} = 0.5$ (Patients P2, P4)

---

#### **Step 2: Attempt standard Naive Bayes (Without Laplace Smoothing)**

**1. For class $Y = \text{Yes}$ (P1, P3):**
* $P(\text{Fatigue} = \text{No} \mid \text{Flu} = \text{Yes}) = \frac{0}{2} = 0$ (both P1 and P3 have Fatigue = Yes)
* $P(\text{Loss of Smell} = \text{Yes} \mid \text{Flu} = \text{Yes}) = \frac{1}{2} = 0.5$ (P1 has Yes, P3 has No)
* $P(\text{Sore Eyes} = \text{No} \mid \text{Flu} = \text{Yes}) = \frac{2}{2} = 1.0$ (both P1 and P3 have Sore Eyes = No)

$$\text{Posterior Numerator}_{\text{Yes}} = P(\text{Flu} = \text{Yes}) \times P(\text{Fatigue} = \text{No} \mid \text{Yes}) \times P(\text{Loss of Smell} = \text{Yes} \mid \text{Yes}) \times P(\text{Sore Eyes} = \text{No} \mid \text{Yes})$$
$$\text{Posterior Numerator}_{\text{Yes}} = 0.5 \times 0 \times 0.5 \times 1.0 = \mathbf{0}$$

**2. For class $Y = \text{No}$ (P2, P4):**
* $P(\text{Fatigue} = \text{No} \mid \text{Flu} = \text{No}) = \frac{2}{2} = 1.0$ (both P2 and P4 have Fatigue = No)
* $P(\text{Loss of Smell} = \text{Yes} \mid \text{Flu} = \text{No}) = \frac{1}{2} = 0.5$ (P2 has No, P4 has Yes)
* $P(\text{Sore Eyes} = \text{No} \mid \text{Flu} = \text{No}) = \frac{0}{2} = 0$ (both P2 and P4 have Sore Eyes = Yes)

$$\text{Posterior Numerator}_{\text{No}} = P(\text{Flu} = \text{No}) \times P(\text{Fatigue} = \text{No} \mid \text{No}) \times P(\text{Loss of Smell} = \text{Yes} \mid \text{No}) \times P(\text{Sore Eyes} = \text{No} \mid \text{No})$$
$$\text{Posterior Numerator}_{\text{No}} = 0.5 \times 1.0 \times 0.5 \times 0 = \mathbf{0}$$

⚠️ **The Zero-Frequency Problem:**  
Because $P(\text{Fatigue} = \text{No} \mid \text{Yes}) = 0$ and $P(\text{Sore Eyes} = \text{No} \mid \text{No}) = 0$, both posterior numerators are $0$. Standard Naive Bayes cannot classify the patient ($0/0$ form). Therefore, we must apply **Laplace Smoothing** to resolve this.

---

#### **Step 3: Apply Laplace Smoothing (Laplace Correction)**
Since each feature is binary ($K = 2$ classes: Yes / No), the smoothed conditional probability formula is:
$$P(X_i = x \mid Y = c) = \frac{\text{count}(X_i = x \text{ and } Y = c) + 1}{\text{count}(Y = c) + 2}$$

Let's compute the smoothed conditional probabilities:

**1. For class $Y = \text{Yes}$ (Denominator = $\text{count}(Yes) + 2 = 2 + 2 = 4$):**
* $P(\text{Fatigue} = \text{No} \mid \text{Flu} = \text{Yes}) = \frac{0 + 1}{2 + 2} = \frac{1}{4} = 0.25$
* $P(\text{Loss of Smell} = \text{Yes} \mid \text{Flu} = \text{Yes}) = \frac{1 + 1}{2 + 2} = \frac{2}{4} = 0.50$
* $P(\text{Sore Eyes} = \text{No} \mid \text{Flu} = \text{Yes}) = \frac{2 + 1}{2 + 2} = \frac{3}{4} = 0.75$

Let's compute the smoothed numerator for $Flu = Yes$:
$$\text{Numerator}_{\text{Yes}} = P(\text{Flu} = \text{Yes}) \times P(\text{Fatigue} = \text{No} \mid \text{Yes}) \times P(\text{Loss of Smell} = \text{Yes} \mid \text{Yes}) \times P(\text{Sore Eyes} = \text{No} \mid \text{Yes})$$
$$\text{Numerator}_{\text{Yes}} = 0.5 \times 0.25 \times 0.50 \times 0.75 = 0.046875 \quad \left(\text{or } \frac{3}{64}\right)$$

**2. For class $Y = \text{No}$ (Denominator = $\text{count}(No) + 2 = 2 + 2 = 4$):**
* $P(\text{Fatigue} = \text{No} \mid \text{Flu} = \text{No}) = \frac{2 + 1}{2 + 2} = \frac{3}{4} = 0.75$
* $P(\text{Loss of Smell} = \text{Yes} \mid \text{Flu} = \text{No}) = \frac{1 + 1}{2 + 2} = \frac{2}{4} = 0.50$
* $P(\text{Sore Eyes} = \text{No} \mid \text{Flu} = \text{No}) = \frac{0 + 1}{2 + 2} = \frac{1}{4} = 0.25$

Let's compute the smoothed numerator for $Flu = No$:
$$\text{Numerator}_{\text{No}} = P(\text{Flu} = \text{No}) \times P(\text{Fatigue} = \text{No} \mid \text{No}) \times P(\text{Loss of Smell} = \text{Yes} \mid \text{No}) \times P(\text{Sore Eyes} = \text{No} \mid \text{No})$$
$$\text{Numerator}_{\text{No}} = 0.5 \times 0.75 \times 0.50 \times 0.25 = 0.046875 \quad \left(\text{or } \frac{3}{64}\right)$$

---

#### **Step 4: Normalization & Final Prediction**
The normalized posterior probabilities are:
$$P(\text{Flu} = \text{Yes} \mid E) = \frac{\text{Numerator}_{\text{Yes}}}{\text{Numerator}_{\text{Yes}} + \text{Numerator}_{\text{No}}} = \frac{0.046875}{0.046875 + 0.046875} = 0.50 \quad (50\%)$$
$$P(\text{Flu} = \text{No} \mid E) = \frac{\text{Numerator}_{\text{No}}}{\text{Numerator}_{\text{Yes}} + \text{Numerator}_{\text{No}}} = \frac{0.046875}{0.046875 + 0.046875} = 0.50 \quad (50\%)$$

* **Final Decision:**
  The smoothed Naive Bayes classifier predicts that the patient has an **equal probability of 50% for having the flu and 50% for not having the flu**. 
  
  $$\boxed{\text{Prediction: } 50\% \text{ Yes (Has Flu) / } 50\% \text{ No (No Flu)}}$$

* **Intuitive Justification:**
  Looking at the symptoms of the new patient:
  - **Fatigue = No** points strongly to $Flu = No$ (since 100% of flu-free patients have Fatigue = No, while 0% of flu patients do).
  - **Sore Eyes = No** points strongly to $Flu = Yes$ (since 100% of flu patients have Sore Eyes = No, while 0% of flu-free patients do).
  - **Loss of Smell = Yes** is perfectly neutral (equally likely in both classes).
  Because the symptoms pointing to "Yes" and "No" are perfectly symmetric and balanced, a probability of exactly $50\%$ for each class is mathematically and logically highly intuitive.

---

## Page 4: Student Information Block
```text
Name: [Your Name]             BITS Student ID: [Your BITS ID]
Course: Introduction to Statistical Methods (S1-26 AIMLCZC418)
Section: Prof. G Krishna Kumari                 Assignment 1
```

---

## 🟩 QUESTION 4: IOT SENSOR ANOMALY DETECTION (2.5 Marks)

**Problem Statement:**  
Actual anomalies are rare — only 1% of all readings come from truly abnormal machine behaviour. The anomaly-detection system works as follows:  
* When a real anomaly occurs, the system correctly flags it 88% of the time.  
* When everything is normal, the system still raises a false alarm 6% of the time.  
One of the latest sensor readings has been flagged as anomalous by the system.  
What is the probability that the reading is actually from a real anomaly?

---

### **Step-by-Step Solution:**

#### **Step 1: Define the Events**
* **Event $A$:** The sensor reading is from a **real anomaly**.
* **Event $N$:** The sensor reading is **normal** (no anomaly).
* **Event $F$:** The system **flags** the reading as anomalous.

---

#### **Step 2: State the Given Probabilities**
* **Prior Probabilities:**
  * $P(A) = 1\% = 0.01$ (Probability of anomaly)
  * $P(N) = 1 - P(A) = 99\% = 0.99$ (Probability of normal machine behavior)
* **Conditional Probabilities:**
  * $P(F \mid A) = 88\% = 0.88$ (Sensitivity / True Positive Rate)
  * $P(F \mid N) = 6\% = 0.06$ (False Alarm Rate / False Positive Rate)

---

#### **Step 3: Formulate Bayes' Theorem**
We need to find the posterior probability $P(A \mid F)$ (the probability that the reading is actually an anomaly, given that it was flagged):
$$P(A \mid F) = \frac{P(F \mid A) \cdot P(A)}{P(F)}$$

Where $P(F)$ is the total probability that the system flags a reading as anomalous:
$$P(F) = P(F \mid A) \cdot P(A) + P(F \mid N) \cdot P(N)$$

---

#### **Step 4: Compute the Total Probability of Being Flagged, $P(F)$**
$$P(F) = (0.88 \times 0.01) + (0.06 \times 0.99)$$
$$P(F) = 0.0088 + 0.0594$$
$$P(F) = 0.0682 \quad \text{(or } 6.82\%\text{)}$$

---

#### **Step 5: Compute the Posterior Probability, $P(A \mid F)$**
$$P(A \mid F) = \frac{0.88 \times 0.01}{0.0682} = \frac{0.0088}{0.0682}$$

Multiply numerator and denominator by 10,000 to work with whole numbers:
$$P(A \mid F) = \frac{88}{682}$$

Simplify the fraction by dividing both numerator and denominator by 22:
$$88 \div 22 = 4$$
$$682 \div 22 = 31$$
$$P(A \mid F) = \frac{4}{31} \approx 0.129032\text{ (or } 12.903\%\text{)}$$

* **Final Answer:** 
  $$\boxed{P(A \mid F) = \frac{4}{31} \approx 0.1290\text{ (or } 12.90\%\text{)}}$$

---

### **Justification & Interpretation of the Result:**
Even though the anomaly detection system is highly accurate (correctly identifying 88% of anomalies and having a low false alarm rate of 6%), the probability that a flagged reading is actually a real anomaly is only **12.90%**. 

**Why is it so low?**  
This is a classic demonstration of the **Base Rate Fallacy**. Because real anomalies are extremely rare (only 1% base rate), the absolute number of false alarms from the 99% normal readings ($0.06 \times 0.99 = 5.94\%$) vastly outnumbers the true positive flags ($0.88 \times 0.01 = 0.88\%$). Therefore, out of all flagged readings, only about 1 in 8 is actually a real anomaly.
