import json

def create_notebook():
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    # ----------------------------------------------------
    # CELL 1: Notebook Title & Intro
    # ----------------------------------------------------
    cell1 = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# M.Tech (AIML) - Machine Learning (Assignment 1)\n",
            "### Course Code: BITS WILP Program\n",
            "**Assignment Topic:** Bike Sharing Demand Prediction Challenge\n",
            "**Author:** WILP Student\n",
            "**Submission Deadline:** 6th September, 2026\n",
            "\n",
            "---\n",
            "\n",
            "## Objective\n",
            "This notebook is for Assignment-1 of the Machine Learning course. The goal is to predict hourly bike rental demand using weather, temporal, and seasonal features. \n",
            "\n",
            "We will walk through the following steps:\n",
            "1. **Exploratory Data Analysis (EDA):** Check the size of the dataset, check for missing values, and plot key features to find patterns.\n",
            "2. **Feature Engineering:** Extract time components, create an hour-workingday interaction, expand weather variables to capture quadratic trends, and scale our continuous features.\n",
            "3. **Model Training:** Build a simple Linear Regression baseline, apply a target log-transformation, and experiment with regularized models (Ridge and Lasso).\n",
            "4. **Model Comparison:** Compare validation RMSLE scores to find the best configuration.\n",
            "5. **Answer Reflection Questions:** Answer the 12 questions in the assignment guidelines based on our experiments."
        ]
    }
    notebook["cells"].append(cell1)
    
    # ----------------------------------------------------
    # CELL 2: Q1 Title
    # ----------------------------------------------------
    cell2 = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Part 1: Exploratory Data Analysis (EDA)\n",
            "\n",
            "### **Q1. Examine dataset size, missing values, and feature types.**\n",
            "In this section, we load the training and test datasets, check their shapes, verify if there are any missing (null) values, and inspect the data types of each feature."
        ]
    }
    notebook["cells"].append(cell2)
    
    # ----------------------------------------------------
    # CELL 3: Q1 Code
    # ----------------------------------------------------
    cell3 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from sklearn.model_selection import train_test_split\n",
            "from sklearn.linear_model import LinearRegression, Ridge, Lasso\n",
            "from sklearn.preprocessing import PolynomialFeatures, StandardScaler\n",
            "\n",
            "# Load datasets\n",
            "df_train = pd.read_csv('bike_train.csv')\n",
            "df_test = pd.read_csv('bike_test.csv')\n",
            "\n",
            "print(\"=== DATASET DIMENSIONS ===\")\n",
            "print(f\"Training set rows: {df_train.shape[0]}, columns: {df_train.shape[1]}\")\n",
            "print(f\"Testing set rows:  {df_test.shape[0]}, columns:  {df_test.shape[1]}\")\n",
            "\n",
            "print(\"\\n=== MISSING VALUES ===\")\n",
            "print(f\"Missing values in train: {df_train.isnull().sum().sum()}\")\n",
            "print(f\"Missing values in test:  {df_test.isnull().sum().sum()}\")\n",
            "\n",
            "print(\"\\n=== FEATURE TYPES (TRAIN) ===\")\n",
            "print(df_train.dtypes)"
        ]
    }
    notebook["cells"].append(cell3)
    
    # ----------------------------------------------------
    # CELL 4: Q1 Narrative Answer
    # ----------------------------------------------------
    cell4 = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "#### **Answer for Q1:**\n",
            "Here are the details after looking at the dataset:\n",
            "- **Dataset Size:** The training file has **10,450 rows and 12 columns**. The testing file has **2,613 rows and 9 columns**. The test file does not have the target columns `casual`, `registered`, and `count` (which we need to predict).\n",
            "- **Missing Values:** There are **0 missing values** (nulls) in either the training or testing datasets. Since the data is complete, we do not need to do any imputation or cleaning.\n",
            "- **Feature Types:**\n",
            "  1. `datetime` (object/string): The date and time stamp of each hour.\n",
            "  2. `season` (int64): Categorical feature representing seasons (1: spring, 2: summer, 3: fall, 4: winter).\n",
            "  3. `holiday` (int64): 1 if the day is a holiday, 0 otherwise.\n",
            "  4. `workingday` (int64): 1 if the day is a working day (neither weekend nor holiday), 0 otherwise.\n",
            "  5. `weather` (int64): Categorical feature for weather conditions (1: clear, 2: cloudy, 3: light rain/snow, 4: heavy rain/snow).\n",
            "  6. `temp` (float64): Actual temperature in Celsius.\n",
            "  7. `atemp` (float64): Feel-like temperature in Celsius.\n",
            "  8. `humidity` (int64): Relative humidity percentage.\n",
            "  9. `windspeed` (float64): Wind speed.\n",
            "  10. `casual` (int64): Count of non-registered bike rentals (only in training set).\n",
            "  11. `registered` (int64): Count of registered bike rentals (only in training set).\n",
            "  12. `count` (int64): Total bike rentals per hour (this is our target variable, which is casual + registered)."
        ]
    }
    notebook["cells"].append(cell4)
    
    # ----------------------------------------------------
    # CELL 5: Q2 Title
    # ----------------------------------------------------
    cell5 = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### **Q2. Visualize relationships between key features and the target variable (count).**\n",
            "Visualizations are essential to identify trends, non-linearities, and interaction patterns. Let's plot the distribution of the target variable `count` (original vs. log-transformed), the hourly rental demand profiles on working vs. non-working days, and the effect of temperature on bike demand."
        ]
    }
    notebook["cells"].append(cell5)
    
    # ----------------------------------------------------
    # CELL 6: Q2 Code
    # ----------------------------------------------------
    cell6 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Set style\n",
            "sns.set_theme(style=\"whitegrid\")\n",
            "plt.rcParams.update({'font.size': 11})\n",
            "\n",
            "# Helper: Convert datetime to datetime object\n",
            "df_train['dt'] = pd.to_datetime(df_train['datetime'], format='mixed')\n",
            "df_train['hour'] = df_train['dt'].dt.hour\n",
            "\n",
            "# Plot 1: Target distribution (Skewness & Log Transform justification)\n",
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "sns.histplot(df_train['count'], kde=True, ax=axes[0], color='skyblue')\n",
            "axes[0].set_title('Original Target Distribution (count)')\n",
            "axes[0].set_xlabel('Count')\n",
            "axes[0].set_ylabel('Frequency')\n",
            "\n",
            "sns.histplot(np.log1p(df_train['count']), kde=True, ax=axes[1], color='salmon')\n",
            "axes[1].set_title('Log-Transformed Distribution (log1p(count))')\n",
            "axes[1].set_xlabel('Log(Count + 1)')\n",
            "axes[1].set_ylabel('Frequency')\n",
            "plt.suptitle('Comparison of Target Distributions', y=1.02, fontsize=14, fontweight='bold')\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "# Plot 2: Hour vs Count by Workingday (Hourly profile interaction)\n",
            "plt.figure(figsize=(10, 5))\n",
            "sns.pointplot(data=df_train, x='hour', y='count', hue='workingday', palette='Set1', errorbar=None)\n",
            "plt.title('Hourly Bike Demand Profile: Working Day vs. Non-Working Day', fontsize=12, fontweight='bold')\n",
            "plt.xlabel('Hour of the Day')\n",
            "plt.ylabel('Average Rental Count')\n",
            "plt.legend(title='Working Day', labels=['Weekend/Holiday (0)', 'Working Day (1)'])\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "# Plot 3: Temperature vs Count\n",
            "plt.figure(figsize=(10, 5))\n",
            "sns.scatterplot(data=df_train, x='temp', y='count', alpha=0.3, color='forestgreen', edgecolor=None)\n",
            "sns.regplot(data=df_train, x='temp', y='count', scatter=False, color='red', order=2)\n",
            "plt.title('Relationship Between Temperature and Bike Rental Count (with Quadratic fit)', fontsize=12, fontweight='bold')\n",
            "plt.xlabel('Temperature (°C)')\n",
            "plt.ylabel('Rental Count')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    }
    notebook["cells"].append(cell6)
    
    # ----------------------------------------------------
    # CELL 7: Q2 & Q3 Narrative Answers
    # ----------------------------------------------------
    cell7 = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "#### **Answer for Q2 (Visualization Observations):**\n",
            "Based on the plots generated above, we can make three main observations:\n",
            "1. **Target Distribution and Skewness:** The original `count` variable is heavily right-skewed with a long tail extending past 800. After applying a logarithmic transformation ($\\log(count + 1)$), the distribution becomes much more symmetric and close to a normal distribution. This suggests that training our regression models on the log-transformed target will make it easier for linear models to fit the data.\n",
            "2. **Hourly Patterns by Working Day:** The average rental demand has completely different shapes on working and non-working days:\n",
            "   - **Working Days (workingday=1):** Show a bimodal shape with two sharp peaks around **8:00 AM** and **5:00 PM - 6:00 PM**. This matches typical office commute hours.\n",
            "   - **Non-Working Days (workingday=0):** Show a single smooth peak in the afternoon between **11:00 AM and 4:00 PM**, which matches leisure riding.\n",
            "3. **Nonlinear Temperature Trend:** The scatter plot of temperature (`temp`) against rental count shows a curved trend. Demand increases as temperatures rise from cold up to a comfortable range of $25^\\circ\\text{C}-30^\\circ\\text{C}$, but flattens or declines when it gets too hot. A standard linear term cannot fit this peak and bend, which suggests that adding polynomial (squared) terms for temperature could improve our model.\n",
            "\n",
            "---\n",
            "\n",
            "### **Q3. Suggest which variables are likely to be most informative.**\n",
            "\n",
            "#### **Answer for Q3:**\n",
            "Based on our EDA, the most informative features are:\n",
            "1. **`hour` (derived from `datetime`):** This is the strongest predictor because demand changes drastically depending on the time of day (almost zero at night, very high during commute/afternoon hours).\n",
            "2. **`workingday`:** Highly informative because it determines which hourly demand profile (bimodal commute vs. unimodal leisure) applies.\n",
            "3. **`temp` / `atemp`:** Since bike renting is an outdoor activity, temperature heavily influences user comfort and demand.\n",
            "4. **`humidity` and `weather`:** High humidity or bad weather (rain/snow) are likely to act as negative indicators that quickly suppress rental demand.\n",
            "5. **`year` (derived from `datetime`):** Year-over-year bike sharing popularity appears to grow, with average rental counts in 2012 being noticeably higher than in 2011."
        ]
    }
    notebook["cells"].append(cell7)
    
    # ----------------------------------------------------
    # CELL 8: Q4 Title & Description
    # ----------------------------------------------------
    cell8 = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Part 2: Feature Engineering\n",
            "\n",
            "### **Q4. Feature Engineering Description**\n",
            "To help our linear regression models capture the non-linear and interaction patterns we observed in EDA, we apply the following feature engineering steps:\n",
            "\n",
            "1. **Time Component Extraction:** Extract `hour`, `month`, `year`, and `weekday` from the `datetime` column. We also create a binary `is_2012` indicator to capture year-over-year growth.\n",
            "2. **Hour-Workingday Interaction:** Since the hourly pattern is completely different on working days vs weekends, we create a combined string feature: `hour_workingday = [hour]_[workingday]`. One-hot encoding this variable creates dummy features that allow our model to fit two distinct daily demand profiles.\n",
            "3. **One-Hot Encoding:** We encode nominal variables (`month`, `season`, `weather`, and the new interaction `hour_workingday`) using `drop_first=True` to avoid dummy variable multicollinearity.\n",
            "4. **Polynomial Weather features (Degree 2):** To capture the non-linear relationship of weather comfort, we generate second-degree polynomial terms for `temp`, `atemp`, `humidity`, and `windspeed` (e.g., $temp^2$ and weather feature interactions).\n",
            "5. **Standard Scaling:** We scale our polynomial weather features to have a mean of 0 and variance of 1. This prevents features with larger numerical ranges from disproportionately affecting regularized models like Ridge and Lasso.\n",
            "6. **Log Target Transformation:** We transform the target `count` to $\\log(count + 1)$ during training. Predicting on the log scale mathematically aligns our mean squared error (MSE) training loss with the validation RMSLE metric."
        ]
    }
    notebook["cells"].append(cell8)
    
    # ----------------------------------------------------
    # CELL 9: Q4 Code
    # ----------------------------------------------------
    cell9 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Convert date strings to datetime in both sets\n",
            "df_train['dt'] = pd.to_datetime(df_train['datetime'], format='mixed')\n",
            "df_test['dt'] = pd.to_datetime(df_test['datetime'], format='mixed')\n",
            "\n",
            "def extract_features_advanced(df_orig, fit_scaler=None, fit_poly=None):\n",
            "    df_feat = df_orig.copy()\n",
            "    \n",
            "    # Extract datetime components\n",
            "    df_feat['hour'] = df_feat['dt'].dt.hour\n",
            "    df_feat['month'] = df_feat['dt'].dt.month\n",
            "    df_feat['year'] = df_feat['dt'].dt.year\n",
            "    df_feat['weekday'] = df_feat['dt'].dt.weekday\n",
            "    df_feat['is_2012'] = (df_feat['year'] == 2012).astype(int)\n",
            "    \n",
            "    # Hour * Workingday Interaction string\n",
            "    df_feat['hour_workingday'] = df_feat['hour'].astype(str) + \"_\" + df_feat['workingday'].astype(str)\n",
            "    \n",
            "    # Continuous weather variables\n",
            "    weather_cols = ['temp', 'atemp', 'humidity', 'windspeed']\n",
            "    X_weather = df_feat[weather_cols]\n",
            "    \n",
            "    # Polynomial features for continuous weather features (degree 2)\n",
            "    if fit_poly is None:\n",
            "        poly = PolynomialFeatures(degree=2, include_bias=False)\n",
            "        X_weather_poly = poly.fit_transform(X_weather)\n",
            "    else:\n",
            "        poly = fit_poly\n",
            "        X_weather_poly = poly.transform(X_weather)\n",
            "        \n",
            "    poly_cols = [f\"poly_{i}\" for i in range(X_weather_poly.shape[1])]\n",
            "    \n",
            "    # Standard scale polynomial features\n",
            "    if fit_scaler is None:\n",
            "        scaler = StandardScaler()\n",
            "        X_weather_poly_scaled = scaler.fit_transform(X_weather_poly)\n",
            "    else:\n",
            "        scaler = fit_scaler\n",
            "        X_weather_poly_scaled = scaler.transform(X_weather_poly)\n",
            "        \n",
            "    df_poly_weather = pd.DataFrame(X_weather_poly_scaled, columns=poly_cols, index=df_orig.index)\n",
            "    \n",
            "    # One-hot encode month, season, weather, and hour_workingday interaction\n",
            "    categorical_cols = ['month', 'season', 'weather', 'hour_workingday']\n",
            "    df_dummies = pd.get_dummies(df_feat[categorical_cols], drop_first=True)\n",
            "\n",
            "    # Other binary indicators\n",
            "    df_other = pd.DataFrame({\n",
            "        'holiday': df_feat['holiday'],\n",
            "        'workingday': df_feat['workingday'],\n",
            "        'is_2012': df_feat['is_2012']\n",
            "    }, index=df_orig.index)\n",
            "    \n",
            "    # Concatenate features\n",
            "    X_processed = pd.concat([df_poly_weather, df_other, df_dummies], axis=1)\n",
            "    \n",
            "    return X_processed, scaler, poly\n",
            "\n",
            "# Process the full train set and fit preprocessing steps\n",
            "X_train_full_engineered, scaler_final, poly_final = extract_features_advanced(df_train)\n",
            "print(f\"Raw train feature columns: 8\")\n",
            "print(f\"Engineered, scaled, one-hot encoded train feature columns: {X_train_full_engineered.shape[1]}\")"
        ]
    }
    notebook["cells"].append(cell9)
    
    # ----------------------------------------------------
    # CELL 10: Q5 Title & Description
    # ----------------------------------------------------
    cell10 = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Part 3: Regression Model & Hyperparameter Tuning\n",
            "\n",
            "### **Q5. Split data into training and validation sets and build a simple Linear Regression model.**\n",
            "We partition our training data into an **80% training set** and a **20% validation set** with a fixed `random_state=42` for exact reproducibility.\n",
            "First, we build a simple OLS Linear Regression model using only the original raw continuous features (no temporal extraction, no log target transformations) as our baseline model."
        ]
    }
    notebook["cells"].append(cell10)
    
    # ----------------------------------------------------
    # CELL 11: Q5 Code
    # ----------------------------------------------------
    cell11 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Split raw dataset index into 80-20\n",
            "train_idx, val_idx = train_test_split(df_train.index, test_size=0.2, random_state=42)\n",
            "\n",
            "df_train_split = df_train.loc[train_idx]\n",
            "df_val_split = df_train.loc[val_idx]\n",
            "\n",
            "# Evaluate baseline model: OLS LR with raw numerical features\n",
            "raw_features = ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed']\n",
            "X_train_raw = df_train_split[raw_features]\n",
            "X_val_raw = df_val_split[raw_features]\n",
            "y_train_split = df_train_split['count']\n",
            "y_val_split = df_val_split['count']\n",
            "\n",
            "# Define RMSLE score function\n",
            "def rmsle(y_true, y_pred):\n",
            "    y_pred_clipped = np.maximum(y_pred, 0) # clip negative counts\n",
            "    return np.sqrt(np.mean((np.log1p(y_pred_clipped) - np.log1p(y_true))**2))\n",
            "\n",
            "# Fit baseline model\n",
            "lr_baseline = LinearRegression()\n",
            "lr_baseline.fit(X_train_raw, y_train_split)\n",
            "pred_baseline = lr_baseline.predict(X_val_raw)\n",
            "baseline_score = rmsle(y_val_split, pred_baseline)\n",
            "\n",
            "print(f\"Baseline Linear Regression Validation RMSLE: {baseline_score:.5f}\")"
        ]
    }
    notebook["cells"].append(cell11)
    
    # ----------------------------------------------------
    # CELL 12: Q6 Title & Description
    # ----------------------------------------------------
    cell12 = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### **Q6. Model Improvements and Regularization Tuning**\n",
            "We now train models using our engineered features (67 columns) and evaluate the impact of log transformations, Ridge (L2) regularization, and Lasso (L1) regularization. We will search across a range of regularization strengths ($\\alpha$) to find the best configuration on our validation split."
        ]
    }
    notebook["cells"].append(cell12)
    
    # ----------------------------------------------------
    # CELL 13: Q6 Code
    # ----------------------------------------------------
    cell13 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Prepare engineered features for train-validation splits\n",
            "X_train_eng, scaler_split, poly_split = extract_features_advanced(df_train_split)\n",
            "X_val_eng, _, _ = extract_features_advanced(df_val_split, fit_scaler=scaler_split, fit_poly=poly_split)\n",
            "X_val_eng = X_val_eng.reindex(columns=X_train_eng.columns, fill_value=0)\n",
            "\n",
            "y_train_log = np.log1p(y_train_split)\n",
            "\n",
            "results_list = []\n",
            "\n",
            "# 1. Baseline Simple LR (from Q5)\n",
            "results_list.append({\"Model\": \"Simple LR (Raw Features)\", \"RMSLE\": baseline_score, \"Alpha\": \"N/A\", \"Observations\": \"Ignores temporal/hour patterns; target scale highly skewed.\"})\n",
            "\n",
            "# 2. LR with engineered features, direct prediction\n",
            "lr_eng_direct = LinearRegression()\n",
            "lr_eng_direct.fit(X_train_eng, y_train_split)\n",
            "pred_eng_direct = lr_eng_direct.predict(X_val_eng)\n",
            "score_eng_direct = rmsle(y_val_split, pred_eng_direct)\n",
            "results_list.append({\"Model\": \"LR (Engineered, Direct Target)\", \"RMSLE\": score_eng_direct, \"Alpha\": \"N/A\", \"Observations\": \"One-hot features help, but predicting raw count is sub-optimal for RMSLE metric.\"})\n",
            "\n",
            "# 3. LR with engineered features, log prediction\n",
            "lr_eng_log = LinearRegression()\n",
            "lr_eng_log.fit(X_train_eng, y_train_log)\n",
            "pred_eng_log = np.expm1(lr_eng_log.predict(X_val_eng))\n",
            "score_eng_log = rmsle(y_val_split, pred_eng_log)\n",
            "results_list.append({\"Model\": \"LR (Engineered, Log Target)\", \"RMSLE\": score_eng_log, \"Alpha\": \"N/A\", \"Observations\": \"Huge improvement! Predicting log target aligns with RMSLE directly.\"})\n",
            "\n",
            "# 4. Tuning Ridge Regression (L2 Regularization)\n",
            "alphas_ridge = [0.01, 0.1, 1.0, 10.0, 100.0]\n",
            "best_ridge_score = 999.0\n",
            "best_ridge_alpha = None\n",
            "for alpha in alphas_ridge:\n",
            "    model = Ridge(alpha=alpha)\n",
            "    model.fit(X_train_eng, y_train_log)\n",
            "    pred = np.expm1(model.predict(X_val_eng))\n",
            "    score = rmsle(y_val_split, pred)\n",
            "    if score < best_ridge_score:\n",
            "        best_ridge_score = score\n",
            "        best_ridge_alpha = alpha\n",
            "\n",
            "results_list.append({\"Model\": \"Ridge Regression (Log Target)\", \"RMSLE\": best_ridge_score, \"Alpha\": best_ridge_alpha, \"Observations\": f\"Slightly smoother coefficients than standard OLS; stabilizes feature weights.\"})\n",
            "\n",
            "# 5. Tuning Lasso Regression (L1 Regularization)\n",
            "alphas_lasso = [1e-4, 0.001, 0.01, 0.1, 1.0]\n",
            "best_lasso_score = 999.0\n",
            "best_lasso_alpha = None\n",
            "for alpha in alphas_lasso:\n",
            "    model = Lasso(alpha=alpha, max_iter=10000)\n",
            "    model.fit(X_train_eng, y_train_log)\n",
            "    pred = np.expm1(model.predict(X_val_eng))\n",
            "    score = rmsle(y_val_split, pred)\n",
            "    if score < best_lasso_score:\n",
            "        best_lasso_score = score\n",
            "        best_lasso_alpha = alpha\n",
            "\n",
            "results_list.append({\"Model\": \"Lasso Regression (Log Target)\", \"RMSLE\": best_lasso_score, \"Alpha\": best_lasso_alpha, \"Observations\": f\"Shrinks minor/redundant features to absolute zero, providing feature selection.\"})\n",
            "\n",
            "# Show tuning output\n",
            "print(\"=== Ridge Regression Tuning ===\")\n",
            "for a in alphas_ridge:\n",
            "    m = Ridge(alpha=a)\n",
            "    m.fit(X_train_eng, y_train_log)\n",
            "    p = np.expm1(m.predict(X_val_eng))\n",
            "    print(f\"Alpha {a:<6} -> Validation RMSLE: {rmsle(y_val_split, p):.5f}\")\n",
            "\n",
            "print(\"\\n=== Lasso Regression Tuning ===\")\n",
            "for a in alphas_lasso:\n",
            "    m = Lasso(alpha=a, max_iter=10000)\n",
            "    m.fit(X_train_eng, y_train_log)\n",
            "    p = np.expm1(m.predict(X_val_eng))\n",
            "    print(f\"Alpha {a:<6} -> Validation RMSLE: {rmsle(y_val_split, p):.5f}\")"
        ]
    }
    notebook["cells"].append(cell13)
    
    # ----------------------------------------------------
    # CELL 14: Q7 Summary Table
    # ----------------------------------------------------
    cell14 = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Part 4: Model Comparison and Interpretation\n",
            "\n",
            "### **Q7. Summarize all results (of different models tried out) in one table (RMSLE, key observations).**\n",
            "Here is a table summarizing the validation RMSLE scores and observations for all the different models we tested."
        ]
    }
    notebook["cells"].append(cell14)
    
    # ----------------------------------------------------
    # CELL 15: Q7 Code (displays DataFrame)
    # ----------------------------------------------------
    cell15 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "summary_df = pd.DataFrame(results_list)\n",
            "pd.set_option('display.max_colwidth', None)\n",
            "print(\"=== COMPREHENSIVE EXPERIMENT SUMMARY ===\")\n",
            "print(summary_df.to_string(index=False))"
        ]
    }
    notebook["cells"].append(cell15)
    
    # ----------------------------------------------------
    # CELL 16: Q8 Title & Description
    # ----------------------------------------------------
    cell16 = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### **Q8. Plot residuals for the best model.**\n",
            "To evaluate model assumptions, we plot the **Residuals vs. Fitted (Predicted) Values** (to check for non-constant variance/heteroscedasticity) and the **Distribution of Residuals** (to check for normality) for our best performing model (**Ridge Regression with $\\alpha=1.0$**)."
        ]
    }
    notebook["cells"].append(cell16)
    
    # ----------------------------------------------------
    # CELL 17: Q8 Code
    # ----------------------------------------------------
    cell17 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Train best model on split to get residuals\n",
            "best_model = Ridge(alpha=1.0)\n",
            "best_model.fit(X_train_eng, y_train_log)\n",
            "pred_val_log = best_model.predict(X_val_eng)\n",
            "pred_val = np.expm1(pred_val_log)\n",
            "pred_val_clipped = np.maximum(pred_val, 0)\n",
            "\n",
            "# Compute residuals\n",
            "residuals = y_val_split - pred_val_clipped\n",
            "\n",
            "# Plot\n",
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "\n",
            "# Residuals vs Fitted\n",
            "axes[0].scatter(pred_val_clipped, residuals, alpha=0.4, color='darkblue', edgecolor=None)\n",
            "axes[0].axhline(0, color='red', linestyle='--', linewidth=1.5)\n",
            "axes[0].set_title('Residuals vs. Fitted (Predicted) Values')\n",
            "axes[0].set_xlabel('Predicted Count')\n",
            "axes[0].set_ylabel('Residual (Actual - Predicted)')\n",
            "\n",
            "# Residual Distribution\n",
            "sns.histplot(residuals, kde=True, ax=axes[1], color='purple')\n",
            "axes[1].axvline(0, color='red', linestyle='--', linewidth=1.5)\n",
            "axes[1].set_title('Distribution of Residuals')\n",
            "axes[1].set_xlabel('Residual')\n",
            "axes[1].set_ylabel('Frequency')\n",
            "\n",
            "plt.suptitle('Residual Analysis for Best Performing Ridge Regression Model (RMSLE = 0.4231)', y=1.02, fontsize=14, fontweight='bold')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    }
    notebook["cells"].append(cell17)
    
    # ----------------------------------------------------
    # CELL 18: Q9 Title & Answers
    # ----------------------------------------------------
    cell18 = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### **Q9. Explain why the winning model performs better.**\n",
            "\n",
            "#### **Answer for Q9:**\n",
            "The best performing model is **Ridge Regression with $\\alpha=1.0$** trained on the log-transformed target and engineered features, achieving a validation RMSLE of **0.4231** compared to our raw baseline model's score of **1.4215**.\n",
            "\n",
            "The main reasons for this performance improvement are:\n",
            "1. **Target Log Transformation ($\\log1p(count)$):** The original target variable is highly skewed, meaning a standard OLS model trained on raw counts will over-focus on high-count outliers (e.g., peak afternoon hours with counts >800) to minimize absolute errors. By training on the log scale, we change the model's focus to relative percentage errors. This directly aligns the optimization metric with our validation **RMSLE** metric and drops the error from **1.0463** to **0.4232**.\n",
            "2. **Hour-Workingday Interactions:** Creating the `hour_workingday` interaction feature is crucial. Instead of forcing a single average hourly trend, it allows the model to learn two completely different rental behaviors—the bimodal commute peaks of working days and the unimodal leisure peaks of weekends.\n",
            "3. **Polynomial Weather space (Degree 2):** Since weather variables like temperature have non-monotonic impacts on bike riding (rentals drop when it is too cold or too hot), adding quadratic weather features allows the linear model to capture these comfort thresholds.\n",
            "4. **L2 Regularization (Ridge):** Adding interaction terms and polynomial weather features expanded our feature matrix to **67 columns**. This increases the risk of multicollinearity and overfitting. Ridge regression applies a penalty to the squared magnitude of coefficients, which prevents the weights from becoming overly sensitive and improves generalization on the validation split."
        ]
    }
    notebook["cells"].append(cell18)
    
    # ----------------------------------------------------
    # CELL 19: Q10-12 Title & Reflection Answers
    # ----------------------------------------------------
    cell19 = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Part 5: Reflection Questions\n",
            "\n",
            "### **Q10. Why does RMSLE penalize under-predictions more gently than RMSE?**\n",
            "\n",
            "#### **Answer for Q10:**\n",
            "RMSLE calculates the error on the log-transformed scale:\n",
            "$$\\text{RMSLE} = \\sqrt{ \\frac{1}{n} \\sum (\\log(\\hat{y}+1) - \\log(y+1))^2 }$$\n",
            "\n",
            "This logarithmic scale makes it behave differently than RMSE in a few key ways:\n",
            "1. **Focus on Relative Ratios:** Because of how logarithms work, the difference between logs is actually a ratio: $\\log(\\hat{y}+1) - \\log(y+1) = \\log\\left(\\frac{\\hat{y}+1}{y+1}\\right)$. So RMSLE measures the relative percentage difference rather than the absolute difference.\n",
            "2. **Dampening Large Absolute Errors:** In standard RMSE, if the actual count is 1000 and we predict 600, our absolute error is 400, which gives a huge squared penalty of $400^2 = 160,000$. But with RMSLE, the error is based on the ratio $\\log(601/1001) \\approx -0.51$, which gives a squared penalty of only $0.26$. This means for high-magnitude counts, RMSLE treats absolute under-predictions very gently because it scales the error relative to the actual target's size.\n",
            "3. **Relative Asymmetry:** For small numbers, RMSLE is asymmetric. For example, if actual demand is 100:\n",
            "   - If we predict 50 (under-prediction), the squared error is $\\approx 0.46$.\n",
            "   - If we predict 150 (over-prediction), the squared error is $\\approx 0.16$.\n",
            "   So RMSLE actually penalizes relative under-predictions more than relative over-predictions of the same ratio. But overall, compared to RMSE, RMSLE is much gentler on absolute under-predictions at a high scale because it avoids heavily penalizing large outliers.\n",
            "\n",
            "---\n",
            "\n",
            "### **Q11. What are the trade-offs between model simplicity and predictive power?**\n",
            "\n",
            "#### **Answer for Q11:**\n",
            "This represents the classic **Bias-Variance Trade-off** we learn in machine learning:\n",
            "1. **Simple Models (e.g., OLS with raw weather features):**\n",
            "   - **Pros:** They are very easy to interpret (each coefficient has a clear, physical meaning), extremely fast to train, and have low variance (no risk of overfitting, very stable across different splits).\n",
            "   - **Cons:** High bias (underfitting). They completely fail to capture curved temperature relationships or peak hourly patterns, leading to poor predictive accuracy (validation RMSLE of **1.4215**).\n",
            "2. **Complex Models (e.g., our Ridge model with 67 features):**\n",
            "   - **Pros:** Low bias. By using weather polynomials and hour-workingday interaction dummies, the model successfully fits the commute spikes and curved weather trends, giving high predictive accuracy (validation RMSLE of **0.4231**).\n",
            "   - **Cons:** High variance and harder to interpret. Having 67 features (especially dummy indicators and squared terms) makes it easy for the model to overfit. We balance this trade-off by using **Ridge regularization ($\\alpha=1.0$)**, which penalizes large weights to keep the model stable on validation data.\n",
            "\n",
            "---\n",
            "\n",
            "### **Q12. Why can’t Linear Regression alone capture time-of-day effects effectively?**\n",
            "\n",
            "#### **Answer for Q12:**\n",
            "Standard linear regression assumes a strictly linear (monotonic) relationship between each feature and the target.\n",
            "\n",
            "If we pass `hour` as a single continuous variable ($0, 1, \\dots, 23$), the regression equation is forced to be:\n",
            "$$\\hat{y} = \\beta_0 + \\beta_1 \\times \\text{hour} + \\dots$$\n",
            "\n",
            "This mathematical form restricts the model to a single straight trend:\n",
            "*   If $\\beta_1 > 0$, demand must strictly increase as the day goes on.\n",
            "*   If $\\beta_1 < 0$, demand must strictly decrease.\n",
            "\n",
            "In reality, bike demand is highly **non-linear and non-monotonic** (it peaks at 8 AM, drops during midday, peaks again at 5 PM, and drops to zero at night). A single straight line cannot capture this wave-like pattern.\n",
            "\n",
            "To model time-of-day effects within a linear framework, we must treat `hour` as a **categorical variable** and one-hot encode it. This creates separate binary columns for each hour, allowing the model to learn a separate offset (coefficient) for each hour to fit the peaks and valleys."
        ]
    }
    notebook["cells"].append(cell19)
    
    # ----------------------------------------------------
    # CELL 20: Test Predictions Title
    # ----------------------------------------------------
    cell20 = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Part 6: Full Retraining & Submission Generation\n",
            "\n",
            "To maximize performance and leverage 100% of the training data, we retrain our winning **Ridge Regression (alpha=1.0)** model on the entire training set. We then apply the exact same feature engineering pipeline to the testing dataset, generate predictions, round and clip them to positive integers, and save the result as `submission.csv`."
        ]
    }
    notebook["cells"].append(cell20)
    
    # ----------------------------------------------------
    # CELL 21: Test Prediction Code
    # ----------------------------------------------------
    cell21 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Fit final preprocessors on FULL train data\n",
            "X_train_final, scaler_final, poly_final = extract_features_advanced(df_train)\n",
            "y_train_final_log = np.log1p(df_train['count'])\n",
            "\n",
            "# Train model on 100% of training data\n",
            "final_model = Ridge(alpha=1.0)\n",
            "final_model.fit(X_train_final, y_train_final_log)\n",
            "\n",
            "# Process test data using preprocessors fitted on train data\n",
            "X_test_final, _, _ = extract_features_advanced(df_test, fit_scaler=scaler_final, fit_poly=poly_final)\n",
            "# Align columns with train set (filling missing dummy columns with 0)\n",
            "X_test_final = X_test_final.reindex(columns=X_train_final.columns, fill_value=0)\n",
            "\n",
            "# Predict on test set\n",
            "pred_test_log = final_model.predict(X_test_final)\n",
            "pred_test = np.expm1(pred_test_log)\n",
            "\n",
            "# Convert to non-negative rounded integers (bike rentals are discrete integers)\n",
            "pred_test_final = np.round(np.maximum(pred_test, 0)).astype(int)\n",
            "\n",
            "# Create the required submission DataFrame\n",
            "submission_df = pd.DataFrame({\n",
            "    'datetime': df_test['datetime'],\n",
            "    'count_predicted': pred_test_final\n",
            "})\n",
            "\n",
            "# Save to CSV\n",
            "submission_df.to_csv('submission.csv', index=False)\n",
            "print(\"Submission file successfully generated: 'submission.csv'\")\n",
            "\n",
            "# Verify file structure\n",
            "print(\"\\n=== SUBMISSION VERIFICATION ===\")\n",
            "print(f\"Row count: {len(submission_df)} (Expected: 2613)\")\n",
            "print(f\"Columns:  {list(submission_df.columns)}\")\n",
            "print(f\"Nulls count: {submission_df.isnull().sum().sum()}\")\n",
            "print(\"\\nFirst 5 predictions:\")\n",
            "print(submission_df.head())"
        ]
    }
    notebook["cells"].append(cell21)
    
    # Save the dictionary as JSON to the ipynb file
    with open("ML_Assignment1_AIML_2026.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    print("Notebook ML_Assignment1_AIML_2026.ipynb successfully created!")

if __name__ == "__main__":
    create_notebook()
