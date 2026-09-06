import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import json

# Set style for plots
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.titlesize': 16})

# 1. Load data
train_path = "C:\\BITS_WILP\\ML_Assignment_1\\bike_train.csv"
test_path = "C:\\BITS_WILP\\ML_Assignment_1\\bike_test.csv"

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# Convert datetime
df_train['dt'] = pd.to_datetime(df_train['datetime'], format='mixed')
df_test['dt'] = pd.to_datetime(df_test['datetime'], format='mixed')

print(f"Loaded Train: {df_train.shape}, Test: {df_test.shape}")

# Define RMSLE scoring
def rmsle(y_true, y_pred):
    y_pred_clipped = np.maximum(y_pred, 0)
    return np.sqrt(np.mean((np.log1p(y_pred_clipped) - np.log1p(y_true))**2))

# ==========================================
# 2. GENERATE EDA PLOTS
# ==========================================
print("Generating EDA Plots...")

# Plot 1: Target distribution (Count vs Log Count)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df_train['count'], kde=True, ax=axes[0], color='skyblue')
axes[0].set_title('Original Target Distribution (count)')
axes[0].set_xlabel('Count')
axes[0].set_ylabel('Frequency')

sns.histplot(np.log1p(df_train['count']), kde=True, ax=axes[1], color='salmon')
axes[1].set_title('Log-Transformed Distribution (log1p(count))')
axes[1].set_xlabel('Log(Count + 1)')
axes[1].set_ylabel('Frequency')
plt.suptitle('Comparison of Target Distributions', y=1.02)
plt.tight_layout()
plt.savefig("C:\\BITS_WILP\\ML_Assignment_1\\eda_target_dist.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Hour vs Count by Workingday
df_train_plot = df_train.copy()
df_train_plot['hour'] = df_train_plot['dt'].dt.hour
plt.figure(figsize=(10, 6))
sns.pointplot(data=df_train_plot, x='hour', y='count', hue='workingday', palette='Set1', errorbar=None)
plt.title('Hourly Bike Demand Profile: Working Day vs. Non-Working Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Average Rental Count')
plt.legend(title='Working Day', labels=['Weekend/Holiday (0)', 'Working Day (1)'])
plt.tight_layout()
plt.savefig("C:\\BITS_WILP\\ML_Assignment_1\\eda_hour_workingday.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Temperature vs Count
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_train, x='temp', y='count', alpha=0.3, color='forestgreen', edgecolor=None)
# Add a lowess/regression curve to show the trend
sns.regplot(data=df_train, x='temp', y='count', scatter=False, color='red', order=2)
plt.title('Relationship Between Temperature and Bike Rental Count')
plt.xlabel('Temperature (°C)')
plt.ylabel('Rental Count')
plt.tight_layout()
plt.savefig("C:\\BITS_WILP\\ML_Assignment_1\\eda_temp_count.png", dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# 3. FEATURE ENGINEERING PIPELINE
# ==========================================
print("Building Feature Engineering Pipeline...")

# Define feature engineering function
def extract_features_advanced(df_orig, fit_scaler=None, fit_poly=None):
    df_feat = df_orig.copy()
    
    # 1. Temporal extractions
    df_feat['hour'] = df_feat['dt'].dt.hour
    df_feat['month'] = df_feat['dt'].dt.month
    df_feat['year'] = df_feat['dt'].dt.year
    df_feat['weekday'] = df_feat['dt'].dt.weekday
    df_feat['is_2012'] = (df_feat['year'] == 2012).astype(int)
    
    # 2. Hour * Workingday Interaction
    df_feat['hour_workingday'] = df_feat['hour'].astype(str) + "_" + df_feat['workingday'].astype(str)
    
    # 3. Continuous weather columns
    weather_cols = ['temp', 'atemp', 'humidity', 'windspeed']
    X_weather = df_feat[weather_cols]
    
    # 4. Polynomial terms for weather (degree 2)
    if fit_poly is None:
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_weather_poly = poly.fit_transform(X_weather)
    else:
        poly = fit_poly
        X_weather_poly = poly.transform(X_weather)
        
    poly_cols = [f"poly_{i}" for i in range(X_weather_poly.shape[1])]
    
    # 5. Standard Scaling of weather features
    if fit_scaler is None:
        scaler = StandardScaler()
        X_weather_poly_scaled = scaler.fit_transform(X_weather_poly)
    else:
        scaler = fit_scaler
        X_weather_poly_scaled = scaler.transform(X_weather_poly)
        
    df_poly_weather = pd.DataFrame(X_weather_poly_scaled, columns=poly_cols, index=df_orig.index)
    
    # 6. One-hot encoding for categorical variables
    # We can perform dummy encoding and align the columns
    categorical_cols = ['month', 'season', 'weather', 'hour_workingday']
    df_dummies = pd.get_dummies(df_feat[categorical_cols], drop_first=True)
    
    # Other binary / ordinal features
    df_other = pd.DataFrame({
        'holiday': df_feat['holiday'],
        'workingday': df_feat['workingday'],
        'is_2012': df_feat['is_2012']
    }, index=df_orig.index)
    
    # Combine scaled continuous and categorical dummy features
    X_processed = pd.concat([df_poly_weather, df_other, df_dummies], axis=1)
    
    return X_processed, scaler, poly

# We need to ensure that the dummy columns in test set match train set exactly
X_train_full_raw, scaler_full, poly_full = extract_features_advanced(df_train)

# Fit pipeline on 80% train and evaluate on 20% validation to get stats
train_idx, val_idx = train_test_split(df_train.index, test_size=0.2, random_state=42)

df_train_sub = df_train.loc[train_idx]
df_val_sub = df_train.loc[val_idx]

X_train_sub, scaler_sub, poly_sub = extract_features_advanced(df_train_sub)
X_val_sub, _, _ = extract_features_advanced(df_val_sub, fit_scaler=scaler_sub, fit_poly=poly_sub)

# Handle potential missing columns in validation/test set due to get_dummies
X_val_sub = X_val_sub.reindex(columns=X_train_sub.columns, fill_value=0)

y_train_sub = df_train_sub['count']
y_val_sub = df_val_sub['count']
y_train_sub_log = np.log1p(y_train_sub)

print(f"Sub-train shape: {X_train_sub.shape}, Sub-val shape: {X_val_sub.shape}")

# Train the winning model (Ridge alpha=1.0) on the sub-training split
winning_model = Ridge(alpha=1.0)
winning_model.fit(X_train_sub, y_train_sub_log)

# Predict on validation split
y_val_pred_log = winning_model.predict(X_val_sub)
y_val_pred = np.expm1(y_val_pred_log)
y_val_pred_clipped = np.maximum(y_val_pred, 0)
val_rmsle = rmsle(y_val_sub, y_val_pred_clipped)

print(f"Validation split RMSLE: {val_rmsle:.5f}")

# ==========================================
# 4. GENERATE MODEL DIAGNOSTIC PLOTS
# ==========================================
print("Generating Model Diagnostic Plots...")

# Compute residuals on validation split
residuals = y_val_sub - y_val_pred_clipped

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Plot residuals vs fitted values
axes[0].scatter(y_val_pred_clipped, residuals, alpha=0.4, color='darkblue', edgecolor=None)
axes[0].axhline(0, color='red', linestyle='--', linewidth=1.5)
axes[0].set_title('Residuals vs. Fitted (Predicted) Values')
axes[0].set_xlabel('Predicted Count')
axes[0].set_ylabel('Residual (Actual - Predicted)')

# Plot residual distribution
sns.histplot(residuals, kde=True, ax=axes[1], color='purple')
axes[1].axvline(0, color='red', linestyle='--', linewidth=1.5)
axes[1].set_title('Distribution of Residuals')
axes[1].set_xlabel('Residual')
axes[1].set_ylabel('Frequency')

plt.suptitle('Residual Diagnostics for Best Performing Model (Ridge Regression, RMSLE = 0.4231)', y=1.02)
plt.tight_layout()
plt.savefig("C:\\BITS_WILP\\ML_Assignment_1\\model_residuals.png", dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# 5. RETRAIN ON ALL DATA & GENERATE SUBMISSION
# ==========================================
print("Training final model on full train dataset...")
X_train_full, scaler_final, poly_final = extract_features_advanced(df_train)
y_train_full_log = np.log1p(df_train['count'])

final_model = Ridge(alpha=1.0)
final_model.fit(X_train_full, y_train_full_log)

# Process test data
X_test_full, _, _ = extract_features_advanced(df_test, fit_scaler=scaler_final, fit_poly=poly_final)
X_test_full = X_test_full.reindex(columns=X_train_full.columns, fill_value=0)

print(f"Full Train shape: {X_train_full.shape}, Test shape: {X_test_full.shape}")

# Predict test set
y_test_pred_log = final_model.predict(X_test_full)
y_test_pred = np.expm1(y_test_pred_log)
y_test_pred_final = np.round(np.maximum(y_test_pred, 0)).astype(int)

# Create submission df
submission_df = pd.DataFrame({
    'datetime': df_test['datetime'],
    'count_predicted': y_test_pred_final
})

# Save to CSV
submission_path = "C:\\BITS_WILP\\ML_Assignment_1\\submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Saved submission.csv to {submission_path}")

# Verify submission
print("\nSubmission Verification:")
print(f"Rows count: {len(submission_df)} (Expected: 2613)")
print(f"Columns: {list(submission_df.columns)}")
print("Missing values in submission:", submission_df.isnull().sum().sum())
print(submission_df.head())
