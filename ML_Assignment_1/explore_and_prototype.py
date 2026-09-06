import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# 1. Load data
train_path = "C:\\BITS_WILP\\ML_Assignment_1\\bike_train.csv"
df = pd.read_csv(train_path)

# Convert datetime
df['dt'] = pd.to_datetime(df['datetime'], format='mixed')

# Define RMSLE scoring function
def rmsle(y_true, y_pred):
    y_pred_clipped = np.maximum(y_pred, 0) # Clip negative predictions to 0
    return np.sqrt(np.mean((np.log1p(y_pred_clipped) - np.log1p(y_true))**2))

# Split data into train and validation (80-20)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

y_train = train_df['count']
y_val = val_df['count']
y_train_log = np.log1p(y_train)

# Advanced features with interaction and polynomial weather features
def get_poly_adv_features(df_orig, degree=2):
    # Continuous cols
    weather_cols = ['temp', 'atemp', 'humidity', 'windspeed']
    X_weather = df_orig[weather_cols]
    
    # Polynomial features for weather
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_weather_poly = poly.fit_transform(X_weather)
    poly_cols = [f"poly_{i}" for i in range(X_weather_poly.shape[1])]
    
    # Scale weather
    scaler = StandardScaler()
    X_weather_poly_scaled = scaler.fit_transform(X_weather_poly)
    df_poly_weather = pd.DataFrame(X_weather_poly_scaled, columns=poly_cols, index=df_orig.index)
    
    # Non-weather advanced features (interaction and others)
    df_feat = df_orig.copy()
    df_feat['hour'] = df_feat['dt'].dt.hour
    df_feat['month'] = df_feat['dt'].dt.month
    df_feat['year'] = df_feat['dt'].dt.year
    df_feat['weekday'] = df_feat['dt'].dt.weekday
    
    # Create hour-workingday interaction string to one-hot encode
    df_feat['hour_workingday'] = df_feat['hour'].astype(str) + "_" + df_feat['workingday'].astype(str)
    
    # One-hot encode
    df_encoded = pd.get_dummies(df_feat, columns=['month', 'season', 'weather', 'hour_workingday'], drop_first=True)
    df_encoded['is_2012'] = (df_encoded['year'] == 2012).astype(int)
    
    # Drop columns
    cols_to_drop = ['datetime', 'dt', 'year', 'weekday', 'hour', 'holiday', 'workingday'] + weather_cols
    for col in ['casual', 'registered', 'count']:
        if col in df_encoded.columns:
            cols_to_drop.append(col)
            
    df_other = df_encoded.drop(columns=cols_to_drop, errors='ignore')
    
    # Add holiday and workingday back as binary (since they are useful)
    df_other['holiday'] = df_feat['holiday']
    df_other['workingday'] = df_feat['workingday']
    
    return pd.concat([df_poly_weather, df_other], axis=1)

X_poly2_adv = get_poly_adv_features(df, degree=2)
X_train_poly2, X_val_poly2 = train_test_split(X_poly2_adv, test_size=0.2, random_state=42)

print("Columns in Poly2+Adv:", X_train_poly2.shape[1])

# Linear Regression
lr_poly2 = LinearRegression()
lr_poly2.fit(X_train_poly2, y_train_log)
pred_poly2 = np.expm1(lr_poly2.predict(X_val_poly2))
print("Poly2 + Interaction LR RMSLE:", rmsle(y_val, pred_poly2))

# Ridge Regression tuning
for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_poly2, y_train_log)
    pred_r = np.expm1(ridge.predict(X_val_poly2))
    print(f"Poly2 + Interaction Ridge (alpha={alpha}) RMSLE:", rmsle(y_val, pred_r))

# Lasso Regression tuning
for alpha in [1e-4, 0.001, 0.01, 0.1]:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train_poly2, y_train_log)
    pred_l = np.expm1(lasso.predict(X_val_poly2))
    print(f"Poly2 + Interaction Lasso (alpha={alpha}) RMSLE:", rmsle(y_val, pred_l))
