# Importing necessary libraries
from sqlalchemy import create_engine
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import re
from sklearn.model_selection import RandomizedSearchCV
import shap
import numpy as np
import joblib
from scipy.stats import ks_2samp

# Establishing connection to the database
engine = create_engine("postgresql://user:password@localhost:5432/database_name")

# Reading data from the database into a DataFrame
df = pd.read_sql("SELECT * FROM master_training_data_v4", con=engine)

# Displaying initial memory usage of the DataFrame
df.info(memory_usage='deep')

# Function to reduce memory usage of the DataFrame
def reduce_memory_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == "float64":
            df[col] = pd.to_numeric(df[col], downcast = "float" )
        elif col_type == "int64":
            df[col] = pd.to_numeric(df[col], downcast = "integer" )

# Reducing memory usage of the DataFrame
reduce_memory_usage(df)

# Displaying memory usage after reduction
df.info(memory_usage="deep")

# Separating target variable and features
y = df["TARGET"]
x = df.drop(columns=["SK_ID_CURR","TARGET"], inplace=False)

# Converting categorical variables to dummy variables
x = pd.get_dummies(x, drop_first=True)

# Cleaning column names by removing special characters
x = x.rename(columns=lambda col: re.sub('[^A-Za-z0-9_]+','_',col))

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x ,y , test_size=0.2, random_state=42)

# Count of each class in the training set
y_train_count = y_train.value_counts()

# Checking imbalance in the target variable
imbalance_ratio = y_train_count[0] / y_train_count[1]

# 1. Define the Grid (The settings we want the robot to test)
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, -1],
    'num_leaves': [15, 31, 63, 127],
    'n_estimators': [100, 300, 500],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

# 2. Initialize the Base Model
base_lgbm = LGBMClassifier(random_state=42, verbose=-1, scale_pos_weight=imbalance_ratio)

# 3. Set up the Search Robot
# n_iter=15 means it will randomly test 15 different combinations from the grid above
# cv=3 means it will cross-validate each test 3 times to ensure it wasn't a fluke
random_search = RandomizedSearchCV(
    estimator=base_lgbm, 
    param_distributions=param_grid, 
    n_iter=15, 
    scoring='roc_auc', 
    cv=3, 
    verbose=2, 
    random_state=42, 
    n_jobs=-1  # Uses all your CPU cores to make it faster
)

# 4. Release the Robot on the Training Data
print("Starting Hyperparameter Tuning... This may take a few minutes.")
random_search.fit(X_train, y_train)

# 5. Extract the Winner
Lgbm_model = random_search.best_estimator_

print("\n--- TUNING COMPLETE ---")
print("Best Parameters Found:")
print(random_search.best_params_)

# Making predictions with the LightGBM model
y_pred_Lgbm = Lgbm_model.predict(X_test)
y_pred_prob_Lgbm = Lgbm_model.predict_proba(X_test)[:, 1]

# creating custom threshold for classification
custom_prediction = (y_pred_prob_Lgbm >= 0.1).astype(int)

# Testing different thresolds for optimal business decision-making
for thres in np.arange(0.40, 0.85, 0.05):
    cust_pred = (y_pred_prob_Lgbm >= thres).astype(int)
    print(f"\n--- Threshold: {thres:.2f} ---")
    print(classification_report(y_test, cust_pred))
    cm = confusion_matrix(y_test, cust_pred)
    print("Confusion Matrix:\n", cm)
    print("Rejection %: ", cust_pred[cust_pred == 1].shape[0] / cust_pred.shape[0])
    profit_sim = cm[0, 0] * 5000
    loss_sim = cm[1, 0] * -50000
    total_yield = profit_sim + loss_sim
    print(f"Simulated Portfolio Yield: ${total_yield:,.2f}")

# printing AUC-ROC score for all models
print("ROC_AUC_Score Report:")
print("LightGBM Model: ", roc_auc_score(y_test, y_pred_prob_Lgbm))

# Prediction distributions for KS test
y_test_good = y_pred_prob_Lgbm[y_test == 0]
y_test_bad = y_pred_prob_Lgbm[y_test == 1]

# KS Test for distribution difference between good and bad predictions
stats = ks_2samp(y_test_good, y_test_bad)
result = stats.statistic * 100

#printing KS statistic
print(f"KS Statistic: {result:.2f}%")

# printing classification report for all models
print("Classification Report:")
print("LightGBM Model:\n", classification_report(y_test, custom_prediction))

#printing confusion matrix for all models
print("Confusion Matrix:")
print("LightGBM Model:\n", confusion_matrix(y_test, custom_prediction))

# Feature importance for LightGBM model
feature_importance_ = Lgbm_model.feature_importances_
feature_names = x.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance_})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print("Feature Importance for LightGBM Model:")
print(feature_importance_df.head(15))

print("Freezing the AI and exporting to disk...")

# Save the mathematically optimized LightGBM model
joblib.dump(Lgbm_model, 'production_lgbm_risk_model.joblib')

explainer = shap.TreeExplainer(Lgbm_model)
joblib.dump(explainer, 'production_shap_explainer.joblib')

# Save the exact column order of your 158-feature matrix
# If the future test data has columns in a different order, the AI will crash.
joblib.dump(list(X_train.columns), 'model_training_columns.joblib')

print("SUCCESS: Model architecture exported for production.")