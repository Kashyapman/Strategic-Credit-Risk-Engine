import pandas as pd
import numpy as np
import joblib
import shap
from sqlalchemy import create_engine

engine = create_engine("postgresql://user:password@localhost:5432/database_name")

print("Initializing Strategic Risk Engine...")

# 1. Load the frozen assets
model = joblib.load('production_lgbm_risk_model.joblib')
explainer = joblib.load('production_shap_explainer.joblib')
golden_columns = joblib.load('model_training_columns.joblib')

# 2. Load the batch of customers you want to analyze
# (For your portfolio, you can just load a slice of your testing data)
df_batch = pd.read_sql("SELECT * FROM master_training_data_v4 LIMIT 5000", con=engine)
X_batch_raw = df_batch.drop(columns=['TARGET', 'SK_ID_CURR'])

# 3. Preprocess the raw database text into math
X_batch_math = pd.get_dummies(X_batch_raw)

# 4. Align to the Golden Matrix (Fills missing dummy columns with 0)
X_batch_aligned = X_batch_math.reindex(columns=golden_columns, fill_value=0)
# --------------------------

# 5. Generate the Risk Probabilities (Using the aligned data!)
print("Calculating Risk Probabilities...")
probabilities = model.predict_proba(X_batch_aligned)[:, 1]

# Categoring customer based on the risk threshold
decision = []
for pre in probabilities:
    if pre <= 0.40:
        decision.append("Prime")
    elif pre <= 0.45:
        decision.append("Standard")
    elif pre <= 0.55:
        decision.append("Watchlist")
    else:
        decision.append("Decline")

# 6. Generate the SHAP Reasons
print("Calculating localized risk factors via SHAP...")
shap_values = explainer.shap_values(X_batch_aligned)

if isinstance(shap_values, list):
    shap_values = shap_values[1] 

# Find the top feature pushing the customer toward default
top_risk_indices = np.argmax(shap_values, axis=1)
top_risk_reasons = X_batch_aligned.columns[top_risk_indices]

# Top approval reasons (negative SHAP values)
top_approval_indices = np.argmin(shap_values, axis=1)
top_approval_reasons = X_batch_aligned.columns[top_approval_indices]

# Convert Decision into np array
decision_array = np.array(decision)

# Combine risk and approval reasons into a single explanation
final_reason = np.where(np.isin(decision_array, ["Decline", "Watchlist"]), top_risk_reasons, top_approval_reasons)

# 7. Build and Export the Business Report
business_report = pd.DataFrame({
    'Customer_ID': df_batch['SK_ID_CURR'],
    'Default_Probability': np.round(probabilities, 4),
    'System_Decision': decision_array,
    'Primary_Risk_Driver': final_reason
})

business_report.to_csv("strategic_risk_report.csv", index=False)
print("SUCCESS: Strategic Risk Report generated.")