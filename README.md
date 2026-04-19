\# Strategic Credit Risk Engine: Profit-Optimized AI



\### \*\*Executive Summary\*\*

This repository contains an end-to-end Machine Learning architecture designed to predict retail credit defaults with high precision. Moving beyond standard classification, this pipeline integrates a \*\*Profit/Loss (P\&L) Financial Simulator\*\* to dynamically optimize risk thresholds. By balancing interest gained against principal lost, the optimized LightGBM engine identified a peak portfolio yield at a 0.55 probability cutoff, effectively shielding an estimated $127.4M in portfolio value.



\### \*\*System Architecture\*\*

The pipeline is decoupled into two production environments:

1\. \*\*Model Training \& Financial Simulation (`src/model\_v4.py`):\*\* \* Connects to a PostgreSQL database to ingest a 158-feature matrix engineered from historical ledgers.

&#x20;  \* Utilizes `RandomizedSearchCV` to optimize LightGBM hyperparameters, achieving a \*\*KS Statistic of 41.83%\*\* and an \*\*ROC-AUC of 0.774\*\*.

&#x20;  \* Integrates a custom Class Imbalance penalty (`scale\_pos\_weight`) to force the algorithm to prioritize severe delinquency capture.

&#x20;  \* Runs a dynamic loop to simulate portfolio yield across various thresholds, establishing Tiered Risk Bands (Prime, Standard, Watchlist, Decline).



2\. \*\*Strategic Explainability Engine (`src/risk\_explainability.py`):\*\* \* Black-box AI is unacceptable in regulatory banking. This script loads the frozen `.joblib` model and deploys SHAP (Game Theory) to evaluate new loan batches.

&#x20;  \* Automatically generates a boardroom-ready CSV report detailing the System Decision, Default Probability, and the single largest localized "Risk Driver" for every individual applicant.



\### \*\*Risk Band Strategy\*\*

Based on the P\&L simulation, the model routes applicants into the following categories:

\* \*\*Prime (< 0.40):\*\* Auto-approval with premium limits.

\* \*\*Standard (0.40 - 0.45):\*\* Core revenue engine with standard pricing.

\* \*\*Watchlist (0.46 - 0.55):\*\* Danger zone requiring manual underwriting or Risk-Based Pricing.

\* \*\*Decline (> 0.55):\*\* Auto-Reject. Simulation proves taking on risk past this exact threshold results in immediate portfolio decay.



\### \*\*How to Run\*\*

1\. Clone the repository: `git clone https://github.com/yourusername/Strategic-Credit-Risk-Engine.git`

2\. Install dependencies: `pip install -r requirements.txt`

3\. Generate the models: `python src/model\_v4.py`

4\. Run the SHAP explainability engine: `python src/risk\_explainability.py`

