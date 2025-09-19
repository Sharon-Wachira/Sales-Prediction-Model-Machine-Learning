# Sales-Prediction-Model-Machine-Learning

The project addresses how to predict sales performance to drive better decisions and boost profitability.
The Superstore dataset is used to develop ML models for predicting sales and providing actionable insights. 
A retail sales forecasting and strategy project that uses EDA, feature engineering, and tree-based machine learning (Random Forest, XGBoost, ensemble) on the Superstore dataset to predict sales, identify top sales drivers, and recommend optimal discount ranges for profit maximization. 

 #### What I worked on
Built an end-to-end ML pipeline: data cleaning, EDA, outlier handling, feature engineering, categorical encoding, model training, hyperparameter tuning, cross-validation and interpretation. 

Compared Random Forest, XGBoost and an ensemble (average of RF + XGB) to find the most reliable predictor for log-sales. 

### Data

Dataset: Superstore transactions (Kaggle sample). ~9,994 rows, 21 columns (order, product, region, sales, profit, discount, etc.). 

#### Methods 
Preprocessing: dropped uninformative columns, standardized column names, converted dates, handled missing values and duplicates, log-transformed skewed targets (sales, profit) and removed/treated outliers (IQR + log transforms). 

Feature engineering: log_sales, log_profit, profit_margin, discount_impact, discount_efficiency, plus encoded region/category/segment variables. 

Models: Random Forest (baseline), XGBoost (tuned), and an ensemble (averaged predictions). Evaluation via R², RMSE, MAE and k-fold cross-validation. 

#### Key results

Random Forest: R² ≈ 0.8039, RMSE ≈ 0.5830, MAE ≈ 0.3097 (metrics on the log-sales target). Feature importance showed profit margin and discount efficiency as top predictors for RF. 

XGBoost: R² ≈ 0.8206, RMSE ≈ 0.5577, MAE ≈ 0.3137 — the highest single-model R² in the analysis. XGBoost ranked Office Supplies as the single most important categorical feature in its importance scores. 

Ensemble (RF + XGB avg): produced the strongest combined performance (reported ensemble R² ≈ 0.82, RMSE ≈ 0.57), suggesting the two models capture complementary patterns. 

#### Discount / pricing insight

Discount analysis (grouped into ranges 0–10%, 11–20%, 21–30%, etc.) found 11–20% to be the most profitable discount band — producing the largest total profit (reported $53,380.49) with moderate average sales (~$96.45). This indicates moderate discounts maximize the trade-off between volume and margin for this dataset. 

Code used to define and evaluate discount ranges is in the report (binning by [0, 0.1, 0.2, 0.3, 0.4, 1] and aggregating sales & profit by bin). 

### Business recommendations
Prioritise inventory & marketing for Office Supplies. This category was consistently the top driver of predicted sales. 

Adopt moderate discounting (≈11–20%) where possible to improve total profitability rather than blanket deep discounts. 

Monitor profit margin and discount efficiency as core KPIs. These were top predictive features and help explain which promotions preserve profit. 

#### Limitations

Analysis used the public Superstore sample (no external competitor or market data). Results are data-specific and should be validated with company sales pipelines before operational rollout. 
City-level high cardinality was simplified/aggregated to avoid overfitting; localized strategies may require more granular local data.
