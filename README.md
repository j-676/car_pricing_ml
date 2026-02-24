# 🚘 Car Pricing ML 

**Estimating Car Prices for a Local Dealership**  
*Machine Learning pipeline to predict used car market value from features like age, engine, mileage, and condition.*

[![RandomForest Feature Importance](feature_importance.png)]
*Top drivers: year (~30%), car_age (~25%), engine_volume (~23%)*

##  Business Problem
Local dealership faces inconsistent pricing → **lost revenue** (undervalued) or **inventory costs** (overpriced).  
**Solution**: ML model predicts fair market price based on car characteristics.

##  Dataset
- **Source**: `cars-cars.csv` (used car listings) [file:1]
- **Features**: make, model, year, mileage_km, engine_volume_cm3, fuel_type, transmission, drive_unit, condition, segment
- **Target**: `priceUSD`
- **Size**: ~100K+ rows (full dataset)

##  ML Pipeline
Complete end-to-end workflow in [Google Colab](Car_Pricing_ML.ipynb):

1. **EDA**: Distributions, correlations, categorical analysis
2. **Preprocessing**: Imputation, scaling, one-hot encoding, feature engineering (car_age, mileage_per_year)
3. **Models Compared**:
   | Model | RMSE ↓ | R² ↑ | MAE ↓ |
   |-------|--------|------|-------|
   | **RandomForest** | **2,627** | **0.89** | **1,055** |
   | XGBoost | 2,739 | 0.89 | 1,277 |
   | GradientBoosting | 3,031 | 0.86 | 1,488 |
   | LinearRegression | 4,417 | 0.70 | 2,088 |

4. **Best Model**: RandomForestRegressor (R²=89%, RMSE=~15% of median price)

## Key Insights

**Top Price Drivers** (from feature importance):

- **year**: ~30% → Newer registration year significantly increases price.
- **car_age**: ~25% → Older cars lose value quickly; age is almost as important as year itself.
- **engine_volume_cm3**: ~23% → Larger engines tend to command a strong price premium.
- **mileage_km**: smaller but meaningful impact → Higher mileage reduces price, reflecting wear and usage.
- Automatic transmission and premium segments (luxury brands, upper segments) still add a noticeable price uplift, even if their individual importances are smaller.


**Pricing Error**: Typical ±1,055 USD → actionable for dealership decisions.

##  Usage
```python
# Predict price for new inventory
new_car = {"make": "bmw", "model": "3", "year": 2018, ...}
predicted_price = best_model.predict([new_car])

```
##  Tech Stack

| Category | Tools |
|----------|-------|
| **ML** | scikit-learn, XGBoost, RandomForestRegressor |
| **Visualization** | matplotlib, seaborn |
| **Preprocessing** | pandas, StandardScaler, OneHotEncoder |
| **Environment** | **Google Colab)** |
| **Deployment** | Ready for Streamlit/Flask/FastAPI |




##  Recommendations

- **Deploy** as pricing assistant tool (web app, Excel integration, CRM plugin)
- **Retrain** quarterly to capture market shifts (fuel prices, EV adoption, economic conditions)
- **Add features**: 
  - Accident history & service records
  - Regional pricing differences (location)
  - Number of owners, days on market
  - Seasonal demand patterns
- **Monitor** model error drift as market conditions change (EVs, fuel prices, interest rates)
- **Human-in-loop**: Use predictions as strong baseline + sales team adjustment


##  License

MIT License** - Free to use, modify, and deploy for commercial purposes.


---





