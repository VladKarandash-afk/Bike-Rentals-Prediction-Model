# Bike Rental Demand Prediction

This project focuses on forecasting daily bike rental demand using historical data, weather conditions, and engineered calendar features.

## Project Results & Evaluation
The core achievement of this project is a robust, highly-optimized prediction pipeline evaluated using **Mean Squared Error (MSE)** through strict Time-Series Cross-Validation (`TimeSeriesSplit`). 

**Key Results & Outputs:**
- **Final Predictions:** The final test set predictions are successfully compiled and exported to `bike_predictions_results.csv`.
- **Hybrid Ensemble Approach:** The best-performing model is a custom ensemble. A baseline **Ridge Regression** model captures the overarching linear trends, while a **Random Forest Regressor** is sequentially trained on the residuals (errors) of the Ridge model to capture complex, non-linear relationships.
- **Automated Performance Visualizations:** The cross-validation pipeline generates and saves detailed evaluation scatter plots as PDFs (e.g., `errors_ridge_input_data.pdf`, `errors_ensemble_with_added_features.pdf`). These plots explicitly annotate the best hyperparameter combinations comparing Train vs. Validation MSE.
- **Data Insights:** The script generates Exploratory Data Analysis (EDA) charts, including a "Rented bikes and temperature by days" dual-axis timeline and a feature correlation heatmap.

## Advanced Feature Engineering
To achieve high prediction accuracy, the data undergoes significant transformations:
- **Temporal Lags:** Incorporates historical behavior using features like `count_lag`, `wcond_lag1`, and `atemp_lag1`.
- **Calendar & Holiday Features:** Distinguishes weekends, seasons, and dynamically identifies **Polish public holidays** using the `holidays` Python library.
- **Custom Weather Indicators:** Creates logical features for extreme weather conditions, ideal cycling temperatures, and day-to-day weather improvements.

## Repository Structure
- `main.py` — The primary pipeline: handles data loading, EDA, feature engineering execution, model training, and final CSV result generation.
- `preprocessing.py` — Encapsulates custom feature creation, missing value imputation, and dataset transformations.
- `ensemble.py` — Implements the Ridge + Random Forest (residuals) hybrid model evaluation.
- `forest.py` & `boosting.py` & `ridge.py` — Cross-validation and visualization modules for individual algorithms (Random Forest, HistGradientBoosting, Ridge Regression).
- `encoding.py` / `pca.py` — Utilities for categorical encoding and Principal Component Analysis on numeric weather data.

## Installation & Usage
1. Ensure Python 3.8+ is installed along with the required dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn holidays
   ```
2. Place your training (`bike_train.csv`) and testing (`bike_test.csv`) datasets in the root directory.
3. Run the main pipeline:
   ```bash
   python main.py
   ```
4. Review the generated `bike_predictions_results.csv` and the output `.pdf` evaluation plots in your project folder.
