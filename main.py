import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import holidays
import seaborn as sns
import sklearn as sk


from pca import pca
from ridge import visualise_errors_ridge
from forest import visualise_errors_forest
from encoding import encoding
from ensemble import visualise_ensemble_errors
from preprocessing import preprocessing

# PREPROCESSING

train_data = pd.read_csv("bike_train.csv", index_col=0)
df = pd.DataFrame(train_data)
pl_holidays = holidays.PL()

df['date'] = pd.to_datetime(df['date'])
df['weekend'] = (df['date'].dt.dayofweek > 4) | (df['date'].isin(pl_holidays))
df['weekend'] = df['weekend'].astype('int')
cat_weather_type = pd.CategoricalDtype(categories=sorted(df['wcond'].dropna().unique()), ordered=True)
df['wcond'] = df['wcond'].astype(cat_weather_type)
df['count'] = df['count'].astype('float')

df = df.replace(r'^\s*$', np.nan, regex=True)
empty_rows_count = df.isna().any(axis=1).sum()
print(f"Rows with empty values: {empty_rows_count}\n")

# TEMPERATURE AND BIKES RENTED DEPENDENCE ON DATE

days = df["date"]
bikes_count = df["count"]
temperature = df["temp"]
fig, ax1 = plt.subplots(figsize = (14, 6))

ax1.set_xlabel("Day")
color1 = 'tab:blue'
ax1.set_ylabel("Number of rented bikes", color = color1)
ax1.plot(days,
         bikes_count,
         color = color1,
         linewidth = 2)
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()

color2 = 'tab:red'
ax2.set_ylabel("Temperature", color = color2)
ax2.plot(days,
         temperature,
         color = color2,
         linewidth = 1,
         alpha = 0.6)
ax2.tick_params(axis='y', labelcolor=color2)

fig.tight_layout()
fig.autofmt_xdate()
plt.title("Rented bikes and temperature by days")
plt.show()

# CORRELATIONS HEATMAP

numeric_df = df.select_dtypes(include = ['number'])
numeric_df = numeric_df.drop('weekend', axis=1)

corr_matrix = numeric_df.corr()

plt.figure(figsize=(10,8))

sns.heatmap(
    corr_matrix,
    annot = True,
    cmap='coolwarm',
    vmin=-1, vmax=1,
    fmt = ".2f",
    linewidths=0.5
)

plt.title("Correlations heatmap")
plt.show()

# PCA FOR NUMERIC FEATURES

weather_df = numeric_df.iloc[:, 0:4]
m, n = weather_df.shape
names = weather_df.columns
for i in range(n):
    column = weather_df.iloc[:, i]
    weather_df.iloc[:, i] = (column - np.mean(column))/np.std(column)
matrix = weather_df.to_numpy()
components = pca(matrix, names)

# MODELS CROSS-VALIDATION BEFORE DATA TRANSFORMING

rand_state = 42
X = df.drop(columns = ["count", "date"])
y = df['count']

param_array = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50]
errors_ridge = visualise_errors_ridge(rand_state, X, y, param_array, "input data")

n_trees_array = [100, 200, 300, 400, 500]
max_depth_array = [1, 3, 5, 7, 9]
max_predictors_array = [2, 3, 4, 5]
visualise_errors_forest(rand_state, X, y, n_trees_array, max_depth_array, max_predictors_array, "input data")

# ADDING FEATURES

df["precipitation"] = df["wcond"] == 3
conditions = [
    (df['atemp'] < 10),
    (df['atemp'] >= 10) & (df['atemp'] <= 25),
    (df['atemp'] > 25)
]
choices = [1, 2, 3]
df['temp_rank'] = np.select(conditions, choices, default=0)

for i in range(1, 4):
    df[f'count_lag_{i}'] = df['count'].shift(i)
df['wcond_lag1'] = df['wcond'].shift(1)
df['atemp_lag1'] = df['temp'].shift(1)
df = df.dropna()
df['extreme_wcond'] = (df['wind'] > df['wind'].quantile(0.9)) | (df['hum'] > df['hum'].quantile(0.9)) | df['precipitation']
df['perfect_temp'] = (df['atemp'] >= 18) & (df['atemp'] <= 25)
df['temp_diff'] = df['atemp'] - df['atemp_lag1']
wcond_codes = df['wcond'].astype('float')
wcond_lag1_codes = df['wcond_lag1'].astype('float')
df['weather_improved'] = (wcond_codes - wcond_lag1_codes) < 0
df['weekend_and_bad_weather'] = df['weekend'].astype('int') * df['precipitation'].astype('int')

month = df['date'].dt.month
conditions = [
    (month == 12) | (month <= 2),
    (month >= 3) & (month <= 5),
    (month >= 6) & (month <= 8),
    (month >= 9) & (month <= 11)
]
choices = [1, 2, 3, 4]
df['month'] = pd.CategoricalDtype(categories=sorted(month.dropna().unique()), ordered=True)
df['season'] = np.select(conditions, choices, default=np.nan)

indices = df.index + 1
df["day_number"] = pd.CategoricalDtype(categories=sorted(indices.dropna().unique()), ordered=True)

train_data = encoding(df)

# MODELS CROSS-VALIDATION AFTER DATA TRANSFORMING

X_train = train_data.drop(columns = ["count", "date"])
y_train = train_data['count']

param_array = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50]
errors_ridge = visualise_errors_ridge(rand_state, X_train, y_train, param_array, "with added features")

n_trees_array = [100, 200, 300, 400, 500]
max_depth_array = [1, 3, 5, 7, 9]
max_predictors_array = [2, 3, 4, 5]
visualise_errors_forest(rand_state, X_train, y_train, n_trees_array, max_depth_array, max_predictors_array,
                        "with added features")

# CHOOSING PARAMETERS VALUES FOR RIDGE-FOREST ENSEMBLE

n_trees_array = [100, 200, 300, 500]
max_depth_array = [6, 7, 8, 9]
min_samples_array = [20, 30, 40]
visualise_ensemble_errors(5, X_train, y_train, n_trees_array, max_depth_array, min_samples_array, "with added features")

# BUILDING MODEL FOR PREDICTION

ridge = sk.pipeline.make_pipeline(sk.preprocessing.StandardScaler(), sk.linear_model.Ridge(alpha=0.018))
ridge.fit(X_train, y_train)
prediction_train_ridge = ridge.predict(X_train)
y_train_res = y_train - prediction_train_ridge
forest = sk.ensemble.RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=20,
    max_depth=7,
    random_state=5
)
forest.fit(X_train, y_train_res)

# TESTING MODEL

X_test = pd.read_csv("bike_test.csv")
test_data = preprocessing(X_test)
y_test = test_data["count"].to_numpy()
X_test = test_data.drop(columns = ["date", "count"])
print("\n=== PREDICTION ===")
prediction_test_ridge = ridge.predict(X_test)
prediction_test_forest = forest.predict(X_test)
results = prediction_test_ridge + prediction_test_forest
results_table = pd.DataFrame({'Predicted_Bikes_Count' : results})
results_table['Predicted_Bikes_Count'] = results_table['Predicted_Bikes_Count'].round().astype(int)
error = sk.metrics.mean_squared_error(y_test.astype(float), results)
file_name = 'bike_predictions_results.csv'
results_table.to_csv(file_name, index=True)
print("Prediction was successfully made")
print(f"Prediction error: {error:.2f}")