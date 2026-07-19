import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt

# FUNCTION FOR RANDOM FOREST CROSS-VALIDATION

def forest(rand_state, X, y, n_trees_array, max_depth_array, max_predictors_array):
    encoder = sk.preprocessing.OrdinalEncoder()
    X["wcond"] = encoder.fit_transform(X[["wcond"]])

    kf = sk.model_selection.TimeSeriesSplit(n_splits=5, gap=0)

    print("Random forest cross-validation. Please wait")
    data_for_table = []
    for n_trees in n_trees_array:
        for max_depth in max_depth_array:
            for max_predictors in max_predictors_array:

                model = sk.ensemble.RandomForestRegressor(n_estimators=n_trees,
                                                          max_depth=max_depth,
                                                          max_features=max_predictors,
                                                          random_state=rand_state)

                terrors_for_params = []
                verrors_for_params = []
                for train_index, test_index in kf.split(X):
                    X_train, y_train = X.iloc[train_index, :], y.iloc[train_index]
                    X_test, y_test = X.iloc[test_index, :], y.iloc[test_index]
                    model.fit(X_train, y_train)

                    prediction = model.predict(X_train)
                    terrors_for_params.append(sk.metrics.mean_squared_error(y_train.astype(float), prediction))
                    prediction = model.predict(X_test)
                    verrors_for_params.append(sk.metrics.mean_squared_error(y_test.astype(float), prediction))
                train_mse = sum(terrors_for_params) / len(terrors_for_params)
                validation_mse = sum(verrors_for_params) / len(verrors_for_params)
                row = {
                    "Number of trees": n_trees,
                    "Maximal depth": max_depth,
                    "Maximal features": max_predictors,
                    "Train MSE": train_mse,
                    "Validation MSE": validation_mse
                }
                data_for_table.append(row)

    print("Random forest cross-validation ended")
    errors = pd.DataFrame(data_for_table, columns=["Number of trees", "Maximal depth", "Maximal features",
                                                   "Train MSE", "Validation MSE"])
    return errors

# FUNCTION FOR VISUALISE PREDICTION ERRORS CORRESPONDING TO DIFFERENT FOREST PARAMETERS COMBINATIONS

def visualise_errors_forest(rand_state, X, y, n_trees_array, max_depth_array, max_predictors_array, text):
    print("\n=== RANDOM FOREST EVALUATION ===")

    results = forest(rand_state, X, y, n_trees_array, max_depth_array, max_predictors_array)

    best_idx = results['Validation MSE'].idxmin()
    best_model = results.loc[best_idx]

    plt.figure(figsize=(10, 6))

    plt.scatter(
        results["Train MSE"],
        results["Validation MSE"],
        color = 'blue',
        alpha = 0.6,
        s = 50,
        label = 'Parameters combination'
    )

    plt.scatter(best_model['Train MSE'], best_model['Validation MSE'], color='red', s=120)
    annotation_text = f"Best model:\nDepth: {best_model['Maximal depth']}\nFeatures: {best_model['Maximal features']}" \
                      f"\nTrees: {best_model['Number of trees']}"

    print(annotation_text)

    plt.annotate(
        annotation_text,
        xy=(best_model['Train MSE'], best_model['Validation MSE']),
        xytext=(15, -25),
        textcoords='offset points',
        color='red',
        fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8)
    )

    plt.title(f'Random forest errors({text})', fontsize=14)
    plt.xlabel('Train MSE', fontsize=12)
    plt.ylabel('Validation MSE', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.savefig(f'errors_forest_{text.replace(" ", "_")}.pdf', format='png', bbox_inches='tight')

    plt.show()