import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt

# FUNCTION FOR RANDOM FOREST PREDICTING RESIDUALS CROSS-VALIDATION

def ensemble(rand_state, X, y, n_trees_array, max_depth_array, min_samples_array):
    ridge = sk.pipeline.make_pipeline(sk.preprocessing.StandardScaler(), sk.linear_model.Ridge(alpha=0.018))
    kf = sk.model_selection.TimeSeriesSplit()

    data_for_table = []
    print("Ensemble cross-validation. Please wait")
    for n_trees in n_trees_array:
        for max_depth in max_depth_array:
            for min_samples in min_samples_array:
                terrors_for_params = []
                verrors_for_params = []
                for train_index, test_index in kf.split(X):
                    X_train, y_train = X.iloc[train_index, :].copy(), y.iloc[train_index].copy()
                    X_test, y_test = X.iloc[test_index, :].copy(), y.iloc[test_index].copy()
                    ridge.fit(X_train, y_train)
                    prediction_ridge_train = ridge.predict(X_train)
                    y_train_res = y_train - prediction_ridge_train
                    prediction_ridge_test = ridge.predict(X_test)
                    y_test_res = y_test - prediction_ridge_test

                    forest = sk.ensemble.RandomForestRegressor(
                        n_estimators=n_trees,
                        min_samples_leaf=min_samples,
                        max_depth=max_depth,
                        random_state=rand_state
                    )
                    X_train = X_train.drop(columns=["day_number"])
                    X_test = X_test.drop(columns=["day_number"])
                    forest.fit(X_train, y_train_res)
                    prediction_train = forest.predict(X_train) + prediction_ridge_train
                    terrors_for_params.append(sk.metrics.mean_squared_error(y_train.astype(float), prediction_train))
                    prediction_test = forest.predict(X_test) + prediction_ridge_test
                    verrors_for_params.append(sk.metrics.mean_squared_error(y_test.astype(float), prediction_test))
                train_mse = sum(terrors_for_params) / len(terrors_for_params)
                validation_mse = sum(verrors_for_params) / len(verrors_for_params)
                row = {
                    "Number of trees": n_trees,
                    "Maximal depth": max_depth,
                    "Minimum samples in leaf": min_samples,
                    "Train MSE": train_mse,
                    "Validation MSE": validation_mse
                }
                data_for_table.append(row)

    print("Ensemble cross-validation ended")
    errors = pd.DataFrame(data_for_table, columns=["Number of trees", "Maximal depth", "Minimum samples in leaf",
                                                   "Train MSE", "Validation MSE"])

    return errors

# FUNCTION FOR VISUALISE PREDICTION ERRORS CORRESPONDING TO DIFFERENT FOREST PARAMETERS COMBINATIONS

def visualise_ensemble_errors(rand_state, X, y, n_trees_array, max_depth_array, min_samples_array, text):
    print("\n=== ENSEMBLE EVALUATION ===")
    errors = ensemble(rand_state, X, y, n_trees_array, max_depth_array, min_samples_array)

    best_idx = errors['Validation MSE'].idxmin()
    best_model = errors.loc[best_idx]

    plt.figure(figsize=(10, 6))

    plt.scatter(
        errors["Train MSE"],
        errors["Validation MSE"],
        color='blue',
        alpha=0.6,
        s=50,
        label='Parameters combination'
    )

    plt.scatter(best_model['Train MSE'], best_model['Validation MSE'], color='red', s=120)
    annotation_text = f"Best model:\nDepth: {best_model['Maximal depth']}\nTrees: {best_model['Number of trees']}\n" \
                      f"Samples in leaf: {best_model['Minimum samples in leaf']}"

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

    plt.title(f'Forest predicting residuals errors({text})', fontsize=14)
    plt.xlabel('Train MSE', fontsize=12)
    plt.ylabel('Validation MSE', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.savefig(f'errors_ensemble_{text.replace(" ", "_")}.pdf', format='png', bbox_inches='tight')

    plt.show()