import numpy as np
import pandas as pd

# FUNCTION FOR CONDEMNING PCA FOR NUMERIC FEATURES

def pca(matrix, names):
    print("=== PCA RESULTS ===")
    U, S, Vh = np.linalg.svd(matrix)

    explain_percentage = np.cumsum(S**2)/np.sum(S**2)
    print("Percentage of data variance explained with adding new principal components:")
    for per in explain_percentage:
        print(f"{per:.3f}")

    data_for_table = Vh[0:3].transpose()
    table = pd.DataFrame(data_for_table, columns=["1st component", "2nd component", "3rd component"])
    table['Variable'] = names
    print("\nFirst 3 components:")
    print(table)

    components = matrix @ Vh.transpose()

    return components
