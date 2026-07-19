import sklearn as sk
import pandas as pd

# FUNCTION FOR ENCODING FEATURES TO USE FOR MODEL TRAINING

def encoding(data):
    encoder_ohe = sk.preprocessing.OneHotEncoder(sparse_output=False)
    categorical_cols = ["precipitation", "extreme_wcond", "perfect_temp", 'weather_improved', 'weekend_and_bad_weather']
    encoded_data = encoder_ohe.fit_transform(data[categorical_cols])
    encoded_columns = encoder_ohe.get_feature_names_out(categorical_cols)
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns, index=data.index)
    data = data.drop(columns=categorical_cols)
    data = pd.concat([data, encoded_df], axis=1)

    encoder_ohe_mult = sk.preprocessing.OneHotEncoder(sparse_output=False, drop='first')
    categorical_cols_date = ["month", "season"]
    encoded_data = encoder_ohe_mult.fit_transform(data[categorical_cols_date])
    encoded_columns = encoder_ohe_mult.get_feature_names_out(categorical_cols_date)
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns, index=data.index)
    data = data.drop(columns=categorical_cols_date)
    data = pd.concat([data, encoded_df], axis=1)

    encoder_rank = sk.preprocessing.OrdinalEncoder()
    data["day_number"] = encoder_rank.fit_transform(data[["day_number"]])
    data["temp_rank"] = encoder_rank.fit_transform(data[["temp_rank"]])
    data["wcond"] = encoder_rank.fit_transform(data[["wcond"]])

    return data