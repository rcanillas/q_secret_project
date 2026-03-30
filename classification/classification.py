import pandas as pd
from icecream import ic
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


numeric_features = ["amount_ht", "vat_rate", "vat", "amount_ttc", "quantity"]
categorical_features = [
    "supplier",
    "item",
    "category",
    "label",
    "client",
    "legal_form",
    "tax_regime",
    # "merchant", dict is messing with pipeline
]
targets = ["codes", "accounts", "sub_accounts", "sub_details"]


def prepare_data(data_df, numeric_features, categorical_features, targets):
    X_train = data_df[numeric_features + categorical_features]
    ic(X_train)
    y_train_dict = {}
    mlb_dict = {}
    for target in targets:
        # TODO: better preparation for lower hierachy (separate by code, etc)
        mlb_target = MultiLabelBinarizer()
        ic(data_df[target])
        y_train_dict[target] = mlb_target.fit_transform(data_df[target])
        mlb_dict[target] = mlb_target
    return X_train, y_train_dict, mlb_dict


def get_processing_pipeline(numeric_features, categorical_features):
    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
    return preprocessor


def get_model():
    model = MultiOutputClassifier(RandomForestClassifier(n_estimators=200, n_jobs=-1))
    return model


def get_pipeline(numeric_features, categorical_features):
    preprocessor = get_processing_pipeline(numeric_features, categorical_features)
    model = get_model()
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", model)])
    return pipeline


def train_pipeline(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)
    return pipeline


if __name__ == "__main__":
    import joblib

    historical_data = pd.read_json("outputs/aggregated_data.jsonl", lines=True)
    X_train, y_train_dict, mlb_dict = prepare_data(
        historical_data, numeric_features, categorical_features, targets
    )
    pipeline = get_pipeline(numeric_features, categorical_features)
    level = "codes"
    trained_pipeline = train_pipeline(pipeline, X_train, y_train_dict[level])
    y_pred = trained_pipeline.predict(X_train)
    labels = mlb_dict[level].inverse_transform(y_pred)
    joblib.dump(trained_pipeline, "outputs/pipeline.pkl")
    joblib.dump(mlb_dict, "outputs/mlb_dict.pkl")
    y_probas = trained_pipeline.predict_proba(X_train)
    ic(y_probas)
