import pandas as pd
import joblib
from icecream import ic
from sklearn.metrics import accuracy_score, hamming_loss, f1_score, jaccard_score


def evaluate_results(y_true, y_pred):
    eval_dict = {
        "Subset accuracy": accuracy_score(y_true, y_pred),
        "Hamming loss": hamming_loss(y_true, y_pred),
        "F1 micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "F1 macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "F1 samples": f1_score(y_true, y_pred, average="samples", zero_division=0),
        "Jaccard": jaccard_score(y_true, y_pred, average="samples"),
    }
    return eval_dict


if __name__ == "__main__":
    import joblib

    mlb_dict = joblib.load("outputs/mlb_dict_test.pkl")
    pipeline = joblib.load("outputs/pipeline_test.pkl")

    eval_data_path = "data/eval_data/eval_data_mini.jsonl"
    eval_data_df = pd.read_json(eval_data_path, lines=True)
    # ic(eval_data_df)
    x_eval = eval_data_df[
        [
            c
            for c in eval_data_df
            if c not in ["codes", "accounts", "sub_accounts", "sub_details"]
        ]
    ]
    # ic(x_eval)
    level = "codes"
    y_eval = eval_data_df[level]
    y_eval_transformed = mlb_dict[level].transform(
        y_eval.apply(set)
    )  # must take into account the 4 happening 2 times, else we get an error
    y_pred_raw = pipeline.predict(x_eval)
    ic(y_eval.apply(set), y_eval_transformed, y_pred_raw)
    y_pred = mlb_dict[level].inverse_transform(y_pred_raw)
    ic(y_eval, y_pred)
    eval_dict = evaluate_results(y_eval_transformed, y_pred_raw)
    ic(eval_dict)
