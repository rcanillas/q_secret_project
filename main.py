import pandas as pd
from sklearn.model_selection import train_test_split
from retrieval import aggregate_data
from matching import get_matching_lines
from icecream import ic
import json


def main():
    # Load data
    full_dataset_df = pd.DataFrame()
    for user_folder in ["coach_agile", "it_consulting"]:
        folder = f"data/users/{user_folder}"
        bk_path = f"{folder}/bank_transactions.jsonl"
        bank_transactions = pd.read_json(bk_path, lines=True)
        il_path = f"{folder}/invoice_lines.jsonl"
        invoice_lines = pd.read_json(il_path, lines=True)
        with open(f"{folder}/user_context.json") as uc_file:
            user_context = json.load(uc_file)
        ac_path = f"{folder}/accounting_codes.jsonl"
        accounting_codes = pd.read_json(ac_path, lines=True)
        user_data_df = aggregate_data(
            bank_transactions, invoice_lines, user_context, accounting_codes
        )
        full_dataset_df = pd.concat([full_dataset_df, user_data_df], ignore_index=True)
    full_dataset_df = full_dataset_df.sample(frac=1, random_state=42)
    histo_dataset_df = full_dataset_df.iloc[:150]
    histo_dataset_path = "outputs/aggregated_data.jsonl"
    histo_dataset_df.to_json(histo_dataset_path, orient="records", lines=True)
    test_dataset_df = full_dataset_df.iloc[150:]
    test_dataset_path = "data/input_data/input_data.jsonl"
    test_dataset_df.to_json(test_dataset_path, orient="records", lines=True)

    # ic(histo_dataset_df, test_dataset_df)
    matching_lines_df = get_matching_lines(histo_dataset_df, test_dataset_df)
    # ic(matching_lines_df)
    ic(len(test_dataset_df))
    ic(len(matching_lines_df))
    matching_lines_df.to_json(
        "outputs/matching_lines_df.jsonl", orient="records", lines=True
    )


if __name__ == "__main__":
    main()
