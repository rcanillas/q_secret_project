import pandas as pd
from icecream import ic
import json


def aggregate_data(bank_transactions, invoice_lines, user_context):
    # Minimal working example
    aggregated_df = invoice_lines.copy()
    aggregated_df["client"] = user_context["user_context"]["company_name"]
    aggregated_df["legal_form"] = user_context["user_context"]["legal_form"]
    aggregated_df["tax_regime"] = user_context["user_context"]["tax_regime"][
        "vat_regime"
    ]
    # TODO: special case when transaction is composed of several different lines
    return aggregated_df


if __name__ == "__main__":
    folder = "data/users/mini"
    bk_path = f"{folder}/bank_transactions.jsonl"
    bank_transactions = pd.read_json(bk_path, lines=True)
    il_path = f"{folder}/invoice_lines.jsonl"
    invoice_lines = pd.read_json(il_path, lines=True)
    with open(f"{folder}/user_context.json") as uc_file:
        user_context = json.load(uc_file)
    aggregated_df = aggregate_data(bank_transactions, invoice_lines, user_context)
    aggregated_df.to_csv("outputs/aggregated_data.csv")
