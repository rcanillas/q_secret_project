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
    # TODO: special case when transaction is composed of several invoice lines -
    # ! an invoice can be paid in several installments, so we have 1-1, 1-N, N-1, N-N...
    # Probably easiest is to create a "order" to create link between transactions and invoices
    # for now we postulate 1-1
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
