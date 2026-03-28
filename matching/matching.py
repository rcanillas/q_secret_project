import pandas as pd
from icecream import ic


def get_matching_lines(historical_data, new_data):
    matching_lines_df = pd.DataFrame()
    for hl_idx, historical_line in historical_data.iterrows():
        # potential bottleneck here
        ic(historical_line)
        matched_data = new_data.loc[
            (new_data["supplier"].str.lower() == historical_line["supplier"].lower())
            & (new_data["amount_ht"] == historical_line["amount_ht"])
            & (new_data["item"].str.lower() == historical_line["item"].lower())
        ]
        if not matched_data.empty:
            matched_data["codes"] = [historical_line["codes"]] * len(matched_data)
            matched_data["accounts"] = [historical_line["accounts"]] * len(matched_data)
            matched_data["sub_accounts"] = [historical_line["sub_accounts"]] * len(
                matched_data
            )
            matched_data["sub_details"] = [historical_line["sub_details"]] * len(
                matched_data
            )
            # ic(matched_data)
            matching_lines_df = pd.concat([matching_lines_df, matched_data])
    return matching_lines_df


if __name__ == "__main__":
    historical_data = pd.read_json("outputs/aggregated_data.jsonl", lines=True)
    # New data must follow the structure of aggregated data. Need to create the same aggregation func.
    new_data = pd.read_json("data/input_data/input_data_mini.jsonl", lines=True)
    matching_lines_df = get_matching_lines(historical_data, new_data)
    # ic(historical_data)
    # ic(new_data)
    ic(matching_lines_df)
    matching_lines_df.to_json(
        "outputs/matching_lines_df.jsonl", orient="records", lines=True
    )
