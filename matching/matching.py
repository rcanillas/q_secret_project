import pandas as pd
from icecream import ic


def get_matching_lines(historical_data, new_data):
    matched_list = []
    for hl_idx, new_data_line in new_data.iterrows():
        # potential bottleneck here
        # ic(historical_line)
        matched_data = historical_data.loc[
            (
                historical_data["supplier"].str.lower()
                == new_data_line["supplier"].lower()
            )
            & (historical_data["amount_ht"] == new_data_line["amount_ht"])
            & (historical_data["item"].str.lower() == new_data_line["item"].lower())
        ]
        if not matched_data.empty:
            matched_data = matched_data.iloc[0]
            # ic(matched_data)
            matched_list.append(matched_data)
    matching_lines_df = pd.DataFrame(matched_list)
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
        "outputs/matching_lines_df_test.jsonl", orient="records", lines=True
    )
