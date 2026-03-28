import pandas as pd
from icecream import ic
import json

df = pd.read_json("test.jsonl", lines=True)
ic(df)

df_2 = pd.read_json("data/users/mini/bank_transactions.jsonl", lines=True)
ic(df_2)
