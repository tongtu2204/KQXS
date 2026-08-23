from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


data_path = "data/kqxsmb_2002_2026.csv"

df = pd.read_csv(
    data_path,
    dtype={
        "full_result": str,
        "last_2_digits": str,
    },
    parse_dates=["date"],
)


print(df.head())