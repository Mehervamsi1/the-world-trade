import pandas as pd
import numpy as np



df = pd.read_csv('world_trade_synth_fast.csv')
df.to_csv("world_trade_synth_fast.csv.gz", index=False, compression="gzip")