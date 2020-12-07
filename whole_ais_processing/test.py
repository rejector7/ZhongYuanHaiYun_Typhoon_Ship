import pandas as pd

processed_file = "G:/Reins/datasets/ais/processed_ais.csv"

dset_reader = pd.read_csv(processed_file, chunksize=100000)
length = 0
for chunk in dset_reader:
    print(chunk.columns)
    length += len(chunk.index)
    break

print(length)