import pandas as pd
import numpy as np
import codecs


# processed_file = "G:/Reins/datasets/ais/processed_ais.csv"
# standard_ais_2019 = "G:/Reins/datasets/AIS_2019_01_01/AIS_2019_01_01.csv"
#
#
# df = pd.read_csv(processed_file)
# print(df.head())
# print(df.dtypes)
# dset_reader = pd.read_csv(processed_file, chunksize=100000, encoding='utf-8')
# length = 0
# for i, chunk in enumerate(dset_reader):
#     # print(chunk.columns)
#     print(i)
#     length += len(chunk.index)
#
#
# print(length)

# np.random.randint(0, len(ais_dset), 10)

# [1, 2] + [3, 4]

# a = {'abc': 123, 'fdsa': 34234}
# np.save("test.npy", a)

a = np.asarray([1, 2, 3, 4])
print(np.concatenate(([a[0]], a[2:])))
print(a[[0, 2, 3]])

