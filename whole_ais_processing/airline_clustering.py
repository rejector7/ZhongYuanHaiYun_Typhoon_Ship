import pandas as pd


port_file = "G:/Reins/projects/中远海运/数据集/dset/DIM_PORT.csv"

port_dset = pd.read_csv(port_file)
# print(port_dset.head())
# port_dset = port_dset.sort_values(by='port_id')
# print(port_dset.head())
# print(port_dset.describe())

port_dset = port_dset[['port_id', 'latitude', 'longitude']]

# print(port_dset.columns)