import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ship_path_prediction.const import SCALE1, SCALE2, SCALE3

'''
process data
'''
ais_inship = pd.read_csv("./data/ordered_inship.csv")
# ais_inship['index_num'] = ais_inship['Unnamed: 0']

ais_inship['date'] = pd.to_datetime(ais_inship['posi_dt'])
ais_inship.set_index("date", inplace=True)
# ais_inship.drop(['posi_dt'], axis=1)
# print(ais_inship.index)
ais_inship['lati_knot'] = np.cos(ais_inship['shipment_dir']) * ais_inship['knot']
ais_inship['loti_knot'] = np.sin(ais_inship['shipment_dir']) * ais_inship['knot']
ais_inship['latitude'] = ais_inship['latitude'] * SCALE1/90
ais_inship['longitude'] = ais_inship['longitude'] * SCALE1/360
ais_inship['lati_knot'] = ais_inship['lati_knot']/100
ais_inship['loti_knot'] = ais_inship['loti_knot']/100
# print(ais_inship['lati_knot'].max())
# print(ais_inship['loti_knot'].max())
#
# ais_inship.to_csv("./data/ordered_inship_processed.csv")

# ais_inship = pd.read_csv("./data/ordered_inship_processed.csv", parse_dates=True, index_col='date')
#
ais_inship = ais_inship.resample('30min').mean()
ais_inship = ais_inship.fillna(method='ffill')

index_num_column = range(len(ais_inship.index))
ais_inship['index_num'] = index_num_column
print(ais_inship.columns)
print(ais_inship.shape)
# print(ais_inship['index_num'].values[:20])
# print(ais_inship.index.shape)
ais_inship = ais_inship[ais_inship['knot'] >= 0.01]
# print(ais_inship.index.shape)

# ais_inship['index_num'].hist()
# plt.show()
ais_inship = ais_inship[['latitude', 'longitude', 'lati_knot', 'loti_knot', 'index_num']].values
print(ais_inship.shape)
# # ais_inship['index_num']
#
ship_voyage_record = []
index_num = ais_inship[:, 4]
record_i = index_num[0]
gap = 1
temp_record = []
num_length = 13
'''
after cut the small-knot off, the index/time continuity suffers
to solve: get the continuous points strings and seperate different stringa
'''
for i in range(len(ais_inship)):
    if record_i + gap >= index_num[i]:
        record_i += 1
        temp_record.append(ais_inship[i, :4])
    else:
        ship_voyage_record.append(temp_record)
        temp_record = []
        record_i = index_num[i + 1]


print(len(ship_voyage_record))
leng_list = []
enough_leng_list = []
enough_num = 0
ship_voyage_long_record = []
sum_length = 0
for record in ship_voyage_record:
    leng_list.append(len(record))
    sum_length += len(record)
    if len(record) >= num_length:
        ship_voyage_long_record.append(record)
        enough_num += 1
        enough_leng_list.append(len(record))

print(sum_length)
print(leng_list)
print(enough_num)
print(enough_leng_list)

ship_voyage_dset = []
for record in ship_voyage_long_record:
    num_record = int(len(record)//num_length)
    for i in range(num_record):
        ship_voyage_dset.append(record[i * num_length:i * num_length + num_length])

ship_voyage_dset = np.array(ship_voyage_dset)

np.savez("./data/ship.npz", ship=ship_voyage_dset)
print(ship_voyage_dset.shape)


