import pandas as pd
import numpy as np
from ship_path_prediction_v1.const import SAMPLE_POINT_SPACE, POINTS_PER_PATH, POINTS_CAN_LOW_IN_PATH, \
    SAMPLE_PATH_POINT_SPACE, MAX_LATI_LOTI_VAR, MAX_KNOT, LOW_KNOT


'''
process raw AIS csv data
remove 
get equally-spaced data
'''


def process_raw_ais_data(csv_file_path):
    ais_dset = pd.read_csv(csv_file_path, encoding=' ISO-8859-1')
    print(ais_dset.head())
    # get needed columns and sort by datetime, set range index
    ais_dset = ais_dset[['latitude', 'longitude', 'shipment_dir', 'knot', 'posi_dt', 'mmsi_code']]
    ais_dset = ais_dset.sort_values(by=["posi_dt"])
    ais_dset.set_index(pd.Index(range(len(ais_dset.index))), inplace=True)

    #  remove and replace impossible high knot dirty data by the last one
    high_knot_index = ais_dset[ais_dset['knot'] >= MAX_KNOT].index
    # print(ais_dset.loc[high_knot_index, ['knot', 'posi_dt']])
    # ais_dset.loc[high_knot_index, ['knot']] = ais_dset.loc[high_knot_index - 1, ['knot']]
    for i in high_knot_index:
        ais_dset.loc[i, ['knot']] = ais_dset.loc[i - 1, ['knot']]
    # print(ais_dset.loc[high_knot_index, ['knot', 'posi_dt']])
    # print(ais_dset.loc[high_knot_index - 1, ['knot', 'posi_dt']])

    # set index to date
    ais_dset['date'] = pd.to_datetime(ais_dset['posi_dt'])
    ais_dset.set_index("date", inplace=True)

    # resample by timefreq = SAMPLE_POINT_SPACE
    ais_dset = ais_dset.resample(SAMPLE_POINT_SPACE).mean()  # mean() first()
    # ais_dset = ais_dset.fillna()

    # add continuous index for separating paths and drop nan and low knot data
    index_num_column = range(len(ais_dset.index))
    ais_dset['index_num'] = index_num_column
    ais_dset = ais_dset[ais_dset['knot'] > LOW_KNOT]
    ais_dset.dropna(axis=0)

    # get lati and loti knot
    ais_dset['lati_knot'] = np.cos(ais_dset['shipment_dir']) * ais_dset['knot']
    ais_dset['loti_knot'] = np.sin(ais_dset['shipment_dir']) * ais_dset['knot']
    # print(ais_dset['lati_knot'].max(), ais_dset['loti_knot'].max())
    # print(ais_dset['lati_knot'].min(), ais_dset['loti_knot'].min())

    # get lati and loti var between continous two points
    ais_dset['last_lati'] = np.concatenate((np.array([0]), ais_dset['latitude'].values[:-1]), 0)
    ais_dset['lati_var'] = ais_dset['latitude'] - ais_dset['last_lati']
    ais_dset['last_loti'] = np.concatenate((np.array([0]), ais_dset['longitude'].values[:-1]), 0)
    ais_dset['loti_var'] = ais_dset['longitude'] - ais_dset['last_loti']
    ais_dset.drop(ais_dset.index[0], inplace=True)

    # normalization
    ais_dset['latitude'] = (ais_dset['latitude'] / 180) + 0.5
    ais_dset['longitude'] = ais_dset['longitude'] / 360
    ais_dset['lati_knot'] = ais_dset['lati_knot'] / (MAX_KNOT * 2) + 0.5
    ais_dset['loti_knot'] = ais_dset['loti_knot'] / (MAX_KNOT * 2) + 0.5

    ais_dset['lati_var'] = ais_dset['lati_var'] / (MAX_LATI_LOTI_VAR * 2) + 0.5
    ais_dset['loti_var'] = ais_dset['loti_var'] / (MAX_LATI_LOTI_VAR * 2) + 0.5

    # print(ais_dset['lati_var'].values.max(), ais_dset['loti_var'].values.max())
    # print(ais_dset['lati_var'].values.min(), ais_dset['loti_var'].values.min())

    # print(ais_dset.columns)
    # print(ais_dset.shape)

    # print("Cutted by knot, ais dset totol points: ", len(ais_dset.index))
    # print(ais_dset.index.shape)
    # ais_dset['index_num'].hist()
    # plt.show()
    ais_dset = ais_dset[['lati_var', 'loti_var', 'lati_knot', 'loti_knot', 'latitude', 'longitude', 'index_num']].values
    ship_path_records = []
    index_num = ais_dset[:, -1]
    path_record_point_index = index_num[0]
    cur_path = []

    # max_lati_var = 0
    # min_lati_var = 0
    # max_loti_var = 0
    # min_loti_var = 0
    '''
    after cut the small-knot off, the index/time continuity suffers
    to solve: get the continuous points strings and seperate different stringa
    '''
    for i in range(len(ais_dset)):
        if path_record_point_index + POINTS_CAN_LOW_IN_PATH >= index_num[i]:
            path_record_point_index += 1
            cur_path.append(ais_dset[i, :-1])
        else:
            ship_path_records.append(cur_path)
            cur_path = []
            path_record_point_index = index_num[i + 1]

    # print("Total path nums:", len(ship_path_records))
    lenth_list = []
    long_path_list = []
    long_path_num = 0
    ship_long_path_records = []
    sum_length = 0

    for record in ship_path_records:
        lenth_list.append(len(record))
        sum_length += len(record)
        if len(record) >= POINTS_PER_PATH:
            ship_long_path_records.append(record[:-1])
            long_path_num += 1
            long_path_list.append(len(record))

    # print("cutted by knot, total points:", sum_length)
    # print("cutted by knot, path lenth list:", lenth_list)
    # print("valid long path num:",long_path_num)
    # print("valid long path length list", long_path_list)
    # print(np.sum(long_path_list) - len(long_path_list) * POINTS_PER_PATH)
    ship_path_dset = []
    for record in ship_long_path_records:
        num_record = int((len(record) - POINTS_PER_PATH) // SAMPLE_PATH_POINT_SPACE + 1)
        for i in range(num_record):
            ship_path_dset.append(record[i * SAMPLE_PATH_POINT_SPACE:i * SAMPLE_PATH_POINT_SPACE + POINTS_PER_PATH])

    ship_path_dset = np.array(ship_path_dset)
    # print(np.max(ship_path_dset[:, :, 0]), np.min(ship_path_dset[:, :, 0]))
    # print(np.max(ship_path_dset[:, :, 1]), np.min(ship_path_dset[:, :, 1]))
    # np.savez("../ship_path_prediction/data/ship.npz", ship=ship_path_dset)
    # print("ship_path_dset shape:", ship_path_dset.shape)
    return ship_path_dset


ais_file = "../ship_path_prediction/data/ODS_JT_AIS_INSHIPHISTORY.csv"
process_raw_ais_data(ais_file)

# df = pd.DataFrame({'Animal': ['Falcon', 'Falcon',
#                               'Parrot', 'Parrot'],
#                    'Max Speed': [380., 370., 24., 26.]})
#
# print(type(df.groupby(['Animal'])))