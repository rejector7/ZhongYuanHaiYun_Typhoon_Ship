import pandas as pd
from whole_ais_processing.const import MAX_KNOT, LOW_KNOT
import chardet
import datetime
import time
import numpy as np

standard_ais_2019 = "G:/Reins/datasets/AIS_2019_01_01/AIS_2019_01_01.csv"
ais_file = "G:/Reins/datasets/ais/ais.csv"
processed_file = "G:/Reins/datasets/ais/processed_ais.csv"
f = open(processed_file, mode='w')
f.close()
iterator=True
ais_dset_reader = pd.read_csv(ais_file, delimiter='^', chunksize=100000, low_memory=False, error_bad_lines=False,
                              encoding='utf-8')
total_length = 0
for i, chunk_data in enumerate(ais_dset_reader):
    if i % 10 == 0:
        print("Processing chunk {:5d}".format(i))
    # print(chunk_data.dtypes)
    chunk_data = chunk_data[['latitude', 'longitude', 'shipment_dir', 'knot', 'posi_dt', 'mmsi_code', 'imo_number']]
    chunk_data = chunk_data[chunk_data.knot.apply(type) == float]
    chunk_data = chunk_data[chunk_data.latitude.apply(type) == float]
    chunk_data = chunk_data[chunk_data.longitude.apply(type) == float]
    chunk_data = chunk_data[chunk_data.shipment_dir.apply(type) == float]
    chunk_data = chunk_data[chunk_data.mmsi_code.apply(type) == int]
    # chunk_data = chunk_data[chunk_data.posi_dt.apply(type) == object]

    # print(len(chunk_data[chunk_data['posi_dt'] != np.nan]))
    # print(len(chunk_data.index))
    # print(chunk_data.describe())
    # break
    # chunk_data['knot'].astype(float)
    # print(chunk_data['knot'].max())
    # print(len(chunk_data[(chunk_data['knot'] > 20) & (chunk_data['knot'] <= 30)]))
    chunk_data = chunk_data[(chunk_data['knot'] > LOW_KNOT) & (chunk_data['knot'] <= MAX_KNOT)]

    chunk_data['MMSI'] = chunk_data['mmsi_code']
    chunk_data['posi_dt'] = pd.to_datetime(chunk_data['posi_dt'], errors="coerce")
    chunk_data.dropna(subset=['posi_dt'], inplace=True)
    chunk_data['BaseDateTime'] = chunk_data['posi_dt'].dt.strftime("%Y-%m-%dT%H:%M:%S")
    chunk_data.dropna(subset=['BaseDateTime'], inplace=True)
    # , format="%Y-%m-%dT%H-%M-%S"
    # chunk_data['BaseDateTime'] = datetime.datetime.strftime(chunk_data['BaseDateTime'], "%Y-%m-%dT%H-%M-%S")
    chunk_data['LAT'] = chunk_data['latitude']
    chunk_data['LON'] = chunk_data['longitude']
    chunk_data['SOG'] = chunk_data['knot']
    chunk_data['COG'] = chunk_data['shipment_dir']
    chunk_data['Heading'] = np.nan
    chunk_data['VesselName'] = np.nan
    chunk_data['IMO'] = chunk_data['imo_number']
    chunk_data['CallSign'] = np.nan
    chunk_data['VesselType'] = np.nan
    chunk_data['Status'] = np.nan
    chunk_data['Length'] = np.nan
    chunk_data['Width'] = np.nan
    chunk_data['Draft'] = np.nan
    chunk_data['Cargo'] = np.nan
    chunk_data['TranscieverClass'] = np.nan
    chunk_data.drop(['latitude', 'longitude', 'shipment_dir', 'knot', 'posi_dt', 'mmsi_code', 'imo_number'], axis=1, inplace=True)

    # del chunk_data['mmsi']
    # print(chunk_data.columns)
    # break
    # print(chunk_data['knot'].max())

    # print(chunk_data.head())
    if i != 0:
        chunk_data.to_csv(processed_file, mode='a', header=False, encoding='utf-8', index=False)
    else:
        chunk_data.to_csv(processed_file, mode='a', header=True, encoding='utf-8', index=False)
    # print(chunk_data.head())
    # break
    # print(chunk_data.columns)
    # print(len(chunk_data.columns))
    # print(chunk_data.head())
    # print(chunk_data.describe())
    # break
    # print(i, len(chunk_data.index))
    # total_length += len(chunk_data.index)

# print(total_length)


# def get_encoding(file):
#     with open(file, 'rb') as f:
#         print(chardet.detect(f.read())['encoding'])
#
#
# get_encoding()
# time.strptime()