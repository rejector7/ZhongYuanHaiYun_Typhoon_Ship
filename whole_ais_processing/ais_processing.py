import pandas as pd
from whole_ais_processing.const import MAX_KNOT, LOW_KNOT

standard_ais_2019 = "G:/Reins/datasets/AIS_2019_01_01/AIS_2019_01_01.csv"
ais_file = "G:/Reins/datasets/ais/ais.csv"
processed_file = "G:/Reins/datasets/ais/processed_ais.csv"
# f = open(processed_file, mode='w')
# f.close()
# iterator=True
ais_dset_reader = pd.read_csv(ais_file, delimiter='^', chunksize=100000, low_memory=False, error_bad_lines=False)
total_length = 0
for i, chunk_data in enumerate(ais_dset_reader):
    # if i % 10 == 0:
    #     print("Processing chunk {:5d}".format(i))
    # chunk_data = chunk_data[['latitude', 'longitude', 'shipment_dir', 'knot', 'posi_dt', 'mmsi_code']]
    # chunk_data = chunk_data[chunk_data.knot.apply(type) == float]
    # # chunk_data['knot'].astype(float)
    # # print(chunk_data['knot'].max())
    # # print(len(chunk_data[(chunk_data['knot'] > 20) & (chunk_data['knot'] <= 30)]))
    # chunk_data = chunk_data[(chunk_data['knot'] > LOW_KNOT) & (chunk_data['knot'] <= MAX_KNOT)]
    # print(chunk_data['knot'].max())
    # if i != 0:
    #     chunk_data.to_csv(processed_file, mode='a', header=False)
    # else:
    #     chunk_data.to_csv(processed_file, mode='a', header=True)

    # print(chunk_data.columns)
    # print(len(chunk_data.columns))
    # print(chunk_data.head())
    # print(chunk_data.describe())
    # break
    print(i, len(chunk_data.index))
    total_length += len(chunk_data.index)

print(total_length)
