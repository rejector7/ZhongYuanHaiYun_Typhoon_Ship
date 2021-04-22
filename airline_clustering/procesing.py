import pandas as pd
import folium
import matplotlib.pyplot as plt
from collections import defaultdict
from global_land_mask import globe
import time


def get_ports():
    """
    dim_port file processing and show
    :return:
    """
    port_file = "../dataset/DIM_PORT.csv"
    port_dset = pd.read_csv(port_file)
    # print(port_dset.head())
    # port_dset = port_dset.sort_values(by='port_id')
    # print(port_dset.head())
    # print(port_dset.describe())
    port_dset = port_dset[['port_id', 'port_cd', 'country_cd', 'latitude', 'longitude']]
    port_dset.dropna(inplace=True)
    # remove invalid lon data
    port_dset = port_dset[port_dset['longitude'] <= 180]
    port_dset = port_dset[port_dset['latitude'] <= 90]
    # print(port_dset['longitude'].max(), port_dset['longitude'].min())
    # print(len(port_dset.values))
    return port_dset


def get_shorted_ports_cd_to_position_dict():
    """
    port_cd - country_cd = shorted port_cd
    but some shorted cd are the same, no solution, because the jy file are shorted cd.
    :return:
    """
    ports_shorted_cd_position_dict = {}
    port_dset = get_ports()
    for port in port_dset.values:
        ports_shorted_cd_position_dict[port[1][2:]] = [port[3], port[4]]
    # print(len(ports_shorted_cd_position_dict))
    return ports_shorted_cd_position_dict


def get_jy_voyages():
    """
    jy_ports file: get jy ports voyages list divided by verssel_nm and voyage
    then produce the voyage ports position lists.
    :return:
    """
    jy_ports_file = "../dataset/jy_ports.csv"
    jy_ports_dset = pd.read_csv(jy_ports_file)
    jy_ports_dset = jy_ports_dset[['verssel_nm', 'voyage', 'port_seq', 'port_cd', 'port_nm']]
    jy_ports_dset.dropna(inplace=True)
    jy_ports_dset = jy_ports_dset.values
    jy_ports_dict = {}
    for voyage in jy_ports_dset:
        if (voyage[0], voyage[1]) not in jy_ports_dict:
            jy_ports_dict[(voyage[0], voyage[1])] = [voyage[3]]
        else:
            jy_ports_dict[(voyage[0], voyage[1])].append(voyage[3])
    # print(jy_ports_dict.values())
    ports_shorted_cd_position_dict = get_shorted_ports_cd_to_position_dict()
    jy_ports_positions = []
    for voyage in jy_ports_dict.values():
        voyage_po = []
        for port_cd in voyage:
            if port_cd[0:3] not in ports_shorted_cd_position_dict:
                continue
            position = ports_shorted_cd_position_dict[port_cd[0:3]]
            voyage_po.append(position)
        jy_ports_positions.append(voyage_po)
    return jy_ports_positions


def cutting_path_by_land(path, k=2):
    """
    continuos k point on the land, cut it.
    init k = 2
    :param path: predicted path
    :param k:
    :return: cutted path
    """
    land_count = 0
    for i, point in enumerate(path):
        if globe.is_land(point[0], point[1]):
            land_count += 1
        if land_count >= k:
            return path[:i]
    return path


def cut_by_closer_to_ports(path):
    """

    :param path:
    :return:
    """


def test_cutting_time():
    paths = []
    for i in range(100):
        path = []
        for j in range(48):
            path.append([31, 122])
        paths.append(path)
    start = time.time()
    for path in paths:
        cutting_path_by_land(path)
    end = time.time()
    print(end - start)


test_cutting_time()
# port_dset = get_ports()
# true_count = 0
# false_count = 0
# for port in port_dset.values:
#     if globe.is_land(port[3], port[4]):
#         true_count += 1
#     else:
#         false_count += 1
# print(true_count, false_count)
# print(get_jy_voyages())


# print(get_ports())
# print(get_shorted_ports_cd_to_position_dict())


