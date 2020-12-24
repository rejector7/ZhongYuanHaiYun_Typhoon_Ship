import torch
import numpy as np
import pandas as pd
import math
import pickle


Threshold = 0.1
# Threshold

# valid_ports = 0
# valid_port = 0


def ais_path_dset_to_np():
    ais_dset_file = "G:/Reins/datasets/ais/ais.pkl"
    ship_path_dset = torch.load(ais_dset_file)
    # print(len(ship_path_dset))

    # temp = np.asarray(ship_path_dset[0])
    # print(type(temp))
    # print(temp.shape)
    np_dset = []
    # print(np_dset)
    for record in ship_path_dset:
        temp_np_array = np.asarray(record)
        np_dset.append(temp_np_array)

    np_dset = np.array(np_dset, dtype=object)
    # print(np_dset.shape)
    # print(type(np_dset))
    save_file = "G:/Reins/datasets/ais/ais_dset.npz"
    np.savez(save_file, ais_path_record=np_dset)


def port_csv_to_np():
    port_csv_file = "G:/Reins/projects/中远海运/数据集/dset/DIM_PORT.csv"
    port_df = pd.read_csv(port_csv_file)
    port_df = port_df[['port_id', 'port_cd', 'longitude', 'latitude']]
    port_np = port_df.values
    # print(type(port_np))
    # print(port_np.shape)
    save_file = "G:/projects/python_projects/ZhongYuanHaiYun_Typhoon_Ship/airline_clustering/data/port.npz"
    np.savez(save_file, port=port_np)

# port_csv_to_np()

def ports_to_cd_dict():
    ports_file = "G:/projects/python_projects/ZhongYuanHaiYun_Typhoon_Ship/airline_clustering/data/port.npz"
    ports = np.load(ports_file, allow_pickle=True)['port']
    ports_dict = {}
    for port in ports:
        ports_dict[port[1]] = port[[0, 2, 3]]
    return ports_dict


def ports_to_dict():
    ports_file = "G:/projects/python_projects/ZhongYuanHaiYun_Typhoon_Ship/airline_clustering/data/port.npz"
    ports = np.load(ports_file, allow_pickle=True)['port']
    ports_dict = {}
    for port in ports:
        ports_dict[port[0]] = port[1:]
    return ports_dict

# ports_dict = ports_to_dict()
# print(ports_dict.values())


def map_end_to_port(end):
    longi, lati = end
    ports_dict = ports_to_dict()
    for port_id, port_posi in ports_dict.items():
        if abs(lati - port_posi[0]) <= Threshold and abs(longi - port_posi[1]) <= Threshold:
            # print(port_posi)
            # break
            # global valid_port
            # valid_port += 1
            # print("valid port:", valid_port)
            return port_id
        else:
            continue
    return None

# map_end_to_port((1,2))


def map_path_ends_pair_to_ports(ends_pair):
    ports_id = []
    for end in ends_pair:
        port_id = map_end_to_port(end)
        if port_id is None:
            return None
        ports_id.append(port_id)
    # global valid_ports
    # valid_ports += 1
    # print("valid ports:", valid_ports)
    return ports_id


#dict (start_port_id, end_port_id): [path, path2]
def map_path_to_ports():
    path_dset_file = "G:/Reins/datasets/ais/ais_dset.npz"
    path_dset = np.load(path_dset_file, allow_pickle=True)['ais_path_record']
    path_port_dict = {}
    # valid_path = 0
    for path in path_dset:
        # print(type(path))
        # print(path)
        # path = np.asarray(path)
        # print(path.shape)
        ends_pair = np.stack((path[0, 1:3], path[-1, 1:3]), axis=0)
        # print(ends_pair.shape)
        # break
        ports_id = map_path_ends_pair_to_ports(ends_pair)
        if ports_id is None:
            continue
        else:
            path_port_dict[tuple(ports_id)] = path_port_dict.get(tuple(ports_id), [])
            path_port_dict[tuple(ports_id)].append(list(path[1:3]))
    #         valid_path += 1
    # print(valid_path)
    # path_port_np = "G:/projects/python_projects/ZhongYuanHaiYun_Typhoon_Ship/airline_clustering/data/path_port.npy"
    # np.save(path_port_np, path_port_dict)
    return path_port_dict


def get_valid_ports():
    path_port_np = "G:/projects/python_projects/ZhongYuanHaiYun_Typhoon_Ship/airline_clustering/data/path_port.npy"
    path_port_dict = dict(np.load(path_port_np, allow_pickle=True).item())
    # print("in get valid ports:", len(path_port_dict))
    valid_ports = path_port_dict.keys()
    # print(len(valid_ports))
    valid_ports_list = []
    for ports in valid_ports:
        valid_ports_list.append(ports)
    # print("in get valid ports:")
    # print(np.array(valid_ports_list, dtype=int).shape)
    return np.array(valid_ports_list, dtype=int)




# get_valid_ports()


# map_path_to_ports()
# path_port_dict = np.load("G:/projects/python_projects/ZhongYuanHaiYun_Typhoon_Ship/airline_clustering/data/path_port.npy", allow_pickle=True).item()
# print(len(path_port_dict))
# print(len(dict(path_port_dict).keys()))



# path_port_dict = map_path_to_ports()
# total_valid_path = 0
# for paths in path_port_dict.values():
#     total_valid_path += len(paths)
#
# print(total_valid_path)
# path_dset_file = "G:/Reins/datasets/ais/ais_dset.npz"
# path_dset = np.load(path_dset_file, allow_pickle=True)['ais_path_record']
# # path_port_dict = {}
# i = 0
# for path in path_dset:
#     if i < 10:
#         i += 1
#     else:
#         break
#     # print(type(path))
#     # print(path)
#     # path = np.asarray(path)
#     # print(path.shape)
#     ends_pair = np.stack((path[0, 1:3], path[-1, 1:3]), axis=0)
#     print(ends_pair)

