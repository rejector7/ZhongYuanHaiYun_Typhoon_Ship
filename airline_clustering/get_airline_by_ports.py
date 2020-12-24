import numpy as np
import pandas as pd
from airline_clustering.util import ports_to_dict, ports_to_cd_dict, get_valid_ports, Threshold

# port_cd -> lat, lon -> near ports


def get_port_id_by_port_cd(port_cd):
    ports_cd_dict = ports_to_cd_dict()
    # lat, lon = ports_cd_dict[port_cd][1:]
    return ports_cd_dict[port_cd][0]    # port_id


def calculateNear(planned_port_posi, checked_port_posi, threshold=Threshold):
    # print(planned_port_posi, checked_port_posi)
    # if abs(planned_port_posi[0] - checked_port_posi[0]) <= Threshold:
    #     print(abs(planned_port_posi[0] - checked_port_posi[0]), abs(planned_port_posi[1] - checked_port_posi[1]))
    if abs(planned_port_posi[0] - checked_port_posi[0]) <= threshold \
            and abs(planned_port_posi[1] - checked_port_posi[1]) <= threshold:
        # print(abs(planned_port_posi[0] - checked_port_posi[0]), abs(planned_port_posi[1] - checked_port_posi[1]))
        return True
    else:
        return False


def get_near_valid_ports_by_ports_id(ports_id, threshold=Threshold):
    print(threshold)
    # count = 0
    valid_ports = get_valid_ports()
    ports_dict = ports_to_dict()
    start_port, end_port = ports_id
    # print(start_port, end_port)
    start_posi = ports_dict[start_port][1:]
    end_posi = ports_dict[end_port][1:]
    # print(start_posi, end_posi)
    near_valid_ports = []
    # breakpoint()
    for ports in valid_ports:
        # print(ports)
        port_posi_0 = ports_dict[ports[0]][1:]
        port_posi_1 = ports_dict[ports[1]][1:]
        # print(ports_dict[ports[0]])
        # print(port_posi_0)
        # break
        # print(calculateNear(start_posi, port_posi_0), calculateNear(end_posi, port_posi_1))
        # print("-----")
        if calculateNear(start_posi, port_posi_0, threshold) and calculateNear(end_posi, port_posi_1, threshold):
            # count += 1
            # print(count)
            near_valid_ports.append(ports)
    print(near_valid_ports)
    return near_valid_ports


def get_near_valid_ports_by_times_threshold(ports_id, threshold=Threshold):
    near_valid_ports = get_near_valid_ports_by_ports_id(ports_id, threshold)
    while len(near_valid_ports) == 0:
        threshold = threshold * 2
        if threshold > 1:
            print("path not found!")
            return None
        near_valid_ports = get_near_valid_ports_by_ports_id(ports_id, threshold)
    return near_valid_ports

# get_near_valid_ports_by_ports_id((45, 166))
# get_near_valid_ports_by_times_threshold((54, 46), Threshold)

def get_path_by_ports(ports_id):
    path_port_dict = dict(np.load("G:/projects/python_projects/ZhongYuanHaiYun_Typhoon_Ship/airline_clustering/data/path_port.npy", allow_pickle=True).item())
    # print(len(path_port_dict))
    near_valid_ports = get_near_valid_ports_by_times_threshold(ports_id)
    if near_valid_ports is None:
        return None
    else:
        valid_ports_pair = near_valid_ports[0]
        path = path_port_dict[tuple(valid_ports_pair)][0]
        # path_list = []
        # for point in path:
        #     path_list.append(np.array(point))
        path = np.array(path)
        # print(type(path))
        # print(len(path))
        # print(path.shape)
        # print(type(path[0]))
        # print(len(path[0]))
        # # print(type(path[1]))
        # print(len(path[1]))
        # print(path)
        # print(path[:, 1:3])
    return path[:, 1:3]


def get_paths_by_multiple_ports_pair(ports_list):
    path_port_dict = dict(np.load("G:/projects/python_projects/ZhongYuanHaiYun_Typhoon_Ship/airline_clustering/data/path_port.npy", allow_pickle=True).item())
    paths = []
    for ports_id in ports_list:
        near_valid_ports = get_near_valid_ports_by_times_threshold(ports_id)
        if near_valid_ports is None:
            path = np.array([])
        else:
            valid_ports_pair = near_valid_ports[0]
            path = path_port_dict[tuple(valid_ports_pair)][0]
            path = np.array(path)[:, 1:3]
        paths.append(path)
    return np.array(paths)

# get_path_by_ports((45, 167))

def get_path_by_ports_cd(start_port_cd, end_port_cd):
    ports_id = (get_port_id_by_port_cd(start_port_cd), get_port_id_by_port_cd(end_port_cd))
    path = get_path_by_ports(ports_id)
    # print(len(path))
    return path



# get_path_by_ports_cd()