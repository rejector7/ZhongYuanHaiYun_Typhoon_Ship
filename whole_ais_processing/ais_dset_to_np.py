import torch
import numpy as np

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


# save_file = "G:/Reins/datasets/ais/ais_dset.npz"
# np.savez(save_file, ais_path_record=np_dset)
