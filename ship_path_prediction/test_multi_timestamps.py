import torch
from ship_path_prediction.version00 import LstmShip
import numpy as np
from ship_path_prediction.const import SCALE1, SCALE2, NUM_IN_FEATURE, NUM_OUT_FEATURE, NUM_HIDDEN, AHEAD_TIME
from sklearn.model_selection import train_test_split

lstm_ship = LstmShip()
lstm_ship.load_state_dict(torch.load("./data/ship_model_224.pth"))


npz_file = "./data/ship.npz"
ship_path_dset_15 = np.load(npz_file)['ship']
ship_path_dset_15 = ship_path_dset_15[:, :, :NUM_IN_FEATURE]
# print(ship_path_dset_15.shape)
# tensor_dset = torch.from_numpy(ship_path_dset_15)
device = "cuda" if torch.cuda.is_available() else "cpu"
lstm_ship = lstm_ship.to(device)
lstm_ship.eval()
# print(device)
index = np.random.randint(0, 300, (10, ))
# print(index)
record = torch.from_numpy(ship_path_dset_15[:, :, :]).float().to(device).squeeze()
# print(record)
record_input = record[:, :AHEAD_TIME, :]
# print(record_input.size())
out_string = record_input
for i in range(AHEAD_TIME, 15):
    record_output = torch.cat((lstm_ship(record_input), record_input[:, AHEAD_TIME-1, 2:4].squeeze()), dim=1)
    # print(record_output.size())
    out_string = torch.cat((out_string, record_output.unsqueeze(1)), dim=1)
    # print(out_string.size())
    record_input = torch.cat((record_input[:, 1:, :], record_output.unsqueeze(1)), dim=1)
    # print(record_input.size())
    # break
# print(out_string.size())

record[:, :, 0] = record[:, :, 0] * 90
record[:, :, 1] = record[:, :, 1] * 360
out_string[:, :, 0] = out_string[:, :, 0] * 90
out_string[:, :, 1] = out_string[:, :, 1] * 360
lati_48 = record[:, -1, 0]
loti_48 = record[:, -1, 1]
pred_lati_48 = out_string[:, -1, 0]
pred_loti_48 = out_string[:, -1, 1]

print(torch.mean(torch.abs(lati_48 - pred_lati_48)), torch.std(torch.abs(lati_48 - pred_lati_48)))
print(torch.mean(torch.abs(loti_48 - pred_loti_48)), torch.std(torch.abs(loti_48 - pred_loti_48)))