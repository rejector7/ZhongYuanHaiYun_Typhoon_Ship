import torch
from ship_path_prediction_v1.version_01 import LstmShip
import numpy as np
import time
from ship_path_prediction_v1.const import NUM_IN_FEATURE, NUM_OUT_FEATURE, NUM_HIDDEN, AHEAD_TIME, POINTS_PER_PATH, MAX_KNOT, MAX_LATI_LOTI_VAR
from sklearn.model_selection import train_test_split

lstm_ship = LstmShip()
lstm_ship.load_state_dict(torch.load("./data/ship_model_var_446.pth"))


npz_file = "./data/ship.npz"
ship_path_dset = np.load(npz_file)['ship']
ship_path_dset = ship_path_dset[:, :, :NUM_IN_FEATURE]

device = "cuda" if torch.cuda.is_available() else "cpu"
lstm_ship = lstm_ship.to(device)
lstm_ship.eval()
# print(device)
# index = np.random.randint(0, 300, (10, ))
# print(index)

record = torch.from_numpy(ship_path_dset[:, :, :]).float().to(device).squeeze()
# print(record)
record_input = record[:, :AHEAD_TIME, :]
# print(record_input.size())
out_string = record_input
for i in range(AHEAD_TIME, POINTS_PER_PATH - 1):
    # record_output = torch.cat((lstm_ship(record_input), record_input[:, AHEAD_TIME-1, 2:4].squeeze()), dim=1)
    record_output = lstm_ship(record_input)
    # print(record_output.size())
    out_string = torch.cat((out_string, record_output.unsqueeze(1)), dim=1)
    # print(out_string.size())
    record_input = torch.cat((record_input[:, 1:, :], record_output.unsqueeze(1)), dim=1)
    # print(record_input.size())
    # break
# print(out_string.size())

record[:, :, 0] = record[:, :, 0] * MAX_LATI_LOTI_VAR * 2 - MAX_LATI_LOTI_VAR
record[:, :, 1] = record[:, :, 1] * MAX_LATI_LOTI_VAR * 2 - MAX_LATI_LOTI_VAR
out_string[:, :, 0] = out_string[:, :, 0] * MAX_LATI_LOTI_VAR * 2 - MAX_LATI_LOTI_VAR
out_string[:, :, 1] = out_string[:, :, 1] * MAX_LATI_LOTI_VAR * 2 - MAX_LATI_LOTI_VAR


def multi_timestamp_diff(time_periods):
    # print(record.shape)
    # print(out_string.shape)

    lati_48 = torch.sum(record[:, AHEAD_TIME:AHEAD_TIME + time_periods, 0], dim=1)

    loti_48 = torch.sum(record[:, AHEAD_TIME:AHEAD_TIME + time_periods, 1], dim=1)
    pred_lati_48 = torch.sum(out_string[:, AHEAD_TIME:AHEAD_TIME + time_periods, 0], dim=1)
    pred_loti_48 = torch.sum(out_string[:, AHEAD_TIME:AHEAD_TIME + time_periods, 1], dim=1)
    # print(lati_48.shape)
    # print(pred_lati_48.shape)
    # print(out_string[:, AHEAD_TIME:AHEAD_TIME + time_periods, 0].shape)
    print("{:.1f} hours prediction lati/loti diff mean and std and %mean:".format(time_periods/2), file=log_file)
    print(torch.mean(torch.abs(lati_48 - pred_lati_48)), torch.std(torch.abs(lati_48 - pred_lati_48)),
          torch.mean(torch.abs((lati_48 - pred_lati_48)/lati_48)), file=log_file)
    print(torch.mean(torch.abs(loti_48 - pred_loti_48)), torch.std(torch.abs(loti_48 - pred_loti_48)),
          torch.mean(torch.abs((loti_48 - pred_loti_48)/loti_48)), file=log_file)


training_logfile = "./data/log.txt"
log_file = open(training_logfile, 'a')
print("\nStart training: ", file=log_file)
print(time.strftime("%Y-%m-%d %H:%M:%S %z"), file=log_file)

multi_timestamp_diff(6)
multi_timestamp_diff(12)
multi_timestamp_diff(24)
multi_timestamp_diff(48)
multi_timestamp_diff(96)
