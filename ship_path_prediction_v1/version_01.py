import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from ship_path_prediction_v1.const import NUM_IN_FEATURE, NUM_OUT_FEATURE, NUM_HIDDEN, AHEAD_TIME, \
    MAX_LATI_LOTI_VAR, MAX_KNOT


class LstmShip(nn.Module):
    def __init__(self):
        super(LstmShip, self).__init__()
        self.lstm_ship = nn.LSTM(input_size=NUM_IN_FEATURE, num_layers=2, hidden_size=NUM_HIDDEN, batch_first=True,
                                 dropout=0.2)
        self.fc = nn.Linear(in_features=NUM_HIDDEN, out_features=NUM_OUT_FEATURE)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        x, (hn, cn) = self.lstm_ship(x)
        # print(x.shape)
        x = x[:, -1, :].squeeze()
        x = self.fc(x)
        return x

    # def reset_parameters(self):


#
def train():
    training_logfile = "./data/log.txt"
    log_file = open(training_logfile, 'a')

    npz_file = "./data/ship.npz"
    ship_path_dset = np.load(npz_file)['ship']
    # print(ship_path_dset_15.shape)
    ship_path_dset = ship_path_dset[:, :, :NUM_IN_FEATURE]
    # print(ship_path_dset_15.shape)
    # tensor_dset = torch.from_numpy(ship_path_dset_15)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(device)

    train_dset, test_dset = train_test_split(ship_path_dset, test_size=0.3, shuffle=True)
    tensor_train_dset = torch.from_numpy(train_dset).float().to(device)
    tensor_test_dset = torch.from_numpy(test_dset).float().to(device)

    tensor_train_y = tensor_train_dset[:, AHEAD_TIME, :NUM_OUT_FEATURE].squeeze()
    tensor_test_y = tensor_test_dset[:, AHEAD_TIME, :NUM_OUT_FEATURE].squeeze()

    lstm_ship = LstmShip().to(device)

    optimizer = optim.SGD(lstm_ship.parameters(), lr=0.01)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000, 10000, 20000], gamma=0.1)
    criterion = nn.MSELoss().to(device)

    num_epoch = 100000
    print("\nStart training: ", file=log_file)
    print(time.strftime("%Y-%m-%d %H:%M:%S %z"), file=log_file)
    for epoch in range(num_epoch):
        lstm_ship.train()
        train_data = tensor_train_dset[:, :AHEAD_TIME, :]

        lstm_output = lstm_ship(train_data)

        cri0 = criterion(lstm_output[:, 0].squeeze(), tensor_train_y[:, 0].squeeze())
        cri1 = criterion(lstm_output[:, 1].squeeze(), tensor_train_y[:, 1].squeeze())
        cri2 = criterion(lstm_output[:, 2].squeeze(), tensor_train_y[:, 2].squeeze())
        cri3 = criterion(lstm_output[:, 3].squeeze(), tensor_train_y[:, 3].squeeze())
        train_loss = cri0 + cri1 + cri2 + cri3
        # train_loss = cri0 + cri1
        # loss = criterion(logits, train_y[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE])
        # train_diff = np.mean(np.sqrt(np.square((torch.sub(lstm_output[:, 0].squeeze(), tensor_train_y[:, 0].squeeze())).detach().cpu().numpy())
        #                    + np.square((torch.sub(lstm_output[:, 1].squeeze(), tensor_train_y[:, 1].squeeze())).detach().cpu().numpy())))

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        # lr_scheduler.step()

        lstm_ship.eval()
        with torch.no_grad():

            test_data = tensor_test_dset[:, :AHEAD_TIME, :]

            train_lstm_output = lstm_ship(train_data)
            test_lstm_output = lstm_ship(test_data)

            # train_cri0 = criterion(train_lstm_output[:, 0].squeeze(), tensor_train_y[:, 0].squeeze())
            # train_cri1 = criterion(train_lstm_output[:, 1].squeeze(), tensor_train_y[:, 1].squeeze())
            test_cri0 = criterion(test_lstm_output[:, 0].squeeze(), tensor_test_y[:, 0].squeeze())
            test_cri1 = criterion(test_lstm_output[:, 1].squeeze(), tensor_test_y[:, 1].squeeze())
            cri2 = criterion(test_lstm_output[:, 2].squeeze(), tensor_test_y[:, 2].squeeze())
            cri3 = criterion(test_lstm_output[:, 3].squeeze(), tensor_test_y[:, 3].squeeze())
            # test_loss = SCALE2*cri0 + SCALE2*cri1 + cri2 + cri3
            train_diff = np.mean(np.sqrt(
                np.square((torch.sub(train_lstm_output[:, 0].squeeze(), tensor_train_y[:, 0].squeeze())).detach().cpu().numpy())
                + np.square((torch.sub(train_lstm_output[:, 1].squeeze(), tensor_train_y[:, 1].squeeze())).detach().cpu().numpy()))
            ) * MAX_LATI_LOTI_VAR * 2

            test_diff = np.mean(np.sqrt(
                np.square((torch.sub(test_lstm_output[:, 0].squeeze(), tensor_test_y[:, 0].squeeze())).cpu().numpy())
                + np.square((torch.sub(test_lstm_output[:, 1].squeeze(), tensor_test_y[:, 1].squeeze())).cpu().numpy()))
            ) * MAX_LATI_LOTI_VAR * 2

            test_loss = test_cri0 + test_cri1 + cri2 + cri3

            if epoch == num_epoch - 1:
                # print(train_lstm_output[0:10, 0].squeeze() * 90 / SCALE1, tensor_train_y[0:10, 0].squeeze() * 90 / SCALE1)
                # print(sqrt_arr[:100])
                print("Training lati mean and std:", file=log_file)
                train_lati_diff = torch.abs(
                    (train_lstm_output[:, 0].squeeze() - 0.5) - (tensor_train_y[:, 0].squeeze() - 0.5))
                print(torch.mean(train_lati_diff) * MAX_LATI_LOTI_VAR * 2,
                      torch.std(train_lati_diff) * MAX_LATI_LOTI_VAR * 2, file=log_file)
                print("Training loti mean and std:", file=log_file)
                train_loti_diff = torch.abs(
                    (train_lstm_output[:, 1].squeeze() - 0.5) - (tensor_train_y[:, 1].squeeze() - 0.5))
                print(torch.mean(train_loti_diff) * MAX_LATI_LOTI_VAR * 2,
                      torch.std(train_loti_diff) * MAX_LATI_LOTI_VAR * 2, file=log_file)

                print("Training lati_knot mean and std:", file=log_file)
                train_lati_knot_diff = torch.abs(
                    (train_lstm_output[:, 2].squeeze() - 0.5) - (tensor_train_y[:, 2].squeeze() - 0.5))
                print(torch.mean(train_lati_diff) * MAX_KNOT * 2,
                      torch.std(train_lati_diff) * MAX_KNOT * 2, file=log_file)
                print("Training loti_knot mean and std:", file=log_file)
                train_loti_knot_diff = torch.abs(
                    (train_lstm_output[:, 3].squeeze() - 0.5) - (tensor_train_y[:, 3].squeeze() - 0.5))
                print(torch.mean(train_lati_diff) * MAX_KNOT * 2,
                      torch.std(train_lati_diff) * MAX_KNOT * 2, file=log_file)

                print("Testing lati mean and std:", file=log_file)
                test_lati_diff = torch.abs(
                    (test_lstm_output[:, 0].squeeze() - 0.5) - (tensor_test_y[:, 0].squeeze() - 0.5))
                print(torch.mean(test_lati_diff) * MAX_LATI_LOTI_VAR * 2, torch.std(test_lati_diff) * MAX_LATI_LOTI_VAR * 2,
                      file=log_file)
                print("Testing loti mean and std:", file=log_file)
                test_loti_diff = torch.abs(
                    (test_lstm_output[:, 1].squeeze() - 0.5) - (tensor_test_y[:, 1].squeeze() - 0.5))
                print(torch.mean(test_loti_diff) * MAX_LATI_LOTI_VAR * 2, torch.std(test_loti_diff) * MAX_LATI_LOTI_VAR * 2,
                      file=log_file)
                print("Testing lati_knot mean and std:", file=log_file)
                test_lati_knot_diff = torch.abs(
                    (train_lstm_output[:, 2].squeeze() - 0.5) - (tensor_train_y[:, 2].squeeze() - 0.5))
                print(torch.mean(train_lati_diff) * MAX_KNOT * 2,
                      torch.std(train_lati_diff) * MAX_KNOT * 2, file=log_file)
                print("Testing loti_knot mean and std:", file=log_file)
                test_loti_knot_diff = torch.abs(
                    (train_lstm_output[:, 3].squeeze() - 0.5) - (tensor_train_y[:, 3].squeeze() - 0.5))
                print(torch.mean(train_lati_diff) * MAX_KNOT * 2,
                      torch.std(train_lati_diff) * MAX_KNOT * 2, file=log_file)

                torch.save(lstm_ship.state_dict(), "./data/ship_model_var_446.pth")
        if epoch % 5000 == 0 or epoch == num_epoch - 1:
            print("Epoch {:09d}: Train Loss {:.12f}, Test Loss {:.12f}, Train Diff {:.12f}, Test Diff {:.12f}".format(epoch,
                                                                                                                      train_loss / len(
                                                                                                                          train_data),
                                                                                                                      test_loss / len(
                                                                                                                          test_data),
                                                                                                                      train_diff,
                                                                                                                      test_diff),
                  file=log_file)
            print("Epoch {:09d}: Train Loss {:.12f}, Test Loss {:.12f}, Train Diff {:.12f}, Test Diff {:.12f}".format(epoch,
                                                                                                                      train_loss / len(
                                                                                                                          train_data),
                                                                                                                      test_loss / len(
                                                                                                                          test_data),
                                                                                                                      train_diff,
                                                                                                                      test_diff),
                  )
    log_file.close()


# train()
