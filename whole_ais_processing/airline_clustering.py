import pandas as pd
import folium
import matplotlib.pyplot as plt

"""
dim_port file processing and show
"""

def get_ports():
    port_file = "../dataset/DIM_PORT.csv"
    port_dset = pd.read_csv(port_file)
    # print(port_dset.head())
    # port_dset = port_dset.sort_values(by='port_id')
    # print(port_dset.head())
    # print(port_dset.describe())
    port_dset = port_dset[['port_id', 'port_cd', 'country_cd', 'latitude', 'longitude']]
    # print(port_dset.columns)
    return port_dset


def plot_ports():
    port_dset = get_ports()
    fm = folium.Map(location=[25, 120])
    fm.save("../processed_dset/ports.html")
    plt.show()

plot_ports()