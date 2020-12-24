import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

#increase figure size
from pylab import rcParams
rcParams['figure.figsize'] = (14,10)


path_port_dict = np.load("G:/projects/python_projects/ZhongYuanHaiYun_Typhoon_Ship/airline_clustering/data/path_port.npy", allow_pickle=True).item()
