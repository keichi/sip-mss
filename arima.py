import numpy as np
import pandas as pd
from pmdarima.arima import auto_arima
from threadpoolctl import threadpool_limits

population = np.load("ntt_mss_2019.npy")
areas = np.load("ntt_mss_2019_areas.npy")

aobayama = np.where(areas == 574036064)[0][0]
tokyo = np.where(areas == 533946113)[0][0]
kabukicho = np.where(areas == 533945363)[0][0]
shibuya = np.where(areas == 533935961)[0][0]

ts = pd.Series(data=population[:, tokyo],
               index=pd.date_range(start="2019-01-01", end="2020-01-01", inclusive="left", freq="H")) 
ts = ts.loc[pd.date_range(start="2019-4-1", end="2019-4-30", freq="H")]

with threadpool_limits(limits=1, user_api="blas"):
    model = auto_arima(ts, m=24, seasonal=True)
