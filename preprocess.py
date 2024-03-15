#!/usr/bin/env python3

from datetime import datetime, timedelta

import os
import pandas as pd
import numpy as np

data = {}

from_date = datetime(year=2019, month=1, day=1)
to_date = datetime(year=2020, month=1, day=1)
num_rows = (to_date - from_date) // timedelta(hours=1)

cur = from_date
i = 0
while cur < to_date:
    path = cur.strftime("/scratch/keichi/ntt/%Y/%Y%m%d/clipped_mesh_pop_%Y%m%d%H00_00000.csv.zip")

    if not os.path.isfile(path):
        continue

    df = pd.read_csv(path)
    for row in df.itertuples(index=False):
        if row.area not in data:
            data[row.area] = np.full(num_rows, -1, dtype=np.int32)
        data[row.area][i] = row.population

    print(path, len(data))

    cur += timedelta(hours=1)
    i += 1

out = np.lib.format.open_memmap("ntt_mss_2019.npy", mode="w+", dtype=np.int32,
                                shape=(num_rows, len(data)), fortran_order=True)

for i, v in enumerate(data.values()):
    out[:, i] = v

out.flush()

np.save("ntt_mss_2019_areas.npy", np.fromiter(data.keys(), dtype=np.int32))
