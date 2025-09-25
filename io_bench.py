#!/usr/bin/env python

from datetime import datetime, timedelta
import statistics
import time

import pandas as pd


def main():
    from_date = datetime(year=2019, month=1, day=1)
    to_date = datetime(year=2020, month=1, day=1)

    timings = []

    cur = from_date
    while cur < to_date:
        path = cur.strftime("/uhome/y90186/mss-raw/%Y/%Y%m%d/clipped_mesh_pop_%Y%m%d%H00_00000.csv.zip")

        start = time.perf_counter()

        df = pd.read_csv(path)

        end = time.perf_counter()

        print(path, end - start)
        timings.append(end - start)

        cur += timedelta(hours=1)

    print(f"Mean: {statistics.mean(timings)}")
    print(f"Median: {statistics.median(timings)}")
    print(f"Stdev: {statistics.stdev(timings)}")


if __name__ == "__main__":
    main()
