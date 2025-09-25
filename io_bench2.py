#!/usr/bin/env python

import statistics
import time

import numpy as np


def main():
    timings = []

    data = np.load("/uhome/y90186/mss-numpy/ntt_mss_2016.npy", mmap_mode="r")

    for i in np.random.permutation(data.shape[1])[:10000]:
        start = time.perf_counter()
        ts = np.asarray(data[:, i], copy=True)
        end = time.perf_counter()

        print(i, end - start)
        timings.append(end - start)

    print(f"Mean: {statistics.mean(timings)}")
    print(f"Median: {statistics.median(timings)}")
    print(f"Stdev: {statistics.stdev(timings)}")


if __name__ == "__main__":
    main()
