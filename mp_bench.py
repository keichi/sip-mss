#!/usr/bin/env python

# pip install stumpy==1.13.0
# FORCE_NO_CUDA=1 CXX=/opt/AMD/aocc-compiler-5.0.0/bin/clang++ pip3 install pyscamp==4.0.1
# CXX=/opt/AMD/aocc-compiler-5.0.0/bin/clang++ pip3 install git+https://github.com/keichi/fastmp.git
# taskset -c 0 python3 mp_bench.py

import os
import statistics
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numba
import numpy as np
import stumpy
import pyscamp
import fastmp


def benchmark(f):
    print("n\tmean\tmedian\tstdev")
    for n in [1 << i for i in range(10, 19)]:
        timings = []

        for i in range(3):
            T = np.random.rand(n)
            f(T)

        for i in range(10):
            T = np.random.rand(n)
            start = time.perf_counter()
            f(T)
            end = time.perf_counter()

            timings.append(end - start)

        mean = statistics.mean(timings)
        median = statistics.median(timings)
        stdev =  statistics.stdev(timings)

        print(f"{n}\t{mean}\t{median}\t{stdev}", flush=True)


def main():
    print("pyscmap")
    benchmark(lambda T: pyscamp.selfjoin(T, m=100, threads=1))

    print("stumpy")
    numba.set_num_threads(1)
    benchmark(lambda T: stumpy.stump(T, m=100))

    print("fastmp")
    benchmark(lambda T: fastmp.stomp(T, 100))


if __name__ == "__main__":
    main()
