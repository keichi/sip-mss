#!/usr/bin/env python3

import os
import time
from datetime import datetime, timedelta
from collections import defaultdict
import warnings

import pandas as pd
import numpy as np
import psutil
from stumpy import stumpi
from multiprocessing import Process, JoinableQueue


TIME_SERIES_LENGTH = 720
WINDOW_SIZE = 24
NUM_WORKERS = psutil.cpu_count()
CHUNK_SIZE = 1000

CMD_LOAD = 0
CMD_BUILD = 1
CMD_UPDATE = 2
CMD_KILL = 3


class Worker(Process):
    def __init__(self, idx):
        super().__init__()

        psutil.Process().cpu_affinity([idx])
        warnings.filterwarnings("ignore", module="stumpy.core")
        os.environ["NUMBA_NUM_THREADS"] = 1

        self.data = {}
        self.queue = JoinableQueue()

    def run(self):
        time = 0
        while True:
            cmd, args = self.queue.get(True)

            if cmd == CMD_LOAD:
                for (area, population) in args.items():
                    if area not in self.data:
                        self.data[area] = np.zeros(TIME_SERIES_LENGTH, dtype=np.float64)
                    self.data[area][time] = population
                time += 1

            elif cmd == CMD_BUILD:
                for key in self.data.keys():
                    self.data[key] = stumpi(self.data[key], WINDOW_SIZE)

            elif cmd == CMD_UPDATE:
                todo = set(self.data.keys())
                for (area, population) in args.items():
                    if area not in self.data:
                        self.data[area] = np.zeros(TIME_SERIES_LENGTH, dtype=np.float64)
                        self.data[area] = stumpi(self.data[area], WINDOW_SIZE)
                        todo.add(area)

                    self.data[area].update(population)
                    todo.remove(area)

                for area in todo:
                    self.data[area].update(0)

            elif cmd == CMD_KILL:
                break

            self.queue.task_done()


def main():
    workers = []
    cur = datetime(year=2019, month=1, day=1)

    print(f"Launching {NUM_WORKERS} workers")

    for worker_id in range(NUM_WORKERS):
        workers.append(Worker(worker_id))
        workers[worker_id].start()

    print("Loading data")

    start = time.time()

    for i in range(WINDOW_SIZE):
        path = cur.strftime("/scratch/keichi/ntt/%Y/%Y%m%d/clipped_mesh_pop_%Y%m%d%H00_00000.csv.zip")

        df = pd.read_csv(path)
        d = defaultdict(dict)

        for row in df.itertuples(index=False):
            d[row.area % NUM_WORKERS][row.area] = row.population

        for worker_id in range(NUM_WORKERS):
            workers[worker_id].queue.put_nowait((CMD_LOAD, d[worker_id]))

        print("Loaded:", path)

        cur += timedelta(hours=1)

    for worker in workers:
        worker.queue.join()

    end = time.time()

    print(f"Loading data took {end - start:.3f}s")

    print("Building initial STUMPI models ")

    start = time.time()

    for worker in workers:
        worker.queue.put_nowait((CMD_BUILD, ()))

    for worker in workers:
        worker.queue.join()

    end = time.time()

    print(f"Building initial models {end - start:.3f}s")

    for i in range(100):
        print("Updating models")

        start = time.time()

        path = cur.strftime("/scratch/keichi/ntt/%Y/%Y%m%d/clipped_mesh_pop_%Y%m%d%H00_00000.csv.zip")
        df = pd.read_csv(path)

        d = defaultdict(dict)

        for row in df.itertuples(index=False):
            d[row.area % NUM_WORKERS][row.area] = row.population

        for worker_id in range(NUM_WORKERS):
            workers[worker_id].queue.put_nowait((CMD_UPDATE, d[worker_id]))

        cur += timedelta(hours=1)

        for worker in workers:
            worker.queue.join()

        end = time.time()

        cur += timedelta(hours=1)

        print(f"Updating models took {end - start:.3f}s")

    for worker in workers:
        worker.queue.put((CMD_KILL, ()))
        worker.join()


if __name__ == "__main__":
    main()
