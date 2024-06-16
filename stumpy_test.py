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
CMD_CHECKPOINT = 3
CMD_RESTART = 4
CMD_KILL = 5


class Worker(Process):
    def __init__(self, idx):
        super().__init__()

        psutil.Process().cpu_affinity([idx])
        warnings.filterwarnings("ignore", module="stumpy.core")
        os.environ["NUMBA_NUM_THREADS"] = "1"

        self.idx = idx
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
                        self.data[area] = stumpi(self.data[area], m=WINDOW_SIZE)
                        todo.add(area)

                    self.data[area].update(population)
                    todo.remove(area)

                for area in todo:
                    self.data[area].update(0)

                print(f"Worker #{self.idx} has {len(self.data)} meshes")

            elif cmd == CMD_CHECKPOINT:
                T = np.zeros([len(self.data), TIME_SERIES_LENGTH])
                mp = np.zeros([len(self.data), TIME_SERIES_LENGTH - WINDOW_SIZE + 1, 4])

                for i, model in enumerate(self.data.values()):
                    T[i, :] = model.T_
                    mp[i, :, 0] = model.P_
                    mp[i, :, 1] = model.I_
                    mp[i, :, 2] = model.left_I_

                with open(f"checkpoint_{self.idx}.dat", "wb") as f:
                    np.save(f, np.array(list(self.data.keys())))
                    np.save(f, T)
                    np.save(f, mp)

            elif cmd == CMD_RESTART:
                with open(f"checkpoint_{self.idx}.dat", "rb") as f:
                    areas = np.load(f)
                    T = np.load(f)
                    mp = np.load(f)

                for i, area in enumerate(areas):
                    self.data[area] = stumpi(T[i, :], m=WINDOW_SIZE, mp=mp[i, :, :])

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

    # print("Loading data")

    # start = time.time()

    # for i in range(WINDOW_SIZE):
    #     path = cur.strftime("/scratch/keichi/ntt/%Y/%Y%m%d/clipped_mesh_pop_%Y%m%d%H00_00000.csv.zip")

    #     df = pd.read_csv(path)
    #     d = defaultdict(dict)

    #     for row in df.itertuples(index=False):
    #         d[row.area % NUM_WORKERS][row.area] = row.population

    #     for worker_id in range(NUM_WORKERS):
    #         workers[worker_id].queue.put_nowait((CMD_LOAD, d[worker_id]))

    #     print("Loaded:", path)

    #     cur += timedelta(hours=1)

    # for worker in workers:
    #     worker.queue.join()

    # end = time.time()

    # print(f"Loading data took {end - start:.3f}s")

    # print("Building initial STUMPI models ")

    # start = time.time()

    # for worker in workers:
    #     worker.queue.put_nowait((CMD_BUILD, ()))

    # for worker in workers:
    #     worker.queue.join()

    # end = time.time()

    # print(f"Building initial models {end - start:.3f}s")

    print("Reading STUMPI models from checkpoint")

    start = time.time()

    for worker in workers:
        worker.queue.put_nowait((CMD_RESTART, ()))

    for worker in workers:
        worker.queue.join()

    end = time.time()

    print(f"Read STUMPI models {end - start:.3f}s")


    for i in range(3):
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

    # print("Checkpoint STUMPI models ")

    # start = time.time()

    # for worker in workers:
    #     worker.queue.put_nowait((CMD_CHECKPOINT, ()))

    # for worker in workers:
    #     worker.queue.join()

    # end = time.time()

    # print(f"Checkpoint STUMPI models {end - start:.3f}s")

    for worker in workers:
        worker.queue.put((CMD_KILL, ()))
        worker.join()


if __name__ == "__main__":
    main()
