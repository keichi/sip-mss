#!/usr/bin/env python3

import os
import sys

import pandas as pd
import numpy as np


def main():
    population_path = sys.argv[1]
    areas_path = sys.argv[2]
    mss_path = sys.argv[3]

    print("Loading latest MSS dataset")

    # Unarchive zip and parse csv into a dataframe
    df = pd.read_csv(mss_path)
    data = {row.area: row.population for row in df.itertuples(index=False)}

    print("Reading latest population file")

    # Read latest population file into memory
    if os.path.isfile(population_path):
        population = np.load(population_path)
    else:
        population = np.zeros((0, 0), dtype=np.int32)

    print("Reading latest mesh ID file")

    # Read latest mesh ID file into memory
    if os.path.isfile(areas_path):
        areas = np.load(areas_path)
    else:
        areas = np.zeros((0), dtype=np.int32)

    print(f"population.shape: {population.shape}")
    print(f"areas.shape: {areas.shape}")

    # Find mesh IDs first seen
    new_areas = np.setdiff1d(np.fromiter(data, dtype=np.int32), areas)
    areas = np.concat([areas, new_areas])

    print(f"Total: {len(data)} Known: {len(data) - len(new_areas)} New: {len(new_areas)}")

    # Build mapping from mesh ID to column index
    id_to_idx = {id: idx for idx, id in enumerate(areas)}

    # Extend population array
    population = np.pad(population, pad_width=((0, 1), (0, len(new_areas))),
                        mode="constant", constant_values=-1)

    print("Updating population data")

    # Update
    for id, pop in data.items():
        population[-1, id_to_idx[id]] = pop

    print(f"population.shape: {population.shape}")
    print(f"areas.shape: {areas.shape}")

    print("Saving updated population and mesh ID")

    np.save(population_path, population)
    np.save(areas_path, areas)


if __name__ == "__main__":
    main()
