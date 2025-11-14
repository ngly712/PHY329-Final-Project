# Planned imports
# numpy
import numpy as np

# Class (no inheritance)
# standardMap
class standardMap():

    # Initialization
    # User can set K, nIters, random seed
    # K is a positive real number (checks) -- defaults to 1
    # nIters is a positive integer (checks) -- defaults to 500
    # seed is a positive integer w/ 0 (checks) -- defaults to None
    # Object stores list of dicts w/ K, nIters, seed, and batched arrays for I
    # and theta
    # Call the list "runs"
    # Initialization makes an empty list
    def __init__(self, K: int=1, nIters: int=500, seed=None) -> None:
        # Checks for K and nIters
        assert isinstance(K, float)
        assert isinstance(nIters, int)
        assert K >= 0
        assert nIters > 0
        # Add to class
        self.K = K
        self.nIters = nIters
        self.seed = seed
        # Initialize the list of dicts
        self.runs = []


    # Function: simulate
    # Option to append new run or overwrite -- default is append
    # User supplies initial values for I and theta -- defaults to random \
    # (seeded or not)
    # I and theta are limited to [0, 2pi] (checks)
    # Initial conditions can be a batch vector or a number of random ICs
    # Overwrite replaces most recent run in list with dict entry
    # Append creates new dict entry in list
    def simulate(self, option: str = "append", ic: np.ndarray | int = 1) -> None:
        # Initialize the array
        if isinstance(ic, int):
            assert ic > 0
            state = np.zeros((ic, 2, self.nIters))
            state[..., 0] = np.random.uniform(0, 2*np.pi, (ic, 2, 1))
        elif isinstance(ic, np.ndarray):
            assert len(ic) > 2
            assert len(ic) % 2 == 0
            state = np.zeros((len(ic), 2, self.nIters))
            state[..., 0] = ic
        else:
            raise # Appropriate error here
        # Run the map
        for i in range(self.nIters - 1):
            state[:, 0, i + 1] = (state[:, 0, i] + self.K * np.sin(state[:, 1, i])) % (2 * np.pi)
            state[:, 1, i + 1] = (state[:, 1, i] + state[:, 0, i + 1]) % (2 * np.pi)
        # Store the run
        run = {"K": self.K, "nIters": self.nIters, "seed": self.seed, "run": state}        
        if option == "append":
            self.runs.append(run)
        elif option == "overwrite":
            self.runs[-1] = run
        else:
            raise # Appropriate error here

# Function: get and set K
# Implement as callable

# Function: get and set nIters
# Implement as callable

# Function: get and set seed
# Implement as callable

# Function: metadata (ALL kwargs)
# Returns the number of runs and the range of K -- default
# Returns the K, initial condition, and length of ith run in list
# (Implement checks for i): keyword is "run" {values start at 1}
# i can also be a range (two element array-like), inclusive
# Must perform sanity checks for i[0] and i[1]
# Returns the indices of runs with a given K, same metadata as "run"
# (same K checks): return "None" if not found, keyword is "K"
# K can also be a range (two element array-like), check is inclusive
# Must perform sanity checks for K[0] and K[1]
# Returns the indices of runs with a given length, same metadata as "run"
# (same nIters checks): return "None" if not found, keyword is "N"
# nIters can also be a range (two element array-like), check is inclusive
# Must perform sanity checks for nIters[0] and nIters[1]

# Redundancy: __str__
# Returns the number of runs and the range of K

# Function: clearRuns
# does not remove current values of K/nIter/seed in object
# removes all runs from history -- default
# removes ith runs (Implement checks for i): keyword is "run"
# i can also be a range (two element array-like), inclusive
# Must perform sanity checks for i[0] and i[1]
# removes runs with given K (Implement checks for K): keyword is "K"
# K can also be a range (two element array-like), inclusive
# Must perform sanity checks for K[0] and K[1]

# Function: write to CSV
# Option to select ith array to write
# i can be a range
# Option to write ALL arrays to CSVs -- default
# Supply kwargs to savetxt function
# Reserve kwarg "name" as filename (can be path)
# Default has header txt of "K = [val]"
# Default saves I then theta w/ col headers "I,theta"
# Default names file "[K]-[val]-len-[nIters].csv"
# Default adds " ([number])" if filename taken
# Default save location is "results/csvs"
