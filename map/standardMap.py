# Planned imports
# numpy
import numpy as np


# Class (no inheritance)
# standardMap
class StandardMap:

    # Initialization
    # User can set K, nIters, random seed
    # K is a positive real number (checks) -- defaults to 1
    # nIters is a positive integer (checks) -- defaults to 500
    # seed is a positive integer w/ 0 (checks) -- defaults to None
    # Object stores list of dicts w/ K, nIters, seed, and batched arrays for I
    # and theta
    # Call the list "runs"
    # Initialization makes an empty list
    def __init__(self, K: float = 1.0, nIters: int = 500, seed=None) -> None:
        # Checks for K and nIters
        assert K >= 0
        assert nIters > 0
        # Add to class
        self._K = float(K)
        self._nIters = int(nIters)
        self._seed = seed
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
            state = np.zeros((ic, 2, self._nIters))
            np.random.seed(self._seed)
            state[..., 0] = np.random.uniform(
                0,
                2 * np.pi,
                (ic, 2),
            )
        elif isinstance(ic, np.ndarray):
            assert ic.shape[1] == 2
            assert ic.shape[0] > 0
            assert np.min(ic) >= 0
            assert np.max(ic) <= 2 * np.pi
            state = np.zeros((len(ic), 2, self._nIters))
            state[..., 0] = ic
        else:
            raise Exception("ic must be an integer or a batch of initial values.")
        # Run the map
        for i in range(self._nIters - 1):
            state[:, 0, i + 1] = (state[:, 0, i] + self._K * np.sin(state[:, 1, i])) % (
                2 * np.pi
            )
            state[:, 1, i + 1] = (state[:, 1, i] + state[:, 0, i + 1]) % (2 * np.pi)
        # Store the run
        run = {
            "K": self._K,
            "nIters": self._nIters,
            "seed": self._seed,
            "run": state,
            "nSim": state.shape[0],
        }
        if option == "append":
            self.runs.append(run)
        elif option == "overwrite":
            self.runs[-1] = run
        else:
            raise Exception(
                'Invalid option. Only "append" and "overwrite" are allowed.'
            )

    # Function: get and set K
    # Implement as callable
    @property
    def K(self):
        return self._K

    @K.setter
    def K(self, K: float):
        assert K >= 0
        self._K = float(K)

    # Function: get and set nIters
    # Implement as callable
    @property
    def nIters(self):
        return self._nIters

    @nIters.setter
    def nIters(self, nIters: int):
        assert nIters > 0
        self._nIters = int(nIters)

    # Function: get and set seed
    # Implement as callable
    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed

    # Function: metadata (ALL kwargs)
    # Returns the number of runs and the range of K -- default
    # Returns the K, initial condition, and length of ith run in list
    # (Implement checks for i): keyword is "run" {values start at 1}
    # i can also be a range (two element tuple), inclusive
    # Must perform sanity checks for i[0] and i[1]
    # Returns the indices of runs with a given K, same metadata as "run"
    # (same K checks): return "None" if not found, keyword is "K"
    # K can also be a range (two element tuple), check is inclusive
    # Must perform sanity checks for K[0] and K[1]
    # Returns the indices of runs with a given length, same metadata as "run"
    # (same nIters checks): return "None" if not found, keyword is "N"
    # nIters can also be a range (two element tuple), check is inclusive
    # Must perform sanity checks for nIters[0] and nIters[1]

    # Redundancy: __str__
    # Returns the number of runs and the range of K
    def __str__(self) -> str:
        if len(self.runs) > 0:
            minK = self.runs[0]["K"]
            maxK = self.runs[0]["K"]
            minN = self.runs[0]["nIters"]
            maxN = self.runs[0]["nIters"]
            for run in self.runs:
                if run["K"] < minK:
                    minK = run["K"]
                if run["K"] > maxK:
                    maxK = run["K"]
                if run["nIters"] < minN:
                    minN = run["nIters"]
                if run["nIters"] > maxN:
                    maxN = run["nIters"]
            return f"Number of runs: {len(self.runs)}\nRange of K: [{minK}, {maxK}]\nRange of lengths: [{minN}, {maxN}]"
        return f"Current K: {self._K}\nCurrent run length: {self._nIters}\nNo runs yet."

    # Function: clearRuns
    # does not remove current values of K/nIter/seed in object
    # removes all runs from history -- default
    # removes ith runs (Implement checks for i): keyword is "run"
    # i can also be a range (two element tuple), inclusive
    # Must perform sanity checks for i[0] and i[1]
    # removes runs with given K (Implement checks for K): keyword is "K"
    # K can also be a range (two element tuple), inclusive
    # Must perform sanity checks for K[0] and K[1]
    # removes runs with given length (Implement checks for nIters): keyword is "N"
    # nIters can also be a range (two element tuple)
    def clearRuns(self, **options) -> None:
        # Check for keyword correctness
        assert len(options) < 2
        ind = []
        # Default clearing
        if len(options) < 1:
            self.runs.clear()
            print("All runs cleared.")
            return
        # List index clearing
        if "run" in options:
            # Single run clear
            if isinstance(options["run"], int):
                del self.runs[options["run"]]
                if options["run"] > -1:
                    print(f"Run {options["run"]} cleared.")
                else:
                    print(f"Run {len(self.runs) + 1 + options["run"]} cleared.")
            # Range of runs cleared
            elif isinstance(options["run"], tuple):
                assert len(options["run"]) == 2
                for i in options["run"]:
                    assert isinstance(i, int)
                assert options["run"][0] < options["run"][1]
                assert (
                    options["run"][0] > -1 and options["run"][1] < len(self.runs)
                ) or (options["run"][0] > -len(self.runs) and options["run"][1] < 0)
                del self.runs[options["run"][0] : options["run"][1] + 1]
                for i in range(options["run"][0], options["run"][1] + 1):
                    if i > -1:
                        print(f"Run {i} cleared.")
                    else:
                        print(f"Run {len(self.runs) + 1 + i} cleared.")
            else:
                raise Exception(
                    "Only single values or an ascending tuple of two inclusive bounds are allowed."
                )
        # K parameter clearing
        elif "K" in options:
            # Single K clear
            if isinstance(options["K"], float):
                assert options["K"] >= 0.0
                for i in range(len(self.runs)):
                    if self.runs[i]["K"] == options["K"]:
                        ind.append(i)
                        print(f"Run {i + 1} cleared.")
                if len(ind) == 0:
                    print(f"No runs with K = {options["K"]} found.")
                for j in ind:
                    del self.runs[j]
            # Range of Ks cleared
            elif isinstance(options["K"], tuple):
                assert len(options["K"]) == 2
                for i in options["K"]:
                    assert isinstance(i, float)
                    assert i >= 0.0
                assert options["K"][0] < options["K"][1]
                for i in range(len(self.runs)):
                    if (
                        self.runs[i]["K"] >= options["K"][0]
                        and self.runs[i]["K"] <= options["K"][1]
                    ):
                        ind.append(i)
                        print(f"Run {i + 1} cleared.")
                if len(ind) == 0:
                    print(
                        f"No runs within K = [{options["K"][0]}, {options["K"][1]}] found."
                    )
                for j in ind:
                    del self.runs[j]
            else:
                raise Exception(
                    "Only single values or an ascending tuple of two inclusive bounds are allowed."
                )
        # Run length clearing
        elif "N" in options:
            # Single length clear
            if isinstance(options["N"], int):
                assert options["N"] > 0
                for i in range(len(self.runs)):
                    if self.runs[i]["nIters"] == options["N"]:
                        ind.append(i)
                        print(f"Run {i + 1} cleared.")
                if len(ind) == 0:
                    print(f"No runs of length {options["N"]} found.")
                for j in ind:
                    del self.runs[j]
            # Range of lengths cleared
            elif isinstance(options["N"], tuple):
                assert len(options["N"]) == 2
                for i in options["N"]:
                    assert isinstance(i, float)
                assert options["N"][0] < options["N"][1]
                for i in range(len(self.runs)):
                    if (
                        self.runs[i]["nIters"] >= options["N"][0]
                        and self.runs[i]["nIters"] <= options["N"][1]
                    ):
                        ind.append(i)
                        print(f"Run {i + 1} cleared.")
                if len(ind) == 0:
                    print(
                        f"No runs from length {options["N"][0]} to {options["N"][1]} found."
                    )
                for j in ind:
                    del self.runs[j]
            else:
                raise Exception(
                    "Only single values or an ascending tuple of two inclusive bounds are allowed."
                )
        else:
            raise Exception("Only run index and K search supported.")

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
    def write(self):
        pass
