import numpy as np
import os
import re


class StandardMap:
    """
    A class to store the trajectories of the Chirikov-Taylor Map. Available to use from
    a command line interface.

    Attributes
    ----------
    K : float
        A nonnegative "kick value" that provides an angular momentum boost. See
        `simulate` for more details.
    nIters : int
        The number of iterations to travel through. Must be positive.
    seed : int or None
        Used to store information about NumPy's random number generator.
    runs : list of dict
        All previous simulations are stored here with their respective parameters.

    Methods
    -------
        simulate(option="append", ic=1)
            Iterates through the map from an initial condition or a batch of initial
            conditions.
        metadata(**options)
            Returns information about runs or a list of indices satisfying a search
            term.
        clearRuns(**options)
            Removes specified runs from the `runs` list.
        write(**options)
            Writes the specified runs to a `.csv` file using NumPy's `savetxt` function.
        read(fname, **options)
            Reads in `.csv` files to store in the `runs` list.
    """

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

    def simulate(self, option: str = "append", ic: np.ndarray | int = 1) -> None:
        """
        Simulates the Chirikov-Taylor map from a given initial condition and adds the
        result to `runs`.

        Parameters
        ----------
        option : str
            How the simulation is added to `runs`. `"append"` (default) adds the
                simulation to the end of the list, while `"overwrite"` replaces the
                last run with the current simulation.
        ic : ndarray or int
            The initial conditions for `I` and `theta`.
            - Passing an integer will result
                in a batch of `ic` (I, theta) pairs iterated simultaneously, where the
                values are drawn from a uniform distribution in [0, 2 pi].
            - Passing an ndarray of shape (number of points, 2), where the first column
                are the I values and the second column are the theta values, will start
                the simulation at the specified points.

        Notes
        -----
        The Chirikov-Taylor map is given by the following recurrence relation:
        .. math::
            I_{n+1}=I_n+K\sin(\omega_n)\mod2\pi
            \omega_{n+1}=\omega_n+I_{n+1}\mod2\pi

        Each run in `runs` is a dictionary listing:
        - the seed of the random number generator (`"seed"`)
        - the time trajectories of the simulation (`"run"`)
        - the number of iterations in the simulation (`"nIters"`)
        - the "kick value" of the simulation (`"K"`)
        - the number of initial conditions (`"nSim"`)
        Use the keys above to access the desired run in the list.

        Examples
        --------
        First, instantiate the object:

        >>> import numpy as np
        >>> from map.standardMap import StandardMap as sMap
        >>> obj = sMap()

        A default simulation:

        >>> obj.simulate()
        >>> print(obj.runs[-1])
        {'K': 1.0, 'nIters': 500, 'seed': None, 'run': array([[[0.99831564, 0.8406607
            , 1.47127561, ..., 5.38026427, 4.54137696, 4.14777316], [6.12486987,
            0.68234526, 2.15362087, ..., 5.28794956, 3.54614121, 1.41072907]]], shape=
            (1, 2, 501)), 'nSim': 1}

        Simulate four trajectories at a time:

        >>> obj.simulate(ic=4)
        >>> print(obj.runs[-1]["run"].shape)
        (4, 2, 501)

        Simulate from a specified initial condition:
        >>> ic = np.array([[0.0, np.pi / 2], [np.pi, 3 * np.pi / 2], [2 * np.pi, 0.75]])
        >>> obj.simulate(ic=ic)
        >>> print(obj.runs[-1]["run"][:, :, 0])
        [[0.0       1.57079633]
        [3.14159265 4.71238898]
        [6.28318531 0.75      ]]

        Replace the previous run when simulating:
        >>> print(obj.runs[-1]["nSim"])
        3
        >>> print(len(obj.runs))
        3
        >>> obj.simulate(ic=5, option="overwrite")
        >>> print(obj.runs[-1]["nSim"])
        5
        >>> print(len(obj.runs))
        3
        """
        # Initialize the array
        if isinstance(ic, int):
            assert ic > 0
            state = np.zeros((ic, 2, self._nIters + 1))
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
            state = np.zeros((len(ic), 2, self._nIters + 1))
            state[..., 0] = ic
        else:
            raise Exception('"ic" must be an integer or a batch of initial values.')
        # Run the map
        for i in range(self._nIters):
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
    def K(self) -> float:
        return self._K

    @K.setter
    def K(self, K: float) -> None:
        assert K >= 0
        self._K = float(K)

    # Function: get and set nIters
    # Implement as callable
    @property
    def nIters(self) -> int:
        return self._nIters

    @nIters.setter
    def nIters(self, nIters: int) -> None:
        assert nIters > 0
        self._nIters = int(nIters)

    # Function: get and set seed
    # Implement as callable
    @property
    def seed(self) -> int:
        return self._seed

    @seed.setter
    def seed(self, seed) -> None:
        self._seed = seed

    # Function: metadata (ALL kwargs)
    # Returns the number of runs, range of K, and range of run lengths -- default
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
    def metadata(self, **options) -> dict | list[dict,] | list[int,]:
        # Check for keyword correctness
        assert len(options) < 2
        ind = []
        # Default info
        if len(options) < 1:
            if len(self.runs) > 0:
                minK = self.runs[0]["K"]
                maxK = self.runs[0]["K"]
                minN = self.runs[0]["nIters"]
                maxN = self.runs[0]["nIters"]
                minI = np.min(self.runs[0]["run"][:, 0, 0])
                maxI = np.max(self.runs[0]["run"][:, 0, 0])
                minT = np.min(self.runs[0]["run"][:, 1, 0])
                maxT = np.max(self.runs[0]["run"][:, 1, 0])
                for run in self.runs:
                    if run["K"] < minK:
                        minK = run["K"]
                    if run["K"] > maxK:
                        maxK = run["K"]
                    if run["nIters"] < minN:
                        minN = run["nIters"]
                    if run["nIters"] > maxN:
                        maxN = run["nIters"]
                    check = np.min(run["run"][:, 0, 0])
                    if check < minI:
                        minI = check
                    check = np.max(run["run"][:, 0, 0])
                    if check > maxI:
                        maxI = check
                    check = np.min(run["run"][:, 1, 0])
                    if check < minT:
                        minT = check
                    check = np.max(run["run"][:, 1, 0])
                    if check > maxT:
                        maxT = check
                return {
                    "runCount": len(self.runs),
                    "K": (minK, maxK),
                    "nIters": (minN, maxN),
                    "I_0": (minI, maxI),
                    "theta_0": (minT, maxT),
                }
            return {"runCount": len(self.runs), "K": self.K, "nIters": self.nIters}
        # List index info
        if "run" in options:
            # Single run info
            if isinstance(options["run"], int):
                return {
                    "K": self.runs[options["run"]]["K"],
                    "nIters": self.runs[options["run"]]["nIters"],
                    "IC": self.runs[options["run"]]["run"][:, :, 0],
                }
            # Info for range of runs
            elif isinstance(options["run"], tuple):
                runs = []
                assert len(options["run"]) == 2
                for i in options["run"]:
                    assert isinstance(i, int)
                assert options["run"][0] < options["run"][1]
                assert (
                    options["run"][0] > -1 and options["run"][1] < len(self.runs)
                ) or (options["run"][0] > -len(self.runs) and options["run"][1] < 0)
                for i in range(options["run"][0], options["run"][1] + 1):
                    run = {
                        "K": self.runs[i]["K"],
                        "nIters": self.runs[i]["nIters"],
                        "IC": self.runs[i]["run"][:, :, 0],
                    }
                    runs.append(run)
                return runs
            else:
                raise Exception(
                    "Only single values or an ascending tuple of two inclusive bounds are allowed."
                )
        # K parameter searching
        elif "K" in options:
            # Single K search
            if isinstance(options["K"], float):
                assert options["K"] >= 0.0
                for i in range(len(self.runs)):
                    if self.runs[i]["K"] == options["K"]:
                        ind.append(i)
                if len(ind) == 0:
                    print(f"No runs with K = {options["K"]} found.")
                return ind
            # Range of K search
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
                if len(ind) == 0:
                    print(
                        f"No runs within K = [{options["K"][0]}, {options["K"][1]}] found."
                    )
                return ind
            else:
                raise Exception(
                    "Only single values or an ascending tuple of two inclusive bounds are allowed."
                )
        # Run length search
        elif "N" in options:
            # Single length search
            if isinstance(options["N"], int):
                assert options["N"] > 0
                for i in range(len(self.runs)):
                    if self.runs[i]["nIters"] == options["N"]:
                        ind.append(i)
                if len(ind) == 0:
                    print(f"No runs of length {options["N"]} found.")
                return ind
            # Range of length search
            elif isinstance(options["N"], tuple):
                assert len(options["N"]) == 2
                for i in options["N"]:
                    assert i > 0
                    assert isinstance(i, int)
                assert options["N"][0] < options["N"][1]
                for i in range(len(self.runs)):
                    if (
                        self.runs[i]["nIters"] >= options["N"][0]
                        and self.runs[i]["nIters"] <= options["N"][1]
                    ):
                        ind.append(i)
                if len(ind) == 0:
                    print(
                        f"No runs from length {options["N"][0]} to {options["N"][1]} found."
                    )
                return ind
            else:
                raise Exception(
                    "Only single values or an ascending tuple of two inclusive bounds are allowed."
                )
        else:
            raise Exception("Only run index, K search, and run length supported.")

    # Redundancy: __str__
    # Prints the number of runs and the range of K
    def __str__(self) -> str:
        info = self.metadata()
        if len(self.runs) > 0:
            return (
                f"Number of runs: {info['runCount']}\n"
                f"Range of K: [{info['K'][0]}, {info['K'][1]}]\n"
                f"Range of lengths: [{info['nIters'][0]}, {info['nIters'][1]}]\n"
                f"Range of I(0): [{info['I_0'][0]}, {info['I_0'][1]}]\n"
                f"Range of theta(0): [{info['theta_0'][0]}, {info['theta_0'][1]}]"
            )
        return (
            f"Current K: {info["K"]}\n"
            f"Current run length: {info["nIters"]}\n"
            f"No runs yet."
        )

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
        # K parameter and run length clearing
        elif "K" in options or "N" in options:
            ind = self.metadata(**options)
            for i in sorted(ind, reverse=True):
                print(f"Run {i} cleared.")
                del self.runs[i]
        else:
            raise Exception("Only run index, K search, and run length supported.")

    # Function: write to CSV
    # Option to select ith array to write - keyword is "run"
    # i can be a range or a list (from metadata)
    # Option to write ALL arrays to CSVs -- default
    # Supply kwargs to savetxt function
    # Reserve kwarg "name" as filename (do not add .csv)
    # Default has header txt of "K = [val]\nseed = [val]"
    # Default saves I then theta w/ col headers "I,theta"
    # Default names file "K-[val]-len-[nIters].csv"
    # Default adds " ([number])" if filename taken
    # Default save location is "results/csvs"
    def write(self, **options) -> None:
        # Helper for duplicate files
        def _fileCount(name: str, path: os.PathLike) -> int:
            files = os.listdir(path)
            count = 0
            for file in files:
                match = re.match(rf"{re.escape(name)}_(\d+)\.csv$", file)
                if match:
                    count += 1
            return count

        # Check how many to save
        if "run" in options:
            i = options.pop("run")
            if isinstance(i, int):
                ind = [i]
            elif isinstance(i, tuple):
                assert len(i) == 2
                for j in i:
                    assert isinstance(j, int)
                assert i[0] < i[1]
                assert (i[0] > -1 and i[1] < len(self.runs)) or (
                    i[0] > -len(self.runs) and i[1] < 0
                )
                ind = [*range(i[0], i[1] + 1)]
            elif isinstance(i, list):
                assert len(i) > 0
                seen = set()
                rep = 0
                for j in i:
                    assert isinstance(j, int)
                    assert j > -1 and j < len(self.runs)
                    if j in seen:
                        rep += 1
                    else:
                        seen.add(j)
                assert rep < 1
                ind = i
            else:
                raise Exception(
                    "Only single values, an ascending tuple of two inclusive bounds, or a list of nonnegative indices are allowed."
                )
        else:
            if len(self.runs) > 0:
                ind = [*range(len(self.runs))]
            else:
                print("No runs to export yet.")
                return
        # Check for given names
        if "name" in options:
            fname = options.pop("name")
            if len(ind) < 2:
                assert isinstance(fname, str)
                rep = _fileCount(fname, "results\\csvs")
                if rep > 0:
                    fname += f"_{rep}"
                fname = ["results\\csvs\\" + fname + ".csv"]
            else:
                assert isinstance(fname, list)
                assert len(fname) == len(ind)
                for i, name in enumerate(fname):
                    assert isinstance(name, str)
                    rep = _fileCount(name, "results\\csvs")
                    if rep > 0:
                        fname[i] += f"_{rep}"
                    fname[i] = "results\\csvs\\" + name + ".csv"
        else:
            fname = []
            for i, j in enumerate(ind):
                run = self.runs[j]
                fname.append(f"K-{run["K"]}-len-{run["nIters"]}")
                rep = _fileCount(fname[-1], "results\\csvs")
                if rep > 0:
                    fname[-1] += f"_{rep}"
                fname[-1] = "results\\csvs\\" + fname[-1] + ".csv"
        # Saving using savetxt
        for i, j in enumerate(ind):
            run = self.runs[j]
            arr = run["run"].reshape((run["nSim"] * 2, run["nIters"] + 1))
            htxt = ",".join(["I,theta"] * run["nSim"])
            if "fmt" not in options:
                options["fmt"] = "%.10f"
            if "delimiter" not in options:
                options["delimiter"] = "\t"
            if "header" not in options:
                options["header"] = f"K = {run["K"]}\nseed = {run["seed"]}\n{htxt}"
            if "comments" not in options:
                options["comments"] = ""
            np.savetxt(fname[i], arr.T, **options)

    # Function: read CSV and add to runs
    # File name is name without directory or .csv tag
    # Option to append (default) or replace last ith runs
    # File must be in the same format as produced by "write"
    # Pass kwargs to loadtxt function
    def read(self, fname: str | list[str,], insert: str = "append", **options) -> None:
        if "delimiter" not in options:
            options["delimiter"] = "\t"
        if "skiprows" not in options:
            options["skiprows"] = 3
        # Read files
        runs = []
        if isinstance(fname, str):
            file = open(f"results\\csvs\\{fname}.csv")
            kVal = float(file.readline().split(" ")[-1])
            seedVal = file.readline().split(" ")[-1]
            seedVal = seedVal.replace("\n", "")
            if seedVal != "None":
                seedVal = int(seedVal)
            else:
                seedVal = None
            arr = np.loadtxt(
                f"results\\csvs\\{fname}.csv",
                **options,
            )
            run = {
                "K": kVal,
                "nIters": arr.shape[0] - 1,
                "seed": seedVal,
                "run": arr.T.reshape((int(arr.shape[1] / 2), 2, arr.shape[0])),
                "nSim": arr.shape[1] / 2,
            }
            runs.append(run)
        elif isinstance(fname, list):
            for name in fname:
                assert isinstance(name, str)
                file = open(f"results\\csvs\\{name}.csv")
                kVal = float(file.readline().split(" ")[-1])
                seedVal = int(file.readline().split(" ")[-1])
                arr = np.loadtxt(
                    f"results\\csvs\\{name}.csv",
                    **options,
                )
                run = {
                    "K": kVal,
                    "nIters": arr.shape[0] - 1,
                    "seed": seedVal,
                    "run": arr.T.reshape((arr.shape[1] / 2, 2, arr.shape[0])),
                    "nSim": arr.shape[1] / 2,
                }
                runs.append(run)
        else:
            raise Exception("Only a file name or a list of file names are allowed.")
        # Add to current object
        if insert == "append":
            self.runs.extend(runs)
        elif insert == "overwrite":
            self.runs[-len(runs) :] = runs
        else:
            raise Exception(
                'Invalid option. Only "append" and "overwrite" are allowed.'
            )
