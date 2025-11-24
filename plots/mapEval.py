import numpy as np


class MapEvaluator:
    """
    Helper class for analyzing batches of runs of the Standard (Chirikov) Map.

    Parameters
    ----------
    runs : list of dict
        Typically ``aMap.runs`` from a ``standardMap`` instance.
        Each dict is expected to contain at least the keys
        ``"K"`` (float) and ``"run"`` (np.ndarray of shape (nSim, 2, nIters)).
    """

    def __init__(self, runs):
        self.runs = runs

    def _checkRunIdx(self, run_idx: int) -> None:
        """Internal helper: validate run_idx."""
        assert isinstance(run_idx, int), "run_idx must be an integer."
        assert len(self.runs) > 0, "No runs are available in self.runs."
        assert 0 <= run_idx < len(self.runs), (
            f"run_idx must satisfy 0 <= run_idx < {len(self.runs)}; "
            f"got run_idx = {run_idx}."
        )

    def _checkNTail(self, n_tail: int, n_iters: int) -> None:
        """Internal helper: validate n_tail against n_iters."""
        assert isinstance(n_tail, int), "n_tail must be an integer."
        assert 1 <= n_tail <= n_iters, (
            f"n_tail must satisfy 1 <= n_tail <= {n_iters}; "
            f"got n_tail = {n_tail}."
        )

    def getKickValues(self):
        """
        Return the kick strength K for each run.

        Returns
        -------
        K_values : np.ndarray, shape (nRuns,)
            One-dimensional array containing the K value stored in each run
            in ``self.runs``, in order.
        """
        return np.array([run["K"] for run in self.runs])

    def getTheta(self, run_idx: int):
        """
        Return the theta array for a single run.

        Parameters
        ----------
        run_idx : int
            Index of the run in ``self.runs``.
            Must satisfy ``0 <= run_idx < len(self.runs)``.

        Returns
        -------
        theta : np.ndarray, shape (nSim, nIters)
            Theta values for all trajectories and all time steps in the
            selected run. ``nSim`` is the number of initial conditions
            simulated in that run; ``nIters`` is the number of iterations.
        """
        self._checkRunIdx(run_idx)
        sim_data = self.runs[run_idx]["run"]   # shape (nSim, 2, nIters)
        theta = sim_data[:, 1, :]              # pick theta component
        return theta

    def getI(self, run_idx: int):
        """
        Return the I (momentum) array for a single run.

        Parameters
        ----------
        run_idx : int
            Index of the run in ``self.runs``.
            Must satisfy ``0 <= run_idx < len(self.runs)``.

        Returns
        -------
        I_vals : np.ndarray, shape (nSim, nIters)
            Momentum values for all trajectories and all time steps in the
            selected run. ``nSim`` is the number of initial conditions
            simulated in that run; ``nIters`` is the number of iterations.
        """
        self._checkRunIdx(run_idx)
        sim_data = self.runs[run_idx]["run"]   # shape (nSim, 2, nIters)
        I_vals = sim_data[:, 0, :]             # pick I component
        return I_vals

    def thetaTail(self, run_idx: int, n_tail: int):
        """
        Return the last n_tail theta values for a single run.

        Parameters
        ----------
        run_idx : int
            Index of the run in ``self.runs``.
            Must satisfy ``0 <= run_idx < len(self.runs)``.
        n_tail : int
            Number of final iterations to extract. Must satisfy
            ``1 <= n_tail <= nIters`` for the selected run.

        Returns
        -------
        theta_tail : np.ndarray, shape (nSim, n_tail)
            Array containing the last ``n_tail`` theta values for each
            trajectory in the selected run.
        """
        self._checkRunIdx(run_idx)
        theta = self.getTheta(run_idx)   # (nSim, nIters)
        n_iters = theta.shape[1]
        self._checkNTail(n_tail, n_iters)
        return theta[:, -n_tail:]

    def ITail(self, run_idx: int, n_tail: int):
        """
        Return the last n_tail I (momentum) values for a single run.

        Parameters
        ----------
        run_idx : int
            Index of the run in ``self.runs``.
            Must satisfy ``0 <= run_idx < len(self.runs)``.
        n_tail : int
            Number of final iterations to extract. Must satisfy
            ``1 <= n_tail <= nIters`` for the selected run.

        Returns
        -------
        I_tail : np.ndarray, shape (nSim, n_tail)
            Array containing the last ``n_tail`` I values for each trajectory
            in the selected run. ``nSim`` is the number of initial conditions
            simulated in that run.
        """
        self._checkRunIdx(run_idx)
        I_vals = self.getI(run_idx)      # (nSim, nIters)
        n_iters = I_vals.shape[1]
        self._checkNTail(n_tail, n_iters)
        return I_vals[:, -n_tail:]

    def thetaBifData(self, n_tail: int = 100):
        """
        Collect (K, theta) pairs for a theta-K bifurcation diagram.

        For each run, this method takes the last ``n_tail`` theta values of
        every trajectory, then flattens and pairs them with the corresponding
        K value for that run.

        Parameters
        ----------
        n_tail : int, optional
            Number of final iterations to use from each trajectory.
            Default is 100.

        Returns
        -------
        K_vals : np.ndarray, shape (N_points,)
            One-dimensional array of K values, with entries repeated so that
            each theta point has a matching K.
        theta_vals : np.ndarray, shape (N_points,)
            One-dimensional array of theta values taken from the tails of all
            runs and all trajectories. ``N_points`` is the total number of
            plotted points, i.e. the sum over all runs of
            ``nSim * n_tail`` for that run.

        Notes
        -----
        If ``self.runs`` is empty, both ``K_vals`` and ``theta_vals`` are
        returned as empty arrays.
        """
        assert isinstance(n_tail, int), "n_tail must be an integer."
        assert n_tail > 0, "n_tail must be positive."

        K_list = []
        theta_list = []

        for run_idx, run in enumerate(self.runs):
            K = run["K"]
            theta_tail = self.thetaTail(run_idx, n_tail)  # (nSim, n_tail)

            # Flatten to 1D: all sims, all tail times
            theta_flat = theta_tail.ravel()

            # Make a matching array of K values
            K_flat = np.full(theta_flat.shape, K, dtype=float)

            theta_list.append(theta_flat)
            K_list.append(K_flat)

        if not K_list:
            return np.array([]), np.array([])

        K_vals = np.concatenate(K_list)
        theta_vals = np.concatenate(theta_list)

        return K_vals, theta_vals

    def IBifData(self, n_tail: int = 100):
        """
        Collect (K, I) pairs for an I-K bifurcation diagram.

        For each run, this method takes the last ``n_tail`` I values of every
        trajectory, then flattens and pairs them with the corresponding K
        value for that run.

        Parameters
        ----------
        n_tail : int, optional
            Number of final iterations to use from each trajectory.
            Default is 100.

        Returns
        -------
        K_vals : np.ndarray, shape (N_points,)
            One-dimensional array of K values, with entries repeated so that
            each I point has a matching K.
        I_vals : np.ndarray, shape (N_points,)
            One-dimensional array of I values taken from the tails of all
            runs and all trajectories. ``N_points`` is the total number of
            plotted points, i.e. the sum over all runs of
            ``nSim * n_tail`` for that run.

        Notes
        -----
        If ``self.runs`` is empty, both ``K_vals`` and ``I_vals`` are returned
        as empty arrays.
        """
        assert isinstance(n_tail, int), "n_tail must be an integer."
        assert n_tail > 0, "n_tail must be positive."

        K_list = []
        I_list = []

        for run_idx, run in enumerate(self.runs):
            K = run["K"]
            I_tail = self.ITail(run_idx, n_tail)  # (nSim, n_tail)

            # Flatten to 1D: all sims, all tail times
            I_flat = I_tail.ravel()

            # Make a matching array of K values
            K_flat = np.full(I_flat.shape, K, dtype=float)

            I_list.append(I_flat)
            K_list.append(K_flat)

        if not K_list:
            return np.array([]), np.array([])

        K_vals = np.concatenate(K_list)
        I_vals = np.concatenate(I_list)

        return K_vals, I_vals

    def phaseSpaceData(self, run_idx: int, n_tail: int = 100):
        """
        Return I and theta values for a phase-space plot of one run.

        Parameters
        ----------
        run_idx : int
            Index of the run in ``self.runs``.
            Must satisfy ``0 <= run_idx < len(self.runs)``.
        n_tail : int, optional
            Number of final iterations to extract for each trajectory.
            Must satisfy ``1 <= n_tail <= nIters``. Default is 100.

        Returns
        -------
        I_vals : np.ndarray, shape (N_points,)
            Flattened I values from the last ``n_tail`` iterations of all
            trajectories in the selected run. ``N_points = nSim * n_tail``.
        theta_vals : np.ndarray, shape (N_points,)
            Flattened theta values from the last ``n_tail`` iterations of all
            trajectories in the selected run.

        Notes
        -----
        This method is intended for generating (theta, I) scatter plots
        (phase-space diagrams) for a single run.
        """
        self._checkRunIdx(run_idx)

        theta = self.getTheta(run_idx)
        n_iters = theta.shape[1]
        self._checkNTail(n_tail, n_iters)

        I_tail = self.getI(run_idx)[:, -n_tail:]
        theta_tail = theta[:, -n_tail:]
        return I_tail.ravel(), theta_tail.ravel()