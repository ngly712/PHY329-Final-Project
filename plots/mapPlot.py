import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def plot_phase_generic(
    run: np.ndarray,
    mode: str = "phase",
    traj_index: int = 0,
    point_size: float = 0.1,
    title: Optional[str] = None,
) -> None:
    """
    Plot phase-space data from a single standard map run array.

    Parameters
    ----------
    run : np.ndarray
        Array of shape (nSim, 2, nIters) containing the simulation data for
        one run. The first index selects the trajectory, the second index
        selects the component (0 for I, 1 for theta), and the third index
        selects the iteration.
    mode : {"phase", "poincare"}, optional
        If "phase", plot all trajectories in the run (full phase-space
        evolution). If "poincare", plot a single chosen trajectory.
        Default is "phase".
    traj_index : int, optional
        Index of the trajectory to use when mode="poincare". Ignored for
        mode="phase". Default is 0.
    point_size : float, optional
        Marker size passed to ``plt.scatter``. Default is 0.1.
    title : str or None, optional
        Figure title. If None, a default title is chosen depending on mode.
    """
    nSim, _, nIters = run.shape

    if mode not in ("phase", "poincare"):
        raise ValueError('mode must be "phase" or "poincare"')

    if title is None:
        if mode == "phase":
            title = "Phase Space Plot"
        else:
            title = f"Poincaré Section (trajectory {traj_index})"

    plt.figure(figsize=(6, 6))

    if mode == "phase":
        color_choices = [
            "#1b1f3b",
            "#283655",
            "#4d648d",
            "#1e434c",
            "#2c7873",
            "#553d67",
            "#5d3a58",
            "#472d30",
        ]
        rand_colors = np.random.choice(a=len(color_choices), size=nSim)

        for k in range(nSim):
            color = color_choices[rand_colors[k]]
            theta_vals = run[k, 1, :]
            I_vals = run[k, 0, :]
            plt.scatter(x=theta_vals, y=I_vals, s=point_size, color=color)

    else:  # mode == "poincare"
        if not (0 <= traj_index < nSim):
            raise ValueError(
                f"traj_index must satisfy 0 <= traj_index < {nSim}; "
                f"got {traj_index}."
            )
        theta_traj = run[traj_index, 1, :]
        I_traj = run[traj_index, 0, :]
        plt.scatter(x=theta_traj, y=I_traj, s=point_size, color="black")

    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$I$")
    plt.title(title)
    plt.xlim(0, 2 * np.pi)
    plt.ylim(0, 2 * np.pi)
    plt.tight_layout()
    plt.show()


def plot_phase_tail(
    evaluator: "MapEvaluator",
    run_idx: int = 0,
    n_tail: int = 100,
    point_size: float = 0.1,
    title: str = "Phase Space (tail points)",
) -> None:
    """
    Plot a phase-space diagram using the tail data from a MapEvaluator.

    This function uses the last ``n_tail`` points from all trajectories in
    a single run and plots them as a scatter cloud in (theta, I) space.

    Parameters
    ----------
    evaluator : MapEvaluator
        Instance constructed from a list of standard map runs. Expected to
        provide a ``phaseSpaceData(run_idx, n_tail)`` method that returns
        flattened I and theta arrays.
    run_idx : int, optional
        Index of the run in the underlying runs list. Default is 0.
    n_tail : int, optional
        Number of final iterations to use from each trajectory. Default is 100.
    point_size : float, optional
        Marker size passed to ``plt.scatter``. Default is 0.1.
    title : str, optional
        Plot title. Default is "Phase Space (tail points)".
    """
    I_vals, theta_vals = evaluator.phaseSpaceData(
        run_idx=run_idx,
        n_tail=n_tail,
    )

    if I_vals.size == 0:
        print("No data available for the requested run; nothing to plot.")
        return

    plt.figure(figsize=(6, 6))
    plt.scatter(x=theta_vals, y=I_vals, s=point_size, color="black")
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$I$")
    plt.title(title)
    plt.xlim(0, 2 * np.pi)
    plt.ylim(0, 2 * np.pi)
    plt.tight_layout()
    plt.show()


def plot_IK_diagnostic(
    evaluator: "MapEvaluator",
    n_tail: int = 100,
    K_min: Optional[float] = None,
    K_max: Optional[float] = None,
    title: str = "I-K Diagnostic Plot",
    max_points: Optional[int] = None,
    point_size: float = 0.1,
    alpha: float = 0.3,
) -> None:
    """
    Plot an I-K diagnostic diagram using a MapEvaluator instance.
    It shows a sweep over K with late-time I_n values to
    visualize the breakdown of invariant structures and the onset of
    chaos.

    Parameters
    ----------
    evaluator : MapEvaluator
        Instance constructed from a list of standard map runs. Expected to
        provide an ``IKDiagnosticData(n_tail)`` method that returns
        (K_vals, I_vals).
    n_tail : int, optional
        Number of final iterations to use from each trajectory when generating
        the diagnostic data. Default is 100.
    K_min : float or None, optional
        Minimum K value to include in the plot. If None, no lower bound is
        applied. Default is None.
    K_max : float or None, optional
        Maximum K value to include in the plot. If None, no upper bound is
        applied. Default is None.
    title : str, optional
        Plot title. Default is "I-K Diagnostic Plot".
    max_points : int or None, optional
        If not None, randomly subsample to at most this many points for
        plotting. Useful to keep the figure readable and lightweight.
    point_size : float, optional
        Matplotlib scatter marker size. Default is 0.1.
    alpha : float, optional
        Marker transparency (0-1). Default is 0.3.
    """
    # Get flattened diagnostic data from the evaluator
    K_vals, I_vals = evaluator.IKDiagnosticData(n_tail=n_tail)

    if K_vals.size == 0:
        print("No runs available; nothing to plot.")
        return

    # Apply K-range mask if requested
    if K_min is not None or K_max is not None:
        if K_min is None:
            K_min = float(np.min(a=K_vals))
        if K_max is None:
            K_max = float(np.max(a=K_vals))
        mask = (K_vals >= K_min) & (K_vals <= K_max)
        K_vals = K_vals[mask]
        I_vals = I_vals[mask]

    # Optional subsampling for readability / file size
    if max_points is not None and K_vals.size > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(K_vals.size, size=max_points, replace=False)
        K_vals = K_vals[idx]
        I_vals = I_vals[idx]

    plt.figure(figsize=(7, 5))
    plt.scatter(
        x=K_vals,
        y=I_vals,
        s=point_size,
        alpha=alpha,
        color="black",
    )
    plt.xlabel(r"$K$")
    plt.ylabel(r"$I_n$ (late-time)")
    plt.title(title)
    plt.tight_layout()
    plt.show()
