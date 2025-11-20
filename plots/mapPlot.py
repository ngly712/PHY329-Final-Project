def plot_phase_generic(run, mode="phase", traj_index=0, point_size=0.1, title=None):
    
    # Plots phase-space data from a standardMap run.
    # -------------------------------------------------------------------
    # Parameters:
    # run: np.ndarray
    #     run belonging to an instance of the StandardMap class
    # mode: {"phase", "poincare"}
    #     if user sets mode = "phase", this function plots a phase space plot of all trajectories in the run (many ICs).
    #     if user sets mode = "poincare", this function plots a single chosen trajectory of the run (Poincaré section).
    # traj_index: int
    #     Index of trajectory to use in "poincare" mode.
    # point_size: float
    #     Matplotlib scatter marker size.
    # title: str or None
    #     Optional figure title. Defaults to "Phase Space Plot" for mode = 'phase' and 
    #     "Poincaré Section (trajectory {traj_index})" for mode = 'poincare'
    # -------------------------------------------------------------------

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
        rand_colors = np.random.choice(len(color_choices), size=nSim)

        for k in range(nSim):
            color = color_choices[rand_colors[k]]
            theta_vals = run[k, 1,:]
            I_vals = run[k, 0,:]
            plt.scatter(theta_vals, I_vals, s=point_size, color=color)

    else: 
        theta_traj = run[traj_index, 1,:]
        I_traj = run[traj_index, 0,:]
        plt.scatter(theta_traj, I_traj, s=point_size, color="black")

    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$I$")
    plt.title(title)
    plt.xlim(0, 2*np.pi)
    plt.ylim(0, 2*np.pi)
    plt.tight_layout()
    plt.show()


def compute_bifurcation(Ks, ic, nIters=2000, burn_in=500, seed=None):
    
    # Helper function for the plot_bifurcation function.
    # This function computes bifurcation data (I values and corresponding 
    # k values) for a StandardMap instance. 
    # Parameters:
    # ---------------------------------------------------------------------
    # Ks: np.ndarray
    #     Sequence of K values to sweep over.
    # ic: np.ndarray
    #     Initial condition array of shape (1, 2).
    # nIters: int
    #     Total iterations per K. defaults to 2000
    # burn_in: int
    #     Number of initial steps to discard.
    # seed: int or None
    # ---------------------------------------------------------------------
    # Returns:
    # bifK : np.ndarray
    #     K values repeated for each late-time point I_n.
    # bifI : np.ndarray
    #     Corresponding I_n values.
   
    bifK = []
    bifI = []

    m = standardMap(K=0.0, nIters=nIters, seed=seed)

    for K in Ks:
        m.K = K
        m.simulate(ic=ic)

        run = m.runs[-1]["run"]  
        I_traj = run[0, 0, :]        

        I_asym = I_traj[burn_in:]

        bifK.extend([K] * len(I_asym))
        bifI.extend(I_asym)

    return np.array(bifK), np.array(bifI)


def plot_bifurcation(Ks, ic, nIters=2000, burn_in=500,
                     K_min=None, K_max=None,
                     title="Bifurcation Diagram of the Standard Map"):
    
    # k_min and k_max set the range for K on the bifurcation plot. Both default to None.

    bifK, bifI = compute_bifurcation(Ks, ic, nIters=nIters,
                                     burn_in=burn_in )

    
    if K_min is not None or K_max is not None:
        if K_min is None:
            K_min = np.min(Ks)
        if K_max is None:
            K_max = np.max(Ks)
        mask = (bifK >= K_min) & (bifK <= K_max)
        bifK = bifK[mask]
        bifI = bifI[mask]

    plt.figure(figsize=(7, 5))
    plt.scatter(bifK, bifI, s=0.1, color="black")
    plt.xlabel(r"$K$")
    plt.ylabel(r"$I_n$ (late-time)")
    plt.title(title)
    plt.tight_layout()
    plt.show()

                       
