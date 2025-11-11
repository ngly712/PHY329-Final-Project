# Analyzing the Chaotic Behavior of the Chirikov Map
The [Taylor-Greene-Chirikov Map](https://mathworld.wolfram.com/StandardMap.html), or Standard Map, is a two-dimensional discrete-time recurrence relation that exhibits chaotic behavior. The system is as follows:

> <p align="center">
> $I_{n+1}=I_n+K\sin{\theta_n}$
> </p>
> <p align="center">
> $\theta_{n+1}=\theta_n+I_{n+1}$
> </p>

$I$ and $\theta$ are periodic real-valued variables within $\[0, 2\pi\]$, while $K$ is a positive real number. The exact value of $K$ that results in chaotic behavior is not known, but several papers ([here](https://arxiv.org/pdf/2509.11593) and [here](https://pubs.aip.org/aip/jmp/article-abstract/20/6/1183/449401/A-method-for-determining-a-stochastic-transition?redirectedFrom=fulltext), for example) have attempted to identify a reasonable bound for the coefficient.

## Our Contributions
To experimentally determine the onset of chaos, we will implement a collection of data analysis scripts that act upon a Standard Map class instance to extract the value of $K$. These will produce a bifurcation diagram, Poincar&eacute; plots, and phase space maps that demonstrate the formation of periodic islands littered in a dense mapping. If time permits, we will expand the model to a classical [kicked rotator](https://www.sciencedirect.com/science/article/pii/S0960077905005485?via%3Dihub) system upon which the Standard map is derived.

# Code Structure
`map` folder:
- `standardMap.py` will contain the class implementation of the standard map with a function to iterate from an initial condition

`plots` folder:
- `mapEval.py` will contain the class implementation for evaluating different aspects of the batch of standard map runs
- `mapPlot.py` will contain the plotting function for the phase space plots and bifurcation diagrams
- This folder will also contain plots for different values of $K$ in labeled subfolders.

`results` folder:
- These will store the raw arrays for different $K$ values as well as any miscellaneous data used in the plotting (subfolders expected).

` unitTests` folder:
- These will contain scripts to check the implementation of the Standard Map class. Useful for anyone who wants to modify it and see if their changes work.

`map.ipynb` will contain the top-level report on our results

# Planned Contributions
- Nguyen: writing the `standardMap.py` class structure, utility functions, and unit tests, helping with the final `map.ipynb` report
- Hazel: writing the final `map.ipynb` report, helping with the `standardMap.py` class structure
- Enrique: writing the `mapEval.py` class structure, utility functions, and unit tests, helping with the `mapPlot.py` script
- Carlos: writing the `mapPlot.py` script, helping with the `mapEval.py` class structure

## Timeline
1. Implement the `standardMap.py` class structure, utility functions, and unit tests
2. Implement the `mapEval.py` class structure, utility functions, and unit tests
3. Implement the `mapPlot.py` functions and generate the required plots
4. Generate the final report in `map.ipynb`.

Optional:
5. Implement the kicked rotator as a separate class and make the Standard Map a subclass of it.
6. Modify `mapEval.py` to account for additional variables like the kick strength and duration.
7. Perform a similar analysis of the chaotic behavior for the kicked rotator.
8. Add results to `map.ipynb`.
