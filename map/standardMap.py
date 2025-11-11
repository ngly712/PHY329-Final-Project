# Planned imports
# numpy

# Class (no inheritance)
# standardMap

# Initialization
# User can set K, nIters, random seed
# K is a positive real number (checks) -- defaults to 1
# nIters is a positive integer (checks) -- defaults to 500
# seed is a positive integer w/ 0 (checks) -- defaults to None
# Object stores list of dicts w/ K, nIters, seed, and initialized arrays for I
# and theta
# Call the list "runs"
# Initialization makes an empty list

# Function: simulate
# Option to append new run or overwrite -- default is append
# User supplies initial values for I and theta -- defaults to random (seeded
# or not)
# I and theta are limited to [0, 2pi] (checks)
# Overwrite replaces most recent run in list with dict entry
# Append creates new dict entry in list

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
# Returns the indices of runs with a given I_0, same metadata as "run"
# (same I_0 checks): return "None" if not found, keyword is "I_0"
# I_0 can also be a range (two element array-like), check is inclusive
# Must perform sanity checks for I_0[0] and I_0[1]
# Returns the indices of runs with a given theta_0, same metadata as "run"
# (same theta_0 checks): return "None" if not found, keyword is "theta_0"
# theta_0 can also be a range (two element array-like), check is inclusive
# Must perform sanity checks for theta_0[0] and theta_0[1]
# Returns the indices of runs with a given initial condition, same metadata as
# "run", must be at least two rows/columns
# (same ic checks): return "None" if not found, keyword is "I_0"
# I_0 can also be a range (two element array-like), check is inclusive
# Must perform sanity checks for I_0[0] and I_0[1]

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
