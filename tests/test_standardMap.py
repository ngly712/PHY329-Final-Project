# Unit tests using the pytest library
import numpy as np
from map.standardMap import StandardMap as sMap


# Planned tests:
# # Initialization:
# ## K
# ### default
# ### user
# ## nIters
# ### default
# ### user
# ## seed
# ### default
# ### user
# ## Check list properly initialized
def test_initialization_K():
    obj = sMap()
    assert obj.K == 1.0
    obj = sMap(K=0.75)
    assert obj.K == 0.75
    obj = sMap(K=3)
    assert obj.K == 3
    try:
        obj = sMap(K=-1)
    except Exception:
        print("Not possible")
    else:
        raise Exception("K is invalid")


def test_initialization_seed():
    obj = sMap()
    assert obj.seed is None
    obj = sMap(seed=42)
    assert obj.seed == 42


def test_initialization_nIters():
    obj = sMap()
    assert obj.nIters == 500
    obj = sMap(nIters=1)
    assert obj.nIters == 1
    obj = sMap(nIters=23.7)
    assert obj.nIters == 23
    try:
        obj = sMap(nIters=0)
    except Exception:
        print("Not possible")
    else:
        raise Exception("nIters is invalid")


# # Simulate:
# ## I_0
# ### default
# ### user
# ## theta_0
# ### default
# ### user
# ## option - append vs replace
# ### list length check
# ### run parameters check
def test_simulate_ic():
    objs = [
        sMap(),
        sMap(K=0.02),
        sMap(nIters=1200),
        sMap(K=0.02, nIters=1200),
    ]
    testIc = np.array([[0.0, np.pi / 2], [np.pi, 3 * np.pi / 2], [2 * np.pi, 0.75]])
    invalidIc = np.array([[1.1, 3.4, 2.7]])
    invalidIc2 = np.array(
        [[-1.0, np.pi / 2], [np.pi, 3 * np.pi / 2], [2 * np.pi, 0.75]]
    )
    for obj in objs:
        print(obj)
        obj.simulate()
        assert obj.runs[-1]["K"] == obj.K
        assert obj.runs[-1]["nIters"] == obj.nIters
        assert obj.runs[-1]["nSim"] == 1
        assert obj.runs[-1]["run"].shape[2] == obj.nIters
        assert obj.runs[-1]["run"].shape[1] == 2
        assert np.min(obj.runs[-1]["run"]) >= 0
        assert np.max(obj.runs[-1]["run"]) <= 2 * np.pi
        print(obj)
        obj.simulate(ic=6)
        assert obj.runs[-1]["K"] == obj.K
        assert obj.runs[-1]["nIters"] == obj.nIters
        assert obj.runs[-1]["nSim"] == 6
        assert obj.runs[-1]["run"].shape[2] == obj.nIters
        assert obj.runs[-1]["run"].shape[0] == 6
        assert np.min(obj.runs[-1]["run"]) >= 0
        assert np.max(obj.runs[-1]["run"]) <= 2 * np.pi
        obj.simulate(ic=testIc)
        assert obj.runs[-1]["K"] == obj.K
        assert obj.runs[-1]["nIters"] == obj.nIters
        assert obj.runs[-1]["nSim"] == 3
        assert obj.runs[-1]["run"].shape[2] == obj.nIters
        assert obj.runs[-1]["run"].shape[0] == 3
        assert np.min(obj.runs[-1]["run"]) >= 0
        assert np.max(obj.runs[-1]["run"]) <= 2 * np.pi
        assert np.allclose(obj.runs[-1]["run"][..., 0], testIc)
        assert len(obj.runs) == 3
        obj.simulate(ic=4, option="overwrite")
        assert obj.runs[-1]["K"] == obj.K
        assert obj.runs[-1]["nIters"] == obj.nIters
        assert obj.runs[-1]["nSim"] == 4
        assert obj.runs[-1]["run"].shape[2] == obj.nIters
        assert obj.runs[-1]["run"].shape[0] == 4
        assert np.min(obj.runs[-1]["run"]) >= 0
        assert np.max(obj.runs[-1]["run"]) <= 2 * np.pi
        assert len(obj.runs) == 3
        assert obj.runs[1]["K"] == obj.K
        assert obj.runs[1]["nIters"] == obj.nIters
        assert obj.runs[1]["nSim"] == 6
        assert obj.runs[1]["run"].shape[2] == obj.nIters
        assert obj.runs[1]["run"].shape[0] == 6
        assert np.min(obj.runs[1]["run"]) >= 0
        assert np.max(obj.runs[1]["run"]) <= 2 * np.pi
        try:
            obj.simulate(ic=-1)
        except Exception:
            print("Not possible")
        else:
            raise Exception("Invalid integer ic.")
        try:
            obj.simulate(ic=invalidIc)
        except Exception:
            print("Not possible")
        else:
            raise Exception("Invalic ic array shape.")
        try:
            obj.simulate(ic=invalidIc2)
        except Exception:
            print("Not possible")
        else:
            raise Exception("Invalic ic array values.")
        print(obj)


# # Getters and setters:
# ## K
# ### same as initialization
# ## nIters
# ### same as initialization
# ## seed
# ### same as initialization
def test_mid_K():
    obj = sMap()
    assert obj.K == 1.0
    obj.K = 0.75
    assert obj.K == 0.75
    obj.K = 3
    assert obj.K == 3.0
    try:
        obj.K = -1
    except Exception:
        print("Not possible")
    else:
        raise Exception("Invalid K value passed")
    print(obj)


def test_mid_nIters():
    obj = sMap()
    assert obj.nIters == 500
    obj.nIters = 1
    assert obj.nIters == 1
    obj.nIters = 23.7
    assert obj.nIters == 23
    try:
        obj.nIters = 0
    except Exception:
        print("Not possible")
    else:
        raise Exception("nIters is invalid")


def test_mid_seed():
    obj = sMap()
    assert obj.seed is None
    obj.seed = 42
    assert obj.seed == 42


# # Metadata:
# ## K range check
# ## list length check
# ## ind
# ### user, single
# ### user, double
# ## K
# ### user, single
# ### user, double
# ## nIters
# ### user, single
# ### user, double


# # Clear:
# ## total clear check
# ## ind
# ### user, single
# ### user, double
# ## K
# ### user, single
# ### user, double
# ## nIters
# ### user, single
# ### user, double
def test_clear_ind():
    obj = sMap()
    for i in range(64):
        obj.simulate()
    runBefore = obj.runs[3]
    runAfter = obj.runs[5]
    obj.clearRuns(run=4)
    assert runBefore == obj.runs[3]
    assert runAfter == obj.runs[4]
    assert len(obj.runs) == 63
    runBefore = obj.runs[-3]
    runAfter = obj.runs[-1]
    obj.clearRuns(run=-2)
    assert runBefore == obj.runs[-2]
    assert runAfter == obj.runs[-1]
    assert len(obj.runs) == 62
    try:
        obj.clearRuns(run=68)
    except Exception:
        print("Not possible")
    else:
        raise Exception("Invalid run index passed.")
    runBefore = obj.runs[34]
    runAfter = obj.runs[42]
    obj.clearRuns(run=(35, 41))
    assert runBefore == obj.runs[34]
    assert runAfter == obj.runs[35]
    assert len(obj.runs) == 55
    runBefore = obj.runs[-8]
    runAfter = obj.runs[-3]
    obj.clearRuns(run=(-7, -4))
    assert runBefore == obj.runs[-4]
    assert runAfter == obj.runs[-3]
    assert len(obj.runs) == 51
    try:
        obj.clearRuns(run=(32, 29))
    except Exception:
        print("Not possible")
    else:
        raise Exception("Invalid run range passed.")
    try:
        obj.clearRuns(run=(-4, 35))
    except Exception:
        print("Not possible")
    else:
        raise Exception("Invalid run range passed.")
    try:
        obj.clearRuns(run=(3, 99))
    except Exception:
        print("Not possible")
    else:
        raise Exception("Invalid run range passed.")
    try:
        obj.clearRuns(run=(5,))
    except Exception:
        print("Not possible")
    else:
        raise Exception("Invalid run range passed.")
    obj.clearRuns()
    assert len(obj.runs) == 0
