# Unit tests using the pytest library
import numpy as np
from map.standardMap import standardMap as sMap


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
    assert obj.seed == None
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
        obj.simulate()
        assert obj.runs[-1]["K"] == obj.K
        assert obj.runs[-1]["nIters"] == obj.nIters
        assert obj.runs[-1]["run"].shape[2] == obj.nIters
        assert obj.runs[-1]["run"].shape[1] == 2
        assert np.min(obj.runs[-1]["run"]) >= 0
        assert np.max(obj.runs[-1]["run"]) <= 2 * np.pi
        obj.simulate(ic=6)
        assert obj.runs[-1]["K"] == obj.K
        assert obj.runs[-1]["nIters"] == obj.nIters
        assert obj.runs[-1]["run"].shape[2] == obj.nIters
        assert obj.runs[-1]["run"].shape[0] == 6
        assert np.min(obj.runs[-1]["run"]) >= 0
        assert np.max(obj.runs[-1]["run"]) <= 2 * np.pi
        obj.simulate(ic=testIc)
        assert obj.runs[-1]["K"] == obj.K
        assert obj.runs[-1]["nIters"] == obj.nIters
        assert obj.runs[-1]["run"].shape[2] == obj.nIters
        assert obj.runs[-1]["run"].shape[0] == 3
        assert np.min(obj.runs[-1]["run"]) >= 0
        assert np.max(obj.runs[-1]["run"]) <= 2 * np.pi
        assert np.allclose(obj.runs[-1]["run"][..., 0], testIc)
        assert len(obj.runs) == 3
        obj.simulate(ic=4, option="overwrite")
        assert obj.runs[-1]["K"] == obj.K
        assert obj.runs[-1]["nIters"] == obj.nIters
        assert obj.runs[-1]["run"].shape[2] == obj.nIters
        assert obj.runs[-1]["run"].shape[0] == 4
        assert np.min(obj.runs[-1]["run"]) >= 0
        assert np.max(obj.runs[-1]["run"]) <= 2 * np.pi
        assert len(obj.runs) == 3
        assert obj.runs[1]["K"] == obj.K
        assert obj.runs[1]["nIters"] == obj.nIters
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


# # Getters and setters:
# ## K
# ### same as initialization
# ## nIters
# ### same as initialization
# ## seed
# ### same as initialization

# # Metadata:
# ## K range check
# ## list length check
# ## ind
# ### user, single
# ### user, double
# ## K
# ### user, single
# ### user, double

# # Clear:
# ##
