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
