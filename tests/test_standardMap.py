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


def initK():
    # K-number of runs
    # 0.7881-4 0.7396-3 0.0891-2 0.6525-4 0.7328-1 0.4611-4 0.6583-3 0.5794-4
    # 0.3456-2 0.2407-4 0.5814-4 0.5815-1 0.7173-3 0.4169-4 0.3698-4 0.7669-2
    # 0.3756-3 0.6728-4 0.0247-1 0.0693-4 0.3200-4 0.4641-2 0.6293-3 0.9244-4
    # 0.6172-1 0.1010-4 0.3184-3 0.8270-2 0.1815-4 0.4466-4 0.1840-3 0.1768-1
    # 0.8292-4
    obj = sMap(nIters=10)
    for i in range(4):
        if i < 1:
            obj.K = 0.7328
            obj.simulate()
            obj.K = 0.5815
            obj.simulate()
            obj.K = 0.0247
            obj.simulate()
            obj.K = 0.6172
            obj.simulate()
            obj.K = 0.1768
            obj.simulate()
        if i < 2:
            obj.K = 0.0891
            obj.simulate()
            obj.K = 0.3456
            obj.simulate()
            obj.K = 0.7669
            obj.simulate()
            obj.K = 0.4641
            obj.simulate()
            obj.K = 0.827
            obj.simulate()
        if i < 3:
            obj.K = 0.7396
            obj.simulate()
            obj.K = 0.6583
            obj.simulate()
            obj.K = 0.7173
            obj.simulate()
            obj.K = 0.3756
            obj.simulate()
            obj.K = 0.6293
            obj.simulate()
            obj.K = 0.3184
            obj.simulate()
            obj.K = 0.184
            obj.simulate()
        obj.K = 0.7881
        obj.simulate()
        obj.K = 0.6525
        obj.simulate()
        obj.K = 0.4611
        obj.simulate()
        obj.K = 0.5794
        obj.simulate()
        obj.K = 0.2407
        obj.simulate()
        obj.K = 0.5814
        obj.simulate()
        obj.K = 0.4169
        obj.simulate()
        obj.K = 0.3698
        obj.simulate()
        obj.K = 0.6728
        obj.simulate()
        obj.K = 0.0693
        obj.simulate()
        obj.K = 0.32
        obj.simulate()
        obj.K = 0.9244
        obj.simulate()
        obj.K = 0.101
        obj.simulate()
        obj.K = 0.1815
        obj.simulate()
        obj.K = 0.4466
        obj.simulate()
        obj.K = 0.8292
        obj.simulate()
    # Run order by K:
    # #.7328 | 0.5815 | 0.0247 | >.6172 | 0.1768 | 0.0891 | 0.3456 | #.7669 | 0.4641 | #.8270 = #3 >1
    # #.7396 | 0.6583 | #.7173 | 0.3756 | 0.6293 | 0.3184 | 0.1840 | #.7881 | 0.6525 | 0.4611 = #3
    # 0.5794 | 0.2407 | <.5814 | 0.4169 | 0.3698 | 0.6728 | 0.0693 | 0.3200 | 0.9244 | 0.1010 = <1
    # 0.1815 | 0.4466 | #.8292 | 0.0891 | 0.3456 | #.7669 | 0.4641 | #.8270 | #.7396 | 0.6583 = #4
    # #.7173 | 0.3756 | 0.6293 | 0.3184 | 0.1840 | #.7881 | 0.6525 | 0.4611 | 0.5794 | 0.2407 = #2
    # <.5814 | 0.4169 | 0.3698 | 0.6728 | 0.0693 | 0.3200 | 0.9244 | 0.1010 | 0.1815 | 0.4466 = <1
    # #.8292 | #.7396 | 0.6583 | #.7173 | 0.3756 | 0.6293 | 0.3184 | 0.1840 | #.7881 | 0.6525 = #4
    # 0.4611 | 0.5794 | 0.2407 | <.5814 | 0.4169 | 0.3698 | 0.6728 | 0.0693 | 0.3200 | 0.9244 = <1
    # 0.1010 | 0.1815 | 0.4466 | #.8292 | #.7881 | 0.6525 | 0.4611 | 0.5794 | 0.2407 | <.5814 = #2 <1
    # 0.4169 | 0.3698 | 0.6728 | 0.0693 | 0.3200 | 0.9244 | 0.1010 | 0.1815 | 0.4466 | #.8292 = #1
    return obj


def initN():
    # nIters-number of runs
    # 57-4 37-3 94-2 31-4 53-1 89-4 40-3 18-4
    # 67-2 36-4 32-4 12-1 51-3 66-4 29-4 63-2
    # 82-3 47-4 98-1 39-4 71-4 75-2 90-3 85-4
    # 49-1 62-4 41-3 64-2 87-4 50-4 59-3 58-1
    # 99-4
    obj = sMap(nIters=10)
    for i in range(4):
        if i < 1:
            obj.nIters = 53
            obj.simulate()
            obj.nIters = 12
            obj.simulate()
            obj.nIters = 98
            obj.simulate()
            obj.nIters = 49
            obj.simulate()
            obj.nIters = 58
            obj.simulate()
        if i < 2:
            obj.nIters = 94
            obj.simulate()
            obj.nIters = 67
            obj.simulate()
            obj.nIters = 63
            obj.simulate()
            obj.nIters = 75
            obj.simulate()
            obj.nIters = 64
            obj.simulate()
        if i < 3:
            obj.nIters = 37
            obj.simulate()
            obj.nIters = 40
            obj.simulate()
            obj.nIters = 51
            obj.simulate()
            obj.nIters = 82
            obj.simulate()
            obj.nIters = 90
            obj.simulate()
            obj.nIters = 41
            obj.simulate()
            obj.nIters = 59
            obj.simulate()
        obj.nIters = 57
        obj.simulate()
        obj.nIters = 31
        obj.simulate()
        obj.nIters = 89
        obj.simulate()
        obj.nIters = 18
        obj.simulate()
        obj.nIters = 36
        obj.simulate()
        obj.nIters = 32
        obj.simulate()
        obj.nIters = 66
        obj.simulate()
        obj.nIters = 29
        obj.simulate()
        obj.nIters = 47
        obj.simulate()
        obj.nIters = 39
        obj.simulate()
        obj.nIters = 71
        obj.simulate()
        obj.nIters = 85
        obj.simulate()
        obj.nIters = 62
        obj.simulate()
        obj.nIters = 87
        obj.simulate()
        obj.nIters = 50
        obj.simulate()
        obj.nIters = 99
        obj.simulate()
    # Run order by nIters:
    # _53 | %12 | _98 | _49 | _58 | _94 | _67 | _63 | >75 | _64 = 1> 1%
    # %37 | %40 | _51 | _82 | _90 | _41 | _59 | _57 | %31 | _89 = 3%
    # %18 | %36 | %32 | _66 | %29 | _47 | %39 | _71 | _85 | _62 = 5%
    # _87 | <50 | _99 | _94 | _67 | _63 | >75 | _64 | %37 | %40 = 1> 1< 2%
    # _51 | _82 | _90 | _41 | _59 | _57 | %31 | _89 | %18 | %36 = 3%
    # %32 | _66 | %29 | _47 | %39 | _71 | _85 | _62 | _87 | <50 = 1< 3%
    # _99 | %37 | %40 | _51 | _82 | _90 | _41 | _59 | _57 | %31 = 3%
    # _89 | %18 | %36 | %32 | _66 | %29 | _47 | %39 | _71 | _85 = 5%
    # _62 | _87 | <50 | _99 | _57 | %31 | _89 | %18 | %36 | %32 = 1< 4%
    # _66 | %29 | _47 | %39 | _71 | _85 | _62 | _87 | <50 | _99 = 1< 2%
    return obj


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
    obj = sMap(nIters=10)
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


def test_clear_K():
    obj = initK()
    runBefore = obj.runs[2]
    runAfter = obj.runs[4]
    obj.clearRuns(K=0.6172)
    assert runBefore == obj.runs[2]
    assert runAfter == obj.runs[3]
    assert len(obj.runs) == 99
    runBefore = [obj.runs[20], obj.runs[48], obj.runs[71], obj.runs[87]]
    runAfter = [obj.runs[22], obj.runs[50], obj.runs[73], obj.runs[89]]
    obj.clearRuns(K=0.5814)
    assert runBefore == [obj.runs[20], obj.runs[47], obj.runs[69], obj.runs[84]]
    assert runAfter == [obj.runs[21], obj.runs[48], obj.runs[70], obj.runs[85]]
    assert len(obj.runs) == 95
    obj.clearRuns(K=1.5)
    assert len(obj.runs) == 95
    try:
        obj.clearRuns(K=-2)
    except Exception:
        print("Not possible")
    else:
        raise Exception("Invalid K passed.")
    runBefore = [
        obj.runs[5],
        obj.runs[7],
        obj.runs[10],
        obj.runs[15],
        obj.runs[29],
        obj.runs[32],
        obj.runs[34],
        obj.runs[37],
        obj.runs[42],
        obj.runs[56],
        obj.runs[59],
        obj.runs[64],
        obj.runs[78],
        obj.runs[93],
    ]
    runAfter = [
        obj.runs[1],
        obj.runs[7],
        obj.runs[10],
        obj.runs[12],
        obj.runs[17],
        obj.runs[31],
        obj.runs[34],
        obj.runs[37],
        obj.runs[39],
        obj.runs[44],
        obj.runs[59],
        obj.runs[61],
        obj.runs[66],
        obj.runs[81],
    ]
    obj.clearRuns(K=(0.7, 0.9))
    assert runBefore == [
        obj.runs[4],
        obj.runs[5],
        obj.runs[6],
        obj.runs[10],
        obj.runs[23],
        obj.runs[25],
        obj.runs[26],
        obj.runs[27],
        obj.runs[31],
        obj.runs[44],
        obj.runs[45],
        obj.runs[49],
        obj.runs[62],
        obj.runs[75],
    ]
    assert runAfter == [
        obj.runs[0],
        obj.runs[5],
        obj.runs[6],
        obj.runs[7],
        obj.runs[11],
        obj.runs[24],
        obj.runs[26],
        obj.runs[27],
        obj.runs[28],
        obj.runs[32],
        obj.runs[45],
        obj.runs[46],
        obj.runs[50],
        obj.runs[63],
    ]
    assert len(obj.runs) == 76
    obj.clearRuns(K=(1.5, 3.5))
    assert len(obj.runs) == 76
    try:
        obj.clearRuns(K=(-4, 5))
    except Exception:
        print("Not possible")
    else:
        raise Exception("Invalid K range passed.")
    try:
        obj.clearRuns(K=(2, -8))
    except Exception:
        print("Not possible")
    else:
        raise Exception("Invalid K range passed.")
    try:
        obj.clearRuns(K=(4, 1))
    except Exception:
        print("Not possible")
    else:
        raise Exception("Invalid K range passed.")
    try:
        obj.clearRuns(K=(4,))
    except Exception:
        print("Not possible")
    else:
        raise Exception("Invalid K range passed.")


def test_clear_nIters():
    obj = initN()
    runBefore = [obj.runs[7], obj.runs[35]]
    runAfter = [obj.runs[9], obj.runs[37]]
    obj.clearRuns(N=75)
    assert runBefore == [obj.runs[7], obj.runs[34]]
    assert runAfter == [obj.runs[8], obj.runs[35]]
    assert len(obj.runs) == 98
    runBefore = [obj.runs[29], obj.runs[56], obj.runs[79], obj.runs[95]]
    runAfter = [obj.runs[31], obj.runs[58], obj.runs[81], obj.runs[97]]
    obj.clearRuns(N=50)
    assert runBefore == [obj.runs[29], obj.runs[55], obj.runs[77], obj.runs[92]]
    assert runAfter == [obj.runs[30], obj.runs[56], obj.runs[78], obj.runs[93]]
    assert len(obj.runs) == 94
    obj.clearRuns(N=135)
    assert len(obj.runs) == 94
    try:
        obj.clearRuns(N=0)
    except Exception:
        print("Not possible")
    else:
        raise Exception("Invalid run length passed.")
    runBefore = [
        obj.runs[0],
        obj.runs[8],
        obj.runs[16],
        obj.runs[18],
        obj.runs[22],
        obj.runs[24],
        obj.runs[34],
        obj.runs[42],
        obj.runs[44],
        obj.runs[48],
        obj.runs[50],
        obj.runs[56],
        obj.runs[64],
        obj.runs[66],
        obj.runs[70],
        obj.runs[72],
        obj.runs[79],
        obj.runs[81],
        obj.runs[85],
        obj.runs[87],
    ]
    runAfter = [
        obj.runs[2],
        obj.runs[11],
        obj.runs[18],
        obj.runs[22],
        obj.runs[24],
        obj.runs[26],
        obj.runs[37],
        obj.runs[44],
        obj.runs[48],
        obj.runs[50],
        obj.runs[52],
        obj.runs[59],
        obj.runs[66],
        obj.runs[70],
        obj.runs[72],
        obj.runs[74],
        obj.runs[81],
        obj.runs[85],
        obj.runs[87],
        obj.runs[89],
    ]
    obj.clearRuns(N=(10, 40))
    assert runBefore == [
        obj.runs[0],
        obj.runs[7],
        obj.runs[13],
        obj.runs[14],
        obj.runs[15],
        obj.runs[16],
        obj.runs[25],
        obj.runs[31],
        obj.runs[32],
        obj.runs[33],
        obj.runs[34],
        obj.runs[39],
        obj.runs[45],
        obj.runs[46],
        obj.runs[47],
        obj.runs[48],
        obj.runs[54],
        obj.runs[55],
        obj.runs[56],
        obj.runs[57],
    ]
    assert runAfter == [
        obj.runs[1],
        obj.runs[8],
        obj.runs[14],
        obj.runs[15],
        obj.runs[16],
        obj.runs[17],
        obj.runs[26],
        obj.runs[32],
        obj.runs[33],
        obj.runs[34],
        obj.runs[35],
        obj.runs[40],
        obj.runs[46],
        obj.runs[47],
        obj.runs[48],
        obj.runs[49],
        obj.runs[55],
        obj.runs[56],
        obj.runs[57],
        obj.runs[58],
    ]
    assert len(obj.runs) == 63
    obj.clearRuns(N=(100, 125))
    assert len(obj.runs) == 63
    try:
        obj.clearRuns(N=(-1, 57))
    except Exception:
        print("Not possible")
    else:
        raise Exception("Invalid run length range passed.")
    try:
        obj.clearRuns(N=(39, -10))
    except Exception:
        print("Not possible")
    else:
        raise Exception("Invalid run length range passed.")
    try:
        obj.clearRuns(N=(80, 60))
    except Exception:
        print("Not possible")
    else:
        raise Exception("Invalid run length range passed.")
    try:
        obj.clearRuns(N=(17,))
    except Exception:
        print("Not possible")
    else:
        raise Exception("Invalid run length range passed.")
