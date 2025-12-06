import numpy as np
from plots.mapEval import MapEvaluator

# run in directory one level up from tests folder: python -m tests.test_MapEvaluator

def make_dummy_runs():
    """
    Create a small dummy runs list for testing.

    We construct two runs with very simple data so we can easily check
    shapes and values.

    Returns
    -------
    runs : list of dict
        Each dict has keys "K" and "run".
        "run" has shape (nSim, 2, nIters).
    """
    nSim = 2
    nIters = 5

    # Run 0: K = 0.5
    # I = 10 + sim_idx + t
    # theta = 100 + sim_idx + t
    run0_data = np.zeros((nSim, 2, nIters), dtype=float)
    for s in range(nSim):
        for t in range(nIters):
            run0_data[s, 0, t] = 10 + s + t          # I component
            run0_data[s, 1, t] = 100 + s + t         # theta component

    run0 = {
        "K": 0.5,
        "run": run0_data,
    }

    # Run 1:
    # I = 20 + 2*sim_idx + t
    # theta = 200 + 2*sim_idx + t
    run1_data = np.zeros((nSim, 2, nIters), dtype=float)
    for s in range(nSim):
        for t in range(nIters):
            run1_data[s, 0, t] = 20 + 2 * s + t      # I component
            run1_data[s, 1, t] = 200 + 2 * s + t     # theta component

    run1 = {
        "K": 1.0,
        "run": run1_data,
    }

    return [run0, run1]


def assert_raises_assertion(fn, *args, **kwargs):
    """
    Helper to check that a function raises an AssertionError.
    """
    try:
        fn(*args, **kwargs)
    except AssertionError:
        return
    else:
        raise AssertionError("Expected AssertionError was not raised.")


def test_getKickValues():
    runs = make_dummy_runs()
    evaluator = MapEvaluator(runs)
    K_vals = evaluator.getKickValues()
    assert K_vals.shape == (2,)
    assert K_vals[0] == 0.5
    assert K_vals[1] == 1.0


def test_getTheta_getI():
    runs = make_dummy_runs()
    evaluator = MapEvaluator(runs)

    theta0 = evaluator.getTheta(0)
    I0 = evaluator.getI(0)

    # Shapes
    assert theta0.shape == (2, 5)
    assert I0.shape == (2, 5)

    # Check that they match the underlying data
    np.testing.assert_allclose(theta0, runs[0]["run"][:, 1, :])
    np.testing.assert_allclose(I0, runs[0]["run"][:, 0, :])

    # Invalid run_idx
    assert_raises_assertion(evaluator.getTheta, -1)
    assert_raises_assertion(evaluator.getTheta, 2)
    assert_raises_assertion(evaluator.getI, 100)


def test_thetaTail_ITail_valid():
    runs = make_dummy_runs()
    evaluator = MapEvaluator(runs)

    n_tail = 3
    theta_tail = evaluator.thetaTail(0, n_tail)
    I_tail = evaluator.ITail(1, n_tail)

    # Shapes
    assert theta_tail.shape == (2, n_tail)
    assert I_tail.shape == (2, n_tail)

    # Compare with manual slice
    full_theta0 = evaluator.getTheta(0)
    full_I1 = evaluator.getI(1)
    np.testing.assert_allclose(theta_tail, full_theta0[:, -n_tail:])
    np.testing.assert_allclose(I_tail, full_I1[:, -n_tail:])


def test_thetaTail_ITail_invalid_n_tail():
    runs = make_dummy_runs()
    evaluator = MapEvaluator(runs)

    # nIters = 5 in our dummy data
    assert_raises_assertion(evaluator.thetaTail, 0, 0)   # n_tail = 0
    assert_raises_assertion(evaluator.thetaTail, 0, 6)   # n_tail > nIters
    assert_raises_assertion(evaluator.ITail, 1, -1)      # negative n_tail


def test_IKDiagnosticData_thetaKDiagnosticData():
    runs = make_dummy_runs()
    evaluator = MapEvaluator(runs)

    n_tail = 2
    K_theta, theta_vals = evaluator.thetaKDiagnosticData(n_tail=n_tail)
    K_I, I_vals = evaluator.IKDiagnosticData(n_tail=n_tail)

    # There are 2 runs, each with nSim = 2 trajectories and n_tail = 2 points
    # per trajectory -> 2 * 2 * 2 = 8 points total.
    assert K_theta.shape == (8,)
    assert theta_vals.shape == (8,)
    assert K_I.shape == (8,)
    assert I_vals.shape == (8,)

    # All K's are either 0.5 or 1.0
    assert set(np.unique(K_theta)) == {0.5, 1.0}
    assert set(np.unique(K_I)) == {0.5, 1.0}

    # Empty runs case
    empty_eval = MapEvaluator([])
    K_empty_I, I_empty = empty_eval.IKDiagnosticData()
    K_empty_theta, theta_empty = empty_eval.thetaKDiagnosticData()
    assert K_empty_I.size == 0
    assert I_empty.size == 0
    assert K_empty_theta.size == 0
    assert theta_empty.size == 0


def test_IKDiagnosticData_thetaKDiagnosticData_invalid_n_tail():
    runs = make_dummy_runs()
    evaluator = MapEvaluator(runs)

    # Non-integer
    assert_raises_assertion(evaluator.thetaKDiagnosticData, 1.5)
    assert_raises_assertion(evaluator.IKDiagnosticData, 0.0)

    # Non-positive
    assert_raises_assertion(evaluator.thetaKDiagnosticData, 0)
    assert_raises_assertion(evaluator.IKDiagnosticData, -3)


def test_phaseSpaceData():
    runs = make_dummy_runs()
    evaluator = MapEvaluator(runs)

    n_tail = 4
    I_vals, theta_vals = evaluator.phaseSpaceData(run_idx=0, n_tail=n_tail)

    # There are nSim = 2 trajectories, so N_points = nSim * n_tail = 8
    assert I_vals.shape == (2 * n_tail,)
    assert theta_vals.shape == (2 * n_tail,)

    # Invalid run_idx
    assert_raises_assertion(evaluator.phaseSpaceData, 3, n_tail)

    # Invalid n_tail (0, or > nIters=5)
    assert_raises_assertion(evaluator.phaseSpaceData, 0, 0)
    assert_raises_assertion(evaluator.phaseSpaceData, 0, 6)


if __name__ == "__main__":
    # Run all tests when this file is executed directly.
    test_getKickValues()
    test_getTheta_getI()
    test_thetaTail_ITail_valid()
    test_thetaTail_ITail_invalid_n_tail()
    test_IKDiagnosticData_thetaKDiagnosticData()
    test_IKDiagnosticData_thetaKDiagnosticData_invalid_n_tail()
    test_phaseSpaceData()
    print("All MapEvaluator tests passed.")
