from PyRAY.library import constants
from PyRAY.library import den2freq
from PyRAY.library import find_mu_mup
from PyRAY.library import find_vh
from PyRAY.library import find_X
from PyRAY.library import find_Y
from PyRAY.library import freq2den
from PyRAY.library import regrid_to_nonuniform_grid
from PyRAY.library import smooth_nonuniform_grid
from PyRAY.library import vertical_to_magnetic_angle
from PyRAY.library import virtical_forward_operator
import numpy as np
import pytest

#!/usr/bin/env python
# --------------------------------------------------------
"""Unit tests for PyRAY.library functions."""



def test_constants_output():
    """Test that constants() returns expected constant values."""
    cp_expected = 8.97866275
    g_p_expected = 2.799249247e10

    cp, g_p = constants()

    assert isinstance(cp, float), "cp should be a float"
    assert isinstance(g_p, float), "g_p should be a float"
    assert cp == pytest.approx(cp_expected, rel=1e-8), "cp value mismatch"
    assert g_p == pytest.approx(g_p_expected, rel=1e-8), "g_p value mismatch"


def test_den2freq_scalar():
    """Test den2freq with a scalar input."""
    density = 1.0e12
    expected = np.sqrt(density) * 8.97866275
    result = den2freq(density)
    assert isinstance(result, float), "Result should be a float"
    assert result == pytest.approx(expected,
                                   (rel=1e-8),
                                   "Incorrect frequency for scalar input")


def test_den2freq_array():
    """Test den2freq with an array input."""
    density = np.array([1.0e12, 2.5e12, 0.0])
    expected = np.sqrt(density) * 8.97866275
    result = den2freq(density)
    assert isinstance(result, np.ndarray), "Result should be a NumPy array"
    assert np.allclose(result, (expected,
                                rtol=1e-8),
                       "Incorrect frequency for array input")


def test_den2freq_negative_input():
    """Test that den2freq raises ValueError on negative input."""
    with pytest.raises(ValueError, match="Density must be non-negative"):
        den2freq(-1.0e11)


def test_freq2den_scalar():
    """Test freq2den with a scalar input."""
    frequency = 8.97866275e6  # corresponds to density = 1e12 m^-3
    expected = (frequency / 8.97866275) ** 2
    result = freq2den(frequency)
    assert isinstance(result, float), "Result should be a float"
    assert result == pytest.approx(expected,
                                   (rel=1e-8),
                                   "Incorrect density for scalar input")


def test_freq2den_array():
    """Test freq2den with an array input."""
    frequencies = np.array([8.97866275e6, 1.0e7, 0.0])
    expected = (frequencies / 8.97866275) ** 2
    result = freq2den(frequencies)
    assert isinstance(result, np.ndarray), "Result should be a NumPy array"
    assert np.allclose(result,
                       (expected,
                        rtol=1e-8), "Incorrect density for array input")


def test_freq2den_negative_result_not_expected():
    """Test that freq2den doesn't produce negative densities for valid input."""
    frequencies = np.array([1.0e6, 5.0e6])
    result = freq2den(frequencies)
    assert np.all(result >= 0), "Densities must be non-negative"


def test_find_X_scalar():
    """Test find_X with scalar inputs."""
    n_e = 1.0e12
    f = 1.0e7
    expected = ((np.sqrt(n_e) * 8.97866275) ** 2) / (f ** 2)
    result = find_X(n_e, f)
    assert isinstance(result, float), "Result should be a float"
    assert result == pytest.approx(expected,
                                   (rel=1e-8),
                                   "Incorrect X value for scalar input")


def test_find_X_array():
    """Test find_X with array inputs."""
    n_e = np.array([1.0e12, 2.0e12])
    f = np.array([1.0e7, 2.0e7])
    expected = ((np.sqrt(n_e) * 8.97866275) ** 2) / (f ** 2)
    result = find_X(n_e, f)
    assert isinstance(result, np.ndarray), "Result should be a NumPy array"
    assert np.allclose(result,
                       (expected,
                        rtol=1e-8), "Incorrect X values for array input")


def test_find_Y_scalar():
    """Test find_Y with scalar inputs."""
    f = 1.0e7  # Hz
    b = 5.0e-5  # Tesla
    g_p = 2.799249247e10
    expected = g_p * b / f
    result = find_Y(f, b)
    assert isinstance(result, float), "Result should be a float"
    assert result == pytest.approx(expected,
                                   (rel=1e-8),
                                   "Incorrect Y value for scalar input")


def test_find_Y_array():
    """Test find_Y with array inputs."""
    f = np.array([1.0e7, 2.0e7])
    b = np.array([5.0e-5, 4.0e-5])
    g_p = 2.799249247e10
    expected = g_p * b / f
    result = find_Y(f, b)
    assert isinstance(result, np.ndarray), "Result should be a NumPy array"
    assert np.allclose(result,
                       (expected,
                        rtol=1e-8), "Incorrect Y values for array input")


def test_find_mu_mup_edge_cases():
    """Test find_mu_mup function with edge-case inputs including nan output."""
    aX = np.array([
        0.02926785, 0.70981059, 0.7962579, 0.88113642, 0.95958365,
        0.98667663, 0.99572596, 0.99870493, 0.99968559, 1.00000841
    ])
    aY = np.array([
        0.17123449, 0.16205801, 0.15903717, 0.15804273, 0.15771537,
        0.15760761, 0.15757213, 0.15756045, 0.15755661, 0.15755534
    ])
    bpsi = np.array([
        60.91523271, 61.66028645, 61.90555292, 61.98629292, 62.01287197,
        62.0216216, 62.02450192, 62.0254501, 62.02576223, 62.02586499
    ])
    expected_mu = np.array([
        0.98626092, 0.56890941, 0.48473479, 0.37867353, 0.22583644,
        0.1304215, 0.07397953, 0.04074072, 0.02007668, np.nan
    ])
    expected_mup = np.array([
        1.01313137, 1.79819742, 2.17991583, 2.97852345, 5.47867872,
        9.74206832, 17.28626019, 31.44650382, 63.84888313, np.nan
    ])

    mu, mup = find_mu_mup(aX, aY, bpsi, mode="O")

    # Check shapes match
    assert mu.shape == expected_mu.shape
    assert mup.shape == expected_mup.shape

    # Mask NaNs for comparison
    valid_idx = ~np.isnan(expected_mu)
    np.testing.assert_allclose(mu[valid_idx],
                               expected_mu[valid_idx], rtol=1e-5)
    np.testing.assert_allclose(mup[valid_idx],
                               (expected_mup[valid_idx], rtol=1e-5))

    # Ensure NaNs are in correct positions
    assert np.isnan(mu[~valid_idx]).all()
    assert np.isnan(mup[~valid_idx]).all()


def test_find_vh_edge_case():
    """Test virtual height calculation using known edge-case inputs."""
    aX = np.array([
        0.02926785, 0.70981059, 0.7962579, 0.88113642, 0.95958365,
        0.98667663, 0.99572596, 0.99870493, 0.99968559, 1.00000841
    ])
    aY = np.array([
        0.17123449, 0.16205801, 0.15903717, 0.15804273, 0.15771537,
        0.15760761, 0.15757213, 0.15756045, 0.15755661, 0.15755534
    ])
    bpsi = np.array([
        60.91523271, 61.66028645, 61.90555292, 61.98629292, 62.01287197,
        62.0216216, 62.02450192, 62.0254501, 62.02576223, 62.02586499
    ])
    dh = np.ones_like(aX) * 10.0  # 10 km vertical layer thickness
    alt_min = 100.0  # km
    expected_mup = np.array([
        1.01313137, 1.79819742, 2.17991583, 2.97852345, 5.47867872,
        9.74206832, 17.28626019, 31.44650382, 63.84888313, np.nan
    ])

    # Reshape for 2D broadcasting along axis=1
    X = aX.reshape(1, -1)
    Y = aY.reshape(1, -1)
    bpsi_arr = bpsi.reshape(1, -1)
    dh_arr = dh.reshape(1, -1)

    vh = find_vh(X, Y, bpsi_arr, dh_arr, alt_min, mode="O")

    expected_vh = np.nansum(expected_mup * 10.0) + alt_min
    np.testing.assert_allclose(vh[0], expected_vh, rtol=1e-5)


def test_smooth_nonuniform_grid_basic():
    """Test non-uniform grid generator returns expected structure."""
    start = 100.0
    end = 500.0
    n_points = 10
    sharpness = 5.0

    grid = smooth_nonuniform_grid(start, end, n_points, sharpness)

    # Check number of points
    assert len(grid) == n_points

    # Ensure grid is increasing
    assert np.all(np.diff(grid) > 0)

    # Check boundaries are approximately correct
    assert np.isclose(grid[0], start, atol=1.0)
    assert np.isclose(grid[-1], end, atol=1.0)


def test_regrid_to_nonuniform_grid_simple_case():
    """Test regridding on a small, simple input."""
    f = np.array([5e6])  # ionosonde frequency in Hz
    n_e = np.array([1e10, 2e11, 5e11, 1e12, 5e11, 1e10])  # electron density
    b = np.linspace(3e-5, 3.5e-5, 6)  # magnetic field in Tesla
    bpsi = np.linspace(60, 70, 6)  # angle to magnetic field in degrees
    aalt = np.linspace(100, 200, 6)  # altitude in km
    npoints = 5

    result = regrid_to_nonuniform_grid(f, n_e, b, bpsi, aalt, npoints)

    # Check basic properties of result
    assert isinstance(result, dict)
    for key in ['freq', 'den', 'bmag', 'bpsi', 'dist', 'alt',
                'crit_height', 'ind']:
        assert key in result
        assert result[key].shape == (1, npoints)

    # Check critical height is below the max alt
    assert np.all(result['crit_height'] < np.max(aalt))
    # Check densities are non-negative
    assert np.all(result['den'] >= 0)


def test_vertical_to_magnetic_angle_basic_cases():
    """Test conversion from inclination to vertical angle."""
    # Scalar input
    assert vertical_to_magnetic_angle(60.0) == 30.0
    assert vertical_to_magnetic_angle(-60.0) == 30.0
    assert vertical_to_magnetic_angle(0.0) == 90.0
    assert vertical_to_magnetic_angle(90.0) == 0.0
    assert vertical_to_magnetic_angle(-90.0) == 0.0

    # Array input
    inclinations = np.array([-90, -45, 0, 45, 90])
    expected = np.array([0, 45, 90, 45, 0])
    result = vertical_to_magnetic_angle(inclinations)
    np.testing.assert_array_equal(result, expected)


def test_virtical_forward_operator_basic_O_mode():
    """Basic test for virtical_forward_operator in O mode with short arrays."""
    # Generate a simple test case with increasing density and constant B-field
    freq = np.array([1.0, 2.0, 3.0])  # MHz
    alt = np.array([100, 200, 300])  # km
    den = np.array([1e11, 5e11, 1e12])  # m^-3
    bmag = np.array([5e-5, 5e-5, 5e-5])  # Tesla
    bpsi = np.array([60.0, 60.0, 60.0])  # degrees

    vh = virtical_forward_operator(freq, den, bmag, bpsi, alt, mode='O',
                                   n_points=50)

    # Check that output has correct shape and types
    assert isinstance(vh, np.ndarray)
    assert vh.shape == freq.shape
    assert np.isnan(vh[-1])  # Last freq is above foF2, so should be NaN
    assert np.all(np.isfinite(vh[:-1]))  # Lower freqs should be finite