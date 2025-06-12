#!/usr/bin/env python
"""Tests for core PyRayHF library functions."""

import numpy as np
import pytest

from copy import deepcopy
from lmfit import Parameters
from numpy.testing import assert_allclose
from PyRayHF.library import constants
from PyRayHF.library import den2freq
from PyRayHF.library import find_mu_mup
from PyRayHF.library import find_vh
from PyRayHF.library import find_X
from PyRayHF.library import find_Y
from PyRayHF.library import freq2den
from PyRayHF.library import model_VH
from PyRayHF.library import regrid_to_nonuniform_grid
from PyRayHF.library import residual_VH
from PyRayHF.library import smooth_nonuniform_grid
from PyRayHF.library import vertical_forward_operator
from PyRayHF.library import vertical_to_magnetic_angle


def test_constants_output():
    """Test that constants function returns correct values."""
    cp, g_p = constants()
    assert np.isclose(cp, 8.97866275, rtol=1e-8)
    assert np.isclose(g_p, 2.799249247e10, rtol=1e-8)


def test_den2freq_scalar():
    """Test den2freq with a scalar input."""
    density = 1.0e12
    expected = np.sqrt(density) * 8.97866275
    result = den2freq(density)
    assert isinstance(result, float), "Result should be a float"
    assert result == pytest.approx(expected,
                                   rel=1e-8), "Incorrect freq for scalar input"


def test_den2freq_array():
    """Test den2freq with an array input."""
    density = np.array([1.0e12, 2.5e12, 0.0])
    expected = np.sqrt(density) * 8.97866275
    result = den2freq(density)
    assert isinstance(result, np.ndarray), "Result should be a NumPy array"
    assert np.allclose(result,
                       expected,
                       rtol=1e-8), "Incorrect frequency for array input"


def test_freq2den_scalar():
    """Test freq2den with a scalar input."""
    frequency = 8.97866275e6  # corresponds to density = 1e12 m^-3
    expected = (frequency / 8.97866275) ** 2
    result = freq2den(frequency)
    assert isinstance(result, float), "Result should be a float"
    assert result == pytest.approx(expected,
                                   rel=1e-8), "Incorrect den for scalar input"


def test_freq2den_array():
    """Test freq2den with an array input."""
    frequency = np.array([8.97866275e6, 2 * 8.97866275e6, 0.0])
    expected = (frequency / 8.97866275) ** 2
    result = freq2den(frequency)
    assert isinstance(result, np.ndarray), "Result should be a NumPy array"
    assert np.allclose(result,
                       expected,
                       rtol=1e-8), "Incorrect density for array input"


def test_find_X_scalar():
    """Test find_X with scalar inputs."""
    n_e = 1.0e12
    f = 1.0e7
    expected = ((np.sqrt(n_e) * 8.97866275) ** 2) / (f ** 2)
    result = find_X(n_e, f)
    assert isinstance(result, float), "Result should be a float"
    assert result == pytest.approx(expected,
                                   rel=1e-8), "Incorrect X for scalar input"


def test_find_X_array():
    """Test find_X with array inputs."""
    n_e = np.array([1.0e12, 2.5e12, 0.0])
    f = np.array([1.0e7, 1.5e7, 2.0e7])
    expected = ((np.sqrt(n_e) * 8.97866275) ** 2) / (f ** 2)
    result = find_X(n_e, f)
    assert isinstance(result, np.ndarray), "Result should be a NumPy array"
    assert np.allclose(result,
                       expected,
                       rtol=1e-8), "Incorrect X values for array input"


def test_find_Y_scalar():
    """Test find_Y with scalar inputs."""
    f = 1.0e7  # Hz
    b = 5.0e-5  # Tesla
    g_p = 2.799249247e10
    expected = g_p * b / f
    result = find_Y(f, b)
    assert isinstance(result, float), "Result should be a float"
    assert result == pytest.approx(expected,
                                   rel=1e-8), "Incorrect Y for scalar input"


def test_find_Y_array():
    """Test find_Y with array inputs."""
    f = np.array([1.0e7, 2.0e7, 3.0e7])  # Hz
    b = np.array([5.0e-5, 6.0e-5, 7.0e-5])  # Tesla
    g_p = 2.799249247e10
    expected = g_p * b / f
    result = find_Y(f, b)
    assert isinstance(result, np.ndarray), "Result should be a NumPy array"
    assert np.allclose(result,
                       expected,
                       rtol=1e-8), "Incorrect Y values for array input"


def test_find_mu_mup_basic():
    """Test find_mu_mup with sample inputs."""
    aX = np.array([0.02926785, 0.70981059, 0.99672596])
    aY = np.array([0.17123449, 0.16205801, 0.15757213])
    bpsi = np.array([60.91523271, 61.66028645, 62.02450192])
    mode = 'O'
    expected_mu = np.array([0.98626092, 0.56890941, 0.06475905])
    expected_mup = np.array([1.01313137, 1.79819741, 19.76001084])

    mu, mup = find_mu_mup(aX, aY, bpsi, mode)

    valid_idx = ~np.isnan(expected_mu)
    np.testing.assert_allclose(mu[valid_idx], expected_mu[valid_idx],
                               rtol=1e-5)
    np.testing.assert_allclose(mup[valid_idx], expected_mup[valid_idx],
                               rtol=1e-5)


def test_find_vh_basic():
    """Test find_vh with sample inputs."""
    aX = np.array([[0.5, 0.6]])
    aY = np.array([[0.1, 0.2]])
    bpsi = np.array([[45.0, 45.0]])
    dh = np.array([[1.0, 1.0]])
    alt_min = 100.0
    mode = 'O'

    vh = find_vh(aX, aY, bpsi, dh, alt_min, mode)

    assert isinstance(vh, np.ndarray)
    assert vh.shape == (1,)
    assert vh[0] > alt_min


def test_smooth_nonuniform_grid_basic():
    """Test non-uniform grid generator returns expected structure."""
    start = 0.
    end = 1.
    n_points = 10
    sharpness = 5.0

    grid = smooth_nonuniform_grid(start, end, n_points, sharpness)

    # Check number of points
    assert len(grid) == n_points

    # Ensure grid is increasing
    assert np.all(np.diff(grid) > 0)

    # Check boundaries
    assert np.isclose(grid[0], start, atol=1e-6)  # Lower boundary
    assert np.isclose(grid[-1], end, atol=1e-6)   # Upper boundary


def test_regrid_to_nonuniform_grid_basic():
    """Test regrid_to_nonuniform_grid with basic inputs."""
    f = np.array([1.0e6, 2.0e6])
    n_e = np.array([1.0e11, 5.0e11, 1.0e12])
    b = np.array([5.0e-5, 5.0e-5, 5.0e-5])
    bpsi = np.array([60.0, 60.0, 60.0])
    aalt = np.array([100, 200, 300])
    npoints = 10

    result = regrid_to_nonuniform_grid(f, n_e, b, bpsi, aalt, npoints)

    assert isinstance(result, dict)
    assert 'freq' in result and 'den' in result
    assert result['freq'].shape[0] == len(f)
    assert result['den'].shape[0] == len(f)


def test_vertical_to_magnetic_angle_basic():
    """Test vertical_to_magnetic_angle with scalar and array inputs."""
    inclination = 60.0
    expected = 90.0 - 60.0
    result = vertical_to_magnetic_angle(inclination)
    assert result == expected

    inclinations = np.array([0.0, 45.0, 90.0])
    expected_array = 90.0 - np.abs(inclinations)
    result_array = vertical_to_magnetic_angle(inclinations)
    assert np.allclose(result_array, expected_array)


def test_vertical_forward_operator_basic_O_mode():
    """Basic test for vertical_forward_operator in O mode with short arrays."""
    freq = np.array([1.0, 2.0, 10.0])  # MHz (10 MHz > fof2)
    alt = np.array([100, 200, 300])
    den = np.array([1e11, 5e11, 1e12])
    bmag = np.array([5e-5, 5e-5, 5e-5])
    bpsi = np.array([60.0, 60.0, 60.0])

    vh = vertical_forward_operator(freq, den, bmag, bpsi, alt,
                                   mode='O', n_points=50)

    assert isinstance(vh, np.ndarray)
    assert vh.shape == freq.shape
    assert np.isnan(vh[-1])  # 10 MHz > fof2, should be NaN
    assert np.all(np.isfinite(vh[:-1]))  # Lower freqs should be finite


def test_model_VH_output():
    """Basic test for model_VH in O mode with short arrays."""
    # Input parameters
    F2 = {'Nm': np.array([[1.17848165e+12]]),
          'fo': np.array([[9.64625394]]),
          'M3000': np.array([[2.64168819]]),
          'hm': np.array([[365.13828931]]),
          'B_top': np.array([[32.52487907]]),
          'B_bot': np.array([[41.26005561]])}
    F1 = {'Nm': np.array([[7.80902301e+11]]),
          'fo': np.array([[7.93574143]]),
          'P': np.array([[0.91422852]]),
          'hm': np.array([[219.26637887]]),
          'B_bot': np.array([[54.63318944]])}
    E = {'Nm': np.array([[1.2846662e+11]]),
         'fo': np.array([[3.2096443]]),
         'hm': np.array([[110.]]),
         'B_bot': np.array([[5.]]),
         'B_top': np.array([[7.]]),
         'solzen': np.array([[22.26668451]]),
         'solzen_eff': np.array([[22.26668451]])}
    freq = np.array([3.0, 4.0, 5.0])
    alt = np.array([100, 200, 300])
    bmag = np.array([5e-5, 5e-5, 5e-5])
    bpsi = np.array([60.0, 60.0, 60.0])

    # Expected outputs
    expected_vh = np.array([198.1695621, 247.07192693, 261.65938426])
    expected_edp = np.array([5.39526841e+10,
                             2.81042886e+11,
                             6.66833261e+11])

    # Run the model
    vh, edp = model_VH(F2, F1, E, freq, alt, bmag, bpsi)

    # Compare results
    assert_allclose(vh, expected_vh, rtol=1e-6)
    assert_allclose(edp, expected_edp, rtol=1e-6)


def test_zero_residual_when_parameters_match():
    """Basic test for residual_VH in O mode with short arrays."""
    # Input dictionaries
    F2 = {'Nm': np.array([[1.17848165e+12]]),
          'fo': np.array([[9.64625394]]),
          'M3000': np.array([[2.64168819]]),
          'hm': np.array([[365.13828931]]),
          'B_top': np.array([[32.52487907]]),
          'B_bot': np.array([[41.26005561]])}
    F1 = {'Nm': np.array([[7.80902301e+11]]),
          'fo': np.array([[7.93574143]]),
          'P': np.array([[0.91422852]]),
          'hm': np.array([[219.26637887]]),
          'B_bot': np.array([[54.63318944]])}
    E = {'Nm': np.array([[1.2846662e+11]]),
         'fo': np.array([[3.2096443]]),
         'hm': np.array([[110.]]),
         'B_bot': np.array([[5.]]),
         'B_top': np.array([[7.]]),
         'solzen': np.array([[22.26668451]]),
         'solzen_eff': np.array([[22.26668451]])}

    freq = np.array([3.0, 4.0, 5.0])
    alt = np.array([100, 200, 300])
    bmag = np.array([5e-5, 5e-5, 5e-5])
    bpsi = np.array([60.0, 60.0, 60.0])

    # Use the true model output as synthetic observations
    vh_obs, _ = model_VH(F2, deepcopy(F1), deepcopy(E), freq, alt, bmag, bpsi)

    # Parameters that match the F2 inputs
    params = Parameters()
    params.add('NmF2', value=1.17848165e+12)
    params.add('hmF2', value=365.13828931)
    params.add('B_bot', value=41.26005561)

    # Compute residual
    residual = residual_VH(params, F2, F1, E, freq, vh_obs, alt, bmag, bpsi)

    # Expect near-zero residual
    assert_allclose(residual, np.zeros_like(vh_obs), rtol=1e-6, atol=1e-6)
