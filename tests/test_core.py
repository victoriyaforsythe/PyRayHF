#!/usr/bin/env python
# --------------------------------------------------------
"""Unit tests for PyRAY.library functions."""

import numpy as np
import pytest
from PyRAY.library import constants
from PyRAY.library import den2freq
from PyRAY.library import find_X
from PyRAY.library import find_Y
from PyRAY.library import find_mu_mup
from PyRAY.library import find_vh
from PyRAY.library import freq2den
from PyRAY.library import regrid_to_nonuniform_grid
from PyRAY.library import smooth_nonuniform_grid
from PyRAY.library import vertical_to_magnetic_angle
from PyRAY.library import virtical_forward_operator


def test_constants_output():
    """Test that constants() returns expected constant values."""
    cp_expected = 8.97866275
    g_p_expected = 2.799249247e10

    cp, g_p = constants()

    assert isinstance(cp, float)
    assert isinstance(g_p, float)
    assert cp == pytest.approx(cp_expected, rel=1e-8)
    assert g_p == pytest.approx(g_p_expected, rel=1e-8)


def test_den2freq_scalar():
    """Test den2freq with a scalar input."""
    density = 1.0e12
    expected = np.sqrt(density) * 8.97866275
    result = den2freq(density)
    assert isinstance(result, float)
    assert result == pytest.approx(expected, rel=1e-8)


def test_den2freq_array():
    """Test den2freq with an array input."""
    density = np.array([1.0e12, 2.5e12, 0.0])
    expected = np.sqrt(density) * 8.97866275
    result = den2freq(density)
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, expected, rtol=1e-8)


def test_den2freq_negative_input():
    """Test that den2freq raises ValueError on negative input."""
    with pytest.raises(ValueError, match="Density must be non-negative"):
        den2freq(-1.0e11)


def test_freq2den_scalar():
    """Test freq2den with a scalar input."""
    frequency = 8.97866275e6
    expected = (frequency / 8.97866275) ** 2
    result = freq2den(frequency)
    assert isinstance(result, float)
    assert result == pytest.approx(expected, rel=1e-8)


def test_freq2den_array():
    """Test freq2den with an array input."""
    frequencies = np.array([8.97866275e6, 1.0e7, 0.0])
    expected = (frequencies / 8.97866275) ** 2
    result = freq2den(frequencies)
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, expected, rtol=1e-8)


def test_freq2den_negative_result_not_expected():
    """Test that freq2den doesn't produce negative densities for valid input."""
    frequencies = np.array([1.0e6, 5.0e6])
    result = freq2den(frequencies)
    assert np.all(result >= 0)


def test_find_X_scalar():
    """Test find_X with scalar inputs."""
    n_e = 1.0e12
    f = 1.0e7
    expected = ((np.sqrt(n_e) * 8.97866275) ** 2) / (f ** 2)
    result = find_X(n_e, f)
    assert isinstance(result, float)
    assert result == pytest.approx(expected, rel=1e-8)


def test_find_X_array():
    """Test find_X with array inputs."""
    n_e = np.array([1.0e12, 2.0e12])
    f = np.array([1.0e7, 2.0e7])
    expected = ((np.sqrt(n_e) * 8.97866275) ** 2) / (f ** 2)
    result = find_X(n_e, f)
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, expected, rtol=1e-8)


def test_find_Y_scalar():
    """Test find_Y with scalar inputs."""
    f = 1.0e7
    b = 5.0e-5
    g_p = 2.799249247e10
    expected = g_p * b / f
    result = find_Y(f, b)
    assert isinstance(result, float)
    assert result == pytest.approx(expected, rel=1e-8)


def test_find_Y_array():
    """Test find_Y with array inputs."""
    f = np.array([1.0e7, 2.0e7])
    b = np.array([5.0e-5, 4.0e-5])
    g_p = 2.799249247e10
    expected = g_p * b / f
    result = find_Y(f, b)
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, expected, rtol=1e-8)
