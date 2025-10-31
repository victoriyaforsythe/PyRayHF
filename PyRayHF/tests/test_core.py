#!/usr/bin/env python
"""Tests for core PyRayHF library functions."""

import numpy as np
import pytest

from copy import deepcopy
from lmfit import Parameters
from numpy.testing import assert_allclose
from PyRayHF.library import build_mup_function
from PyRayHF.library import build_refractive_index_interpolator_cartesian
from PyRayHF.library import build_refractive_index_interpolator_spherical
from PyRayHF.library import constants
from PyRayHF.library import den2freq
from PyRayHF.library import eval_refractive_index_and_grad
from PyRayHF.library import find_mu_mup
from PyRayHF.library import find_vh
from PyRayHF.library import find_X
from PyRayHF.library import find_Y
from PyRayHF.library import freq2den
from PyRayHF.library import minimize_parameters
from PyRayHF.library import model_VH
from PyRayHF.library import n_and_grad
from PyRayHF.library import regrid_to_nonuniform_grid
from PyRayHF.library import residual_VH
from PyRayHF.library import smooth_nonuniform_grid
from PyRayHF.library import tan_from_mu_scalar
from PyRayHF.library import trace_ray_cartesian_gradient
from PyRayHF.library import trace_ray_cartesian_snells
from PyRayHF.library import trace_ray_spherical_gradient
from PyRayHF.library import trace_ray_spherical_snells
from PyRayHF.library import vertical_forward_operator
from PyRayHF.library import vertical_to_magnetic_angle
from scipy.interpolate import RegularGridInterpolator
from unittest.mock import patch


def test_constants_output():
    """Test that constants function returns correct values."""
    pf_const, g_p, R_E, c_km_s = constants()
    assert np.isclose(pf_const, 8.97866275, rtol=1e-8)
    assert np.isclose(g_p, 2.799249247e10, rtol=1e-8)
    assert np.isclose(R_E, 6371., rtol=1e-8)
    assert np.isclose(c_km_s, 299_792.458, rtol=1e-8)


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
    n_points = 10

    result = regrid_to_nonuniform_grid(f, n_e, b, bpsi, aalt,
                                       mode='O',
                                       n_points=n_points)

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
    den = np.array([0, 0.5e12, 1e12])
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
    freq = np.array([3.0, 4.0, 4.5])
    alt = np.array([100, 200, 300])
    bmag = np.array([5e-5, 5e-5, 5e-5])
    bpsi = np.array([60.0, 60.0, 60.0])

    # Expected outputs
    expected_vh = np.array([174.32954286, 253.81598004, 301.25975233])
    expected_edp = np.array([5.39526842e+10,
                             2.81042885e+11,
                             6.66833260e+11])

    # Run the model
    vh, edp = model_VH(F2, F1, E, freq, alt, bmag, bpsi)

    # Compare results
    assert_allclose(vh, expected_vh, rtol=1e-6)
    assert_allclose(edp, expected_edp, rtol=1e-6)


def test_zero_residual_when_parameters_match():
    """Basic test for residual_VH with short arrays."""
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

    freq = np.array([3.0, 4.0, 4.5])
    alt = np.array([100, 200, 300])
    bmag = np.array([5e-5, 5e-5, 5e-5])
    bpsi = np.array([60.0, 60.0, 60.0])

    # Use the true model output as synthetic observations
    vh_obs, _ = model_VH(F2, deepcopy(F1), deepcopy(E), freq, alt, bmag,
                         bpsi)

    # Parameters that match the F2 inputs
    params = Parameters()
    params.add('NmF2', value=1.17848165e+12)
    params.add('hmF2', value=365.13828931)
    params.add('B_bot', value=41.26005561)

    # Compute residual
    residual = residual_VH(params, F2, F1, E, freq, vh_obs, alt, bmag, bpsi)

    # Expect near-zero residual
    assert_allclose(residual, np.zeros_like(vh_obs), rtol=1e-6, atol=1e-6)


def test_minimize_parameters_runs_and_returns_shapes():
    """Basic test for minimize_parameters with short arrays."""
    # Fake input dictionaries (with small 1x1x1 arrays)
    F2 = {"Nm": np.array([[[1e11]]]),
          "hm": np.array([[[300.0]]]),
          "B_bot": np.array([[[200.0]]])}
    F1 = {"Nm": np.array([[[1e10]]]),
          "hm": np.array([[[200.0]]]),
          "B_bot": np.array([[[100.0]]])}
    E = {"Nm": np.array([[[1e9]]]),
         "hm": np.array([[[120.0]]]),
         "B_bot": np.array([[[80.0]]])}

    # Input arrays
    f_in = np.array([5.0, 7.0, 10.0])  # MHz
    vh_obs = np.array([150.0, 200.0, 300.0])  # km
    alt = np.linspace(0, 400, 50)  # km
    b_mag = np.full_like(alt, 50000.0)  # nT
    b_psi = np.full_like(alt, 45.0)  # degrees

    # Patch freq2den, residual_VH, and model_VH so we don’t run heavy
    # physics
    with patch("PyRayHF.library.freq2den", return_value=1e11), \
         patch("PyRayHF.library.residual_VH",
               return_value=np.zeros_like(vh_obs)), \
         patch("PyRayHF.library.model_VH",
               return_value=(vh_obs, np.ones_like(alt))):

        vh_result, EDP_result = minimize_parameters(
            F2, F1, E, f_in, vh_obs, alt, b_mag, b_psi,
            method="brute", percent_sigma=10., step=1.0)

    # Check return types and shapes
    assert isinstance(vh_result, np.ndarray)
    assert isinstance(EDP_result, np.ndarray)

    assert vh_result.shape == vh_obs.shape
    assert EDP_result.shape == alt.shape

    # Check values come from mocked model_VH
    np.testing.assert_allclose(vh_result, vh_obs)
    np.testing.assert_allclose(EDP_result, np.ones_like(alt))


def test_n_and_grad_simple_field():
    """Basic test for n_and_grad_ with simple field."""
    # Define toy grid
    z_grid = np.linspace(0, 10, 6)  # 6 points
    x_grid = np.linspace(0, 10, 6)
    Z, X = np.meshgrid(z_grid, x_grid, indexing="ij")

    # Toy refractive index field: n(x,z) = x + 2z
    n_field = X + 2 * Z

    # Exact derivatives:
    # ∂n/∂x = 1
    # ∂n/∂z = 2
    dn_dx_exact = np.ones_like(n_field)
    dn_dz_exact = 2 * np.ones_like(n_field)

    # Interpolators
    n_interp = RegularGridInterpolator((z_grid, x_grid), n_field)
    dn_dx_interp = RegularGridInterpolator((z_grid, x_grid), dn_dx_exact)
    dn_dz_interp = RegularGridInterpolator((z_grid, x_grid), dn_dz_exact)

    # Test at some points
    x_test = np.array([1.0, 5.0, 9.0])
    z_test = np.array([2.0, 4.0, 8.0])

    n_val, dnx_val, dnz_val = n_and_grad(x_test, z_test,
                                         n_interp,
                                         dn_dx_interp,
                                         dn_dz_interp)

    # Expected values
    n_expected = x_test + 2 * z_test
    dnx_expected = np.ones_like(x_test)
    dnz_expected = 2 * np.ones_like(z_test)

    # Assertions with tolerance
    np.testing.assert_allclose(n_val, n_expected, rtol=1e-12)
    np.testing.assert_allclose(dnx_val, dnx_expected, rtol=1e-12)
    np.testing.assert_allclose(dnz_val, dnz_expected, rtol=1e-12)


def test_eval_refractive_index_and_grad_linear_field():
    """Basic test for eval_refractive_index_and_grad with linear field."""
    # Define toy grid
    z_grid = np.linspace(0, 10, 6)
    x_grid = np.linspace(0, 10, 6)
    Z, X = np.meshgrid(z_grid, x_grid, indexing="ij")

    # Define analytic refractive index: n(x,z) = 3*x + 4*z
    n_field = 3 * X + 4 * Z

    # Exact derivatives
    dn_dx = 3 * np.ones_like(n_field)
    dn_dz = 4 * np.ones_like(n_field)

    # Build interpolators
    n_interp = RegularGridInterpolator((z_grid, x_grid), n_field)
    dn_dx_interp = RegularGridInterpolator((z_grid, x_grid), dn_dx)
    dn_dz_interp = RegularGridInterpolator((z_grid, x_grid), dn_dz)

    # Test points (both scalars and arrays)
    x_test = np.array([1.0, 5.0, 9.0])
    z_test = np.array([2.0, 4.0, 8.0])

    n_val, dnx_val, dnz_val = eval_refractive_index_and_grad(
        x_test, z_test, n_interp, dn_dx_interp, dn_dz_interp
    )

    # Expected analytic results
    n_expected = 3 * x_test + 4 * z_test
    dnx_expected = np.full_like(x_test, 3.0)
    dnz_expected = np.full_like(z_test, 4.0)

    # Assertions
    np.testing.assert_allclose(n_val, n_expected, rtol=1e-12)
    np.testing.assert_allclose(dnx_val, dnx_expected, rtol=1e-12)
    np.testing.assert_allclose(dnz_val, dnz_expected, rtol=1e-12)


def test_eval_refractive_index_and_grad_broadcasting():
    """Basic test for eval_refractive_index_and_grad broadcasting."""
    # Small grid
    z_grid = np.linspace(0, 5, 3)
    x_grid = np.linspace(0, 5, 3)
    Z, X = np.meshgrid(z_grid, x_grid, indexing="ij")

    # n(x,z) = x - z
    n_field = X - Z
    dn_dx = np.ones_like(n_field)
    dn_dz = -np.ones_like(n_field)

    n_interp = RegularGridInterpolator((z_grid, x_grid), n_field)
    dn_dx_interp = RegularGridInterpolator((z_grid, x_grid), dn_dx)
    dn_dz_interp = RegularGridInterpolator((z_grid, x_grid), dn_dz)

    # Provide 2D mesh input
    x_test, z_test = np.meshgrid([1.0, 2.0], [3.0, 4.0])

    n_val, dnx_val, dnz_val = eval_refractive_index_and_grad(
        x_test, z_test, n_interp, dn_dx_interp, dn_dz_interp
    )

    # Expected
    n_expected = x_test - z_test
    dnx_expected = np.ones_like(x_test)
    dnz_expected = -np.ones_like(z_test)

    np.testing.assert_allclose(n_val, n_expected, rtol=1e-12)
    np.testing.assert_allclose(dnx_val, dnx_expected, rtol=1e-12)
    np.testing.assert_allclose(dnz_val, dnz_expected, rtol=1e-12)


def test_build_refractive_index_interpolator_cartesian_linear_field():
    """Basic test for build_refractive_index_interpolator with linear field."""
    # Define grids
    z_grid = np.linspace(0, 10, 6)
    x_grid = np.linspace(0, 10, 6)
    Z, X = np.meshgrid(z_grid, x_grid, indexing="ij")

    # Define analytic refractive index: n(x,z) = 2x + 3z
    n_field = 2 * X + 3 * Z

    # Build interpolator
    n_and_grad = build_refractive_index_interpolator_cartesian(z_grid,
                                                               x_grid,
                                                               n_field)

    # Test points
    x_test = np.array([0.0, 5.0, 10.0])
    z_test = np.array([0.0, 5.0, 10.0])

    n_val, dndx, dndz = n_and_grad(x_test, z_test)

    # Expected analytic results
    n_expected = 2 * x_test + 3 * z_test
    dndx_expected = np.full_like(x_test, 2.0)
    dndz_expected = np.full_like(z_test, 3.0)

    np.testing.assert_allclose(n_val, n_expected, rtol=1e-12)
    np.testing.assert_allclose(dndx, dndx_expected, rtol=1e-12)
    np.testing.assert_allclose(dndz, dndz_expected, rtol=1e-12)


def test_build_refractive_index_interpolator_cartesian_broadcasting():
    """Basic test for build_refractive_index_interpolator broadcasting."""
    # Small grid
    z_grid = np.linspace(0, 2, 3)
    x_grid = np.linspace(0, 2, 3)
    Z, X = np.meshgrid(z_grid, x_grid, indexing="ij")

    # n(x,z) = x - z
    n_field = X - Z

    n_and_grad = build_refractive_index_interpolator_cartesian(z_grid,
                                                               x_grid,
                                                               n_field)

    # Mesh input
    x_test, z_test = np.meshgrid([0.5, 1.5], [0.5, 1.5])

    n_val, dndx, dndz = n_and_grad(x_test, z_test)

    n_expected = x_test - z_test
    dndx_expected = np.ones_like(x_test)
    dndz_expected = -np.ones_like(z_test)

    np.testing.assert_allclose(n_val, n_expected, rtol=1e-12)
    np.testing.assert_allclose(dndx, dndx_expected, rtol=1e-12)
    np.testing.assert_allclose(dndz, dndz_expected, rtol=1e-12)


def test_build_refractive_index_interpolator_spherical_linear_field():
    """Test spherical μ(r,φ) interpolator on a simple linear refractive ind."""
    # Load Earth radius
    _, _, R_E, _ = constants()

    # Create linear refractive index field
    a, b = 1e-4, 2e-4  # linear coefficients
    z_grid = np.linspace(0, 400, 80)
    x_grid = np.linspace(0, 1000, 120)
    Z, X = np.meshgrid(z_grid, x_grid, indexing="ij")
    n_field = 1.0 + a * Z + b * X

    # Build spherical interpolator (handles conversion internally)
    n_and_grad = build_refractive_index_interpolator_spherical(
        z_grid=z_grid,
        x_grid=x_grid,
        n_field=n_field,
        R_E=R_E
    )

    # Pick some test points
    x_test = np.array([0.0, 250.0, 750.0])
    z_test = np.array([0.0, 200.0, 400.0])

    # Evaluate μ, ∂μ/∂r, ∂μ/∂φ
    mu_val, mu_r, mu_phi = n_and_grad(x_test / R_E, R_E + z_test)

    # Expected analytic values
    expected_mu = 1.0 + a * z_test + b * x_test
    expected_mu_r = np.full_like(expected_mu, a)
    expected_mu_phi = np.full_like(expected_mu, b * R_E)

    # Validate
    np.testing.assert_allclose(mu_val, expected_mu, rtol=1e-6)
    np.testing.assert_allclose(mu_r, expected_mu_r, rtol=1e-6)
    np.testing.assert_allclose(mu_phi, expected_mu_phi, rtol=1e-6)


def test_build_refractive_index_interpolator_spherical_broadcasting():
    """Test spherical interpolator with internal Cartesian→spherical."""
    # Get Earth radius
    _, _, R_E, _ = constants()

    # Build a simple test field: n = 1 + 1e-4*(z + x)
    z_grid = np.linspace(0, 500, 50)
    x_grid = np.linspace(0, 1000, 100)
    Z, X = np.meshgrid(z_grid, x_grid, indexing="ij")
    n_field = 1.0 + 1e-4 * (Z + X)

    # Build spherical interpolator (auto converts to r, φ)
    n_and_grad = build_refractive_index_interpolator_spherical(
        z_grid=z_grid,
        x_grid=x_grid,
        n_field=n_field,
        R_E=R_E
    )

    # Evaluate at a few arbitrary points
    x_test = np.array([0.0, 500.0, 1000.0])
    z_test = np.array([0.0, 250.0, 500.0])

    n_val, dn_dr, dn_dphi = n_and_grad(x_test / R_E, R_E + z_test)

    # --- Expected values ---
    # n = 1 + 1e-4 * (z + x)
    expected_n = 1.0 + 1e-4 * (z_test + x_test)
    expected_dn_dr = np.full_like(expected_n, 1e-4)  # d/dz term
    # ∂/∂φ = (∂n/∂x) * (dx/dφ) = 1e-4 * R_E
    expected_dn_dphi = np.full_like(expected_n, 1e-4 * R_E)

    np.testing.assert_allclose(n_val, expected_n, rtol=1e-6)
    np.testing.assert_allclose(dn_dr, expected_dn_dr, rtol=1e-6)
    np.testing.assert_allclose(dn_dphi, expected_dn_dphi, rtol=1e-6)


def test_tan_from_mu_scalar_basic():
    """Test tan_from_mu_scalar against analytic expectation for μ and p."""
    # For μ = 2, p = 1 → tanθ = p / sqrt(μ^2 - p^2) = 1 / sqrt(3)
    mu_val = 2.0
    p = 1.0
    expected = 1.0 / np.sqrt(3.0)
    result = tan_from_mu_scalar(mu_val, p)
    np.testing.assert_allclose(result, expected, rtol=1e-12)

    # Test near the singularity (μ → p)
    mu_val = 1.0000001
    result = tan_from_mu_scalar(mu_val, 1.0)
    assert np.isfinite(result)
    assert result > 0.0


def test_tan_from_mu_scalar_behavior_near_zero():
    """tan_from_mu_scalar should handle near-zero μ gracefully."""
    mu_val = 1e-6
    p = 1e-7
    result = tan_from_mu_scalar(mu_val, p)
    assert np.isfinite(result)
    assert result >= 0.0


def test_find_X_basic():
    """Validate plasma frequency parameter X = (f_p / f)^2."""

    # Get constants from the library
    pf_const, _, _, _ = constants()

    # Set up test data
    f_Hz = 10e6  # 10 MHz
    Ne = np.array([1e11, 1e12])  # electron density [m^-3]

    # Expected formula: X = (f_p / f)^2 = (pf_const * sqrt(Ne) / f)^2
    expected = (pf_const * np.sqrt(Ne) / f_Hz) ** 2

    # Compute using library
    result = find_X(Ne, f_Hz)

    # Verify match
    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_find_Y_basic():
    """Validate magnetoionic parameter Y = f_H / f using library constants."""

    # Get constants from the library
    _, g_p, _, _ = constants()

    # Test data
    f_Hz = 10e6  # 10 MHz
    Babs = np.array([1e-5, 5e-5])  # magnetic field [T]

    # Expected: Y = f_H / f = (g_p * B) / f
    expected = (g_p * Babs) / f_Hz

    # Compute using library
    result = find_Y(f_Hz, Babs)

    # Compare with tight tolerance
    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_find_mu_mup_ordinary_mode():
    """Validate O-mode (μ, μ′) behavior for small X and Y.

    For weakly magnetized plasma (small Y):
      • μ ≈ sqrt(1 - X)
      • μ′ (group index) ≥ μ and close to 1 for small X,Y
    This test checks for physically consistent relationships,
    not exact numeric identity.

    """
    # Inputs
    X = np.array([0.1, 0.2])
    Y = np.array([0.01, 0.02])
    bpsi = np.array([0.0, np.pi / 4])

    mu, mup = find_mu_mup(X, Y, bpsi, mode="O")

    mu_expected = np.sqrt(1 - X)

    # --- Assertions ---
    # μ should be close to analytic limit
    np.testing.assert_allclose(mu, mu_expected, rtol=5e-2)

    # μ′ must always be >= μ (group slower than phase)
    assert np.all(mup >= mu), "Group index should be >= phase index"

    # μ′ must remain finite and near unity for small X, Y
    assert np.all((mup > 0.8) & (mup < 1.5)), "Group index out of range"


def test_find_mu_mup_extraordinary_mode_differs():
    """X-mode should differ slightly from O-mode."""
    X = np.array([0.1])
    Y = np.array([0.02])
    bpsi = np.array([np.pi / 3])

    mu_O, mup_O = find_mu_mup(X, Y, bpsi, mode="O")
    mu_X, mup_X = find_mu_mup(X, Y, bpsi, mode="X")

    # Both must be positive but not identical
    assert np.all(mu_X > 0)
    assert np.all(mup_X > 0)
    assert not np.allclose(mu_O, mu_X)
    assert not np.allclose(mup_O, mup_X)


def test_trace_ray_cartesian_snells_basic():
    """Verify Snell's-law Cartesian raytracer produces finite output."""
    # --- Test profile setup ---
    alt_km = np.linspace(0, 600, 200)
    Ne = 1e12 * np.exp(-(alt_km - 250)**2 / (2 * 60**2))
    Babs = np.full_like(alt_km, 4e-5)  # Tesla
    bpsi = np.full_like(alt_km, 45.0)  # degrees

    f0_Hz = 10e6
    elevation_deg = 45.0

    # --- Run the tracer ---
    result = trace_ray_cartesian_snells(
        f0_Hz=f0_Hz,
        elevation_deg=elevation_deg,
        alt_km=alt_km,
        Ne=Ne,
        Babs=Babs,
        bpsi=bpsi,
        mode="O")

    # --- Structure checks ---
    expected_keys = {
        "x", "z", "group_path_km", "group_delay_sec",
        "x_midpoint", "z_midpoint", "ground_range_km"
    }
    assert expected_keys.issubset(result.keys()), "Missing keys in result dict"

    # --- Finite values ---
    assert np.all(np.isfinite(result["x"])), "x contains NaN"
    assert np.all(np.isfinite(result["z"])), "z contains NaN"
    assert np.isfinite(result["group_path_km"])
    assert np.isfinite(result["group_delay_sec"])
    assert np.isfinite(result["ground_range_km"])

    # --- Physically meaningful ---
    assert result["group_path_km"] > 0
    assert result["group_delay_sec"] > 0
    assert result["ground_range_km"] > 0

    # --- Geometry sanity ---
    z = result["z"]
    assert np.isclose(z[0], 0.0, atol=1e-3), "Ray must start at ground"
    assert np.nanmax(z) > 50.0, "Ray should reach reasonable altitude"
    assert np.isclose(z[-1], 0.0, atol=1e-2), "Ray must return to ground"


def test_cartesian_snells_vs_gradient_consistency():
    """Compare Snells and gradient Cartesian tracers in a uniform medium."""
    # --- Atmospheric & ray parameters ---
    alt_km = np.linspace(0, 600, 200)
    Ne = 1e12 * np.exp(-(alt_km - 250)**2 / (2 * 60**2))
    Babs = np.full_like(alt_km, 4e-5)  # Tesla
    bpsi = np.full_like(alt_km, 45.0)  # degrees
    f0_Hz = 10e6
    elevation_deg = 45.0
    mode = "O"

    # --- Plasma parameters ---
    nx = 200
    xmax = 1000
    x_grid = np.linspace(0, xmax, nx)
    z_grid = alt_km
    Xg, Zg = np.meshgrid(x_grid, z_grid)
    Ne_grid = np.tile(Ne[:, np.newaxis], (1, nx))
    Babs_grid = np.tile(Babs[:, np.newaxis], (1, nx))
    bpsi_grid = np.tile(bpsi[:, np.newaxis], (1, nx))

    X = find_X(Ne_grid, f0_Hz)
    Y = find_Y(f0_Hz, Babs_grid)
    mu, mup = find_mu_mup(X, Y, bpsi_grid, mode)

    # --- Build interpolators ---
    n_and_grad_cartesian = build_refractive_index_interpolator_cartesian(
        z_grid,
        x_grid,
        mu)

    mup_func_cartesian = build_mup_function(
        mup_field=mup,
        x_grid=x_grid,
        z_grid=z_grid,
        geometry="cartesian")

    # --- Run both raytracers ---
    result_snell = trace_ray_cartesian_snells(
        f0_Hz=f0_Hz,
        elevation_deg=elevation_deg,
        alt_km=alt_km,
        Ne=Ne,
        Babs=Babs,
        bpsi=bpsi,
        mode=mode)

    result_grad = trace_ray_cartesian_gradient(
        n_and_grad=n_and_grad_cartesian,
        x0_km=0.0,
        z0_km=0.0,
        elevation_deg=elevation_deg,
        s_max_km=4000.0,
        max_step_km=5.0,
        z_max_km=600.0,
        x_min_km=0.0,
        x_max_km=1000.0,
        mup_func=mup_func_cartesian)

    # --- Consistency checks ---
    for key in ["group_path_km", "group_delay_sec", "ground_range_km"]:
        v1, v2 = result_snell[key], result_grad[key]
        rel_err = abs(v1 - v2) / max(abs(v1), abs(v2))
        assert rel_err < 0.04, f"{key} mismatch >4% ({rel_err * 100:.2f}%)"

    # --- Geometry sanity ---
    assert np.nanmax(result_snell["z"]) > 100.0
    assert np.nanmax(result_grad["z"]) > 100.0
    assert np.isclose(result_snell["z"][-1], 0.0, atol=1e-2)
    assert np.isclose(result_grad["z"][-1], 0.0, atol=1e-2)


def test_spherical_snells_flat_earth_limit():
    """Verify spherical Snell's-law reduces to Cartesian for large R_E."""
    # --- Atmospheric and ray parameters ---
    alt_km = np.linspace(0, 600, 200)
    Ne = 1e12 * np.exp(-(alt_km - 250)**2 / (2 * 60**2))
    Babs = np.full_like(alt_km, 4e-5)  # Tesla
    bpsi = np.full_like(alt_km, 45.0)  # degrees
    f0_Hz = 10e6
    elevation_deg = 50.0
    mode = "O"

    # --- Flat-Earth reference (Cartesian Snell's law) ---
    result_cart = trace_ray_cartesian_snells(
        f0_Hz=f0_Hz,
        elevation_deg=elevation_deg,
        alt_km=alt_km,
        Ne=Ne,
        Babs=Babs,
        bpsi=bpsi,
        mode=mode,
    )

    # --- Spherical Snell's law with huge Earth radius (≈ flat) ---
    result_sph = trace_ray_spherical_snells(
        f0_Hz=f0_Hz,
        elevation_deg=elevation_deg,
        alt_km=alt_km,
        Ne=Ne,
        Babs=Babs,
        bpsi=bpsi,
        mode=mode,
        R_E=6371e9,   # effectively infinite curvature radius
    )

    # --- Key metrics should match within tight tolerance ---
    for key in ["group_path_km", "group_delay_sec", "ground_range_km"]:
        v_cart, v_sph = result_cart[key], result_sph[key]
        rel_err = abs(v_cart - v_sph) / max(abs(v_cart), abs(v_sph))
        assert rel_err < 0.03, f"{key} mismatch >3% ({rel_err * 100:.2f}%)"

    # --- Geometry sanity ---
    assert np.nanmax(result_cart["z"]) > 100.0
    assert np.nanmax(result_sph["z"]) > 100.0
    assert np.isclose(result_cart["z"][-1], 0.0, atol=1e-3)
    assert np.isclose(result_sph["z"][-1], 0.0, atol=1e-3)


def test_trace_ray_spherical_gradient_basic():
    """Test spherical gradient-based ray tracing on a simple plasma profile.

    This validates numerical integration of the full spherical gradient
    equations using μ (phase refractive index) for geometry and μ' (group
    index) for delay. The ray should rise, bend smoothly, and return to
    ground—producing reasonable path and delay metrics.

    The test ensures:
      - Returned arrays are finite and monotonic in arclength.
      - Ground return occurs within expected horizontal range.
      - Group path and delay are positive and physically plausible.

    """
    # --- Basic setup ---
    pf_const, gp, R_E, c_km_s = constants()
    f0_Hz = 10e6
    elevation_deg = 45.
    mode = "O"

    alt_km = np.linspace(0, 600, 200)
    Ne = 1e12 * np.exp(-(alt_km - 250)**2 / (2 * 60**2))
    Babs = np.full_like(alt_km, 4e-5)  # Tesla
    bpsi = np.full_like(alt_km, 45.0)  # degrees

    # Create 2D density grid (Ne assumed horizontally uniform)
    nx = 500
    xmax = 1000
    x_grid = np.linspace(0, xmax, nx)
    z_grid = alt_km
    Xg, Zg = np.meshgrid(x_grid, z_grid)
    Ne_grid = np.tile(Ne[:, np.newaxis], (1, nx))
    Babs_grid = np.tile(Babs[:, np.newaxis], (1, nx))
    bpsi_grid = np.tile(bpsi[:, np.newaxis], (1, nx))

    # Calculate plasma parameters
    X = find_X(Ne_grid, f0_Hz)
    Y = find_Y(f0_Hz, Babs_grid)
    mu, mup = find_mu_mup(X, Y, bpsi_grid, mode)
    mup = np.where((mup < 1e-3) | np.isnan(mup), np.nan, mup)

    # Build interpolator from spherical μ grid
    n_and_grad_spherical = build_refractive_index_interpolator_spherical(
        z_grid,
        x_grid,
        mu)

    mup_func_spherical = build_mup_function(mup_field=mup,
                                            x_grid=x_grid,
                                            z_grid=z_grid,
                                            geometry="spherical")

    # --- Trace ray ---
    result = trace_ray_spherical_gradient(
        n_and_grad_rphi=n_and_grad_spherical,
        x0_km=0.0,
        z0_km=0.0,
        elevation_deg=elevation_deg,
        s_max_km=4000.0,
        R_E=None,
        mup_func=mup_func_spherical,
    )

    # --- Assertions ---
    assert result["status"] in {"ground", "success", "length"}
    assert np.all(np.isfinite(result["x"]))
    assert np.all(np.isfinite(result["z"]))
    assert result["group_path_km"] > 0
    assert result["group_delay_sec"] > 0
    assert 500 < result["ground_range_km"] < 800

    # Physical consistency
    rel_err = abs(result["group_delay_sec"]
                  * c_km_s / result["group_path_km"] - 1)
    assert rel_err < 0.2, f"Delay-path consistency off by {rel_err * 100:.2f}%"
