#!/usr/bin/env python
# --------------------------------------------------------
# Distribution statement A. Approved for public release.
# Distribution is unlimited.
# This work was supported by the Office of Naval Research.
# --------------------------------------------------------
"""This library contains components for PyRayHF software.

"""

# Standard library
from copy import deepcopy
from functools import partial

# Third-party
import lmfit
import numpy as np
import PyIRI
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator

# Typing
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple

# Local
from PyRayHF import logger


def constants():
    """Define constants for virtual height calculation.

    Returns
    -------
    cp : float
        Plasma frequency constant in Hz per sqrt(m^-3).
        f_p [Hz] = cp * sqrt(n_e [m^-3]).
    g_p : float
        Electron gyrofrequency constant in Hz/T (f_ce = g_p * B).
    R_E : float
        Earth radius [km].
    c_km_s : float
        Speed of light [km/s].

    Notes
    -----
    This function provides constants used in virtual height calculations.

    """
    # Constant to convert density (m-3) to frequency (Hz)
    cp = 8.97866275

    # Proton gyrofrequency constant (Hz/T)
    g_p = 2.799249247e10

    # Earth radius (km)
    R_E = 6371.

    # Speed of light (km/s)
    c_km_s = 299_792.458

    return cp, g_p, R_E, c_km_s


def den2freq(density):
    """Convert plasma density to plasma frequency.

    Parameters
    ----------
    density : float or array-like
        Plasma density in m^-3.

    Returns
    -------
    frequency : float or array-like
        Plasma frequency in Hz.

    """
    # Declaring constants
    cp, _, _, _ = constants()

    # Test for negative input
    if np.any(np.asarray(density) < 0):
        raise ValueError("Density must be non-negative")

    frequency = np.sqrt(density) * cp
    return frequency


def freq2den(frequency):
    """Convert plasma frequency to plasma density.

    Parameters
    ----------
    frequency : float or array-like
        Plasma frequency in Hz.

    Returns
    -------
    density : float or array-like
        Plasma density in m^-3.

    """
    # Declaring constants
    cp, _, _, _ = constants()
    density = (frequency / cp)**2
    return density


def find_X(n_e, f):
    """Calculate the square of the plasma freq over the square of the ion freq.

    Parameters
    ----------
    n_e : array-like
        Electron density in m^-3.
    f : array-like
        Ionosonde frequency in Hz.

    Returns
    -------
    X : float or array-like
        Ratio (f_N / f)^2.

    """
    X = (den2freq(n_e))**2 / f**(2)
    return X


def find_Y(f, b):
    """Calculate Y: the electron gyrofrequency to ionosonde frequency ratio.

    Parameters
    ----------
    f : array-like
        Ionosonde frequency in Hz.
    b : array-like
        Magnetic field magnitude in Tesla.

    Returns
    -------
    Y : array-like
        Electron gyrofrequency / ionosonde frequency.

    """
    _, g_p, _, _ = constants()
    Y = g_p * b / f
    return Y


def find_mu_mup(X, Y, bpsi, mode,
                *,
                y_tol: float = 1e-12,) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the phase and group refractive indices (μ and μ′).

    Parameters
    ----------
    X : array-like
        Ratio of plasma and transmission frequencies squared.
    Y : array-like
        Ratio of electron gyrofrequency and transmission frequency.
    bpsi : array-like
        Angle ψ between wave vector and magnetic field in degrees.
    mode : str
        'O' for ordinary or 'X' for extraordinary wave mode.
    y_tol : flt
        Tollerance to determine if plasma is magnetized or not.

    Returns
    -------
    mu : array-like
        Phase refractive index μ.
    mup : array-like
        Group refractive index μ′.

    Notes
    -----
    When Y ≈ 0 (unmagnetized plasma), use isotropic formulas:
    mu = sqrt(1 - X)   for X < 1, else NaN
    mup = 1 / mu
    Otherwise, fall through to the existing magnetized Appleton-Hartree logic.

    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    bpsi = np.asarray(bpsi, dtype=float)

    # Unmagnetized case
    # If cyclotron term is negligible everywhere, use isotropic cold-plasma
    # result.
    if np.nanmax(np.abs(Y)) < y_tol:
        mu2 = 1.0 - X
        # phase index: NaN at/above critical (X >= 1)
        mu = np.where(mu2 > 0.0, np.sqrt(mu2), np.nan)
        # group index: c / vg = 1 / mu (for collisionless plasma)
        mup = np.where(np.isfinite(mu) & (mu > 0.0), 1.0 / mu, np.nan)
        return mu, mup

    # Compute transverse and longitudinal components of Y
    YT = Y * np.sin(np.deg2rad(bpsi))
    YL = Y * np.cos(np.deg2rad(bpsi))

    # Compute 1-X
    Xm1 = 1.0 - X

    # Calculate alpha and beta as intermediate terms for refractive index
    alpha = 0.25 * YT**4 + YL**2 * Xm1**2
    beta = np.sqrt(alpha)

    # Set mode multiplier depending on propagation mode
    if mode == "O":
        modeMult = 1.
    if mode == "X":
        modeMult = -1.
    if (mode != "O") & (mode != "X"):
        raise ValueError("Mode must be O or X")

    # Appleton-Hartree denominator and mu
    D = Xm1 - 0.5 * YT**2 + modeMult * beta

    # Select > 0 part
    under_sqrt = 1. - X * Xm1 / D
    under_sqrt[under_sqrt < 0] = np.nan
    mu = np.sqrt(under_sqrt)

    # Apply physical constraints on refractive index
    mu[np.where(mu < 0.)] = 0.
    mu[np.where(mu > 1.)] = np.nan

    # Derivatives with respect to X and Y
    dbetadX = -YL**2 * Xm1 / beta
    dDdX = -1. + modeMult * dbetadX

    dalphadY = ((YT**3 * np.sin(np.deg2rad(bpsi)))
                + (2. * YL * Xm1**2 * np.cos(np.deg2rad(bpsi))))
    dbetadY = 0.5 * dalphadY / beta
    dDdY = -YT * np.sin(np.deg2rad(bpsi)) + modeMult * dbetadY

    # Compute partial derivatives of mu for corrected index
    dmudY = (X * Xm1 * dDdY) / (2. * mu * D**2)
    dmudX = (1. / (2. * mu * D)) * (2. * X - 1. + X * Xm1 / D * dDdX)

    # Modified refractive index considering dispersion effects
    mup = mu - (2. * X * dmudX + Y * dmudY)

    return mu, mup


def find_vh(X, Y, bpsi, dh, alt_min, mode):
    """Calculate virtual height for given mode.

    Parameters
    ----------
    X : array-like
        Plasma to ionosonde frequency ratio squared.
    Y : array-like
        Electron gyrofrequency to ionosonde frequency ratio.
    bpsi : array-like
        Angle between wave vector and magnetic field (degrees).
    dh : array-like
        Vertical layer thickness in km.
    alt_min : float
        Minimum altitude in km.
    mode : str
        "O" or "X" mode.

    Returns
    -------
    vh : array-like
        Virtual height in km.

    """
    # Find the phase refractive index μ and the group refractive index μ′ for
    # ordinary (O) and extraordinary (X) modes of signal propagation
    _, mup = find_mu_mup(X, Y, bpsi, mode)

    # Find virtual height as vertical integral through μ′
    vh = np.nansum(mup * dh, axis=1) + alt_min
    return vh


def smooth_nonuniform_grid(start, end, n_points, sharpness):
    """Generate smooth non-uniform grid from start to end.

    Parameters
    ----------
    start : float
    end : float
    n_points : int
    sharpness : float
        Controls how sharply resolution increases near `end`.

    Returns
    -------
    x : ndarray
        Non-uniformly spaced grid.

    """
    # Uniform grid [0, 1]
    u = np.linspace(0.0, 1.0, n_points)

    # Flip to make fine resolution near 'end'
    flipped_u = 1.0 - u

    factor = (np.exp(sharpness * flipped_u) - 1.0) / (np.exp(sharpness) - 1.0)
    x = 1. - (start + (end - start) * factor)
    return x


def regrid_to_nonuniform_grid(f, n_e, b, bpsi, aalt, npoints):
    """Regrid profile to smooth non-uniform vertical grid.

    Parameters
    ----------
    f : array-like
        Ionosonde frequency in Hz.
    n_e : array-like
        Electron density in m^-3.
    b : array-like
        Magnetic field magnitude.
    bpsi : array-like
        Angle to magnetic field vector in degrees.
    aalt : array-like
        Altitude profile in km.
    npoints : int
        Points in new vertical grid.

    Returns
    -------
    regridded : dict
        Dictionary with re-gridded arrays

    """
    # Create non-regular grid that has low resolution near zero and hight
    # resolution near one
    start = 0
    end = 1
    sharpness = 10.
    multiplier = smooth_nonuniform_grid(start, end, npoints, sharpness)

    N_grid = multiplier.size
    N_freq = f.size
    ind_grid = np.arange(0, N_grid, 1)

    # Limit input arrays to the fof2 of the ionosphere
    ind_max = np.argmax(n_e)
    n_e = n_e[0: ind_max]
    b = b[0: ind_max]
    bpsi = bpsi[0: ind_max]
    aalt = aalt[0: ind_max]

    # How close to the reflection height do we want to get
    dh = 1e-20

    # An array of critical height for the given ionosonde frequency
    # We subtract dh so that the critical height is not exactly reached
    critical_height = np.interp(f, den2freq(n_e), aalt) - dh

    # Make arrays 2-D
    multiplier_2d = np.full((N_freq, N_grid), multiplier)
    critical_height_2d = np.transpose(np.full((N_grid, N_freq),
                                              critical_height))
    new_alt_2d = multiplier_2d * (critical_height_2d - aalt[0]) + aalt[0]

    dh_2d = np.concatenate((np.diff(new_alt_2d, axis=1), np.full((N_freq, 1),
                                                                 dh)), axis=1)

    new_ind_2d = np.full((N_freq, N_grid), ind_grid)

    # Flattened array of new altitudes where we want to sample the density
    # profile, so we can apply 1-D Numpy interpolation, that is faster
    new_alt_1d = np.reshape(new_alt_2d, new_ind_2d.size)
    # Create arrays on the modified grid
    den_mod = np.reshape(np.interp(new_alt_1d, aalt, n_e), new_alt_2d.shape)
    bmag_mod = np.reshape(np.interp(new_alt_1d, aalt, b), new_alt_2d.shape)
    bpsi_mod = np.reshape(np.interp(new_alt_1d, aalt, bpsi), new_alt_2d.shape)
    ionosonde_freq_mod = np.transpose(np.full((N_grid, N_freq), f))

    # Create a dictionary to hold the new re-gridded arrays
    regridded = {'freq': ionosonde_freq_mod,
                 'den': den_mod,
                 'bmag': bmag_mod,
                 'bpsi': bpsi_mod,
                 'dist': dh_2d,
                 'alt': new_alt_2d,
                 'crit_height': critical_height_2d,
                 'ind': new_ind_2d}
    return regridded


def vertical_to_magnetic_angle(inclination_deg):
    """Calculate angle between vertical and magnetic field vector.

    Parameters
    ----------
    inclination_deg : float or ndarray
        Magnetic inclination in degrees (positive = downward).

    Returns
    -------
    vertical_angle : float or ndarray
        Angle between vertical and magnetic field in degrees.

    """
    vertical_angle = 90.0 - np.abs(inclination_deg)
    return vertical_angle


def vertical_forward_operator(freq, den, bmag, bpsi, alt, mode, n_points=2000):
    """Calculate virtual height from ionosonde freq and ion profile.

    Parameters
    ----------
    freq : array-like
        Frequency in MHz.
    den : array-like
        Electron density in m^-3.
    bmag : array-like
        Magnetic field magnitude in Tesla.
    bpsi : array-like
        Angle to magnetic field vector.
    alt : array-like
        Altitude profile in km.
    mode : str
        "O" or "X" propagation mode.
    n_points : int
        Number of vertical grid points.

    Returns
    -------
    vh : ndarray
        Virtual height in km.

    """
    # Check that input arrays have the same size
    if (den.shape != bmag.shape != bpsi.shape != alt.shape):
        logger.error("Error: freq, den, bmag, bpsi, alt should have same size")

    # Limit the ionosonde frequency array up tp the ionospheric critical
    # frequency foF2 (result is in Hz).
    foF2 = np.max(den2freq(den))

    # Index where ionosonde frequency is less then foF2 value
    ind = np.where((freq * 1e6) < foF2)

    # Select ionosonde frequency with this criteria
    freq_lim = freq[ind] * 1e6

    # Make empty array to collect virtual height of the same size as input
    # frequency array
    vh = np.zeros((freq.size)) + np.nan

    # Interpolate input arrays into a new stretched grid based on the
    # reflective height for each ionosonde frequency
    # Frequency needs to be converted to MHz from Hz
    regridded = regrid_to_nonuniform_grid(freq_lim,
                                          den,
                                          bmag,
                                          bpsi,
                                          alt,
                                          n_points)

    # Find the ratio of the square of the plasma frequency f_N to the square of
    # the ionosonde frequency f.
    aX = find_X(regridded['den'], regridded['freq'])

    # Find the ratio of electron gyrofrequency and the ionosonde frequency
    aY = find_Y(regridded['freq'], regridded['bmag'])

    # Find virtual height
    vh[ind] = find_vh(aX, aY, regridded['bpsi'], regridded['dist'],
                      np.min(alt), mode)
    return vh


def model_VH(F2, F1, E, f_in, alt, b_mag, b_psi):
    """Compute vertical virtual height using a modeled EDP and raytrace.

    Parameters
    ----------
    F2 : dict
        Dictionary of F2 layer parameters. Must include:
        - 'Nm': peak electron density (NmF2)
        - 'hm': peak height (hmF2)
        - 'B_bot': thickness of the bottomside of the F2 layer
    F1 : dict
        Dictionary of F1 layer parameters. Must include:
        - 'P': shape factor or profile parameter
    E : dict
        Dictionary of E layer parameters. Must include:
        - 'hm': peak height of the E layer
    f_in : ndarray
        Input frequency [MHz].
    alt : ndarray
        1D array of altitudes [km].
    b_mag : ndarray
        1D array of magnetic field magnitudes [nT].
    b_psi : ndarray
        1D array of magnetic field dip angles [rad].

    Returns
    -------
    vh_O : ndarray
        Virtual height trace (O-mode) [km].
    EDP : ndarray
        Reconstructed electron density profile [m^-3].

    """
    # Using PyIRI formalizm update the F1 layer parameters, in case F2
    # parameters have changed
    (NmF1,
     foF1,
     hmF1,
     B_F1_bot) = PyIRI.edp_update.derive_dependent_F1_parameters(F1['P'],
                                                                 F2['Nm'],
                                                                 F2['hm'],
                                                                 F2['B_bot'],
                                                                 E['hm'])

    # Update F1 with derived values
    F1['Nm'] = NmF1
    F1['hm'] = hmF1
    F1['fo'] = foF1
    F1['B_bot'] = B_F1_bot

    # Reconstruct electron density profile
    EDP = PyIRI.edp_update.reconstruct_density_from_parameters_1level(F2,
                                                                      F1,
                                                                      E,
                                                                      alt)
    EDP = EDP[0, :, 0]

    # Set ray-tracing parameters
    mode = "O"
    n_points = 200

    # Run vertical raytracing using PyRayHF
    vh_O = vertical_forward_operator(f_in, EDP,
                                     b_mag, b_psi,
                                     alt, mode, n_points)
    return vh_O, EDP


def residual_VH(params, F2_init, F1_init, E_init, f_in, vh_obs, alt, b_mag,
                b_psi):
    """Compute the residual between observed and modeled virtual heights.

    Parameters
    ----------
    params : lmfit.Parameters
        Parameters to be optimized, containing:
        - 'NmF2': peak electron density of F2 layer
        - 'hmF2': peak height of F2 layer
        - 'B_bot': thickness of F2 bottomside
    F2_init : dict
        Initial F2 layer parameters.
    F1_init : dict
        Initial F1 layer parameters.
    E_init : dict
        Initial E layer parameters.
    f_in : float
        Input frequency [MHz].
    vh_obs : ndarray
        Observed virtual heights [km].
    alt : ndarray
        Altitude array [km].
    b_mag : ndarray
        Magnetic field magnitude array [nT].
    b_psi : ndarray
        Magnetic field dip angle array [rad].

    Returns
    -------
    residual : ndarray
        Flattened array of residuals between observed and modeled virtual
        heights [km].

    """
    # Work on deep copies to avoid mutating originals
    F2 = deepcopy(F2_init)
    F1 = deepcopy(F1_init)
    E = deepcopy(E_init)

    # Update F2 parameters from optimization values
    F2['Nm'] = np.full_like(F2_init['Nm'], params['NmF2'].value)
    F2['hm'] = np.full_like(F2_init['Nm'], params['hmF2'].value)
    F2['B_bot'] = np.full_like(F2_init['Nm'], params['B_bot'].value)

    # Run forward model
    vh_model, _ = model_VH(F2, F1, E, f_in, alt, b_mag, b_psi)
    residual = (vh_obs - vh_model).ravel()
    return residual


def minimize_parameters(F2, F1, E, f_in, vh_obs, alt, b_mag, b_psi,
                        method='brute', percent_sigma=20., step=1.):
    """Minimize F2 layer parameters (hmF2 and B_bot) to fit observed VH.

    Parameters
    ----------
    F2 : dict
        Initial F2 layer parameters. Must include 'Nm', 'hm', and 'B_bot'.
    F1 : dict
        Initial F1 layer parameters.
    E : dict
        Initial E layer parameters.
    f_in : ndarray
        Input frequencies [MHz].
    vh_obs : ndarray
        Observed virtual heights [km].
    alt : ndarray
        Altitude array [km].
    b_mag : ndarray
        Magnetic field magnitude array [nT].
    b_psi : ndarray
        Magnetic field dip angle array [degrees].
    method : str
        Method of minimization in lmfit:
        "brute": (default) A grid search method for finding a global
        minimum.
        "levenberg-marquardt": Generally fast and
        effective for many curve-fitting needs.
        "powell": Another derivative-free method.
    percent_sigma : flt
        How far off from the background value to deviate.
        Default is 20%.
        If the speed needs to be increase, decrease this parameter.
    step : flt
        Step size in km for brute minimization.
        If the speed needs to be increase, increase this parameter.

    Returns
    -------
    vh_result : ndarray
        Virtual height after parameter fitting [km].
    EDP_result : ndarray
        Reconstructed electron density profile after fitting [m^-3].

    """
    # Use the last valid (finite) value in vh_obs to estimate initial NmF2
    ind_valid = np.where(np.isfinite(vh_obs))[0][-1]
    # Convert plasma frequency to plasma density and increase it slightly
    # (by 0.01%) to make sure that we can obtain the virtual height for the
    # last data point
    NmF2_new = freq2den(f_in[ind_valid] * 1e6) * 1.0001

    # The input arrays in F2 have shape [1, 1, 1], let's use mean to make it
    # just one number
    mean_hmF2 = np.nanmean(F2['hm'])
    mean_B_bot = np.nanmean(F2['B_bot'])

    # Brute minimization gives the best result
    # The brute step controls the walk
    # If you need to make the code faster, increase the brute_step and
    # decrease the percent sigma
    sigma_hmF2 = mean_hmF2 * (percent_sigma / 100.0)
    sigma_B_bot = mean_B_bot * (percent_sigma / 100.0)

    # Populate lmfit parameters for the minimization
    params = lmfit.Parameters()
    params.add('NmF2', value=NmF2_new, vary=False)
    params.add('hmF2', value=mean_hmF2,
               min=mean_hmF2 - sigma_hmF2,
               max=mean_hmF2 + sigma_hmF2,
               brute_step=step)

    params.add('B_bot', value=mean_B_bot,
               min=mean_B_bot - sigma_B_bot,
               max=mean_B_bot + sigma_B_bot,
               brute_step=step)

    # Perform brute-force minimization
    brute_result = lmfit.minimize(residual_VH, params,
                                  args=(F2, F1, E, f_in, vh_obs,
                                        alt, b_mag, b_psi),
                                  method=method)

    # Extract optimal parameter values
    NmF2_opt = brute_result.params['NmF2'].value
    hmF2_opt = brute_result.params['hmF2'].value
    B_bot_opt = brute_result.params['B_bot'].value

    # Update F2 dictionary with optimized parameters
    F2_fit = deepcopy(F2)
    F1_fit = deepcopy(F1)
    E_fit = deepcopy(E)
    F2_fit['Nm'] = np.full_like(F2['Nm'], NmF2_opt)
    F2_fit['hm'] = np.full_like(F2['Nm'], hmF2_opt)
    F2_fit['B_bot'] = np.full_like(F2['Nm'], B_bot_opt)

    # Run forward model with optimized parameters
    vh_result, EDP_result = model_VH(F2_fit, F1_fit, E_fit,
                                     f_in, alt, b_mag, b_psi)
    return vh_result, EDP_result


def n_and_grad(x: np.ndarray,
               z: np.ndarray,
               n_interp: RegularGridInterpolator,
               dn_dx_interp: RegularGridInterpolator,
               dn_dz_interp: RegularGridInterpolator,) -> Tuple[np.ndarray,
                                                                np.ndarray,
                                                                np.ndarray]:
    """Evaluate n, ∂n/∂x, and ∂n/∂z at points (x, z).

    Parameters
    ----------
    x : array_like
        Horizontal positions [km]; scalar or array.

    z : array_like
        Altitudes [km]; scalar or array.

    Returns
    -------
    n : np.ndarray
        Refractive index at (x, z). Shape equals the broadcasted shape of
        inputs.

    dndx : np.ndarray
        Partial derivative ∂n/∂x [1/km] at (x, z). Same shape as n.

    dndz : np.ndarray
        Partial derivative ∂n/∂z [1/km] at (x, z). Same shape as n.

    Notes
    -----
    Inputs are broadcast to a common shape, flattened for one batched
    interpolation, then reshaped back—this minimizes overhead during ODE
    stepping. Outside the grid hull, values are fill_value_n and
    fill_value_grad (unless bounds_error=True).

    """
    # Broadcast inputs
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    z_arr = np.atleast_1d(np.asarray(z, dtype=float))
    x_arr, z_arr = np.broadcast_arrays(x_arr, z_arr)

    # Interpolation points (z, x)
    pts = np.column_stack([z_arr.ravel(), x_arr.ravel()])

    n_val = n_interp(pts)
    dnx_val = dn_dx_interp(pts)
    dnz_val = dn_dz_interp(pts)

    out_shape = x_arr.shape

    return (
        n_val.reshape(out_shape),
        dnx_val.reshape(out_shape),
        dnz_val.reshape(out_shape),
    )


def eval_refractive_index_and_grad(x: np.ndarray,
                                   z: np.ndarray,
                                   n_interp: RegularGridInterpolator,
                                   dn_dx_interp: RegularGridInterpolator,
                                   dn_dz_interp: RegularGridInterpolator,
                                   ) -> Tuple[np.ndarray,
                                              np.ndarray,
                                              np.ndarray]:
    """Evaluate refractive index and its gradients at (x, z).

    Parameters
    ----------
    x : array_like
        Horizontal positions [km]. Scalar or array.
    z : array_like
        Altitudes [km]. Scalar or array.
    n_interp : RegularGridInterpolator
        Interpolator for n(z, x).
    dn_dx_interp : RegularGridInterpolator
        Interpolator for ∂n/∂x(z, x).
    dn_dz_interp : RegularGridInterpolator
        Interpolator for ∂n/∂z(z, x).

    Returns
    -------
    n : np.ndarray
        Refractive index values at (x, z).
    dndx : np.ndarray
        ∂n/∂x [1/km] values at (x, z).
    dndz : np.ndarray
        ∂n/∂z [1/km] values at (x, z).

    Notes
    -----
    Inputs x and z are broadcast to a common shape.
    Points are stacked as (z, x) before passing to interpolators.
    Output arrays match the broadcasted shape of inputs.

    """
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    z_arr = np.atleast_1d(np.asarray(z, dtype=float))
    x_arr, z_arr = np.broadcast_arrays(x_arr, z_arr)

    pts = np.column_stack([z_arr.ravel(), x_arr.ravel()])

    n_val = n_interp(pts)
    dnx_val = dn_dx_interp(pts)
    dnz_val = dn_dz_interp(pts)

    shape = x_arr.shape

    return (n_val.reshape(shape),
            dnx_val.reshape(shape),
            dnz_val.reshape(shape),)


def make_n_and_grad(n_interp: RegularGridInterpolator,
                    dn_dx_interp: RegularGridInterpolator,
                    dn_dz_interp: RegularGridInterpolator,
                    ) -> Callable[[np.ndarray, np.ndarray],
                                  Tuple[np.ndarray,
                                        np.ndarray,
                                        np.ndarray]]:
    """Construct a wrapper for evaluating refractive index and gradients.

    This function packages three interpolators—one for n(z, x), and two for its
    partial derivatives—into a single callable:
    n_and_grad(x, z) -> (n, dndx, dndz)

    Parameters
    ----------
    n_interp : RegularGridInterpolator
        Interpolator for the refractive index field n(z, x).
    dn_dx_interp : RegularGridInterpolator
        Interpolator for ∂n/∂x(z, x).
    dn_dz_interp : RegularGridInterpolator
        Interpolator for ∂n/∂z(z, x).

    Returns
    -------
    callable
        Function (x, z) → (n, dndx, dndz), with the interpolators baked in.

    Notes
    -----
    Inputs (x, z) can be scalars or arrays; they are broadcast to a common
    shape. The returned function simply delegates to
    eval_refractive_index_and_grad, keeping the interpolators “baked in” so you
    don't need to pass them around separately.
    Typical usage is inside build_refractive_index_interpolator_cartesian.

    """
    def n_and_grad(x: np.ndarray,
                   z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return eval_refractive_index_and_grad(x, z,
                                              n_interp,
                                              dn_dx_interp, dn_dz_interp)
    return partial(eval_refractive_index_and_grad,
                   n_interp=n_interp,
                   dn_dx_interp=dn_dx_interp,
                   dn_dz_interp=dn_dz_interp)


def ray_rhs_cartesian(s: float,
                      y: np.ndarray,
                      n_and_grad: Callable[[np.ndarray,
                                            np.ndarray],
                                           Tuple[np.ndarray,
                                                 np.ndarray,
                                                 np.ndarray]],
                      renormalize_every: int,
                      eval_counter: Dict[str, int],) -> np.ndarray:
    """Right-hand side of the 2D ray equations in Cartesian coordinates.

    Parameters
    ----------
    s : float
        Arc length [km].
    y : ndarray
        State vector [x, z, vx, vz].
    n_and_grad : callable
        Function returning (n, dndx, dndz) at (x, z).
    renormalize_every : int
        Frequency (in calls) to re-normalize direction vector.
    eval_counter : dict
        Mutable counter used for tracking RHS evaluations.

    Returns
    -------
    dyds : ndarray
        Derivatives [dx/ds, dz/ds, dvx/ds, dvz/ds].

    """
    x, z, vx, vz = y
    n, dndx, dndz = n_and_grad(np.array([x]), np.array([z]))
    n, dndx, dndz = float(n[0]), float(dndx[0]), float(dndz[0])

    if not np.isfinite(n) or n <= 0.0:
        return np.zeros(4)

    dxds, dzds = vx, vz
    gv_dot_v = dndx * vx + dndz * vz
    dvxds = (dndx - gv_dot_v * vx) / n
    dvzds = (dndz - gv_dot_v * vz) / n

    # Periodic re-normalization
    eval_counter['n'] += 1
    if renormalize_every and (eval_counter['n'] % renormalize_every == 0):
        vmag = np.hypot(vx, vz)
        if vmag > 0.0:
            scale = 1.0 / vmag
            dxds, dzds = dxds * scale, dzds * scale
            gv_dot_v = dndx * dxds + dndz * dzds
            dvxds = (dndx - gv_dot_v * dxds) / n
            dvzds = (dndz - gv_dot_v * dzds) / n

    return np.array([dxds, dzds, dvxds, dvzds])


def event_ground(s: float, y: np.ndarray, z_ground_km: float) -> float:
    """Stop when ray hits or goes below the ground."""
    return y[1] - z_ground_km - 1e3


def event_z_top(s: float, y: np.ndarray, z_max_km: float) -> float:
    """Stop when ray leaves the top of the domain."""
    return z_max_km - y[1]


def event_z_bottom(s: float, y: np.ndarray, z_min_km: float) -> float:
    """Stop when ray leaves the bottom of the domain."""
    return y[1] - z_min_km


def event_x_left(s: float, y: np.ndarray, x_min_km: float) -> float:
    """Stop when ray exits left boundary."""
    return y[0] - x_min_km


def event_x_right(s: float, y: np.ndarray, x_max_km: float) -> float:
    """Stop when ray exits right boundary."""
    return x_max_km - y[0]


def tan_from_mu_scalar(mu_val: float, p: float) -> float:
    """Compute tanθ safely for Snell's law in plasma.

    Parameters
    ----------
    mu_val : float
        Phase refractive index μ at altitude.
    p : float
        Snell invariant (μ0 sinθ0).

    Returns
    -------
    tanθ : float
        Tangent of propagation angle relative to vertical.
    """
    eps = 1e-10
    mu2 = float(mu_val) ** 2
    root = np.sqrt(max(mu2 - p * p, eps))
    return p / root


def find_turning_point(z: np.ndarray, mu: np.ndarray, p: float) -> float:
    """Locate altitude where μ crosses the Snell invariant p.

    Uses linear interpolation between bracketing nodes.

    Returns
    -------
    z_turn : float
        Altitude of turning point [km].
    """
    for i in range(z.size - 1):
        if (mu[i] >= p) and (mu[i + 1] <= p):
            z0, z1 = z[i], z[i + 1]
            mu0, mu1 = mu[i], mu[i + 1]
            if mu0 == mu1:
                return float(z0)
            t = (mu0 - p) / (mu0 - mu1)
            return float(z0 + t * (z1 - z0))
    return np.nan


def trace_ray_cartesian_snells(f0_Hz: float,
                               elevation_deg: float,
                               alt_km: np.ndarray,
                               Ne: np.ndarray,
                               Babs: np.ndarray,
                               bpsi: np.ndarray,
                               mode: str) -> Dict[str, float]:
    """Stratified Snell's law ray tracing (flat Earth, 2D Cartesian).

    Parameters
    ----------
    f0_Hz : float
        Frequency [Hz].
    elevation_deg : float
        Launch elevation above horizontal [deg].
    alt_km : ndarray
        Altitude grid [km].
    Ne : ndarray
        Electron density [el/m^3].
    Babs : ndarray
        Magnetic field strength [T].
    bpsi : ndarray
        Angle between B and k-vector [degrees].
    mode : str
        Wave mode: "O" or "X".

    Returns
    -------
    result : dict
        {'x': ndarray,           # horizontal positions [km]
        'z': ndarray,           # altitudes [km]
        'group_path_km': float,
        'group_delay_sec': float,
        'x_midpoint': float,
        'z_midpoint': float,
        'ground_range_km': float}

    Notes
    -----
    This function models how a high-frequency radio wave
    propagates through the ionosphere, using Snell's law
    adapted for a plasma medium. It calculates the trajectory
    of the ray as it leaves the ground, bends through the
    ionized atmosphere, reaches a turning point, and (if
    conditions allow) returns back toward Earth.

    Background: Snell's Law in a Plasma.
    In a uniform dielectric, Snell's law states that
    nsinθ=constant, where n is the refractive index and θ
    is the propagation angle relative to the vertical.
    In a plasma, the refractive index isn't constant but
    depends on: electron density (affects plasma frequency),
    magnetic field (splits O and X modes), wave frequency,
    and angle between wave vector and magnetic field.
    This gives two possible wave modes: the ordinary (O) and
    extraordinary (X) mode, each with a different effective
    refractive index. The function uses auxiliary functions
    find_X, find_Y, and find_mu_mup to compute these
    refractive indices as functions of altitude. Thus,
    the plasma-modified Snell's law is applied: μ'sinθ=constant,
    where μ′ is the transverse refractive index for the
    chosen wave mode.

    Specifics:
    Geometry (bending) uses phase index μ.
    Group delay integrates group index μ′ (mup).
    Down-leg is a perfect mirror of the up-leg about the apex.

    """
    # Use constants defined above
    _, _, _, c_km_per_s = constants()

    # Ensure ground present
    h_ground = 0.0
    if alt_km[0] > h_ground:
        Ne0 = np.interp(h_ground, alt_km, Ne)
        Babs0 = np.interp(h_ground, alt_km, Babs)
        bpsi0 = np.interp(h_ground, alt_km, bpsi)
        alt_km = np.insert(alt_km, 0, h_ground)
        Ne = np.insert(Ne, 0, Ne0)
        Babs = np.insert(Babs, 0, Babs0)
        bpsi = np.insert(bpsi, 0, bpsi0)

    # Plasma parameters
    X = find_X(Ne, f0_Hz)
    Y = find_Y(f0_Hz, Babs)
    mu, mup = find_mu_mup(X, Y, bpsi, mode)
    mu = np.where((~np.isfinite(mu)) | (mu <= 0.0), np.nan, mu)
    mup = np.where((~np.isfinite(mup)) | (mup <= 0.0), np.nan, mup)

    # Launch constants
    theta0 = np.radians(90.0 - elevation_deg)  # from vertical
    s0 = np.sin(theta0)
    mu0 = mu[0]
    if not np.isfinite(mu0) or not np.isfinite(s0):
        return {k: np.nan for k in ["x", "z", "group_path_km",
                                    "group_delay_sec", "x_midpoint",
                                    "z_midpoint", "ground_range_km"]}
    # Snell invariant
    p = mu0 * s0

    # Turning point
    valid = np.isfinite(mu)
    zv, muv = alt_km[valid], mu[valid]
    if zv.size < 2:
        return {k: np.nan for k in ["x", "z", "group_path_km",
                                    "group_delay_sec", "x_midpoint",
                                    "z_midpoint", "ground_range_km"]}
    z_turn = find_turning_point(zv, muv, p)
    if not np.isfinite(z_turn):
        return {k: np.nan for k in ["x", "z", "group_path_km",
                                    "group_delay_sec", "x_midpoint",
                                    "z_midpoint", "ground_range_km"]}

    # Up-leg nodes (include apex)
    i_turn = np.searchsorted(zv, z_turn)
    z_up = np.concatenate([zv[:i_turn], [z_turn]])
    mu_up = np.concatenate([muv[:i_turn], [p]])

    # Integrate horizontal displacement using midpoint tanθ
    x_up = np.zeros_like(z_up)
    if z_up.size > 1:
        dz = np.diff(z_up)
        mu_mid = 0.5 * (mu_up[:-1] + mu_up[1:])
        mu_mid[-1] = max(mu_mid[-1], p + 1e-8)  # avoid singularity
        tan_mid = np.array([tan_from_mu_scalar(mm, p) for mm in mu_mid])
        x_up[1:] = np.cumsum(dz * tan_mid)

    # Mirror down-leg
    x_turn = x_up[-1]
    z_down = z_up[::-1]
    x_down = (2.0 * x_turn) - x_up[::-1]
    x_full = np.concatenate([x_up, x_down[1:]])
    z_full = np.concatenate([z_up, z_down[1:]])

    # Metrics
    dx, dz = np.diff(x_full), np.diff(z_full)
    ds = np.hypot(dx, dz)
    group_path_km = float(np.nansum(ds))

    mup_path = np.interp(z_full, alt_km, mup)
    mup_seg = 0.5 * (mup_path[1:] + mup_path[:-1])
    group_delay_sec = float(np.nansum((mup_seg / c_km_per_s) * ds))

    if group_path_km > 0:
        s_cum = np.cumsum(ds)
        mid_idx = int(np.searchsorted(s_cum, 0.5 * group_path_km))
        x_midpoint = float(x_full[mid_idx])
        z_midpoint = float(z_full[mid_idx])
    else:
        x_midpoint = z_midpoint = np.nan

    ground_range_km = float(x_full[-1]) if np.isclose(z_full[-1],
                                                      0.0,
                                                      atol=1e-3) else np.nan

    return {"x": x_full,
            "z": z_full,
            "group_path_km": group_path_km,
            "group_delay_sec": group_delay_sec,
            "x_midpoint": x_midpoint,
            "z_midpoint": z_midpoint,
            "ground_range_km": ground_range_km}


def trace_ray_cartesian_gradient(
    n_and_grad: Callable[[np.ndarray, np.ndarray],
                         Tuple[np.ndarray, np.ndarray, np.ndarray]],
    mup_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x0_km: float,
    z0_km: float,
    elevation_deg: float,
    s_max_km: float = 5000.0,
    *,
    # Integration controls
    rtol: float = 1e-7,
    atol: float = 1e-9,
    max_step_km: Optional[float] = None,
    # Domain / stop conditions
    z_ground_km: float = 0.0,
    z_min_km: float = -1.0,
    z_max_km: float = 1000.0,
    x_min_km: float = -1e6,
    x_max_km: float = 1e6,
    # Numerical hygiene
    renormalize_every: int = 50,
) -> Dict[str, Any]:
    """Trace a 2D ray in a horizontally varying refractive index field μ(x, z).

    Geometry is integrated via:
        dr/ds = v,  ||v|| = 1
        dv/ds = (1/μ) [∇μ - (∇μ·v) v]

    Group delay is integrated using μ′(x, z):
        τ = ∫ (μ′/c) ds

    Parameters
    ----------
    n_and_grad : callable
        (x, z) → (μ, ∂μ/∂x, ∂μ/∂z). Usually built with
        `build_refractive_index_interpolator(...)`.
    mup_func : callable
        (x, z) → μ′(x, z). Must be built with `build_mup_function(...)`.
    x0_km, z0_km : float
        Launch point [km].
    elevation_deg : float
        Launch elevation above horizontal [deg].
    s_max_km : float
        Max path length [km].
    rtol, atol : float
        ODE solver tolerances.
    max_step_km : float or None
        Max solver step size [km].
    z_ground_km, z_min_km, z_max_km : float
        Vertical domain limits [km].
    x_min_km, x_max_km : float
        Horizontal domain limits [km].
    renormalize_every : int
        Re-normalize v every N evaluations.

    Returns
    -------
    result : dict
        {
          'x', 'z', 'vx', 'vz', 'status',
          'group_path_km', 'group_delay_sec',
          'x_midpoint', 'z_midpoint', 'ground_range_km'
        }

    Notes
    -----
    • This model assumes a 2D Cartesian (flat-Earth) geometry.
    • μ controls bending; μ′ controls group delay.
    • NaNs or invalid μ terminate integration.

    """
    # Use constants defined above
    _, _, R_E, _ = constants()

    # --- Check mandatory input
    if mup_func is None:
        string = "mup_func must be provided, build it with build_mup_function."
        raise ValueError(string)

    # --- Initial conditions
    elev_rad = np.deg2rad(elevation_deg)
    vx0, vz0 = np.cos(elev_rad), np.sin(elev_rad)
    vnorm = np.hypot(vx0, vz0)
    y0 = np.array([x0_km, z0_km, vx0 / vnorm, vz0 / vnorm], dtype=float)

    eval_counter = {'n': 0}

    # --- Use global shared event helpers directly (no lambdas)
    events = [partial(event_ground, z_ground_km=z_ground_km),
              partial(event_z_top, z_max_km=z_max_km),
              partial(event_x_left, x_min_km=x_min_km),
              partial(event_x_right, x_max_km=x_max_km)]

    for ev in events:
        ev.terminal, ev.direction = True, -1.0

    # --- Right-hand side
    def rhs_cartesian(s, y):
        return ray_rhs_cartesian(s, y, n_and_grad,
                                 renormalize_every, eval_counter)

    # --- Integrate
    sol = solve_ivp(rhs_cartesian,
                    (0.0, s_max_km),
                    y0,
                    method="RK45",
                    rtol=rtol,
                    atol=atol,
                    max_step=max_step_km,
                    events=events,
                    dense_output=True)

    # --- Determine status
    if sol.status == 1:
        status = "ground" if len(sol.t_events[0]) > 0 else "domain"
    elif sol.status == 0:
        status = "length"
    elif sol.status == -1:
        status = "failure"
    else:
        status = "success"

    # --- Extract results
    y = sol.y
    x_path, z_path = y[0, :], y[1, :]

    dx, dz = np.diff(x_path), np.diff(z_path)
    ds = np.hypot(dx, dz)
    group_path_km = float(np.nansum(ds))

    # --- Group delay (μ′ required)
    c_km_per_s = 299792.458
    x_mid = 0.5 * (x_path[:-1] + x_path[1:])
    z_mid = 0.5 * (z_path[:-1] + z_path[1:])
    mup_mid = np.asarray(mup_func(x_mid, z_mid), dtype=float)
    valid = np.isfinite(mup_mid)
    group_delay_sec = float(np.nansum((mup_mid[valid]
                                       / c_km_per_s) * ds[valid]))

    # --- Midpoint and landing
    mid_idx = len(sol.t) // 2
    x_midpoint = float(x_path[mid_idx]) if x_path.size else np.nan
    z_midpoint = float(z_path[mid_idx]) if z_path.size else np.nan
    ground_range_km = float(x_path[-1]) if status == "ground" else np.nan

    # --- Return dictionary
    return {"sol": sol,
            "t": sol.t,
            "x": x_path,
            "z": z_path,
            "vx": y[2, :],
            "vz": y[3, :],
            "status": status,
            "group_path_km": group_path_km,
            "group_delay_sec": group_delay_sec,
            "x_midpoint": x_midpoint,
            "z_midpoint": z_midpoint,
            "ground_range_km": ground_range_km}


def trace_ray_spherical_snells(
    f0_Hz: float,
    elevation_deg: float,
    alt_km: np.ndarray,
    Ne: np.ndarray,
    Babs: np.ndarray,
    bpsi: np.ndarray,
    mode: str = "O",
    *,
    # new controls for apex refinement (defaults are conservative)
    dz_target_km: float = 1.0,  # aim for 1 km substeps away from apex
    apex_boost: float = 200.0,  # multiply substeps as (1+apex_boost*sharpness)
    max_substeps: int = 400,  # hard cap per coarse interval
    R_E: Optional[float] = None,
) -> Dict[str, float]:
    """Stratified Snell's law ray tracing (spherical Earth, 2D geometry).

    Parameters
    ----------
    f0_Hz : float
        Radio frequency [Hz].
    elevation_deg : float
        Launch elevation above local horizontal [degrees].
    alt_km : ndarray
        Altitude grid [km].
    Ne : ndarray
        Electron density profile [el/m³].
    Babs : ndarray
        Magnetic field strength [T].
    bpsi : ndarray
        Magnetic field inclination [degrees].
    mode : str, default 'O'
        Wave mode: 'O' (ordinary) or 'X' (extraordinary).

    dz_target_km : float, default 1.0
        Target altitude increment for apex refinement [km].
    apex_boost : float, default 200.0
        Sharpness multiplier that increases substeps near turning points.
    max_substeps : int, default 400
        Hard limit on number of adaptive substeps per altitude interval.
    R_E : float or None, optional
        Earth radius [km]. If None, defaults to value from `constants()`.

    Returns
    -------
    result : dict
        {
          'x': ndarray,             # surface distance [km]
          'z': ndarray,             # altitude along the ray [km]
          'group_path_km': float,   # total geometric path length [km]
          'group_delay_sec': float, # group delay [s]
          'x_midpoint': float,      # midpoint horizontal coordinate [km]
          'z_midpoint': float,      # midpoint altitude [km]
          'ground_range_km': float  # surface distance to landing point [km]
        }

    Notes
    -----
    **Physical formulation:**
      In spherical geometry, Snell's invariant becomes:
          μ * r * sin(θ) = constant
      where μ is the phase refractive index, r = R_E + z, and θ is the
      angle between the ray and the local vertical.

      The ray is launched from ground level, bent according to μ(r),
      reflected at the turning point where μ * r = p, and mirrored down.

    **Group delay:**
      The propagation delay is integrated along the ray using the group
      refractive index μ′ (mup):
          τ = ∫ (μ′ / c) ds
      where c is the speed of light in vacuum.

    **Apex refinement:**
      Near the turning point, the derivative
          dφ/dz = p / [ r * sqrt((μ r)² - p²) ]
      becomes sharply peaked, making coarse integration unstable.
      To resolve this, the algorithm adaptively subdivides altitude steps
      where |μ r - p| is small, increasing resolution toward the apex
      while capping the total substeps with `max_substeps`.

    **Comparison to Cartesian model:**
      This function extends the flat-Earth Snell's law formulation
      (`trace_ray_cartesian_snells`) to spherical geometry. For large R_E,
      both solutions converge within numerical precision.

    """
    # Speed of light
    _, _, _, c_km_s = constants()
    # Load Earth radius if not provided
    if R_E is None:
        _, _, R_E, _ = constants()

    # Ensure ground at z=0 present
    if alt_km[0] > 0.0:
        Ne0 = np.interp(0.0, alt_km, Ne)
        B0 = np.interp(0.0, alt_km, Babs)
        psi0 = np.interp(0.0, alt_km, bpsi)
        alt_km = np.insert(alt_km, 0, 0.0)
        Ne = np.insert(Ne, 0, Ne0)
        Babs = np.insert(Babs, 0, B0)
        bpsi = np.insert(bpsi, 0, psi0)

    # Plasma indices
    X = find_X(Ne, f0_Hz)
    Y = find_Y(f0_Hz, Babs)
    mu, mup = find_mu_mup(X, Y, bpsi, mode)
    mu = np.where((~np.isfinite(mu)) | (mu <= 0.0), np.nan, mu)
    mup = np.where((~np.isfinite(mup)) | (mup <= 0.0), np.nan, mup)

    # Invariant p = μ0 r0 sinθ0  (θ0 from radial)
    theta0 = np.radians(90.0 - elevation_deg)
    r0 = R_E + alt_km[0]
    mu0 = mu[0]
    if not np.isfinite(mu0):
        return {k: np.nan for k in ["x",
                                    "z",
                                    "group_path_km",
                                    "group_delay_sec",
                                    "x_midpoint",
                                    "z_midpoint",
                                    "ground_range_km"]}
    p = mu0 * r0 * np.sin(theta0)

    # Find turning point where (μ r) crosses p
    valid = np.isfinite(mu)
    zv, muv = alt_km[valid], mu[valid]
    rv = R_E + zv
    if zv.size < 2:
        return {k: np.nan for k in ["x",
                                    "z",
                                    "group_path_km",
                                    "group_delay_sec",
                                    "x_midpoint",
                                    "z_midpoint",
                                    "ground_range_km"]}
    mu_r = muv * rv
    cross = None
    for i in range(zv.size - 1):
        if (mu_r[i] >= p) and (mu_r[i + 1] <= p):
            cross = (i, i + 1)
            break
    if cross is None:
        return {k: np.nan for k in ["x",
                                    "z",
                                    "group_path_km",
                                    "group_delay_sec",
                                    "x_midpoint",
                                    "z_midpoint",
                                    "ground_range_km"]}

    i0, i1 = cross
    z0, z1 = zv[i0], zv[i1]
    mu_r0, mu_r1 = mu_r[i0], mu_r[i1]
    # linear z_turn where μ(z) r(z) = p
    t = (mu_r0 - p) / (mu_r0 - mu_r1) if mu_r0 != mu_r1 else 0.0
    t = float(np.clip(t, 0.0, 1.0))
    z_turn = z0 + t * (z1 - z0)

    # Up-leg nodes including apex; μ at apex from invariant
    z_up = np.concatenate([zv[:i0 + 1], [z_turn]])
    r_up = R_E + z_up
    mu_up = np.concatenate([muv[:i0 + 1], [p / r_up[-1]]])

    # --- Integrate φ with adaptive substeps over each coarse interval ---
    phi_up = np.zeros_like(z_up)

    def dphi_integrand(mu_r_val, r_val):
        # f = p / [ r sqrt((μ r)^2 - p^2) ]
        denom = max(mu_r_val * mu_r_val - p * p, 1e-16)
        return p / (r_val * np.sqrt(denom))

    for k in range(len(z_up) - 1):
        z_a, z_b = z_up[k], z_up[k + 1]
        r_a, r_b = r_up[k], r_up[k + 1]
        mu_a, mu_b = mu_up[k], mu_up[k + 1]
        mu_r_a, mu_r_b = mu_a * r_a, mu_b * r_b

        dz = z_b - z_a
        if dz <= 0:
            continue

        # base substeps by dz_target
        N = max(1, int(np.ceil(abs(dz) / dz_target_km)))

        # apex sharpness: smaller min gap -> more substeps
        gap_a = max(mu_r_a - p, 1e-12)
        gap_b = max(mu_r_b - p, 1e-12)
        sharpness = 1.0 / min(gap_a, gap_b)
        N = int(min(max_substeps, N * (1.0 + apex_boost * sharpness)))

        # integrate with midpoint rule on the product (μ r)
        dphi_sum = 0.0
        for j in range(N):
            t0 = j / N
            t1 = (j + 1) / N
            z_m = z_a + 0.5 * (t0 + t1) * dz
            r_m = R_E + z_m
            # linear μ at midpoint
            mu_m = mu_a + (mu_b - mu_a) * (0.5 * (t0 + t1))
            mu_r_m = mu_m * r_m
            # nudge away from singularity
            if mu_r_m <= p:
                mu_r_m = p + 1e-8
            f_m = dphi_integrand(mu_r_m, r_m)
            dphi_sum += f_m * (dz / N)

        phi_up[k + 1] = phi_up[k] + dphi_sum

    # Mirror down-leg in (φ, z)
    phi_turn = phi_up[-1]
    phi_down = 2.0 * phi_turn - phi_up[::-1]
    phi_full = np.concatenate([phi_up, phi_down[1:]])
    z_full = np.concatenate([z_up, z_up[::-1][1:]])

    # Coordinates and metrics
    x_full = R_E * phi_full

    dz_seg = np.diff(z_full)
    phi_seg = np.diff(phi_full)
    r_mid = R_E + 0.5 * (z_full[:-1] + z_full[1:])
    ds_seg = np.hypot(r_mid * phi_seg, dz_seg)

    group_path_km = float(np.nansum(ds_seg))
    mup_path = np.interp(z_full, alt_km, mup)
    mup_seg = 0.5 * (mup_path[:-1] + mup_path[1:])

    group_delay_sec = float(np.nansum((mup_seg / c_km_s) * ds_seg))

    if group_path_km > 0:
        s_cum = np.cumsum(ds_seg)
        mid_idx = int(np.searchsorted(s_cum, 0.5 * group_path_km))
        x_midpoint = float(x_full[mid_idx])
        z_midpoint = float(z_full[mid_idx])
    else:
        x_midpoint = z_midpoint = np.nan

    ground_range_km = float(x_full[-1]) if np.isclose(z_full[-1],
                                                      0.0,
                                                      atol=1e-3) else np.nan

    return {
        "x": x_full,
        "z": z_full,
        "group_path_km": group_path_km,
        "group_delay_sec": group_delay_sec,
        "x_midpoint": x_midpoint,
        "z_midpoint": z_midpoint,
        "ground_range_km": ground_range_km,
    }


def trace_ray_spherical_gradient(
    n_and_grad_rphi: Callable[[np.ndarray, np.ndarray],
                              Tuple[np.ndarray, np.ndarray, np.ndarray]],
    mup_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x0_km: float,
    z0_km: float,
    elevation_deg: float,
    s_max_km: float = 6000.0,
    *,
    R_E: Optional[float] = None,
    z_ground_km: float = 0.0,
    r_max_km: Optional[float] = None,
    phi_min: float = -np.pi,
    phi_max: float = +np.pi,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    max_step_km: Optional[float] = 2.0,
    renormalize_every: int = 50,
) -> Dict[str, Any]:
    """Raytrace in 2-D spherical using μ for geometry and μ' for delay.

    Geometry
    --------
    Integrates in spherical coordinates (r, φ) with tangent v = (v_r, v_φ):
        dr/ds   = v_r
        dφ/ds   = v_φ / r
        dv_r/ds = (1/μ)[∂μ/∂r - (∇μ·v)v_r] + (v_φ²)/r
        dv_φ/ds = (1/μ)[(∂μ/∂φ)/r - (∇μ·v)v_φ] - (v_r v_φ)/r
    where ∇μ·v = (∂μ/∂r)v_r + (∂μ/∂φ)(v_φ / r).

    Delay
    -----
    Group delay integrates μ′ along the path:
        τ = ∫ ( μ′(x, z) / c ) ds
    with x = R_E φ (surface arc) and z = r − R_E.

    Parameters
    ----------
    n_and_grad_rphi : callable
        (φ, r) → (μ, ∂μ/∂r, ∂μ/∂φ).
    mup_func : callable  (REQUIRED)
        (x, z) → μ′(x, z). Build with `build_mup_function(...,
        geometry="spherical")`.
    x0_km, z0_km : float
        Launch point [km]; x0 is surface arc distance, z0 altitude.
    elevation_deg : float
        Launch elevation above local horizontal [deg].
    s_max_km : float
        Max arclength to integrate [km].
    R_E : float, optional
        Earth radius [km]. Defaults to `constants()[2]` if None.
    z_ground_km : float
        Ground altitude [km]; stop when r ≤ R_E + z_ground_km.
    r_max_km : float, optional
        Max r; default R_E + 1200 km if None.
    phi_min, phi_max : float
        Azimuth bounds [rad].
    rtol, atol : float
        ODE tolerances.
    max_step_km : float or None
        Max solver step [km].
    renormalize_every : int
        Re-normalize v every N RHS calls.

    Returns
    -------
    dict
        {'t', 'r', 'phi', 'v_r', 'v_phi', 'x', 'z', 'status',
        'group_path_km', 'group_delay_sec', 'x_midpoint', 'z_midpoint',
        'ground_range_km'}

    """

    # --- Required inputs
    if mup_func is None:
        string1 = "mup_func must be provided — build it with "
        string2 = "build_mup_function(..., geometry='spherical')."
        raise ValueError(string1 + string2)

    # --- Constants / defaults
    if R_E is None:
        _, _, R_E, c_km_s = constants()
    else:
        _, _, _, c_km_s = constants()
    if r_max_km is None:
        r_max_km = R_E + 1200.0

    # --- Initial conditions
    r0 = R_E + z0_km
    phi0 = x0_km / R_E
    elev = np.deg2rad(elevation_deg)
    v_r0, v_phi0 = np.sin(elev), np.cos(elev)
    y0 = np.array([r0, phi0, v_r0, v_phi0], dtype=float)

    eval_counter = {'n': 0}

    # --- Use global shared event helpers directly (no lambdas)
    events = [
        partial(event_ground, z_ground_km=R_E + z_ground_km),
        partial(event_z_top, z_max_km=r_max_km),
        partial(event_x_left, x_min_km=phi_min),
        partial(event_x_right, x_max_km=phi_max)]

    for ev in events:
        ev.terminal, ev.direction = True, -1.0

    def rhs_wrapper(s, y):
        """Make thin wrapper for rhs_spherical with fixed parameters."""
        return rhs_spherical(s, y, n_and_grad_rphi,
                             renormalize_every, eval_counter)

    # --- Integrate
    sol = solve_ivp(rhs_wrapper,
                    (0.0, s_max_km),
                    y0,
                    method="RK45",
                    rtol=rtol,
                    atol=atol,
                    max_step=max_step_km,
                    events=events,
                    dense_output=True)

    # --- Status
    if sol.status == 1:
        status = "ground" if len(sol.t_events[0]) > 0 else "domain"
    elif sol.status == 0:
        status = "length"
    elif sol.status == -1:
        status = "failure"
    else:
        status = "success"

    # --- Extract paths
    r_path, phi_path = sol.y[0], sol.y[1]
    v_r_path, v_phi_path = sol.y[2], sol.y[3]
    x_path = R_E * phi_path
    z_path = r_path - R_E

    # --- Geometry metrics
    dx = np.diff(x_path)
    dz = np.diff(z_path)
    ds = np.hypot(dx, dz)
    group_path_km = float(np.nansum(ds))

    # --- Group delay (μ')
    if ds.size > 0:
        x_mid = 0.5 * (x_path[:-1] + x_path[1:])
        z_mid = 0.5 * (z_path[:-1] + z_path[1:])
        mup_mid = np.asarray(mup_func(x_mid, z_mid), dtype=float)
        valid = np.isfinite(mup_mid)
        group_delay_sec = float(np.nansum((mup_mid[valid] / c_km_s)
                                          * ds[valid]))
    else:
        group_delay_sec = 0.0

    # --- Midpoint & landing
    if group_path_km > 0.0:
        s_cum = np.cumsum(ds)
        mid_idx = int(np.searchsorted(s_cum, 0.5 * group_path_km))
        x_midpoint = float(x_path[mid_idx])
        z_midpoint = float(z_path[mid_idx])
    else:
        x_midpoint = z_midpoint = np.nan

    ground_range_km = float(x_path[-1]) if len(sol.t_events[0]) else np.nan

    # --- Return
    return {"t": sol.t,
            "r": r_path,
            "phi": phi_path,
            "v_r": v_r_path,
            "v_phi": v_phi_path,
            "x": x_path,
            "z": z_path,
            "status": status,
            "group_path_km": group_path_km,
            "group_delay_sec": group_delay_sec,
            "x_midpoint": x_midpoint,
            "z_midpoint": z_midpoint,
            "ground_range_km": ground_range_km}


def n_and_grad_rphi(
    phi: np.ndarray,
    r: np.ndarray,
    n_interp: RegularGridInterpolator,
    dn_dr_interp: RegularGridInterpolator,
    dn_dphi_interp: RegularGridInterpolator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate μ(r, φ) and its gradients at given coordinates.

    Parameters
    ----------
    phi : array_like
        Azimuthal coordinate [rad]. Scalar or array. Will broadcast with r.
    r : array_like
        Radial coordinate [km] (Earth radius + altitude). Scalar or array.
    n_interp : RegularGridInterpolator
        Interpolator for μ(r, φ).
    dn_dr_interp : RegularGridInterpolator
        Interpolator for ∂μ/∂r.
    dn_dphi_interp : RegularGridInterpolator
        Interpolator for ∂μ/∂φ.

    Returns
    -------
    n : np.ndarray
        Refractive index μ at (r, φ). Shape matches broadcasted inputs.
    dn_dr : np.ndarray
        Partial derivative ∂μ/∂r at (r, φ).
    dn_dphi : np.ndarray
        Partial derivative ∂μ/∂φ at (r, φ).

    """
    phi_arr = np.atleast_1d(np.asarray(phi, dtype=float))
    r_arr = np.atleast_1d(np.asarray(r, dtype=float))
    phi_arr, r_arr = np.broadcast_arrays(phi_arr, r_arr)
    pts = np.column_stack([r_arr.ravel(), phi_arr.ravel()])

    n_val = n_interp(pts)
    nr_val = dn_dr_interp(pts)
    nphi_val = dn_dphi_interp(pts)

    out_shape = phi_arr.shape
    return (
        n_val.reshape(out_shape),
        nr_val.reshape(out_shape),
        nphi_val.reshape(out_shape),
    )


def build_refractive_index_interpolator_cartesian(
    z_grid: np.ndarray,
    x_grid: np.ndarray,
    n_field: np.ndarray,
    *,
    fill_value_n: float = np.nan,
    fill_value_grad: float = 0.0,
    bounds_error: bool = False,
    edge_order: int = 2) -> Callable[[np.ndarray,
                                      np.ndarray],
                                     Tuple[np.ndarray,
                                           np.ndarray,
                                           np.ndarray]]:
    """Construct interpolators for refractive index n(z, x) and its gradients.

    Parameters
    ----------
    z_grid : ndarray, shape (nz,)
        Altitude coordinates [km], strictly increasing.
    x_grid : ndarray, shape (nx,)
        Horizontal coordinates [km], strictly increasing.
    n_field : ndarray, shape (nz, nx)
        Refractive index values on (z, x) grid.
    fill_value_n : float
        Fill value for n outside grid (default NaN).
    fill_value_grad : float
        Fill value for gradients outside grid (default 0.0).
    bounds_error : bool
        If True, raise error outside grid. If False, use fill values.
    edge_order : int
        Accuracy order for finite differences (default 2).

    Returns
    -------
    n_and_grad : callable
        Function (x, z) → (n, dndx, dndz).
    """
    z_grid = np.asarray(z_grid, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    n_field = np.asarray(n_field, dtype=float)

    if n_field.shape != (z_grid.size, x_grid.size):
        raise ValueError(
            f"`n_field` must have shape (len(z_grid)={z_grid.size}, "
            f"len(x_grid)={x_grid.size}), "
            f"got {n_field.shape}."
        )

    if not (np.all(np.diff(z_grid) > 0) and np.all(np.diff(x_grid) > 0)):
        raise ValueError("`z_grid` and `x_grid` must be strictly increasing.")

    # Interpolator for n
    n_interp = RegularGridInterpolator(
        (z_grid, x_grid), n_field,
        bounds_error=bounds_error,
        fill_value=fill_value_n,
    )

    # Gradients on grid
    dn_dz, dn_dx = np.gradient(n_field, z_grid, x_grid, edge_order=edge_order)

    dn_dx_interp = RegularGridInterpolator(
        (z_grid, x_grid), dn_dx,
        bounds_error=bounds_error,
        fill_value=fill_value_grad,
    )
    dn_dz_interp = RegularGridInterpolator(
        (z_grid, x_grid), dn_dz,
        bounds_error=bounds_error,
        fill_value=fill_value_grad,
    )
    return make_n_and_grad(n_interp, dn_dx_interp, dn_dz_interp)


def build_refractive_index_interpolator_spherical(
    r_grid: np.ndarray,
    phi_grid: np.ndarray,
    n_field: np.ndarray,
    *,
    fill_value_n: float = np.nan,
    fill_value_grad: float = 0.0,
    bounds_error: bool = False,
    edge_order: int = 2) -> Callable[[np.ndarray,
                                      np.ndarray],
                                     Tuple[np.ndarray,
                                           np.ndarray,
                                           np.ndarray]]:
    """Construct interpolators for refractive index μ(r, φ) and its gradients.

    Parameters
    ----------
    r_grid : ndarray, shape (nr,)
        Radial coordinates [km], strictly increasing (R_E + altitude).
    phi_grid : ndarray, shape (nφ,)
        Azimuth coordinates [rad], strictly increasing.
    n_field : ndarray, shape (nr, nφ)
        Refractive index values on (r, φ) grid.
    fill_value_n : float
        Fill value for μ outside grid (default NaN).
    fill_value_grad : float
        Fill value for gradients outside grid (default 0.0).
    bounds_error : bool
        If True, raise error outside grid. If False, use fill values.
    edge_order : int
        Accuracy order for finite differences (default 2).

    Returns
    -------
    n_and_grad_rphi : callable
        Function (φ, r) → (μ, ∂μ/∂r, ∂μ/∂φ).

    """
    r_grid = np.asarray(r_grid, dtype=float)
    phi_grid = np.asarray(phi_grid, dtype=float)
    n_field = np.asarray(n_field, dtype=float)

    if n_field.shape != (r_grid.size, phi_grid.size):
        raise ValueError(
            f"`n_field` shape {n_field.shape} must be "
            f"(len(r_grid)={r_grid.size}, len(phi_grid)={phi_grid.size})."
        )

    if not (np.all(np.diff(r_grid) > 0) and np.all(np.diff(phi_grid) > 0)):
        raise ValueError(
            "`r_grid` and `phi_grid` must be strictly increasing.")

    # Interpolator for μ(r,φ)
    n_interp = RegularGridInterpolator(
        (r_grid, phi_grid), n_field,
        bounds_error=bounds_error,
        fill_value=fill_value_n,
    )

    # Compute gradients (axis 0 = r, axis 1 = φ)
    dn_dr, dn_dphi = np.gradient(
        n_field, r_grid, phi_grid, edge_order=edge_order
    )

    dn_dr_interp = RegularGridInterpolator(
        (r_grid, phi_grid), dn_dr,
        bounds_error=bounds_error,
        fill_value=fill_value_grad,
    )
    dn_dphi_interp = RegularGridInterpolator(
        (r_grid, phi_grid), dn_dphi,
        bounds_error=bounds_error,
        fill_value=fill_value_grad,
    )

    def n_and_grad_rphi_func(phi, r):
        """Evaluate μ(r, φ) and its grad using precomputed interpolators."""
        return n_and_grad_rphi(phi, r, n_interp, dn_dr_interp, dn_dphi_interp)

    return n_and_grad_rphi_func


def build_mup_function(
    mup_field: np.ndarray,
    x_grid: np.ndarray,
    z_grid: np.ndarray,
    *,
    geometry: str = "cartesian",
    R_E: float = None,
    bounds_error: bool = False,
    fill_value: float = np.nan) -> Callable[[np.ndarray,
                                             np.ndarray],
                                            np.ndarray]:
    """Construct callable for evaluating μ'(x, z) for group delay integration.

    Parameters
    ----------
    mup_field : ndarray
        Grid of μ′ (group refractive index) values.
    x_grid : ndarray
        Horizontal coordinate grid:
          - For "cartesian": horizontal distance [km].
          - For "spherical": surface arc distance [km].
    z_grid : ndarray
        Vertical coordinate grid [km].
    geometry : {'cartesian', 'spherical'}, optional
        Coordinate system type. Default is 'cartesian'.
    R_E : float, optional
        Earth radius [km], used only for spherical geometry.
    bounds_error : bool, optional
        Passed to RegularGridInterpolator. If True, raise error outside grid.
    fill_value : float, optional
        Fill value used for extrapolation. Default np.nan.

    Returns
    -------
    mup_func : callable
        Function mup_func(x, z) that evaluates μ′ at given coordinates.

    Notes
    -----
    • For Cartesian geometry, (z, x) order is used in the interpolator.
    • For Spherical geometry, (r, φ) order is used, where
        r = R_E + z   and   φ = x / R_E.

    """
    # Load Earth radius if not provided
    if R_E is None:
        _, _, R_E, _ = constants()

    # Create the interpolator depending on geometry type
    if geometry == "cartesian":
        mup_interp = RegularGridInterpolator(
            (z_grid, x_grid),
            mup_field,
            bounds_error=bounds_error,
            fill_value=fill_value,
        )

        def mup_func(x: np.ndarray, z: np.ndarray) -> np.ndarray:
            """Evaluate μ′(x, z) in Cartesian geometry."""
            pts = np.column_stack([np.ravel(z), np.ravel(x)])
            vals = mup_interp(pts)
            return vals.reshape(np.shape(x))

        return mup_func

    elif geometry == "spherical":
        # Convert to r, φ grids
        r_grid = R_E + np.asarray(z_grid)
        phi_grid = np.asarray(x_grid) / R_E

        mup_interp = RegularGridInterpolator(
            (r_grid, phi_grid),
            mup_field,
            bounds_error=bounds_error,
            fill_value=fill_value,
        )

        def mup_func(x: np.ndarray, z: np.ndarray) -> np.ndarray:
            """Evaluate μ′(x, z) in spherical geometry."""
            r = R_E + np.asarray(z)
            phi = np.asarray(x) / R_E
            pts = np.column_stack([r.ravel(), phi.ravel()])
            vals = mup_interp(pts)
            return vals.reshape(np.shape(x))
        return mup_func

    else:
        raise ValueError("geometry must be 'cartesian' or 'spherical'")


def rhs_spherical(
    s: float,
    y: np.ndarray,
    n_and_grad_rphi: Callable[[np.ndarray, np.ndarray],
                              Tuple[np.ndarray, np.ndarray, np.ndarray]],
    renormalize_every: int,
    eval_counter: Dict[str, int],
) -> np.ndarray:
    r"""Calculate the right-hand side of the spherical ray equations.

    Computes derivatives for the ODE system governing 2D spherical ray
    propagation (r, φ) in a refractive index field μ(r, φ). The equations
    describe the evolution of position and tangent components along the
    raypath with respect to arc length *s*.

    The system is defined as:

    .. math::

        \\frac{dr}{ds}   = v_r
        \\qquad
        \\frac{dφ}{ds}   = \\frac{v_φ}{r}

        \\frac{dv_r}{ds} = \\frac{1}{μ}
            \\left[ \\frac{∂μ}{∂r} - (∇μ · v)v_r \\right] + \\frac{v_φ^2}{r}

        \\frac{dv_φ}{ds} = \\frac{1}{μ}
            \\left[ \\frac{1}{r} \\frac{∂μ}{∂φ} - (∇μ · v)v_φ \\right]
            - \\frac{v_r v_φ}{r}

    where the gradient projection is given by:

    .. math::

        ∇μ · v = \\frac{∂μ}{∂r} v_r + \\frac{1}{r} \\frac{∂μ}{∂φ} v_φ

    Parameters
    ----------
    s : float
        Current arc length along the ray [km].
    y : ndarray, shape (4,)
        State vector [r, φ, v_r, v_φ]:
        - r : radial coordinate [km]
        - φ : azimuthal angle [rad]
        - v_r : radial direction cosine
        - v_φ : tangential direction cosine
    n_and_grad_rphi : callable
        Function (φ, r) → (μ, ∂μ/∂r, ∂μ/∂φ), typically constructed using
        `build_refractive_index_interpolator_rphi`.
    renormalize_every : int
        Frequency (in RHS evaluations) to re-normalize the tangent vector
        to ensure |v| = 1. Use 0 or None to disable.
    eval_counter : dict
        Dictionary used to track number of RHS evaluations:
        e.g., `{'n': 0}` will be incremented in-place.

    Returns
    -------
    dyds : ndarray, shape (4,)
        Derivatives with respect to arc length *s*:
        [dr/ds, dφ/ds, dv_r/ds, dv_φ/ds].

    Notes
    -----
    • Implements 2D spherical geometry (flat-Earth limit not assumed).
    • The equations conserve |v| ≈ 1 under small step sizes.
    • NaN or non-positive μ values return zero derivatives (halts ray).

    References
    ----------
    - Haselgrove, J., "The Hamiltonian Ray Path Equations", *Proc. IEE* (1955)
    - Budden, K. G., *The Propagation of Radio Waves*, Cambridge Univ. Press,
    1985

    """
    r, phi, v_r, v_phi = y
    mu, mu_r, mu_phi = n_and_grad_rphi(phi, r)

    mu = float(mu)
    mu_r = float(mu_r)
    mu_phi = float(mu_phi)

    # Invalid or non-physical μ → terminate derivative
    if not np.isfinite(mu) or mu <= 0.0:
        return np.zeros_like(y)

    # Gradient projection ∇μ · v
    grad_dot_v = mu_r * v_r + (mu_phi / r) * v_phi

    # Position derivatives
    drds = v_r
    dphids = v_phi / r

    # Velocity derivatives
    dv_r = (mu_r - grad_dot_v * v_r) / mu + (v_phi**2) / r
    dv_phi = ((mu_phi / r) - grad_dot_v * v_phi) / mu - (v_r * v_phi) / r

    # Optional periodic renormalization of v
    eval_counter["n"] += 1
    if renormalize_every and (eval_counter["n"] % renormalize_every == 0):
        vmag = np.hypot(v_r, v_phi)
        if vmag > 0.0:
            v_r /= vmag
            v_phi /= vmag

    return np.array([drds, dphids, dv_r, dv_phi], dtype=float)
