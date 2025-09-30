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
        Constant that relates plasma frequency to plasma density in Hz m^-1.5.
    g_p : float
        g_p * B is the electron gyrofrequency in Hz/T.

    Notes
    -----
    This function provides constants used in virtual height calculations.

    """
    # Constant to convert density to frequency (MHz)
    cp = 8.97866275
    # Proton gyrofrequency constant (Hz/T)
    g_p = 2.799249247e10
    return cp, g_p


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
    cp, _ = constants()

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
    cp, _ = constants()
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
    _, g_p = constants()
    Y = g_p * b / f
    return Y


def find_mu_mup(X, Y, bpsi, mode):
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

    Returns
    -------
    mu : array-like
        Phase refractive index μ.
    mup : array-like
        Group refractive index μ′.

    """
    # Compute transverse and longitudinal components of Y
    YT = Y * np.sin(np.deg2rad(bpsi))
    YL = Y * np.cos(np.deg2rad(bpsi))

    # Compute 1-X
    Xm1 = 1.0 - X

    # Calculate alpha and beta as intermediate terms for refractive index
    alpha = 0.25 * YT**4 + YL**2 * Xm1**2
    beta = np.sqrt(alpha)

    # Set mode multiplier depending on propagation mode
    if mode == 'O':
        modeMult = 1.
    if mode == 'X':
        modeMult = -1.

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
        'O' or 'X' mode.

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


def vertical_forward_operator(freq, den, bmag, bpsi, alt, mode='O',
                              n_points=2000):
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
        'O' or 'X' propagation mode.
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
    # frequency foF2 and convert form MHz to Hz.
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
    mode = 'O'
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


def build_refractive_index_interpolator(z_grid: np.ndarray,
                                        x_grid: np.ndarray,
                                        n_field: np.ndarray,
                                        *,
                                        fill_value_n: float = np.nan,
                                        fill_value_grad: float = 0.0,
                                        bounds_error: bool = False,
                                        edge_order: int = 2,
                                        ) -> Callable[[np.ndarray,
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
    n_and_grad : callable
        Function accepting arrays (x, z) and returning:
            n : ndarray
                Refractive index at (x, z).
            dndx : ndarray
                ∂n/∂x [1/km] at (x, z).
            dndz : ndarray
                ∂n/∂z [1/km] at (x, z).

    Notes
    -----
    Inputs (x, z) can be scalars or arrays; they are broadcast to a common
    shape. The returned function simply delegates to
    eval_refractive_index_and_grad, keeping the interpolators “baked in” so you
    don't need to pass them around separately.
    Typical usage is inside build_refractive_index_interpolator.

    """
    def n_and_grad(x: np.ndarray,
                   z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return eval_refractive_index_and_grad(x, z,
                                              n_interp,
                                              dn_dx_interp, dn_dz_interp)
    return n_and_grad


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
    return y[1] - z_ground_km


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


def trace_ray_cartesian_stratified(f0_Hz: float,
                                   elevation_deg: float,
                                   alt_km: np.ndarray,
                                   Ne: np.ndarray,
                                   Babs: np.ndarray,
                                   bpsi: np.ndarray,
                                   mode: str = "O",) -> Dict[str, float]:
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
        Angle between B and k-vector [rad].
    mode : str
        Wave mode: 'O' or 'X'.

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
    c_km_per_s = 299792.458
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
                         Tuple[np.ndarray,
                               np.ndarray,
                               np.ndarray]],
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
    # Group delay input
    mup_func: Optional[Callable[[np.ndarray,
                                 np.ndarray],
                                np.ndarray]] = None,) -> Dict[str, Any]:
    """Gradient-based Cartesian ray tracing (flat Earth, 2D).

    Parameters
    ----------
    n_and_grad : callable
        Function (x, z) -> (μ, dμ/dx, dμ/dz).
        Usually built with `build_refractive_index_interpolator`.
    x0_km, z0_km : float
        Start coordinates [km].
    elevation_deg : float
        Launch elevation above horizontal [deg].
    s_max_km : float, default 5000
        Maximum arc length to integrate [km].

    Integration controls
    --------------------
    rtol, atol : float
        Relative and absolute tolerances for ODE solver.
    max_step_km : float or None
        Maximum solver step [km]. None = adaptive.

    Domain / stop conditions
    ------------------------
    z_ground_km : float
        Ground height [km]. Stops when z <= this.
    z_min_km, z_max_km : float
        Vertical bounds [km].
    x_min_km, x_max_km : float
        Horizontal bounds [km].

    Numerical hygiene
    -----------------
    renormalize_every : int
        Re-normalize velocity every N RHS calls.

    Group delay input
    -----------------
    mup_func : callable or None
        If provided, must evaluate μ′ (group index) at (x, z).
        Used for group delay integration:
            τ = ∫ (μ′/c) ds
        If None, delay falls back to vacuum value:
            τ = path_length / c

    Returns
    -------
    result : dict
        {
          'sol': OdeSolution (SciPy object),
          't': 1D array of arc length samples [km],
          'x': 1D array of horizontal positions [km],
          'z': 1D array of altitudes [km],
          'vx','vz': direction cosines along the ray,
          'status': str,
           # 'ground' | 'domain' | 'length' | 'failure' |'success'
          'group_path_km': float,
          'group_delay_sec': float,
          'x_midpoint': float,
          'z_midpoint': float,
          'ground_range_km': float or NaN
        }

    Notes
    -----
    Geometry is solved by integrating the ray equations using
    the phase index μ(x, z):
    dr/ds = v            with ||v|| = 1
    dv/ds = (1/μ) [∇μ - (∇μ·v) v]

    Here r = (x, z) is position, v = (vx, vz) is the unit tangent,
    and s is arc length [km].
    Geometry uses μ (phase index for bending.
    This ensures consistency with Snell's law in stratified media.

    Group delay is optional:
    If mup_func is given, integrates μ' (group index) along path.
    Otherwise uses vacuum c, which underestimates true delay.

    This is a Cartesian 2D flat-Earth model:
    No curvature of Earth included.
    Good for validation and simple comparisons.
    Extendable to spherical geometry if needed.

    Events terminate integration when the ray:
    Hits the ground (z <= z_ground_km),
    Leaves vertical or horizontal domain bounds,
    Exceeds arc length budget s_max_km.

    Numerical safety:
    Renormalization keeps ||v|| = 1 to avoid drift.
    NaN or nonpositive μ values trigger termination.

    """
    # Initial conditions
    elev = np.deg2rad(elevation_deg)
    vx0, vz0 = np.cos(elev), np.sin(elev)
    vnorm = np.hypot(vx0, vz0)
    vx0, vz0 = vx0 / vnorm, vz0 / vnorm
    y0 = np.array([x0_km, z0_km, vx0, vz0], dtype=float)

    eval_counter = {'n': 0}

    # Event functions
    events = [lambda s, y: event_ground(s, y, z_ground_km),
              lambda s, y: event_z_top(s, y, z_max_km),
              lambda s, y: event_z_bottom(s, y, z_min_km),
              lambda s, y: event_x_left(s, y, x_min_km),
              lambda s, y: event_x_right(s, y, x_max_km),]

    for ev in events:
        ev.terminal, ev.direction = True, -1.0

    # Integrate ODE
    sol = solve_ivp(lambda s, y: ray_rhs_cartesian(s, y,
                                                   n_and_grad,
                                                   renormalize_every,
                                                   eval_counter),
                    (0.0, s_max_km),
                    y0,
                    method="RK45",
                    rtol=rtol,
                    atol=atol,
                    max_step=max_step_km,
                    events=events,
                    dense_output=True,)

    # Termination reason
    if sol.status == 1:
        status = "ground" if len(sol.t_events[0]) > 0 else "domain"
    elif sol.status == 0:
        status = "length"
    elif sol.status == -1:
        status = "failure"
    else:
        status = "success"

    # Path arrays
    y = sol.y
    x_path = y[0, :]
    z_path = y[1, :]

    # Geometric path length
    dx, dz = np.diff(x_path), np.diff(z_path)
    ds = np.hypot(dx, dz)
    group_path_km = float(np.nansum(ds))

    # Group delay
    c_km_per_s = 299792.458
    if mup_func is not None and ds.size > 0:
        x_mid = 0.5 * (x_path[:-1] + x_path[1:])
        z_mid = 0.5 * (z_path[:-1] + z_path[1:])
        mup_mid = np.asarray(mup_func(x_mid, z_mid), dtype=float)
        valid = np.isfinite(mup_mid)
        group_delay_sec = float(np.nansum((mup_mid[valid] / c_km_per_s)
                                          * ds[valid]))
    else:
        group_delay_sec = group_path_km / c_km_per_s

    # Midpoint
    mid_idx = len(sol.t) // 2
    x_midpoint = float(x_path[mid_idx]) if x_path.size else np.nan
    z_midpoint = float(z_path[mid_idx]) if z_path.size else np.nan

    # Ground landing
    ground_range_km = float(x_path[-1]) if status == "ground" else np.nan

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
