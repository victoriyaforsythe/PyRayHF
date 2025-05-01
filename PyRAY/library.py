import numpy as np
import PyIRI
import PyIRI.main_library as ml
from scipy.interpolate import interp1d


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
    cp = 8.97866275
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
    cp, _ = constants()
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
    cp, _ = constants()
    density = (frequency / cp) ** 2
    return density


def find_X(n_e, f):
    """Calculate X: the square of the plasma frequency over the square of the ionosonde frequency.

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
    X = den2freq(n_e) ** 2 / f ** 2
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
    YT = Y * np.sin(np.deg2rad(bpsi))
    YL = Y * np.cos(np.deg2rad(bpsi))
    Xm1 = 1.0 - X

    alpha = 0.25 * YT ** 4 + YL ** 2 * Xm1 ** 2
    beta = np.sqrt(alpha)

    modeMult = 1.0 if mode == 'O' else -1.0
    D = Xm1 - 0.5 * YT ** 2 + modeMult * beta
    mu = np.sqrt(1.0 - X * Xm1 / D)

    # Clamp values
    mu[mu < 0.0] = 0.0
    mu[mu > 1.0] = np.nan

    # Partial derivatives for group index
    dbetadX = -YL ** 2 * Xm1 / beta
    dDdX = -1.0 + modeMult * dbetadX

    dalphadY = (YT ** 3 * np.sin(np.deg2rad(bpsi))) + (
        2.0 * YL * Xm1 ** 2 * np.cos(np.deg2rad(bpsi))
    )
    dbetadY = 0.5 * dalphadY / beta
    dDdY = -YT * np.sin(np.deg2rad(bpsi)) + modeMult * dbetadY

    dmudY = (X * Xm1 * dDdY) / (2.0 * mu * D ** 2)
    dmudX = (1.0 / (2.0 * mu * D)) * (2.0 * X - 1.0 + X * Xm1 / D * dDdX)

    mup = mu - (2.0 * X * dmudX + Y * dmudY)
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
    _, mup = find_mu_mup(X, Y, bpsi, mode)
    vh = np.nansum(mup * dh, axis=1) + alt_min
    return vh


def smooth_nonuniform_grid(start, end, n_points, sharpness):
    """Generate smooth non-uniform grid from `start` to `end`.

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
    u = np.linspace(0.0, 1.0, n_points)
    flipped_u = 1.0 - u
    factor = (np.exp(sharpness * flipped_u) - 1.0) / (np.exp(sharpness) - 1.0)
    x = 1.0 - (start + (end - start) * factor)
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
    Tuple of regridded arrays.
    """
    start, end, sharpness = 0, 1, 10.0
    multiplier = smooth_nonuniform_grid(start, end, npoints, sharpness)

    N_grid = multiplier.size
    N_freq = f.size
    ind_grid = np.arange(N_grid)

    # Trim input to foF2
    ind_max = np.argmax(n_e)
    n_e, b, bpsi, aalt = n_e[:ind_max], b[:ind_max], bpsi[:ind_max], aalt[:ind_max]

    dh = 1e-20
    critical_height = np.interp(f, den2freq(n_e), aalt) - dh

    multiplier_2d = np.full((N_freq, N_grid), multiplier)
    critical_height_2d = np.transpose(np.full((N_grid, N_freq), critical_height))
    new_alt_2d = multiplier_2d * (critical_height_2d - aalt[0]) + aalt[0]

    dh_2d = np.concatenate((np.diff(new_alt_2d, axis=1), np.full((N_freq, 1), dh)), axis=1)
    new_ind_2d = np.full((N_freq, N_grid), ind_grid)
    new_alt_1d = np.reshape(new_alt_2d, new_ind_2d.size)

    den_mod = np.reshape(np.interp(new_alt_1d, aalt, n_e), new_alt_2d.shape)
    bmag_mod = np.reshape(np.interp(new_alt_1d, aalt, b), new_alt_2d.shape)
    bpsi_mod = np.reshape(np.interp(new_alt_1d, aalt, bpsi), new_alt_2d.shape)
    ionosonde_freq_mod = np.transpose(np.full((N_grid, N_freq), f))

    return ionosonde_freq_mod, den_mod, bmag_mod, bpsi_mod, dh_2d, critical_height_2d, new_ind_2d, ind_grid


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
    return 90.0 - np.abs(inclination_deg)


def virtical_forward_operator(freq, den, bmag, bpsi, alt, mode='O', n_points=2000):
    """Calculate virtual height from ionosonde frequency and ionosphere profile.

    Parameters
    ----------
    freq : array-like
        Frequency in MHz.
    den : array-like
        Electron density in m^-3.
    bmag : array-like
        Magnetic field magnitude.
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
    foF2 = np.max(den2freq(den))
    ind = np.where((freq * 1e6) < foF2)
    freq_lim = freq[ind] * 1e6
    vh = np.full(freq.shape, np.nan)

    (freq_mod, den_mod, bmag_mod, bpsi_mod, dh_2d,
     critical_height_2d, new_ind_2d, ind_grid) = regrid_to_nonuniform_grid(
        freq_lim, den, bmag, bpsi, alt, n_points
    )

    aX = find_X(den_mod, freq_mod)
    aY = find_Y(freq_mod, bmag_mod)
    vh[ind] = find_vh(aX, aY, bpsi_mod, dh_2d, np.min(alt), mode)

    return vh
