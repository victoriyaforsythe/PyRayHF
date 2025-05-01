import numpy as np
import PyIRI
import PyIRI.main_library as ml
from scipy.interpolate import interp1d

def constants():
    """Define constants for virtual height calculation.

    Parameters
    ----------
    none

    Returns
    -------
    cp: flt
        Constant that relates plasma frequency to plasma density in Hz m^-1.5
    g_p: flt
        g_p * B is electron gyrofrequency

    Notes
    -----
    This function gives constants for VH.

    """

    cp = 8.97866275
    g_p = 2.799249247e10

    return cp, g_p

def den2freq(density):
    """Convert plasma density to plasma frequency.

    Parameters
    ----------
    density: flt, array-like

    Returns
    -------
    frequency: flt, array-like
        Plasma frequency in (Hz)

    Notes
    -----
    This function converts given plasma density to plasma frequency.

    """
    # Declaring constants
    cp, _ = constants()
    frequency = np.sqrt(density) * cp
    return frequency

def freq2den(frequency):
    """Convert plasma frequency to plasma density.

    Parameters
    ----------
    frequency: flt, array-like
        Plasma frequency in (Hz)
        
    Returns
    -------
    density: flt, array-like
        Plasma density in (m-3)

    Notes
    -----
    This function converts given plasma frequency to plasma density.

    """
    # Declaring constants
    cp, _ = constants()
    density = (frequency / cp)**2
    return density

def find_X(n_e, f):
    """Calculate reflection height

    Parameters
    ----------
    n_e: array-like
        Electron density array in (m-3)
    f: array-like
        Frequency of the ionosonde in (Hz)

    Returns
    -------
    X: flt, array-like
        The ratio of the square of the plasma frequency f_N to the 
        square of the ionosonde frequency f.

    Notes
    -----
    This function returns X, the ratio of the square of the plasma frequency
    f_N to the square of the ionosonde frequency f.

    """

    # Load constants
    cp, _ = constants()
    X = (den2freq(n_e))**2 / f**(2)

    return X


def find_Y(f, b):
    """Calculate the gyrofrequency to the ionosonde frequency ratio

    Parameters
    ----------
    f: array-like
        Ionosonde frequency in Hz
    b: array-like
        Magnitude of the magnetic field in Tesla.

    Returns
    -------
    Y: array-like
        The the ratio of electron gyrofrequency and the ionosonde frequency.

    Notes
    -----
    This function calculates the ratio of electron gyrofrequency and the 
    ionosonde frequency.

    """
    _, g_p = constants()
    Y = g_p * b / f
    return Y


def find_mu_mup(X, Y, bpsi, mode):
    """Calculate group refractive index 

    Parameters
    ----------
    X: array-like
        Ratio of plasma and emission frequncies
    Y: array-like
        Ratio of electron gyrofrequency and emission frequency 
    bpsi: array-like
        The angle ψ between the wave vector and the Earth's magnetic field.
    mode: str
        Mode of propagation, O or X.

    Returns
    -------
    mu: array-like
        Phase refractive index μ
    mup: array-like
        Group refractive index μ′

    Notes
    -----
    This function calculates the phase refractive index μ and the group
    refractive index μ′

    """
    YT = Y * np.sin(np.deg2rad(bpsi))
    YL = Y * np.cos(np.deg2rad(bpsi))
    
    #index = X > 1.0
    #X[index] = 1.0
    Xm1 = 1.0 - X

    
    # Allow imaginary part for sqrt of potentially negative alpha
    alpha = 0.25 * YT**4 + YL**2 * Xm1**2
    beta = np.sqrt(alpha)

    # O-mode / X-mode multiplier
    if mode == 'O':
        modeMult = 1.
    if mode == 'X':
        modeMult = -1.

    # Also allow imaginary D and mu
    
    D = Xm1 - 0.5 * YT**2 + modeMult * beta
    mu = np.sqrt(1. - X * Xm1 / D)

    # Limit the mu array
    mu[np.where(mu < 0.)] = 0.
    mu[np.where(mu > 1.)] = np.nan

    # Derivatives
    dbetadX = -YL**2 * Xm1 / beta
    dDdX = -1. + modeMult * dbetadX

    dalphadY = (YT**3 * np.sin(np.deg2rad(bpsi))) + (2. * YL * Xm1**2 * np.cos(np.deg2rad(bpsi)))
    dbetadY = 0.5 * dalphadY / beta
    dDdY = -YT * np.sin(np.deg2rad(bpsi)) + modeMult * dbetadY

    # Partials of mu
    dmudY = (X * Xm1 * dDdY) / (2. * mu * D**2)
    dmudX = (1. / (2. * mu * D)) * (2. * X - 1. + X * Xm1 / D * dDdX)

    # The group refractive index μ′
    mup = mu - (2. * X * dmudX + Y * dmudY)

    return mu, mup


def find_vh(X, Y, bpsi, dh, alt_min, mode):
    """Calculate virtual height

    Parameters
    ----------
    X: array-like
        The ratio of the square of the plasma frequency f_N to the 
        square of the ionosonde frequency f.
    Y: array-like
        The the ratio of electron gyrofrequency and the ionosonde frequency.
    bpsi: array-like
        The angle ψ between the wave vector and the Earth's magnetic field.
    dh: array-like
        The grid distances in km.
    alt_min: flt
        Minimum altitude in km.
    mode: str
        Mode of propagation, 'O' or 'X'.

    Returns
    -------
    mu: dict
        Phase refractive index μ for O and X modes
    mup: dict
        Group refractive index μ' for O and X modes
    vh: dict
        Virtual height for O and X modes in km

    Notes
    -----
    This function calculates the virtual height for ordinary (O) and
    extraordinary (X) modes of signal propagation

    """

    # Find the phase refractive index μ and the group refractive index μ′ for
    # ordinary (O) and extraordinary (X) modes of signal propagation
    _, mup = find_mu_mup(X, Y, bpsi, mode)

    # Find virtual height as vertical integral through μ′
    vh = np.nansum(mup * dh, axis=1) + alt_min

    return vh


def smooth_nonuniform_grid(start, end, n_points, sharpness):
    """
    Generate a smooth non-uniform grid between `start` and `end`,
    where the resolution gradually increases toward the `end` point.

    The grid starts with coarse spacing near `start` and transitions 
    smoothly to finer spacing near `end`, controlled by the `sharpness`
    parameter.

    Parameters
    ----------
    start: float
        Starting value of the grid.
    end: float
        Ending value of the grid.
    n_points: int
        Number of grid points to generate.
    sharpness: float
        Controls how quickly the resolution transitions; higher values make
        the resolution change more abrupt.

    Returns
    -------
    x: ndarray
        1-D numpy array of grid points with smooth, non-uniform spacing.

    Notes
    -----
    The method uses an exponential transformation of a uniform grid
    to achieve non-uniformity. 
    """

    # Uniform grid [0, 1]
    u = np.linspace(0.0, 1.0, n_points)
    # Flip to make fine resolution near 'end'
    flipped_u = 1.0 - u
    factor = (np.exp(sharpness * flipped_u) - 1.0) / (np.exp(sharpness) - 1.0)
    x = 1. - (start + (end - start) * factor)
    return x


def regrid_to_nonuniform_grid(f, n_e, b, bpsi, aalt, npoints):
    """
    Generate a smooth non-uniform grid between `start` and `end`,
    where the resolution gradually increases toward the `end` point.

    The grid starts with coarse spacing near `start` and transitions 
    smoothly to finer spacing near `end`, controlled by the `sharpness`
    parameter.

    Parameters
    ----------
    f: array-like
        Frequency of the ionosonde in (Hz)
    n_e: array-like
        Electron density array in (m-3)
    b: array-like
        Magnitude of the magnetic field
    bpsi: array-like
        The angle ψ between the wave vector and the Earth's magnetic field.
    aalt: array-like
        Array of altitudes for the given profile in km.
    npoints: int
        Number of points in the new vertical grid. For O-mode even 200 points
        can be enough, for X-mode, this number should be increased to 10000.

    Returns
    -------
    f: array-like
        Frequency of the ionosonde in (Hz)


    Notes
    -----
    The method uses an exponential transformation of a uniform grid
    to achieve non-uniformity. 
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
    critical_height_2d = np.transpose(np.full((N_grid, N_freq), critical_height))
    new_alt_2d = multiplier_2d * (critical_height_2d - aalt[0]) + aalt[0]

    dh_2d = np.concatenate((np.diff(new_alt_2d, axis=1), np.full((N_freq, 1), dh)), axis=1)

    new_ind_2d = np.full((N_freq, N_grid), ind_grid)

    # Flattened array of new altitudes where we want to sample the density
    # profile, so we can apply 1-D Numpy interpolation, that is faster
    new_alt_1d = np.reshape(new_alt_2d, new_ind_2d.size)
    # Create arrays on the modified grid
    den_mod = np.reshape(np.interp(new_alt_1d, aalt, n_e), new_alt_2d.shape)
    bmag_mod = np.reshape(np.interp(new_alt_1d, aalt, b), new_alt_2d.shape)
    bpsi_mod = np.reshape(np.interp(new_alt_1d, aalt, bpsi), new_alt_2d.shape)
    ionosonde_freq_mod = np.transpose(np.full((N_grid, N_freq), f))

    return ionosonde_freq_mod, den_mod, bmag_mod, bpsi_mod, dh_2d, critical_height_2d, new_ind_2d, ind_grid



def vertical_to_magnetic_angle(inclination_deg):
    """
    Calculates the angle (in degrees) between the vertical ray path 
    and the magnetic field vector given the magnetic inclination.

    Parameters
    ----------
    inclination_deg: flt or np.ndarray
        Magnetic inclination angle in degrees.
        Positive = downward field, Negative = upward.

    Returns
    -------
    vertical_angle: float or np.ndarray
        Angle between vertical and magnetic field vector in degrees.

    """

    vertical_angle = 90.0 - np.abs(inclination_deg)

    return vertical_angle


def virtical_forward_operator(freq, den, bmag, bpsi, alt, mode='O', n_points=2000):
    """
    Calculates the virtual height.

    Parameters
    ----------
    freq: array-like
        Frequency of the ionosonde in (MHz)
    den: array-like
        Electron density array in (m-3)
    bmag: array-like
        Magnitude of the magnetic field
    bpsi: array-like
        The angle ψ between the wave vector and the Earth's magnetic field.
    alt: array-like
        Array of altitudes for the given profile in km.
    mode: str
        Mode of propagation, ordinary 'O' or extraordinary 'X'.
        Default is O-mode.
    n_points: int
        Number of points in the vertical grid.
        For O-mode even 200 points can be enough.
        for X-mode, this number should be increased to 10000.
        Default is 2000 points.

    Returns
    -------
    vh: np.ndarray
        Virtual height in km, same size as a given freq array.

    """

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
    (freq_mod,
    den_mod,
    bmag_mod,
    bpsi_mod,
    dh_2d,
    critical_height_2d,
    new_ind_2d,
    ind_grid) = regrid_to_nonuniform_grid(freq_lim,
                                          den,
                                          bmag,
                                          bpsi,
                                          alt,
                                          n_points)

    # Find the ratio of the square of the plasma frequency f_N to the square of
    # the ionosonde frequency f.
    aX = find_X(den_mod, freq_mod)

    # Find the ratio of electron gyrofrequency and the ionosonde frequency
    aY = find_Y(freq_mod, bmag_mod)

    # Find virtual height
    vh[ind] = find_vh(aX, aY, bpsi_mod, dh_2d, np.min(alt), mode)

    return vh