import numpy as np
from scipy.interpolate import interp1d
import PyIRI
import PyIRI.main_library as ml


def constants():
    """Define constants for virtual height calculation."""
    cp = 8.97866275
    g_p = 2.799249247e10
    return cp, g_p


def den2freq(density):
    """Convert plasma density to plasma frequency."""
    cp, _ = constants()
    frequency = np.sqrt(density) * cp
    return frequency


def freq2den(frequency):
    """Convert plasma frequency to plasma density."""
    cp, _ = constants()
    density = (frequency / cp) ** 2
    return density


def find_X(n_e, f):
    """Calculate reflection height ratio X = (fp / f)^2."""
    cp, _ = constants()
    X = den2freq(n_e) ** 2 / f ** 2
    return X


def find_Y(f, b):
    """Calculate the gyrofrequency to ionosonde frequency ratio."""
    _, g_p = constants()
    Y = g_p * b / f
    return Y


def find_mu_mup(X, Y, bpsi, mode):
    """Calculate phase and group refractive indices (mu, mu′)."""
    YT = Y * np.sin(np.deg2rad(bpsi))
    YL = Y * np.cos(np.deg2rad(bpsi))
    Xm1 = 1.0 - X
    alpha = 0.25 * YT ** 4 + YL ** 2 * Xm1 ** 2
    beta = np.sqrt(alpha)

    mode_mult = 1.0 if mode == 'O' else -1.0
    D = Xm1 - 0.5 * YT ** 2 + mode_mult * beta
    mu = np.sqrt(1.0 - X * Xm1 / D)

    mu[mu < 0.0] = 0.0
    mu[mu > 1.0] = np.nan

    dbetadX = -YL ** 2 * Xm1 / beta
    dDdX = -1.0 + mode_mult * dbetadX

    dY_rad = np.deg2rad(bpsi)
    dalphadY = YT ** 3 * np.sin(dY_rad) + 2.0 * YL * Xm1 ** 2 * np.cos(dY_rad)
    dbetadY = 0.5 * dalphadY / beta
    dDdY = -YT * np.sin(dY_rad) + mode_mult * dbetadY

    dmudY = (X * Xm1 * dDdY) / (2.0 * mu * D ** 2)
    dmudX = (1.0 / (2.0 * mu * D)) * (2.0 * X - 1.0 + X * Xm1 / D * dDdX)

    mup = mu - (2.0 * X * dmudX + Y * dmudY)

    return mu, mup


def find_vh(X, Y, bpsi, dh, alt_min, mode):
    """Calculate virtual height."""
    _, mup = find_mu_mup(X, Y, bpsi, mode)
    vh = np.nansum(mup * dh, axis=1) + alt_min
    return vh


def smooth_nonuniform_grid(start, end, n_points, sharpness):
    """Generate a smooth non-uniform vertical grid."""
    u = np.linspace(0.0, 1.0, n_points)
    flipped_u = 1.0 - u
    factor = (np.exp(sharpness * flipped_u) - 1.0) / \
             (np.exp(sharpness) - 1.0)
    x = 1.0 - (start + (end - start) * factor)
    return x


def regrid_to_nonuniform_grid(f, n_e, b, bpsi, aalt, npoints):
    """Regrid ionospheric parameters to a non-uniform vertical grid."""
    start = 0
    end = 1
    sharpness = 10.0
    multiplier = smooth_nonuniform_grid(start, end, npoints, sharpness)

    N_grid = multiplier.size
    N_freq = f.size
    ind_grid = np.arange(N_grid)

    ind_max = np.argmax(n_e)
    n_e = n_e[:ind_max]
    b = b[:ind_max]
    bpsi = bpsi[:ind_max]
    aalt = aalt[:ind_max]

    dh = 1e-20
    critical_height = np.interp(f, den2freq(n_e), aalt) - dh

    multiplier_2d = np.full((N_freq, N_grid), multiplier)
    critical_height_2d = np.transpose(np.full((N_grid, N_freq), 
                                               critical_height))
    new_alt_2d = multiplier_2d * (critical_height_2d - aalt[0]) + aalt[0]
    dh_2d = np.concatenate((np.diff(new_alt_2d, axis=1),
                            np.full((N_freq, 1), dh)), axis=1)

    new_ind_2d = np.full((N_freq, N_grid), ind_grid)
    new_alt_1d = np.reshape(new_alt_2d, new_ind_2d.size)

    den_mod = np.reshape(np.interp(new_alt_1d, aalt, n_e), new_alt_2d.shape)
    bmag_mod = np.reshape(np.interp(new_alt_1d, aalt, b), new_alt_2d.shape)
    bpsi_mod = np.reshape(np.interp(new_alt_1d, aalt, bpsi), new_alt_2d.shape)
    ionosonde_freq_mod = np.transpose(np.full((N_grid, N_freq), f))

    return (ionosonde_freq_mod, den_mod, bmag_mod, bpsi_mod,
            dh_2d, critical_height_2d, new_ind_2d, ind_grid)


def vertical_to_magnetic_angle(inclination_deg):
    """Compute angle between vertical and magnetic field vector."""
    vertical_angle = 90.0 - np.abs(inclination_deg)
    return vertical_angle


def virtical_forward_operator(freq, den, bmag, bpsi, alt,
                              mode='O', n_points=2000):
    """Compute virtual height for O/X mode propagation."""
    foF2 = np.max(den2freq(den))
    ind = np.where((freq * 1e6) < foF2)
    freq_lim = freq[ind] * 1e6

    vh = np.full(freq.size, np.nan)

    (freq_mod, den_mod, bmag_mod, bpsi_mod, dh_2d, _, _, _) = \
        regrid_to_nonuniform_grid(freq_lim, den, bmag, bpsi, alt, n_points)

    aX = find_X(den_mod, freq_mod)
    aY = find_Y(freq_mod, bmag_mod)
    vh[ind] = find_vh(aX, aY, bpsi_mod, dh_2d, np.min(alt), mode)

    return vh
