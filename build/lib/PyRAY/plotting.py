#!/usr/bin/env python
# ---------------------------------------------------------
# Distribution statement A. Approved for public release.
# Distribution is unlimited.
# This work was supported by the Office of Naval Research.
# ---------------------------------------------------------
"""This library contains components visualization routines for PyIRI."""

import os

import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def plot_ionosondes(y_instr_info,
                    dtime,
                    plot_dir,
                    plot_name='Ionosondes_Map.pdf'):
    """Plot map with GIRO ionosondes that were used for validation.

    Parameters
    ----------
    y_instr_info : dict
        Dictionary with ionosonde information
        lon : array-like
            Array of longitudes for the stations in degrees.
        ay_lat : array-like
            Array of latitudes latitudes for the stations in degrees.
        name : array-like
            Array of names of the stations.
    dtime : class:`datetime.datetime`
        Datetime of the validation day.
    plot_dir : str
        Direction where to save the figure.
    plot_name : str
        Output name, without directory, for the saved figure
        (default='Ionosondes_Map.pdf')
    """

    # Plot map with ionosondes before filtering
    projection1 = ccrs.PlateCarree()
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(9, 3),
                           constrained_layout=True,
                           subplot_kw={'projection': projection1})
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    ax_plot = ax
    ax_plot.set_xticks(np.arange(-180, 180 + 45, 90), crs=ccrs.PlateCarree())
    ax_plot.set_yticks(np.arange(-90, 90 + 45, 45), crs=ccrs.PlateCarree())
    ax_plot.set_xlabel('Geo Lon (°)')
    ax_plot.set_ylabel('Geo Lat (°)')
    ax_plot.set_facecolor('lightgray')
    lon_formatter = cticker.LongitudeFormatter()
    lat_formatter = cticker.LatitudeFormatter()
    ax_plot.xaxis.set_major_formatter(lon_formatter)
    ax_plot.yaxis.set_major_formatter(lat_formatter)
    ax_plot.coastlines(lw=0.5, color='black', zorder=1)
    ax_plot.scatter(y_instr_info['lon'], y_instr_info['lat'],
                    zorder=2, linewidth=0.5, edgecolor='black', c='red')
    ax_plot.set_title('GIRO Ionosondes for Validation Period, '
                      + dtime.strftime('%Y%m%d'))

    # Save figure
    figname = os.path.join(plot_dir, plot_name + '.pdf')
    plt.savefig(figname, bbox_inches='tight', facecolor='white')
    print('Figure Ionosonde Map is saved at: ', figname)
    return


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def plot_histogram(residuals,
                   model_units,
                   dtime,
                   plot_dir,
                   plot_name='Residuals.pdf'):
    """Plot residuals for each model parameter.

    Parameters
    ----------
    residuals : dict
        Dictionary with residuals.
    model_units: dict
        Dictionary with units for each key in residuals.
    dtime : class:`datetime.datetime`
        Datetime of the validation day.
    plot_dir : str
        Direction where to save the figure.
    plot_name : str
        Output name, without directory, for the saved figure
        (default='Residuals.pdf')

    """

    fig, ax = plt.subplots(1, len(residuals), sharex=False, sharey=False)
    fig.suptitle('Residuals, ' + dtime.strftime('%Y%m%d'))
    fig.set_size_inches(3 * len(residuals), 3)
    keys_list = np.array(list(residuals.keys()), dtype=str)

    for ikey in range(0, keys_list.size):
        ax_plot = ax[ikey]
        ax_plot.set_facecolor('lightgray')
        ax_plot.set_xlabel(keys_list[ikey]
                           + ' (' + model_units[keys_list[ikey]] + ')')
        ax_plot.set_ylabel('Number of obs')
        ax_plot.axvline(x=0, c='black', zorder=2, linestyle='dashed',
                        linewidth=0.5)
        # Plot the histogram of the model residuals
        ax_plot.hist(residuals[keys_list[ikey]], color='red',
                     bins=50, zorder=1)
        # Making x-axis symmetric
        low, high = ax_plot.get_xlim()
        bound = max(abs(low), abs(high))
        ax_plot.set_xlim((-bound, bound))
    plt.tight_layout()

    # Save figure
    figname = os.path.join(plot_dir, plot_name + '.pdf')
    plt.savefig(figname, bbox_inches='tight', facecolor='white')
    print('Figure Residuals is saved at: ', figname)
    return


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def plot_individual_mean_residuals(res_ion,
                                   ion_info,
                                   model_units,
                                   dtime,
                                   plot_dir,
                                   plot_name='Ion_Residuals.pdf'):
    """Plot residuals for each ionosonde.

    Parameters
    ----------
    res_ion : dict
        Dictionary with mean residuals for individual ionosondes.
    ion_info : dict
        Dictionary with ionosonde information
        lon : array-like
            Array of longitudes for the stations in degrees.
        ay_lat : array-like
            Array of latitudes latitudes for the stations in degrees.
        name : array-like
            Array of names of the stations.
    model_units: dict
        Dictionary with units for each key in res_ion.
    dtime : class:`datetime.datetime`
        Datetime of the validation day.
    plot_dir : str
        Direction where to save the figure.
    plot_name : str
        Output name, without directory, for the saved figure
        (default='Ion_Residuals.pdf')

    """
    # Loop through all parameters in the residuals dictionary
    for key in res_ion:

        fig, ax = plt.subplots(1, 1, sharex=False, sharey=False)
        fig.suptitle(key + ' Residuals, ' + dtime.strftime('%Y%m%d'))
        fig.set_size_inches(7, 3)

        ax_plot = ax
        ax_plot.set_facecolor('lightgray')
        ax_plot.set_ylabel(key + ' Residuals (' + model_units[key] + ')')
        ax_plot.set_xlabel('GIRO Ionosonde Names')
        ax_plot.bar(ion_info['name'], res_ion[key], color='red',
                    edgecolor='black', linewidth=0.5)
        plt.xticks(rotation='vertical')
        ax_plot.axhline(y=0, color='black', linewidth=0.5)

        # Making y-axis symmetric
        low, high = ax_plot.get_ylim()
        bound = max(abs(low), abs(high))
        ax_plot.set_ylim((-bound, bound))

        # Save figure
        figname = os.path.join(plot_dir, plot_name + '_' + key + '.pdf')
        plt.savefig(figname, bbox_inches='tight', facecolor='white')
        print('Figure Ionosonde Mean Residuals is saved at: ', figname)
    return
