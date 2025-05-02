Example: Generate Input Arrays Using PyIRI 
============================================

1. Import libraries.

::

    import datetime
    import pickle

    import numpy as np
    import matplotlib.pyplot as plt

    import PyIRI
    import PyIRI.main_library as pyiri_main
    import PyIRI.igrf_library as igrf
    import PyRAY

2. Select the day and Universal Time of interest.

::

    year = 2020
    month = 4
    day = 1
    UT = 10.0

3. Provide the F10.7 value for this day. You can look it up at OMNIWeb:
https://omniweb.gsfc.nasa.gov/form/dx1.html

::

    F107 = 69.4

4. Create datetime object for the selected day.

::

    dtime_day = datetime.datetime(year, month, day)

5. Define geographic location (longitude, latitude in degrees)

::

    lon = 10.0
    lat = 0.0

6. Create an array of altitudes (km).

::

    aalt = np.arange(90., 600., 1.)

7. Run PyIRI for the selected time and location.

::

    _, _, _, _, _, _, edp = pyiri_main.IRI_density_1day(year,
                                                        month,
                                                        day,
                                                        np.array([UT]),
                                                        np.array([lon]),
                                                        np.array([lat]),
                                                        aalt,
                                                        F107,
                                                        PyIRI.coeff_dir,
                                                        ccir_or_ursi=1)

8. Extract 1-D electron density profile

::

    den = edp[0, :, 0]

9. Compute magnetic inclination and field strength at min and max altitudes.

::

    _, _, _, _, _, inc_min, bmag_min = igrf.inclination(PyIRI.coeff_dir,
                                                    dtime_day,
                                                    np.array([lon]),
                                                    np.array([lat]),
                                                    np.min(aalt))

    _, _, _, _, _, inc_max, bmag_max = igrf.inclination(PyIRI.coeff_dir,
                                                        dtime_day,
                                                        np.array([lon]),
                                                        np.array([lat]),
                                                        np.max(aalt))

10. Compute angles between the magnetic field and vertical ray.

::

    bpsi_min = PyRAY.library.vertical_to_magnetic_angle(inc_min[0])
    bpsi_max = PyRAY.library.vertical_to_magnetic_angle(inc_max[0])


11. Construct array of angles (°) between magnetic field and vertical ray.

::

    abpsi = np.linspace(bpsi_min, bpsi_max, aalt.size)

12. Construct array of magnetic field strengths in Tesla (IGRF outputs nT).

::

    abmag = np.linspace(bmag_min[0], bmag_max[0], aalt.size) / 1e9

13. Generate array of frequencies used by a vertical ionosonde (Hz).

::

    ionosonde_frequency = np.arange(1e6,
                                PyRAY.library.den2freq(np.max(den)),
                                0.1e6)

14. Combine inputs into a dictionary.

::

    input_example = {'den': den,
                 'alt': aalt,
                 'bmag': abmag,
                 'bpsi': abpsi,
                 'freq': ionosonde_frequency}

15. Save inputs to a pickle file

::

    file_save = 'Example_input.p'
    pickle.dump(input_example, open(file_save, "wb"))

.. image:: /docs/figures/Input_Arrays.png
    :width: 800px
    :align: center
    :alt: DP, Magnetic field strength, Magnetic field angle.
