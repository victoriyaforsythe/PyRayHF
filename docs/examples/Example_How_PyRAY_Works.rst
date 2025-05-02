Example: How PyRAY Works
========================


This section describes how to calculate the **virtual height** using only the magnetic field and the refractive index in a cold magnetoplasma.


Let:

- ``X = (f_p / f)^2`` — ratio of plasma frequency squared to wave frequency squared  
- ``Y = f_c / f`` — ratio of gyrofrequency to wave frequency  
- ``μ`` — refractive index  
- ``μ'`` — group refractive index  
- ``Ψ`` — angle between wave vector and magnetic field  

Then, the **group refractive index** μ′ is calculated from:

μ' = d(fμ) / df``

The virtual height h' is then given by:

h' = ∫ μ' dh``

LaTeX-rendered math (for Sphinx/ReadTheDocs)
--------------------------------------------

.. math::

   X &= \left( \frac{f_p}{f} \right)^2

.. math::

   Y &= \frac{f_c}{f}

.. math::

   \mu' = \frac{d(f \mu)}{df}

.. math::

   h' = \int \mu' \, dh

Here:

- :math:`f_p` is the plasma frequency
- :math:`f_c` is the electron gyrofrequency
- :math:`f` is the wave frequency
- :math:`\Psi` is the angle between the propagation direction and the magnetic field

This formulation allows computing virtual height profiles using only the magnetic field model and derived plasma parameters.




.. image:: /docs/figures/Stretched_Grid.png
    :width: 600px
    :align: center
    :alt: Stretched Grid.


.. image:: /docs/figures/Regridded_Input_Matrix.png
    :width: 600px
    :align: center
    :alt: Stretched Grid.


.. image:: /docs/figures/intermediate_Calculations.png
    :width: 600px
    :align: center
    :alt: Stretched Grid.

.. image:: /docs/figures/Virtual_Height.png
    :width: 600px
    :align: center
    :alt: Stretched Grid.