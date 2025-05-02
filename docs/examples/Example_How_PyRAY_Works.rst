Example: How PyRAY Works
========================

In the presence of the Earth's magnetic field (but neglecting collisions), the **virtual height** is computed using the **group refractive index** :math:`n_g(h, f)` from the Appleton-Hartree dispersion relation.

The virtual height :math:`h_v(f)` is given by:

.. math::

   h_v(f) = \int_0^{h_r} n_g(h, f) \, \mathrm{d}h

Intermediate Quantities
------------------------

We define the following dimensionless parameters:

- Plasma-to-wave frequency ratio:

  .. math::

     X(h) = \frac{f_p(h)^2}{f^2}

- Gyro-to-wave frequency ratio:

  .. math::

     Y = \frac{f_B}{f}

- Propagation angle factor:

  .. math::

     \mu = \cos \Psi, \quad \mu' = \sin \Psi

  where :math:`\Psi` is the angle between the wave vector and the geomagnetic field.

The **Appleton-Hartree group refractive index** (for the ordinary or extraordinary mode, neglecting collisions) is then:

.. math::

   n_g(h, f) = \left[ 1 - \frac{X(h)}{1 \pm \frac{Y \mu}{\sqrt{1 - X(h) - Y^2 \mu'^2}}} \right]^{-1/2}

where:

- The :math:`+` sign corresponds to the **ordinary mode**,
- The :math:`-` sign corresponds to the **extraordinary mode**.

Evaluation
----------

To compute the virtual height numerically, discretize the integral using an electron density profile :math:`n_e(h)` and compute :math:`X(h)` and :math:`n_g(h)` at each layer.

The plasma frequency is:

.. math::

   f_p(h) = 8.98 \times 10^{-3} \cdot \sqrt{n_e(h)} \quad \text{[MHz]}

The electron gyrofrequency is:

.. math::

   f_B = \frac{e B}{2 \pi m_e} \approx 2.8 \times 10^{-3} \cdot B \quad \text{[MHz]}

with :math:`B` in nanotesla (nT).

Finally, compute:

.. math::

   h_v(f) \approx \sum_i n_{g, i}(f) \cdot \Delta h_i

where :math:`\Delta h_i` is the step size in altitude.






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