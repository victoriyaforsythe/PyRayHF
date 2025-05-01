#!/usr/bin/env python
# --------------------------------------------------------
"""Unit tests for PyRAY.library functions."""

import numpy as np
import unittest
from PyRAY.library import den2freq


class TestDen2Freq(unittest.TestCase):
    """Unit tests for the `den2freq` function."""

    def test_scalar_input(self):
        """Test that a scalar density returns a positive scalar frequency."""
        density = 1e12
        freq = den2freq(density)
        self.assertIsInstance(freq, float)
        self.assertGreater(freq, 0)

    def test_array_input(self):
        """Test that an array of dens returns an array of positive freq."""
        density = np.array([1e10, 1e11, 1e12])
        freq = den2freq(density)
        self.assertTrue(np.all(freq > 0))
        self.assertEqual(freq.shape, density.shape)

    def test_zero_density(self):
        """Test that zero density returns zero frequency."""
        self.assertEqual(den2freq(0), 0.0)

    def test_negative_density(self):
        """Test that negative density raises a ValueError."""
        with self.assertRaises(ValueError):
            den2freq(-1e10)
