#!/usr/bin/env python
# --------------------------------------------------------
"""Unit tests for PyRAY.library functions."""

import unittest
import numpy as np
import PyRAY
from PyRAY.library import den2freq  # replace with actual import path


class TestDen2Freq(unittest.TestCase):
    def test_scalar_input(self):
        # Known density and expected frequency (using constant cp)
        density = 1e12
        freq = den2freq(density)
        self.assertIsInstance(freq, float)
        self.assertGreater(freq, 0)

    def test_array_input(self):
        density = np.array([1e10, 1e11, 1e12])
        freq = den2freq(density)
        self.assertTrue(np.all(freq > 0))
        self.assertEqual(freq.shape, density.shape)

    def test_zero_density(self):
        self.assertEqual(den2freq(0), 0.0)

    def test_negative_density(self):
        with self.assertRaises(ValueError):
            den2freq(-1e10)
