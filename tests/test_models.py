"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt

import pytest

from inflammation.models import daily_mean

def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""
    

    test_input = np.array([[0, 0],
                           [0, 0],
                           [0, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_mean_integers():
    """Test that mean function works for an array of positive integers."""

    test_input = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])
    test_result = np.array([3, 4])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)

def test_daily_mean_floats():
    """Test that mean function works for an array of positive floats."""

    test_input = np.array([[1.3, 2.0],
                           [3.0, 4.0],
                           [5.0, 6.0]])
    test_result = np.array([3.1, 4.0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)

def test_daily_mean_unexpected_type():
    """Should fail if the dimensions of the input array are wrong."""
    
    test_input = np.array(['a', 'b', 'c'])

    with pytest.raises(TypeError):
        error_expected = daily_mean(test_input)
