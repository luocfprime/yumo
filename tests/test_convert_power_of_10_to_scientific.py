import numpy as np

from yumo.utils import convert_power_of_10_to_scientific


def test_convert_power_of_10_to_scientific():
    x = 1.21e12
    coeff, exp = convert_power_of_10_to_scientific(np.log10(x))
    assert np.isclose(coeff, 1.21)
    assert exp == 12
