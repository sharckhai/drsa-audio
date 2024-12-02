import os
import sys
import math


def round_down(n: float, decimalpoints: int):
    """Custom function to round down to a given number of decimalpoints.
    
    -----
    Args:
        n (float): number to round.
        decimalpoints (int): number of decimalpoitns to round number to.
    Returns:
        rounded_number (float): down-rounded number
    """
    return math.floor(n * 10**decimalpoints) / 10**decimalpoints


class HiddenPrints:
    """Class to hide prints from 'fit_baseline()' function during grid search."""
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout