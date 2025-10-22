#!/usr/bin/env python
"""
Test to see if NumPy usage triggers the teacher.
"""

import numpy as np

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
z = x + y

print(f"NumPy result: {z}")
print(f"Using np.number: {isinstance(x[0], np.number)}")
