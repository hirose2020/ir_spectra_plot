import jcamp
import matplotlib.pyplot as plt
import numpy as np
import ir_spectra_cmp

FILENAME = "./data-dx/benzene.jdx"

dx = jcamp.jcamp_readfile(FILENAME)
x = dx["x"]
y = dx["y"]

ir_spectra_cmp.ir_spectra_disp(x, y, True)

"""
plt.plot(x, y)
plt.title("benzene")
plt.show()

"""
