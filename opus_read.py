from brukeropusreader import read_file
import numpy as np
import matplotlib.pyplot as plt

FILENAME = "./data-bruker/03-WBC717T-BlackFW255-WP.0"
opus = read_file(FILENAME)

wl = opus.get_range("AB")
abs = opus["AB"]

import ir_spectra_cmp
ir_spectra_cmp.ir_spectra_disp(wl, abs, True)

