# -*- coding: utf-8 -*-
"""
IRスペクトルの波数圧縮プロット
Created on Tue Mar  3 09:31:40 2020
@author: yuji hirose
"""
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns

#sns.set()

def ir_spectra_disp(wl, ab, cmp=False):
    #AN, = wl.shape
    N = len(wl)
    wl_cmp = np.zeros([N], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="w")

    if cmp:
        ax.set_xticklabels(["500", "1000", "1500", "2000", "3000", "4000"])
        ax.set_xticks(np.linspace(500, 3000, 6))
        ax.set_xlim(400, 3000)
        for i in range(N):
            if wl[i] > 2000:
                wl_cmp[i] = wl[i] / 2.0 + 1000.0
            else:
                wl_cmp[i] = wl[i]
    else:
        ax.set_xticks(np.linspace(500, 4000, 8))
        ax.set_xlim(400, 4000)
        for i in range(N):
            wl_cmp[i] = wl[i]

    ax.plot(wl_cmp, ab, color="red")
    ax.invert_xaxis()
    ax.grid(axis='x', color='green', lw=1, ls='--')
    ax.grid(axis='y', color='green', lw=1, ls='--')
    ax.set_xlabel("wavenumber [cm$^{-1}$]")
    ax.set_ylabel("absorbance")
    plt.show()

def main():
    filename = "./data/MC02-01.0.dpt"

    with open(filename, "r") as f:
        lines = f.readlines()

    N = len(lines)
    x = np.zeros([N], dtype=np.float)
    y = np.zeros([N], dtype=np.float)

    for i in range(len(lines)):
        try:
            x[i] = float(lines[i].split(",")[0])
            y[i] = float(lines[i].split(",")[1])
        except:
            x[i] = np.nan
            y[i] = np.nan

    ir_spectra_disp(x, y, cmp=False)
    ir_spectra_disp(x, y, cmp=True)

if __name__ == "__main__":
    main()
