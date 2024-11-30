import os, sys
import pandas as pd
import numpy as np
import ROOT as rt
import math

from utils import *
from plotting import *
from sampling import *

if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)

    #energy = np.arange(0.0, 16.0, step=0.01)
    energy = np.arange(0.0, 16.0, step=0.2)
    #energy = np.arange(0.0, 16.0, step=0.5)
    print("=== extract 8B spectrum from csv file ===")
    spectrum_nuL_orig = interpolateSpectrum("data/8BSpectrum.csv", energy) #unit: MeV^-1 cm^-2 s^-1

    #print("=== plot the 8B spectrum ===")
    #plot1DCurve(spectrum_nuL_orig, "^{8}B Solar Neutrino Spectrum", "Neutrino Energy (MeV)", "Neutrino Flux (MeV^{-1} cm^{-2} s^{-1})", "plots/8BSpectrum", 0.1, 16.0, 10.0, 1e6)

    U2 = [1.0, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    mRHN = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    #mRHN = [2.0]

    print("=== doing toy mc for the following U2 and mRHN values ===")
    print("U2:")
    print(U2)
    print("mRHN:")
    print(mRHN)

    grid_U2M = np.zeros((len(U2), len(mRHN), 2))
    for i in range(len(U2)):
        for j in range(len(mRHN)):
            grid_U2M[i][j] = np.array([U2[i], mRHN[j]])

    print("=== calculate RHN spectrum ===")

    spectrums_nuR = getRHNSpectrums(spectrum_nuL_orig, grid_U2M)

    print("=== plot the RHN spectrums ===")
    plotSpectrums_grid1d(spectrums_nuR[0], spectrum_nuL_orig, grid_U2M[0], "Neutrino Energy (MeV)", "Neutrino Flux (MeV^{-1} cm^{-2} s^{-1})", "plots/RHNSpectrum", 0.1, 16.0, 10.0, 1e6)

    #print("=== plot RHN lifetime ===")
    #plotTauCM_vsMU(1e-6, 1e-1, 500, 1.5, 16.5, 500)

    #print("=== plot RHN BR ===")
    #plotBR_vsMU(1e-6, 1e-1, 100, 1.5, 16.5, 100)

    #print("plot 2D distribution of nuL energy and emission angle from nuH decay")
    #for MH in mRHN:
    #    for U2_this in U2:
    #        plotDiff_vs_E_costheta(spectrum_nuL_orig, MH, U2_this)

    #plotDiff_vs_E_costheta(spectrum_nuL_orig, 10.0, 0.1)
    print("=== plot spectrums of nuL from RHN decay ===")
    #plot_nuL_El_grid1d(spectrum_nuL_orig, grid_U2M[0],  "Neutrino #nu_{e} Energy (MeV)", "Neutrino Flux (MeV^{-1} cm^{-2} s^{-1})", "plots/nuLSpectrum", 0.1, 16.0, 10.0, 4.0e6)

    print("=== plot decay angle costheta of nuL from RHN decay ===")
    #plot_nuL_costheta_grid1d(spectrum_nuL_orig, grid_U2M[0],  "#nu_{e} emission angle cos(#theta)", "r.u.", "plots/nuLEmissionAngle", -1.0, 1.0, 1e-2, 40.0)

    #print("=== plot decay angle costheta of nuL from RHN decay by sampling===")
    #plot_nuL_costheta_grid1d_sampling(spectrum_nuL_orig, grid_U2M[0],  "#nu_{e} emission angle cos(#theta)", "r.u.", "plots/nuLEmissionAngleSampling", 100000, -1.0, 1.0, 1e-2, 40.0)

    #print("=== plot energy and decay angle costheta of nuL from RHN decay by sampling===")
    #plot_nuL_El_costheta_grid1d_sampling(spectrum_nuL_orig, grid_U2M[0],  "plots/", 100000)

    print("=== plot electron pair energy distribution from nuR decay inside detector ===")
    volume = 1000.0 # tons
    exposure_time = 1000.0 # days
    detector_size = math.pow(volume, 1.0/3.0) # meters
    #for igrid in range(len(grid_U2M)):
    #    plot_Eee_in_detector_grid1d(spectrum_nuL_orig, grid_U2M[igrid], "E_{e^{+}e^{-}} (MeV)", "Counts / MeV "+str(int(exposure_time)) +" days "+str(int(volume))+" t", "plots/EeeSpectrum_decay_in_detector_integrate", detector_size*detector_size*10000.0, detector_size, exposure_time*24.0*3600.0, 0.1, 16.0, 1e-2, 300.0)

    print("=== plot nuL energy and angle distribution for nuR decay before reaching detector ===")
    for igrid in range(len(grid_U2M)):
        plot_nuL_El_costheta_decay_in_flight_grid1d(spectrum_nuL_orig, grid_U2M[igrid], "plots/")
    #plot_nuL_El_costheta_decay_in_flight_grid1d(spectrum_nuL_orig, grid_U2M[0], "plots/")

    #spectrum_R = getRHNSpectrum(spectrum_nuL_orig, 4.0, 1.0)
    #testSampling2D(spectrum_R, 10000)


