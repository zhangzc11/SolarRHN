import numpy as np
from utils import *
import ROOT as rt
from plotting import *
import time
import matplotlib.pyplot as plt

# Rejection sampling algorithm to generate samples
def rejection_sampling_2Dfunc(f, num_samples, x_bounds, y_bounds, M, *args):
    samples = []
    
    while len(samples) < num_samples:
        # Generate candidate samples uniformly within the bounds
        x = np.random.uniform(x_bounds[0], x_bounds[1])
        y = np.random.uniform(y_bounds[0], y_bounds[1])
        
        # Generate a random number for acceptance check
        u = np.random.uniform(0, M)
        
        # Accept or reject the candidate sample
        if u < f(x, y, *args):
            samples.append([x, y])
    
    return np.array(samples)

# Rejection sampling algorithm to generate samples
def rejection_sampling_1Dfunc(f, num_samples, x_bounds, M, *args):
    samples = []
    
    while len(samples) < num_samples:
        # Generate candidate samples uniformly within the bounds
        x = np.random.uniform(x_bounds[0], x_bounds[1])
        
        # Generate a random number for acceptance check
        u = np.random.uniform(0, M)
        
        # Accept or reject the candidate sample
        if u < f(x, *args):
            samples.append(x)
    
    return np.array(samples)

# input spectrum is a 2D array of distributions
def generate_samples_from_spectrum(spectrum, num_samples):
    x_value = spectrum[:, 0]
    y_value = spectrum[:, 1]

    # normalize the spectrum
    samples = np.zeros(num_samples)
    spectrum_generated = np.zeros((len(x_value), 2))
    for i in range(len(x_value)):
        spectrum_generated[i][0] = x_value[i]

    if np.sum(y_value) < 1e-6:
        return samples, spectrum_generated
        
    y_value = y_value / np.sum(y_value)
    indices = np.arange(len(x_value))
    samples_index = np.random.choice(indices, size=num_samples, p=y_value)

    for i in range(num_samples):
        samples[i] = x_value[samples_index[i]]
        spectrum_generated[samples_index[i]][1] += 1.0
    return samples, spectrum_generated

def getMaximumValue2D(f, x_bounds, y_bounds, *args):
    values = []
    num_points = 100
    for ix in range(num_points):
        x_this = x_bounds[0] + ix*(x_bounds[1]-x_bounds[0])/num_points
        for iy in range(num_points):
            y_this = y_bounds[0] + iy*(y_bounds[1]-y_bounds[0])/num_points
            value = f(x_this, y_this, *args)
            values.append(value)
    return max(values)


def getNuLEAndAngleBySampling(spectrum_R, MH, num_samples, costheta_bins):
    energy_bins = spectrum_R[:, 0]
    flux_R = spectrum_R[:, 1]

    energy_bin_edges = np.zeros(len(energy_bins)+1)
    step_i = 0.0
    for i in range(len(energy_bins)):
        if i > 0:
            step_i = energy_bins[i] - energy_bins[i-1]
        else:
            step_i = energy_bins[i+1] - energy_bins[i]
        energy_bin_edges[i] = energy_bins[i] - 0.999*step_i
    energy_bin_edges[-1] = energy_bins[-1] + 0.001*step_i

    costheta_bin_edges = np.zeros(len(costheta_bins)+1)
    step_i = 0.0
    for i in range(len(costheta_bins)):
        if i > 0:
            step_i = costheta_bins[i] - costheta_bins[i-1]
        else:
            step_i = costheta_bins[i+1] - costheta_bins[i]
        costheta_bin_edges[i] = costheta_bins[i] - 0.999*step_i
    costheta_bin_edges[-1] = costheta_bins[-1] + 0.001*step_i
    
    maxDiff = 0.0
    for EH in energy_bins:
        if EH > MH:
            maxDiff_this = getMaximumValue2D(diff_El_costheta_cms, [energy_bins[0], energy_bins[-1]], [costheta_bins[-1], costheta_bins[0]], MH, EH)
            if maxDiff_this > maxDiff:
                maxDiff = maxDiff_this

    samples_generated, spectrum_generated = generate_samples_from_spectrum(spectrum_R, num_samples)
    Els = []
    Eees = []
    costhetas = []

    for EH in samples_generated:
        sample_this = rejection_sampling_2Dfunc(diff_El_costheta_cms, 1, [energy_bins[0], energy_bins[-1]], [costheta_bins[0], costheta_bins[-1]], maxDiff*2.0, MH, EH)
        El_this = sample_this[0][0]
        costheta_this = sample_this[0][1]
        El_this_lab, costheta_this_lab = cms_to_lab(El_this, costheta_this, MH, EH)
        Els.append(El_this_lab)
        Eees.append(EH - El_this_lab)
        costhetas.append(costheta_this_lab)

    diff_El = np.zeros((len(energy_bins), 2))
    diff_Eee = np.zeros((len(energy_bins), 2))
    diff_costheta = np.zeros((len(costheta_bins), 2))

    Els_count, Els_edges = np.histogram(Els, bins=energy_bin_edges)
    Eees_count, Eees_edges = np.histogram(Eees, bins=energy_bin_edges)
    costhetas_count, costhetas_edges = np.histogram(costhetas, bins=costheta_bin_edges)
    sum_flux_R = np.sum(flux_R)
    sum_Els_count = np.sum(Els_count)
    sum_Eees_count = np.sum(Eees_count)
    sum_costhetas_count = np.sum(costhetas_count)

    for ieL in range(len(Els_count)):
        diff_El[ieL][0] = energy_bins[ieL]
        diff_El[ieL][1] = Els_count[ieL]*1.0*sum_flux_R/sum_Els_count

        diff_Eee[ieL][0] = energy_bins[ieL]
        diff_Eee[ieL][1] = Eees_count[ieL]*1.0*sum_flux_R/sum_Eees_count
    for iTheta in range(len(costhetas_count)):
        diff_costheta[iTheta][0] = costheta_bins[iTheta]
        diff_costheta[iTheta][1] = costhetas_count[iTheta]*1.0*sum_flux_R/sum_costhetas_count
    return diff_El, diff_costheta, diff_Eee


