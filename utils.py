import os, sys
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import ROOT as rt
import math
import ctypes

pi = 3.141592653
GFermi = 1.1663787e-11 # MeV^-2
m_electron = 0.5109989507 # MeV
hbar = 6.582119569e-22 # MeV second
s2w = 0.22305 # weak mixing angle
distance_SE = 1.4960e11 # meters
speed_of_light = 299792458.0 # m/s

def RHN_Gamma_vvv(MH, U2):
    return pow(GFermi,2)*pow(MH, 5)*U2*1.0/(192.0*pow(pi, 3)) # MeV

def RHN_Gamma_vll(MH, U2):
    C1 = 0.25*(1-4*s2w+8*s2w**2)
    C2 = 0.5*s2w*(2*s2w-1)
    C3 = 0.25*(1+4*s2w+8*s2w**2)
    C4 = 0.5*s2w*(2*s2w+1)

    xl = m_electron/MH

    L = math.log((1.0 - 3*xl**2 - (1-xl**2)*math.sqrt(1.0-4.0*xl**2))/(xl**2*(1.0+math.sqrt(1.0-4.0*xl**2))))

    hfactor = (C1*(1.0-1.0)+C3*1.0)*( (1.0-14*xl**2-2.0*pow(xl,4)-12.0*pow(xl,6))*math.sqrt(1.0-4.0*xl**2) +12.0*pow(xl, 4)*(pow(xl,4)-1)*L )  + 4*(C2*(1-1)+C4*1)*( xl**2*(2+10*xl**2-12*pow(xl,4))*math.sqrt(1.0-4.0*xl**2) + 6*pow(xl,4)*(1-2*xl**2+2*pow(xl,4))*L )

    gamma = pow(GFermi,2)*pow(MH, 5)*U2*hfactor/(192.0*pow(pi, 3)) # MeV
    return gamma

def RHN_TauCM(MH, U2):
    return hbar/(RHN_Gamma_vll(MH, U2) + RHN_Gamma_vvv(MH, U2)) # second

def RHN_BR_vll(MH, U2):
    gamma_vll = RHN_Gamma_vll(MH, U2)
    gamma_vvv = RHN_Gamma_vvv(MH, U2)

    return gamma_vll/(gamma_vll+gamma_vvv)

# time of flight for RHN to travel from Sun to Earth in c.m.s
def RHN_TauF(MH, EH):
    if MH >= EH:
        return 0.0

    beta = math.sqrt(EH*EH - MH*MH)/EH
    return MH*distance_SE/(EH*beta*speed_of_light)

def diff_lambda(x,y,z):
    return x*x + y*y + z*z - 2*(x*y + y*z + z*x)

# 2D distribution of nuL energy El and costheta in c.m.s frame
def diff_El_costheta_cms(El, costheta, MH, EH):
    if costheta > 1.0 or costheta < -1.0:
        return 0.0

    beta = math.sqrt(EH*EH - MH*MH)/EH # velocity of H in lab frame

    El_l, costheta_l = cms_to_lab(El, costheta, MH, EH)

    El_max = (MH*MH - 4.0*m_electron*m_electron)/(2.0*MH)
    costheta_min = (1.0/beta)*( (MH/El_max) * (1.0-(EH-El_l)/EH) - 1.0)
    if El >= El_max or costheta <= costheta_min:
        return 0

    Eb = El/MH
    mi = m_electron/MH
    mj = m_electron/MH
    mb = 0.0
    Q2 = 1.0-2.0*Eb+mb*mb
    if abs(Q2) < 1e-6:
        return 0

    lambda_ij = diff_lambda(1.0, mi*mi/Q2, mj*mj/Q2)
    if lambda_ij <= 0:
        return 0
    lambda_Qb = diff_lambda(1.0, Q2, mb*mb)
    if lambda_Qb <= 0:
        return 0

    Aij = pow(lambda_ij, 1.5)
    Bij = 2.0*pow(lambda_ij, 0.5)*(1.0+(mi*mi+mj*mj)/Q2-2.0*pow((mi*mi-mj*mj)/Q2, 2.0))
    f1 = math.sqrt(lambda_Qb) * ( 2.0*Q2*(1+mb*mb-Q2)*Aij + (pow(1-mb*mb,2.0)-Q2*Q2)*Bij )
    fs = lambda_Qb*( 2.0*Q2*Aij - (1.0-mb*mb-Q2)*Bij )
    zeta = 1.0

    return f1+zeta*beta*costheta*fs

# lorentz transformation, input El and costheta in c.m.s frame
def cms_to_lab(El, costheta, MH, EH):
    beta = math.sqrt(EH*EH - MH*MH)/EH # velocity of H in lab frame
    gamma_v = 1.0/math.sqrt(1.0-beta*beta) 
    beta_l = 1.0 # neutrino, zero mass
    sintheta = math.sqrt(1.0-costheta*costheta) # theta in cms frame
    vy_l = beta_l*sintheta # vx of nuL in cms frame
    vx_l = beta_l*costheta # vy of nuL in cms frame

    #tantheta_p = vy_l/(gamma_v*(vx_l - beta))
    costheta_p = math.sqrt( pow(gamma_v*(vx_l + beta), 2.0) / (pow(gamma_v*(vx_l + beta), 2.0) + vy_l*vy_l) )
    if vx_l + beta < 0.0:
        costheta_p = -1.0*costheta_p

    El_p = gamma_v*(El + beta*El*beta_l*costheta)

    return El_p, costheta_p

# lorentz transformation, input El and costheta in lab frame
def lab_to_cms(El, costheta, MH, EH):
    beta = math.sqrt(EH*EH - MH*MH)/EH # velocity of H in lab frame
    gamma_v = 1.0/math.sqrt(1.0-beta*beta) 
    beta_l = 1.0 # neutrino, zero mass
    sintheta = math.sqrt(1.0-costheta*costheta) # theta in lab frame
    vy_l = beta_l*sintheta # vx of nuL in lab frame
    vx_l = beta_l*costheta # vy of nuL in lab frame

    #tantheta_p = vy_l/(gamma_v*(vx_l - beta))
    costheta_p = math.sqrt( pow(gamma_v*(vx_l - beta), 2.0) / (pow(gamma_v*(vx_l - beta), 2.0) + vy_l*vy_l) )
    if vx_l - beta < 0.0:
        costheta_p = -1.0*costheta_p

    El_p = gamma_v*(El - beta*El*beta_l*costheta)

    return El_p, costheta_p

# 2D distribution of nuL energy El and costheta in lab frame
def diff_El_costheta_lab(Elp, costhetap, MH, EH):
    if costhetap > 1.0 or costhetap < -1.0:
        return 0.0
    
    El, costheta = lab_to_cms(Elp, costhetap, MH, EH)

    diff_cms = diff_El_costheta_cms(El, costheta, MH, EH)

    delta_El = 1e-6
    delta_costheta = 1e-6
    if costhetap > 1.0 - 1e-4:
        delta_costheta = -1e-6
    
    El_Edelta, costheta_Edelta = lab_to_cms(Elp+delta_El, costhetap, MH, EH)
    El_Tdelta, costheta_Tdelta = lab_to_cms(Elp, costhetap+delta_costheta, MH, EH)
    
    pE_pEp = (El_Edelta-El)/delta_El
    pE_ptp = (El_Tdelta-El)/delta_costheta
    pt_pEp = (costheta_Edelta-costheta)/delta_El
    pt_ptp = (costheta_Tdelta-costheta)/delta_costheta

    Jacob = abs(pE_pEp*pt_ptp - pE_ptp*pt_pEp)
    return diff_cms*Jacob 


def diff_El_costheta_lab_wrong(El, costheta, MH, EH):
    if costheta > 1.0 or costheta < -1.0:
        return 0.0

    El_p, costheta_p = lab_to_cms(El, costheta, MH, EH)

    return diff_El_costheta_cms(El_p, costheta_p, MH, EH)

# 1D distribution of nuL costheta (in lab frame)
def diff_costheta(costheta, MH, EH):
    El_min = 0.0
    El_max = EH - 2.0*m_electron
    nsteps = 500
    El_step = (El_max-El_min)/nsteps

    int_diff = 0.0
    for istep in range(nsteps):
        int_diff += diff_El_costheta_lab(El_min+istep*El_step, costheta, MH, EH)
    return int_diff

# 1D distribution of nuL energy El (in lab frame)
def diff_El(El, MH, EH):
    costheta_min = -1.0
    costheta_max = 1.0
    nsteps = 500
    costheta_step = (costheta_max-costheta_min)/nsteps

    int_diff = 0.0
    for istep in range(nsteps):
        int_diff += diff_El_costheta_lab(El, costheta_max-istep*costheta_step, MH, EH)

    return int_diff

# 1D distribution of epem energy Eee (in lab frame)
def diff_Eee(Eee, MH, EH):
    El = EH - Eee
    return diff_El(El, MH, EH)

# given a spectrum in CSV file, interpolate to get the y value for any given x value, return the new spectrum
def interpolateSpectrum(inputCSV, xvalues):
    df = pd.read_csv(inputCSV)
    interp_func = interp1d(df['energy'].values, df['flux'].values, kind='linear', fill_value="extrapolate")

    spectrum_return = []
    for ix in xvalues:
        iy = interp_func(ix)*1.0
        spectrum_return.append([ix, iy])
    return np.array(spectrum_return)

def integrateSpectrum(spectrum):
    sum_all = 0.0
    for idx in range(len(spectrum)):
        if idx > 0:
            sum_all += spectrum[idx][1]*(spectrum[idx][0]-spectrum[idx-1][0])
        else:
            sum_all += spectrum[idx][1]*(spectrum[idx+1][0]-spectrum[idx][0])
    return sum_all

def getRHNSpectrum(spectrum_L, MH, U2):
    energy = spectrum_L[:,0]
    flux_L = spectrum_L[:,1]
    spectrum_R = []
    for ie in range(len(energy)):
        if energy[ie] <= MH:
            spectrum_R.append([energy[ie], 0.0])
        else:
            spectrum_R.append([energy[ie], flux_L[ie]*U2*math.sqrt(1.0-(MH/energy[ie])**2)])
    return np.array(spectrum_R)
    
def getRHNSpectrums(spectrum_L, grid_U2M):
    nU2 = len(grid_U2M)
    nM = len(grid_U2M[0])

    spectrums_R = []

    for i in range(nU2):
        spectrums_i = []
        for j in range(nM):
            flux_R = []
            U2_this = grid_U2M[i][j][0]
            M_this = grid_U2M[i][j][1]
            spectrums_ij = getRHNSpectrum(spectrum_L, M_this, U2_this)

            spectrums_i.append(spectrums_ij)
        spectrums_R.append(spectrums_i)

    return spectrums_R

# distance: distance from sun to decay point (e.g. earth), unit m
# length: length of detector, unit m
def getDecayedRHNSpectrum(spectrum_orig, MH, U2, distance, length):
    energy = spectrum_orig[:,0]
    flux_orig = spectrum_orig[:,1]

    spectrum_decayed = []
    for ie in range(len(energy)):
        EH = energy[ie]
        if EH <= MH:
            spectrum_decayed.append([EH, 0.0])
        else:
            tau_cm = RHN_TauCM(MH, U2)
            PH = math.sqrt(EH*EH - MH*MH)
            beta = PH/EH
            tau_f = MH*distance/(EH*beta*speed_of_light)
            delta_tau = MH*length/(EH*beta*speed_of_light)
            spectrum_decayed.append([EH, flux_orig[ie]*math.exp(-1.0*tau_f/tau_cm)*(1.0-math.exp(-1.0*delta_tau/tau_cm))])
            #print(energy[ie], PH, beta, delta_tau, tau_f, tau_cm, flux_orig[ie], math.exp(-1.0*tau_f/tau_cm)*(1.0-math.exp(-1.0*delta_tau/tau_cm)), flux_orig[ie]*math.exp(-1.0*tau_f/tau_cm)*(1.0-math.exp(-1.0*delta_tau/tau_cm)))
    return np.array(spectrum_decayed)


# distance: distance from sun to decay point (e.g. earth), unit m
# length: length of detector, unit m
def findRatioForDistance(MH, EH, U2, distance):
    if EH < MH+0.001:
        return 0.0
    tau_cm = RHN_TauCM(MH, U2)
    PH = math.sqrt(EH*EH - MH*MH)
    beta = PH/EH
    tau_f = MH*distance/(EH*beta*speed_of_light)
    return 1.0-math.exp(-1.0*tau_f/tau_cm)


# distance: distance from sun to decay point (e.g. earth), unit m
# length: length of detector, unit m
def findDistanceForRatio(MH, EH, U2, ratio):
    if ratio < 0.0 or ratio >= 1.0:
        return -999.0
    tau_cm = RHN_TauCM(MH, U2)
    PH = math.sqrt(EH*EH - MH*MH)
    beta = PH/EH
    tau_f = -1.0*tau_cm*math.log(1.0-ratio)
    return tau_f*EH*beta*speed_of_light/MH

# distance: distance from sun to decay point (e.g. earth), unit m
# length: length of detector, unit m
def findRatioForDistanceSpectrum(MH, spectrum, U2, distance):
    energy = spectrum[:,0]
    flux = spectrum[:,1]
    flux_decayed = np.zeros(len(flux))
    for ie in range(len(energy)):
        ratio_this = findRatioForDistance(MH, energy[ie], U2, distance)
        flux_decayed[ie] = flux[ie]*ratio_this
    return np.sum(flux_decayed)/np.sum(flux)

# see definition of phi and theta in yutao's thesis fig 3-4
def transform_phi_to_theta(cosphi, distance):
    if cosphi < -1.0 or cosphi > 1.0:
        return -999.0
    if cosphi > 0.0 and distance > distance_SE:
        return -999.0

    sinphi = math.sqrt(1.0 - cosphi*cosphi)
    sintheta = distance*sinphi/distance_SE
    if abs(sintheta-1.0) < 1e-8:
        sintheta = 1.0
    if sintheta > 1.0:
        return -999.0
    costheta = math.sqrt(1.0-sintheta*sintheta)
    if distance >= distance_SE:
        return -1.0*costheta
    else:
        return costheta

def transform_theta_to_phi(costheta, distance):
    if abs(distance) < 1e-6:
        return -999.0
    if costheta < -1.0 or costheta > 1.0:
        return -999.0
    if costheta < 0.0 and distance < distance_SE:
        return -999.0
    sintheta = math.sqrt(1.0 - costheta*costheta)
    sinphi = distance_SE*sintheta/distance
    if abs(sinphi-1.0) < 1e-8:
        sinphi = 1.0
    if sinphi > 1.0:
        return -999.0
    cosphi = math.sqrt(1.0 - sinphi*sinphi)
    if distance >= distance_SE:
        return -1.0*cosphi
    else:
        return cosphi

# distance: distance from sun to decay point, unit m
# length: length of detector (or flight path to integrate), unit m
def getNulEAndAngleFromRHNDecay(spectrum_orig, MH, U2, distance, length, costheta_bins):
    energy = spectrum_orig[:,0]
    flux_orig = spectrum_orig[:,1]
    npoints_costheta = len(costheta_bins)

    costheta_step = costheta_bins[2]-costheta_bins[1]


    diff_El_decayed = np.zeros((len(energy), 2))
    diff_costheta_decayed = np.zeros((npoints_costheta, 2))
    diff_cosphi_decayed = np.zeros((npoints_costheta, 2))

    for iTh in range(npoints_costheta):
        diff_costheta_decayed[iTh][0] = costheta_bins[iTh]
        diff_cosphi_decayed[iTh][0] = costheta_bins[iTh]

    cosphi_needed = np.zeros(npoints_costheta)
    diff_cosphi_needed = np.zeros(npoints_costheta)

    distance_m = distance+0.5*length
    # convert cosphi distribution to costheta distribution
    for iTh in range(npoints_costheta):
        cosphi_needed[iTh] = -999.0
        costheta_this = costheta_bins[iTh]
        cosphi_temp = transform_theta_to_phi(costheta_this, distance_m)
        if cosphi_temp < -2.0:
            continue
        delta_costheta = 1e-6
        if costheta_this > 1.0-1e-4:
            delta_costheta = -1.0*1e-6
        dcosphi_temp = transform_theta_to_phi(costheta_this+delta_costheta, distance_m)
        if dcosphi_temp < -2.0:
            delta_costheta = -1.0*delta_costheta
            dcosphi_temp = transform_theta_to_phi(costheta_this+delta_costheta, distance_m)
        if dcosphi_temp < -2.0:
            continue
        Jacob = abs((dcosphi_temp-cosphi_temp)/delta_costheta)
        diff_costheta_decayed[iTh][1] = Jacob
        cosphi_needed[iTh] = cosphi_temp

    #print("cosphi_needed")
    #print(cosphi_needed)
    # integrate over all RHNs that decay in the give length
    nRHN_decayed_total = 0.0
    for ie in range(len(energy)):
        diff_El_decayed[ie][0] = energy[ie]

        EH = energy[ie]

        eStep = 0.0
        if ie > 0.0:
            eStep = energy[ie]-energy[ie-1]
        else:
            eStep = energy[ie+1]-energy[ie]
            
        if EH <= MH:
            continue
        else:
            tau_cm = RHN_TauCM(MH, U2)
            PH = math.sqrt(EH*EH - MH*MH)
            beta = PH/EH
            tau_f = MH*distance/(EH*beta*speed_of_light)
            delta_tau = MH*length/(EH*beta*speed_of_light)
            nRHN_decayed = flux_orig[ie]*math.exp(-1.0*tau_f/tau_cm)*(1.0-math.exp(-1.0*delta_tau/tau_cm))
            nRHN_decayed_total += nRHN_decayed

            diff_El_temp = np.zeros(len(energy))
            diff_cosphi_temp = np.zeros(npoints_costheta)
            diff_cosphi_needed_temp = np.zeros(npoints_costheta)

            nTotalDecayed = 0.0
            nReachEarth = 0.0

            for icosphi in range(npoints_costheta):
                costheta_temp = transform_phi_to_theta(costheta_bins[icosphi], distance_m)
                for ieL in range(len(energy)):
                    diff_temp = diff_El_costheta_lab(energy[ieL], costheta_bins[icosphi], MH, EH)
                    nTotalDecayed += diff_temp
                    if costheta_temp < -2.0:
                        continue
                    nReachEarth += diff_temp
                    diff_El_temp[ieL] += diff_temp
                    diff_cosphi_temp[icosphi] += diff_temp

            fractionReachEarth = 0.0
            if nTotalDecayed > 0.0:
                fractionReachEarth = nReachEarth/nTotalDecayed

            for icosphi in range(npoints_costheta):
                if cosphi_needed[icosphi] < -2.0:
                    continue
                costheta_temp = transform_phi_to_theta(cosphi_needed[icosphi], distance_m)
                if costheta_temp < -2.0:
                    continue
                for ieL in range(len(energy)):
                    diff_cosphi_needed_temp[icosphi] +=  diff_costheta_decayed[icosphi][1]*diff_El_costheta_lab(energy[ieL], cosphi_needed[icosphi], MH, EH)
                    if distance_m < distance_SE:
                        diff_cosphi_needed_temp[icosphi] +=  diff_costheta_decayed[icosphi][1]*diff_El_costheta_lab(energy[ieL], -1.0*cosphi_needed[icosphi], MH, EH)

            sum_diff_El_temp = np.sum(diff_El_temp)
            if sum_diff_El_temp > 0.0:
                for ieL in range(len(energy)):
                    diff_El_decayed[ieL][1] += nRHN_decayed*fractionReachEarth*diff_El_temp[ieL]/sum_diff_El_temp
                    #diff_El_decayed[ieL][1] += nRHN_decayed*1*diff_El_temp[ieL]/sum_diff_El_temp

            sum_diff_cosphi_temp = np.sum(diff_cosphi_temp)
            if sum_diff_cosphi_temp > 0.0:
                for icosphi in range(npoints_costheta):
                    diff_cosphi_decayed[icosphi][1] += nRHN_decayed*eStep*(1.0/costheta_step)*fractionReachEarth*diff_cosphi_temp[icosphi]/sum_diff_cosphi_temp

            sum_diff_cosphi_needed_temp = np.sum(diff_cosphi_needed_temp)
            if sum_diff_cosphi_needed_temp > 0.0:
                for icosphi in range(npoints_costheta):
                    diff_cosphi_needed[icosphi] += nRHN_decayed*eStep*(1.0/costheta_step)*fractionReachEarth*diff_cosphi_needed_temp[icosphi]/sum_diff_cosphi_needed_temp

    #print("diff_cosphi_needed")
    #print(diff_cosphi_needed)
    #print("Jacob")
    #print(diff_costheta_decayed[:,1])
    for iTh in range(npoints_costheta):
        diff_costheta_decayed[iTh][1] = diff_cosphi_needed[iTh]
    #print("DEBUG : ", nRHN_decayed_total, ",", np.sum(diff_El_decayed[:,1]), ", ", integrateSpectrum(diff_El_decayed))
    return diff_El_decayed, diff_costheta_decayed, diff_cosphi_decayed


def saveSpectrums(spectrums, column_names, fileName, labels):
    for i in range(len(spectrums)):
        if column_names is None:
            np.savetxt(fileName+labels[i]+".csv", spectrums[i], delimiter=',', fmt='%f')
        else:
            np.savetxt(fileName+labels[i]+".csv", spectrums[i], delimiter=',', header=column_names, fmt='%f', comments='')
