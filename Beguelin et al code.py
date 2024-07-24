#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 15:30:23 2024

@author: paulbeguelin
"""

# To use this code:

# Monte Carlo simulations are ran using the method run from the class mc (see lines 652 and below).
# The random seeds yielding the 50 valid simulations discussed in the article are listed in the list valid_seeds (line 558).
# The results presented in Supplementary Table 2 are the median values of these 50 simulations.
# The code takes a couple of hours to run on a laptop.


# Imports libraries
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import numpy.random as rd
import statsmodels.nonparametric.smoothers_lowess as slw
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Imports model constants
cons = pd.read_csv('Constants.csv')

# Import MORB, Gabbro and sediments libraries
morb_lit = pd.read_csv('MORB.csv')
gabbro_lit = pd.read_csv('Gabbros.csv')
sed_lit = pd.read_csv('Sediments.csv')

# Dictionary of partition coefficients
Kds = {'Sp_per': {'Rb': 0.0003, 'Sr': 0.014, 'La': 0.013, 'Ce': 0.021, 'Nd': 0.046, 'Sm': 0.075, 'Hf': 0.073, 'Lu': 0.186},
       'Sp_per_parent': {'Sr': 0.0003, 'Ce': 0.013, 'Nd': 0.075, 'Hf': 0.186},
       'Gt_per': {'Sr': 0.031, 'Ce': 0.013, 'Nd': 0.03, 'Hf': 0.061},
       'Ecl': {'Sr': 0.058, 'Ce': 0.047, 'Nd': 0.152, 'Hf': 0.269}}

# List of isotopes to work with
isotopes = ['Sr', 'Ce', 'Nd', 'Hf']
concentrations = ['CSr', 'CCe', 'CNd', 'CHf']
concDict = {'Sr': 'CSr', 'Ce': 'CCe', 'Nd': 'CNd', 'Hf': 'CHf'}
isotopes2 = ['Sr', 'Nd', 'Hf']

# Imports the data to model
data = pd.read_csv('Ce_dataset.csv')
data2 = pd.read_csv('Literature_data.csv')

# Normalization parameters for observables
norm = {'Sr': 0.001437, 'Ce': 1.449, 'Nd': 10.611942337, 'Hf': 13.190232862, 'CNd': 8}

# Functions to extract data from dicts
# Transforms a dictionary of isotope values into a numpy array with the values, in the order of the list isotopes
def iso(x, isotopes=isotopes):
    return np.asarray([x.get(i) for i in isotopes])

# Transforms a dictionary of concentration values into a numpy array with the values, in the order of the list isotopes
def con(x, isotopes=isotopes):
    concDict = {'Sr': 'CSr', 'Ce': 'CCe', 'Nd': 'CNd', 'Hf': 'CHf'}
    conc = [concDict.get(i) for i in isotopes]
    return np.asarray([x.get(i) for i in conc])

## Melting and depletion functions
def Cr(F, D, C0): #Solid restite: F = melting degree, D = Partition coefficients, C0 = concentrations before the depletion
    return (1 - F) ** (1/D - 1) * C0

def Cl(F, D, C0): #Melt: F = melting degree, D = Partition coefficients, C0 = source concentrations before the melting
    if F <= 0 or F > 0.99:
        return C0
    else:
        return (1/F) * (1 - (1 - F) ** (1/D)) * C0

# Function to calculate depleted peridotite isotope composition
def DM_R(Fd, Ends, t, isotopes):
    # Returns the isotope composition and concentrations in the solid source for a DM with a time-integrated degree of melt-depletion Fd.
    P_D_DM = iso({'Sr': 0.028772457, 'Ce': 0.154864791, 'Nd': 0.2330047, 'Hf': 0.045405263}, isotopes)
    P_D_DM2 = Cr(Fd, iso(Kds['Sp_per_parent'], isotopes), P_D_DM) / Cr(Fd, iso(Kds['Sp_per'], isotopes), 1)
    branched = iso(dict(zip(cons.Iso, cons.branched)), isotopes)
    decay = iso(dict(zip(cons.Iso, cons.decay)), isotopes)
    Eps_dif = iso(dict(zip(cons.Iso, cons.Eps_dif)), isotopes)
    Eps_mul = iso(dict(zip(cons.Iso, cons.Eps_mul)), isotopes)
    Eps_div = iso(dict(zip(cons.Iso, cons.Eps_div)), isotopes)
    R_DM = iso(Ends['DM'], isotopes) * Eps_div / Eps_mul + Eps_dif
    R = (R_DM + (P_D_DM2 - P_D_DM) * branched * np.expm1(decay * t) - Eps_dif) * Eps_mul / Eps_div
    return R

# Function to calculate depleted peridotite trace element concentrations
def DM_C(Fd, Ends, isotopes): #Just recalls the function Cr, setting up the inputs properly
    C = Cr(Fd, iso(Kds['Sp_per'], isotopes), con(Ends['DM'], isotopes))
    return C

# Isotope composition and trace element concentrations of rc (recycled oceanic crust)
class rc:
    def __init__(self, fSED, F_RC, ends, isos): # fSED is the proportion of sediments in the rc: fSED = fSED * X_RC
        self.isos = isos
        self.ends = ends
        self.F = F_RC
        self.OC = ends['OC'] # OC: recycled mafic crust (gabbros + basalts)
        self.SED = ends['SED']
        self.C0 = fSED * con(self.SED, isos) + (1 - fSED) * con(self.OC, isos)
        self.R = (fSED * iso(self.SED, isos) * con(self.SED, isos) + (1 - fSED) * iso(self.OC, isos) * con(self.OC, isos)) / self.C0
        self.C = Cl(self.F, iso(Kds['Ecl'], isos), self.C0)

# Calculates the solidus T of a peridotite with degree of depletion Fd at a pressure P
def solidus(P, Fd):
    # Uses the parametrization from Hirschmann (2000) and Robinson and Wood (1998)
    # P is in [GPa]
    # Fd is the degree of time integrated melt depletion
    a = -5.104
    b = 123.899 - 10
    c = 1120.661
    return a * P ** 2 + b * P + c + 700 * Fd

# Calculates the total (accumulated) F at each P following figure 7d from Asimow et al. (1997).
def gamma(dP): # Melting parametrization based on the productivity model of Asimow et al. (1997) (Figure 7b, 7d)
    # As an approximation, the shape of the dF/dP curve remains the same but simply shifts along the P axis with different Tp values
    # WARNING: Not valid for dP > 1.8
    a = -15.809245
    b = 77.116899
    c = -133.435262
    d = 102.613742
    e = -34.154555
    f = 5.594125
    g = 0
    return (a * dP ** 6 + b * dP ** 5 + c * dP ** 4 + d * dP ** 3 + e * dP ** 2 + f * dP + g) / 100

# Calculates the relative upwelling velocity of a parcel of melt at each pressure increment
def U(dP):
    # U(P) parameter of Ito and Mahoney (2005), equation 3
    # H = 100 [Km], Lithosphere thickness = 100 km 
    # WARNING: Does not work for P > 6.27 [GPa]
    P = dP + 3
    return -0.0935 * P ** 2 + 1.1726 * P - 2.6762

# Calculates the total degree of peridotite melting, integrated over the whole pressure interval
def Fper_fn(P0, num_out=True):
    if P0 <= 0:
        return 0
    def num(dP):
        return gamma(dP) ** 2 * U(dP)
    def den(dP):
        return gamma(dP) * U(dP)
    num_i = quad(num, 0, P0)
    den_i = quad(den, 0, P0)
    Fper = num_i[0] / den_i[0]
    if num_out:
        return Fper
    else:
        return (Fper, num_i[1], den_i[1])

# Calculates the latent heat of rc melting
def dT_RC(X_RC, F_RC):
    # Values from Ballmer et al. (2013)
    M_melt = X_RC * F_RC
    c_p = 1250 # [J Kg-1 K-1] Specific heat capacity
    L = 5.6e5 # [J/Kg] Latent heat of melt
    heat = M_melt * L # [J] Latent heat per Kg of plume mantle melting
    return heat/c_p # [K] Difference in temperature after melting

# Peridotite class with all its attibutes
# dP_r is a ratio between 0 and 1 controlling what ratio of the maximum P0 allowed ends up being P0 (P0 max allowed based on cpx exhaustion)
class per:
    def __init__(self, Fd, dP_r, t, ends, isos):
        self.isos = isos
        self.ends = ends
        self.t = t
        self.Fd = Fd
        self.mFd = 0.17 - self.Fd
        self.P0_max = dP_r * (-5.1961 *  self.Fd ** 2 - 8.3824 *  self.Fd + 1.783) # T solidus used as the limiting factor (Tp of the plume doesn't increase with Fd)
        self.F_per_max = gamma(self.P0_max)
        self.P0 = 3 + dP_r * self.P0_max
        self.T0 = solidus(self.P0, Fd) # Temperature [C] before peridotite melting
        self.F_per = Fper_fn(self.P0 - 3)
        self.F = self.F_per
        self.C0 = DM_C(self.Fd, self.ends, self.isos)
        self.R = DM_R(self.Fd, self.ends, self.t, self.isos)
        if self.F_per <= 0:
            self.C = self.C0
        else:
            self.C = Cl(self.F_per, iso(Kds['Gt_per'], self.isos), self.C0)

# Melt mixture class with all its attributes. This class also contains the attributes of the associated mantle source, de-facto serving at a "modeled sample" class.
class mix:
    def __init__(self, Fd, dP_r, X_RC, fSED, t, F_RC, ends, isos):
        self.Fd = Fd
        self.dP_r = dP_r
        self.X_RC = X_RC
        self.fSED = fSED
        self.X_SED = self.X_RC * self.fSED
        self.X_OC = self.X_RC - self.X_SED
        self.t = t
        self.isos = isos
        self.ends = ends
        
        self.per = per(Fd, dP_r, t, ends, isos)
        self.rc = rc(fSED, F_RC, ends, isos)
        self.x = (X_RC * self.rc.F) / (X_RC * self.rc.F + (1 - X_RC) * self.per.F)
        self.C = self.x * self.rc.C + (1 - self.x) * self.per.C
        self.R = (self.x * self.rc.C * self.rc.R + (1 - self.x) * self.per.C * self.per.R) / self.C
        self.dT_RC = dT_RC(X_RC, self.rc.F)
        self.Tp = self.per.T0 + self.dT_RC
        self.P0 = self.per.P0
        
        self.F_per = self.per.F
        self.F_per_max = self.per.F_per_max
        self.F_RC = self.rc.F
        self.F_tot = X_RC * self.F_RC + (1 - X_RC) * self.per.F

# Calculates how close a model composition is to that of a sample: (model - sample)^2 for each modeled isotope ratio and the Nd concentration condition.
def mixNM(x, sample, t, F_RC, ends, isos):
    x_Fd = x[0]
    x_dP_r = x[1]
    x_X_RC = x[2]
    x_fSED = x[3]
    m = mix(x_Fd, x_dP_r, x_X_RC, x_fSED, t, F_RC, ends, isos)
    isoN = (m.R - sample) / iso(norm, isos)
    iCNd = isos.index('Nd')
    if abs(m.C[iCNd]) > 100:
        cN = 3 ** 100 + (m.C[iCNd] - 100) ** 2
    else:
        cN = 3 ** (abs(m.C[iCNd] - 16) - 10.5)
    N = np.zeros(len(isos)+1)
    for i in range(len(isos)):
        N[i] = isoN[i]
    N[len(isos)] = cN
    return sum(N**2)

# These three lines of code put the model results into a dictionary (param_l is a list of keys)
def P(p):
    return {'Fd': p[0], 'dP_r': p[1], 'X_RC': p[2], 'fSED': p[3]}

param_l = ['Fd', 'dP_r', 'X_RC', 'fSED']


# Default values for Fd, dP_r, X_RC, fSED, to be used as initial guess in the upcoming minimize function.
x0 = [0.05, 0.85, 0.06, 0.1]

# Bounds of the minimize function
bnds = ((0, 0.15), (0.001, 1.0), (0, 0.5), (0, 0.5))

# Bounds for endmember compositions. Only used for DM.
e_bnds = {'DM': {'min': [0.702, -1.65, 10.5, 18], 'max': [0.7027, -1.2, 15, 25]}}

# Aging function. Calculates the isotope composition of a mantle source after aging.
def growth(t, DM, P_Ds, initial='DM', t_res=2.5E9): # t is the reycling age in years (not Ga), DM is a dict with Sr, Ce, Nd, Hf keys, and P_Ds is a 1-row dataframe with concentrations for individual trace elements. Initial: 'DM', 'BSE', or 'CC'.
    isotopes = ['Sr', 'Ce', 'Nd', 'Hf']
    branched = iso(dict(zip(cons.Iso, cons.branched)), isotopes)
    decay = iso(dict(zip(cons.Iso, cons.decay)), isotopes)
    Eps_dif = iso(dict(zip(cons.Iso, cons.Eps_dif)), isotopes)
    Eps_mul = iso(dict(zip(cons.Iso, cons.Eps_mul)), isotopes)
    Eps_div = iso(dict(zip(cons.Iso, cons.Eps_div)), isotopes)
    R_DM = iso(DM, isotopes) * Eps_div / Eps_mul + Eps_dif
    P_D_DM = iso({'Sr': 0.028772457, 'Ce': 0.154864791, 'Nd': 0.2330047, 'Hf': 0.045405263}, isotopes)
    P_D_CC = iso({'Sr': 0.516580100, 'Ce': 0.22299, 'Nd': 0.1179, 'Hf': 0.01267}, isotopes)
    R_CC = cons.BSE_iso[:4] + (P_D_CC - cons.BSE_P_D[:4]) * branched * np.expm1(decay * t_res)
    P_D_GLOSS = iso({'Sr': 0.506339, 'Ce': 0.240971, 'Nd': 0.129429, 'Hf': 0.014445}, isotopes)
    R_GLOSS = cons.BSE_iso[:4] + (P_D_GLOSS - cons.BSE_P_D[:4]) * branched * np.expm1(decay * t_res)
    P_D = np.array([P_Ds.Rb/P_Ds.Sr, P_Ds.La/P_Ds.Ce, P_Ds.Sm/P_Ds.Nd, P_Ds.Lu/P_Ds.Hf]) * cons.C_C_to_P_D[:4]
    if initial == 'BSE':
        R = (cons.BSE_iso[:4] + (P_D - cons.BSE_P_D[:4]) * branched * np.expm1(decay * t) - Eps_dif) * Eps_mul / Eps_div
    if initial == 'DM':
        R = (R_DM + (P_D - P_D_DM) * branched * np.expm1(decay * t) - Eps_dif) * Eps_mul / Eps_div
    if initial == 'CC':
        R = (R_CC + (P_D - P_D_CC) * branched * np.expm1(decay * t) - Eps_dif) * Eps_mul / Eps_div
    if initial == 'GLOSS':
        R = (R_GLOSS + (P_D - P_D_GLOSS) * branched * np.expm1(decay * t) - Eps_dif) * Eps_mul / Eps_div
    return {'Sr': R[0],
     'Ce': R[1],
     'Nd': R[2],
     'Hf': R[3],
     'CSr': P_Ds.Sr,
     'CCe': P_Ds.Ce,
     'CNd': P_Ds.Nd,
     'CHf': P_Ds.Hf}

# Solid mixing function. Takes two standard endmember dictionaries and mixing ratio. Outputs the dict of the solid mixture.
def smix(end1, end2, x2, isos=isotopes):
    C = x2 * con(end2, isos) + (1 - x2) * con(end1, isos)
    R = (x2 * iso(end2, isos) * con(end2, isos) + (1 - x2) * iso(end1, isos) * con(end1, isos)) / C
    return {'Sr': R[0],
     'Ce': R[1],
     'Nd': R[2],
     'Hf': R[3],
     'CSr': C[0],
     'CCe': C[1],
     'CNd': C[2],
     'CHf': C[3]}

# Endmembers class. This class contains all the parameters of a given endmember combination, plus recycling age and F_RC (all Monte Carlo parameters).
class endm:
    def __init__(self, seed=0, bounds=e_bnds, sed_init = 'CC', GLOSS=False, run=True):
        self.seed = seed
        self.e_bnds = e_bnds
        rd.seed(self.seed)
        se = rd.uniform(0.0, 1.0, size = 8)
        self.t = ((2 - 1) * se[0] + 1) * 1E9 # t is the rc recycling age.
        self.t_res  = (3E9 - self.t) * se[1] + self.t
        self.F_RC = (0.8 - 0.5) * se[2] + 0.5
        self.t_Fd = self.t
        self.mor = morb_lit.iloc[int(se[3] * len(morb_lit))].copy()
        self.gab = gabbro_lit.iloc[int(se[4] * len(gabbro_lit))].copy()
        if GLOSS:
            self.sed = sed_lit.iloc[0].copy()
        else:
            self.sed = sed_lit.iloc[int(se[5] * len(sed_lit))].copy()
        self.xgab = se[6] * 0.8 + 0.1 # Between 10 and 90 % gabbro
        self.DM = {'Sr': self.e_bnds['DM']['min'][0] + (self.e_bnds['DM']['max'][0] - self.e_bnds['DM']['min'][0]) * (1-se[7]),
         'Ce': self.e_bnds['DM']['min'][1] + (self.e_bnds['DM']['max'][1] - self.e_bnds['DM']['min'][1]) * (1-se[7]),
         'Nd': self.e_bnds['DM']['min'][2] + (self.e_bnds['DM']['max'][2] - self.e_bnds['DM']['min'][2]) * se[7],
         'Hf': self.e_bnds['DM']['min'][3] + (self.e_bnds['DM']['max'][3] - self.e_bnds['DM']['min'][3]) * se[7],
         'CSr': 9.8,
         'CCe': 0.772,
         'CNd': 0.713,
         'CHf': 0.199}
        self.MORB = growth(self.t, self.DM, self.mor)
        self.GABBRO = growth(self.t, self.DM, self.gab)
        self.SED = growth(self.t, self.DM, self.sed, initial=sed_init, t_res=self.t_res)
        
        if run:
            self.run()
        
    def run(self): # This method calculates all endmembers parameters
        self.OC = smix(self.MORB, self.GABBRO, self.xgab)
        self.dic = {'OC': self.OC,
         'SED': self.SED,
         'DM': self.DM}
        i_cls = ['Sr', 'Ce', 'Nd', 'Hf', 'CSr', 'CCe', 'CNd', 'CHf']
        self.df3 = pd.DataFrame(data=None,index=['OC', 'SED', 'DM Fd_0'], columns=i_cls)
        for i in i_cls:
            self.df3[i]['OC'] = self.dic['OC'][i]
            self.df3[i]['SED'] = self.dic['SED'][i]
            self.df3[i]['DM Fd_0'] = self.dic['DM'][i]
        self.df1 = pd.DataFrame(data=None,index=[self.seed], columns=
                            ['OC_Sr', 'OC_Ce', 'OC_Nd', 'OC_Hf', 'OC_CSr', 'OC_CCe', 'OC_CNd', 'OC_CHf',
                             'SED_Sr', 'SED_Ce', 'SED_Nd', 'SED_Hf', 'SED_CSr', 'SED_CCe', 'SED_CNd', 'SED_CHf',
                             'DM_Sr', 'DM_Ce', 'DM_Nd', 'DM_Hf', 'DM_CSr', 'DM_CCe', 'DM_CNd', 'DM_CHf'])
    
        i_cols0 = ['OC_Sr', 'OC_Ce', 'OC_Nd', 'OC_Hf', 'OC_CSr', 'OC_CCe', 'OC_CNd', 'OC_CHf']
        i_cols1 = ['SED_Sr', 'SED_Ce', 'SED_Nd', 'SED_Hf', 'SED_CSr', 'SED_CCe', 'SED_CNd', 'SED_CHf']
        i_cols2 = ['DM_Sr', 'DM_Ce', 'DM_Nd', 'DM_Hf', 'DM_CSr', 'DM_CCe', 'DM_CNd', 'DM_CHf']
        for j in i_cols0:
            # self.df1[j].iloc[0] = self.dic['OC'][i_cls[i_cols0.index(j)]]
            self.df1.at[self.df1.index[0], j] = self.dic['OC'][i_cls[i_cols0.index(j)]]
        for j in i_cols1:
            # self.df1[j].iloc[0] = self.dic['SED'][i_cls[i_cols1.index(j)]]
            self.df1.at[self.df1.index[0], j] = self.dic['SED'][i_cls[i_cols1.index(j)]]
        for j in i_cols2:
            # self.df1[j].iloc[0] = self.dic['DM'][i_cls[i_cols2.index(j)]]
            self.df1.at[self.df1.index[0], j] = self.dic['DM'][i_cls[i_cols2.index(j)]]
            

# This is the central class of the model. This class takes an endmember class as an input (that is, a set of endmembers), the measured data (data0: Ce dataset, data2: literature dataset), and the model bounds (bnds).
# It contains methods to run the model and store data
class model:
    def __init__(self, ends=endm(), data0=data, data2=data2, x0=x0, bnds=bnds):
        self.s0 = data # s stands for samples
        self.s1 = data
        self.s2 = data2
        self.x0 = x0
        self.bnds = bnds
        self.en = ends
        self.t = ends.t
        self.F_RC = ends.F_RC
        self.ends = ends.dic
        self.ends_cls = ends
        self.isos = isotopes
        self.isos2 = isotopes2
        
        self.m0 = []
        self.m1 = []
        self.m2 = []
        self.r0 = self.s0.copy() # r stands for results
        self.r1 = self.s1.copy()
        self.r2 = self.s2.copy()
    
    # This method creates a table with the Ce–Nd–Hf–Sr model results for the Ce isotope dataset
    def table0(self):
        conc = []
        for i in self.isos:
            conc.append(concDict[i])
        c_R = [i + '_m' for i in self.isos]
        c_C = [i + '_m' for i in conc]
        c_Rper = [i + '_per' for i in self.isos]
        c_Cper = [i + '_per' for i in conc]
        c_Rrc = [i + '_RC' for i in self.isos]
        c_Crc = [i + '_RC' for i in conc]
        columns = ['res', 'X_RC', 'fSED', 'X_OC', 'X_SED', 'F_RC', 'F_per', 'F_tot', 'Fd', 'dP_r', 'P0', 'Tp', 'dT_RC']
        for i in c_R + c_C + c_Rper + c_Cper + c_Rrc + c_Crc + columns:
            self.r0[i] = 0
        for i in range(len(self.r0)):
            for j in range(len(self.isos)):
                self.r0.at[self.r0.index[i], c_R[j]] = self.m0[i].R[j]
                self.r0.at[self.r0.index[i], c_C[j]] = self.m0[i].C[j]
                self.r0.at[self.r0.index[i], c_Rper[j]] = self.m0[i].per.R[j]
                self.r0.at[self.r0.index[i], c_Cper[j]] = self.m0[i].per.C[j]
                self.r0.at[self.r0.index[i], c_Rrc[j]] = self.m0[i].rc.R[j]
                self.r0.at[self.r0.index[i], c_Crc[j]] = self.m0[i].rc.C[j]
            for k in range(len(columns)):
                self.r0.at[self.r0.index[i], columns[k]] = getattr(self.m0[i], columns[k])
        self.res0 = self.r0.res.sum()
        if self.res0 > 100:
            self.res0 = 100
        return self.r0
    
    # This method creates a table with the Nd–Hf–Sr model results for the Ce isotope dataset
    def table1(self):
        conc = []
        for i in self.isos2:
            conc.append(concDict[i])
        c_R = [i + '_m' for i in self.isos2]
        c_C = [i + '_m' for i in conc]
        c_Rper = [i + '_per' for i in self.isos2]
        c_Cper = [i + '_per' for i in conc]
        c_Rrc = [i + '_RC' for i in self.isos2]
        c_Crc = [i + '_RC' for i in conc]
        columns = ['res', 'X_RC', 'fSED', 'X_OC', 'X_SED', 'F_RC', 'F_per', 'F_tot', 'Fd', 'dP_r', 'P0', 'Tp', 'dT_RC']
        for i in c_R + c_C + c_Rper + c_Cper + c_Rrc + c_Crc + columns:
            self.r1[i] = 0
        for i in range(len(self.r1)):
            for j in range(len(self.isos2)):
                self.r1.at[self.r1.index[i], c_R[j]] = self.m1[i].R[j]
                self.r1.at[self.r1.index[i], c_C[j]] = self.m1[i].C[j]
                self.r1.at[self.r1.index[i], c_Rper[j]] = self.m1[i].per.R[j]
                self.r1.at[self.r1.index[i], c_Cper[j]] = self.m1[i].per.C[j]
                self.r1.at[self.r1.index[i], c_Rrc[j]] = self.m1[i].rc.R[j]
                self.r1.at[self.r1.index[i], c_Crc[j]] = self.m1[i].rc.C[j]
            for k in range(len(columns)):
                self.r1.at[self.r1.index[i], columns[k]] = getattr(self.m1[i], columns[k])
        return self.r1
    
    # This method creates a table with the Nd–Hf–Sr model results for the literature dataset (no Ce)
    def table2(self):
        conc = []
        for i in self.isos2:
            conc.append(concDict[i])
        c_R = [i + '_m' for i in self.isos2]
        c_C = [i + '_m' for i in conc]
        c_Rper = [i + '_per' for i in self.isos2]
        c_Cper = [i + '_per' for i in conc]
        c_Rrc = [i + '_RC' for i in self.isos2]
        c_Crc = [i + '_RC' for i in conc]
        columns = ['res', 'X_RC', 'fSED', 'X_OC', 'X_SED', 'F_RC', 'F_per', 'F_tot', 'Fd', 'dP_r', 'P0', 'Tp', 'dT_RC']
        for i in c_R + c_C + c_Rper + c_Cper + c_Rrc + c_Crc + columns:
            self.r2[i] = 0
        for i in range(len(self.r2)):
            for j in range(len(self.isos2)):
                self.r2.at[self.r2.index[i], c_R[j]] = self.m2[i].R[j]
                self.r2.at[self.r2.index[i], c_C[j]] = self.m2[i].C[j]
                self.r2.at[self.r2.index[i], c_Rper[j]] = self.m2[i].per.R[j]
                self.r2.at[self.r2.index[i], c_Cper[j]] = self.m2[i].per.C[j]
                self.r2.at[self.r2.index[i], c_Rrc[j]] = self.m2[i].rc.R[j]
                self.r2.at[self.r2.index[i], c_Crc[j]] = self.m2[i].rc.C[j]
            for k in range(len(columns)):
                self.r2.at[self.r2.index[i], columns[k]] = getattr(self.m2[i], columns[k])
        return self.r2
    
    # This method runs the model (minimizing the mixNM function). The model argument (0, 1 or 2) refers to the model type (see descriptions of table0, table1 and table2 methods)
    def run(self, model, print_i=True):
        if model == 0:
            s = self.s0
            isos = isotopes
        if model == 1:
            s = self.s1
            isos = isotopes2
        if model == 2:
            s = self.s2
            isos = isotopes2
        for i in range(len(s)):
            sampl = iso(s.iloc[i].copy(), isos)
            res = minimize(mixNM, self.x0, args = (sampl, self.t, self.F_RC, self.ends, isos), method = 'Powell', bounds = self.bnds)
            MIX = mix(res.x[0], res.x[1], res.x[2], res.x[3], self.t, self.F_RC, self.ends, isos)
            MIX.s = s.iloc[i].copy()
            MIX.res = res.fun
            if MIX.res > 100:
                MIX.res = 100
            if model == 0: self.m0.append(MIX)
            if model == 1: self.m1.append(MIX)
            if model == 2: self.m2.append(MIX)
            if print_i: print(i)
        if model == 0: self.table0()
        if model == 1: self.table1()
        if model == 2: self.table2()
        
    # This method runs the 3 models together
    def runall(self):
        self.run(0)
        self.run(1)
        self.run(2)
    
    # This method quantifies the difference in Fd, dP_r, X_RC and fSED between the results of model 0 and model 1.
    # This is done to select sets of endmembers where the Sr–Nd–Hf results are consistent with the Ce–Sr–Nd–Hf results for the samples with Ce istope data, thus calibrating the model for interpreting the literature data without the Ce.
    def valid(self):
        self.run(0, print_i=False)
        self.run(1, print_i=False)
        self.R2_Fd = sum((self.r1.Fd - self.r0.Fd) ** 2) / sum((self.r1.Fd - self.r1.Fd.mean()) ** 2)
        self.R2_X_RC = sum((self.r1.X_RC - self.r0.X_RC) ** 2) / sum((self.r1.X_RC - self.r1.X_RC.mean()) ** 2)
        self.R2_fSED = sum((self.r1.fSED - self.r0.fSED) ** 2) / sum((self.r1.fSED - self.r1.fSED.mean()) ** 2)
        self.R2_dP_r = sum((self.r1.dP_r - self.r0.dP_r) ** 2) / sum((self.r1.dP_r - self.r1.dP_r.mean()) ** 2)
        self.val = (self.R2_Fd ** 2 + self.R2_X_RC ** 2 + self.R2_fSED ** 2 + self.R2_dP_r **2) ** 0.5
        if self.val > 100:
            self.val = 100

# This class contains the Monte Carlo model, testing seeds (endmember combinations + t, F_RC) where model 0 and 1 are consistent and samples can be modeled from input values within the bounds of the model (bnds) and running the successful seeds for literature samples (model 2).
class mc:
    def __init__(self, e_bnds=e_bnds, data0=data, data2=data2, x0=x0, bnds=bnds):
        self.e_bnds = e_bnds
        self.data0 = data0
        self.data2 = data2
        self.x0 = x0
        self.bnds = bnds
        self.r = {}
        self.good_seeds = []
        
    # This method runs the Monte Carlo model. To look through a range of seeds, use the default of_list=False and seed_min, seed_max as the bounds of the range.
    # seed_max is inclusive.
    # To run a list of pre-chosen seeds, use of_list=True and a list of seeds for the seeds argument.
    # To run models 0 and 1 (the Ce isotope dataset with and without Ce), set run2 as False. If run2=True, then the whole literature dataset (100s of samples) will run.
    def run(self, seed_min=0, seed_max=10, of_list=False, seeds=None, run2=True, use_lims=False, lims=[0.25,2.5]):
        
        n = 0
        
        if of_list:
            for i in seeds:
                m = model(endm(seed=i, bounds = self.e_bnds), data0=self.data0, data2=self.data2, x0=self.x0, bnds=self.bnds)
                m.valid()
                if use_lims:
                    if m.res0 < lims[0] and m.val < lims[1]:
                        lims_ok = True
                    else:
                        lims_ok = False
                else:
                    lims_ok = True
                if run2 and lims_ok:
                    print('Seed ' + str(i) + ' is running for all samples...')
                    m.run(2, print_i=False)
                print(i)
                self.r[i] = m
                if m.res0 < lims[0] and m.val < lims[1]:
                    self.good_seeds.append(i)
                n = n + 1
        else:
            for i in range(seed_min, seed_max + 1, 1):
                m = model(endm(seed=i, bounds = self.e_bnds), data0=self.data0, data2=self.data2, x0=self.x0, bnds=self.bnds)
                m.valid()
                if use_lims:
                    if m.res0 < lims[0] and m.val < lims[1]:
                        lims_ok = True
                    else:
                        lims_ok = False
                else:
                    lims_ok = True
                if run2 and lims_ok:
                    print('Seed ' + str(i) + ' is running for all samples...')
                    m.run(2, print_i=False)
                print(i)
                self.r[i] = m
                if m.res0 < lims[0] and m.val < lims[1]:
                    self.good_seeds.append(i)
                n = n + 1
                    
        print('Number of seeds: ' + str(n))
        
# Seeds of valid model simulations discussed in the article
valid_seeds = [1460,
3703,
5027,
7229,
8569,
9187,
9887,
11477,
12013,
12722,

12740,
13219,
13741,
14326,
15491,
16597,
17067,
17134,
17289,
17519,

19094,
20531,
20771,
21210,
23172,
24506,
25082,
25168,
26409,
26695,

27510,
28366,
30217,
30849,
36588,
36971,
37035,
37296,
38432,
38895,

39000,
42095,
42644,
44509,
45609,
45845,
47617,
48024,
48334,
48555]

# This function creates an instance of the model class with the median results of seeds from mc as the data.
def medres(mc, seeds=[]): # mc input is an instance of the mc class, and seeds is the corresponding list of seeds
    seeds_loc = []
    if seeds:
        seeds_loc = seeds
    else:
        for i in mc.r:
            if mc.r[i].m2:
                seeds_loc.append(i)
    cols = ['Sr_m', 'Nd_m', 'Hf_m', 'X_OC', 'X_SED', 'X_RC', 'Fd', 'P0', 'F_per', 'F_tot', 'dT_RC', 'Tp']
    med = model()
    med.r2 = mc.r[seeds_loc[0]].r2.copy()
    for i in cols:
        for j in med.r2.index:
            ij = []
            for k in seeds_loc:
                ij.append(mc.r[k].r2[i][j])
            ij = np.array(ij)
            med.r2[i][j] = np.median(ij)
    med.r2 = med.r2[['Group', 'Distance', 'Age_shield', 'Sample_Name', 'Sr', 'Nd', 'Hf'] + cols].copy()
    return med

# This function calculates source density and plume dynamics parameters from the data (see also embedded equations in Supplementary Table 2)
# r_plume is the assumed radius of the plume in [m] and visc is the assumed asthenospheric viscosity in [Pa s]
def dynamics(model, r_plume=70E3, visc=1.8E19): # model is an instance of the model class
    model.r2['d_rho_RC'] = 180 * model.r2['X_RC']
    model.r2['d_rho_per'] = -214.5 * model.r2['Fd'] * (1 - model.r2['X_RC'])
    model.r2['d_rho_c'] = model.r2['d_rho_RC'] + model.r2['d_rho_per']
    model.r2['d_rho_T'] = -3300 * 0.00003 * (model.r2['Tp'] - 1350)
    model.r2['d_rho'] = model.r2['d_rho_c'] + model.r2['d_rho_T']
    model.r2['v_p'] = (-(2/9) * model.r2['d_rho'] * 9.81 * r_plume ** 2 / visc) * 3155760000
    model.r2['Q_p'] = (model.r2['v_p'] / 3155760000) * r_plume ** 2 * np.pi
    model.r2['Q_v'] = model.r2['Q_p'] * (3300/2900) * model.r2['F_tot']
    return model

# This function finds the plume radius resulting in a Q_v value of 4.88 [m3 s-1] in Kauai, stacking individual model simulation results at the level of the Van Ark and Lin (2004) estimates
def find_radius(model, Qv_Kauai = 4.88, visc=1.8E19): # model is an instance of the model class that has been edited with the dynamics function
    F_tot_mean_kauai = np.mean(model.r2['F_tot'][model.r2['Group'] == 'KAUAI'])
    d_rho_mean_kauai = np.mean(model.r2['d_rho'][model.r2['Group'] == 'KAUAI'])
    radius = (Qv_Kauai / (-(2/9) * d_rho_mean_kauai / visc * 9.81 * np.pi * 3300/2900 * F_tot_mean_kauai)) ** (1/4)
    return radius

# Lines running the model code
monte_carlo = mc()
monte_carlo.run(of_list=True, seeds=valid_seeds)
median_results = medres(monte_carlo, valid_seeds)
median_results_with_dynamics = dynamics(median_results)
Supplementary_Table_2 = median_results_with_dynamics.r2.to_csv('Supplementary_Table_2.csv')


# Calculates the Q_v values of individual simulations with a plume radius yielding values comparable to geophysical estimates. Radius results range from 56.7 to 80.1 [km].
plume_radii = []
individual_models = {}
for i in valid_seeds:
    model_i = dynamics(monte_carlo.r[i])
    radius_i = find_radius(model_i)
    plume_radii.append(radius_i)
    model_i2 = dynamics(monte_carlo.r[i], r_plume=radius_i)
    individual_models[i] = model_i2

# Imports literature geophysical estimates of Q_v
Qv_0 = pd.read_csv('Van_Ark_and_Lin_Qv.csv')
Qv_1 = pd.read_csv('Vidal_and_Bonneville_Qv.csv')
Qv_2 = pd.read_csv('Wessel_Qv.csv')

# Shield age array to perform the LOWESS fit
x_vals = np.linspace(0.05, 42, 840)

# Plots LOWESS fit curves for individual simulations
for i in valid_seeds:
    plt.plot(x_vals, slw.lowess(individual_models[i].r2['Q_v'], individual_models[i].r2['Age_shield'], xvals=x_vals), lw=0.5, c='green', alpha=0.5)    

# Plots LOWESS fit curve for the median results
plt.plot(x_vals, slw.lowess(median_results.r2['Q_v'], median_results.r2['Age_shield'], xvals=x_vals), lw=1.5, c='green')    

# Plots LOWESS fit curves for the geophysical estimates
plt.plot(Qv_0['Age'], Qv_0['Qv_phys'], c='k', lw=0.9, ls='dashed')
plt.plot(Qv_1['Age'], Qv_1['Qv_phys'], c='k', lw=0.9, ls='dotted')
plt.plot(Qv_2['Age'], Qv_2['Qv_phys'], c='k', lw=0.9)

plt.ylim(0, 12)
plt.xlabel('Age of shield volcanoes [Ma]')
plt.ylabel('Qv [m3 s-1]')

# Save plotted curves as a pdf
with PdfPages('Figure_3e_curves.pdf') as pdf:
    pdf.savefig()
    plt.close()








