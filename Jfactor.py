"""
General tools for reading precomputed tables
"""
import os, math
import numpy as np
import scipy

curdir=os.path.dirname(os.path.realpath(__file__))
# *********************************************************
#  Jfactor
def extract_values(filename, pos1, pos2):
    value_file = open(filename).readlines()
    val1 = np.array([])
    val2 = np.array([])
    for l in value_file:
        line = l.split()
        # print (line)
        try:
            float(line[pos1])
            float(line[pos2])
        except ValueError:
            continue
        val1 = np.append(val1, np.float64(line[pos1].strip()))
        val2 = np.append(val2, np.float64(line[pos2].strip()))
    return val1, val2

def Jfactor_Clumpy(profile, process):
    if process=='ann':
        nametag='Jfactor_dJdOmega_GeV2_cm5_sr'
        col=4
    elif process=='decay':
        nametag='Dfactor_dDdOmega_GeV_cm2_sr'
        col=3
    
    clumpyfile = f"{curdir}/resources/Clumpy_precomp/Jfactor//{nametag}_{profile}_NestiSalucci.output"
    psi_values, Jpsi_values = extract_values(clumpyfile, 0, col)

    JPsi_dict = dict()
    JPsi_dict["J"] = Jpsi_values
    JPsi_dict["psi"] = psi_values
    return JPsi_dict

def Interpolate_Jfactor(Jfactor, psival):
    y_interp = scipy.interpolate.splrep(Jfactor["psi"], Jfactor["J"])
    interp_Jpsi = scipy.interpolate.splev(psival, y_interp, der=0)

    return interp_Jpsi