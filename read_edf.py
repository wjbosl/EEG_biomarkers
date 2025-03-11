#!/usr/bin/env python

"""process_edf.py:

__author__ = "William Bosl"
__copyright__ = "Copyright 2020, William J. Bosl"
__credits__ = ["William Bosl"]
__license__ = All rights reserved by William J. Bosl
__version__ = "1.0.0"
__maintainer__ = "William Bosl"
__email__ = "william.bosl@childrens.harvard.edu"
__status__ = "Initial test"
"""

#
#  Copyright (c) 2020 William J. Bosl. All rights reserved.
#

import numpy as np
#import pyedflib
from mne.io import read_raw_edf

# A few global variables
all_features = ["Power", "SampE", "hurstrs", "dfa", "cd", "lyap0", "RR", "DET", "LAM", "Lentr", "Lmax", "Lmean", "TT"]
f_labels = []
FORMAT = "long"
SEGMENT = "beg"  # "beg" = beginning, "mid" = middle, "end" = end. Position from which to extract the segment
SUBSAMPLE = False

# Use these to convert EGI net numbers to standard 10-20 location terminology
# The last entries for each here are not standard, but we've included them for completeness
hydroCell128_channelList = [22,    9,   33,  24,  11, 124, 122,  45,  36, 129, 104, 108,  58,  52,  62,  92,  96,  70,  83,   6]
egi_channel_names =       ["Fp1","Fp2","F7","F3","Fz","F4","F8","T7","C3","Cz","C4","T8","P7","P3","Pz","P4","P8","O1","O2","Fcz"]
EGI_64v2 =                [ 11,    6,   15,  13,  3,   62,  61,  24,  17,  65,  54,  52,  27,  28,  34,  46,  49,  37,  40,   4]
#EGI_64v1 =                [ 10,    5,   18,  12,  6,   60,  58,  24,  20,  65,  50,  52,  30,  28,  34,  42,  44,  35,  39,   8]

# the master list is just used for filtering all the extraneous channels that are typically saved in the Epilepsy Center files
master_channel_list = ["Fp1","Fp2","FP1","FP2","T3","T5","T4","T6","F7","F3","Fz","F4","F8","T7","C3","Cz","C4","T8","P7","P3","Pz","P4","P8","O1","O2","E1","E2"]
processed_channel_names = []

#---------------------------------------------------------------------------------
# Unpack the .edf file and return the data array, channelnames, and sampling rate
#---------------------------------------------------------------------------------
#----------------------------------------------------------------------
# Extract the time series and sensor names from the file. We will
# process only .edf and .csv files at this time.
#----------------------------------------------------------------------
def get_data(filename, max_nt, params):
    
    device = "default"
    if "device" in params:
        device == params["device"]
        
    resample = False # default value
    if "resample" in params:
        if params["resample"] == '0':
            resample = False
        else:
            resampling_rate = float(params["resample"])
            resample = True
        
    # if sleep stage or "awake" is in the filename, extract it
    if "Awake" in filename:
        params["sleep"] = "awake"
    elif "Sleep2" in filename:
        params["sleep"] = "sleep2"
    elif "Sleep3" in filename:
        params["sleep"] = "sleep3"
        
    # Temporary, for BECTS files
    if filename[0] == 'B': # BECTS case
        params["group"] = "BECTS"
    elif filename[0] == 'C': # Control, no epilepsy
        params["group"] = "control"
        
    file = open(filename,'r')
    last = len(filename)
    
    # default value for age
    age = 0

    MNE = True
    
    #-----------  MNE read functions --------------
    if MNE:
        try:
            raw = read_raw_edf(filename, preload=True, verbose='ERROR')
        except:
            print("Cannot process file: %s" %(filename))
            return
        channelNames = raw.ch_names
        nch = len(channelNames)
        srate = int(raw.info['sfreq'])
        data, times = raw.get_data(return_times=True)
        nch, nt = data.shape
        if srate == 0: 
            srate = 128 # Temporary! This error arose in the Supriya data sets
    
        # Resampling, if desired
        if resample and (srate != resampling_rate): 
            raw.resample(sfreq=resampling_rate)
            srate = resampling_rate
    #-----------  MNE read functions --------------


    #-----------  pyedflib read functions --------------
    else:
        try:
            print("Try to open file ", filename)
            f = pyedflib.EdfReader(filename)
        except:
            print("Cannot process file: %s" %(filename))
            return
        nch = f.signals_in_file
        channelNames = f.getSignalLabels()
        srate = f.getSampleFrequency(0)
        data = np.zeros((nch, f.getNSamples()[0]))
        for i in np.arange(nch):
            data[i, :] = f.readSignal(i)
        nch, nt = data.shape
    #-----------  pyedflib read functions --------------

    # Change old names to new: T3->T7, T5->P7, T4->T8, T6->P8
    for c, ch in enumerate(channelNames):
        if ch=='T3': channelNames[c] = 'T7'
        if ch=='T4': channelNames[c] = 'P7'
        if ch=='T5': channelNames[c] = 'T8'
        if ch=='T6': channelNames[c] = 'P8'
        
        # And fix case
        if ch=='FP1': channelNames[c] = 'Fp1'
        if ch=='FP2': channelNames[c] = 'Fp2'
        
        # And a minor substitution
        if ch=='AF3': channelNames[c] = 'Fp1'
        if ch=='AF4': channelNames[c] = 'Fp2'
        
        # For sleep channels
        if ch.endswith("/M1") or ch.endswith("/M2"): ch = ch[:-3]
        if ch.endswith("-M1") or ch.endswith("-M2"): ch = ch[:-3]
        if ch.endswith("/A1") or ch.endswith("/A2"): ch = ch[:-3]
        if ch.endswith("-A1") or ch.endswith("-A2"): ch = ch[:-3]


    # Remove the "-Ref1:8" part of the filename for some sensors
    tag = "-Ref1:8"
    for c, ch in enumerate(channelNames):
        if tag in ch:
            n = len(ch)
            channelNames[c] = ch[0:n-7]

    # We generally want only the standard 10-20 channels (19 max)
    # The high density EGI nets have numbered channel names. We'll need to convert.
    new_data = []
    new_channel_list = []

    # These two sections are for extracting desired channels from high density EGI devices
    # You can ignore this if data from any other devices are being processes.
    if channelNames[1] == 'CC1':  # This is an ECoG from Cook Hospital
        dummy=1
    
    elif nch == 129:  # EGI hydrocell 128
        for c, ch in enumerate(egi_channel_names):
            new_channel_list.append(ch)
            i = hydroCell128_channelList[c] - 1
            new_data.append(data[i])
        params["device"] = "EGI_hydrocell_128"
        channelNames = new_channel_list
        data = new_data
        #channelNames = range(1,129)

    elif nch == 65:  # EGI 64 v2
        for c, ch in enumerate(egi_channel_names):
            new_channel_list.append(ch)
            i = EGI_64v2[c] - 1
            new_data.append(data[i])
        params["device"] = "EGI_64_v2"
        channelNames = new_channel_list
        data = new_data
        #channelNames = range(1,65)
        

    # Let's trim the data array so that time series are not more than max_nt seconds
    max_nt_points = max_nt * srate
    nt = int(min(max_nt_points, nt))

    # Here we're just picking out a segment of max_nt seconds from the entire EEG time series.
    # Start at the beginning (beg), middle (mid), or end of the entire array.
    m = len(data)
    n = len(data[0])
    if "beg" in SEGMENT:
        m1 = 0
    if "end" in SEGMENT:
        m1 = n-nt
    elif "mid" in SEGMENT:
        m1 = int((n-nt)/2)
    else:
        m1 = 0
    m2 = m1 + nt
    new_data = []
    for i in range(m):
        new_data.append( np.array(data[i][m1:m2]) )

    data = np.array(new_data)
    
    # Return a numpy array with the selected data, the channel names, and the sampling rate.
    return data, channelNames, srate


#
#----------------------------------------------------------------------
# Main
#----------------------------------------------------------------------
if __name__ == "__main__":
    print ("This file must be called from elsewhere.")


