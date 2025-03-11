#!/usr/bin/env python

"""process_mat.py:

__author__ = "William Bosl"
__copyright__ = "Copyright 2020, William J. Bosl"
__credits__ = ["William Bosl"]
__license__ = All rights reserved by William J. Bosl
__version__ = "1.0.0"
__maintainer__ = "William Bosl"
__email__ = "wjbosl@gmail.com"
__status__ = "Initial test"
"""

#
#  Copyright (c) 2020 William J. Bosl. All rights reserved.
#
import sys
import numpy as np
import scipy.io as sio
import pathlib
import mne.filter as filter
import pandas as pd
from mne.io import read_raw_edf
import glob


# A few global variables
f_labels = []
DEVICE = "unk"
AGE = 0
FORMAT = "long"
SEGMENT = "beg"  # "beg" = beginning, "mid" = middle, "end" = end. Position from which to extract the segment

"""
hydroCell128: FP1 (E22), FP2 (E9), F7 (E33), F8 (E122), F3 (E24), F4 (E124), 
Fz (E11), T3 (E45), C3 (E36), T8 (E108), C4 (E104), P7 (E58), P8 (E96), 
P3 (E52), P4 (E92), Pz (E62), O1 (E70), O2 (E83)

['Fp1','Fp2','F7','F8','F3','F4','Fz','T7','C3','T8','C4','P7','P8','P3','P4','Pz','O1','O2']
[ 22,    9,   33, 122,  24, 124,  11,  45,  36,  108, 104, 58,  96,  52,  92,  62,  70,  83 ]


64channel net: FP1 (E11), FP2 (E6), F7 (E15), F8 (E61), F3 (E13), F4 (E62), 
Fz (E3), T7 (E24), C3 (E17), T8 (E52), C4 (E54), P7 (E27), P8 (E49), 
P3 (E28), P4 (E46), Pz (E34), O1 (E37), O2 (E40)

['Fp1','Fp2','F7','F8','F3','F4','Fz','T7','C3','T8','C4','P7','P8','P3','P4','Pz','O1','O2']
[ 11,    6,   15,  61,  13,  62,  3,   24,  17,  52,  54,  27,  49,  28,  46,  34,  37,  40 ]

"""
# Use these to convert EGI net numbers to standard 10-20 location terminology
# The last entries for each here are not standard, but we've included them for completeness
hydroCell128_channelList = [ 22,    9,   33, 122,  24, 124,  11,  45,  36,  108, 104, 58,  96,  52,  92,  62,  70,  83,  6]
egi_channel_names =        ['Fp1','Fp2','F7','F8','F3','F4','Fz','T7','C3','T8','C4','P7','P8','P3','P4','Pz','O1','O2','Fcz']
EGI_64v2 =                 [ 11,    6,   15,  61,  13,  62,  3,   24,  17,  52,  54,  27,  49,  28,  46,  34,  37,  40,  4]

# the master list is just used for filtering all the extraneous channels that are typically saved in the Epilepsy Center files
master_channel_list = ['Fp1','Fp2','FP1','FP2','F7','F8','F3','F4','Fz','T3','T5','T4','T6','T7','C3','T8','C4','P7','P8','P3','P4','Pz','O1','O2']
processed_channel_names = []

#----------------------------------------------------------------------
# Convert matlab struct to a Python dict
#----------------------------------------------------------------------
def struct_to_dict(struct, name=None):
    result = dict()
    try:
        vals = struct[0,0]
    except IndexError:
        #print name, struct
        vals = ''
    try:
        for name in struct.dtype.names:
            if vals[name].shape == (1,):
                result[name] = vals[name][0]
            else:
                result[name] = struct_to_dict(vals[name], name=name)
    except TypeError:
        return vals
    return result


#----------------------------------------------------------------------
# Extract the time series and sensor names from the file. We will
# process only .edf and .csv files at this time.
#----------------------------------------------------------------------
def read_matlab(all_files, max_nt, resampling_rate=0):

    index_array = []
    X = []
    Channels = []
    srate_array = []
    
    print("Total number of filenames to read = ", len(all_files))
        
    for filename in all_files:
    #for i in range(100):
    #    filename = all_files[i]
   
        # Extract ID, age, and Index
        id_full = pathlib.Path(filename).stem
        n1 = id_full.partition(".")[0]
        id_age = n1
        n2 = n1.split('_')
        ID = n2[0]
        age = 1
        if len(n2) > 1:
            age = int(n2[1][0])
        index = ID + '.' + str(age)
    
        resample = False # default value
        if resampling_rate != 0:
            resample = True
    
        PROCESS_FILE = False
        channelNames = egi_channel_names

        # ---- matlab v7.3 or greater
        #print("Read h5")
        #mat_contents = h5py.File(filename, 'r')
        #print("keys:")
        #for k in mat_contents.keys():
        #    print(k)
        #eeg3 = mat_contents['EEG3']
        #data = mat_contents['EEG3']
        #print("size of EEG3: ", eeg3.keys())
        #exit()
        #--------------------------------
    
        mat_contents = sio.loadmat(filename)
        #mat_proc_info = struct_to_dict(mat_contents['file_proc_info'])
        #data = mat_contents['Category_1_Segment1']
        keys = list(mat_contents.keys())
        
        if ID[-1] == '_':
            ID = ID[0: -1]            
        if id_age[-1] == '_':
            id_age = id_age[0: -1]            

        # Extract an ID and index from the file name
        if ID in keys:
            mat_id = ID
        elif id_age in keys:
            mat_id = id_age
        elif ID+'_' in keys:
            mat_id = ID+'_'
        elif id_age+'_' in keys:
            mat_id = id_age+'_'
                    
        try:
            data = mat_contents[mat_id]
            index_array.append(index)      
            PROCESS_FILE = True

        except:
            print("Problem reading file %s; ID = %s, mat_id = %s " %(filename,ID,mat_id))
            #print(mat_contents.keys())
            PROCESS_FILE = False
        
        if PROCESS_FILE:
            srate = float(mat_contents['samplingRate'][0,0])
            srate_array.append(srate)
            
            # Here we assume that the desired sampling rate is an integer multiple of srate
            if resample and srate != resampling_rate:
                downsamplerate = float(srate/resampling_rate)
                new_data = filter.resample(data, down=downsamplerate, axis=1)
                data = new_data
                srate = resampling_rate
                       
            data_channels, nt = data.shape
    
            # We generally want only the standard 10-20 channels (19 max)
            # The high density EGI nets have numbered channel names. We'll need to convert.
            new_data = []
            new_channel_list = []
    
            # These two sections are for extracting desired channels from high density EGI devices
            # You can ignore this if data from any other devices are being processes.
            if len(data) >= 128:  # EGI hydrocell 128
                for c, ch in enumerate(egi_channel_names):
                    new_channel_list.append(ch)
                    i = hydroCell128_channelList[c] - 1
                    new_data.append(data[i])
                DEVICE = "EGI hydrocell 128"
                channelNames = new_channel_list
                data = new_data
    
            elif len(data) >= 64:  # EGI 64 v2
                for c, ch in enumerate(egi_channel_names):
                    new_channel_list.append(ch)
                    i = EGI_64v2[c] - 1
                    new_data.append(data[i])
                DEVICE = "EGI 64 v2"
                channelNames = new_channel_list
                data = new_data
    
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
            
            X.append(data)
            Channels.append(channelNames)
        
    X = np.array(X)
    
    print("Shape of X: ", X.shape)
    print("Len of  index_array, Channels, srate_array: ", len(index_array), len(Channels), len(srate_array) )
               
    # dictionary of lists 
    dict = {'Index': index_array, 'Channels': Channels, 'srate': srate_array} 
    df_data = pd.DataFrame(dict)

    # Return a numpy array with the selected data, the channel names, and the sampling rate.
    return df_data, X


def read_edf(all_files, max_time, resampling_rate=0):
    print("Total number of EGI eeg filenames to read = ", len(all_files))
    
    hydroCell128_channelList = [22,    9,   33,  24,  11, 124, 122,  45,  36, 128, 104, 108,  58,  52,  62,  92,  96,  70,  83,   6]
    EGI_64v2 =                [ 11,    6,   15,  13,  3,   62,  61,  24,  17,  64,  54,  52,  27,  28,  34,  46,  49,  37,  40,   4]
    egi_channel_names =       ["Fp1","Fp2","F7","F3","Fz","F4","F8","T7","C3","C4","T8","P7","P3","Pz","P4","P8","O1","O2","Fcz"]

    channel_list = []
    for i in hydroCell128_channelList:
        ch = "EEG " + str(i)
        channel_list.append(ch)
    hydroCell128_channelList = channel_list
    
    channel_list = []
    for i in EGI_64v2:
        ch = "EEG " + str(i)
        channel_list.append(ch)
    EGI_64v2 = channel_list    

    count = 0
    index_array = []
    srate = []
    X = []
    Channels = []
    age_list = [3,6,9,12,18,24,36]
    for filename in all_files:
        success = True
        try:
            raw = read_raw_edf(filename, preload=True, verbose='ERROR')
            sfreq = int(raw.info['sfreq'])
            if sfreq < resampling_rate:
                success = False
                print("Freq too low: ",sfreq, filename)
        except:
            print("Cannot process file: %s" %(filename), "; skipping to next file")
            success = False
            
        nch = len(raw.ch_names)
        if nch >= 128:
            picks_eeg = hydroCell128_channelList
        elif nch >= 64:
            picks_eeg = EGI_64v2
        ch_set = set(picks_eeg)

        # Does this file have the required channels?
        df = raw.to_data_frame()        
        col_list = [s for s in raw.ch_names if (s in picks_eeg)]
        df = df[col_list]
        channel_names = list(df.columns) # raw.ch_names  
                    
        if set(channel_names) != set(picks_eeg):
            success = False
            these_channels = set(channel_names)
            missing_set = ch_set - these_channels
            print("%d channels; Cannot process file: %s" %(len(channel_names),filename), "; missing channels: ", missing_set)
            
            print("failed channel names = ", channel_names)
            print("failed picks_eeg = ", picks_eeg)
            print("diff: ", missing_set)
                    
        if success:            
            # MNE can set the EEG reference if needed
            #raw.set_eeg_reference('average')
            count += 1
            
            # Extract an ID and index from the file name
            index = pathlib.Path(filename).stem
            #ID = id_full[0:7]
            
            index_array.append(index)   
            
            sfreq = int(raw.info['sfreq'])
            sf = sfreq
            if sfreq != resampling_rate:
                raw.resample(sfreq=resampling_rate)
                sf = resampling_rate
            srate.append(sf)

            nt = int(sf*max_time)
            
            x = df.transpose().to_numpy()[0:,0:nt]
            X.append(x)
            Channels.append(channel_names)
            
    if count == 0:
        print("No files processed, exiting")
        exit()
    else:
        print(count," files processed.")

    print("Channels selected & found: ")
    print(egi_channel_names)
    print("Number of files to analyze before selecting = ", count)

    X = np.array(X)
    
    print("After reading, len X = ", len(X))
               
    # dictionary of lists 
    dict = {'Index': index_array, 'Channels': Channels, 'srate': srate} 
    df_data = pd.DataFrame(dict)

    return df_data, X
#
#----------------------------------------------------------------------
# Main
#----------------------------------------------------------------------
if __name__ == "__main__":
    
    argc = len(sys.argv)
    if argc == 1:
        print ("This file must be called from elsewhere.")
    else:
        glob_list = glob.glob(sys.argv[1])
        if "edf" in glob_list[0]:
            read_edf(glob_list, 30, 250.0)
        elif "mat" in glob_list[0]:  
            print("N files = ", len(glob_list))
            read_matlab(glob_list, 30, 250.0)
        else:
            print("file format not found.")



