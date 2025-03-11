#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 11:15:49 2025

@author: wjbosl
"""

# Import packages
import pandas as pd
import numpy as np
import tensorly as tl
from scipy.interpolate import interp1d

# -----------------------------------------
# Normalize data
# -----------------------------------------
def normalize(df):
    # Normalize by measure
    features = df.Feature.unique()

    for f in features:
        df_f = df[df['Feature'] == f]
        mx = np.max(df_f['Value'])
        mn = np.min(df_f['Value'])
        absmx = max(mx, mn)
        df.loc[df['Feature'] == f, 'Value'] = df_f['Value'] / absmx

    return df


# -----------------------------------------
# Impute missing or nan data using group averages
# -----------------------------------------
def impute_missing(df):
    # df.loc[df['Value'].isnull(), 'value_is_NaN'] = 'Yes'
    # df.loc[df['Value'].notnull(), 'value_is_NaN'] = 'No'
    headers = df.columns

    # Get the rows with a NaN
    # df_not_na = df[df['value_is_NaN']=='No']
    mean_value = df['Value'].mean()
    df['Value'].fillna(value=mean_value, inplace=True)
    return df

    # df_nan = df.isna()
    print("Start")
    df_nan = df.isnull()
    print("len of nans: %d of %d" % (len(df_nan), len(df)))

    # Loop over the rows with bad data
    if 'Sleep' in headers:
        for index, row in df_nan.iterrows():
            g = row['Label']
            s = row['Sleep']
            f = row['Feature']
            c = row['Channel']
            new_value = np.mean(
                df[(df['Label'] == g) & (df['Sleep'] == s) & (df['Feature'] == f) & (df['Channel'] == c)].Value)
            df.loc[index, 'Value'] = new_value
    else:
        for index, row in df_nan.iterrows():
            g = row['Label']
            f = row['Feature']
            c = row['Channel']
            new_value = np.mean(df[(df['Label'] == g) & (df['Feature'] == f) & (df['Channel'] == c)].Value)
            df.loc[index, 'Value'] = new_value
    return df


# -----------------------------------------
# Read a long format .csv file and pull into
# a tensor structure
#
# Input: infilename
# Return: dataframe with all data for analysis
# -----------------------------------------
def read_csv_file(infilename, df_clinical):
    global sup_title, plotfilename, RANK

    # Let's make sure white space at the end of NaN's won't confuse the reader
    additional_nans = ['NaN ', 'nan ', 'na ', 'inf', 'inf ']

    # Assume that indir is the full pathname for the directory where data lies.
    # Inside indir there will be one or more zip files whose name is the ID.
    df = pd.read_csv(infilename, skipinitialspace=True, dtype={'ID': str,'DX': str}, na_values=additional_nans)
    df = normalize(df)
    
    # Add the clinical data
    headers = list(df.columns.values)
    cl_headers = list(df_clinical.columns.values)
    
    if "Visit_Age" in headers:
        df2 = pd.merge(df, df_clinical, on=['ID', 'Visit_Age'])
        print("Merging on ID and Visit_Age")
    elif "Index" in headers and "ID" in headers:
        df2 = pd.merge(df, df_clinical, on=['Index','ID'])
        print("Merging on Index and IDs only")
    elif "Index" in headers:
        df2 = pd.merge(df, df_clinical, on=['Index'])
        print("Merging on Index only")
    else:
        df2 = pd.merge(df, df_clinical, on=['ID'])
        print("Merging on ID only")
        #print("New headers: ", list(df2.columns.values))
        #exit()
    df = df2  
    
    headers = list(df.columns.values)        
    print ("Fraction good channels: ", df.Value.count() / len(df.Value))
    df = impute_missing(df)
    good_channels = df.Value.count() / len(df.Value)
    print ("Fraction after imputation: ", good_channels)
    if good_channels < 1.0:
        df.dropna(inplace=True)  # Dropping all the rows with nan values
    if len(df.Value) == 0.0:
        good_channels = 0.0
        print ("Fraction after dropping bad channels: ", good_channels)
        exit()
        
    return df

# -----------------------------------------
# Create a tensor object
# -----------------------------------------
def create_tensor(df, params):
    
    desired_channels = params["channels"]
    df = df[[x in desired_channels for x in df.Channel]] 


    # Read the data into structures appropriate for tensorization
    # Axes: Feature (nonlinear measure), Channel, Freq
    axes = {}
    IDs = df.ID.unique()
    features = df.Feature.unique()
    channels = df.Channel.unique()
    axes['ID'] = IDs
    axes['Feature'] = features
    axes['Channel'] = channels
    axes['Freq'] = []  # to be filled below with new frequencies, x_new

    # Create multiscale curves and interpolate to a standard set of frequencies
    # Use the minimum sampling rate to determine the range of new frequencies
    # delta: 0-4, theta: 4-7, alpha: 7-13, beta: 13-30, gamma: 30-60, gamma+: 60 and above
    # Only use details, wavelets that start with D
    details = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7']
    df = df[df["Wavelet"].isin(details)]
#    x_new = np.array([64., 32., 16., 8., 4., 2.])
    x_new = np.array([100., 50., 25., 12.5, 6.25, 3.125, 1.55])
    # x_new = np.array([1.6, 3.1, 6.2, 12.5, 25., 50., 100.])
    axes['Freq'] = x_new

    # Create the numpy array that will hold the new tensor structures
    # data_np((ID,Feature,Channel,Freq))
    # n1 = len(IDs)
    n3 = len(features)
    n4 = len(channels)
    n5 = len(x_new)
    data_np2 = []

    index = 0
    id_stage = []    
    for i1, id in enumerate(IDs):
        df_id = df[(df["ID"] == id)]
        labels = df_id.Label.unique()
                
        index_list = list(df_id.Index.unique())

        for i2, label in enumerate(labels):
            index += 1
            values = np.full((n3, n4, n5), np.nan)
            id_stage.append([id, label])
            df_id_s = df_id[(df_id["Label"] == label)]

            # Get the sampling rate, then compute wavelet frequencies
            # srate = df_id_s.Rate.unique()[0] # Get the sampling rate for this patient

            x = x_new

            for i3, feature in enumerate(features):
                df_id_feature = df_id_s[df_id_s["Feature"] == feature]
                
                # Randomly choose one channel
                nch = len(channels)
                random_int1 = np.random.randint(nch) 
                random_int2 = np.random.randint(nch) 
    
                for i4, ch in enumerate(channels):
                    df_id_feature_ch = df_id_feature[df_id_feature['Channel'] == ch]
    
                    y = np.array(df_id_feature_ch["Value"])
                    
                    if i4 == random_int1 or i4 == random_int2:
                        y = np.zeros(len(y))
    
                    
                    #if (#feature=='Power') and (ch=='Fp1'): 
                    #print(id,feature,ch,len(x_new), len(y),y)
                        
                    try:
                        func = interp1d(x, y, "linear")
                    except:
                        print(index_list,feature,ch,len(x_new), len(y))
                        print(df_id_feature_ch["Value"])
                        exit()
                    y_new = func(x_new)
                    values[i3, i4, 0:] = y_new[0:]

            data_np2.append(values)
    
    data_np = np.array(data_np2)
    print("--->>>  shape of EEG tensor = ", data_np.shape)
    
    # Create a tensor object and scale
    X = tl.tensor(data_np)

    # Some initializations
    nID = len(X)  # number of subjects
     
    Y = []
    ages = {}
    age_list = []
    id_list = []
    dx_list = {}
    dx_indices = []
    hist_gsz = {}
    hist_gsz_list = []
    abs_type_array = []
    med_array = []
    severity = []
    Meds = {}
    headers = list(df.columns.values)
    
    for i in range(nID):
        label = id_stage[i][1]
        id = id_stage[i][0]
        id_list.append(id)
        
        sub_df = df.loc[df['ID'] == id]
        dx = sub_df.iloc[0]['DX']
        dx_list[id] = dx
        dx_indices.append(dx)

        age = float(sub_df.iloc[0]['Age'])
        ages[id] = age
        age_list.append(age)

        if "gtc_sz_history" in headers:
            hst_df = sub_df.iloc[0]['gtc_sz_history']
            hist_gsz[id] = hst_df
            hist_gsz_list.append(hst_df)

        if "Sex" in headers:
            sex = sub_df.iloc[0]['Sex']
        if "Gender" in headers:
            sex = sub_df.iloc[0]['Gender']
        if "Birth_Weight" in headers:
            bw = sub_df.iloc[0]['Birth_Weight']
        if "absence_epilepsy_type" in headers:
           epitype = sub_df.iloc[0]["absence_epilepsy_type"]
           abs_type_array.append(epitype)
        if "Diagnosis" in headers:
            diag = sub_df.iloc[0]["Diagnosis"]
        if "Epilepsy" in headers:
            epilepsy = sub_df.iloc[0]["Epilepsy"]
        if "Age_1sz" in headers:
            age_1sz = sub_df.iloc[0]["Age_1sz"]
        if "gtc_sz_history" in headers:
            gtc_sz_history = sub_df.iloc[0]["gtc_sz_history"]
        if "age_1sz" in headers:
            age_1sz = sub_df.iloc[0]["age_1sz"]
        if "Meds" in headers:
            meds = sub_df.iloc[0]["Meds"]   
            med_array.append(meds)
            Meds[id] = meds
        if "ASD" in headers:
            asd = sub_df.iloc[0]["ASD"]            
        if "Severity" in headers:
            sev = sub_df.iloc[0]["Severity"]   
            severity.append(sev)
            
        Y.append([label])#,epilepsy]) #,gtc_sz_history,age_1sz])  # Labels for SupParafac
        
    Y = np.array(Y)
    
    return data_np, Y[:,0]
    
    