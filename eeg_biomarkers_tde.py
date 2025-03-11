#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 07:55:52 2024

@author: wjbosl
"""
# Import packages
import sys
import os
import numpy as np
from mne.io import read_raw_edf
import glob
#from emucore_direct.client import EmuCoreClient
import pathlib
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn import metrics
from tensorly.regression.cp_regression import CPRegressor
import pySupCP
from distutils.util import strtobool
import read_egi
import read_csv_features


# -----------------------------------------
# Convert tensor X to weights using model V
# -----------------------------------------
def get_weights(Xnew, V, R):
    pySCP = pySupCP.pySupCP(R)
    # Compute some parameters
    m = Xnew.shape         #
    n = m[0]            # sample size under test/train
    L = len(m)          # number of modes (as in multi-modal data), in case of BECTS, it's 4
    p = np.prod(m[1:L]) # p1*p2*...*pK           % GL: potentially very large
    
    # Matricize the factor matrices
    Vmat = np.zeros((p,R)) # a very long matrix (p can be very large)
    for r in range(R):      # if rank is 10, this r goes from 0 to 9
        Temp = pySCP.TensProd(V, [r])
        Vmat[:,r] = Temp.flatten(order='F')

    # Matricize Xnew
    Xmat = np.moveaxis(Xnew, 0, -1)  # move the first dimension, n, to the last; Xmat is a matricization of X samples
    Xmat = Xmat.reshape(-1, n ,order='F')
    
    # Compute weights
    U = Xmat.conj().T @ Vmat
    return U

#------------------------------------------
# Compute AUC
#------------------------------------------
def get_auc(y, y_pred, score, labels):

    lab0 = labels[0]
    lab1 = labels[1]
    negative_lab = lab0
    positive_lab = lab1
    prevalence = np.count_nonzero(y == positive_lab)/len(y)    

    # Compute F1 scores for each threshold
    precision_curve, recall_curve, thresholds = metrics.precision_recall_curve(y, score, pos_label=positive_lab)
    sum_array = precision_curve + recall_curve
    sum_array = np.where(sum_array == 0, 1, sum_array)  # Replace 0 with 1
    f1_scores = 2 * (precision_curve * recall_curve) / (sum_array)

    # Find the index of the maximum F1 score; get precision and recall at the best threshold
    best_index = np.argmax(f1_scores)      
    precision = precision_curve[best_index]
    recall = recall_curve[best_index]
    F1 = f1_scores[best_index]
    
    # Get the scores directly
    recall = metrics.recall_score(y, y_pred, pos_label=positive_lab)
    precision = metrics.precision_score(y, y_pred, pos_label=positive_lab, zero_division=0.0)
    F1 = metrics.f1_score(y, y_pred, pos_label=positive_lab)
    spec = metrics.recall_score(y, y_pred, pos_label=negative_lab)
    sens = recall

    
    # Sensitivity and specificity from scores
    fpr, tpr, thresholds = metrics.roc_curve(y, score, pos_label=positive_lab)
    auc = metrics.auc(fpr, tpr)
    #auc = metrics.roc_auc_score(y, score)
    sens_array = tpr # 1.0 - fnr
    spec_array = 1.0 - fpr
    ss = (sens_array+spec_array).tolist()
    max_value = max(ss)
    best_index = ss.index(max_value)
#    sens = sens_array[best_index]
#    spec = spec_array[best_index]  
        
    # Compute PPV and NPV from sens and spec
    p = prevalence
    num = sens*p
    d1 = (1.0-spec)*(1.0-p)
    if num + d1 > 0:
        ppv = num / (num + d1 )
    else: 
        ppv = 0.0
    num = spec*(1.0-p)
    d1 = (1-sens)*p
    if num + d1 > 0:
        npv = num / (num + d1)
    else:
        npv = 0.0
        
    return auc, recall, precision, sens, spec, F1, ppv, npv

#------------------------------------------
# Read parameter file
#------------------------------------------
def read_parameters(argv):
    argc = len(argv)
    parameters = {}
    TENSOR_FAC = True
    TENSOR_RANK = 3
    TENSOR_METHOD = "tensorly"  # "tensorly" or "supcp"
    REGRESS = False
    EEG_FORMAT = "edf"
    MAX_NT = 30
    RESAMPLING = 200.0
    SLEEP_STAGE = 0
    PREPROCESSING_METHOD = 1
    vbias = 0.3
    gain = 0.1
    nnodes = 100
    density = 0.1
    feature_scaling = 0.7
    emucore = False
    covariates = []
    channels = ['Fp1', 'Fp2', 'C3', 'C4' ,'O1', 'O2', 'F3', 'F4', 'F7', 'F8', 'Fz' , 'P3', 'P4', 'Pz','T7', 'T8', 'P7' ,'P8']
    
    # Read the input parameter file. Assume key, value pairs
    if argc == 1:  # Use default parameters
        print("Use: > python emu_EEG.py params.txt")
        exit()
    elif argc == 2:
        param_filename = argv[1]
        lines = open(param_filename).read().split('\n')
    else:
        print("Use:")
        print("> python emu_EEG.py parameter_file.txt")
        exit()

    # Parse the parameter file
    target = "default" # default value
    for i,line in enumerate(lines):   
        if line.isspace() or len(line)==0 or line[0]=='#':
            continue
        else:
            kval = line.split(',')
            key = kval[0].strip()
            value = kval[1].strip()
            
            if key.lower() == "path":
                path = value
            elif key.lower() == "files":
                files = value
            elif key.lower() == "clinical":
                clinical_filename = value
            elif key.lower() == "target":
                target = value
            elif key.lower() == "tensor":
                TENSOR_FAC = strtobool(value)
            elif key.lower() == "rank":
                TENSOR_RANK = int(value)
            elif key.lower() == "tensor_method":
                TENSOR_METHOD = value
            elif key.lower() == "regress":
                REGRESS = strtobool(value)
            elif key.lower() == "eeg_format":
                EEG_FORMAT = value
            elif key.lower() == "max_nt":
                MAX_NT = float(value)
            elif key.lower() == "resampling":
                RESAMPLING = float(value)
            elif key.lower() == "sleep_stage":
                SLEEP_STAGE = int(value)
            elif key.lower() == "preprocessing_method":
                PREPROCESSING_METHOD = int(value)
            elif key.lower() == "vbias":
                vbias = float(value)
            elif key.lower() == "gain":
                gain = float(value)
            elif key.lower() == "nnodes":
                nnodes = int(value)
            elif key.lower() == "density":
                density = float(value)
            elif key.lower() == "feature_scaling":
                feature_scaling = float(value)
            elif key.lower() == "emucore":
                emucore = strtobool(value)
            elif key.lower() == "covariates":
                kval = line.split(',')
                covariates = kval[1:]
            elif key.lower() == "channels":
                kval = line.split(',')
                cleaned_list = [s.strip() for s in kval]
                channels = cleaned_list[1:]

    all_file_names = path+files
    
    # Assign parameters to a dictionary for easy passing
    parameters["all_file_names"] = all_file_names
    parameters["clinical_filename"] = clinical_filename
    parameters["target"] = target
    parameters["TENSOR_FAC"] = TENSOR_FAC
    parameters["TENSOR_RANK"] = TENSOR_RANK
    parameters["TENSOR_METHOD"] = TENSOR_METHOD
    parameters["REGRESS"] = REGRESS
    parameters["EEG_FORMAT"] = EEG_FORMAT
    parameters["MAX_NT"] = MAX_NT
    parameters["RESAMPLING"] = RESAMPLING
    parameters["SLEEP_STAGE"] = SLEEP_STAGE
    parameters["PREPROCESSING_METHOD"] = PREPROCESSING_METHOD
    parameters["vbias"] = vbias
    parameters["gain"] = gain
    parameters["nnodes"] = nnodes
    parameters["density"] = density
    parameters["feature_scaling"] = feature_scaling
    parameters["emucore"] = emucore
    parameters["covariates"] = covariates
    parameters["channels"] = channels

    return parameters    


#------------------------------------------
# Read eeg data from clinical file
#------------------------------------------
def read_clinical_data(parameters):
    
    # Get needed parameters
    filename = parameters["clinical_filename"]

    #annotations=['Index','Diagnosis','ID','DX','Age','Sex','Sleep','Spikes','Epilepsy']    
    
    # Let's make sure white space at the end of NaN's won't confuse the reader
    additional_nans = ['NaN ', 'nan ', 'na ', 'inf', 'inf ']

    # Read the file, then extract the desired columns if they exist
    Y_df = pd.read_csv(filename, skipinitialspace=True, na_values=additional_nans)   
            
    return Y_df


#------------------------------------------
# Read all data
#------------------------------------------
def read_eeg_files(Y_df, parameters):

    # Get needed parameters
    all_file_names = parameters["all_file_names"]    
    EEG_FORMAT = parameters["EEG_FORMAT"]    
    RESAMPLING = parameters["RESAMPLING"]    
    MAX_NT = parameters["MAX_NT"]
    TENSOR_FAC = parameters["TENSOR_FAC"]
    TENSOR_RANK = parameters["TENSOR_RANK"]

    # Read the eeg data and the clinical data, then merge
    glob_list = glob.glob(all_file_names)
    
    if EEG_FORMAT == "edf":
        eeg_df, X = read_edf_files( glob_list,parameters) 
           
    elif EEG_FORMAT == "mat":
        resampling_rate = RESAMPLING
        eeg_df, X = read_egi.read_matlab(glob_list, MAX_NT, resampling_rate)

    elif EEG_FORMAT == "egi":
        resampling_rate = RESAMPLING
        eeg_df, X = read_egi.read_edf(glob_list, MAX_NT, resampling_rate)
   
    X_indices = eeg_df.Index.tolist()    
    print("eeg_df columns: ", list(eeg_df.columns),"; len = ", len(eeg_df))
    print("Y_df columns: ", list(Y_df.columns),"; len = ", len(Y_df))
    #print("Y_df columns: ", list(Y_df.columns),"; len = ", len(Y_df))
    print("Using tensor factorization? ", TENSOR_FAC,"; rank = ", TENSOR_RANK)

    print("Len of eeg_df, Y_df: ", len(eeg_df), len(Y_df))
    df = pd.merge(eeg_df, Y_df, on=['Index'])
    print("merged len df, X = ", len(df), len(X))
    print("Columns after merge: ", list(df.columns),", len = ", len(df))
    
    return df, X, X_indices
            
# ------------------------------------------------------------------------------
# Choose selected EEG channels
# ------------------------------------------------------------------------------
def update_channel_names(df):
    
    channelNames = list(df.columns)
        
    # Change old names to new: T3->T7, T5->P7, T4->T8, T6->P8
    for c, ch in enumerate(channelNames):
        
        df.columns = ['T7' if x=='T3' else x for x in df.columns]
        df.columns = ['P7' if x=='T5' else x for x in df.columns]
        df.columns = ['T8' if x=='T4' else x for x in df.columns]
        df.columns = ['P8' if x=='T6' else x for x in df.columns]
        
        # And fix case
        df.columns = ['Fp1' if x=='FP1' else x for x in df.columns]
        df.columns = ['Fp2' if x=='FP2' else x for x in df.columns]
         
        # And a minor substitution
        df.columns = ['Fp1' if x=='AF3' else x for x in df.columns]
        df.columns = ['Fp2' if x=='AF4' else x for x in df.columns]
        
    return df

# ------------------------------------------------------------------------------
# Read eeg data from all files
# ------------------------------------------------------------------------------
def read_edf_files(all_files, params, resampling_rate=200):
    
    max_time = params["MAX_NT"]    
    resampling_rate = params["RESAMPLING"]
    target = params["target"]
    offset = 0 # place to start reading from EEG file, in seconds
    
    # Read each file and place in a data dictionary structure
#    picks_eeg = ['Fp1','Fp2']
#    picks_eeg = ['T7', 'T8']
#    picks_eeg = ['T7','T8','C3','C4','F3','F4']
#    picks_eeg = ['C3', 'C4' ,'O1', 'O2', 'F3', 'F4']
#    picks_eeg = ['Fp2', 'C3', 'C4' ,'O1', 'O2', 'F3', 'F4', 'F7', 'F8', 'Fz' ,'P3', 'P4', 'Pz','T7', 'T8', 'P7' ,'P8']
    picks_eeg = ['Fp1', 'Fp2', 'C3', 'C4' ,'O1', 'O2', 'F3', 'F4', 'F7', 'F8', 'Fz' , 'P3', 'P4', 'Pz','T7', 'T8', 'P7' ,'P8']
#    picks_eeg = ['C3', 'C4' ,'O1', 'O2', 'F3', 'F4', 'F7', 'F8', 'Fz' , 'P3', 'P4', 'Pz']

    #picks_eeg = [ 'O1', 'O2', 'F7', 'F8']
 
    if target.lower() == "spikes": # or target.lower() == "sleep":
        picks_eeg_ied = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2']   
        picks_eeg = picks_eeg_ied 
    elif target.lower() == "absence":
        #picks_eeg = ['Fp1', 'Fp2', 'C3', 'C4' ,'O1', 'O2', 'F3', 'F4', 'F7', 'F8', 'Fz' , 'P3', 'P4', 'Pz','T7', 'T8', 'P7' ,'P8']
        picks_eeg = ['Fp1', 'Fp2', 'C3', 'C4' ,'O1', 'O2', 'F3', 'F4', 'F7', 'F8', 'Fz' , 'P3', 'P4', 'Pz','T7', 'T8', 'P7' ,'P8']
    elif target.lower() == "bects":
        picks_eeg = ['C3', 'C4' ,'O1', 'O2', 'F3', 'F4', 'F7', 'F8', 'Fz' , 'P3', 'P4', 'Pz','T7', 'T8', 'P7' ,'P8']
        picks_eeg = ['C3', 'C4' ,'O1', 'O2', 'F3', 'F4', 'F7', 'F8', 'Fz' , 'P3', 'P4', 'Pz']
        #picks_eeg = ['Fp1', 'Fp2', 'C3', 'C4' ,'O1', 'O2', 'F3', 'F4', 'F7', 'F8', 'Fz' , 'P3', 'P4', 'Pz','T7', 'T8', 'P7' ,'P8']
        #picks_eeg = ['T7','T8','C3','C4','F3','F4']

    
    picks_eeg = params["channels"]
    
    X = []
    index_array = []
    Channels = []
    srate = []
    count = 0
    ch_set = set(picks_eeg)
    
    print("Total number of filenames to read = ", len(all_files))
    for f, filename in enumerate(all_files):
        basename = os.path.basename(filename)
        #print("Processing file: ", basename)
        success = True
        try:
            raw = read_raw_edf(filename, preload=True, verbose='ERROR')
            sfreq = int(raw.info['sfreq'])
            if sfreq < resampling_rate:
                success = False
                #print("Freq too low: ",sfreq, basename)
        except:
            print("Cannot process file: %s" %(basename), "; skipping to next file")
            success = False

        if success:
            # Does this file have the required channels?
            df = raw.to_data_frame()   
            
            # Some name changes
            update_channel_names(df)
    
            col_list = [s for s in list(df.columns) if (s in picks_eeg)]
            df = df[col_list]
            channel_names = list(df.columns) # raw.ch_names  
                                    
            if set(channel_names) == set(picks_eeg):            
                # Always use an average EEG reference for dynamical analysis
                #raw.set_eeg_reference('average')
                count += 1
                
                id_full = pathlib.Path(filename).stem
                id = id_full.partition(".")[0]
                index_array.append(id)            
                #print(id, channel_names)
                                                   
                sfreq = int(raw.info['sfreq'])
                srate.append(sfreq)
                if sfreq > resampling_rate:
                    raw.resample(sfreq=resampling_rate)
                    sfreq = resampling_rate
    
                nt = int(sfreq*max_time)
                            
                x = df.transpose().to_numpy()[0:, offset: offset+nt]
                X.append(x)
                Channels.append(channel_names)
                
            else:
                these_channels = set(channel_names)
                missing_set = ch_set - these_channels
                print("%d channels; Cannot process file: %s" %(len(channel_names),basename), "; missing channels: ", missing_set)
        
    if count == 0:
        print("No files processed, exiting")
        exit()
    else:
        print(count," files processed.")
                
    X = np.array(X)
    
    # dictionary of lists 
    dict = {'Index': index_array, 'Channels': Channels, 'srate': srate} 
    df_data = pd.DataFrame(dict)

    return df_data, X


#--------------------------------------------------------------------
# Tensor factorization
#--------------------------------------------------------------------
def tensor_factorization(df, params, resp_train, resp_test, Y, train_index, test_index):

    # Get a few parameters
    TENSOR_RANK = params["TENSOR_RANK"]
    TENSOR_METHOD = params["TENSOR_METHOD"]

    rank = TENSOR_RANK
    pySCP = pySupCP.pySupCP(rank)
    kwargs = {'AnnealIters': 100, 'ParafacStart': 1, 'max_niter': 5000, 'convg_thres': 0.01, 'Sf_diag': 1}
    cp_regress = CPRegressor(rank, verbose=0)        

    u_resp_train = resp_train.copy()
    u_resp_test = resp_test.copy()
    
    # Labels
    labels = set(Y)
    
    # Covariates
    Indices = df.Index.unique()
    N = len(Indices)
    Age = np.zeros(N)
    Sex = np.zeros(N)
    Bects = np.zeros(N)
    
    for i, index in enumerate(Indices):
        df_ind = df[(df["Index"] == index)]
    
        Age[i] = df_ind.Age.unique()
        Sex[i] = df_ind.Sex.unique()     
        if "BECTS" in list(df_ind.columns):
            Bects[i] = df_ind.BECTS.unique()  
        
    # Include additional covariates in the training phase
    Possible_Additional_Features = params["covariates"]
    ADDITIONAL_FEATURES = []
    headers = list(df.columns.values)    
    for f in Possible_Additional_Features:
        if f in headers:
            ADDITIONAL_FEATURES.append(f)
            
    q = 1 + len(ADDITIONAL_FEATURES)
    n1 = len(train_index)
    n2 = len(test_index)
    Y_train = Y[train_index]
    cov_train = np.zeros((n1,q))         
    cov_test = np.zeros((n2,q))         
    cov_train[0:,0] = Y[train_index]
    cov_test[0:,0] = Y[test_index]
        
    for f, feat in enumerate(ADDITIONAL_FEATURES):
        if "Age" == feat:
            # Include patient age as a feature
            cov_train[0:,f+1] = Age[train_index]
            cov_test[0:,f+1] = Age[test_index]

        elif "Sex" == feat:
            cov_train[0:,f+1] = Sex[train_index]
            cov_test[0:,f+1] = Sex[test_index]
                        
        elif "BECTS" == feat:
            cov_train[0:,f+1] = Bects[train_index]
            cov_test[0:,f+1] = Bects[test_index]
                        
    # Get latent features using tensor factorization
    if TENSOR_METHOD == "tensorly":
        cp_regress.fit(u_resp_train, Y_train)
        (weights, factors) = cp_regress.cp_weight_
        prob_array = np.zeros(len(test_index))        

    elif TENSOR_METHOD == "supcp":
        (B, factors, U_train, se2, Sf, rec) = pySCP.fit(u_resp_train, cov_train, kwargs)  
        prob_array = np.zeros(len(test_index))        
        for k in range(len(cov_test)):
            yi = cov_test[k:k+1,:]
            prob_array[k] = pySCP.classify(u_resp_test, yi, labels)

    U_train = get_weights(u_resp_train, factors, rank)
    U_test = get_weights(u_resp_test, factors, rank)
    
    n_cov = len(ADDITIONAL_FEATURES)
    if n_cov > 0:
        U_train = np.append(U_train, cov_train[0:,1:], axis=1)
        U_test = np.append(U_test, cov_test[0:,1:], axis=1)

    return U_train, U_test, prob_array
    
#--------------------------------------------------------------------
# Classification
#--------------------------------------------------------------------
def classify(df,resp, Y, params, skf_splits):
    
    # Get a few parameters
    TENSOR_FAC = params["TENSOR_FAC"]
    TENSOR_RANK = params["TENSOR_RANK"]
    TENSOR_METHOD = params["TENSOR_METHOD"]
    TARGET = params["target"]
    
    print("Tensors?, method, rank:", TENSOR_FAC,TENSOR_METHOD, TENSOR_RANK)
    
    Ypred = np.zeros(Y.shape)
    Yprob = np.zeros(Y.shape)
        
    # Make some arrays to keep track of ID,Truth, proba
    N = len(df)
    id_array = df.Index.unique()
    truth_array = {}
    for id in id_array:
        truth_array[id] = df[[x == id for x in df.Index]].Label.unique()[0]

    # Covariates
    a = df.Age.to_numpy()
    Age = np.zeros((len(a),1))
    Age[0:,0] = a[0:]
    s = df.Sex.to_numpy()
    Sex = np.zeros((len(s),1))
    Sex[0:,0] = s[0:]
    if "Diagnosis" in list(df.columns):
        d = df.Diagnosis.to_numpy()
        Diag = np.zeros((len(d),1))
        Diag[0:,0] = d[0:]

    if "absence_epilepsy_type" in list(df.columns):
        d = df.absence_epilepsy_type.to_numpy()
        Type = np.zeros((len(d),1))
        Type[0:,0] = d[0:]

    if "Meds" in list(df.columns):
        d = df.Meds.to_numpy()
        meds = np.zeros((len(d),1))
        meds[0:,0] = d[0:]
    
    if "Epilepsy" in list(df.columns):
        d = df.Epilepsy.to_numpy()
        epilepsy = np.zeros((len(d),1))
        epilepsy[0:,0] = d[0:]
    
    # Set up the classifiers
    # Create the stratified splits
    Ylist = list(Y)
    labels = list(set(Ylist))
    n0 = Ylist.count(labels[0])
    n1 = Ylist.count(labels[1])
    kfolds = 5  # default
    kfolds = min(kfolds, n0, n1) # must have more labels than folds
    skf = StratifiedKFold(kfolds)   
    
    # Some classifiers
    knn = KNeighborsClassifier(9)
    RF = RandomForestClassifier(max_depth=3, n_estimators=11, max_features=3)
    GB = GradientBoostingClassifier(max_depth=3, n_estimators=21)
    ada = AdaBoostClassifier(n_estimators=50, learning_rate=1.0)
    svmrbf = SVC(gamma='auto', kernel='rbf', probability=True)#, C=0.025)]

    clf_array = [RF,GB,ada,knn,svmrbf]
    clf_names = ["RF","GB","ada","knn","svmrbf"]    
 
    # 3. Classification. Loop through the mapping from output layer to labels
    # Loop through training and testing data and run the features through the reservoir        
    print("\n%6s %6s %6s %6s %6s %6s %6s %6s %6s (%4s,%4s), (%4s,%4s), %8s, %10s" % ("CLF","AUC","Recall","Precis", "Sens", "Spec","PPV","NPV","F1","mean","std","mean","std","P-val","Severity_R"))

    train_index_array = []
    test_index_array = []
    resp_train_array = []
    resp_test_array = []
    Y_train_array = []
    Y_test_array = []

    if TENSOR_FAC:        
        U_train_array = []
        U_test_array = []
        prob_array = []
        for train_index, test_index in skf.split(X, Y):
            train_index_array.append(train_index)
            test_index_array.append(test_index)

            resp_train = resp[train_index]
            resp_test = resp[test_index]
            Y_train = Y[train_index]
            Y_test = Y[test_index]
            U1, U2, prob = tensor_factorization(df, params, resp_train, resp_test, Y, train_index, test_index)
            
            prob_array.append(prob)
            U_train_array.append(U1)
            U_test_array.append(U2)
            resp_train_array.append(resp_train)
            resp_test_array.append(resp_test)
            Y_train_array.append(Y_train)
            Y_test_array.append(Y_test)
            
    else:
        for train_index, test_index in skf.split(X, Y):
            resp_train = resp[train_index]
            resp_test = resp[test_index]
            Y_train = Y[train_index]
            Y_test = Y[test_index]
            
            train_index_array.append(train_index)
            test_index_array.append(test_index)
            resp_train_array.append(resp_train)
            resp_test_array.append(resp_test)
            Y_train_array.append(Y_train)
            Y_test_array.append(Y_test)
        

    for i,clf in enumerate(clf_array):
        # Test all classifiers
        clfname = clf_names[i]
        means = {labels[0]:[], labels[1]:[]}
        
        fold = 0
        for train_index, test_index in skf.split(X, Y):
                               
            resp_train = resp_train_array[fold] # resp[train_index]
            resp_test = resp_test_array[fold] # resp[test_index]
            Y_train = Y_train_array[fold] #Y[train_index]
            Y_test = Y_test_array[fold] #Y[train_index]
                        
            if TENSOR_FAC:
                U_train = U_train_array[fold]
                U_test = U_test_array[fold]   
                #U_test = np.append(U_test, Y_test[0:,1:],axis=1)

                if clf == "supCP":                    
                    yp = np.zeros((len(U_test),2))
                    yp[:,1] = prob_array[fold]
                    # Convert probabilities to binary 0 or 1
                    Ypred[test_index] = [int(p > 0.5) for p in prob_array[fold]]
                else:
                    model = clf.fit(U_train,Y_train)
                    Ypred[test_index] = model.predict(U_test)
                    try:
                        yp = model.predict_proba(U_test) 
                    except:
                        yp = np.ones((len(U_test),2))
                    
            else:                
                (x1,x2,x3) = resp_train.shape
                resp_train = resp_train.reshape(x1,x2*x3)
                (x1,x2,x3) = resp_test.shape
                resp_test = resp_test.reshape(x1,x2*x3)
                                
                model = clf.fit(resp_train,Y_train)
                Ypred[test_index] = model.predict(resp_test)
                try:
                    yp = model.predict_proba(resp_test)  
                except:
                    yp = np.ones((len(resp_test),2))   
                    
            Yprob[test_index] = yp[:,1]                 
            fold += 1
                
            for i in test_index:
                truth = int(Y[i])
                p = Yprob[i]
                means[truth].append(p)

        # Print the scores for all
        print_all_scores = False
        if print_all_scores:
            m0 = 0.0
            m1 = 0.0
            n0 = 0
            n1 = 0
            N = len(Y)
            for i in range(N):
                id = id_array[i]
                truth = Y[i]
                if truth == 0:
                    #print("id, truth, score: ", id, Y[i], Yprob[i])
                    n0 += 1
                    m0 += Yprob[i]
            m0 = m0/n0
                    
            #print("\n")
            for i in range(N):
                id = id_array[i]
                truth = Y[i]
                if truth > 0:
                    #print("id, truth, score: ", id, Y[i], Yprob[i])
                    n1 += 1
                    m1 += Yprob[i]
            m1 = m1/n1
            
            #print("m0, m1: ", m0, m1)

        if (1==0) and clfname == "GB" and TARGET=="absence" and "Meds" in list(df.columns):
            print("%12s, %6s, %6s, %6s, %6s" %("ID","AGE","Abs","Meds","Score"))
            for i,id in enumerate(id_array):
                df_id = df.loc[df['Index'] == id]
                age = df_id.Age.unique()
                abs = df_id.Absence.unique()
                meds = df_id.Meds.unique()[0]
                truth = Y[i]
                pred = Ypred[i]
                score = Yprob[i]
                
                if pred != truth:
                    print("%12s, %6.1f, %6d, %6s, %6.2f " %(id,age,abs,meds,score))
                
        # How did we do?
        labels = list(df.Label.unique())
        labels.sort()
        nlabels = len(labels)
        lab0 = labels[0]
        lab1 = labels[1]
        s0 = means[lab0]
        s1 = means[lab1]    
        m0, std0 = (np.mean(s0), np.std(s0))
        m1, std1 = (np.mean(s1), np.std(s1))
        t, p = stats.ttest_ind(s0, s1)
                
        if nlabels == 2:
            #brier = brier_score_loss(Y, Yprob, pos_label=labels[1])
            auc, recall, precision, sens, spec, F1, ppv, npv = get_auc(Y, Ypred, Yprob, labels)
            data_text = ("%6s %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f  %6.2f  (%4.2f,%4.2f),  (%4.2f,%4.2f),  %8.2e " % (clfname,auc, recall, precision, sens, spec, ppv, npv,F1,m0,std0,m1,std1,p))
            print(data_text)
        else:
            print("Currently no stats for > 2 labels")
                

# ------------------------------------------------------------------------------
# Select subsets for special cases, labels, etc.
# This section is currently messy, but is flexible for research
# ------------------------------------------------------------------------------
def assign_labels(df, params, X=None, X_indices=None):
    # Get needed parameters
    SLEEP_STAGE = params["SLEEP_STAGE"]
    target = params["target"]
    emucore = params["emucore"]
           
    # This step will apply only to csv files
    if not emucore:
        df = df[[(not  x.startswith('s_')) for x in df.Feature]]
    
    # Select the labels for classification
    if target.lower() == "sleep":
        df = df[[((x == 0) or (x == 2)) for x in df.SleepStage]]  # Sleep stage 0 or 2
        df["Label"] = df.SleepStage.copy()
        df["DX"] = df.SleepStage.copy()

        
    elif target.lower() == "bects":        
        df = df[[ (x.startswith('B') or x.startswith('D')) for x in df.Index]]  # Diagnostic group
        df = df[[x == SLEEP_STAGE for x in df.SleepStage]]  # Sleep stage 0 or 2        
        list_of_labels = df.BECTS.tolist().copy()
        df["DX"] = list_of_labels
        df["Label"] = list_of_labels
        
    else:
        print("Not set up for problem == ", target)
        exit()
        
    # Select rows from X to keep
    if X_indices != None:
        X_new = []
        X_new_indices = []
        selected_indices = df.Index.tolist()
        for i, ind in enumerate(X_indices):
            if ind in selected_indices:
                X_new.append(X[i])
                X_new_indices.append(ind)
        X = np.array(X_new)
    
    # Process the data
    Y = df.Label.to_numpy()
    
    return df, X, Y
        
# ------------------------------------------------------------------------------
# Main driver
# ------------------------------------------------------------------------------
if __name__ == '__main__':

    # Read parameter file
    params = read_parameters(sys.argv)    
    
    # Read the clinical data file
    df_clinical = read_clinical_data(params)
    
    # Create the cross validation splits
    kfolds = 5
    skf = StratifiedKFold(kfolds)   
    
    # --------------------
    # Read nonlinear EEG features previously computed
    # --------------------
    csv_file = params["all_file_names"]
    df = read_csv_features.read_csv_file(csv_file, df_clinical)
    
    # Assign labels, select subsets, etc based on specified target
    df, blank1, blank2 = assign_labels(df, params)
    
    eeg_feature_tensor, Y = read_csv_features.create_tensor(df, params)
    X = eeg_feature_tensor
    
    # Create the stratified splits (may or may not be used here)
    skf_splits = skf.split(X, Y)
                            
    # Classification or regression
    classify(df,eeg_feature_tensor, Y, params, skf_splits)
                
