"""
function [B,V,U,se2,Sf,rec]=SupParafacEM(Y,X,R,args)
% EM algorithm to fit the SupCP model:
% X=U X V1 X ... X VK + E 
% U=YB + F 
%
%
% Input
%       Y           n*q full column rank response matrix (necessarily n>=q)
%                  
%        
%       X           n*p1*...*pK design array
%      
%       R          fixed rank of approximation, R<=min(n,p)   
%
%       args       struct parameter with optional additional parameters
%                  args = struct('field1',values1, ...
%                            'field2',values2, ...
%                            'field3',values3) .
%                  The following fields may be specified:
%
%           AnnealIters:  Annealing iterations (default =100)
%
%           fig: binary argument for whether log likelihood should be
%           plotted at each iteration (default = 0)
%
%           ParafacStart:binary argument for whether to initialize with
%           Parafac factorization (default = 0)
%
%           max_niter: maximum number of iterations (default = 1000)
%
%           convg_thres: convergence threshold for difference in log
%           likelihood (default = 10^(-3))
%
%           Sf_diag: whether Sf is diagnal (default =1, diagonal)
%
% Output
%       B           q*r coefficient matrix for U~Y, 
%                   
%       V           list of length K-1. V{k} is a pXr coefficient matrix 
%                    with columns of norm 1; these are the loading matrices for each mode (except mode 1, which are the samples)
%
%       U           Conditional expectation of U: nXr; latent score matrix for the samples
%
%       se2         scalar, var(E)
%       Sf          r*r diagonal matrix, cov(F)
%  
%       rec        log likelihood for each iteration
%      
%
% Created by: Eric Lock (elock@umn.edu) and Gen Li

Python version written by William J. Bosl, 2022.
Updates:
    
"""

import sys
import numpy as np
from scipy import stats as sps
import math
from tensorly.decomposition import parafac
from tensorly.regression.cp_regression import CPRegressor
import tensorly
from numpy.lib import scimath as SM
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
#from scipy.optimize import curve_fit
#from scipy.misc import derivative
from scipy.optimize import minimize


#--------------------------------------------------
# Initial the class
#--------------------------------------------------
class pySupCP:
    
    #----------------------------------------------
    #----------------------------------------------
    def __init__(self, R):
        # Computed SupCP model parameters
        self.R = R
        self.B = None
        self.V = None
        self.U = None
        self.se2 = None
        self.Sf = None
        self.rec = None
        self.Vmat = None
        self.F = None
        
    #----------------------------------------------
    #----------------------------------------------
    def get_model(self):
        # Create a database object for pickling
        db = {}
        db['R'] = self.R
        db['B'] = self.B
        db['U'] = self.U
        db['V'] = self.V
        db['se2'] = self.se2
        db['Sf'] = self.Sf
        db['rec'] = self.rec
             
        # Create a pickle database file and return
        return pickle.dumps(db)          


    #----------------------------------------------
    #----------------------------------------------
    def get_model_params(self):
        return (self.B,self.V,self.U,self.se2,self.Sf,self.rec)
     
    #----------------------------------------------
    #----------------------------------------------
    def set_model(self, db):
        # Create a database object for pickling
        self.R = db['R']
        self.B = db['B']
        self.U = db['U'] 
        self.V = db['V'] 
        self.se2 = db['se2'] 
        self.Sf = db['Sf'] 
        self.rec = db['rec'] 
             
    #----------------------------------------------
    #----------------------------------------------
    def write_model(self, filename):
        # Create a database object for pickling
        db = {}
        db['R'] = self.R
        db['B'] = self.B
        db['U'] = self.U
        db['V'] = self.V
        db['se2'] = self.se2
        db['Sf'] = self.Sf
        db['rec'] = self.rec
        
        # Open a file for writing. 
        # Its important to use binary mode
        dbfile = open(filename, 'ab')
     
        # Write to the specified file: source, destination
        pickle.dump(db, dbfile)                    
        dbfile.close()
     
    #----------------------------------------------
    #----------------------------------------------
    def read_model(self, filename):
        # Read model parameters from a pickled file
        # for reading also binary mode is important
        dbfile = open(filename, 'rb')    
        db = pickle.load(dbfile)
        dbfile.close()
        self.R = db['R']
        self.B = db['B']
        self.U = db['U']
        self.V = db['V']
        self.se2 = db['se2'] 
        self.Sf = db['Sf']
        self.rec = db['rec'] 


    #----------------------------------------------
    # This function will accept new tensor data
    # and use it to update an existing model
    #----------------------------------------------
    def update_model(self, Y_new, X_new, R_new, kwargs=None):
        a = Y_new

    #----------------------------------------------
    ##### DS comments
    ##### function to calculate array descriptive statistics
    #----------------------------------------------
    def descriptive_stats(self,arr, ax = None, rf = 6):
        ct = len(arr)
        mean = np.round(np.mean(arr, axis=ax), rf)
        median = np.round(np.median(arr, axis = ax), rf)
        min, max = np.round(np.min(arr), rf+1), np.round(np.max(arr), rf+1)
        var = np.round(np.var(arr, axis=ax), rf)
        sd = np.round(np.std(arr, axis=ax), rf)
        return {'count':ct, 'min':min, 'max':max, 'mean':mean, 'median':median, 'var':var, 'sd':sd}

    #----------------------------------------------
    #----------------------------------------------
#    def calc_ds(self,arr): ### wrapper function around descritive_stats()
#        ds = {'count':[], 'min':[], 'max':[], 'mean':[], 'median':[], 'var':[], 'sd':[]}
#        for v in np.array(np.array(arr).T):
#            st = self.descriptive_stats(v)
#            for k in ds.keys():
#                ds[k].append(st[k])
#        df_ds = pd.DataFrame(ds)
#        return df_ds        
    #####
    
    #----------------------------------------------
    #----------------------------------------------
    def normc(self,m):
        m = m.astype(float)
        for c in range(m.shape[1]):
            col = m[:,c]
            d = np.sqrt(np.square(col).sum())
            m[:,c] = col/d
        return m
    
    
    #----------------------------------------------
    #----------------------------------------------
    def TensProd(self,L, cols=None):
        K = len(L)
        m = []
        for i in range(K):
            [mi, R] = list(L[i].shape)
            m.append(mi)
        # print(f'm: {m}, R: {R}')
        if cols != None: # check
            if max(cols) > R:
                print('Range exceeds rank!')
                sys.exit()
            # customize L
            # DS comment: changed from newL = [] to newL = L.copy()
            newL = L.copy()
            for i in range(K):
                # DS comment - changed the below line
                newL[i] = L[i][:,cols]
                #newL.append(L[i][:,cols])
            # DS comment: changed from L = newL to L = newL.copy()
            L = newL
    
        L = np.array(L, dtype=object)
    
        order = np.array(range(K-1,0,-1))
        tempL = L[order]
     
        krL = tensorly.kr(tempL)
    
        matX = np.array(L[0] @ krL.conj().T)
    
        ans = matX.real.reshape(m, order='F')
        return ans
    
    #--------------------------------------------
    # Use SupParafacEM to fit model to data
    #--------------------------------------------
    def fit(self, X, Y, kwargs=None):
    
        R = self.R    
        AnnealIters = 100   # default to 100 annealing iterations
        fig = 0             # 1=plot likelihood at each iteration; 0=no show
        ParafacStart = 0    # 0 = random start; 1 = parafac start
        max_niter=20000      # maximum number of iterations
        convg_thres = 10**(-2)   # for log likelihood difference
        Sf_diag=1           # This was added by DS, coming from kwargs
    
#        print("Starting SupParafac function in Python... ")
    
        #Update args if specified
        if kwargs != None:
    
          if 'AnnealIters' in kwargs:
            AnnealIters = kwargs['AnnealIters']
    
          if 'fig' in kwargs:
            fig = kwargs['fig']
    
          if 'ParafacStart' in kwargs:
            ParafacStart = kwargs['ParafacStart']
    
          if 'max_niter' in kwargs:
            max_niter = kwargs['max_niter']
    
          if 'convg_thres' in kwargs:
            convg_thres = kwargs['convg_thres']
    
          if 'Sf_diag' in kwargs:
            Sf_diag = kwargs['Sf_diag']
    
        ##### DSY comment
        #print(f'AnnealIters: {AnnealIters}, fig: {fig}, ParafacStart: {ParafacStart}, max_niter: {max_niter}, convg_thres: {convg_thres}, Sf_diag: {Sf_diag}\n\n')
        #####
    
        # resetting of input parameters
        (n1,q) = Y.shape    # 72 x 3 for cross validation and 90 x 3 otherwise
        m = X.shape         #
        n = m[0]            # sample size under test/train
        L = len(m)          # number of modes (as in multi-modal data), in case of BECTS, it's 4
        #K = L-1             # 
        p = np.prod(m[1:L]) # p1*p2*...*pK           % GL: potentially very large
    
        # Pre-Check
        if (n != n1):
            print('X does not match Y! exit...')
            sys.exit()
        elif (np.linalg.matrix_rank(Y) != q):
            print('Columns of Y are linearly dependent! exit...')
            print(Y)
            sys.exit()
    
        Index = list(range(0,L))
        IndexV = list(range(0,L-1))
    
        # Annie add random seed
        # print('Annie add random seed - python')
        #print('using random num generator')
        np.random.seed(1)
    
    
        # initialize via parafac
        V = []
        test = []
        test_norm = []
        if(ParafacStart):   #this is initiated as ParafacStart = 0, and never changed in further code
#            print("... Parafac initialization")
            (weights, factors) = parafac(X,R)
            for i in range(1,L):
                V.append(factors[i])
        else:
          for l in range(1,L):
            # Annie changed the random num generator
            test.append(sps.norm.ppf(np.random.rand(R, m[l])).T)
            test_norm.append(self.normc(sps.norm.ppf(np.random.rand(R, m[l])).T))
            V.append(self.normc(sps.norm.ppf(np.random.rand(R, m[l])).T))
    
        Vmat = np.zeros((p,R)) # a very long matrix (p can be very large)     
        for r in range(R):      # if rank is 10, this r goes from 0 to 9
            Temp = self.TensProd(V, [r])
            Vmat[:,r] = Temp.flatten(order='F')

        Xmat = np.moveaxis(X, 0, -1)  # move the first dimension, n, to the last; Xmat is a matricization of X samples
        Xmat = Xmat.reshape(-1, n ,order='F')
        
        U = Xmat.conj().T @ Vmat
            
        UV = [U] + V
        
        tp_UV = self.TensProd(UV)
    
        E = X - tp_UV
        se2 = np.var(E.flatten(order='F'), ddof=1)
    
        Yp = Y.conj().T
        Ytemp = np.linalg.inv(Yp @ Y) @ Yp
        B = Ytemp @ U
    
        UYB = U - Y@B
        sf_temp = (UYB.conj().T @ UYB)/n

        if Sf_diag:
            Sf = np.diag(np.diag(sf_temp))  # R*R, diagonal
        else:
            Sf = sf_temp
    
        # Compute determinant exactly, using Sylvester's determinant theorem
        # https://en.wikipedia.org/wiki/Determinant#Properties_of_the_determinant
        sqrtSf = SM.sqrt(Sf)
        MatForDet = (sqrtSf @ Vmat.conj().T @ Vmat @ sqrtSf / se2) + np.eye(R)  # R X R
        logdet_VarMat = 2.0 * np.sum(np.log(np.diag(np.linalg.cholesky(MatForDet)))) + p * math.log(se2)
        ResidMat  = Xmat.conj().T - (Y @ B @ Vmat.conj().T)
    
        if Sf_diag:
            Sfinv = np.diag(1./np.diag(Sf))
        else:
            Sfinv = np.linalg.inv(Sf)
    
        term1 = (1/se2) * (ResidMat @ ResidMat.conj().T).trace()
        term2 = Vmat.conj().T @ ResidMat.conj().T @ ResidMat @ Vmat
        term3 = np.linalg.inv(Sfinv + (1/se2) * (Vmat.conj().T @ Vmat)) 
        Trace = term1 - (1/(se2*se2)) * (term2 @ term3).trace()
        logl = (-n/2) * (logdet_VarMat) - 0.5*Trace
        rec=[logl]
    
        niter=1
        Pdiff = convg_thres+1
        
        while(niter<=max_niter and (abs(Pdiff)>convg_thres)):
            niter=niter+1
    
            # record last iter
            logl_old = logl
            #### DS comments
            #se2_old=se2;
            #Sf_old=Sf;
            #Vmat_old = Vmat;
            #V_old=V;
            #B_old=B;
            #####
            
            #E step
            if Sf_diag:
                Sfinv = np.diag(1./np.diag(Sf))
            else:
                Sfinv = np.linalg.inv(Sf)
    
            weight = np.linalg.inv(Vmat.conj().T @ Vmat + se2*Sfinv) # r*r
    
            cond_Mean = (se2 * Y @ B @ Sfinv + Xmat.conj().T @ Vmat) @ weight  # E(U|X), n*r
            U = cond_Mean
    
            cond_Var = np.linalg.inv( (1/se2) * Vmat.conj().T  @ Vmat + Sfinv) # cov(U(i)|X), r*r
    
            # Add noise to the conditional mean of U.
            # Variance of noise is a decreasing percentage of the variance of the
            # true conditional mean.
            # if(niter<AnnealIters): # this line needs to be uncommented to add noise
            #     anneal = (AnnealIters-niter)/AnnealIters # this line needs to be uncommented to add noise
            #     diag = anneal*np.diag(U.var(0,ddof=1)) #% this line needs to be uncommented to add noise
            #     #diag = anneal*np.diag(np.var(U.flatten(order='F'),ddof=1))
    
            #     for i in range(R): #% this line needs to be uncommented to add noise
            #         U[i,:] = np.random.multivariate_normal(cond_Mean[i], diag) #% this line needs to be uncommented to add noise
            #         #U = mvnrnd(cond_Mean,anneal*diag(var(U)));
            #     #print("Shape of cond_Mean, diag: ", cond_Mean.shape, diag.shape)
            #     #U = np.random.multivariate_normal(cond_Mean, diag)
    
            cond_Mean = U
            cond_quad = n*cond_Var + U.conj().T @ U  # E(U'U|X)  r*r
    
            # Estimate V's
            for l in range(1,L): # it means l can take values (1, 2, 3) in case of BECTS data
                # Create some index sets; note the copy() for each l in this loop
                ind = Index.copy()  # [0, 1, 2, 3]
                del ind[l]          # we're deleting ind[1] when l = 1, so new ind becomes [0, 2, 3]
                ind_l = ind + [l]   # ind_l becomes [0, 2, 3] + [1] = [0, 2, 3, 1]
                indV = IndexV.copy() # (0, 1, 2)
                del indV[l-1]       # delete indV = 0, new indV = [1, 2]
    
                ResponseMat = X.transpose(ind_l).reshape(-1,m[l], order='F')
    
                tempI = list(m).copy()  
                del tempI[l]            
                PredMat = np.zeros((np.prod(tempI),R))
                del tempI[0]            
                VParams = np.zeros((np.prod(tempI),R))
    
    
                #IndexV = list(range(0,L-1))
                #% V{IndexV~=(l-1)}
                
                V_mod = [V[i] for i in indV]
                UV = [U] + V_mod
       
                for r in range(R):
                    Temp = self.TensProd(UV,[r])
                    PredMat[:,r] = Temp.flatten(order='F')
                    
                    if(L==3):
                        x1,x2 = V_mod[0].shape
                        V_new = np.zeros(x1)
                        for kk in range(x1):
                            V_new[kk] = V_mod[0][kk,r]
                        Temp = V_new
                        #Temp = V_mod[0][:,r] 
                        #print("len of V_mod = ", len(V_mod))
                        #print("r,  V_mod[0].shape, Temp.shape: ", r, V_mod[0].shape,Temp.shape)
                        #print('using L = 3 modes')
                    else:
                        Temp = self.TensProd(V_mod,[r])
                        
                    #if(L==3) Temp =           V{IndexV~=(l-1)}(:,r);
                    #else     Temp = TensProd({V{IndexV~=(l-1)}},[r]);               
                    #end

                    VParams[:,r] = Temp.reshape(-1,1, order='F')[:, 0]    

                ##### DS - uncomment these 4 lines to get the original version back
                # mat1 = np.matrix((ResponseMat.conj().T @ PredMat)).T
                # mat2 = np.matrix((np.multiply(VParams.conj().T @ VParams, cond_quad))).T
                # sol = (np.linalg.lstsq(mat1, mat2, rcond=None))[0]
                # V[l-1] = sol
                #####
    
                ##### DS comments - alternate method to solve matrix equation
                # print(f'=====PYTHON Start of Alternative approach to calculating V=====')
                mat2 = np.matrix((ResponseMat.conj().T @ PredMat))
                mat1 = np.matrix((np.multiply(VParams.conj().T @ VParams, cond_quad)))
                sol = (np.linalg.lstsq(mat1.T, mat2.T, rcond=None))[0].T
                V[l-1] = sol
                #####
           
            a = Y.conj().T @ Y
            b = Y.conj().T @ U
            B = np.linalg.lstsq(a,b, rcond=None)[0]        # equivalent to x = A\b in matlab
    
            for r in range(R):
                Temp = self.TensProd(V,[r])
                Vmat[:,r] = Temp.reshape(-1,1, order='F')[:,0] # Vmat(:, r) has Temp reshaped into 0 rows and 1 column
    
            term0 = Xmat - 2*Vmat@cond_Mean.conj().T
            term1 = np.trace( Xmat.conj().T @ term0 )
            term2 = n * np.trace(Vmat.conj().T @ Vmat @ cond_Var)
            term3 = np.trace(cond_Mean @ Vmat.conj().T @ Vmat @ cond_Mean.conj().T)
            se2 = (term1 + term2 + term3) / (n*p)
            
            #estimate diagonal entries for covariance:
            YB = Y @ B
            YBcT = YB.conj().T
            sf_temp = (cond_quad + (YBcT @ YB) - (YBcT @ cond_Mean) - (cond_Mean.conj().T @ YB) )/n
    
            if Sf_diag:
                Sf = np.diag(np.diag(sf_temp))
            else: # estimate full covariance
                Sf = sf_temp
    
            # scaling
            for l in range(1,L):
                V[l-1] = self.normc(V[l-1])
    
            VmatS=Vmat
    
            for r in range(R):
               Temp = self.TensProd(V,[r])
               VmatS[:,r] = Temp.reshape(-1, 1, order='F')[:, 0]
    
            ##### DS comments - changed axis = 0, earlier it was not provided as a parameter
            sum_Vmat_2 = np.sqrt(np.sum(np.square(Vmat), axis = 0))
            Bscaling = np.ones((q,1)) * sum_Vmat_2
            B  = B * Bscaling
            Sfscaling = sum_Vmat_2.conj().T * sum_Vmat_2
            Sf = Sf * Sfscaling
            Vmat=VmatS
                
            # calc likelihood
            if Sf_diag:
                Sfinv = np.diag(1/np.diag(Sf))
            else:
                Sfinv = np.linalg.inv(Sf)
    
            ResidMat = Xmat.conj().T - (Y @ B @ Vmat.conj().T)  # n*p
            Sfsq = SM.sqrt(Sf)
            MatForDet = (Sfsq @ Vmat.conj().T @ Vmat @ Sfsq ) /se2 + np.eye(R) # R X R
            logdet_VarMat = 2.0 * np.sum(np.log(np.diag(np.linalg.cholesky(MatForDet)))) + p*math.log(se2) #uncomment
    
            term1 = (1/se2) * np.trace(ResidMat @ ResidMat.conj().T)
            term2 = Vmat.conj().T @ ResidMat.conj().T @ ResidMat @ Vmat
            term3 = np.linalg.inv(Sfinv + (1/se2)*(Vmat.conj().T @ Vmat))
            Trace = term1 - (1/(se2**2))*np.trace(term2@term3)
            logl = (-n/2) * (logdet_VarMat) - 0.5*Trace #uncomment
            rec.append(logl) #uncomment
    
            # iteration termination
            # Updates
            Ldiff = logl - logl_old   # should be positive
            # logl_old = logl
            Pdiff = Ldiff
            
            # Updates
            logl_old = logl
    
            #print("Sf: ", Sf)
            #print("logl, Ldiff = ", logl, Ldiff)
            
        # end while loop
    
        #if niter < max_niter:
            #print("tol = ", convg_thres)
            #print("EM converges at precision %6.3e after %6d iterations." %(convg_thres, niter))
        #else:
            #print("EM does not converge at precision %6.3e after %6d iterations!!!" %(convg_thres, max_niter))
    
        # re-order parameters
        #[~,I]=sort(diag(Sf),'descend')
        #for k in range(L-1):
        #    V{k} = V{k}(:,I)
        #B = B(:,I);
        #Sf = Sf(I,I);
        #U = U(:,I);
        
        #### WJB added this to keep track of F for solving inverse
        self.F = U - Y@B
        
        self.B = B
        self.V = V
        self.U = U
        self.se2 = se2
        self.Sf = Sf
        self.rec = rec
        self.Vmat = Vmat
        
        YB = Y@B
        F = self.F
        FVmat = F@Vmat.T
        YBVmat = Y@B@Vmat.T
        #print("Size of B: ", B.shape)
        #print("Size of U: ", U.shape)
        #print("Size of V: ", V[0].shape, V[1].shape, V[2].shape)
        #print("Size of Vmat: ", Vmat.shape)
        #print("Size of F: ", F.shape)
        #print("Size of X: ", X.shape)
        #print("Size of Y: ", Y.shape)
        #print("Size of Y@B: ", YB.shape)
        #print("Size of FVmat: ", FVmat.shape)
        #print("Size of YBVmat: ", YBVmat.shape)
        #print("Size of Sf: ", Sf.shape)
        #print("Size of se2: ", se2.shape)
            
        return [B,V,U,se2,Sf,rec]
    
    
    #----------------------------------------------
    # Once a model is learned, use this to return
    # feature weights for a new X
    #----------------------------------------------
#    def get_weights(self):
#        return self.U

    def get_weights(self, Xnew):
        # Compute some parameters
        m = Xnew.shape         #
        n = m[0]            # sample size under test/train
            
        # Matricize Xnew
        Xmat = np.moveaxis(Xnew, 0, -1)  # move the first dimension, n, to the last; Xmat is a matricization of X samples
        Xmat = Xmat.reshape(-1, n ,order='F')
        
        # Compute weights
        U = Xmat.conj().T @ self.Vmat
        return U


    
    #----------------------------------------------
    # Computes a probability of membership for each
    # class label in Y
    #
    # Given an EEG tensor for a new subject X_i, you could compute the likelihood 
    # of X_i under each of the classes:
    # p(X_i | Y*),  where Y* could give the specified class 
    # (and can also include the true age etc).  
    # This can be computed as by taking the exponent of Equation (4) in the SupCP 
    # manuscript (which gives the log-likelihood, marginalized over U).  
    # Given a prior probability for each class (e.g., P(Y=k)), you can apply 
    # Bayes rule to compute the posterior probabilities for X_i belonging to each 
    # class (e.g., P(Y_i=k | X_i)).  
    #
    # X - computed features for one subject
    # Y = labels
    #
    #----------------------------------------------
    def classify(self, X, Y, labels, kwargs=None):
    
        # Input from the model
#        B = input[0]
#        V = input[1]
#        U = input[2]
#        se2 = input[3]
#        Sf = input[4]
    #    rec = input[5]
        R = self.R
        B = self.B
        V = self.V
        U = self.U
        se2 = self.se2
        Sf = self.Sf
    
        m = X.shape[0]
        n = len(Y)
#        Y = np.zeros((1,n))
#        for i in range(n):
#            Y[0,i] = Y_in[i]
        #Y = np.zeros((n,1))
#        Y = Y_in
        
        # Optional arguments
        Sf_diag = 1
        if kwargs != None:
          if 'Sf_diag' in kwargs:
            Sf_diag = kwargs['Sf_diag']
        
        # Compute
        p = 1
        for i in range(len(V)):
            p *= V[i].shape[0]
#        Vmat = np.zeros((p,R))
#        for r in range(R):
#            Temp = self.TensProd(V, [r])
#            Vmat[:,r] = Temp.flatten(order='F')
        Vmat = self.Vmat
    
        Xmat = np.moveaxis(X, 0, -1)  # move the first dimension, n, to the last; Xmat is a matricization of X samples
        Xmat = Xmat.reshape(-1, m ,order='F')        
        #Xmat = X.flatten(order='F')
            
        # Compute probability of each class
        prob = {}
#        n = len(labels)
        loglike = {}
        denom = 0.0
        for i, lab in enumerate(labels):
            Y[0,0] = lab
            # calc likelihood
            if Sf_diag:
                Sfinv = np.diag(1 / np.diag(Sf))
            else:
                Sfinv = np.linalg.inv(Sf)
    
            ResidMat = Xmat.conj().T - (Y @ B @ Vmat.conj().T)  # n*p
            Sfsq = SM.sqrt(Sf)
            MatForDet = (Sfsq @ Vmat.conj().T @ Vmat @ Sfsq) / se2 + np.eye(R)  # R X R
            logdet_VarMat = 2.0 * np.sum(np.log(np.diag(np.linalg.cholesky(MatForDet)))) + p * math.log(se2)
    
            term1 = (1 / se2) * np.trace(ResidMat @ ResidMat.conj().T)
            term2 = Vmat.conj().T @ ResidMat.conj().T @ ResidMat @ Vmat
            term3 = np.linalg.inv(Sfinv + (1 / se2) * (Vmat.conj().T @ Vmat))
            Trace = term1 - (1 / (se2 ** 2)) * np.trace(term2 @ term3)
            logl = (-n / 2) * (logdet_VarMat) - 0.5 * Trace
            loglike[lab] = logl
    
        # Compute odds ratio
#        n = len(labels)
        prob = {}
        for i, ilab in enumerate(labels):
            denom = 0.0
            for j, jlab in enumerate(labels):
                diff = loglike[jlab] - loglike[ilab]
                denom = denom + np.exp(diff)
                prob[ilab] = 1.0/denom
    
        #diff = loglike[labels[1]] - loglike[labels[0]]
        #odds = np.exp(diff)
        #if odds == np.inf:
        #    prob[labels[1]] = 1.0
        #else:
        #    prob[labels[1]] = odds/(1.0+odds)
        #prob[labels[0]] = 1 - prob[labels[1]]
        
        return prob[0]
    
    #----------------------------------------------
    """
    Essentially, what the SupCP results provide is a probability 
    model for the tensor data given covariates, and with a 
    categorical covariate we are able to invert this to get 
    the probability of each class.  In the regression context 
    we have a continuous variable, and it is not quite as 
    straightforward to invert the model to get a full distribution 
    for a continuous variable.  
    
    The approach here is fully Bayesian approach: 
        (1) begin with a prior distribution for the real-valued covariate 
            e.g., fit a Normal distribution to the training data
        (2) update the prior using Bayes rule with the likelihood from the SupCP model.  
    
    This is essentially what we are doing with the  classification 
    approach as well, where the "prior" is that each of the classes have 
    equal probability. 
    
    X = [YB + F, V1, V2, ..., Vk], where U = YB + F
    Each of these is known from the model (computed in the fit function).
    
    Now with a new X, we should be able to compute a new Y.
    
    """ 
    #----------------------------------------------
    def regress(self, X, Y_in, kwargs=None):
    
        # Input from the model
        R = self.R
        B = self.B
        V = self.V
        U = self.U
        se2 = self.se2
        Sf = self.Sf
    
        m = X.shape[0]
        n = len(Y_in)
        Y = np.zeros((1,n))
        for i in range(n):
            Y[0,i] = Y_in[i]
            
        # Compute probability distribution for the continuous variable
        values = np.arange(0,95,1)
        n = len(values)
        
        # Optional arguments
        Sf_diag = 1
        if kwargs != None:
          if 'Sf_diag' in kwargs:
            Sf_diag = kwargs['Sf_diag']
    
        # Compute
        p = 1
        for i in range(len(V)):
            p *= V[i].shape[0]
        #Vmat = np.zeros((p,R))
        #for r in range(R):
        #    Temp = self.TensProd(V, [r])
        #    Vmat[:,r] = Temp.flatten(order='F')           
        ####################
        Vmat = self.Vmat  # Vmat hasn't changed from the fit function
        ####################
        Xmat = np.moveaxis(X, 0, -1)  # move the first dimension, n, to the last; Xmat is a matricization of X samples
        Xmat = Xmat.reshape(-1, m ,order='F')        
#        Xmat = X.flatten(order='F')
            
        prob = np.zeros(n)
        loglike = {}
        denom = 0.0
        for i, val in enumerate(values):
            Y[0,0] = val
            
            # calculate the log n likelihood for each value
            if Sf_diag:
                Sfinv = np.diag(1 / np.diag(Sf))
            else:
                Sfinv = np.linalg.inv(Sf)

            ResidMat = Xmat.conj().T - (Y @ B @ Vmat.conj().T)  # n*p
            Sfsq = SM.sqrt(Sf)
            MatForDet = (Sfsq @ Vmat.conj().T @ Vmat @ Sfsq) / se2 + np.eye(R)  # R X R
            logdet_VarMat = 2.0 * np.sum(np.log(np.diag(np.linalg.cholesky(MatForDet)))) + p * math.log(se2)
    
            term1 = (1 / se2) * np.trace(ResidMat @ ResidMat.conj().T)
            term2 = Vmat.conj().T @ ResidMat.conj().T @ ResidMat @ Vmat
            term3 = np.linalg.inv(Sfinv + (1 / se2) * (Vmat.conj().T @ Vmat))
            Trace = term1 - (1 / (se2 ** 2)) * np.trace(term2 @ term3)
            logl = (-n / 2) * (logdet_VarMat) - 0.5 * Trace
            loglike[val] = logl
    
        # Compute likelihood (odds ratio ?) for each value
        for i, ival in enumerate(values):
            denom = 0.0
            for j, jval in enumerate(values):
                diff = loglike[jval] - loglike[ival]
                denom = denom + np.exp(diff)
            prob[i] = 1.0/denom
                
        # Compute regression value (maximum likelihood) and std dev      
        
        # Fit a spline to the probability distribution
        x = values
        y = prob
        #spline = UnivariateSpline(x, y, s=1)
        
        # Generate some x values for plotting the spline and its derivative
        #ynew = spline(x)
        
#        k = np.argmax(y)
        x0 = np.dot(x,y)
#        x0 = x[k]
        
        # Choose a window size
        #window_size = min(x0-a,b-x0)
        #y1 = x.index(x0-window_size)
        #y2 = x.index(x0+window_size)
        #ysd = y[y1:y2]
        
        # Get the data points within the window around x0
#        mask = (xnew > x0 - window_size) & (xnew < x0 + window_size)
#        x_window = xnew[mask]
        
        # Calculate the standard deviation
        std_dev = np.std(y)
        #std_dev = 0.0
        
        # For testing only
        if 1==0:        
            print("Sum ynew = ", np.sum(y))
            print("The maximum occurs at x = {x0}; truth = {truth}")
            print(f"The standard deviation of y around x0 = {x0} is {std_dev}")
                   
            # Plot the data, the spline, and its derivative
            plt.axvline(x=x0, color='red')
            #plt.scatter(x, y, label='Data')
            plt.plot(x, y, label='Prob Density Function')
            #plt.plot(x, ynew, label='Spline')
            #plt.plot(xnew, label='Derivative')
            plt.legend()
            plt.show()
        
    
        return x0, std_dev
    
    
    
    