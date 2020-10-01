#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Jul  9 14:55:51 2020

@author: Vera Zheng 
"""
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as si
from sympy.stats import Normal, cdf
import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import scipy.optimize as opt
from scipy.special import ndtr
def simulate_spot(a,b,c,T,m,dt):
        """
        function to model the normalised fictitious spot price
        Input:
        a: param: scaler
        b/c: param: a list if more than one future
        dt: time interval
        T: time to maturity(a list if more than one future)
        m: # of simulation paths
        Output:
        St: spot prices dynamics
           
        """
        # number of steps for each future (simulate until the furthest future)
        n = T / dt
        # longest 
        N = int(n[-1])
         
        S0 = 1
        St = np.zeros((N,m))
        St[0,:] = S0
       
        sigma = np.zeros((N,m))
        sigma0 = 0.1
        sqrt_dt = dt ** 0.5
       
        dw1 = np.random.normal(size=(N,m))* sqrt_dt 
        sigma[0,:] = sigma0
        for j in range (m):
                for i in range(int(n[0]) - 1):
               
                     sigma[i+1,j] = b[0] * St[i,j] +c[0] 
                     St[i+1,j] = St[i,j] + (1-St[i,j])* a * dt + sigma[i,j] * St[i,j] * dw1[i,j]
        if len(T)==1:
            return St
        else:
            for j in range (m):
            
                for k in range(len(T)-1):
                    
                    for i in range(int(n[k]),int(n[k+1])-1):
                        sigma[i+1,j] = b[k] * St[i,j] +c[k] 
                        St[i+1,j] = St[i,j] + (1-St[i,j])* a * dt + sigma[i,j] * St[i,j] * dw1[i,j]
        
        return St
     
def FutureDynamics(a, N_E,St,F0):
        """
        function to model the normalised fictitious spot price
        Input:
        
        N_E: difference between option maturity and futures maturity
        F0: Future price at the begining of the contact
        
        Output:
        Ft: spot prices dynamics size n X m 
            
        """
       
       
        
        
        Ft = F0*(1-(1-St)*np.exp(a*(-N_E)))

        return Ft
   
def OptionPricing(S_T,K,N_E,F_T,a,m):
        """
        function to price call options on the futures 
        Input:
       
        N_E: difference between option maturity and futures maturity
        S_T: spot price at maturity
        T: time to maturity
        Output:
        Call: Call price
        Put: Put price
            
        """
       
        
        effective_K= 1-np.exp(a*N_E)*(1-K/F_T)
        
        
           
        call = np.maximum(0,S_T - effective_K)*F_T
             
        Call =  np.sum(call)/m *np.exp(-a*N_E)
        
        SE_call = np.sqrt(sum(call-Call)/(m*(m-1)))
       
        return Call,  SE_call



def BS(F,K,sigma,T,r):
    d1 = (np.log(F / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = F* ndtr(d1)- K * np.exp(-r * T) * ndtr(d2)
    return call    
# 


def implied_vol(mkt_price, F, K, T_maturity, *args):
        Max_iteration = 500
        PRECISION = 1.0e-5
        sigma = 0.4
        for i in range(0, Max_iteration):
            d1 = (np.log(F / K) + ( 0.5 * sigma ** 2) * T_maturity) / (sigma * np.sqrt(T_maturity))
            d2 = d1 - sigma * np.sqrt(T_maturity)
            bls_price = F * ndtr(d1) -  K * ndtr(d2)
            vega = F * norm._pdf(d1) * np.sqrt(T_maturity)
            diff = mkt_price - bls_price  
            if (abs(diff) < PRECISION):
                return sigma
            sigma = sigma + diff/vega # f(x) / f'(x)
        return sigma 
     

class calibration_quadratic:
     def __init__(self,a,K,T_M,Future):
     
        self._K =K
        self._T =T_M
        self._a = a
        self._F = Future
     
       
     def sum_difference(self,K,F,T,market_price,param):
        diff = np.ones(len(K))
        for i in range(len(K)):
            price_i=BS(F+param[0],K[i]+param[0],param[1],T,0)
            diff[i]=abs(price_i- market_price[i])
        
            if abs(K[i]-F)<0.5:
                diff[i]=10000*diff[i]
            else:
                diff[i]=diff[i]
        total=sum(diff)  
        
        return total
     
     def optimise(self,K,F,T,market_price,param):
         start_params = np.array([0.01, 0.1])
         
         sum_diff = lambda param: self.sum_difference(K,F,T,market_price,param)
         #bnds = ((0,1),(0,1))
         all_results = opt.minimize(fun=sum_diff, x0=start_params,
                                        method="Nelder-Mead")#"BFGS"
         if (self.sum_difference(K,F,T,market_price,all_results.x))>0.01:
             all_results = opt.minimize(fun=sum_diff, x0=all_results.x,   method="BFGS")  
         return all_results.x
          
     def estimateSigma(self,gamma_sigma,param,K,F,T,N_E):
       
       
        a= self._a
        
        #param=np.exp(param) #transform to restrict a to be postive 
        effective_K = 1-np.exp(-a*N_E)*(1-K/F)
        f0 = F + gamma_sigma[0]
        k0 = effective_K*F+gamma_sigma[0]
        d1 = (np.log(f0 / k0) + (0.5 * gamma_sigma[1] ** 2) * T) / (gamma_sigma[1] * np.sqrt(T))
        d2 = d1 - gamma_sigma[1] * np.sqrt(T)
        #call_price = np.exp(a*T)/F *(f0*norm.cdf(d1)-k0*norm.cdf(d2))
       
        f1 = -2*a*(effective_K -1)*ndtr(d2)*np.exp(a*N_E)
        f2 = f1**2
        f3 = (f0 * norm._pdf(d1)*np.exp(0.5*a*N_E)*effective_K) **2/(k0**2*T)
        f4 = f0*norm._pdf(d1)/(F*np.sqrt(T))
        estimate_sigma = (f1+np.sqrt(f2+f3*(param[0]*effective_K +param[1])**2))/f4
        
        return estimate_sigma
     #for DD process 
     def object_func(self,gamma_sigma_list,F,K,T,N_E,param):
         "minimise difference in IV"
         
         n= np.shape(K)[0]
         diff = np.ones(n)
         for i in range (n):
             diff[i]= abs(gamma_sigma_list[1]- self.estimateSigma(gamma_sigma_list,param,K[i],F,T,N_E))
        
             if abs(K[i]-F)<0.5:
                diff[i]=10000*diff[i]
             else:
                diff[i]=diff[i]
        
         return  sum(diff)
    
     def find_param(self,gamma_sigma_list,F,K,T,N_E,param):
         start_params = np.array([0.01, 0.1])
         
         difference = lambda param: self.object_func(gamma_sigma_list,F,K,T,N_E,param)
         #bnds = ((0,1),(0,1))
         all_results = opt.minimize(fun=difference, x0=start_params,
                                        method="BFGS")#Nelder-Mead bounds=bnds)
         
         error=self.object_func(gamma_sigma_list,F,K,T,N_E,all_results.x)
         if (error)>0.001:
             all_results = opt.minimize(fun=difference, x0=all_results.x,  method="Nelder-Mead")
         return all_results.x
 

class time_dependent:
    def __init__(self,a,dt,m,b_list,c_list):
     
        self._a =a
        self._dt =dt
        self._m = m
        self._b_list = b_list
        self._c_list = c_list
     
    def object_func(self,gamma_sigma,F_front,F_back,T,N_E,K_list,mkt_vols,obj,b):
        """
        function to calculate deifferece between market and model 
        Future prices/IV
        input: 
        F_front: futures at the start of this month
        F_back: futures at the start of next month
        mkt_vols: mkts vols for F_back (for comparison to simulated futures)
        obj: Future Price/IV
        """
        a = self._a
        dt = self._dt
        m=self._m
        b_list = self._b_list
        c_list = self._c_list
        f0 = F_front + gamma_sigma[0]
        b_list[-1] =b
        c =gamma_sigma[1]*f0*np.exp(-0.5*a*N_E)/F_front-b
        c_list[-1]=c
        St = simulate_spot(a,b_list,c_list,T,m,dt)
        
        F_T = FutureDynamics(a, N_E,St[-1,:],F_front)
        if obj == "Future Price":
            diff = sum((F_T-F_back)**2)
        
        else:
             ones=np.ones(len(K_list))
             call=ones
             SE_call =ones
             for i in range(len(K_list)):
                 call[i],SE_call[i] = OptionPricing(St[-1,:],K_list[i],N_E,F_T,a,m)
                 params = np.vstack((call,F_T*ones,K_list, T*ones))
             vols = list(map(implied_vol, *params))
             n= len(K_list)
             diff = np.ones(n)
             for i in range (n):
                 if abs(K_list[i]-F_T)<0.5:
                     diff[i]= (mkt_vols[i]- vols[i])**2*1000
                 else:
                      diff[i]=(mkt_vols[i]- vols[i])**2
             diff =sum(diff)
         
        return diff
        
     
        
    def find_bc(self,gamma_sigma,F_front,F_back,T,N_E,K_list,mkt_vols,obj,start_params,b):
       
        
         difference = lambda b: self.object_func(gamma_sigma,F_front,F_back,T,N_E,K_list,mkt_vols,obj,b)
         #bnds = ((0,1),(0,1))
         all_results = opt.minimize(fun=difference, x0=start_params,
                                        method="Nelder-Mead")#BFGS bounds=bnds)
         
         error=self.object_func(gamma_sigma,F_front,F_back,T,N_E,K_list,mkt_vols,obj,all_results.x)
         while(error)>0.001:
             all_results = opt.minimize(fun=difference, x0=all_results.x,  method="Nelder-Mead")
             error=self.object_func(gamma_sigma,F_front,F_back,T,N_E,K_list,mkt_vols,obj,all_results.x)
         print('Error',error)
         return all_results.x
        
            
            

if __name__ == "__main__":
    output_path = "/Users/veraisice/Desktop/Comodity-Market-Research/thesis_1/"
    input_path  = "/Users/veraisice/Desktop/Comodity-Market-Research/Input/"
   
    Month_List = ["August","September","October","November","December"]
    N_E = np.ones((len(Month_List)))
    T_M = np.ones((len(Month_List)))
    Future= np.ones((len(Month_List)))
    market_price=np.zeros((len(Month_List),10), dtype=np.ndarray)
    K=np.zeros((len(Month_List),10), dtype=np.ndarray)
    vols_mkt = np.zeros((len(Month_List),10), dtype=np.ndarray)
    vols_mdl = np.zeros((len(Month_List),10), dtype=np.ndarray)
    effective_K=np.zeros((len(Month_List),10), dtype=np.ndarray)
    
    #magic number
    a =0.1 
    m=1000
    dt=1/365
    
    for i in range(len(Month_List)):
        Option_Data = pd.read_excel(input_path+ "TTFdata"+".xlsx",sheet_name = Month_List[i])    #SEPopt
    #strike 
        K[i] = Option_Data["Strike"].values
    #market option price
        market_price[i] = Option_Data["Call"].values
    #time to maturity
        T_M[i]= Option_Data["Time to Maturity"].values[0]
    #expiry to notification date
        N_E[i]=Option_Data["N-E"].values[0]
    #future price
        Future[i] = Option_Data["1-Month Future"].values[0]
    
        effective_K[i] = 1-np.exp(-a*N_E[i])*(1-K[i]/Future[i])
        ones= np.ones(np.size(K[i]))
        params_mkt = np.vstack((market_price[i], Future[i]*ones,K[i], T_M[i]*ones))
    
        vols_mkt[i] = list(map(implied_vol, *params_mkt))
       
   
#get first set of b and c for Augustto initialise the process
    gamma_sigma_list= calibration_quadratic(a, K, T_M, Future).optimise(np.asarray(K[0],float), Future[0],T_M[0],np.asarray(market_price[0],float),param=[])
    b0,c0=calibration_quadratic(a, K, T_M, Future).find_param(gamma_sigma_list,Future[0],K[i],T_M[0],N_E[0],param=[])
    print("Calibration Result for "+str(Month_List[0])+ " is b=" + str(b0)+" and c=" + str(c0))
 

#calibrate for the second month 
    b_list = np.ones(2)
    b_list[0]= b0
    c_list = np.ones(2)
    c_list[0]= c0
    b1 = time_dependent(a, dt, m, b_list, c_list).find_bc(gamma_sigma_list, Future[1], Future[2], T_M[:1], N_E[1], K[1], vols_mkt[1], "Future Price", 0.1, [])
    


           