#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 18:02:20 2020

@author: veraisice
"""

import scipy.stats as si
from sympy.stats import Normal, cdf
import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import scipy.optimize as opt
from scipy.special import ndtr
class Process:
   
    def __init__(self, sigma0, a,m,alpha,rho,S0,b,c):
        """
        Parameters
        ----------
        sigma0 : float
            initial vol if the dynamic is chosen to be "stochastic"
        a : float
            param in the fititatious spot price process in paper Smile Modelling
        m : int
            number of simulations
        alpha : float
            param in the vol process if the dynamic is chosen to be "Stochastic"
           
        rho : float between 0 and 1
             correlation between vol and spot if the dynamic is chosen to be "Stochastic"
        S0 : float
              initial ficticious spot price
        b : float
            param in the vol process if the dynamic is chosen to be "Quadratic"
        c : float
           param in the vol process if the dynamic is chosen to be "stochastic"
 
        Returns
        -------
     
 
        """
        self._S0 =S0
        self._sigma0 = sigma0
        self._a = a
        self._m = m
        self._alpha = alpha
        self._rho = rho
        self._b = b
        self._c =c
   
    def simulate_spot(self,T, dt,dynamics):
        """
        function to model the normalised fictitious spot price
        Input:
        dt: time interval
        T: time to maturity
        dynamics: param to determine the spot process
        Output:
        St: spot prices dynamics
           
        """
        n = int(T / dt)
        a = self._a
        m = self._m
        b = self._b
        c = self._c
        S0 = self._S0
        St = np.zeros((n,m))
        St[0,:] = S0
        alpha = self._alpha
        rho = self._rho
        sigma = np.zeros((n,m))
        sigma0 = self._sigma0
        sqrt_dt = dt ** 0.5
       
        #dw1 = np.random.normal(size=(n,m))* sqrt_dt
        mP=int(m*0.5)
        dw1 = np.random.normal(size=(n,mP))* sqrt_dt
        
        if dynamics == "Quadratic":
            sigma[0,:] = sigma0
            for j in range (mP):
                for i in range(n - 1):
                    sigma[i,j] = b*St[i,j] +c
                    sigma[i,j+mP] = b*St[i,j+mP] +c
                    St[i+1,j] = St[i,j] + (1-St[i,j])* a * dt + sigma[i,j] * St[i,j] * dw1[i,j]
                    St[i+1,j+mP] = St[i,j+mP] + (1-St[i,j+mP])* a * dt - sigma[i,j+mP] * St[i,j+mP] * dw1[i,j]
       
        if dynamics == "Stochastic":
             sigma[0,:] = sigma0
             dw2 = np.random.normal(size=(n,m))* sqrt_dt
             dw3 = rho * dw1 + np.sqrt(1- rho **2) *dw2
             for j in range (m):
                 for i in range(n - 1):
                  
                     sigma[i+1,j] = sigma[i,j] +  alpha *sigma[i,j] * dw3[i,j]
                     St[i+1,j] = St[i,j] + (1-St[i,j])* a * dt + sigma[i,j] * St[i,j]^0.5 * dw1[i,j]
        return St
  
    def OptionPricing(self,S_T,K_mkt,r,dt,T,F_T,a,method):
        """
       function to price call options on the futures
        Input:
        dt: time interval
        T: maturity     
        
        Output:
        Call: Call price
        Put: Put price
           
        """
        sigma = self._sigma0      
        K= 1-np.exp(a*T)*(1-K_mkt/F_T)
       
        if method == "MC":
           call = np.maximum(0,S_T - K)*F_T
        if method == "bls":  
            
                d1 = (np.log(S_T / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
                d2 = (np.log(S_T/ K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))              
                call = (S_T * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))*F_T
 
        Call = np.exp(-r*T)* np.sum(call)/m *np.exp(-a*T)       
        SE_call = np.sqrt(sum(call-Call)/(m*(m-1)))      
        return Call,  SE_call
 
 
def implied_vol(mkt_price, F, K, T_maturity, r, *args):
        Max_iteration = 500
        PRECISION = 1.0e-5
        sigma = 0.4
        for i in range(0, Max_iteration):
            d1 = (np.log(F / K) + (r + 0.5 * sigma ** 2) * T_maturity) / (sigma * np.sqrt(T_maturity))
            d2 = d1 - sigma * np.sqrt(T_maturity)
            bls_price = F * ndtr(d1) - np.exp(-r * T_maturity) * K * ndtr(d2)
            vega = F * norm._pdf(d1) * np.sqrt(T_maturity)
            diff = mkt_price - bls_price 
            if (abs(diff) < PRECISION):
                return sigma
            sigma = sigma + diff/vega # f(x) / f'(x)
        return sigma
 
 
class calibration:
     def __init__(self,a,K,T_M,Future):
    
        self._K =K
        self._T =T_M
        self._a = a
        self._F = Future
 
     def DD_diff(self,K,F,T,market_price,param):
        diff = np.ones(len(K))
        for i in range(len(K)):
            price_i=BS(F+param[0],K[i]+param[0],param[1],T,0)
            diff[i]=1.0*abs(price_i- market_price[i])
        
            if abs(K[i]-F)<0.5:
                diff[i]=10000*diff[i]
        total=sum(diff)   
        return total
    
     def CalibrateDD(self,K,F,T,market_price,param):
         start_params = np.array([1, 0.1])
        
         sum_diff = lambda param: self.DD_diff(K,F,T,market_price,param)
         #bnds = ((0,1),(0,1))
         all_results = opt.minimize(fun=sum_diff, x0=start_params,   method="Nelder-Mead")#bounds=bnds)
        
         if (self.DD_diff(K,F,T,market_price,all_results.x))>0.01:
             all_results = opt.minimize(fun=sum_diff, x0=all_results.x,   method="Nelder-Mead")  
         return all_results.x
         
     def estimateSigma(self,gamma_sigma,param,K,F,T):
       
        b=param[0]
        c=param[1]
 
        effective_K = 1-np.exp(-a*T)*(1-K/F)   
        estimate_sigma=effective_K*(c+b*effective_K)*F/(effective_K*F+gamma_sigma[0])
        return estimate_sigma       
     
     def BCparams_func(self,gamma_sigma_list,F,K,T,param):       
         
         n= np.shape(K)[0]
         diff = np.ones(n)
         for i in range (n):
             diff[i]= 1.0*abs(gamma_sigma_list[1]- self.estimateSigma(gamma_sigma_list,param,K[i],F,T))
             if abs(K[i]-F)<0.5:
                diff[i]=10000.0*diff[i]
             else:
                diff[i]=diff[i]
       
         return  sum(diff)
 
        
        
     def find_BCparam(self,gamma_sigma_list,F,K,T,param):
         start_params = np.array([-0.2, 0.1])        
         difference = lambda param: self.BCparams_func(gamma_sigma_list,F,K,T,param)
       
         all_results = opt.minimize(fun=difference, x0=start_params,  method="Nelder-Mead")#bounds=bnds)
         error=self.BCparams_func(gamma_sigma_list,F,K,T,all_results.x)
         if (error)>0.001:
             all_results = opt.minimize(fun=difference, x0=all_results.x,  method="Nelder-Mead")
         return all_results.x
 
def BS(F,K,sigma,T,r):
    d1 = (np.log(F / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = F* ndtr(d1)- K * np.exp(-r * T) * ndtr(d2)
    return call
        
if __name__ == "__main__":     
    
    output_path = "/Users/veraisice/Desktop/Comodity-Market-Research/thesis_1/"
    input_path  = "/Users/veraisice/Desktop/Comodity-Market-Research/Input/"
   
    a=0.0
    Month_List = np.array(["October","November","December"])
    c= np.ones(len(Month_List))
    b= np.ones(len(Month_List))
    T_M = np.ones((len(Month_List)))
    Future= np.ones((len(Month_List)))
    market_price=np.zeros((len(Month_List),10), dtype=np.ndarray)
   
    K=np.zeros((len(Month_List),10), dtype=np.ndarray)
    gamma_sigma_list= np.zeros((len(Month_List),2), dtype=np.ndarray)
    for i in range(len(Month_List)):
        Option_Data = pd.read_excel(input_path+ "TTFdata"+".xlsx",sheet_name = Month_List[i])    #SEPopt
    #strike
        K[i] = Option_Data["Strike"].values
    #market option price
        market_price[i] = Option_Data["Call"].values
    #time to maturity
        T_M[i]= Option_Data["Time to Maturity"].values[0]
    #future price
        Future[i] = Option_Data["1-Month Future"].values[0]
        gamma_sigma_list[i]= calibration(a, K, T_M, Future).CalibrateDD(np.asarray(K[i],float), Future[i],T_M[i],np.asarray(market_price[i],float),param=[1,1])
        print('gamma',gamma_sigma_list[0][0])
        print('sigma',gamma_sigma_list[0][1])
        print('error', calibration(a, K, T_M, Future).DD_diff(K[i],Future[i],T_M[i],market_price[i],gamma_sigma_list[i]))
 
        b[i],c[i]=calibration(a, K, T_M, Future).find_BCparam(gamma_sigma_list[i],Future[i],K[i],T_M[i],param=[])
        print("Calibration Result for "+str(Month_List[i])+ " is b=" +str(b[i])+" and c=" +str(c[i]))
        print('Error',calibration(a, K, T_M, Future).BCparams_func(gamma_sigma_list[i],Future[i],K[i],T_M[i],[b[i],c[i]]))
    #####calculate the model price######
    Month ="November"
    sigma0= gamma_sigma_list[1][1]
    gamma0= gamma_sigma_list[1][0]
    r=0
    m=50000
    rho =0.0
    S0=1
    alpha=0
   
    dt=1/365
    Option_data = pd.read_excel(input_path+ "TTFdata"+".xlsx",sheet_name = Month)
    F0 =  Option_data["1-Month Future"].values[0]  
    K_list = Option_data["Strike"].values
   
    
    ones =np.ones(np.size(K_list))  
    
    #time to maturity
    T_M = Option_data["Time to Maturity"].values[0]
    Call_list = Option_data["Call"].values
  
    S = Process(sigma0, a, m, alpha, rho,S0,b[1],c[1]).simulate_spot(T_M,dt,"Quadratic")
    S_T = S[-1,:]
   
    call = np.zeros((len(K_list)))
    SE_call = np.zeros((len(K_list)))
    market_price_dd=np.zeros((len(K_list)))
    for i in range(len(K_list)):
     call[i],SE_call[i] = Process(sigma0, a ,m, alpha, rho,S0,b[1],c[1]).OptionPricing(S_T,K_list[i],r,dt,T_M,F0,a,"MC")
     market_price_dd[i]=BS(F0+gamma0,K_list[i]+gamma0,sigma0,T_M,0) #Process(sigma0, a ,m, alpha, rho,S0,b[0],c[0]).OptionPricing(F_T+gamma0,K_list[i]+gamma0,0,dt,T_M,np.mean(F_T),a,"bls")
    params = np.vstack((call, F0*ones,K_list, T_M*ones, r*ones))
    vols = list(map(implied_vol, *params))
   
    effective_K = 1-np.exp(-a*T_M)*(1-K_list/F0)
   
    ###implied vol from the market prices
    params_mkt = np.vstack((market_price_dd, F0*ones,K_list, T_M*ones, r*ones))
    vols_mkt = list(map(implied_vol, *params_mkt))
   
    plt.figure(dpi=1000)
    plt.plot(effective_K,vols,'--b*',label="Model IV")
    plt.plot(effective_K,vols_mkt,'--r*',label="Market IV")
    plt.xlabel("Strike")
    plt.ylabel("Implied Volatility")
    plt.title("Implied Volatilities of TTF Futures Options Expires in " + str(Month)  )
    plt.legend(loc= 'best')
    plt.savefig(output_path + "Figures/IV_model_mkt2"+str(Month))
   
    
    
    plt.figure(dpi=1000)
    plt.plot(effective_K,call,'--b*',label="Model Prices VS Strikes")
    #plt.plot(effective_K,Call_list,'--r',label="Market Prices VS Strikes")
    plt.plot(effective_K,Call_list,linestyle='None',color='g',marker='o',label="DD Prices VS Strikes")
    plt.xlabel("Strike")
    plt.ylabel("Option Price")
    plt.title("Comparison of TTF Futures Option Price Expired in " + str(Month)  )
    plt.legend(loc= 'best')
    plt.savefig(output_path + "Figures/price_model_mkt2"+str(Month))
   
 
 

