#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 15:19:40 2020

@author: vera
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
        
        dw1 = np.random.normal(size=(n,m))* sqrt_dt 
        
        if dynamics == "Quadratic":
            sigma[0,:] = sigma0
            for j in range (m):
                for i in range(n - 1):
               
                     sigma[i+1,j] = b * St[i,j] +c 
                     St[i+1,j] = St[i,j] + (1-St[i,j])* a * dt + sigma[i,j] * St[i,j] * dw1[i,j]
        
        if dynamics == "Stochastic":
             sigma[0,:] = sigma0
             dw2 = np.random.normal(size=(n,m))* sqrt_dt 
             dw3 = rho * dw1 + np.sqrt(1- rho **2) *dw2
             for j in range (m):
                 for i in range(n - 1):
                   
                     sigma[i+1,j] = sigma[i,j] +  alpha *sigma[i,j] * dw3[i,j] 
                     St[i+1,j] = St[i,j] + (1-St[i,j])* a * dt + sigma[i,j] * St[i,j]^0.5 * dw1[i,j]
        return St
   
        
 
   
   
    def FutureDynamics(self, T,dt,St,F0):
        """
        function to model the normalised fictitious spot price
        Input:
        dt: time interval
        T: maturity
        F0: Future price at the begining of the contact
        
        Output:
        Ft: spot prices dynamics size n X m 
            
        """
        a = self._a
       
        #save only date of interest 
        St = St[::21]
        
        Ft = F0*(1-(1-St)*np.exp(a*(i*dt-T)))

        return Ft
   
    def OptionPricing(self,S_T,K,r,dt,T,F_T,a,method):
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
        
        K= 1-np.exp(a*T)*(1-K/F_T)
        
        if method == "MC":
           call = np.maximum(0,S_T - K)*F_T
        if method == "bls":   
            
                d1 = (np.log(S_T / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
                d2 = (np.log(S_T/ K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
               
                call = (S_T * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))*F_T

        Call = np.exp(-r*T)* np.sum(call)/m *np.exp(-a*T)
        
        SE_call = np.sqrt(sum(call-Call)/(m*(m-1)))
       
        return Call,  SE_call
    
   

     
 ###############main############
class AsianBasketOption:
    def __init__(self,weight,a,b,c,alpha,dt,T):
       
        self._a = a
        self._b = b
        self._c = c
        self._alpha = alpha
        self._dt = dt
        self._T = T
        self._weight = weight
   
    def simulate_correlated_paths(self,corr,nSims,nAssets,nSteps,dynamics,F0):
        """ Inputs: S0 - stock price
            mu - expected return
            sig - volatility
            corr - correlation matrix
            dt - size of time steps
            steps - number of time steps to calculate
            nsims - number of simulation paths to generate

            Output: F - a (steps+1)-by-nsims-by-nassets 3-dimensional matrix where
            each row represents a time step, each column represents a
            seperate simulation run and each 3rd dimension represents a
             different asset.
        """
        a = self._a
        b = self._b
        c = self._c
        alpha = self._alpha
        T= self._T
        sigma = np.zeros((nSteps,nSims,nAssets))
        St = np.zeros((nSteps,nSims,nAssets))
        #if incoporate control variate simulation
        # sigma1 = np.zeros((nSteps,nSims,nAssets))
        # St1 = np.zeros((nSteps,nSims,nAssets))
        R = np.linalg.cholesky(corr)
        if dynamics == "Quadratic":
            sigma[0,:,:] = b * S0 + c
            for j in range (nSims):
                dw = np.random.normal(size=(nSteps,nAssets))
                eps = np.dot(dw , R)
                for i in range(nSteps - 1):
                   
                        sigma[i+1,j,:] = b * St[i,j,:] +c 
                       # sigma1[i+1,j,:] = b * St1[i,j,:] +c 
                        #St1[i+1,j,:] = St1[i,j,:] + (1-St1[i,j,:])* a * dt + sigma1[i,j,:] * St1[i,j,:]*-eps[i,:]
                        St[i+1,j,:] = St[i,j,:] + (1-St[i,j,:])* a * dt + sigma[i,j,:] * St[i,j,:]*eps[i,:]
                       
        
        
        if dynamics == "Stochastic":
             dw1 = np.random.normal(size=(nSteps,nSims))
             dw2 = np.random.normal(size=(nSteps,nSims))
             sigma[0,:,:] = sigma0
             dw3 = rho * dw1 + np.sqrt(1- rho **2) *dw2
             for j in range (nSims):
                 dw = np.random.normal(size=(nSteps,nAssets))
                 eps = np.dot(dw ,R)
                 for i in range(nSteps - 1):
                     
                         sigma[i+1,j,:] = sigma[i,j,:] +  alpha *sigma[i,j,:] * dw3[i,j]
                         St[i+1,j,:] = St[i,j,:] + (1-St[i,j,:])* a * dt + sigma[i,j,:] * St[i,j,:]^0.5* eps[i,:]
                        # sigma1[i+1,j,:] = sigma[i,j,:] +  alpha *sigma[i,j,:] * dw3[i,j]
                        # St1[i+1,j,:] = St[i,j,:] + (1-St[i,j,:])* a * dt + sigma[i,j,:] * St[i,j,:]^0.5 * eps[i,:]
       # St1 = St1[::21,:,:] 
        St = St[::21,:,:] 
       # StTot =  np.concatenate((St,St1),1)
        Ft = F0*(1-(1-St)*np.exp(a*T))  
        return Ft   
    
    def pricing(self,n1,n2,Ft,k):
      
        #[n1,n2]is the interval to take averge of the future prices
        weight = self._weight
        #k=np.dot(Ft[0,:,:],weight)
   
        Ftot = np.dot(np.mean(Ft[n1:n2,:,:],0),weight)
   
        V = np.maximum(Ftot - k,0)
        avgV = np.mean(V)
    
         
        return V, avgV
    
# 
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
     
       
        
        
     
      
     def price_difference(self,param,K,F,T,market_price):
        
         d1 = (np.log((F+param[0]) / (K+param[0])) + (r + 0.5 * param[1] ** 2) * T) / (param[1] * np.sqrt(T))
         d2 = d1 - param[1] * np.sqrt(T)
         bls_price = (F+param[0]) * ndtr(d1) - np.exp(-r * T) * K * ndtr(d2)
         difference = bls_price - market_price
         return abs(difference)
       
     def sum_difference(self,K,F,T,market_price,param):
        diff = np.ones(len(K))
        for i in range(len(K)):
           
            diff[i]=self.price_difference(param,K[i],F,T,market_price[i])
         
            if abs(K[i]-F)<1:
                diff[i]=10000*diff[i]
            else:
                diff[i]=diff[i]
        return sum(diff)
     
     def optimise(self,K,F,T,market_price,param):
         start_params = np.array([0.01, 0.1])
         
         sum_diff = lambda param: self.sum_difference(K,F,T,market_price,param)
         #bnds = ((0,1),(0,1))
         all_results = opt.minimize(fun=sum_diff, x0=start_params,
                                        method="BFGS")#bounds=bnds)
         return all_results.x
          
     def estimateSigma(self,gamma_sigma,param,K,F,T):
       
       
        a= self._a
        
        #param=np.exp(param) #transform to restrict a to be postive 
        effective_K = 1-np.exp(-a*T)*(1-K/F)
        f0 = F + gamma_sigma[0]
        k0 = effective_K*F+gamma_sigma[0]
        d1 = (np.log(f0 / k0) + (r + 0.5 * gamma_sigma[1] ** 2) * T) / (gamma_sigma[1] * np.sqrt(T))
        d2 = d1 - gamma_sigma[1] * np.sqrt(T)
        #call_price = np.exp(a*T)/F *(f0*norm.cdf(d1)-k0*norm.cdf(d2))
       
        f1 = -2*a*(effective_K -1)*ndtr(d2)*np.exp(a*T)
        f2 = 4*a**2*(effective_K -1)**2*ndtr(d2)**2*np.exp(2*a*T)
        f3 = f0**2 * norm._pdf(d1)**2*np.exp(a*T)*effective_K **2
        f4 = f0*norm._pdf(d1)/(F*np.sqrt(T))
        estimate_sigma = (f1+np.sqrt(f2+(f3*(param[0]*effective_K +param[1])**2)/(k0**2*np.sqrt(T))))/f4
        
        return estimate_sigma
    
    
   
        
     
     def obeject_func(self,gamma_sigma_list,F,K,T,param):
         
         
         n= np.shape(K)[0]
         diff = np.ones(n)
         for i in range (n):
             diff[i]= abs(gamma_sigma_list[1]- self.estimateSigma(gamma_sigma_list,param,K[i],F,T))
        
             if abs(K[i]-F)<1:
                diff[i]=10000*diff[i]
             else:
                diff[i]=diff[i]
        
         return  sum(diff)
    
     
        
   
        
        
     def find_param(self,gamma_sigma_list,F,K,T,param):
         start_params = np.array([0.01, 0.1])
         
         difference = lambda param: self.obeject_func(gamma_sigma_list,F,K,T,param)
         #bnds = ((0,1),(0,1))
         all_results = opt.minimize(fun=difference, x0=start_params,
                                        method="BFGS")#bounds=bnds)
         return all_results.x
         


class Sabr:
    def SABR(self,alpha,beta,rho,nu,F,K,time,MKT):
        """
        function to evaluate model implied vol at each strike with each tenor
        Input:
        F: Current Future price 
        Time: time to maturity
        MKT: market vol
        Output:
        diff: difference between model implied vol and market implied vol
            
        """
        if F == K: # ATM formula
            V = (F*K)**((1-beta)/2.)
            logFK = np.log(F/K)
            A = 1 + ( ((1-beta)**2*alpha**2)/(24.*(V**2)) + (alpha*beta*nu*rho)/(4.*V) + ((nu**2)*(2-3*(rho**2))/24.) ) * time
            vol = (alpha/F**(1-beta))*A
            diff = vol - MKT
        elif F != K: # not-ATM formula 
            V = (F*K)**((1-beta)/2.) 
            logFK = np.log(F/K)
            z = nu*alpha*F**(1-beta)-K**(1-beta)/(alpha*(1-beta))
            x =np.log( ( np.sqrt(1-2*rho*z+z**2) + z - rho ) / (1-rho) )
            A= 1 + ( ((1-beta)**2*alpha**2)/(24.*(V**2)) + (alpha*beta*nu*rho)/(4.*V) + ((nu**2)*(2-3*(rho**2))/24.) ) * time
            
            Vol = (nu*logFK*A)/x 
            diff = Vol - MKT
        return diff
  


    def smile(self,alpha,beta,rho,nu,F,K,time,MKT,i):
        """
        The smile function computes the implied volatilities for
        a given ”smile” pointed out by the index i. F, time and 
        the parameters are scalars; K and MKT are vectors.
        """
        for j in range(len(K)):
        
            self.SABR(alpha,beta,rho,nu,F,K[j],time,MKT[j])



    def SABR_vol_matrix(self,alpha,beta,rho,nu,F,K,time,MKT): 
        """function computes the implied volatilities for all 
        combinations of swaptions. F, time and the parameters are vectors; 
        K and MKT are matrices.
        """
        for i in range(len(F)):
            self.smile(alpha[i],beta[i],rho[i],nu[i],F[i],K[i],time[i],MKT[i],i)


    def objfunc(self,par,F,K,time,MKT): 
        sum_sq_diff = 0
        alpha,beta,rho,nu=par
        for j in range(len(K)):
            diff = self.SABR(alpha,beta,rho,nu,F,K[j],time,MKT[j])  
            sum_sq_diff = sum_sq_diff + diff**2
        obj = np.sqrt(sum_sq_diff) 
        return obj
    
    
    
    
    def calibration(self,starting_par,F,K,time,MKT): 
        """The function used to calibrate each smile 
        (different strikes within the same tenor"""
        alpha = np.zeros(len(F))
        beta = np.zeros(len(F))
        rho = np.zeros(len(F))
        nu = np.zeros(len(F))
        
        for i in range(len(F)):
            x0 = starting_par
            bnds = ((0.001,None) , (0,1) , (-0.999,0.999) , (0.001,None) ) 
            Diff = lambda param: self.objfunc(param,F[i],K[i],time[i],MKT[i])
            res = opt.minimize(Diff,x0, method="SLSQP",bounds = bnds) 
            alpha[i] = res.x[0]
            beta[i] = res.x[1] 
            rho[i] = res.x[2] 
            nu[i] = res.x[3]

        return alpha, beta, rho, nu

    def Sabr_underlying(self,alpha,beta,rho,nu,m,F0,a,T_M,K):
        
          """function to simulate future prices and eta 
          given known param"""
          n = int(T_M / dt)
          # Ft = np.zeros((n,m))
          
          # Ft[0,:] = F0
          eta = np.zeros((n,m))
          eta_S = np.zeros((n,m))
          St = np.zeros((n,m))
          St[0,:]=1
          sqrt_dt = dt ** 0.5
          dw1 = np.random.normal(size=(n,m))* sqrt_dt 
          eta[0,:] = alpha
          dw2 = np.random.normal(size=(n,m))* sqrt_dt 
          dw3 = rho * dw1 + np.sqrt(1- rho **2) *dw2
          K_F=1-np.exp(a*T_M)*(1-K/F0)
          for j in range (m):
              for i in range(n - 1):
                   eta[i+1,j] = eta[i,j] +  nu *eta[i,j] * dw3[i,j] 
                   #Ft[i+1,j] = Ft[i,j] + Ft[i,j]** beta *  eta[i,j] * dw1[i,j]
                   eta_S[i+1,j] = eta[i+1,j]/(F0*(1+K_F*np.exp(-a*T_M)))
                   St[i+1,j] = St[i,j] + (1-St[i,j])* a * dt + eta_S[i,j] * St[i,j] * np.random.normal(size=(1,1))* sqrt_dt 
            #St = 1-(1-Ft/F0)*np.exp(a*T_M)
          #choose ATM strike          
          return St


if __name__ == "__main__":
    output_path = "/Users/veraisice/Desktop/Comodity-Market-Research/thesis_1/"
    input_path  = "/Users/veraisice/Desktop/Comodity-Market-Research/Input/"
   
    a=0.103
    Month_List = np.array(["August","September","October","November"])
    
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
        gamma_sigma_list[i]= calibration(a, K, T_M, Future).optimise(np.asarray(K[i],float), Future[i],T_M[i],np.asarray(market_price[i],float),param=[])
    
    
    ####Calibration for different months
    c= np.ones(len(Month_List))
    b= np.ones(len(Month_List))
    for i in range(len(Month_List)):  
    
         b[i],c[i]=calibration(a, K, T_M, Future).find_param(gamma_sigma_list[i],Future[i],K[i],T_M[i],param=[])
         print("Calibration Result for "+str(Month_List[i])+ " is b=" +str(b[i])+" and c=" +str(c[i]))
    #####calculate the model price######
    Month ="October"
    sigma0= gamma_sigma_list[2][1]
    r=0
    m=10000
    alpha = 0.2
    rho =0.15
    S0=1
    #
    
    dt=1/252
    Option_data = pd.read_excel(input_path+ "TTFdata"+".xlsx",sheet_name = Month)
    F0 =  Option_data["1-Month Future"].values[0]   
    K_list = Option_data["Strike"].values 
   
    
    ones =np.ones(np.size(K_list))   
    
    #time to maturity
    T_M = Option_data["Time to Maturity"].values[0]
    Call_list = Option_data["Call"].values
    #c_mkt = Call_list*np.exp(a*T_M)/F0 ##call price on the normalised spot
   
    S = Process(sigma0, a, m, alpha, rho,S0,b[2],c[2]).simulate_spot(T_M,dt,"Quadratic")
    F= Process(sigma0,a,m, alpha,rho,S0,b[2],c[2]).FutureDynamics(T_M,dt,S,F0)
    F_T = F[-1,:]
    S_T = S[-1,:]
    
    call = np.zeros((len(K_list)))
    SE_call = np.zeros((len(K_list)))
    
    for i in range(len(K_list)):
     call[i],SE_call[i] = Process(sigma0, a ,m, alpha, rho,S0,b[2],c[2]).OptionPricing(S_T,K_list[i],r,dt,T_M,np.mean(F_T),a,"MC")
    
    params = np.vstack((call, F0*ones,K_list, T_M*ones, r*ones))
    vols = list(map(implied_vol, *params))
    
    effective_K = 1-np.exp(-a*T_M)*(1-K_list/np.mean(F_T))
    
    ###implied vol from the market prices
    params_mkt = np.vstack((Call_list, F0*ones,K_list, T_M*ones, r*ones))
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
    plt.plot(effective_K,Call_list,'--r*',label="Market Prices VS Strikes")
    plt.xlabel("Strike")
    plt.ylabel("Option Price")
    plt.title("Comparison of TTF Futures Option Price Expired in " + str(Month)  )
    plt.legend(loc= 'best')
    plt.savefig(output_path + "Figures/price_model_mkt2"+str(Month))
    
    Month ="October"
    sigma0= gamma_sigma_list[2][1]
    a1 = 0.103#0.103018 #
    a2=0.15325
    a3=0.1522
    m = 10000
    alpha = 0
    rho =0.15
    S0=1
    #
    
    dt=1/365
    Option_data = pd.read_excel(input_path+ "TTFdata"+".xlsx",sheet_name = Month)
    F0 =  Option_data["1-Month Future"].values[0]   
    K_list = Option_data["Strike"].values 
   
    
    ones =np.ones(np.size(K_list))   
    
    #time to maturity
    N_E = Option_data["N-E"].values[0]
    T_M = Option_Data["Time to Maturity"].values[0]
    Call_list = Option_data["Call"].values
    #c_mkt = Call_list*np.exp(a*T_M)/F0 ##call price on the normalised spot
   
    S1 = Process(sigma0, a1, m, alpha, rho,S0,b[2],c[2]).simulate_spot(T,dt,"Quadratic")
    F1= Process(sigma0,a1,m, alpha,rho,S0,b[2],c[2]).FutureDynamics(N_E,dt,S1,F0)
    F_T1 = F1[-1,:]
    S_T1 = S1[-1,:]
    
    S1 = Process(sigma0, a1, m, alpha, rho,S0,b[2],c[2]).simulate_spot(T,dt,"Quadratic")
    F1= Process(sigma0,a1,m, alpha,rho,S0,b[2],c[2]).FutureDynamics(N_E,dt,S1,F0)
    F_T1 = F1[-1,:]
    S_T1 = S1[-1,:]
    
    S2 = Process(sigma0, a2, m, alpha, rho,S0,b[2],c[2]).simulate_spot(T,dt,"Quadratic")
    F2= Process(sigma0,a2,m, alpha,rho,S0,b[2],c[2]).FutureDynamics(T,dt,S2,F0)
    F_T2 = F2[-1,:]
    S_T2 = S2[-1,:]
    
    
    S3 = Process(sigma0, a3, m, alpha, rho,S0,b[2],c[2]).simulate_spot(T,dt,"Quadratic")
    F3= Process(sigma0,a3,m, alpha,rho,S0,b[2],c[2]).FutureDynamics(T,dt,S3,F0)
    F_T3 = F3[-1,:]
    S_T3 = S3[-1,:]
   
    call = np.zeros((len(K_list)))
    SE_call = np.zeros((len(K_list)))
     ##effective strike
    effective_K = 1-np.exp(-a*N_E)*(1-K_list/F0)
    
    
    for i in range(len(K_list)):
     call[i],SE_call[i] = Process(sigma0, a1,m, alpha, rho,S0,b[2],c[2]).OptionPricing(S_T1,K_list[i],r,dt,N_E,np.mean(F_T1),a1,"MC")

    for i in range(len(K_list[4:7])):
      call[i+4],SE_call[i+4] = Process(sigma0,a2,m, alpha, rho,S0,b[2],c[2]).OptionPricing(S_T2,K_list[i+4],r,dt,T,np.mean(F_T2),a2,"MC")
    
    for i in range(len(K_list[7:])):
      call[i+7],SE_call[i+7] = Process(sigma0,a3,m, alpha, rho,S0,b[2],c[2]).OptionPricing(S_T3,K_list[i+7],r,dt,T,np.mean(F_T3),a3,"MC")
    
