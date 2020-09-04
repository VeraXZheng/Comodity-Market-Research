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
        n = round(T / dt)
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

        Call = np.exp(-r*T)* np.sum(call)/m  *np.exp(-a*T)
        
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
        Ft = F0*(1-(1-St)*np.exp(a*(i*dt-T)))  
        return Ft   
    
    def pricing(self,n1,n2,Ft,k):
      
        #[n1,n2]is the interval to take averge of the future prices
        weight = self._weight
        #k=np.dot(Ft[0,:,:],weight)
   
        Ftot = np.dot(np.mean(Ft[n1:n2,:,:],0),weight)
   
        V = np.exp(-r*T)*np.maximum(Ftot - k,0)
        avgV = np.mean(V)
    
         
        return V, avgV
    

   
def implied_vol(mkt_price, F, K, T_maturity, r,gamma,Type, *args):
        Max_iteration = 5000
        PRECISION = 1.0e-5
        sigma = 0.5
        #1 for market
        if Type ==1:
            for i in range(0, Max_iteration):
                d1 = (np.log(F/ K) + (r + 0.5 * sigma ** 2) * T_maturity) / (sigma * np.sqrt(T_maturity))
                d2 = d1 - sigma * np.sqrt(T)
                bls_price = F * norm.cdf(d1) - np.exp(-r * T_maturity) * K * norm.cdf(d2)
                vega = F * norm.pdf(d1) * np.sqrt(T_maturity)
                diff = mkt_price - bls_price  
                if (abs(diff) < PRECISION):
                    return sigma
            sigma = sigma + diff/vega # f(x) / f'(x)
        
        if Type==2:
           
            K=1-np.exp(-a*T)*(1-K/F0)
            f0 = F + gamma
            k0 = K*F + gamma
       
            for i in range(0, Max_iteration):
            
                d1 = (np.log(f0 / k0) + (r + 0.5 * sigma ** 2) * T_maturity) / (sigma * np.sqrt(T_maturity))
         
                d2 = d1 - sigma * np.sqrt(T_maturity)
                bls_price = np.exp(a*T_maturity) *(f0*norm.cdf(d1)-k0*norm.cdf(d2))/F
       
            # d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            # d2 = d1 - sigma * np.sqrt(T)
            # bls_price = S * norm.cdf(d1) - np.exp(-r * T) * K * norm.cdf(d2)
                vega =   np.exp(a*T_maturity)*f0* norm.pdf(d1) * np.sqrt(T_maturity)/F
                diff = mkt_price - bls_price  
                if (abs(diff) < PRECISION):
                    return sigma
            sigma = sigma + diff/vega # f(x) / f'(x)
        
        return sigma 
  


class calibration:
     def __init__(self,a,K,T_M,Future):
     
        self._K =K
        self._T_M =T_M
        self._a = a
        self._Future = Future
      
        
        
        
     def find_constant(self,initial_guess,market_price,T,K,F):
      
        def price_difference(initial_guess):
            d1 = (np.log((F+initial_guess[0]) / (K+initial_guess[0])) + (r + 0.5 * initial_guess[1] ** 2) * T) / (initial_guess[1] * np.sqrt(T))
            d2 = d1 - initial_guess[1] * np.sqrt(T)
            bls_price = (F+initial_guess[0]) * norm.cdf(d1) - np.exp(-r * T) * K * norm.cdf(d2)
            difference = bls_price - market_price
            return difference
        return fsolve(price_difference,initial_guess)

   
     def estimateSigma(self,gamma_sigma,param,K,F,T):
       
       
        a= self._a
        
        param=np.exp(param) #transform to restric a be postive 
        effective_K = 1-np.exp(-a*T)*(1-K/F)
        f0 = F + gamma_sigma[0]
        k0 = effective_K*F+gamma_sigma[0]
        d1 = (np.log(f0 / k0) + (r + 0.5 * gamma_sigma[1] ** 2) * T) / (gamma_sigma[1] * np.sqrt(T))
        d2 = d1 - gamma_sigma[1] * np.sqrt(T)
        #call_price = np.exp(a*T)/F *(f0*norm.cdf(d1)-k0*norm.cdf(d2))
       
        f1 = -2*a*(K-1)*norm.cdf(d2)*np.exp(a*T)
        f2 = 4*a**2*(K-1)**2*norm.cdf(d2)**2*np.exp(2*a*T)
        f3 = f0**2 * norm.pdf(d1)**2*np.exp(a*T)*K**2
        f4 = f0*norm.pdf(d1)/(F*np.sqrt(T))
        estimate_sigma = (f1+np.sqrt(f2+(f3*(param[0]*K+param[1])**2)))/f4
        
        return estimate_sigma
     
     def obeject_func(self,gamma_sigma_list,param):
         estimate_sigma = np.ones((len(gamma_sigma_list)))
         F = self._Future
         K = self._K
         T = self._T_M
         for i in range(len(gamma_sigma_list)):
             estimate_sigma[i]= self.estimateSigma(gamma_sigma_list[i],param,K[i][1],F[i],T[i])
         return  sum(abs(gamma_sigma_list[:,0]-estimate_sigma))
    
     def find_param(self,gamma_sigma_list,param):
         start_params = np.array([0.01, 0.1])
         
         difference = lambda param: self.obeject_func(gamma_sigma_list,param)
         #bnds = ((0,1),(0,1))
         all_results = opt.minimize(fun=difference, x0=start_params,
                                      method="BFGS")#bounds=bnds)
         return np.exp(all_results.x)#all_results.x #np.exp(all_results.x) #with transformation
    

      
   ###########################################################
### start main
if __name__ == "__main__":
    output_path = "/Users/veraisice/Desktop/Comodity-Market-Research/thesis_1/"
    input_path  = "/Users/veraisice/Desktop/Comodity-Market-Research/Input/"
   
    Future_prices = pd.read_excel(input_path+ "TTFdata"+".xlsx",sheet_name="Futures")     
   
    
    T = 1
    dt = 1/252
    S0 = 1
   
    sigma0 = 0.1
    alpha = 0.5
    rho = 0.5
    m = 1000
    K0 = 1
    r =0.01
    
    a =0.01   
    blist = np.linspace(0,1,5)
    c = 0.3
    
    F10 = Future_prices.loc[Future_prices['Month']== '2020-08-25','1-Month Future'].values[0]
    F20 = Future_prices.loc[Future_prices['Month']== '2020-09-25','1-Month Future'].values[0]
    
    
    np.random.seed(123)
    
    #list of mean reverting speed of interest
    alist=[0,0.5,1]
    
   

    dynamics = "Stochastic"
    avgS = np.zeros((int(T/dt),len(alist)))
    avgF = np.zeros((T*12,len(alist)))
    call = np.zeros(len(alist))
    
    SE_call = np.zeros(len(alist))
   
    for i in range(len(alist)):
        S = Process(sigma0, alist[i],m, alpha, rho,S0,blist[i],c).simulate_spot(T, dt,dynamics)
        F= Process(sigma0,a,m, alpha, rho,S0,blist[i],c).FutureDynamics(T,dt,S,F10)
        S_T=S[-1,:]

        F_T=F[-1,:]
        call[i],SE_call[i] = Process(sigma0, a,m, alpha, rho,S0,blist[i],c).OptionPricing(S_T, K0, r, dt, T,F_T,a,"bls")
        avgF[:,i] =  np.mean(F,1)
        avgS[:,i] = np.mean(S,1)
    print ("Call option prices by MC method with " +str(m)+ " simulations and different MR speed are " + str(call) +\
  
           "  with Standard Error for Call " +str(SE_call))

    
    plt.figure(dpi=1000)
    for i in range(len(blist)):
        plt.plot(avgS[:,i],label=r"Linear(a =" + str(a)+ ", b = "+str(blist[i])+",c = "+str(c)+")")
    plt.xlabel('Time index')
    plt.ylabel('Value')
    plt.title("Realization of " +str(dynamics) +"vol spot prices dynamics")
    plt.legend(loc='best')
    #plt.savefig(output_path + "Figures/"+str(dynamics)+"linear_spot")
 
    ####plot of future process 
    plt.figure(dpi=1000)
    for i in range(len(blist)):
        plt.plot(avgF[:,i],label=r"Stochastic(a =" + str(a)+ ",b = "+ str(blist[i])+")")
    plt.xlabel('Time index')
    plt.ylabel('Value')
    plt.title("Realization of " +str(dynamics) +"  vol futures prices dynamics")
    plt.legend(loc='best')
    plt.savefig(output_path + "Figures/linear_futures")       
    
    
    
    
   
    

################linear vol #########################

#plot spot and futures prices with linear vol
    
    
    avgS = np.zeros((int(T/dt),len(blist)))
    avgF = np.zeros((T*12,len(blist)))
    call = np.zeros(len(blist))
    SE_call = np.zeros(len(blist))
   
    for i in range(len(blist)):
        S = Process(sigma0, a, m, alpha, rho,S0,blist[i],c).simulate_spot(T,dt,"Linear")
        F= Process(sigma0,a,m, alpha,rho,S0,blist[i],c).FutureDynamics(T,dt,S,F10)
        F_T = F[-1,:]
        S_T = S[-1,:]
        call[i],SE_call[i] = Process(sigma0, a,m, alpha, rho,S0,blist[2],c).OptionPricing(S_T, K0, r, dt, T,F_T,a,"bls")
        avgF[:,i] =  np.mean(F,1)
        avgS[:,i] = np.mean(S,1)
   #plot of spot process with linear vol
    plt.figure(dpi=1000) 
    for i in range(len(blist)):
        plt.plot(avgS[:,i],label=r"Linear(a =" + str(a)+ ", b = "+str(blist[i])+",c = "+str(c)+")")
    plt.xlabel('Time index')
    plt.ylabel('Value')
    plt.title("Realization of linear vol spot prices dynamics")
    plt.legend(loc='best')
    plt.savefig(output_path + "Figures/linear_spot")
  #plot of future process with linear vol  
    plt.figure(dpi=1000)
    for i in range(len(blist)):
        plt.plot(avgF[:,i],label=r"Stochastic(a =" + str(a)+ ",b = "+ str(blist[i])+")")
    plt.xlabel('Time index')
    plt.ylabel('Value')
    plt.title("Realization of linear vol futures prices dynamics")
    plt.legend(loc='best')
    plt.savefig(output_path + "Figures/linear_futures")       
  
  #plot option prices vs b
    plt.figure(dpi=1200)
    plt.plot(blist,call) 
    plt.xlabel("b value")
    plt.ylabel("European Call Price")
    plt.title("European Call Price vs vol Coefficient b")


###################Asian Basket Option#########
    m=1000
    K0 = Strike_list[10]
    b=0.01
    c=0.1
    alpha = 0.1
    alpha_list =[0.002,0.004,0.006,0.008,0.010,0.012,0.014,0.016,0.018,0.20]
    a =0.1
    alist=[0,0.15,0.3,0.45,0.6,0.75,0.9]
    blist=np.linspace(0.0,0.02,25)
    clist=np.linspace(0,0.08,30)
    corr = ([1,0.2],[0.2,1])#([1, 0.5,0.4],[0.5,1,0.5],[0.4,0.5,1]) #([1,0.5],[0.5,1])
    corr_list = [[[1, 0.2,0.1],[0.2,1,0.2],[0.1,0.2,1]],[[1, 0.2,0.3],[0.3,1,0.3],[0.3,0.3,1]],[[1, 0.4,0.3],[0.4,1,0.4],[0.3,0.4,1]],[[1, 0.5,0.4],[0.5,1,0.5],[0.4,0.5,1]],[[1, 0.6,0.5],[0.6,1,0.6],[0.5,0.6,1]],[[1, 0.7,0.6],[0.7,1,0.7],[0.6,0.7,1]],[[1, 0.8,0.7],[0.8,1,0.8],[0.8,0.7,1]],[[1, 0.9,0.8],[0.9,1,0.9],[0.8,0.5,0.9]]]
    nSims =10000
    nSteps = int(T/dt)
    F0 = [5.495, 5.805]#[5.495, 5.805, 6.49]
    nAssets = 2 #2
    dynamics = "Quadratic"
    weight=[1/2,1/2] #[1/3 , 1/3 ,1/3] #[1/2,1/2]
    avgV= np.zeros((len(blist)))
    for i in range(len(blist)):
        
        Ft = AsianBasketOption(weight,a,blist[i],c,alpha,dt,T).simulate_correlated_paths(corr,nSims,nAssets,nSteps,dynamics,F0)
        k=np.dot(Ft[0,:,:],weight)
        V, avgV[i] = AsianBasketOption(weight,a,blist[i],c,alpha,dt,T).pricing( len(Ft)-3,len(Ft)-1, Ft,k)    
        #print('Call price of the Asian basket Option with ' + str(nSims)+" simulations and mean reversion speed "+str(alist[i]) + " is " +str(avgV))     
   
    plt.figure(dpi=1000)
    plt.plot(blist,avgV,label="Asian Basket Price")
    #plt.plot(np.linspace(0.1,0.8,len(corr_list)),avgV,label="Asian Basket Price")
    plt.xticks(rotation=45, ha='right')
    plt.xlabel(r"$b$")
    plt.ylabel("Price")
    plt.title(r"Effect of Parameter $b$ in the " +str(dynamics) + " Model" )
    plt.legend(loc= 'best')
    plt.savefig(output_path+"Figures/SLV_b")
    
    
    
    
   


#Price vs Strike


    m=1000
    K0 = Strike_list
    b=0.01
    c=0.1
    alpha = 0.01
    alpha_list =[0.002,0.004,0.006,0.008,0.010,0.012,0.014,0.016,0.018,0.20]
    a =0.05
    alist=[0,0.15,0.3,0.45,0.6,0.75,0.9]
    blist=np.linspace(0,0.1,20)
    clist=np.linspace(0,0.1,30)
    corr = ([1,0.5],[0.5,1])#([1, 0.5,0.4],[0.5,1,0.5],[0.4,0.5,1]) #([1,0.5],[0.5,1])
    corr_list = [[[1, 0.2,0.1],[0.2,1,0.2],[0.1,0.2,1]],[[1, 0.2,0.3],[0.3,1,0.3],[0.3,0.3,1]],[[1, 0.4,0.3],[0.4,1,0.4],[0.3,0.4,1]],[[1, 0.5,0.4],[0.5,1,0.5],[0.4,0.5,1]],[[1, 0.6,0.5],[0.6,1,0.6],[0.5,0.6,1]],[[1, 0.7,0.6],[0.7,1,0.7],[0.6,0.7,1]],[[1, 0.8,0.7],[0.8,1,0.8],[0.8,0.7,1]],[[1, 0.9,0.8],[0.9,1,0.9],[0.8,0.5,0.9]]]
    nSims =10000
    nSteps = int(T/dt)
    F0 = [5.495, 5.805]#[5.495, 5.805, 6.49]
    nAssets = 2 #2
    dynamics = "Quadratic"
    weight=[1/2,1/2] #[1/3 , 1/3 ,1/3] #[1/2,1/2]
    avgV= np.zeros((len(Strike_list)))
    for i in range(len(Strike_list)):
        
        Ft = AsianBasketOption(weight,a,b,c,alpha,dt,T).simulate_correlated_paths(corr,nSims,nAssets,nSteps,dynamics,F0)
        V, avgV[i] = AsianBasketOption(weight,a,b,c,alpha,dt,T).pricing( -30,len(Ft)-1, Ft,Strike_list[i])    
        #print('Call price of the Asian basket Option with ' + str(nSims)+" simulations and mean reversion speed "+str(alist[i]) + " is " +str(avgV))     
   
    plt.figure(dpi=1000)
    plt.plot(Strike_list,avgV,label="Asian Basket Price")
    #plt.plot(np.linspace(0.1,0.8,len(corr_list)),avgV,label="Asian Basket Price")
    plt.xlabel(r"$K$")
    plt.ylabel("Price")
    plt.title(r"Effect of Parameter $Strike$ in the " +str(dynamics) + " Model" )
    plt.legend(loc= 'best')
    plt.savefig(output_path+"Figures/SLV_K")


 ######plot of European Option against parameters#######
    

    call = np.zeros(len(alist))
    
    SE_call = np.zeros(len(alist))
   
    for i in range(len(alist)):
        S = Process(sigma0, alist[i],m, alpha, rho,S0,b,c).simulate_spot(T, dt,"Linear")
       
        S_T = S[-1,:]
        call[i],SE_call[i] = Process(sigma0, alist[i],m, alpha, rho,S0,b,c).OptionPricing(S_T, K0, r, dt, T,Future,a,"bls")
    
    plt.figure(dpi=1000)
    plt.plot(alist,call,label="European Option Price")
    plt.xlabel(r"$a$")
    plt.ylabel("Price")
    plt.title(r"Effect of Parameter $\alpha$ in the" +str(dynamics) + " Model" )
    plt.legend(loc= 'best')
    plt.savefig(output_path+"Figures/SLV_a_EU")
           

    call = np.zeros(len(alist))
    
    SE_call = np.zeros(len(alist))
   
    for i in range(len(alist)):
        S = Process(sigma0, a,m, alist[i], rho,S0,b,c).simulate_spot(T, dt,dynamics)
       
        S_T = S[-1,:]
        call[i],SE_call[i] = Process(sigma0, a,m, alist[i], rho,S0,blist,c).OptionPricing(S_T, K0, r, dt, T,Future,a,"bls")
    
    plt.figure(dpi=1000)
    plt.plot(alist,call,label="European Option Price")
    plt.xlabel(r"$a$")
    plt.ylabel("Price")
    plt.title(r"Effect of Parameter $a$ in the SV Model"  )
    plt.legend(loc= 'best')
    plt.savefig(output_path+"Figures/SV_a_EU")



 #############implied vol##################### 
    Month = "Sep"
    Option_Data = pd.read_excel(input_path+ "TTFdata"+".xlsx",sheet_name = Month)    #SEPopt
    #strike 
    Strike_list = Option_Data["Strike"]
    #market option price
    Call_list = Option_Data["Call"]
    #time to maturity
    T_M = Option_Data["Time to Maturity"].values[0]
    #future price
    Future = Option_Data["1-Month Future"].values[0]
    ones =np.ones( np.size(Call_list) )        
    params = np.vstack((Call_list, Future*ones, Strike_list, T_M*ones, r*ones, sigma0*ones))
    vols = list(map(implied_vol, *params))
    
    plt.figure(dpi=1000)
    plt.plot(Strike_list,vols,label="Implied Volatilities")
    plt.xlabel("Strike")
    plt.ylabel("Implied Volatility")
    plt.title("Implied Volatilities of TTF Futures Options Expired in " + str(Month)  )
    plt.legend(loc= 'best')
    plt.savefig(output_path+"Figures/IV_sep")
    
    
    avgS = np.zeros((int(T/dt),len(Strike_list)))
    avgF = np.zeros((T*12,len(Strike_list)))
    call_bls = np.zeros(len(Strike_list))

    SE_call_mc = np.zeros(len(Strike_list))
    call_mc = np.zeros(len(Strike_list))

    SE_call_bls = np.zeros(len(Strike_list))
    
    S = Process(sigma0, a, m, alpha, rho,S0,blist[2],c).simulate_spot(T,dt,dynamics)
    # F= Process(sigma0,a,m, alpha,rho,S0,blist[2],c).FutureDynamics(T,dt,S,F10)

    S_T = S[-1,:]
    for i in range(len(Strike_list)):
            call_bls[i],SE_call_bls[i] = Process(sigma0, a,m, alpha, rho,S0,blist[1],c).OptionPricing(S_T, Strike_list[i], r, dt, T,Future*np.ones((1000)),a,"bls")
            call_mc[i],SE_call_mc[i] = Process(sigma0, a,m, alpha, rho,S0,blist[1],c).OptionPricing(S_T, Strike_list[i], r, dt, T,Future*np.ones((1000)),a,"MC")
            avgF[:,i] =  np.mean(F,1)
            avgS[:,i] = np.mean(S,1)
    plt.figure(dpi=1200)
    plt.plot(Strike_list,call_bls,"-o",label="Model Call Price (Black-Scholes)")
    plt.plot(Strike_list,call_mc,"-o",label="Model Call Price (Monte Carlo)")
    plt.plot(Strike_list,Call_list,"-o",label= "Market Call Price") 
    plt.legend(loc="best")
    plt.xlabel("Strike")
    plt.ylabel("European Call Price")
    plt.title("European Call Price vs Strike")
    plt.savefig(output_path + "Figures/option_LV")
    
    
    
    
    
    
    
    ####calibration##################
    r=0
    initial_guess=np.array([0.5,0.5])
    Month_List = np.array(["August","September","October"])
    T_M = np.ones((3))
    Future= np.ones((3))
    market_price=np.zeros((3,2), dtype=np.ndarray)
    K=np.zeros((3,2), dtype=np.ndarray)
    gamma_sigma_list= np.zeros((3,2), dtype=np.ndarray)
    for i in range(len(Month_List)):
        Option_Data = pd.read_excel(input_path+ "TTFdata"+".xlsx",sheet_name = Month_List[i])    #SEPopt
    #strike 
        K[i] = Option_Data["Strike"][2:4].values
    #market option price
        market_price[i] = Option_Data["Call"][2:4].values
    #time to maturity
        T_M [i]= Option_Data["Time to Maturity"].values[0]
    #future price
        Future[i] = Option_Data["1-Month Future"].values[0]
        gamma_sigma_list[i] = calibration(0.1,K,T_M,Future).find_constant(initial_guess, np.asarray(market_price[i],float), T_M[i], np.asarray(K[i],float), Future[i])
    
    
    [b,c]= calibration(0.1,K,T_M,Future).find_param(gamma_sigma_list,param=[])
    
    #####calculate the model price######
    Month ="Sep"
    sigma0= 0.1
    a = 0.1
    m = 1000
    alpha = 0.1
    rho =0.2
    S0=1
    T=1
    dt=T/252
    Option_data = pd.read_excel(input_path+ "TTFdata"+".xlsx",sheet_name = Month)
    F0 =  Option_data["1-Month Future"].values[0]   
    K_list = Option_data["Strike"].values 
    ##effective strike
    effective_K = 1-np.exp(-a*T)*(1-K_list/F0)
    
    
    
    #time to maturity
    T_M = Option_data["Time to Maturity"].values[0]
    Call_list = Option_data["Call"].values
    c_mkt = Call_list*np.exp(a*T_M)/F0
   
    S = Process(sigma0, a, m, alpha, rho,S0,b,c).simulate_spot(T,dt,"Quadratic")
    F= Process(sigma0,a,m, alpha,rho,S0,b,c).FutureDynamics(T,dt,S,F0)
    F_T = F[-1,:]
    S_T = S[-1,:]
    call = np.zeros((len(K_list)))
    SE_call = np.zeros((len(K_list)))
    for i in range(len(K_list)):
     call[i],SE_call[i] = Process(sigma0, a,m, alpha, rho,S0,b,c).OptionPricing(S_T,K_list[i], r, dt, T,F_T,0.01,"MC")
    
    c_model = call*np.exp(a*T_M)/F0
    ###implied vol from the model prices
    ones =np.ones(np.size(c_mkt))        
    params = np.vstack((call, F0*ones, K_list, T_M*ones, r*ones,gamma_sigma_list[1][0]*ones,2*ones))
    vols = list(map(implied_vol, *params))
    
    ###implied vol from the market prices
    params_mkt = np.vstack((Call_list, F0*ones,K_list, T_M*ones, r*ones,gamma_sigma_list[1][0]*ones,1*ones))
    vols_mkt = list(map(implied_vol, *params_mkt))
    
    
    plt.figure(dpi=1000)
    plt.plot(effective_K[5:],vols[5:],label="Model Implied Volatilities")
    plt.plot(effective_K[8:],vols_mkt[8:],label="Market Implied Volatilities")
    plt.xlabel("Strike")
    plt.ylabel("Implied Volatility")
    plt.title("Implied Volatilities of TTF Futures Options Expired in " + str(Month)  )
    plt.legend(loc= 'best')
    