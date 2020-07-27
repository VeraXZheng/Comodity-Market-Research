#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Jul  9 14:55:51 2020

@author: Vera Zheng 
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class ProcessOneFactor:

    def __init__(self, sigma0, a,m,alpha,rho):
     
        self._sigma0 = sigma0
        self._a = a
        self._m = m 
        self._alpha = alpha
        self._rho = rho
   
    def Simulate(self, T=1, dt=0.001, S0=1.):
        """
        function to model the normalised fictitious spot price
        Input:
        dt: time interval
        T: maturity
        S0: initial fictious spot price
        sigma0: Initial stochastic volatility
        m:    number of paths
        alpha: prarameters in loc vol thus controls the height of the ATM implied volatility level
        rho: controls the slope of the implied skew 
        Output:
        St: spot prices dynamics
            
        """
        #number of steps
        n = round(T / dt)
 
        sigma0 = self._sigma0
        
        a = self._a
        
        m = self._m
        
        alpha = self._alpha
        
        rho = self._rho
        
        sigma = np.zeros((n,m))
        sigma[0,:] = sigma0
        
        St = np.zeros((n,m))
        St[0,:] = S0
       
        sqrt_dt = dt ** 0.5
        
        dw1 = np.random.normal(size=(n,m))* sqrt_dt 
        dw2 = rho * dw1 + np.sqrt(1- rho **2) *dw1
        
        for j in range (m):
            for i in range(n - 1):
               dSigma = alpha *sigma[i,j] * dw2[i,j] 
               sigma[i+1,j] = sigma[i,j] + dSigma
               St[i+1,j] = St[i,j] + (1-St[i,j])* a * dt + sigma[i,j] * St[i,j]  * dw1[i,j]
        
        
        return St, sigma
   
    def FutureDynamics(self, T,dt,St,F0):
        """
        function to model the normalised fictitious spot price
        Input:
        dt: time interval
        T: maturity
        F0: Future price at the begining of the contact
        
        Output:
        Ft: spot prices dynamics
            
        """
        a = self._a
        n = round(T / (21*dt))
        Ft = np.zeros((n,m))
        Ft[0,:] = F0
        #save only date of interest 
        St = St[::21]
        for j in range(m):
            for i in range(n):
                Ft[i,j] = F0*(1-(1-St[i,j])*np.exp(a*(i*dt-T)))

        return Ft
   
    def OptionPricing(self,St,K0,r,dt,T):
        """
        function to price call and put options
        Input:
        dt: time interval
        T: maturity
       
        
        Output:
        Call: Call price
        Put: Put price
            
        """
        
        call = np.zeros(m)
        put = np.zeros(m)
    
        for i in range (m):
            call[i] = max(0,St[-1,i] - K0)
            put[i] = max(0,-St[-1,i] + K0)

        Call = np.exp(-r*T)* np.sum(call)/m
        Put = np.exp(-r*T)* np.sum(put)/m
        SE_call = np.sqrt(sum(call-Call)/(m*(m-1)))
        SE_put = np.sqrt(sum(put-Put)/(m*(m-1)))
        return Call, Put, SE_call, SE_put
        
 ###############main############
class AsianBasketOption:
    def __init__(self,weight):
       
        
        self._weight = weight
   
      
    def pricing(self,n1,n2,F1t,F2t):
        #[n1,n2]is the interval to take averge of the future prices
        weight = self._weight
        k=np.dot([F10,F20],weight)
        Ftot = weight[0]*np.mean(F1t[n1:n2,:],1)+weight[1]* np.mean(F2t[n1:n2,:],1)
      
        V = np.exp(-r*T)*np.maximum(Ftot - k,0)
        avgV = np.mean(V)
         
        return V, avgV
          
           
         
         
         
        
   ###########################################################
### start main
if __name__ == "__main__":
    output_path = "/Users/veraisice/Desktop/Comodity-Market-Research/thesis_1/"
    input_path  = "/Users/veraisice/Desktop/Comodity-Market-Research/Input/"
    Future_prices = pd.read_excel(input_path+ "1M_ForwardCurve"+".xlsx")     
    T = 2
    dt = 1/252
    sigma0 = 0.5
    alpha = 0.5
    rho = 0.5
    m = 1000
    K0 =1
    r =0.01
    F10 = Future_prices.loc[Future_prices['Month']== '2020-08-01','1-Month Future'].values[0]
    F20 = Future_prices.loc[Future_prices['Month']== '2020-09-01','1-Month Future'].values[0]
    #list of mean reverting speed of interest
    alist=[0,0.5,1]
    styl_list=['-','--','-.']
    # plot of the fictitious spot prices dynamics
    plt.figure(dpi=1200)
    for i in range(len(alist)):
        S,sigma = ProcessOneFactor(sigma0,alist[i], m, alpha, rho).Simulate(T, dt)
        avgS = np.mean(S,1)
        plt.plot(avgS, ls =styl_list[i], label=r"OneFactor(a =" + str(alist[i]) +", $\sigma_0$ ="+ str(sigma0)+")")
       
    plt.xlabel('Time index')
    plt.ylabel('Value')
    plt.title("Realization of fictious spot prices dynamics")
    plt.legend(loc='best')
    plt.savefig(output_path + "Figures/OneFactor_spot")
   # plot of the futures prices dynamics 
    plt.figure(dpi=1200)
    for i in range(len(alist)):
            St,sigma = ProcessOneFactor(sigma0, alist[i],m, alpha, rho).Simulate(T, dt)
            Ft = ProcessOneFactor(sigma0,alist[i],m, alpha, rho).FutureDynamics(T,dt,St,F10)
            avgFt =  np.mean(Ft,1)
            plt.plot(avgFt,ls =styl_list[i],label=r"OneFactor(a = "+ str(alist[i]) +", $\sigma_0$ = "+ str(sigma0)+")")
    
    plt.xlabel('Time index')
    plt.ylabel('Value')
    plt.title("Realization of Future processes")
    plt.legend(loc='best')
    plt.savefig(output_path + "Figures/OneFactor_futures")
#call and put prices
    call = np.zeros(len(alist))
    put = np.zeros(len(alist))
    SE_call = np.zeros(len(alist))
    SE_put = np.zeros(len(alist))
    for i in range(len(alist)):
        St,sigma = ProcessOneFactor(sigma0, alist[i],m, alpha, rho).Simulate(T, dt)
        
        call[i], put[i],SE_call[i], SE_put[i] = ProcessOneFactor(sigma0, alist[i],m, alpha, rho).OptionPricing(St, K0, r, dt, T)
       
    print ("Call and put option prices by MC method with " +str(m)+ " simulations and different MR speed are " + str(call) + " and "+str(put)+\
           " respectively; with Standard Error for Call " +str(SE_call)+ " and Put "+str(SE_put))
           
           
###################Asian Basket Option#########
    m=1000
    rho1 =0.4
    rho2 =0.3
    sigma1 =0.1
    sigma2 = 0.1
    alpha = 0.2
    for i in range(len(alist)):
        S1t,Sigma1 = ProcessOneFactor(sigma1, alist[i],m, alpha, rho1).Simulate(T, dt)
        F1t = ProcessOneFactor(sigma1,alist[i],m, alpha,rho1 ).FutureDynamics(T,dt,S1t,F0=F10)
        S2t,Sigma2 = ProcessOneFactor(sigma2, alist[i],m, alpha, rho2).Simulate(T, dt)
        F2t = ProcessOneFactor(sigma2,alist[i],m, alpha, rho2).FutureDynamics(T,dt,S2t,F0=F20)
        V, avgV = AsianBasketOption(weight=[0.4,0.6]).pricing(0,len(F1t) , F1t, F2t)    
        print('Call price of the Asian basket Option with ' + str(m)+" simulations and mean reversion speed "+str(alist[i]) + "is " +str(avgV))     
           
           
   