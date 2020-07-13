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
        n = round(T / dt)
        Ft = np.zeros((n,m))
        Ft[0,:] = F0
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
        F0: Future price at the begining of the contact
        
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
        return Call, Put
        
 ###############main############
class AsianBasketOption:
    def __init__(self, sigma1,sigma2,F1_0,F2_0,rho,m,k,weight):
     
        self._sigma1 = sigma1
        self._sigma2 = sigma2
        self._F1_0 = F1_0
        self._F2_0 = F2_0
        self._rho = rho
        self._m = m
        self._k = k 
        self._weight = weight
    def simulate(self,T, dt):
        F1_0 = self._F1_0
        F2_0 = self._F2_0
        rho = self._rho
        sigma1 = self._sigma1 
        sigma2 = self._sigma2 
        n = round(T / dt)
        F1t = np.zeros((n,m))
        F2t = np.zeros((n,m))
      
        for i in range(m):
            dw1 = np.random.normal(size= n)*np.sqrt(dt)
            dw2 = np.random.normal(size = n)*np.sqrt(dt)
            F1t[:,i] = F1_0 * np.cumprod(np.exp((-0.5*sigma1 **2)*dt+sigma1*dw1),0)
            F2t[:,i] = F2_0 * np.cumprod(np.exp((-0.5*sigma2 **2)*dt+sigma2 * (rho*dw1+np.sqrt(1-rho**2)*dw2)),0)
        return F1t, F2t
    
    def pricing(self,n1,n2,F1t,F2t):
        #[n1,n2]is the interval to take averge of the future prices
        weight = self._weight
        k = self._k
        Ftot = weight[0]*np.mean(F1t[n1:n2,:],1)+weight[1]* np.mean(F2t[n1:n2,:],1)
      
        V = np.exp(-r*T)*np.maximum(Ftot - k,0)
        avgV = np.mean(V)
         
        return V, avgV
          
           
         
         
         
        
   ###########################################################
### start main
if __name__ == "__main__":
    T = 10
    dt = 0.05
    sigma0 = 0.5
    alpha = 0.5
    rho = 0.5
    m = 1
    K0 =1
    r =0.01
    #list of mean reverting speed of interest
    alist=[0,0.5,1]
    # plot of the fictitious spot prices dynamics
    plt.figure()
    for a in alist:
        S,sigma = ProcessOneFactor(sigma0,a, m, alpha, rho).Simulate(T, dt)
        
        plt.plot(S, label=r"OneFactor(a =" + str(a) +", $\sigma_0$ ="+ str(sigma0)+")")
       
    plt.xlabel('Time index')
    plt.ylabel('Value')
    plt.title("Realization of fictious spot prices dynamics")
    plt.legend(loc='best')
   # plot of the futures prices dynamics 
    plt.figure()
    for a in alist:
            St,sigma = ProcessOneFactor(sigma0, a,m, alpha, rho).Simulate(T, dt)
            plt.plot(ProcessOneFactor(sigma,a,m, alpha, rho).FutureDynamics(T,dt,St,F0=1),label=r"OneFactor(a = "+ str(a) +", $\sigma_0$ = "+ str(sigma0)+")")
    
    plt.xlabel('Time index')
    plt.ylabel('Value')
    plt.title("Realization of Future processes")
    plt.legend(loc='best')

#call and put prices
    call = np.zeros(len(alist))
    put = np.zeros(len(alist))
    for i in range(len(alist)):
        St,sigma = ProcessOneFactor(sigma0, alist[i],m, alpha, rho).Simulate(T, dt)
        call[i], put[i] = ProcessOneFactor(sigma0, alist[i],m, alpha, rho).OptionPricing(St, K0, r, dt, T)
    print ("Call and put option prices by MC method with" +str(m)+ " simulations and different MR speed are " + str(call) + " and "+str(put)+" respectively")
           
           
###################Asian Basket Option#########
    m=1000
    F1t,F2t = AsianBasketOption(sigma1=0.3, sigma2=0.4, F1_0=100, F2_0=80, rho=0.5, m=1000, k=88, weight=(0.4,0.6)).simulate(1,1/250)     
    V, avgV = AsianBasketOption(sigma1=0.3, sigma2=0.4, F1_0=100, F2_0=80, rho=0.5, m=1000, k=88, weight=(0.4,0.6)).pricing(100, 200, F1t, F2t)    
    print('Call price of the Asian basket Option with ' + str(m)+" simulations is " +str(avgV))     
           
           
           