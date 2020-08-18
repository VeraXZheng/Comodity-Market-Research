#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Jul  9 14:55:51 2020

@author: Vera Zheng 
"""
import scipy.stats as si
from sympy.stats import Normal, cdf
import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
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
            param in the vol process if the dynamic is chosen to be "Linear"
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
        
        if dynamics == "Linear":
            sigma[0,:] = b * S0 + c
            for j in range (m):
                for i in range(n - 1):
               
                     sigma[i+1,j] = b * St[i+1,j] +c 
                     St[i+1,j] = St[i,j] + (1-St[i,j])* a * dt + sigma[i,j] * St[i,j] * dw1[i,j]
        
        if dynamics == "Stochastic":
             sigma[0,:] = sigma0
             dw2 = rho * dw1 + np.sqrt(1- rho **2) *dw1
             for j in range (m):
                 for i in range(n - 1):
                     dSigma = alpha *sigma[i,j] * dw2[i,j] 
                     sigma[i+1,j] = sigma[i,j] + dSigma
                     St[i+1,j] = St[i,j] + (1-St[i,j])* a * dt + sigma[i,j] * St[i,j] * dw1[i,j]
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
        n = round(T / (21*dt))
        Ft = np.zeros((n,m))
        Ft[0,:] = F0
        #save only date of interest 
        St = St[::21]
        for j in range(m):
            for i in range(n):
                Ft[i,j] = F0*(1-(1-St[i,j])*np.exp(a*(i*dt-T)))

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
    
   
def option_bls(sigma,K,T,S):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        call = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
        return call

     
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
    

   
def implied_vol(mkt_price, S, K, T_maturity, r, *args):
        Max_iteration = 500
        PRECISION = 1.0e-5
        sigma = 0.5
        for i in range(0, Max_iteration):
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            bls_price = S * norm.cdf(d1) - np.exp(-r * T) * K * norm.cdf(d2)
            vega = S * norm.pdf(d1) * np.sqrt(T)
            diff = mkt_price - bls_price  
            if (abs(diff) < PRECISION):
                return sigma
            sigma = sigma + diff/vega # f(x) / f'(x)
        return sigma 
  






      
   ###########################################################
### start main
if __name__ == "__main__":
    output_path = "/Users/veraisice/Desktop/Comodity-Market-Research/thesis_1/"
    input_path  = "/Users/veraisice/Desktop/Comodity-Market-Research/Input/"
   
    Future_prices = pd.read_excel(input_path+ "TTFdata"+".xlsx",sheet_name="Futures")     
   
    
    T = 2
    dt = 1/252
    S0 = 1
   
    sigma0 = 0.4
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
    
    
    
    
    #list of mean reverting speed of interest
    alist=[0,0.5,1]
    
   

    dynamics = "Stochastic"
    avgS = np.zeros((int(T/dt),len(alist)))
    avgF = np.zeros((T*12,len(alist)))
    call = np.zeros(len(alist))
    put = np.zeros(len(alist))
    SE_call = np.zeros(len(alist))
    SE_put = np.zeros(len(alist))
    for i in range(len(alist)):
        S = Process(sigma0, alist[i],m, alpha, rho,S0,blist[i],c).simulate_spot(T, dt,dynamics)
        F= Process(sigma0,a,m, alpha, rho,S0,blist[i],c).FutureDynamics(T,dt,S,F10)
        F_T=F[:,-1]
        call[i],SE_call[i] = Process(sigma0, a,m, alpha, rho,S0,blist[i],c).OptionPricing(S[:,-1], K0, r, dt, T,F_T,a,"bls")
        avgF[:,i] =  np.mean(F,1)
        avgS[:,i] = np.mean(S,1)
    print ("Call option prices by MC method with " +str(m)+ " simulations and different MR speed are " + str(call) +\
  
           "  with Standard Error for Call " +str(SE_call))
 #plot of spot process with linear vol
    
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
    m=100
    rho1 =0.4
    rho2 =0.3
    sigma1 =0.1
    sigma2 = 0.1
    alpha = 0.2
    for i in range(len(alist)):
        S1t = Process(sigma1, alist[i],m, alpha, rho1,S0,blist[i],c).Simulate_spot(T, dt)
        F1t = Process(sigma1,alist[i],m, alpha,rho1,S0,blist[i],c).FutureDynamics(T,dt,S1t,F0=F10)
        S2t = Process(sigma2, alist[i],m, alpha, rho2,S0,blist[i],c).Simulate_Stochastic(T, dt)
        F2t = Process(sigma2,alist[i],m, alpha, rho2,S0,blist[i],c).FutureDynamics(T,dt,S2t,F0=F20)
        V, avgV = AsianBasketOption(weight=[0.4,0.6]).pricing(0,len(F1t) , F1t, F2t)    
        print('Call price of the Asian basket Option with ' + str(m)+" simulations and mean reversion speed "+str(alist[i]) + "is " +str(avgV))     
           
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
    
    S = Process(sigma0, a, m, alpha, rho,S0,blist[2],c).simulate_spot(T,dt,"Linear")
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