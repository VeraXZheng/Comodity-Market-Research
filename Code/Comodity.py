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
class Process:
    
    def __init__(self, sigma0, a,m,alpha,nu,rho,beta,S0,b,c):
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
        self._c = c
        self._nu = nu
        self._beta = beta
    
    def simulate_spot(self,T,dynamics):
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
        beta = self._beta
        St = np.zeros((n+1,m))
        Yt = np.zeros((n+1,m))
        St[0,:] = S0
        alpha = self._alpha
        nu = self._nu
        rho = self._rho
        sigma = np.zeros((n+1,m))
        sigma0 = self._sigma0
        sqrt_dt = dt ** 0.5
        mP=int(m*0.5)
        dw1 = np.random.normal(size=(n,mP))* sqrt_dt 
        
        if dynamics == "Quadratic":
            sigma[0,:] = sigma0
            for i in range (n-1):
                for j in range(mP):
                     vol = b * St[i,j] +c 
                     Yt[i+1,j]=Yt[i,j]+vol*  dw1[i,j]+a*dt*(1-St[i,j])/St[i,j]-0.5*vol**2*dt
                     vol = b * St[i,j+mP] +c 
                     Yt[i+1,j+mP]=Yt[i,j+mP]+vol*  dw1[i,j]+a*dt*(1-St[i,j+mP])/St[i,j+mP]-0.5*vol**2*dt
                     St[i+1,j]=np.exp(Yt[i+1,j])
                     St[i+1,j+mP]=np.exp(Yt[i+1,j+mP])
        if dynamics == "Stochastic":
             sigma[0,:] = alpha
             
             dw2 = np.random.normal(size=(n,mP))* sqrt_dt 
             dw3 = rho * dw1 + np.sqrt(1- rho **2) *dw2
             for i in range (n):
                 for j in range(mP):
                      sigma[i+1,j] = sigma[i,j] + nu *sigma[i,j] * dw3[i,j]
                      eta=sigma[i,j]*St[i,j]**(beta-1)
                      Yt[i+1,j]=Yt[i,j]+ eta*  dw1[i,j]+a*dt*(1-St[i,j])/St[i,j]-0.5* eta**2*dt
                      
                      sigma[i+1,j+mP] = sigma[i,j+mP] -  nu *sigma[i,j+mP] * dw3[i,j] 
                      eta=sigma[i,j+mP]*St[i,j+mP]**(beta-1)
                     
                      Yt[i+1,j+mP]=Yt[i,j+mP]- eta*dw1[i,j]+a*dt*(1-St[i,j+mP])/St[i,j+mP]-0.5* eta**2*dt
                      St[i+1,j]=np.exp(Yt[i+1,j])
                      St[i+1,j+mP]=np.exp(Yt[i+1,j+mP])  
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
        
        Ft = F0*(1-(1-St)*np.exp(a*(-T)))

        return Ft
   
    def OptionPricing(self,S_T,K,r,T,F_T,a,method):
        """
        function to price call options on the futures 
        Input:
       
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
   
    def simulate_correlated_paths(self,corr,nSims,nAssets,dynamics,F0):
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
        T = self._T
        nSteps = T*365
        
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
            d1 = (np.log(F / K) + (0.5 * sigma ** 2) * T_maturity) / (sigma * np.sqrt(T_maturity))
            d2 = d1 - sigma * np.sqrt(T_maturity)
            bls_price = F * ndtr(d1) - np.exp(-r * T_maturity) * K * ndtr(d2)
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
         "to get the constant param in DD process"
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
        d1 = (np.log(f0 / k0) + (r + 0.5 * gamma_sigma[1] ** 2) * T) / (gamma_sigma[1] * np.sqrt(T))
        d2 = d1 - gamma_sigma[1] * np.sqrt(T)
        #call_price = np.exp(a*T)/F *(f0*norm.cdf(d1)-k0*norm.cdf(d2))
       
        f1 = -2*a*(effective_K -1)*ndtr(d2)*np.exp(a*N_E)
        f2 = f1**2
        f3 = (f0 * norm._pdf(d1)*np.exp(0.5*a*N_E)*effective_K) **2/(k0**2*T)
        f4 = f0*norm._pdf(d1)/(F*np.sqrt(T))
        estimate_sigma = (f1+np.sqrt(f2+f3*(param[0]*effective_K +param[1])**2))/f4
        
        return estimate_sigma
    
     def object_func(self,gamma_sigma_list,F,K,T,N_E,param):
         
         
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
 
         
    
    
class Sabr:
    
    def SABR(self,alpha,rho,nu,beta,F,K,time,MKT):
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
            A = 1 + ( ((1-beta)**2*alpha**2)/(24.*(V**2)) + (alpha*beta*nu*rho)/(4.*V)+ ((nu**2)*(2-3*(rho**2))/24.) ) * time
            B = 1 + (1/24.)*(((1-beta)*logFK)**2) + (1/1920.)*(((1-beta)*logFK)**4) 
            VOL = (alpha/V)*A
            diff = VOL - MKT
        elif F != K: # not-ATM formula 
             V = (F*K)**((1-beta)/2.) 
             logFK = np.log(F/K)
             z = (nu/alpha)*V*logFK
             x =np.log( ( np.sqrt(1-2*rho*z+z**2) + z - rho ) / (1-rho) ) 
             A = 1 + ( ((1-beta)**2*alpha**2)/(24.*(V**2)) + (alpha*beta*nu*rho)/(4.*V) + ((nu**2)*(2-3*(rho**2))/24.) ) * time
             B = 1 + (1/24.)*(((1-beta)*logFK)**2) + (1/1920.)*(((1-beta)*logFK)**4)
             VOL = (nu*logFK*A)/(x*B) 
             diff = VOL - MKT
        return diff
  

  


    def smile(self,alpha,rho,nu,beta,F,K,time,MKT,i):
        """
        The smile function computes the implied volatilities for
        a given ”smile” pointed out by the index i. F, time and 
        the parameters are scalars; K and MKT are vectors.
        """
        for j in range(len(K)):
        
            self.SABR(alpha,rho,nu,beta,F,K[j],time,MKT[j])



    def SABR_vol_matrix(self,alpha,rho,nu,beta,F,K,time,MKT): 
        """function computes the implied volatilities for all 
        combinations of swaptions. F, time and the parameters are vectors; 
        K and MKT are matrices.
        """
        for i in range(len(F)):
            self.smile(alpha[i],rho[i],nu[i],beta[i],F[i],K[i],time[i],MKT[i],i)


    def objfunc(self,par,beta,F,K,time,MKT): 
        sum_sq_diff = 0
        alpha,rho,nu=par
        for j in range(len(K)):
            diff = self.SABR(alpha,rho,nu,beta,F,K[j],time,MKT[j])  
            sum_sq_diff = sum_sq_diff + diff**2
        obj = np.sqrt(sum_sq_diff) 
        return obj
    
    
    
    
    def calibration(self,starting_par,beta,F,K,time,MKT): 
        """The function used to calibrate each smile 
        (different strikes within the same tenor"""
        alpha = np.zeros(len(F))
        nu = np.zeros(len(F))
        rho = np.zeros(len(F))
        for i in range(len(F)):
            x0 = starting_par
            bnds = ( (0.001,None) ,(-0.9999,0.9999), (0.001,None) ) 
            Diff = lambda param: self.objfunc(param,beta,F[i],K[i],time[i],MKT[i])
            res = opt.minimize(Diff,x0, method="SLSQP",bounds = bnds) 
            alpha[i] = res.x[0]
            rho[i] = res.x[1]
            nu[i] = res.x[2]

        return alpha, rho,nu

    

###############################################################

### start main
if __name__ == "__main__":
    output_path = "/Users/veraisice/Desktop/Comodity-Market-Research/thesis_1/"
    input_path  = "/Users/veraisice/Desktop/Comodity-Market-Research/Input/"
   
    Future_prices = pd.read_excel(input_path+ "TTFdata"+".xlsx",sheet_name="Futures")     
   
    
    T = 1
    dt = 1/365
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
    m=50000
    klist = np.linspace(0.5*np.mean(Future),1.5*np.mean(Future),10)
    b=0.15
    c=0.4
    alpha = 0.1
    alpha_list =[0.002,0.004,0.006,0.008,0.010,0.012,0.014,0.016,0.018,0.20]
    a =0.1
    alist=np.linspace(0.0,0.8,40)
    blist=np.linspace(0.0,0.2,20)
    clist=np.linspace(0,0.08,30)
    corr = ([1,0.2],[0.2,1])#([1, 0.5,0.4],[0.5,1,0.5],[0.4,0.5,1]) #([1,0.5],[0.5,1])
    corr_list = [[[1, 0.2,0.1],[0.2,1,0.2],[0.1,0.2,1]],[[1, 0.2,0.3],[0.3,1,0.3],[0.3,0.3,1]],[[1, 0.4,0.3],[0.4,1,0.4],[0.3,0.4,1]],[[1, 0.5,0.4],[0.5,1,0.5],[0.4,0.5,1]],[[1, 0.6,0.5],[0.6,1,0.6],[0.5,0.6,1]],[[1, 0.7,0.6],[0.7,1,0.7],[0.6,0.7,1]],[[1, 0.8,0.7],[0.8,1,0.8],[0.8,0.7,1]],[[1, 0.9,0.8],[0.9,1,0.9],[0.8,0.5,0.9]]]
    nSims =10000
    nSteps = int(T/dt)
    F0 = [5.495, 5.805]#[5.495, 5.805, 6.49]
    nAssets = 3 #2
    dynamics = "Quadratic"
    weight=[1/3 , 1/3 ,1/3]#[1/2,1/2] #[1/3 , 1/3 ,1/3] #[1/2,1/2]
    avgV= np.zeros((len(clist)))
    
    T=1/3
    for i in range(len(clist)):
       
       
        S = Process(sigma0, a, m, alpha,nu,rho,S0,b,clist[i]).simulate_spot(T,dt,"Quadratic")
        F1= np.ones(len(S))
        F2=np.ones(len(S))
        F3=np.ones(len(S))
        F1= Process(sigma0,a,m, alpha,nu,rho,S0,b,c).FutureDynamics(T,dt,S,Future[0])
        F2= Process(sigma0,a,m, alpha,nu,rho,S0,b,c).FutureDynamics(T,dt,S,Future[1])
        F3= Process(sigma0,a,m, alpha,nu,rho,S0,b,c).FutureDynamics(T,dt,S,Future[2])
        Ft = np.dstack((F1,F2,F3))
        k=np.dot(Ft[0,:,:],weight)
        V, avgV[i] = AsianBasketOption(weight,alist[i],b,c,alpha,dt,T).pricing( len(Ft)-3,len(Ft)-1, Ft,k)    
    
    plt.figure(dpi=1000)
    plt.plot(alist,avgV,label="Asian Basket Price")
    #plt.plot(np.linspace(0.1,0.8,len(corr_list)),avgV,label="Asian Basket Price")
    plt.xticks(rotation=45, ha='right')
    plt.xlabel(r"$a$")
    plt.ylabel("Price")
    plt.title(r"Effect of Parameter $a$ in the " +str(dynamics) + " Model" )
    plt.legend(loc= 'best')
    plt.savefig(output_path+"Figures/SLV_c")
    
    
    
 #for sabr 
    dynamics = "stochastic"
    alpha =0.5
    nu=0.4
    rho=-0.1
    nu_list = np.linspace(0,1,20)
    rho_list = np.linspace(-0.5,0.5,20)
    alpha_list = np.linspace(0,0.8,20)
    weight=[1/3 , 1/3 ,1/3]#[1/2,1/2] #[1/3 , 1/3 ,1/3] #[1/2,1/2]
   
    avgV= np.zeros((len(nu_list)))
   
    for i in range(len(nu_list)):
       
       
        S = Process(sigma0, a, m, alpha,nu_list[i],rho,S0,b,c).simulate_spot(T,dt,"Stochastic")
        F1= np.ones(len(S))
        F2=np.ones(len(S))
        F3=np.ones(len(S))
        F1= Process(sigma0,a,m, alpha,nu,rho,S0,b,c).FutureDynamics(T,dt,S,Future[0])
        F2= Process(sigma0,a,m, alpha,nu,rho,S0,b,c).FutureDynamics(T,dt,S,Future[1])
        F3= Process(sigma0,a,m, alpha,nu,rho,S0,b,c).FutureDynamics(T,dt,S,Future[2])
        Ft = np.dstack((F1,F2,F3))
        k=np.dot(Ft[0,:,:],weight)
        V, avgV[i] = AsianBasketOption(weight,a,b,c,alpha,dt,T).pricing( len(Ft)-3,len(Ft)-1, Ft,k)    
    
    plt.figure(dpi=1000)
    plt.plot(nu_list,avgV,label="Asian Basket Price")
    #plt.plot(np.linspace(0.1,0.8,len(corr_list)),avgV,label="Asian Basket Price")
    plt.xticks(rotation=45, ha='right')
    plt.xlabel(r"$\nu$")
    plt.ylabel("Price")
    plt.title(r"Effect of Parameter $\nu$ in the " +str(dynamics) + " Model" )
    plt.legend(loc= 'best')
    plt.savefig(output_path+"Figures/SLV_nu")
    
    
    
    
    
    avgV= np.zeros((len(rho_list)))
   
    for i in range(len(rho_list)):
       
       
        S = Process(sigma0, a, m, alpha,nu,rho_list[i],S0,b,c).simulate_spot(T,dt,"Stochastic")
        F1= np.ones(len(S))
        F2=np.ones(len(S))
        F3=np.ones(len(S))
        F1= Process(sigma0,a,m, alpha,nu,rho,S0,b,c).FutureDynamics(T,dt,S,Future[0])
        F2= Process(sigma0,a,m, alpha,nu,rho,S0,b,c).FutureDynamics(T,dt,S,Future[1])
        F3= Process(sigma0,a,m, alpha,nu,rho,S0,b,c).FutureDynamics(T,dt,S,Future[2])
        Ft = np.dstack((F1,F2,F3))
        k=np.dot(Ft[0,:,:],weight)
        V, avgV[i] = AsianBasketOption(weight,a,b,c,alpha,dt,T).pricing( len(Ft)-3,len(Ft)-1, Ft,k)    
    
    plt.figure(dpi=1000)
    plt.plot(rho_list,avgV,label="Asian Basket Price")
    #plt.plot(np.linspace(0.1,0.8,len(corr_list)),avgV,label="Asian Basket Price")
    plt.xticks(rotation=45, ha='right')
    plt.xlabel(r"$\rho$")
    plt.ylabel("Price")
    plt.title(r"Effect of Parameter $\rho$ in the " +str(dynamics) + " Model" )
    plt.legend(loc= 'best')
    plt.savefig(output_path+"Figures/SLV_rho")
    
    avgV= np.zeros((len(alpha_list)))
   
    for i in range(len(alpha_list)):
       
       
        S = Process(sigma0, a, m, alpha_list[i],nu,rho,S0,b,c).simulate_spot(T,dt,"Stochastic")
        F1= np.ones(len(S))
        F2=np.ones(len(S))
        F3=np.ones(len(S))
        F1= Process(sigma0,a,m, alpha,nu,rho,S0,b,c).FutureDynamics(T,dt,S,Future[0])
        F2= Process(sigma0,a,m, alpha,nu,rho,S0,b,c).FutureDynamics(T,dt,S,Future[1])
        F3= Process(sigma0,a,m, alpha,nu,rho,S0,b,c).FutureDynamics(T,dt,S,Future[2])
        Ft = np.dstack((F1,F2,F3))
        k=np.dot(Ft[0,:,:],weight)
        V, avgV[i] = AsianBasketOption(weight,a,b,c,alpha,dt,T).pricing( len(Ft)-3,len(Ft)-1, Ft,k)    
    
    plt.figure(dpi=1000)
    plt.plot(rho_list,avgV,label="Asian Basket Price")
    #plt.plot(np.linspace(0.1,0.8,len(corr_list)),avgV,label="Asian Basket Price")
    plt.xticks(rotation=45, ha='right')
    plt.xlabel(r"$\alpha$")
    plt.ylabel("Price")
    plt.title(r"Effect of Parameter $\alpha$ in the " +str(dynamics) + " Model" )
    plt.legend(loc= 'best')
    plt.savefig(output_path+"Figures/SLV_alpha")
    




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

    #
    a=0
    Month_List = np.array(["October","November","December"])
    c= np.ones(len(Month_List))
    b= np.ones(len(Month_List))
    N_E = np.ones((len(Month_List)))
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
    #expiry to notification date
        N_E[i]=Option_Data["N-E"].values[0]
    #future price
        Future[i] = Option_Data["1-Month Future"].values[0]
        gamma_sigma_list[i]= calibration_quadratic(a, K, T_M, Future).optimise(np.asarray(K[i],float), Future[i],T_M[i],np.asarray(market_price[i],float),param=[])
        print('gamma',gamma_sigma_list[i][0])
        print('sigma',gamma_sigma_list[i][1])
        print('error', calibration_quadratic(a, K, T_M, Future).sum_difference(K[i],Future[i],T_M[i],market_price[i],gamma_sigma_list[i]))
 
    
        b[i],c[i]=calibration_quadratic(a, K, T_M, Future).find_param(gamma_sigma_list[i],Future[i],K[i],T_M[i],N_E[i],param=[])
        
        print("Calibration Result for "+str(Month_List[i])+ " is b=" +str(b[i])+" and c=" +str(c[i]))
        print('Error',calibration_quadratic(a, K, T_M, Future).object_func(gamma_sigma_list[i],Future[i],K[i],T_M[i],N_E[i],[b[i],c[i]]))
    #####calculate the model price######
    Month ="November"#“November” “December”
    nM=1
    sigma0= gamma_sigma_list[nM][1]
    
    gamma0= gamma_sigma_list[nM][0]
   
    a=0
    m = 100000
    alpha = 0
    rho =0
    S0=1
    #
    
    dt=1/365
   
    
    
    ones =np.ones(np.size(K[nM]))   
    
    
   
    S = Process(sigma0, a, m, alpha,nu, rho,S0,b[nM],c[nM]).simulate_spot(T[nM],N_E[nM],"Quadratic")
   
    S_T = S[-1,:]
    
    Call = market_price[nM]
   
    call = np.zeros((len(K[nM])))
    SE_call = np.zeros((len(K[nM])))
    market_price_dd=np.zeros((len(K[nM])))
    for i in range(len(K[nM])):
       call[i],SE_call[i] = Process(sigma0, a ,m, alpha,nu,rho,S0,b[nM],c[nM]).OptionPricing(S_T,K[nM][i],r,N_E[nM],Future[nM],a,"MC")#N_E
       market_price_dd[i]=BS(Future[nM]+gamma0,K[nM][i]+gamma0,sigma0,T_M[nM],0) 
    
    
    params = np.vstack((call, Future[nM]*ones,K[nM], T_M[nM]*ones, r*ones))
    vols = list(map(implied_vol, *params))
   
    effective_K = 1-np.exp(-a*N_E[nM])*(1-K[nM]/Future[nM])
   
    ###implied vol from the market prices
    params_mkt = np.vstack((market_price_dd, Future[nM]*ones,K[nM], T_M[nM]*ones, r*ones))
    vols_mkt = list(map(implied_vol, *params_mkt))
   
   
    
    plt.figure(dpi=1000)
    plt.plot(effective_K,vols,'--b*',label="Model IV")
    plt.plot(effective_K,vols_mkt,'--r*',label="Market IV")
    plt.xlabel("Strike")
    plt.ylabel("Implied Volatility")
    plt.title("Implied Volatilities of TTF Future Style Option Expires in " + str(Month)  )
    plt.legend(loc= 'best')
    plt.savefig(output_path + "Figures/LV_model_mkt_vol"+str(Month))
    
    
    
    plt.figure(dpi=1000)
    plt.plot(effective_K,Call,'--b*',label="Market Prices VS Strikes")
    plt.plot(effective_K,market_price_dd,linestyle='None',color='g',marker='o',label="Model Prices VS Strikes")
    
    plt.xlabel("Strike")
    plt.ylabel("Option Price")
    plt.title("Comparison of TTF Future Style Option Price Expired in " + str(Month)  )
    plt.legend(loc= 'best')
    plt.savefig(output_path + "Figures/LV_model_mkt_price"+str(Month))






###Calibration Sabr####
    
    Month_List = Month_List = ["October","November","December"]
    n_month=len(Month_List)
    N_E = np.ones((len(Month_List)))
    T_M = np.ones((len(Month_List)))
    Future= np.ones((len(Month_List)))
    market_price=np.zeros((len(Month_List),10), dtype=np.ndarray)
    K=np.zeros((len(Month_List),10), dtype=np.ndarray)
    vols_mkt = np.zeros((len(Month_List),10), dtype=np.ndarray)
    vols_mdl = np.zeros((len(Month_List),10), dtype=np.ndarray)
    effective_K=np.zeros((len(Month_List),10), dtype=np.ndarray)
    
    #magic number
    a =0.
    m=500
    dt=1/365
    np.random.seed(111)
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
       
   


    beta=0.5
    # Future_Y = np.exp(-a*N_E)*Future
    # K_Y=np.dot(K*np.exp(-a*N_E).reshape(3,1),1)
    # starting_par = np.array([0.1,0.1,0.1])
    # [alpha,rho,nu] = Sabr().calibration(starting_par,beta,Future_Y,K_Y,T_M,vols_mkt)
    
    # call = np.zeros((len(Month_List),10), dtype=np.ndarray)
    # SE_call= np.zeros((len(Month_List),10), dtype=np.ndarray)
    # market_price_dd= np.zeros((len(Month_List),10), dtype=np.ndarray)
    # alpha_spot= alpha*(np.exp(-a*N_E)*Future)**(beta-1)
    # nu_spot= nu*(np.exp(-a*N_E)*Future)**(beta-1)
    Future_Y = np.exp(-a*N_E)*Future
    K_Y=K/Future.reshape(3,1)
    starting_par = np.array([0.1,0.1,0.1])
    [alpha,rho,nu] = Sabr().calibration(starting_par,beta,np.ones(len(Future)),K_Y,T_M,vols_mkt)
    
    call = np.zeros((len(Month_List),10), dtype=np.ndarray)
    SE_call= np.zeros((len(Month_List),10), dtype=np.ndarray)
   
    # alpha_spot= alpha*(np.exp(-a*N_E)*Future)**(beta-1)
    # nu_spot= nu*(np.exp(-a*N_E)*Future)**(beta-1)
    #dynamic of spot
   
    for i in range(len(Month_List)):
        
        St= Process(sigma0,a,m,alpha[i],nu[i],rho[i],beta,1,b,c).simulate_spot(T_M[i],"Stochastic")
        S_T = St[-1,:]
        for j in range(len(K[i])):
               call[i,j],SE_call[i,j] = Process(sigma0, a,m, alpha, nu,rho,beta,S0,b,c).OptionPricing(S_T,K[i][j],r,T_M[i],Future[i],0.01,"MC")
               market_price_dd[i,j]=BS(Future[i]+gamma_sigma_list[i][0],K[i][j]+gamma_sigma_list[i][0],gamma_sigma_list[i][1],T_M[i],0) 
        params_mdl = np.vstack((call[i,:], Future[i]*ones,K[i], T_M[i]*ones))
       
       
   
       
        vols_mdl[i] = list(map(implied_vol, *params_mdl))
       
    for i in range(len(Month_List)): 
        plt.figure(dpi=1000)
        plt.plot(effective_K[i],call[i],linestyle='None',color='g',marker='o',label="Model Prices VS Strikes")
        plt.plot(effective_K[i],market_price[i],'--r*',label="Market Prices VS Strikes")
        plt.xlabel("Strike")
        plt.ylabel("Option Price")
        plt.title("Comparison of TTF Futures Option Price Expired in " + str(Month_List[i])  )
        plt.legend(loc= 'best')
        plt.savefig(output_path + "Figures/price_mdl_mkt"+str(Month_List[i]))
    for i in range(len(Month_List)):
         plt.figure(dpi=1000)
         plt.plot(effective_K[i],vols_mdl[i],'--b*',label="Model IV")
         plt.plot(effective_K[i],vols_mkt[i],'--r*',label="Market IV")
         plt.xlabel("Strike")
         plt.ylabel("Implied Volatility")
         plt.title("Implied Volatilities of TTF Futures Options Expires in " + str(Month_List[i])  )
         plt.legend(loc= 'best')
         plt.savefig(output_path + "Figures/vol_mdl_mkt"+str(Month_List[i]))
         
    