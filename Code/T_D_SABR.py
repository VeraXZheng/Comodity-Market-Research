#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 18:12:31 2020

@author: veraisice
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
def sabr_spot(a,m,T,beta,nu,alpha):
        """
        function to model the normalised fictitious spot price
        Input:
        dt: time interval
        T: time to maturity
        
        Output:
        St: spot prices dynamics
            
        """
        dt=1/365
        np.random.seed(1)
        n = T / dt
        
        if np.size(T)==1:
            N=int(n)
            
        else:
            N = int(n[-1])
           
        St = np.zeros((N+1,m))
        Yt = np.zeros((N+1,m))
        St[0,:] = 1
      
        sigma = np.zeros((N+1,m))
        
        sqrt_dt = dt ** 0.5
        mP=int(m*0.5)
      
        dw1 = np.random.normal(size=(N,mP))* sqrt_dt
        
        dw2 = np.random.normal(size=(N,mP))* sqrt_dt 
        if np.size(T)==1:
            sigma[0,:]=alpha
            for i in range(N - 1):  
                    for j in range (mP):
                        eta=sigma[i,j]*St[i,j]**(beta-1)
                     
                        Yt[i+1,j]=Yt[i,j]+ eta*  dw1[i,j]+a*dt*(1-St[i,j])/St[i,j]-0.5* eta**2*dt
                      
                        sigma[i+1,j+mP] = sigma[i,j+mP] -  nu *sigma[i,j+mP] * dw2[i,j] 
                        eta=sigma[i,j+mP]*St[i,j+mP]**(beta-1)
                     
                        Yt[i+1,j+mP]=Yt[i,j+mP]- eta*dw1[i,j]+a*dt*(1-St[i,j+mP])/St[i,j+mP]-0.5* eta**2*dt
                        St[i+1,j]=np.exp(Yt[i+1,j])
                        St[i+1,j+mP]=np.exp(Yt[i+1,j+mP])  
            return St
       
        else:  
            n=np.insert(n,0,0,axis=0)    
            for k in range(len(T)):
                sigma[int(n[k]),:]=alpha[k]
                
                for i in range(int(n[k]),int(n[k+1])):
                    for j in range (mP): 
                        eta=sigma[i,j]*St[i,j]**(beta-1)
                     
                        Yt[i+1,j]=Yt[i,j]+ eta*  dw1[i,j]+a*dt*(1-St[i,j])/St[i,j]-0.5* eta**2*dt
                      
                        sigma[i+1,j+mP] = sigma[i,j+mP] -  nu[k] *sigma[i,j+mP] * dw2[i,j] 
                        eta=sigma[i,j+mP]*St[i,j+mP]**(beta-1)
                     
                        Yt[i+1,j+mP]=Yt[i,j+mP]- eta*dw1[i,j]+a*dt*(1-St[i,j+mP])/St[i,j+mP]-0.5* eta**2*dt
                        St[i+1,j]=np.exp(Yt[i+1,j])
                        St[i+1,j+mP]=np.exp(Yt[i+1,j+mP])  
        return St  
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
        call_mc = np.maximum(0,S_T - effective_K)*F_T            
        Call_mc =  np.sum(call_mc)/m *np.exp(-a*N_E)       
        SE_call =0# np.sqrt(sum(call-Call)/(m*(m-1)))
      
        return Call_mc,  SE_call
 
 
#object function for brut force
def object_func(n_calib,market_price,F,T,N_E,K_list,alpha_list,nu_list,beta,param):
        """
        function to calculate deifferece between market and model
        Future prices/IV
        input:
        F: futures
        obj: Future Price/IV
        """
       
       
        #print("param",param[0],param[1])
       
        # b_list = self._b_list
        # c_list = self._c_list
        index_S=int(T[n_calib]/dt)
        nu_list[n_calib] =param[0]
        alpha_list[n_calib]=param[1]       
        
        
        St = sabr_spot(a,m,T,beta,nu_list,alpha_list)       
        S_time=St[index_S]
        diff=0
        market_list=market_price[n_calib]
        
        for i in range(len(K_list)):
               
            v,error = OptionPricing(S_time,K_list[i],N_E,F,a,m)
            diff+=10*abs(v-market_list[i])
 
        return diff
       
     
        
def find_param(n_calib,market_price,F,T,N_E,K_list,nu_list,alpha_list,beta):
      
         start_params=np.array([nu_list[n_calib-1],alpha_list[n_calib-1]])
         #start_params=np.array([-1,1.2])
         difference = lambda param: object_func(n_calib,market_price,F,T,N_E,K_list,alpha_list,nu_list,beta,param)
         bnds = ((0.001,None),(0.001,None))
         #cons = [{'type':'ineq', 'fun': lambda param:param[1]-0.2+param[0]}]
         #cons = [{'type':'ineq', 'fun': lambda param:-0.2-param[0]}]
        
         all_results = opt.minimize(fun=difference, x0=start_params,
                                          #method="Nelder-Mead",options= {"disp":True,"maxiter":20})
                                           method="SLSQP",options= {"disp":True,"maxiter":30},bounds=bnds)
         error=object_func(n_calib,market_price,F,T,N_E,K_list,alpha_list,nu_list,beta,all_results.x)
 
         print('Error'+str(n_calib),error)
         return all_results.x           


    #object function for numeric method
def obj_num(a,m,beta,nu,alpha,T,param):
    LHS=
    EY=0
    VY=0
    for i in range(len(T-1)):
           
        EY+=alpha[i]**2(alpha[i]**2*(np.exp(T[i+1])-np.exp(T[i]))+(alpha[i]-alpha[i]**2)*(T[i+1]-T[i]))
        
        VY+= alpha[i]**8*(1/6*(np.exp(6*T[i+1])-np.exp(T[i]))-1/2*(np.exp(2*T[i+1])-np.exp(2*T[i])))
    EY=EY+param**2(param**2*(np.exp(T[i+1])-np.exp(T[i]))+(param-param**2)*(T[i+1]-T[i]))  
    VY =VY + param**8*(1/6*(np.exp(6*T[i+1])-np.exp(T[i]))-1/2*(np.exp(2*T[i+1])-np.exp(2*T[i])))
    EsqrtY= np.sqrt(EY**2/np.sqrt(VY+EY**2))*(1+VY/EY**2)**(1/8)
    diff =np.mean(St[-1,:])-1- EsqrtY/np.sqrt(2*np.pi)
    return diff

#calibrate nu
def solvefornu(x,*param):  
    T,alpha,nu=param
    lhs=0
    rhs=0
   
    sumj=0
    nu=np.insert(nu,len(T)-1,x)
    T=np.insert(T,0,0)
    
    for i in range(len(T)-1):
        for j in range(1,max(i,1)+2):
            #sum wrt j 
            sumj+= nu[j-1]**2*(T[j]-T[j-1])
        lhs+=alpha[i]**4*np.exp(6*(sumj-T[i]*nu[i]**2))*(np.exp(4*T[i+1]*nu[i]**2)-np.exp(4*T[i]*nu[i]**2)+4*(np.exp(-T[i+1]*nu[i]**2)-np.exp(-T[i]*nu[i]**2)))/(4*nu[i]**4)   
        rhs+=alpha[i]**2*np.exp(sumj+(T[i+1]-2*T[i])*nu[i]**2)/(nu[i]**2)
    rhs= (rhs/(np.exp(nu[-1]**2*T[-1])-1))**2*(np.exp(6*nu[-1]**2*T[-1])/6-np.exp(nu[-1]**2*T[-1])+5/6)
    return lhs-rhs 







if __name__ == "__main__":  
    output_path = "/Users/veraisice/Desktop/Comodity-Market-Research/thesis_1/"
    input_path  = "/Users/veraisice/Desktop/Comodity-Market-Research/Input/"
   

   
    Month_List = ["October","November","December"]
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
    a =0 
    m=5000
    dt=1/365
    np.random.seed(1)
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
    
    beta=0.5
    nu0=0.4182074
    alpha0= 0.73064841
    nu_list = np.ones(n_month)*nu0
    alpha_list = np.ones(n_month)*alpha0
   
    
    n_calib=1 # Nov
    
    param1 = find_param(n_calib,market_price, Future[n_calib], T_M[:n_calib+1], N_E[n_calib], K[n_calib], nu_list,alpha_list,beta)
   
    n_calib=2 # Dec
    param2 = find_param(n_calib,market_price, Future[n_calib], T_M[:n_calib+1], N_E[n_calib], K[n_calib], nu_list,alpha_list,beta)
    
   
    # b_list1=np.array([-0.845309,-0.902326,-1.5,-1.6,-1])
    # c_list2=np.array([1.8962,1.85955,1.57,2.1,2])
   
    # ________________________
    n_calib=1
    K_list=K[n_calib]
    S_TArray=sabr_spot(a,m,T_M[0:2],beta,nu_list,alpha_list) 
    S_T=S_TArray[int(T_M[n_calib]/dt)]
    call = np.zeros((len( K[n_calib])))
    SE_call = np.zeros((len( K[n_calib])))
 
    for i in range(len( K[n_calib])):
        call[i],SE_call[i] = OptionPricing(S_T,K_list[i],N_E[n_calib],Future[n_calib],a,m)
 
    plt.figure(dpi=1000)
    plt.plot(effective_K[n_calib],call,'--b*',label="Model Prices VS Strikes")
    plt.plot(effective_K[n_calib],market_price[n_calib],'--r*',label="Market Prices VS Strikes")
    plt.title("Comparison of TTF Futures Option Price Expired in " + str(Month_List[n_calib])  )
    plt.legend(loc= 'best')
      # ________________________
    n_calib=2
    K_list=K[n_calib]
    S_T=S_TArray[int(T_M[n_calib]/dt)]
    call = np.zeros((len( K[n_calib])))
    SE_call = np.zeros((len( K[n_calib])))
 
    for i in range(len( K[n_calib])):
        call[i],SE_call[i] = OptionPricing(S_T,K_list[i],N_E[n_calib],Future[n_calib],a,m)
 
    plt.figure(dpi=1000)
    plt.plot(effective_K[n_calib],call,'--b*',label="Model Prices VS Strikes")
    plt.plot(effective_K[n_calib],market_price[n_calib],'--r*',label="Market Prices VS Strikes")
    plt.title("Comparison of TTF Futures Option Price Expired in " + str(Month_List[n_calib])  )
    plt.legend(loc= 'best')
      # ________________________
       
   


    
    
   

    
    #magic number
    a =0.1
    m=1000
    dt=1/365
    np.random.seed(1)
    for i in range(len(Month_List)):
        Option_Data = pd.read_excel(input_path+ "TTFdata"+".xlsx",sheet_name = Month_List[i])    #SEPopt
   
    #time to maturity
        T_M[i]= Option_Data["Time to Maturity"].values[0]
    
    #start alpha and nu
    alpha_start= [0.74426606, 0.53440506]
    nu = [0.7931,0.4753256]
    # calibrate for novmber nu
    param=(T_M[0:2],alpha_start,nu)
    nu_nov = fsolve(solvefornu,0.5,args=param)
    print(" november nu is" + str(nu_nov) + "error is" +str(solvefornu(nu_nov,*param)) )       
    
    Diff = lambda param: obj(alpha,T,param)
   
    all_results = opt.minimize(fun=Diff, x0=0.01,
                                          method="Nelder-Mead",options= {"disp":True,"maxiter":60})
                                          
    error=obj(a,m,nu,alpha,T,all_results.x)
 
    print(all_results.x,error)
    

       