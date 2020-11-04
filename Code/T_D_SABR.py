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
       
        n = T / dt
       
        if np.size(T)==1:
            N=int(n)
           
        else:
            N = int(n[-1])
          
        St = np.zeros((N+1,m))
        Yt = np.zeros((N+1,m))
        St[0,:] = 1
     
        sigma = np.ones((N+1,m))
        sqrt_dt = dt ** 0.5
        mP=int(m*0.5)
     
        dw1 = np.random.normal(size=(N,m))* sqrt_dt
       
        dw2 = np.random.normal(size=(N,mP))* sqrt_dt
        np.random.seed(1)
        if np.size(T)==1:
           
            for i in range(N - 1): 
                    for j in range (mP):
                        sigma[i+1,j] = sigma[i,j] -  nu *sigma[i,j] * dw2[i,j]
                        eta=sigma[i,j]*St[i,j]**(beta-1)*alpha
                    
                        Yt[i+1,j]=Yt[i,j]+ eta*  dw1[i,j]+a*dt*(1-St[i,j])/St[i,j]-0.5* eta**2*dt
                     
                        sigma[i+1,j+mP] = sigma[i,j+mP] -  nu *sigma[i,j+mP] * dw2[i,j]
                        eta=sigma[i,j+mP]*St[i,j+mP]**(beta-1)
                    
                        Yt[i+1,j+mP]=Yt[i,j+mP]- eta*dw1[i,j]+a*dt*(1-St[i,j+mP])/St[i,j+mP]-0.5* eta**2*dt
                        St[i+1,j]=np.exp(Yt[i+1,j])
                        St[i+1,j+mP]=np.exp(Yt[i+1,j+mP])
                        St[i+1,j]
                       
            return St
      
        else: 
            n=np.insert(n,0,0,axis=0)   
            for k in range(len(T)):
               
               
                for i in range(int(n[k]),int(n[k+1])):
                    for j in range (mP):
                       
                      
                        St[i+1,j]=np.maximum(0.00001,St[i,j]+St[i,j]**beta*alpha[k] *sigma[i,j]*dw1[i,j]+dt*a*(1-St[i,j]))
                        St[i+1,j+mP]=np.maximum(0.00001,St[i,j+mP]-St[i,j+mP]**beta*alpha[k] *sigma[i,j+mP]*dw1[i,j]+dt*a*(1-St[i,j]))
 
                        sigma[i+1,j] = sigma[i,j] *np.exp(nu[k] * dw1[i,j+mP] -0.5*nu[k]*nu[k]*dt)
                        sigma[i+1,j+mP] = sigma[i,j+mP] *np.exp(nu[k] * dw1[i,j+mP] -0.5*nu[k]*nu[k]*dt)
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
  
    
        index_S=int(T[n_calib]/dt)
        nu_list[n_calib] =param[0]
        alpha_list[n_calib]=param[1]     
       
       
        St = sabr_spot(a,m,T,beta,nu_list,alpha_list)     
        S_time=St[index_S]
        diff=0
        market_list=market_price[n_calib]
       
        for i in range(len(K_list)):
              
            v,error = OptionPricing(S_time,K_list[i],N_E,F,a,m)
            diff+=10*(v-market_list[i])**2
            if abs(K_list[i]-F)<0.5:
                    diff+=1000*(v-market_list[i])**2
        return diff
def object_funcATM(n_calib,market_price,F,T,N_E,K_list,alpha_list,nu_list,beta,nu,param):
        """
        function to calculate deifferece between market and model
        Future prices/IV
        input:
        F: futures
        obj: Future Price/IV
        """
      
    
        index_S=int(T[n_calib]/dt)
        nu_list[n_calib] =nu
        alpha_list[n_calib]=param  
       
       
        St = sabr_spot(a,m,T,beta,nu_list,alpha_list)     
        S_time=St[index_S]
        diff=0
        market_list=market_price[n_calib]
       
       
        for i in range(len(K_list)):            
           
            if abs(K_list[i]-F)<0.5:
               
                v,error = OptionPricing(S_time,K_list[i],N_E,F,a,m)
                diff+=1000*(v-market_list[i])**2
        return diff
      
       
def find_param(n_calib,market_price,F,T,N_E,K_list,nu_list,alpha_list,beta):
     
         start_params=np.array([nu_list[n_calib-1],alpha_list[n_calib-1]])
         #start_params=np.array([-1,1.2])
         difference = lambda param: object_func(n_calib,market_price,F,T,N_E,K_list,alpha_list,nu_list,beta,param)
        
         bnds = [ (0.001,None),(0.001,None)]
         bnds0 = [ (0.001,None)]
         cons = [{'type':'ineq', 'fun': lambda param:0.999-param[0]}, {'type':'ineq', 'fun': lambda param:-0.01+param[0]},
                  {'type':'ineq', 'fun': lambda param:-0.01+param[1]},
                   {'type':'ineq', 'fun': lambda param:0.999-param[1]}]
         cons0 = [{'type':'ineq', 'fun': lambda param:-0.01+param},{'type':'ineq', 'fun': lambda param:0.999-param}]    
       
         all_results = opt.minimize(fun=difference, x0=start_params,
                                           #method="Nelder-Mead",options= {"disp":True,"maxiter":20})
                                           method="SLSQP",options= {"disp":True,"maxiter":50},constraints=cons,tol=1e-8)
         error=object_func(n_calib,market_price,F,T,N_E,K_list,alpha_list,nu_list,beta,all_results.x)
         alpha0=all_results.x[1]
         if error>1.1:
             nu0=all_results.x[0]
             differenceATM = lambda param0: object_funcATM(n_calib,market_price,F,T,N_E,K_list,alpha_list,nu_list,beta,nu0,param0)
        
             all_resultsATM = opt.minimize(fun=differenceATM, x0=all_results.x[1],method="SLSQP",options= {"disp":True,"maxiter":10},constraints=cons0)
             alpha0=all_resultsATM.x
             
             
         error=object_func(n_calib,market_price,F,T,N_E,K_list,alpha_list,nu_list,beta,np.array([all_results.x[0], alpha0]))
         print('Error'+ str(n_calib),error)
         return np.array([all_results.x[0], alpha0])        
 

 
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
 

   
       

  
def SE_cali(a,m,n_calib,market_price,F,T,N_E,K_list,St):
        """
        function to calculate SE of model
        option Future prices
        input:
        F: futures
        
        """
       
        
        index_S=int(T[n_calib]/dt)
        S_time=St[index_S]
        
        
        diff=0
        market_list=market_price[n_calib]
        for i in range(len(K_list)):
                   
                v,error = OptionPricing(S_time,K_list[i],N_E,F,a,m)
                diff+=(v-market_list[i])**2
        
        
        return np.sqrt(diff/len(K_list))

  
    
#object function for numerical method
def solveforalpha(x,*param):
    alpha_effective,alpha,T=param
    alpha=np.insert(alpha,len(T)-1,x)
    T=np.insert(T,0,0)
  
    EYR=0
    VYR=0
  
    for i in range(len(T)-1):
         
        EYR+=alpha[i]**3*((np.exp(T[i+1])-np.exp(T[i]))*alpha[i]+(1-alpha[i])*(T[i+1]-T[i]))
      
        VYR+= alpha[i]**8*((np.exp(6*T[i+1])-np.exp(6*T[i]))/6-(np.exp(2*T[i+1])-np.exp(2*T[i]))/2)
    RHS= EYR/((VYR+EYR**2)**(1/4))*(1+VYR/(EYR**2))**(1/8)
    EYL=alpha_effective**3*(alpha_effective*(np.exp(T[-1])-1)-(alpha_effective-1)*T[-1])
    VYL= alpha_effective**8*((np.exp(6*T[-1])-1)/6-(np.exp(2*T[-1])-1)/2)
    LHS= EYL/((VYL+EYL**2)**(1/4))*(1+VYL/(EYL**2))**(1/8)
    return LHS-RHS
#calibrate nu
def solvefornu(x,*param):
    T,alpha,nu=param
    lhs=0
    rhs=0
    sump=0
    sumj=0
    sumjR=0
    nu=np.insert(nu,len(T)-1,x)
    T=np.insert(T,0,0)
  
    for k in range(len(T)-1):
        for j in range(1,max(k,1)+1):
            for p in range(1,max(j,1)+1):
            #sum wrt p
                sump+= nu[p-1]**2*(T[p]-T[p-1])-T[k]*nu[k]**2
            sumj+=alpha[j-1]**2*np.exp(6*sump)*(np.exp(T[k+1]*nu[k]**2)-np.exp(T[k]*nu[k]**2)*(np.exp(5*T[j]*nu[k]**2)-np.exp(5*T[j-1]*nu[k]**2)))
            sumjR+= nu[j-1]**2*(T[j]-T[j-1])
        lhs+=alpha[k]**2/nu[k]**4*sumj+alpha[k]**2*(np.exp(6*sump)+(np.exp(6*T[k+1]*nu[k]**2)-np.exp(6*T[k]*nu[k]**2))/6+ np.exp(5*T[k]*nu[k]**2)*np.exp(T[k+1]*nu[k]**2)-np.exp(T[k]*nu[k]**2))
        rhs+=alpha[k]**2*np.exp(sumjR+(T[k+1]-2*T[k])*nu[k]**2)/(nu[k]**2)
    rhs= (rhs/(np.exp(nu[-1]**2*T[-1])-1))**2*(np.exp(6*nu[-1]**2*T[-1])/6-np.exp(nu[-1]**2*T[-1])+5/6)
    return lhs-rhs
 
 
if __name__ == "__main__":
    output_path = "/Users/veraisice/Desktop/Comodity-Market-Research/thesis_1/"
    input_path  = "/Users/veraisice/Desktop/Comodity-Market-Research/Input/"
    
    
   
  
   # Month_List = ["October","November","December","January"]
    Month_List = ["November","December","January","Febunary","March","April","May","June"]
    # Month_List = ["September","October","November","December"]
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
    m=2400
    dt=1/365
    np.random.seed(1)
    for i in range(len(Month_List)):
        Option_Data = pd.read_excel(input_path+ "TTFdata"+".xlsx",sheet_name = Month_List[i])   #"TTFdata" #SEPopt
    #strike
        K[i] = Option_Data['Strike'].values
    #market option price
        market_price[i] = Option_Data["Call"].values
    #time to maturity
        T_M[i]= Option_Data["Time to Maturity"].values[0]
    #expiry to notification date
        N_E[i]=Option_Data["N-E"].values[0]
    #future price
        Future[i] = Option_Data["1-Month Future"].values[0]
  
        effective_K[i] = 1-np.exp(-a*N_E[i])*(1-K[i]/Future[i])
       
    beta=0.5
   
    alpha_list=np.array([0.986,0.392565,0.02,0.0784501,0.23,0.237])
    nu_list=np.array([0.1   ,0.2,0.2,0.2,0.2,0.2])
  
 
    # alpha_list= np.array([ 0.73534803, 0.53236472, 0.4875991 ])
    # nu_list=np.array([0.49887788, 0.52797162, 0.58208504])
    
    n_calib=1 # Sep  
    param1 = find_param(n_calib,market_price, Future[n_calib], T_M[:n_calib+1], N_E[n_calib], K[n_calib], nu_list,alpha_list,beta)
 
    n_calib=2 # Oct
    param2 = find_param(n_calib,market_price, Future[n_calib], T_M[:n_calib+1], N_E[n_calib], K[n_calib], nu_list,alpha_list,beta)
    if(n_month>3):
        n_calib=3 # Nov
        param3 = find_param(n_calib,market_price, Future[n_calib], T_M[:n_calib+1], N_E[n_calib], K[n_calib], nu_list,alpha_list,beta)
    if(n_month>4):
        n_calib=4 # Dec
        param4 = find_param(n_calib,market_price, Future[n_calib], T_M[:n_calib+1], N_E[n_calib], K[n_calib], nu_list,alpha_list,beta)

    if(n_month>5):
        n_calib=5# jan
        param4 = find_param(n_calib,market_price, Future[n_calib], T_M[:n_calib+1], N_E[n_calib], K[n_calib], nu_list,alpha_list,beta)
    
    S_TArray=sabr_spot(a,m,T_M,beta,nu_list,alpha_list)
    r=0
   
    T_M_cor = (T_M/dt).astype(int)
    S_T = S_TArray[T_M_cor]
    k_list =np.ones(n_month)
    call_mkt_TD= np.ones(n_month)
    call = np.ones(n_month)
    SE_call= np.ones(n_month)
   
    for i in range(n_month):
        k_list[i] = np.array([y for (x,y) in enumerate(K[i]) if abs(K[i][x]-Future[i])<0.3])
        call_mkt_TD[i] =np.array([market_price[i][x] for (x,y) in enumerate(K[i]) if abs(K[i][x]-Future[i])<0.3])
        call[i],SE_call[i] = OptionPricing(S_T[i,:],k_list[i],N_E[i],Future[i],a,m)
       
    params = np.vstack((call,Future,k_list, T_M, r*np.ones(n_month)))
    vols_model = list(map(implied_vol, *params))
    
       
    params_mkt_TD =np.vstack((call_mkt_TD,Future,k_list, T_M, r*np.ones(n_month)))
    vols_mkt_TD= list(map(implied_vol, *params_mkt_TD))
    
   
    plt.figure(dpi=1000)
    plt.xticks(rotation=45, ha='right')
    plt.plot(Month_List,vols_model,'gX',label="TD Model IV")
    plt.plot(Month_List,vols_model_Static,'b^',label="Effective Market Model IV")
    plt.plot(Month_List,vols_mkt_TD,'--r*',label="Market IV")
    plt.xlabel("Month")
    plt.ylabel("Implied Volatility")
    plt.title(" Market Quote ATM IV vs Market Effective Model ATM IV"  )
    plt.legend(loc= 'best')
       
  
    
    
        
    
    
    # ________________________
    n_calib=0
    K_list=K[n_calib]
  
    S_T=S_TArray[int(T_M[n_calib]/dt)]
    call = np.zeros((len( K[n_calib])))
    SE_call = np.zeros((len( K[n_calib])))
    for i in range(len( K[n_calib])):
        call[i],SE_call[i] = OptionPricing(S_T,K_list[i],N_E[n_calib],Future[n_calib],a,m)
    plt.figure(dpi=1000)
    plt.plot(effective_K[n_calib],call,'--b*',label="Model Prices VS Strikes")
    plt.plot(effective_K[n_calib],market_price[n_calib],'--r*',label="Market Prices VS Strikes")
    plt.xlabel("Strike")
    plt.ylabel("Option Price")
    plt.title("Comparison of TTF Futures Option Price Expired in " + str(Month_List[n_calib])  )
    plt.legend(loc= 'best')
    plt.savefig(output_path + "Figures/TD_sabr"+str(Month_List[n_calib]))
    # ________________________
    n_calib=1
    K_list=K[n_calib]
    S_T=S_TArray[int(T_M[n_calib]/dt)]
    call = np.zeros((len( K[n_calib])))
    SE_call = np.zeros((len( K[n_calib])))
    for i in range(len( K[n_calib])):
        call[i],SE_call[i] = OptionPricing(S_T,K_list[i],N_E[n_calib],Future[n_calib],a,m)
    plt.figure(dpi=1000)
    plt.plot(effective_K[n_calib],call,'--b*',label="Model Prices VS Strikes")
    plt.plot(effective_K[n_calib],market_price[n_calib],'--r*',label="Market Prices VS Strikes")
    plt.xlabel("Strike")
    plt.ylabel("Option Price")
    plt.title("Comparison of TTF Futures Option Price Expired in " + str(Month_List[n_calib])  )
    plt.legend(loc= 'best')
    plt.savefig(output_path + "Figures/TD_Sabr"+str(Month_List[n_calib]))
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
    plt.xlabel("Strike")
    plt.ylabel("Option Price")
    plt.title("Comparison of TFF Futures Option Price Expired in " + str(Month_List[n_calib])  )
    plt.legend(loc= 'best')
    plt.savefig(output_path + "Figures/TD_Sabr"+str(Month_List[n_calib]))
    
#    # ________________________
    if(n_month>3):
        n_calib=3
        K_list=K[n_calib]
        S_T=S_TArray[int(T_M[n_calib]/dt)]
        call = np.zeros((len( K[n_calib])))
        SE_call = np.zeros((len( K[n_calib])))
   
        for i in range(len( K[n_calib])):
            call[i],SE_call[i] = OptionPricing(S_T,K_list[i],N_E[n_calib],Future[n_calib],a,m)
   
        plt.figure(dpi=1000)
        plt.plot(effective_K[n_calib],call,'--b*',label="Model Prices VS Strikes")
        plt.plot(effective_K[n_calib],market_price[n_calib],'--r*',label="Market Prices VS Strikes")
        plt.title("Comparison of TFF Futures Option Price Expired in " + str(Month_List[n_calib])  )
        plt.xlabel("Strike")
        plt.ylabel("Option Price")  
        plt.legend(loc= 'best')
        plt.savefig(output_path + "Figures/TD_Sabr"+str(Month_List[n_calib]))
## ________________________
    if(n_month>4):
        n_calib=4
        K_list=K[n_calib]
        S_T=S_TArray[int(T_M[n_calib]/dt)]
        call = np.zeros((len( K[n_calib])))
        SE_call = np.zeros((len( K[n_calib])))
   
        for i in range(len( K[n_calib])):
            call[i],SE_call[i] = OptionPricing(S_T,K_list[i],N_E[n_calib],Future[n_calib],a,m)
   
        plt.figure(dpi=1000)
        plt.plot(effective_K[n_calib],call,'--b*',label="Model Prices VS Strikes")
        plt.plot(effective_K[n_calib],market_price[n_calib],'--r*',label="Market Prices VS Strikes")
        plt.title("Comparison of TFF Futures Option Price Expired in " + str(Month_List[n_calib])  )
        plt.xlabel("Strike")
        plt.ylabel("Option Price")
        plt.legend(loc= 'best')
        plt.savefig(output_path + "Figures/TD_Sabr"+str(Month_List[n_calib]))
     
        
        
        n_calib=5
        K_list=K[n_calib]
        S_T=S_TArray[int(T_M[n_calib]/dt)]
        call = np.zeros((len( K[n_calib])))
        SE_call = np.zeros((len( K[n_calib])))
   
        for i in range(len( K[n_calib])):
            call[i],SE_call[i] = OptionPricing(S_T,K_list[i],N_E[n_calib],Future[n_calib],a,m)
   
        plt.figure(dpi=1000)
        plt.plot(effective_K[n_calib],call,'--b*',label="Model Prices VS Strikes")
        plt.plot(effective_K[n_calib],market_price[n_calib],'--r*',label="Market Prices VS Strikes")
        plt.title("Comparison of TFF Futures Option Price Expired in " + str(Month_List[n_calib])  )
        plt.xlabel("Strike")
        plt.ylabel("Option Price")
        plt.legend(loc= 'best')
        plt.savefig(output_path + "Figures/TD_Sabr"+str(Month_List[n_calib]))
        
  ####Calculate SE of calibration 
        for n_calib in range (len(Month_List)):
            error =SE_cali(a,m,n_calib,market_price,Future[n_calib],T_M, N_E[n_calib], K[n_calib],S_TArray)
            print('Error for'+str(Month_List[n_calib]),error)