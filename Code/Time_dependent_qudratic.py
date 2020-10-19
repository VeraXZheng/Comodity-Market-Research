#!/usr/bin/env python3
# -*- coding: utf-8 -*-




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
        np.random.seed(1)
        n = T / dt
        S0 = 1
        # longest
       
        if np.size(T)==1:
            N=int(n)
            b0=b
            c0=c
        else:
            N = int(n[-1])
            
        St = np.zeros((N+1,m))
        Yt = np.zeros((N+1,m))
        St[0,:] = S0
      
        sigma = np.zeros((N+1,m))
        
        sqrt_dt = dt ** 0.5
        mP=int(m*0.5)
      
        dw1 = np.random.normal(size=(N,mP))* sqrt_dt
        sigma[0,:] = b[0]+c[0]
       
        if np.size(T)==1:
            for i in range(N - 1):  
                    for j in range (mP):                     
                         sigma[i,j] =( b0 * St[i,j] +c0)*St[i,j]
                         sigma[i,j+mP] = (b0 * St[i,j+mP] +c0)*St[i,j+mP]#check!
                         St[i+1,j] = St[i,j] + (1-St[i,j])* a * dt + sigma[i,j] * dw1[i,j]
                         St[i+1,j+mP] = St[i,j+mP] + (1-St[i,j+mP])* a * dt -sigma[i,j+mP] * dw1[i,j]
   
            return St
       
        else:
            n=np.insert(n,0,0,axis=0)           
            
            for k in range(len(T)):
               
                for i in range(int(n[k]),int(n[k+1])):
                    for j in range (mP): 
                        b_inf=b[k]
                        c_inf=c[k]
                        vol=(b_inf*St[i,j]+c_inf)
 
                        Yt[i+1,j]=Yt[i,j]+vol* dw1[i,j]+a*dt*(1-St[i,j])/St[i,j]-0.5*vol*vol*dt
                        vol=(b_inf*St[i,j+mP]+c_inf)
                        Yt[i+1,j+mP]=Yt[i,j+mP]-vol* dw1[i,j]+a*dt*(1-St[i,j+mP])/St[i,j+mP]-0.5*vol*vol*dt
                       
                        St[i+1,j]=np.exp(Yt[i+1,j])
                        St[i+1,j+mP]=np.exp(Yt[i+1,j+mP])
       
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
        call_mc = np.maximum(0,S_T - effective_K)*F_T            
        Call_mc =  np.sum(call_mc)/m *np.exp(-a*N_E)       
        SE_call =0# np.sqrt(sum(call-Call)/(m*(m-1)))
      
        return Call_mc,  SE_call
 
 
 
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
    
       
     def DD_diff(self,K,F,T,market_price,param):
        diff = np.ones(len(K))
        for i in range(len(K)):
            price_i=BS(F+param[0],K[i]+param[0],param[1],T,0)
            diff[i]=1.0*abs(price_i- market_price[i])
        
            if abs(K[i]-F)<0.5:
                diff[i]=10000*diff[i]
          
        return sum(diff)
    
     def CalibrateDD(self,K,F,T,market_price,param):
         start_params = np.array([1, 0.1])
        
         sum_diff = lambda param: self.DD_diff(K,F,T,market_price,param)
         #bnds = ((0,1),(0,1))
         all_results = opt.minimize(fun=sum_diff, x0=start_params,   method="Nelder-Mead")#bounds=bnds)
        
         if (self.DD_diff(K,F,T,market_price,all_results.x))>0.01:
             all_results = opt.minimize(fun=sum_diff, x0=all_results.x,   method="Nelder-Mead")  
         return all_results.x
         
     def estimateSigma(self,gamma_sigma,K,F,T,b):
         
            bb=b[0]
            effective_K = 1-np.exp(-a*T)*(1-K/F) 
            c=gamma_sigma[1]/F*(F+gamma_sigma[0])-bb
            estimate_sigma=effective_K*(c+bb*effective_K)*F/(effective_K*F+gamma_sigma[0])
            return estimate_sigma       
         
     def BCparams_func(self,gamma_sigma_list,F,K,T,b):       
             
        n= np.shape(K)[0]
        diff = np.ones(n)
        for i in range (n):
            diff[i]= 10.0*abs(gamma_sigma_list[1]- self.estimateSigma(gamma_sigma_list,K[i],F,T,b))
            if abs(K[i]-F)<0.5:
                diff[i]=100.0*diff[i]
            else:
                diff[i]=diff[i]
           
        return  sum(diff)
     
            
            
     def find_BCparam(self,gamma_sigma_list,F,K,T,param):
         
        difference = lambda b: self.BCparams_func(gamma_sigma_list,F,K,T,b)
             # #bnds = ((0,1),(0,1))
             # all_results = opt.minimize(fun=difference, x0=start_params,  method="Nelder-Mead")#bounds=bnds)
             # error=self.BCparams_func(gamma_sigma_list,F,K,T,all_results.x)
             # if (error)>0.001:
             #     all_results = opt.minimize(fun=difference, x0=all_results.x,  method="Nelder-Mead")
        b=opt.minimize(difference,-1.2).x
        c=gamma_sigma_list[1]/F*(F+gamma_sigma_list[0])-b   
        return np.array([b, c])
 
 
class time_dependent:
    def __init__(self,a,dt,m):
    
        self._a =a
        self._dt =dt
        self._m = m
        # self._b_list = b_list
        # self._c_list = c_list
    
    def object_func(self,n_calib,market_price,F,T,N_E,K_list,obj,b_list,c_list,param):
        """
        function to calculate deifferece between market and model
        Future prices/IV
        input:
        F: futures
        obj: Future Price/IV
        """
        a = self._a
        dt = self._dt
        m=self._m
       
        #print("param",param[0],param[1])
       
        # b_list = self._b_list
        # c_list = self._c_list
        index_S=int(T[n_calib]/dt)
           
        
       
        b_list[n_calib] =param[0]
        c_list[n_calib]=param[1]       
        
        
        St = simulate_spot(a,b_list,c_list,T,m,dt)       
        S_time=St[index_S]
        ones=np.ones(len(K_list))
        call=ones
        SE_call =ones
        diff=0
        market_list=market_price[n_calib]
        if obj == "Opt":
            for i in range(len(K_list)):
               
                v,error = OptionPricing(S_time,K_list[i],N_E,F,a,m)
                diff+=10*abs(v-market_list[i])
 
        else:
             ones=np.ones(len(K_list))
             call=ones
             SE_call =ones
             market_price_dd=ones
             for i in range(len(K_list)):
                 call[i],SE_call[i] = OptionPricing(St[-1,:],K_list[i],N_E,F_T,a,m)
                 market_price_dd[i]=BS(F+gamma_sigma[0],K_list[i]+gamma_sigma[0],gamma_sigma[1],N_E,0)
     
             
             params = np.vstack((call,F*ones,K_list, T[-1]*ones))
             params_dd = np.vstack((market_price_dd,F*ones,K_list, T[-1]*ones))
             vols_mdl = list(map(implied_vol, *params))
             vols_dd = list(map(implied_vol, *params_dd))
             n= len(K_list)
            
             diff = np.ones(n)
             for i in range (n):
                 if abs(K_list[i]-F)<0.5:
                     diff[i]= (vols_mdl[i]- vols_dd[i])**2*1000
                 else:
                      diff[i]=(vols_mdl[i]- vols_dd[i])**2
            
             diff =sum(diff)
        # print("bcerror",diff)
        return diff
       
     
        
    def find_bc(self,n_calib,market_price,F,T,N_E,K_list,obj,b_list,c_list):
      
         start_params=np.array([b_list[n_calib-1],c_list[n_calib-1]])
         #start_params=np.array([-1,1.2])
         difference = lambda param: self.object_func(n_calib,market_price,F,T,N_E,K_list,obj,b_list,c_list,param)
         #bnds = ((0,1),(0,1))
         #cons = [{'type':'ineq', 'fun': lambda param:param[1]-0.2+param[0]}]
         cons = [{'type':'ineq', 'fun': lambda param:-0.2-param[0]}]
        
         all_results = opt.minimize(fun=difference, x0=start_params,
                                          #method="Nelder-Mead",options= {"disp":True,"maxiter":20})
                                           method="SLSQP",options= {"disp":True,"maxiter":30},constraints=cons)
         error=self.object_func(n_calib,market_price,F,T,N_E,K_list,obj,b_list,c_list,all_results.x)
 
         print('Error'+str(n_calib),error)
         return all_results.x           

if __name__ == "__main__":
    output_path = "/Users/veraisice/Desktop/Comodity-Market-Research/thesis_1/"
    input_path  = "/Users/veraisice/Desktop/Comodity-Market-Research/Input/"
   
    Month_List = ["August","September","October","November","December"]
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
    
        vols_mkt[i] = list(map(implied_vol, *params_mkt))
       
   


    
    
   
     
   
#get first set of b and c for Augustto initialise the process
    nR=0
    gamma_sigma_list= calibration_quadratic(a, K, T_M, Future).CalibrateDD(np.asarray(K[nR],float), Future[nR],T_M[nR],np.asarray(market_price[nR],float),param=[1,1])
    b0,c0 = calibration_quadratic(a,K,T_M,Future).find_BCparam(gamma_sigma_list,Future[nR],K[nR],T_M[nR],np.array([-1, 2]))
   
    print("Calibration Result for "+str(Month_List[0])+ " is b=" + str(b0)+" and c=" + str(c0))
    print('Error',calibration_quadratic(a,K,T_M,Future).BCparams_func(gamma_sigma_list,Future[0],K[0],T_M[0],np.array([b0,c0])))
         
 
 
#calibrate for the second month
    b_list = np.ones(n_month)*b0
    c_list = np.ones(n_month)*c0
    # c_list[0]= c0
    #gamma_sigma= calibration_quadratic(a, K, T_M, Future).optimise(np.asarray(K[1],float), Future[1],T_M[1],np.asarray(market_price[1],float),np.array([b0,c0]))
    n_calib=1 # September
    bc1 = time_dependent(a, dt, m).find_bc(n_calib,market_price, Future[n_calib], T_M[:n_calib+1], N_E[n_calib], K[n_calib],"Opt", b_list,c_list)
   
    n_calib=2 # October
    bc2 = time_dependent(a, dt, m).find_bc(n_calib,market_price, Future[n_calib], T_M[:n_calib+1], N_E[n_calib], K[n_calib],"Opt", b_list,c_list)
    
    n_calib=3 # November
    bc3 = time_dependent(a, dt, m).find_bc(n_calib,market_price, Future[n_calib], T_M[:n_calib+1], N_E[n_calib], K[n_calib],"Opt", b_list,c_list)
    
    n_calib=4 # December
    bc4 = time_dependent(a, dt, m).find_bc(n_calib,market_price, Future[n_calib], T_M[:n_calib+1], N_E[n_calib], K[n_calib],"Opt", b_list,c_list)
   
    # b_list1=np.array([-0.845309,-0.902326,-1.5,-1.6,-1])
    # c_list2=np.array([1.8962,1.85955,1.57,2.1,2])
   
    # ________________________
    n_calib=1
    K_list=K[n_calib]
    S_TArray=simulate_spot(a,b_list,c_list,T_M,m,dt)
    S_T=S_TArray[int(T_M[n_calib]/dt)]
    call = np.zeros((len( K[n_calib])))
    SE_call = np.zeros((len( K[n_calib])))
 
    for i in range(len( K[n_calib])):
        call[i],SE_call[i] = OptionPricing(S_T,K_list[i],N_E[n_calib],Future[n_calib],a,m)
 
    plt.figure(dpi=1000)
    plt.plot(effective_K[n_calib],call,'--b*',label="Model Prices VS Strikes")
    plt.plot(effective_K[n_calib],market_price[n_calib],'--r',label="Market Prices VS Strikes")
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
    plt.plot(effective_K[n_calib],market_price[n_calib],'--r',label="Market Prices VS Strikes")
    plt.title("Comparison of TTF Futures Option Price Expired in " + str(Month_List[n_calib])  )
    plt.legend(loc= 'best')
      # ________________________
    n_calib=3
    K_list=K[n_calib]
 
    
    S_T=S_TArray[int(T_M[n_calib]/dt)]
    call = np.zeros((len( K[n_calib])))
    SE_call = np.zeros((len( K[n_calib])))
 
    for i in range(len( K[n_calib])):
        call[i],SE_call[i] = OptionPricing(S_T,K_list[i],N_E[n_calib],Future[n_calib],a,m)
    
    plt.figure(dpi=1000)
    plt.plot(effective_K[n_calib],call,'--b*',label="Model Prices VS Strikes")
    plt.plot(effective_K[n_calib],market_price[n_calib],'--r',label="Market Prices VS Strikes")
    plt.title("Comparison of TTF Futures Option Price Expired in " + str(Month_List[n_calib])  )
    plt.legend(loc= 'best')
    n_calib=4
    K_list=K[n_calib]
 
    
    S_T=S_TArray[int(T_M[n_calib]/dt)]
    call = np.zeros((len( K[n_calib])))
    SE_call = np.zeros((len( K[n_calib])))
 
    for i in range(len( K[n_calib])):
        call[i],SE_call[i] = OptionPricing(S_T,K_list[i],N_E[n_calib],Future[n_calib],a,m)
    
    plt.figure(dpi=1000)
    plt.plot(effective_K[n_calib],call,'--b*',label="Model Prices VS Strikes")
    plt.plot(effective_K[n_calib],market_price[n_calib],'--r',label="Market Prices VS Strikes")
    plt.title("Comparison of TTF Futures Option Price Expired in " + str(Month_List[n_calib])  )
    plt.legend(loc= 'best')

 

 