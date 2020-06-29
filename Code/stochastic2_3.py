"""
Stochastics2_Case1_Final.py

Purpose:
    Case 1 stochastics

Version:
    1       First start, 
    2
    3
    4
    5
    Final       final version

Date:
    10-09-2019 - started
    27-09-2019 - submitted

Author:
    Lisa Lansu
   xiaotian zheng
"""

###########################################################
### Imports
import numpy as np
import matplotlib.pyplot as plt

import scipy.stats as stats

###########################################################
#Q(a)
def MC(S1, S2, sigma1, sigma2, rho, N, r,K,n_years):

    ST = np.zeros((N,2))
    Yt = np.zeros((N,2))
    Wt1 = np.random.normal(0.0, 1, (1, N))
    Wt2 = np.random.normal(0.0, 1, (1, N))
    Yt[:,0] = Wt1
    Yt[:,1] = rho * Wt1 + np.sqrt(1-rho**2)*Wt2
    ST[:,0] = S1*np.exp((r-0.5*sigma1**2)*n_years+sigma1*np.sqrt(n_years)*Yt[:,0])
   
    ST[:,1] = S2*np.exp((r-0.5*sigma2**2)*n_years+sigma2*np.sqrt(n_years)*Yt[:,1])
    price = np.zeros((N,3))
    for i in range(N):
        price[i,0] = np.exp(-r * n_years) * max(ST[i,0]-S1,0) # Price of the first call option
        price[i,1] = np.exp(-r * n_years) * max(ST[i,1]-S2,0) # Price of the sercond call option
        price[i,2] =np.exp(-r * n_years) * max(ST[i,0]-ST[i,1]-K,0) # price of the spread option
    Spread = np.sum(price[:,2])/N 
    return Spread
#Q(b)
def Plot(c_list,S1, S2, sigma1, sigma2, rho, N, r,K,n_years):
    n = len(c_list)
    spread = np.zeros((n,1))
    for i in range(n):
        rho = c_list[i]
        spread[i] = MC(S1, S2, sigma1, sigma2, rho, N, r,K,n_years)
    plt.plot(c_list,spread)
    plt.show()
    return spread
#Q（c)
def portfolio_vol(sigma1,sigma2,rho):
    weight1 = 1
    weight2 = -1
    spread_vol = np.sqrt(weight1 ** 2 * sigma1**2 + weight2 ** 2* sigma2 ** 2 - 2*rho*weight1*weight2*sigma1*sigma2)
    return spread_vol
#Q(d)
def bls_spread(spread_vol,K,S1,S2,n_years,r):
    """
    Purpose:
        Calculate spread call price using Black Scholes

    Inputs:
        S0         price at the start date
        K0          strike price
       spread_vol      simga , double
        n_years           time in years
        r           3-month interest rate

    Return value:
        call        Blackscholes calculated call price
        Put         Blackscholes calculated put price
    """
    S = S1 - S2
    d1 = (np.log(S / K)+(r + 0.5 * spread_vol**2) * n_years)/(spread_vol * np.sqrt(n_years))
    d2 = (np.log(S / K)+(r - 0.5 * spread_vol**2 ) * n_years)/(spread_vol * np.sqrt(n_years))

    bls_price = (S*stats.norm.cdf(d1, 0.0, 1.0) - K*np.exp(-r * n_years)*stats.norm.cdf(d2, 0.0, 1.0))
    
    return bls_price
#Q(f)
# the drift for the ABM rf - 0.5*sigma^2
def MC_ABM(S1,S2,n_years,r,K,sigma1,sigma2):
    S = S1-S2
    mT = S - K*np.exp(-r *n_years) 
    sT =  np.sqrt(S1 **2 * (np.exp(sigma1 **2*n_years) -1) - 2*S1*S2*(np.exp(0.3 * sigma1 *sigma2 * n_years) -1) + S2 **2 * (np.exp(sigma2 ** 2 * n_years) -1))
    ABM_price = mT * stats.norm.cdf(mT/sT,0,1) +  sT* stats.norm.pdf(mT/sT,0,1)

    return ABM_price

def discrepancy_plot(S1, S2, sigma1, sigma2, rho, N, r,K,n_years,spread_vol):
    k_list = np.linspace(0.5 * K, 1.5 * K,20)
    n = len(k_list)
    MC_Bls_diff,Bls_ABM_diff,MC_ABM_diff = np.zeros((n,1)) , np.zeros((n,1)),np.zeros((n,1))
    for i in range (n):
        K = k_list[i]
        MC_Bls_diff[i] = abs(MC(S1, S2, sigma1, sigma2, rho, N, r,K,n_years)-bls_spread(spread_vol,K,S1,S2,n_years,r))
        Bls_ABM_diff[i] = abs(bls_spread(spread_vol,K,S1,S2,n_years,r) - MC_ABM(S1,S2,n_years,r,K,sigma1,sigma2))
        MC_ABM_diff[i]  = abs(MC_ABM(S1,S2,n_years,r,K,sigma1,sigma2)-MC(S1, S2, sigma1, sigma2, rho, N, r,K,n_years))
    plt.plot(k_list,MC_Bls_diff,label = 'price difference between BS and  MC spread option')
    plt.plot(k_list,MC_ABM_diff,label = 'price difference between Bachelier and  MC spread option')
    plt.plot(k_list,Bls_ABM_diff,label = 'price difference between Bachelier and  bS spread option')
    plt.xlabel('strike')
    plt.ylabel('price difference')
    plt.legend()
###########################################################
def main():
    # Magic numbers
    c_list = np.linspace(-1,1,100)
   
    S1= 19*10
    S2= 14*10
    sigma1= 7*3*0.01
    
    sigma2 = 5*3*0.01
   
    rho= 0.3  #correlation between the Brownian motions driving the stock prices 
    r= np.log(1+0.02) # continuously compounding
    N = 1000
    K = S1-S2
    n_years = 1
    MC_price_spread = MC(S1, S2, sigma1, sigma2, rho, N, r,K,n_years)
    print(' spread option price is:',MC_price_spread)

    Plot(c_list,S1, S2, sigma1, sigma2, rho, N, r,K,n_years)
    spread_vol = portfolio_vol(sigma1,sigma2,rho)
    print('spread option volatility is :',spread_vol)
    bls_price_spread = bls_spread(spread_vol,K,S1,S2,n_years,r)
    print('black_scholes price for spread option is:',bls_price_spread )
    ABM_price_spread = MC_ABM(S1,S2,n_years,r,K,sigma1,sigma2)
    print('ABM price for spread option is:',ABM_price_spread )
    discrepancy_plot(S1, S2, sigma1, sigma2, rho, N, r,K,n_years,spread_vol)
###########################################################
### start main
if __name__ == "__main__":
    main()