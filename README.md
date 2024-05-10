# Distributions-of-returns
tool to look at log returns of stocks and compare them to the VIX

This started as a way to inspect the distributions of returns for stocks and compare them to a ratio of their volume on that day to the previous day.  There was no correlation there, to we shifted to looking at realized volatility.  There wasn't much correlation there either, now it currently looks at the relationship between the VIX and SP500.  

Enter the ticker symbol in the space provided.  select the period over which you want to inspect, the the frequency of time intervals to look at log returns.  
## The app then returns:
### Time series plot of the asset,
### An ACF of VIX over that time period, 
### a QQ plot of G(t).  
  #### G(t) is a function equal to VIX at time t, V(t) divided by VIX at time t-1, V(T-1), raised to the beta times e^alpha.  alpha and beta are a result of an AR(1) time   series regression.
### G(t) versus time
### ACF of G(t)
### Histogram of G(t)
### qqplot of Z(t) 
  #### regressing for the coefficients that build the relationship between log returns and g(t) yeilds a standard normal error.  
### Log Returns versus time
### ACF of log returns
### Histogram of log returns
### Log returns v G(t)
### The MLE's of the distributions
### Plots of simulated data

