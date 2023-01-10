# About the project

The code fetches the latest option prices data for all the strikes and the maturity dates, and estimates the implied volatility and the greeks for each of the option. 

Implied volatility (IV) is estimated using scipy.optimize. With the estimated, the corresponding Greek values are estimated. The Greeks that are measured in this code are delta, gamma, theta, vega, rho, vanna, charm, and volga. 

To fetch the underlying data with one-minute delay, http://aeron7.github.io is used.

The output to the code is a CSV file which will be downloaded to the folder path that is chosen.
