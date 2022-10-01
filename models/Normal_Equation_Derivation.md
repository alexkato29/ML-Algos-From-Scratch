## Deriving the Normal Equation
Linear Regressions take the form ![equation](https://latex.codecogs.com/gif.image?%5Cdpi%7B110%7D%5Cmathbf%7B%5Chat%7By_i%7D%7D%20=%20%5CTheta%20_0&plus;%5CTheta%20_1x_1&plus;...&plus;%5CTheta%20_nx_n) 
where ![equation](https://latex.codecogs.com/gif.image?%5Cdpi%7B110%7D%5CTheta%5Cepsilon%5Cmathbb%7BR%5E%7B%5Ctext%7Bn%7D%7D%7D),
![equation](https://latex.codecogs.com/gif.image?%5Cdpi%7B110%7D%5Cvec%7Bx%7D%5Cepsilon%5Cmathbb%7BR%5E%7B%5Ctext%7Bn%7D%7D%7D)
such that n is the number of features.

To maximize the accuracy of the regression, we want to minimize the mean squared error function ![equation](https://latex.codecogs.com/gif.image?%5Cdpi%7B110%7DMSE%20=%20%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi=1%7D%5E%7Bm%7D(%5Ctheta%20%5Ccdot%20x_i%20-%20y_i)%5E2)
.
