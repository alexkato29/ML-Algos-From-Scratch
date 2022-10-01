## Deriving the Normal Equation
Linear Regressions take the form 

![equation](https://latex.codecogs.com/svg.image?%5Cdpi%7B110%7D%5Cmathbf%7B%5Chat%7By_i%7D%7D%20=%20%5CTheta%20_0&plus;%5CTheta%20_1x_1&plus;...&plus;%5CTheta%20_nx_n) 

where ![equation](https://latex.codecogs.com/svg.image?%5Cdpi%7B110%7D%5CTheta%5Cepsilon%5Cmathbb%7BR%5E%7B%5Ctext%7Bn%7D%7D%7D),
![equation](https://latex.codecogs.com/svg.image?%5Cdpi%7B110%7D%5Cvec%7Bx%7D%5Cepsilon%5Cmathbb%7BR%5E%7B%5Ctext%7Bn%7D%7D%7D)
such that n is the number of features. To maximize the accuracy of the regression, we want to minimize the mean squared error function 

![equation](https://latex.codecogs.com/svg.image?%5Cdpi%7B110%7DMSE%20=%20%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi=1%7D%5E%7Bm%7D(%5Ctheta%20%5Ccdot%20x_i%20-%20\bold{y}_i)%5E2)

so that the residual values are minimized. 

![equation](https://latex.codecogs.com/svg.image?%5Cdpi%7B110%7DMSE%20=%20%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi=1%7D%5E%7Bm%7D(%5Ctheta%5E%5Ctop%20x_i%20-%20\bold{y}_i)%5E2)

The dot product is analogous to the transposed matrix product.

![equation](https://latex.codecogs.com/gif.image?%5Cdpi%7B110%7DMSE%20=%20%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi=1%7D%5E%7Bm%7D(%5Ctheta%5E%5Ctop%20x_i%20-%20\bold{y}_i)%5E2) 

The sum of all elements in X by definition of matrix vector multiplication

![equation](https://latex.codecogs.com/svg.image?%5Cdpi%7B110%7DMSE%20=%20%5Cfrac%7B1%7D%7Bm%7D(X%5Ctheta%20-%20\bold{y})%5E%5Ctop%20(X%5Ctheta%20-%20\bold{y}))

The transposes and dot products then distribute 

![equation](https://latex.codecogs.com/svg.image?MSE&space;=&space;\frac{1}{m}[(X\theta)^\top&space;X\theta&space;-&space;(X\theta)^\top&space;\bold{y}&space;-&space;\bold{y}^\top&space;X\theta&space;-&space;\bold{y}^\top&space;\bold{y}])

Again distributing transposes and applying the commutative property of vector dot products yields

![equation](https://latex.codecogs.com/svg.image?MSE&space;=&space;\frac{1}{m}[X^\top&space;\theta^\top&space;X\theta&space;-&space;2X^\top&space;\theta^\top&space;\bold{y}&space;&plus;&space;\bold{y}^\top&space;\bold{y}])

![equation](https://latex.codecogs.com/svg.image?\frac{\partial&space;MSE}{\partial&space;\theta}&space;=&space;\frac{1}{m}[2X^\top&space;X\theta&space;-&space;2X^\top&space;\bold{y}])

To minimize the function, we want to look for the point where the partial is zero

![equation](https://latex.codecogs.com/svg.image?0&space;=&space;\frac{1}{m}[2X^\top&space;X\theta&space;-&space;2X^\top&space;\bold{y}])

![equation](https://latex.codecogs.com/svg.image?X^\top&space;X\theta&space;=&space;X^\top&space;\bold{y})

![equation](https://latex.codecogs.com/svg.image?\theta&space;=&space;(X^\top&space;X)^{-1}X^\top&space;\bold{y})

And thus, we have found our normal equation. Note it was not necessary to compute the second derivative. By nature of 
the MSE function, there is no local or global maximum. Our error can always increase by making the model fit the data 
arbitrarily worse. The only extrema is the global minimum that minimizes error.