## Deriving the Gradients of MSE
Recall from the derivation of the normal equation that

![equation](https://latex.codecogs.com/svg.image?%5Cfrac%7B%5Cpartial%20MSE%7D%7B%5Cpartial%20%5Ctheta%7D%20=%20%5Cfrac%7B2%7D%7Bm%7D%20X%5E%5Ctop(X%5Ctheta%20-%20%5Cbold%7By%7D))

Because $ \theta $ is a vector of weights, this derivative represents 
a new vector where each component *i* is ![equation](https://latex.codecogs.com/svg.image?%5Cpartial%20MSE/%5Cpartial%20%5Ctheta_i).

Thus, by definition,

![equation](https://latex.codecogs.com/svg.image?%5Cbigtriangledown_%5Ctheta%20MSE=%5Cfrac%7B2%7D%7Bm%7DX%5E%5Ctop(X%5Ctheta%20-%20%5Cbold%7By%7D))
