## ML Algorithms from Scratch
This project features my custom-built implementations of common data preprocessing and machine learning algorithms.
I did so to gain a deep understanding of the math and code behind them. ***This project is a work in progress***!
I am working to add new models, training options, and features.
### Preprocessing
##### Train/Test Split
Splits a dataframe into training and testing data.
##### Imputer
Replaces missing data entries with the median of a column. Also can 
delete missing instances or features altogether.
##### Feature Scaling
Scales a column of features using min-max scale or standardization.
Standardization is not bound while min-max is always between 0 and 1.
### Models
##### Linear Regression
Current model implementation supports multiple types of training: 
singular value decomposition (default), normal equation, gradient descents
(batch, mini-batch, and stochastic).
##### Ridge Regression
Implementation of ridge regression to achieve regularization. Supports
training using stochastic gradient descent. Make sure to scale data before
training.
##### Lasso Regression
Implementation of Lasso Regression. Trains using stochastic gd. Good for
removing weights of relatively unimportant features.
### Model Analysis
##### MSE
Returns the MSE of the model's residuals.