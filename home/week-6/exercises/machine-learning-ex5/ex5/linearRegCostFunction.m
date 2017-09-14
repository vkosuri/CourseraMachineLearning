function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


h = X*theta;
hError = h - y;
sumSquaredError = sum(hError .^ 2);
regTermLeft = (1/(2 * m)) * sumSquaredError;

thetaWithoutBias = theta(2:end);
sumSquaredTheta = sum(thetaWithoutBias .^ 2);
regTermRight = (lambda / (2 * m)) * sumSquaredTheta;

J = regTermLeft + regTermRight;

grad = (1/m) * X' * hError;
grad(2:end) += (lambda / m) * thetaWithoutBias;




% =========================================================================

grad = grad(:);

end
