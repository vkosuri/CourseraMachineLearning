function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

testValues = [0.01, 0.03, 0.1, 0.3,  1, 3, 10, 30];

results = [];

for loopC=1:8,
    for loopSigma=1:8,
      
      
      testC = testValues(loopC);
      testSigma = testValues(loopSigma);
      
      model= svmTrain(X, y, testC, @(x1, x2) gaussianKernel(x1, x2, testSigma)); 
      predictions = svmPredict(model, Xval);
      
      testError = mean(double(predictions ~= yval));
      
      fprintf("C: %f\nsigma: %f\nerror: %f\n", testC, testSigma, testError);
      
      results = [results; testC, testSigma, testError];
      
    end
end

[minError, minIndex] = min(results(:,3));



C = results(minIndex,1);
sigma = results(minIndex,2);



fprintf("\n\nLeast error:\nC: %f\nsigma: %f\nerror: %f\n", C, sigma, minError);

% Wil return this one
% C: 0.300000
% sigma: 0.100000
% error: 0.035000

% but this one will work too
% C: 1.000000
% sigma: 0.100000
% error: 0.035000






% =========================================================================

end
