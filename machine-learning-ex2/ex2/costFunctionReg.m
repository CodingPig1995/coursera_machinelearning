function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n=size(theta,1);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

prediction=sigmoid(X*theta);
cost_f=(-1).*y.*log(prediction)-(1.+(-1).*y).*log(1.+(-1)*prediction);
J=1/m*sum(cost_f)+lambda/(2*m)*(sum(theta.^2)-theta(1)^2);

%grad(1)=1/m*sum((prediction-y).*X(:,1));
%grad([2:n])=1/m*sum((prediction-y).*X(:,[2:n]))+lambda/(m*theta([2:n]));
grad(1) = (X(:, 1).' * (prediction - y)) /m;

grad(2:end) = (X(:, 2:end).' * (prediction - y)) /m + (lambda/m) * theta(2:end); 



% =============================================================

end
