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
% theta1= [0;theta(2:end)];
% 
% Theta = theta;
% for i=2:m
%     Theta = [Theta,theta];
% end
% 
% 
% h = sum(X .* Theta',2);
% J = sum(( h - y).^2,1) / m / 2 + sum(theta1 .^ 2,1) * lambda / 2 / m;
% 
% % Theta(1,:) = Theta(1,:) * 0;
% 
% 
% [m1,n1] = size(X);
% tmp = h - y;
% for i=2:n1
%     tmp = [tmp,h - y];
% end
% 
% grad_tmp = tmp .* X;
% grad = sum(grad_tmp,1) /m + (lambda * theta1 /m)';
% 
% grad = grad';

hx = X * theta;
theta_temp = [0;theta(2:end)];
J = (1 / (2 * m)) * (hx  - y)' * (hx - y) + (lambda / (2 * m)) * (theta_temp' * theta_temp);
grad = (1 / m) * X' * (hx - y) + (lambda / m) * theta_temp;
% =========================================================================

grad = grad(:);

end
