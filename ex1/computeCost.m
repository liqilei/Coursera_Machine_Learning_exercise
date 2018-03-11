function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
[m,n] = size(X);
h_x = zeros(m,1);
for ii = 1 : m
    h_x(ii,:) = theta(1)+theta(2)*X(ii,2);
    J = J + (h_x(ii,:) - y(ii))^2;
end
J = J/(2 * m);




% =========================================================================

end
