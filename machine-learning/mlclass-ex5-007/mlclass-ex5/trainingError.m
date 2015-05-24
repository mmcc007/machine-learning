function [ Jt Jval ] = trainingError( X, y )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%  Train linear regression with lambda = 0
lambda = 0;
[theta] = trainLinearReg([ones(m, 1) X], y, lambda);
Jt = linearRegCostFunction(X, y, theta, lambda)
end

