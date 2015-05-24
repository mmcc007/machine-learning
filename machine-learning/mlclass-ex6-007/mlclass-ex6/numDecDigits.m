function [ n ] = numDecDigits(x)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
x = abs(x); %in case of negative numbers
n=0;
while (floor(x*10^n)-x*10^n) == eps(x*10^n)*2
    n=n+1;
end
end

