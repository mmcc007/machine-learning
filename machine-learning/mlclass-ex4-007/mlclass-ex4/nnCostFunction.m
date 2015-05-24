function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%Hx = sigmoid(X*theta);
%J = -1/m*sum(sum(y.*log(Hx) + (1 - y).*log(1 - Hx)));

% Part 1: Feedforward
Theta1Filtered = Theta1(:,2:end);
Theta2Filtered = Theta2(:,2:end);
h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
%Hx = (h2 >= 0.5);
Hx = h2;

% convert y into output format
yout = zeros(m,num_labels);
for i = 1:m
    yout(i,y(i)) = 1;
end

Jk = 0;
% costFunction
for i = 1:m
    Hxi = Hx(i, :);
    yi = yout(i, :);
    for k = 1:num_labels

        Hxik = Hxi(k);
        yik = yi(k);
    
        Jk = Jk - yik.*log(Hxik) - (1 - yik).*log(1 - Hxik);
    end
end

%add regularization
reg = lambda/(2*m) * (sum(sum(Theta1(:, 2:size(Theta1,2)).^2)) + sum(sum(Theta2(:, 2:size(Theta2,2)).^2)));

J = Jk/m + reg;

% Part 2: Backpropagation
% Step 1: Generate activations

% Prepend ones column to the X data matrix
X = [ones(m, 1) X];

% generate level 2 activations
Z2 = X * Theta1';
A2 = (sigmoid(Z2) >= 0.5);

% Prepend ones column to the A2 data matrix
A2 = [ones(size(A2, 1), 1) A2];

% generate and return level 3 activations
Z3 = A2 * Theta2';
A3 = (sigmoid(Z3) >= 0.5);

[C, p] = max(A3, [], 2);

% Step 2: Generate delta for output layer
Delta3 = A3 - yout;

% Step 3: Generate delta for hidden layer
Delta2 = Delta3 * Theta2;
Delta2 = Delta2(:, 2:end); % remove Delta2_0 (first column)
Delta2 = Delta2 .* sigmoidGradient(Z2);

% Step 4: Accumulate Gradient
Theta2_grad = Delta3' * A2;
Theta1_grad = Delta2' * X;

% Step 5: Generate Gradient
Theta1_grad = (1/m) * Theta1_grad;
Theta2_grad = (1/m) * Theta2_grad;


% -------------------------------------------------------------
% Part 3: Implement regularization with the cost function and gradients.
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + ((lambda / m) * Theta1Filtered);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + ((lambda / m) * Theta2Filtered);

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
