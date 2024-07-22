import torch
import time
import random

def polynomial_fun(w, x):

    w_size = w.shape[0]
    x_powers = torch.pow(x.unsqueeze(-1), torch.arange(w_size).float())
    y = torch.matmul(x_powers, w)

    return y

# fit a polynomial model where the degree M is learnable through gates
def fit_polynomial_sgd_g(x, t, M_max, lr, mbs):
    
    w_sgd = torch.randn(M_max + 1, requires_grad=True) # initial weights
    w_g = torch.randn(M_max + 1, requires_grad=True)  # corresponding gates for each weight

    optimizer = torch.optim.Adam([w_sgd, w_g], lr=lr) # optimizes both weights and gates

    shuffled_i = torch.randperm(x.numel()) # shuffle indices for stochastic gradient descent

    epochs_no = 15000

    for epoch in range(epochs_no):
        for i in range(0, x.numel(), mbs):
            indices = shuffled_i[i:i+mbs]
            x_batch = x[indices] # select a minibatch of inputs
            t_batch = t[indices] # select corresponding minibatch of targets

            optimizer.zero_grad() # reset gradients

            adjusted_w = w_sgd * torch.sigmoid(w_g) # calculate adjusted weights
            y_pred = polynomial_fun(adjusted_w, x_batch) # predict with adjusted weights

            loss = torch.mean((y_pred - t_batch) ** 2) # calculate loss

            loss.backward() # backpropagate
            optimizer.step() # update weights and gates

    # calculate effective weights and determine effective polynomial degree
    effective_weights = w_sgd.detach() * torch.sigmoid(w_g).detach() > 0.2
    effective_degree = effective_weights.sum().item()-1

    optimal_w = w_sgd.detach() * torch.sigmoid(w_g).detach()
    
    # adjust effective degree
    if optimal_w[0] <= 0.2:
        effective_degree += 1

    return effective_degree, optimal_w[:effective_degree+1]

def task1a():
    
    print( 'The algorithm dynamically adjusts a models complexity by introducing gates alongside weights in a polynomial function. These gates, modified through training, control the impact of each polynomial term. The model learns not only the best weights but also the optimal polynomial degree required for accurate predictions, thus automatically preventing overfitting by discarding unnecessary terms. The final model complexity is determined by applying a threshold to the gates outputs after training, ensuring only significant terms are retained.')
    
    # Generate a training set and test sest: m = 2, w = [1, 2, 3] using polynomial_fun
    
    w = torch.tensor([1,2,3], dtype=torch.float32) # underlying polynomial degree: m = 2
    xT_size = 20
    xTe_size = 10
    x_range = [-20, 20]
    noise_sdv = 0.5

    # Training set
    xT = torch.linspace(x_range[0], x_range[1], steps = xT_size)
    yT_true = polynomial_fun(w, xT) # true polynomial curve values
    yT = yT_true + torch.randn(yT_true.shape) * noise_sdv # observed training data values

    # Test set
    xTe = torch.linspace(x_range[0], x_range[1], steps = xTe_size)
    yTe_true = polynomial_fun(w, xTe) # true polynomial curve values
    yTe = yTe_true + torch.randn(yTe_true.shape) * noise_sdv # observed test data values
    
    # Run fit_polynomial_sgd_g
    
    optimal_m, optimal_w = fit_polynomial_sgd_g(xT, yT, M_max=7, lr=1e-2, mbs=5)
    
    print(f'The optimised M value: {optimal_m}')
    
    # Report, using printed messages, the optimised M value and the mean (and standard deviation) in
    # difference between the model-predicted values and the underlying “true” polynomial curve.
    
    yT_pred = polynomial_fun(optimal_w, xT)
    
    diff_m = yT_pred - yT_true
    
    diff_m_mean = diff_m.mean().item()
    diff_m_std = diff_m.std().item()
    
    print(f'The mean in difference between the “SGD-predicted” values and the underlying “true” polynomial curve for M = {optimal_m}: {diff_m_mean:.4f}')
    print(f'The standard deviation in difference between the “SGD-predicted” values and the underlying “true” polynomial curve for M = {optimal_m}: {diff_m_std:.4f}')
    

task1a()