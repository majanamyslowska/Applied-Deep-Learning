import torch
import time
import random

# define a polynomial function using the given weights and inputs
def polynomial_fun(w, x):
    
    w_size = w.shape[0]
    x_powers = torch.pow(x.unsqueeze(-1), torch.arange(w_size).float())
    y = torch.matmul(x_powers, w)

    return y

# fit a polynomial using least squares, given inputs, targets, and polynomial degree M
def fit_polynomial_ls(x, t, M):

    nx, nt = x.shape[0], t.shape[0]
    assert nx == nt

    # generate powers of x for the polynomial terms up to degree M
    x_powers = torch.pow(x.unsqueeze(-1), torch.arange(M + 1).float())
    
    # solve the least squares problem
    result = torch.linalg.lstsq(x_powers, t.unsqueeze(-1)) 
    optimal_w = result.solution.squeeze(-1)
    
    return optimal_w

# fit a polynomial using stochastic gradient descent, given inputs, targets, polynomial degree M, learning rate, and minibatch size
def fit_polynomial_sgd(x, t, M, lr, mbs):

    w_sgd = torch.randn(M + 1, requires_grad = True) # initialize weights randomly

    optimizer = torch.optim.Adam([w_sgd], lr = lr) # use the Adam optimizer

    shuffled_i = list(range(x.numel())) # shuffle indices for SGD
    random.shuffle(shuffled_i)

    epochs_no = 1000
    mbs_times = x.numel()//mbs
    losses = []

    for epoch in range(epochs_no):

        for b in range(mbs_times):  
            
            # select minibatch
            x_batch = torch.tensor([x[i] for i in shuffled_i[b * mbs : (b + 1) * mbs]], dtype=torch.float32) 
            t_batch = torch.tensor([t[i] for i in shuffled_i[b * mbs : (b + 1) * mbs]], dtype=torch.float32) 

            optimizer.zero_grad()

            loss = torch.mean(torch.square((polynomial_fun(w_sgd, x_batch) - t_batch))) # mse loss

            loss.backward()
            optimizer.step()
        if (epoch + 1) % 100 == 0:
            losses.append(loss.item())
            print('epoch {} loss {}'.format(epoch+1, loss.item()))
    
    return w_sgd.detach() # return the trained weights


# compare the time taken by least squares and SGD methods
def compare_time(x_train, t_train, M = 2):
    # ls_time = 0
    # sgd_time = 0

    start_ls = time.time()
    fit_polynomial_ls(x_train, t_train, M)
    end_ls = time.time()
    ls_time = end_ls - start_ls

    start_sgd = time.time()
    fit_polynomial_sgd(x_train, t_train, M, lr=1e-2, mbs=5)
    end_sgd = time.time()
    sgd_time = end_sgd - start_sgd

    return ls_time, sgd_time

# calculate the root mean square error between predictions and true values
def rmse(y_pred, y_true):
  return torch.sqrt(torch.mean(torch.square((y_pred - y_true))))

def task1():
    
    # Generate a training set and test set: m = 2, w = [1, 2, 3] using polynomial_fun
    
    w = torch.tensor([1, 2, 3], dtype=torch.float32) # underlying polynomial degree: m = 2
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
    
    # fit and predict using least squares for different polynomial degrees (M)
    w_ls_m2 = fit_polynomial_ls(xT, yT, M=2)
    w_ls_m3 = fit_polynomial_ls(xT, yT, M=3)
    w_ls_m4 = fit_polynomial_ls(xT, yT, M=4)    
    
    # predictions for training and test sets using the least squares method
    yT_pred_ls_m2 = polynomial_fun(w_ls_m2, xT)
    yTe_pred_ls_m2 = polynomial_fun(w_ls_m2, xTe)
    
    yT_pred_ls_m3 = polynomial_fun(w_ls_m3, xT)
    yTe_pred_ls_m3 = polynomial_fun(w_ls_m3, xTe)
    
    yT_pred_ls_m4 = polynomial_fun(w_ls_m4, xT)
    yTe_pred_ls_m4 = polynomial_fun(w_ls_m4, xTe)
    
    # Report, using printed messages, the mean (and standard deviation) in difference 
    # a) between the observed training data and the underlying “true” polynomial curve (this is independent of M)
    diff_a = yT - yT_true
    diff_a_mean = diff_a.mean().item()
    diff_a_std = diff_a.std().item()
    
    print(f'The mean in difference between the observed training data and the underlying “true” polynomial curve: {diff_a_mean:.4f}')
    print(f'The standard deviation in difference between the observed training data and the underlying “true” polynomial curve: {diff_a_std:.4f}')
    
    # b) between the “LS-predicted” values and the underlying “true” polynomial curve for M = 2,3,4.
    diff_ls_b_m2 = yT_pred_ls_m2 - yT_true
    diff_ls_b_m3 = yT_pred_ls_m3 - yT_true
    diff_ls_b_m4 = yT_pred_ls_m4 - yT_true
    
    diff_ls_b_m2_mean = diff_ls_b_m2.mean().item()
    diff_ls_b_m2_std = diff_ls_b_m2.std().item()
    
    print(f'The mean in difference between the “LS-predicted” values and the underlying “true” polynomial curve for M = 2: {diff_ls_b_m2_mean:.4f}')
    print(f'The standard deviation in difference between the “LS-predicted” values and the underlying “true” polynomial curve for M = 2: {diff_ls_b_m2_std:.4f}')   
    
    diff_ls_b_m3_mean = diff_ls_b_m3.mean().item()
    diff_ls_b_m3_std = diff_ls_b_m3.std().item()
    
    print(f'The mean in difference between the “LS-predicted” values and the underlying “true” polynomial curve for M = 3: {diff_ls_b_m3_mean:.4f}')
    print(f'The standard deviation in difference between the “LS-predicted” values and the underlying “true” polynomial curve for M = 3: {diff_ls_b_m3_std:.4f}')
    
    diff_ls_b_m4_mean = diff_ls_b_m4.mean().item()
    diff_ls_b_m4_std = diff_ls_b_m4.std().item()
    
    print(f'The mean in difference between the “LS-predicted” values and the underlying “true” polynomial curve for M = 4: {diff_ls_b_m4_mean:.4f}')
    print(f'The standard deviation in difference between the “LS-predicted” values and the underlying “true” polynomial curve for M = 4: {diff_ls_b_m4_std:.4f}')
    
    
    # fit and predict using stochastic gradient descent for different polynomial degrees (M)    
    w_sgd_m2 = fit_polynomial_sgd(xT, yT, M=2, lr=1e-2, mbs=5)
    w_sgd_m3 = fit_polynomial_sgd(xT, yT, M=3, lr=1e-2, mbs=5)
    w_sgd_m4 = fit_polynomial_sgd(xT, yT, M=4, lr=1e-2, mbs=5)    
    
    # predictions for training and test sets using the SGD method
    yT_pred_sgd_m2 = polynomial_fun(w_sgd_m2, xT)
    yTe_pred_sgd_m2 = polynomial_fun(w_sgd_m2, xTe)
    
    yT_pred_sgd_m3 = polynomial_fun(w_sgd_m3, xT)
    yTe_pred_sgd_m3 = polynomial_fun(w_sgd_m3, xTe)
    
    yT_pred_sgd_m4 = polynomial_fun(w_sgd_m4, xT)
    yTe_pred_sgd_m4 = polynomial_fun(w_sgd_m4, xTe)
    
    
    # Report, using printed messages, the mean (and standard deviation) in difference between the
    # “SGD-predicted” values and the underlying “true” polynomial curve.
    
    diff_sgd_m2 = yT_pred_sgd_m2 - yT_true
    diff_sgd_m3 = yT_pred_sgd_m3 - yT_true
    diff_sgd_m4 = yT_pred_sgd_m4 - yT_true
    
    diff_sgd_m2_mean = diff_sgd_m2.mean().item()
    diff_sgd_m2_std = diff_sgd_m2.std().item()
    
    print(f'The mean in difference between the “SGD-predicted” values and the underlying “true” polynomial curve for M = 2: {diff_sgd_m2_mean:.4f}')
    print(f'The standard deviation in difference between the “SGD-predicted” values and the underlying “true” polynomial curve for M = 2: {diff_sgd_m2_std:.4f}')
    
    
    diff_sgd_m3_mean = diff_sgd_m3.mean().item()
    diff_sgd_m3_std = diff_sgd_m3.std().item()
    
    print(f'The mean in difference between the “SGD-predicted” values and the underlying “true” polynomial curve for M = 3: {diff_sgd_m3_mean:.4f}')
    print(f'The standard deviation in difference between the “SGD-predicted” values and the underlying “true” polynomial curve for M = 3: {diff_sgd_m3_std:.4f}')
    
    
    diff_sgd_m4_mean = diff_sgd_m4.mean().item()
    diff_sgd_m4_std = diff_sgd_m4.std().item()
    
    print(f'The mean in difference between the “SGD-predicted” values and the underlying “true” polynomial curve for M = 4: {diff_sgd_m4_mean:.4f}')
    print(f'The standard deviation in difference between the “SGD-predicted” values and the underlying “true” polynomial curve for M = 4: {diff_sgd_m4_std:.4f}')

    
    # Compare the accuracy of your implementation using the two methods with ground-truth on
    # test set and report the root-mean-square-errors (RMSEs) in both w and y using printed messages.
    
    # RMSE in y
    
    rmse_ls_m2_y = rmse(yTe_pred_ls_m2, yTe_true).item()
    rmse_ls_m3_y = rmse(yTe_pred_ls_m3, yTe_true).item()
    rmse_ls_m4_y = rmse(yTe_pred_ls_m4, yTe_true).item()
    
    print(f'RMSE in y for M = {2, 3, 4} using LS: {rmse_ls_m2_y:.4f}, {rmse_ls_m3_y:.4f}, {rmse_ls_m4_y:.4f}')
    
    rmse_sgd_m2_y = rmse(yTe_pred_sgd_m2, yTe_true).item()
    rmse_sgd_m3_y = rmse(yTe_pred_sgd_m3, yTe_true).item()
    rmse_sgd_m4_y = rmse(yTe_pred_sgd_m4, yTe_true).item()
    
    print(f'RMSE in y for M = {2, 3, 4} using SGD: {rmse_sgd_m2_y:.4f}, {rmse_sgd_m3_y:.4f}, {rmse_sgd_m4_y:.4f}')
    
    # RMSE in w
    
    rmse_ls_m2_w = rmse(w_ls_m2, w).item()

    print(f'RMSE in w for M = 2 using LS: {rmse_ls_m2_w:.4f}')
    
    rmse_sgd_m2_w = rmse(w_sgd_m2, w).item()
    
    print(f'RMSE in w for M = 2 using SGD: {rmse_sgd_m2_w:.4f}')
    
    # Compare the speed of the two methods and report time spent in fitting/training (in seconds) using printed messages.
    
    ls_time, sgd_time = compare_time(xT, yT, M = 2)
    
    print(f'Time spent in fitting/training using LS: {ls_time:.4f}')
    print(f'Time spent in fitting/training using SGD: {sgd_time:.4f}')
    
task1()