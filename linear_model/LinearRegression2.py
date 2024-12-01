
import numpy as np

def grad_calculate(alpha=None, weights=None, b=None, data_X=None, data_y=None ):
    size_n = data_X.shape[0]
    
    ## grad w.r.t w
    ## y_i_hat = w.T * data_X.T+b (multiplying all X instead of x_i, hence transpose of X)
    y_hat = np.matmul(weights.T, data_X.T).T + b
    
    ## (y_i - y_i_hat) * data_X.T (as summation of (y_i - y_i_hat)*x_i can be written as (y_i - y_i_hat)*data_X)
    grad_w = (-2/size_n) * (np.matmul((data_y.reshape(-1,1)-y_hat).T, data_X)).T
    
    
    
    ## grad w.r.t b --> -2/n * summation(y_i - wTx_i - b)
    grad_b = (-2/size_n) * np.sum(data_y.reshape(-1,1) - y_hat)
    
    #print(grad_b)
    
    return grad_w, grad_b




def cal_error_loss(y_actual=None, y_pred=None):
    dif_sqr = (y_actual - y_pred)**2
    error = 1/y_actual.shape[0] * (np.sum(dif_sqr))
    #print(error)
    return error    

def cal_LR(X_data=None, w=None, b=None):
    y_pred = np.matmul(w.T, X_data.T).T + b    
    return y_pred



def sgd_LinearRegression(X_data=None, y=None, alpha=None, batch_size=None, max_iter=None, tol=None):
    
    ## initialize weights -- randomly
    w = np.random.rand(X_data.shape[1],1)
    b = np.random.rand()    
    
    ## loops: iterations for calculating weights
    for iter in range(0, max_iter):
        
        ## select random batch_size indexes: https://stackoverflow.com/questions/14262654/numpy-get-random-set-of-rows-from-2d-array
        random_indx = np.random.randint(X_data.shape[0], size=batch_size)
        batch_data_X = X_data[random_indx]
        batch_data_y = y[random_indx]

        
        ## return gradient calculated w.r.t w & b
        temp_w, temp_b = grad_calculate(alpha=alpha, weights=w, b=b, data_X=batch_data_X, data_y=batch_data_y)
        
        old_w = w
        old_b = b
        
        w = w - alpha*temp_w
        b = b - alpha*temp_b

        
        ## For terminating the loop: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html
        '''
            tol : float or None, optional (default=1e-3)
            The stopping criterion.
        '''
        temp_Y_pred = cal_LR(X_data=X_data, w=w, b=b)
        cur_error = cal_error_loss(y_actual=y, y_pred=temp_Y_pred)
        
        ## will use --> if error is less than a tolerance value or if old weights and new weights are very close, i.e., difference between them be less than 0.0001
        if tol is not None:
            if (cur_error <= tol) or ((old_w - w)<0.00001).all():
                print("Iteration No.: ", iter)
                break
        elif tol is None:
            if ((old_w - w)<0.00001).all():
                print("Iteration No.: ", iter)
                break

        ## use 0.001 as epsilon to avolid alpha become 0 and get invalid nan or inf (it became 0 after some iterations)
        alpha = (alpha+0.001)/2
    
    return w, b
    