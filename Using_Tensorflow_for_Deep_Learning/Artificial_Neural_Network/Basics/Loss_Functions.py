import numpy as np

# Sample Data
y_predicted = np.array([1, 1, 0, 0, 1])
y_true = np.array([0.30, 0.7, 1, 0, 0.5])

# Mean Absolute Error
def mae(y_true, y_predicted):
    total_error = np.sum(np.abs(y_predicted - y_true))
    # for yt, yp in zip(y_true, y_predicted):
    #     total_error += abs(yt - yp)     # To calculate absolute error
    print("Mean Absolute Error")
    print("Total Error = ", total_error)
    mae = np.mean(np.abs(y_predicted - y_true))
    print("MAE = ", mae)
    return mae

# Mean Squared Error
def mse(y_true, y_predicted):
    total_error = np.sum(np.abs(np.power((y_predicted - y_true), 2)))
    # for yt, yp in zip(y_true, y_predicted):
    #     total_error += abs(yt - yp)     # To calculate absolute error
    print("Mean Squared Error")
    print("Total Error = ", total_error)
    mse = np.mean(np.abs(np.power((y_predicted - y_true), 2)))
    print("MAE = ", mse)
    return mse

# Binary Cross-Entropy
def bce(y_true, y_predicted):
    epsilon = 1e-15
    y_predicted_new = [max(i, epsilon) for i in y_predicted]    # To avoid log(1) = 0
    y_predicted_new = [min(i, 1-epsilon) for i in y_predicted_new]      # To avoid log(0) = inf
    y_predicted_new = np.array(y_predicted_new)
    bce = -np.mean(y_true*np.log(y_predicted_new)+(1-y_true)*np.log(1-y_predicted_new))
    print("Binary Cross-Entropy = ", bce)
    return bce

mae(y_true, y_predicted)
mse(y_true, y_predicted)
bce(y_true, y_predicted)