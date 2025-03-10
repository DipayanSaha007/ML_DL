import numpy as np

# all of the m_curr,b_curr,iterations,learning rate values are trial & error values.....these values are taken imaginarily

def gradient_descent(x,y):
    m_curr = b_curr = 0
    iterations = 10000
    n= len(x)
    learning_rate = 0.08   # the rate will be such that we will get decreased cost after each iterations

    for i in range(iterations):
        y_predict = m_curr * x + b_curr
        cost = (1/n) * sum([val**2 for val in (y-y_predict)])  # cost(mse) = (1/n) * (yi - y_predict)^2
        md = -(2/n) * sum(x * (y - y_predict))  # getting partial derivative of 'm' : d/dm = -(2/n) * xi(yi - (m*xi+b))
        bd = -(2/n) * sum(y - y_predict)    # getting partial derivative of 'b' : d/db = -(2/n) * (yi - (m*xi+b))

        m_curr = m_curr - learning_rate * md    # m = m - learning rate * d/dm
        b_curr = b_curr - learning_rate * bd    # b = b - learning rate * d/db
        print("m {}, b {}, cost {}, iteration {}".format(m_curr,b_curr,cost, i))



x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])

gradient_descent(x,y)