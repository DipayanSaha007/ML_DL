import numpy as np
import pandas as pd
import math


def gradient_descent(x,y):
    m_curr = b_curr = 0
    iterations = 1000
    n = len(x)
    learning_rate = 0.0005

    for i in range(iterations):
        y_predict = m_curr * x + b_curr
        cost = (1/n) * sum([val**2 for val in (y - y_predict)])
        math.isclose(a,b,*,rel_tol=1e-20,abs_tol=0.0)
        md = -(2/n) * sum(x * (y - y_predict))
        bd = -(2/n) * sum(y - y_predict)

        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        print("m {}, b {}, cost {}, iteration {}".format(m_curr,b_curr,cost,i))

df = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\ML Projects\Using VSCode\gradient_descent\exercise\test_scores.csv")
x = np.array(df.math)
y = np.array(df.cs)

gradient_descent(x,y)