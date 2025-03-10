from sklearn.datasets import load_digits
from my_package import Best_Model_and_Parameters

digits = load_digits()
# print(dir(digits))
Best_Model_and_Parameters.get_best(digits.data, digits.target)
# print(len(digits.data))