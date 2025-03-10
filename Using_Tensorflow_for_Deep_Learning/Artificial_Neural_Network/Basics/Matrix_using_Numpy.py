import numpy as np

# # Matrix Division
# revenue = np.array([[180, 200, 220], [24, 36, 40], [12, 18, 20]])
# expenses = np.array([[80, 90, 100], [10, 16, 20], [8, 10, 10]])
# profit = revenue - expenses
# print("profit = ", profit)

# # Matrix Multiplication
# price_per_unit = np.array([1000, 400, 1200])
# units = np.array([[30, 40, 50], [5, 10, 15], [2, 5, 7]])
# price = price_per_unit*units
# print("Price = ", price)

# # Matrix (.)Dot product
# price_per_unit1 = np.array([1000, 400, 1200])
# units1 = np.array([[30, 40, 50], [5, 10, 15], [2, 5, 7]])
# price1 = np.dot(price_per_unit, units)
# print("Price1 = ", price1)


#### Exercise = https://github.com/codebasics/deep-learning-keras-tf-tutorial/blob/master/4_matrix_math/4_matrix_math.md
## Question no. 1
revenue = np.array([[200, 220, 250], [68, 79, 105], [110, 140, 180], [80, 85, 90]])
# 1 USD to 75 INR
revenue = 75*revenue
print("Question 1 ans = \n", revenue)

## Question no. 2
units_sold = np.array([[50, 60, 25], [10, 13, 5], [40, 70, 52]])
price_per_unit = np.array([20, 30, 15])
sales = np.dot(price_per_unit, units_sold)
# print("\n Question 2 ans = \n", sales)
# total_sales = np.sum(sales, axis=1)
print("\n Question 2 ans = \n", sales)