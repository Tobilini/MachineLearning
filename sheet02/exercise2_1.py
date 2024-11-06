"""
Exercise 2.1
"""


import numpy as np

# ** Indexes ** #
AGE = 0
AREA = 1

# ** Data ** #
age = np.array([1, 42, 13, 25, 63, 15])
area = np.array([50.73, 41.83, 46.54, 58.27, 72.53, 51.47])
price = np.array([523902.67, 325104.45, 434919.86, 575719.18, 629274.54, 390576.98])

# ** Matrix ** #
X = np.vstack([age, area]).T

# ---- 2.1.1 ---- #
w = np.linalg.inv(X.T @ X) @ X.T @ price  # (X^T X)^-1 X^T y i.e analytical linear regression

print("[Ans 2.1.1] Weights:", w)

# ---- 2.1.2 ---- #
# Given age and area for the prediction
age_test = 10
area_test = 50.0

# Predicted price using the linear model
predicted_price = w[AGE] * age_test + w[AREA] * area_test
print("[Ans 2.1.2] predicted price for test values: ", predicted_price)

# ---- 2.1.3 ---- #
# Real value of test
real_price = 427451.10

# won't do a summation because there's only one pair of values
l2_loss = (predicted_price - real_price) ** 2
l1_loss = abs(predicted_price - real_price)

print("[Ans 2.1.3] l2_loss: ", l2_loss, "l1_loss: ", l1_loss)
