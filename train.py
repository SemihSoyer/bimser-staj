import numpy as np
from sklearn.linear_model import LinearRegression
import pickle


X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y = 1 * x_0 + 2 * x_1 + 3
y = np.dot(X, np.array([1, 2])) + 3

# model
reg = LinearRegression().fit(X, y)
with open("model.pkl", "wb") as f:
    pickle.dump(reg, f)

# prediction
result = reg.predict(np.array([[3, 5]]))
print(result)
