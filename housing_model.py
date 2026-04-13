import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# training data
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

print(f"x_train = {x_train}")
print(f"y_train = {y_train}")

# number of examples
m = x_train.shape[0]
print(f"Number of training examples: {m}")

# visualize data
plt.scatter(x_train, y_train, marker='x', c='r')
plt.title("Housing Prices")
plt.ylabel('Price (in 1000s of dollars)')
plt.xlabel('Size (1000 sqft)')
plt.show()

# model parameters
w = 120
b = 100

def compute_model_output(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)

    for i in range(m):
        f_wb[i] = w * x[i] + b

    return f_wb

# predictions
predictions = compute_model_output(x_train, w, b)

# plot prediction vs actual
plt.plot(x_train, predictions, c='b', label='Prediction')
plt.scatter(x_train, y_train, marker='x', c='r', label='Actual')

plt.title("Housing Prices")
plt.xlabel("Size (1000 sqft)")
plt.ylabel("Price")
plt.legend()
plt.show()

# new prediction
x_i = 1.2
cost = w * x_i + b

print(f"Predicted price for 1200 sqft: ${cost:.0f}k")