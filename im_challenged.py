import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# y = mx + b
def func(b, m, x):
    return np.dot(m,x) + b

# m is slope, b is y-intercept
def compute_error(b, m, x, y):
    totalError = np.sum(np.power(y - (np.dot(m,x) + b), 2))
    return totalError / float(len(x))

def step_gradient(b, m, x, y, learningRate):
    N = float(len(x))

    # partial derivitive of error with respect to b
    # -(2/N) * (y - ((m * x) + b))
    b_gradient = np.sum(np.dot(-(2/N), y - (np.dot(m, x) + b)))

    # partial derivitive of error with respect to m
    # -(2/N) * x * (y - ((m * x) + b))
    m_gradient = np.sum(np.dot(-(2/N), np.dot(x, y - (np.dot(m, x) + b))))

    b = b - (learningRate * b_gradient)
    m = m - (learningRate * m_gradient)
    return [b, m]

def gradient_descent_runner(x, y, b, m, learning_rate, num_iterations):

    for i in range(num_iterations):
        b, m = step_gradient(b, m, x, y, learning_rate)

    return b, m

def run():
    # read data
    points = np.genfromtxt('challenge_dataset.txt', delimiter=',')
    x = points[:, 0]
    y = points[:, 1]
    b = 0 # initial y-intercept guess
    m = 0 # initial slope guess

    learning_rate = 0.0001
    num_iterations = 1000000

    print "Starting gradient descent at b = {0}, m = {1}, error = {2}" \
        .format(b, m, compute_error(b, m, x, y))
    print "Running..."

    [b, m] = gradient_descent_runner(x, y, b, m, learning_rate, num_iterations)

    print "After {0} iterations b = {1}, m = {2}, error = {3}" \
        .format(num_iterations, b, m, compute_error(b, m, x, y))

    # visualize results
    plt.scatter(x, y, color='k')
    y_comp = func(b, m, x)
    plt.plot(x, y_comp, color='r')
    plt.show()

if __name__ == '__main__':
    run()
