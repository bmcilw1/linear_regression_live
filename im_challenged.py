import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# y = mx + b
# m is slope, b is y-intercept
def y_fun(b, m, x):
    return m * x + b

def error(y, b, m, x):
    return (y - y_fun(b, m, x)) ** 2;

# partial derivitive of error with respect to b
def b_grad(y, b, m, x, N):
    return -(2/N) * (y - ((m * x) + b))

# partial derivitive of error with respect to m
def m_grad(y, b, m, x, N):
    return -(2/N) * x * (y - ((m * x) + b))

def compute_error_for_line_given_points_fast(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += error(y, b, m, x)
    return totalError / float(len(points))

def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += error(y, b, m, x)
    return totalError / float(len(points))

def step_gradient(b, m, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += b_grad(y, b, m, x, N)
        m_gradient += m_grad(y, b, m, x, N)
    b = b - (learningRate * b_gradient)
    m = m - (learningRate * m_gradient)
    return [b, m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, np.array(points), learning_rate)
    return [b, m]

def run():
    # read data
    #dataframe = pd.read_csv('challenge_dataset.txt')
    points = np.genfromtxt('challenge_dataset.txt', delimiter=',')
    
    learning_rate = 0.0001
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    num_iterations = 1000

    print "Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points))
    print "Running..."
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print "After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points))

    # visualize results
    plt.scatter(points[:, 0], points[:, 1])
    y_vect = np.vectorize(y_fun)
    plt.plot(points[:, 0], y_vect(b, m, points[:, 0]))
    plt.show()

if __name__ == '__main__':
    run()
