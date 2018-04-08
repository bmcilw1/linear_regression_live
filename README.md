# linear_regression_live
This is the code for the "How to Do Linear Regression the Right Way" live session by Siraj Raval on Youtube


## Overview

This is the code for [this](https://youtu.be/uwwWVAgJBcM) video on Youtube by Siraj Raval. I'm using a small dataset of student test scores and the amount of hours they studied. Intuitively, there must be a relationship right? The more you study, the better your test scores should be. We're going to use [linear regression](https://onlinecourses.science.psu.edu/stat501/node/250) to prove this relationship. 

## Challenge

Re-implement the [previous challenge](https://github.com/bmcilw1/linear_regression_demo), but eliminate the dependency on sklearn. Instead, manually implement a gradient descent approach to arive at the line of best fit. You may find it easier to also eliminate pandas (in favor of numpy's genfromtxt). Use matplotlib or other tool to visualize the results. You will want to import numpy directly and use it as much as you can for performance gains (it was being used internally by sklearn). Bonus points to the fastest-executing code THAT STILL GO THROUGH ALL STEPS of gradient descent and converge on the solution (with learning_rate = 0.0001, num_iterations = 100000). Experiment with different learning rates and number of iterations.

Here are some helpful links:

#### Gradient descent visualization
https://raw.githubusercontent.com/mattnedrich/GradientDescentExample/master/gradient_descent_example.gif

#### Sum of squared distances formula (to calculate our error)
https://spin.atomicobject.com/wp-content/uploads/linear_regression_error1.png

#### Partial derivative with respect to b and m (to perform gradient descent)
https://spin.atomicobject.com/wp-content/uploads/linear_regression_gradient1.png

## Dependencies

* numpy

Python 2 and 3 both work for this. Use [pip](https://pip.pypa.io/en/stable/) to install any dependencies.

## Usage

Just run ``python3 demo.py`` to see the results:

   ```
Starting gradient descent at b = 0, m = 0, error = 5565.107834483211
Running...
After 1000 iterations b = 0.08893651993741346, m = 1.4777440851894448, error = 112.61481011613473
   ```

## Credits

Credits for this code go to [mattnedrich](https://github.com/mattnedrich). I've merely created a wrapper to get people started. 
