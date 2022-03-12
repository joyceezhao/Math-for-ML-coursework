import numpy as np
import matplotlib.pyplot as plt

import argparse

from solution import f1, f2, f3
from solution import f1_grad_fd, f2_grad_fd, f3_grad_fd
from solution import f1_grad_exact, f2_grad_exact, f3_grad_exact
from solution import gradient_descent


def main(args):
    # Parse arguments
    if args.function == 1:
        fn = f1
        if args.gradient == 'exact':
            fn_grad = f1_grad_exact
        else:
            fn_grad = f1_grad_fd
        title = 'Plotting function f1'
    elif args.function == 2:
        fn = f2
        if args.gradient == 'exact':
            fn_grad = f2_grad_exact
        else:
            fn_grad = f2_grad_fd
        title = 'Plotting function f2'
    elif args.function == 3:
        fn = f3
        if args.gradient == 'exact':
            fn_grad = f3_grad_exact
        else:
            fn_grad = f3_grad_fd
        title = 'Plotting function f3'
    else:
        raise NotImplementedError(f"Trying to plot unknown function. Use '--function [1, 2, 3]'.")

    # Define plotting range for x- and y- axis.
    x1min, x1max = -3, -0.5
    x2min, x2max = -2, 2

    # Evaluate function everywhere within the defined range for the contour plot
    x1 = np.linspace(x1min, x1max, 100)
    x2 = np.linspace(x2min, x2max, 100)

    X1, X2 = np.meshgrid(x1, x2)

    Y = [fn(np.array([[p1], [p2]])) for p1, p2 in zip(X1.flatten(), X2.flatten())]
    Y = np.array(Y).reshape(X1.shape)

    # Plot contour
    plt.title(title)
    plt.xlim(x1min, x1max)
    plt.ylim(x2min, x2max)
    plt.contourf(X1, X2, Y)
    plt.colorbar()

    # Plot gradient descent trajectory
    trajectory, found_minimum, found_minimum_value = gradient_descent(fn, fn_grad)

    p1, p2 = zip(*trajectory)
    plt.plot(p1, p2, '.-', color='red')

    # Show plot to user
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plotting function for coursework 1 in Mathematics for Machine Learning course.")
    parser.add_argument('--function', type=int, help="Function to plot (e.g. 'fn1', 'fn2', 'fn3') ", default=1)
    parser.add_argument('--gradient', type=str, help="Use either 'fd' for finite-differences, or 'exact' for exact derivation.", default='exact')

    args = parser.parse_args()
    main(args)

