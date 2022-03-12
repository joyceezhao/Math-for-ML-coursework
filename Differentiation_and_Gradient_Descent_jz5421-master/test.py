import unittest
import numpy as np

from solution import f1, f2, f3
from solution import f1_check_minimum
from solution import f1_grad_fd, f2_grad_fd, f3_grad_fd
from solution import f1_grad_exact, f2_grad_exact, f3_grad_exact
from solution import gradient_descent

class TestCoursework(unittest.TestCase):

    def test_f1(self):
        """ Test whether (given) function f1 is correct by evaluating one point. """
        dummy_input = np.array([[2.5], [3.5]])
        output = f1(dummy_input)
        target_output = 25.5
        self.assertAlmostEqual(output, target_output, delta=1e-3)

    def test_f2(self):
        """ Test whether (given) function f2 is correct by evaluating one point. """
        dummy_input = np.array([[2.5], [3.5]])
        output = f2(dummy_input)
        target_output = 25.202135120387183
        self.assertAlmostEqual(output, target_output, delta=1e-3)

    def test_f3(self):
        """ Test whether (given) function f3 is correct by evaluating one point. """
        dummy_input = np.array([[2.5], [3.5]])
        output = f3(dummy_input)

        target_output = 0.8313103674065578
        self.assertAlmostEqual(output, target_output, delta=1e-3)

    def test_f1_minimum_check(self):
        """ Unit test to check whether the 'f1_minimum_check' function is correct. """

        # Check with minimum
        B = np.array([[4, -2], [-2, 4]])
        a = np.array([[0], [1]])
        b = np.array([[-2], [1]])

        check = f1_check_minimum(B, a, b)

        self.assertTrue(check, f"The function 'f1_minimum_check' does not seem to be correct, yet.")

        # Check without minimum
        B = np.array([[1, 0], [1, 0]])
        a = np.array([[0], [0]])
        b = np.array([[0], [0]])

        check = f1_check_minimum(B, a, b)

        self.assertFalse(check, f"The function 'f1_minimum_check' does not seem to be correct, yet.")

    def test_f1_grad_fd(self):
        """ Unit test to check whether the 'f1_grad_fd' function is correct. """
        dummy_input = np.array([[2.5], [3.5]])
        output = f1_grad_fd(dummy_input)

        target_output = np.array([[3.00003, 11.00003]])
        correct = np.isclose(output, target_output, atol=1e-3).all()

        self.assertTrue(correct, f"The function 'f1_grad_fd' does not seem to return the correct gradient yet.")

    def test_f2_grad_fd(self):
        """ Unit test to check whether the 'f2_grad_fd' function is correct. """
        dummy_input = np.array([[2.5], [3.5]])
        output = f2_grad_fd(dummy_input)

        target_output = np.array([[1.18572957, 5.10321673]])
        correct = np.isclose(output, target_output, atol=1e-3).all()

        self.assertTrue(correct, f"The function 'f2_grad_fd' does not seem to return the correct gradient yet.")

    def test_f3_grad_fd(self):
        """ Unit test to check whether the 'f3_grad_fd' function is correct. """

        dummy_input = np.array([[2.5], [3.5]])
        output = f3_grad_fd(dummy_input)

        target_output = np.array([[0.02703108, 0.03783601]]) # assuming delta=1e-5, but likely to pass due to tolerant atol
        correct = np.isclose(output, target_output, atol=1e-3).all()

        self.assertTrue(correct, f"The function 'f3_grad_fd' does not seem to return the correct gradient yet.")

    def test_f1_grad_exact(self):
        """ Unit test to check whether the 'f1_grad_exact' function is correct. """
        dummy_input = np.array([[2.5], [3.5]])
        output = f1_grad_exact(dummy_input)

        target_output = np.array([[3.0, 11.0]])
        correct = np.isclose(output, target_output, atol=1e-3).all()

        self.assertTrue(correct, f"The function 'f1_grad_exact' does not seem to return the correct gradient yet.")

    def test_f2_grad_exact(self):
        """ Unit test to check whether the 'f2_grad_exact' function is correct. """
        dummy_input = np.array([[2.5], [3.5]])
        output = f2_grad_exact(dummy_input)

        target_output = np.array([[1.1858, 5.1032]])
        correct = np.isclose(output, target_output, atol=1e-3).all()

        self.assertTrue(correct, f"The function 'f2_grad_exact' does not seem to return the correct gradient yet.")

    def test_f3_grad_exact(self):
        """ Unit test to check whether the 'f3_grad_exact' function is correct. """
        dummy_input = np.array([[2.5], [3.5]])
        output = f3_grad_exact(dummy_input)

        target_output = np.array([[0.0270, 0.0378]])
        correct = np.isclose(output, target_output, atol=1e-3).all()

        self.assertTrue(correct, f"The function 'f3_grad_exact' does not seem to return the correct gradient yet.")

    def test_gradient_descent(self):
        """ Unit test to check whether the 'gradient_descent' function is correct. """

        c = np.array([[4], [5]])

        def dummy_fn(x):
            return float(c.T @ x)

        def dummy_fn_grad(x):
            return c.T

        dummy_start_x = 1.0
        dummy_start_y = 2.0

        trajectory, minimum_loc, minimum_value = gradient_descent(dummy_fn, dummy_fn_grad, start_x=dummy_start_x, start_y=dummy_start_y, lr=0.01, n_steps=5)

        target_trajectory = [np.array([[1], [2]]), np.array([[0.96], [1.95]]), np.array([[0.92], [1.9]]),
                             np.array([[0.88], [1.85]]), np.array([[0.84], [1.8]]), np.array([[0.8], [1.75]])]
        target_minimum_loc = np.array([[0.8], [1.75]])
        target_minimum_value = 11.949999999999998

        correct_trajectory = np.array([np.isclose(target.flatten(), output.flatten()) for target, output in zip(target_trajectory, trajectory)]).all()
        correct_minimum_loc = np.isclose(target_minimum_loc, minimum_loc).all()
        correct_minimum_value = target_minimum_value == minimum_value

        self.assertTrue(correct_trajectory, f"The function 'gradient_descent' does not yet seem to return a correct trajectory yet.")
        self.assertTrue(correct_minimum_loc, f"The function 'gradient_descent' does not yet seem to return a correct minimum location yet.")
        self.assertTrue(correct_minimum_value, f"The function 'gradient_descent' does not yet seem to return a correct minimum value yet.")


if __name__ == '__main__':
    unittest.main()

