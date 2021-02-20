'''

'''
from exampleRegression.Optimizer import Optimizer
import numpy as np


class NewtonGaussOptimizer(Optimizer):
    def next_point(self):
        # Solve (J_t * J)d_ng = -J*f
        jacobi = self.jacobi(self.x)
        jacobisLeft = np.dot(jacobi.T, jacobi)
        jacobiLeftInverse = np.linalg.inv(jacobisLeft)
        jjj = np.dot(jacobiLeftInverse, jacobi.T)  # (J_t * J)^-1 * J_t
        nextX = self.x - self.learningRate * np.dot(jjj, self.function_array(self.x)).reshape((-1))
        return self.move_next(nextX)