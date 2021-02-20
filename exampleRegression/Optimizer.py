class Optimizer:
    def __init__(self, function, initialPoint, gradient=None, jacobi=None, hesse=None,
                 interval=None, epsilon=1e-7, function_array=None, metaclass=ABCMeta):
        self.function_array = function_array
        self.epsilon = epsilon
        self.interval = interval
        self.function = function
        self.gradient = gradient
        self.hesse = hesse
        self.jacobi = jacobi
        self.name = self.__class__.__name__.replace('Optimizer', '')
        self.x = initialPoint
        self.y = self.function(initialPoint)


"Возвращает следующую координату по ходу оптимизационного процесса"


@abstractmethod
def next_point(self):
    pass


"""
Движемся к следующей точке
"""


def move_next(self, nextX):
    nextY = self.function(nextX)
    self.y = nextY
    self.x = nextX
    return self.x, self.y
