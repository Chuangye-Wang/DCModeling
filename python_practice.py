""" Multi-inheritance
"""


class Calculus:
    def Sum(self, a, b):
        return a + b;


class Calculus1:
    def Sum(self, a, b):
        return a * b;


# Inherit from left to right.
class Derived(Calculus1, Calculus):
    def Div(self, a, b):
        return a / b;


d = Derived()
print(d.Sum(10, 30))
# print(d.Mul(10,30))
print(d.Div(10, 30))

""" Abstract class 
(1). An Abstract class can contain the both method normal and abstract method.
(2). If a method is not decorated with @abstractmethod decorator, it is a normal method.
(3). An Abstract cannot be instantiated; we cannot create objects for the abstract class.
(4). All the abstract method must be overridden in the subclass
"""

from abc import ABC, abstractmethod


class Absclass(ABC):
    def print(self, x):
        print("Passed value: ", x)

    @abstractmethod
    def task(self):
        print("We are inside Absclass task")

    # @abstractmethod
    def catch(self):
        print("To catch an object.")


class test_class(Absclass):
    def task(self):
        print("We are inside test_class task")


class example_class(Absclass):
    def task(self):
        print("We are inside example_class task")


# object of test_class created
test_obj = test_class()
test_obj.task()
test_obj.print(100)

# object of example_class created
example_obj = example_class()
example_obj.task()
example_obj.print(200)

# show all abstract methods
print(Absclass.__abstractmethods__)
print(getattr(Absclass, "print", False))
print(getattr(Absclass, "prin", False))

""" Methods with same name
(1). The last method will override all the pre-defined methods with the same name.
(2). The method (same name) with different number of parameters will also be overridden in Python, while it will not be 
    overridden in Java.
"""


def func(a, b=2):
    return a + b


def func(a):
    return a


print(func(1))
# print(func(1, 2))


""" MRO (Method resolution order)
Depth-first left-to-right,
Removing all duplicates, except for the last one
"""
class First(object):
    def __init__(self):
        print("first-1")
        super(First, self).__init__()
        print("first")

class Second(First):
    def __init__(self):
        print("second-1")
        super(Second, self).__init__()
        print("second")

class Third(First): # add parent class first will create different MRO.
    def __init__(self):
        print("third-1")
        super(Third, self).__init__()
        print("third")

class Fourth(Second, Third):
    def __init__(self):
        super(Fourth, self).__init__()
        print("fourth")
        self.a = 10
        self._b = 101
        # self.__b = 102

    # @property
    # def b(self):
    #     return self.__b
    @property
    def b(self):
        return self._b


fourth = Fourth()
# print method resolution order
print(Fourth.__mro__)
fourth.a = 11
fourth._b = 111
print(fourth.a, fourth.b)
print("super returns: ", super(Second))
# print(Fourth.__bases__)
print(Fourth.__dir__)
print(Second)


from scipy.optimize import least_squares
import numpy as np


def fun(model, x, y):
    return model[0] * x + model[1] - y


# init_model = {"a": 1, "b": 1}
init_model = [1, 2]
x_d = np.array([1, 2, 3, 4])
y_d = np.array([2.1, 2.95, 3.98, 5.04])


res_lsq = least_squares(fun, init_model, args=(x_d, y_d))

print(res_lsq)


def func1(**kwargs):
    if "line" not in kwargs:
        kwargs["line"] = "dashed"
    print(kwargs)


func1()

