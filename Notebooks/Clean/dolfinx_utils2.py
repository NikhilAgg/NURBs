from dolfinx import fem
import numpy as np

def create_func(expr, func_space):
    f = fem.Function(func_space)
    if callable(expr) or type(expr) == fem.Expression:
        f.interpolate(expr)
    elif np.isscalar(expr):
        f.interpolate(lambda x: x[0]+expr)
    else:
        raise ValueError("Expression must be a constant or a function")

    return f

def conjugate_function(p, V):
    p_conj = fem.Function(V)
    p_conj.vector[:] = np.conjugate(p.vector[:])

    return p_conj

def quadratic_product(func1, mat, func2, V):
    func3 = fem.Function(V)
    mat.mult(func2.vector, func3.vector)

    func3 = conjugate_function(func3, V) #Since vector.dot conjugates the second vector

    product = func1.vector.dot(func3.vector)

    return product
