
from ufl import (FacetNormal, FiniteElement, Coefficient, FunctionSpace, Mesh,
                 VectorElement, as_vector, dS, inner, triangle)

coord_element = VectorElement("Lagrange", triangle, 1)
mesh = Mesh(coord_element)


n = FacetNormal(mesh)
vec = as_vector((1, 0))

el = FiniteElement("Discontinuous Lagrange", mesh.ufl_cell(), 0)
W = FunctionSpace(mesh, el)
weight = Coefficient(W)


def torque(restriction):
    func = weight * inner(n, vec)
    return func(restriction)*dS(3)


J = torque("+") + torque("-")

J_plus = inner(n("+"), vec) * dS(3)
J_minus = inner(n("-"), vec) * dS(3)

forms = [J, J_plus, J_minus]
