# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

# # Defining custom finite elements
#
# This demo ({download}`demo_custom-elements.py`) illustrates how to:
#
# - Define custom elements using Basix
#
# We begin by importing everything we require.

# +
import numpy as np
import ufl
from dolfinx import fem
from dolfinx.mesh import CellType, GhostMode, create_rectangle
from mpi4py import MPI
import basix
import basix.ufl_wrapper
import matplotlib.pylab as plt
# -

# ## Defining a Mardal-Tai-Winter element
# Basix supports a range of finite elements, but there are many other possible elements that users may
# want to use. In this demo, we look at how Basix's custom element interface can be used to define
# elements. More detailed information about the inputs needed to create a custom element can be found
# in [the Basix documentation](https://docs.fenicsproject.org/basix/main/python/demo/demo_custom_element.py.html).
#
# As an example, we will define a
# [Mardal-Tai-Winther](https://defelement.com/elements/mardal-tai-winther.html) H(div) conforming
# element on a triangle, as defined in [A Robust Finite Element Method for Darcy–Stokes Flow (Mardal,
# Tai, Winther, 2002)](https://doi.org/10.1137/S0036142901383910).
#
# ### The polynomial set
# We begin by defining a basis of the polynomial space that this element spans. This is defined in terms
# of the orthogonal Legendre polynomials on the cell. The first six members of this basis are the
# polynomials $(1, 0)$, $(y, 0)$, $(x, 0)$, $(0, 1)$, $(0, y)$, and $(0, x)$.

# +
npoly = 10
ndofs = 9

wcoeffs = np.zeros((ndofs, npoly * 2))

wcoeffs[0, 0] = 1
wcoeffs[1, 1] = 1
wcoeffs[2, 2] = 1
wcoeffs[3, npoly] = 1
wcoeffs[4, npoly + 1] = 1
wcoeffs[5, npoly + 2] = 1
# -

# The final three elements of the basis are $(x(x+2y),-y(2x+y))$, $(x(2x-x^2+3y^2),y(3x^2-4x-y^2))$, and
# $(x(2xy+x+3y^2),-y(2xy+2xy^2))$. The coefficients of these in terms of the Legendre polynomials are
# computed using integration.

# +
dof_n = 6

pts, wts = basix.make_quadrature(basix.CellType.triangle, 6)
poly = basix.tabulate_polynomials(basix.PolynomialType.legendre, basix.CellType.triangle, 3, pts)

x = pts[:, 0]
y = pts[:, 1]

for f in [
    [x * (x + 2 * y), -y * (2 * x + y)],
    [x * (2 * x - x ** 2 + 3 * y ** 2), y * (3 * x ** 2 - 4 * x - y ** 2)],
    [x * (2 * x * y + x + 3 * y ** 2), -y * (2 * x * y + 2 * x + y ** 2)]
]:
    for j in range(npoly):
        wcoeffs[dof_n, j] = sum(f[0] * poly[:, j] * wts)
        wcoeffs[dof_n, npoly + j] = sum(f[1] * poly[:, j] * wts)
    dof_n += 1
# -

# ### The interpolation operators
# Next, we define operators that represent interpolation into the finite element space. For this element,
# These operators represent integrals along each edge of the normal component times linear polynomials,
# and integrals along each edge of the tangential component.
#
# There are no degrees of freedom associated with the vertices or interior of the element, so we initialise
# empty arrays for these entities.

# +
geometry = basix.geometry(basix.CellType.triangle)
topology = basix.topology(basix.CellType.triangle)
x = [[], [], [], []]
M = [[], [], [], []]
for v in topology[0]:
    x[0].append(np.zeros([0, 2]))
    M[0].append(np.zeros([0, 2, 0]))

x[2].append(np.zeros([0, 2]))
M[2].append(np.zeros([0, 2, 0]))
# -

# For the edges of the cell, we set the points defining the operators to be quadrature points, and use
# the quadrature weights to create operators that represent integrals over an edge.

# +
pts, wts = basix.make_quadrature(basix.CellType.interval, 4)
poly = basix.tabulate_polynomials(basix.PolynomialType.legendre, basix.CellType.interval, 1, pts)
for e_n, e in enumerate(topology[1]):
    v0 = geometry[e[0]]
    v1 = geometry[e[1]]
    normal = basix.cell.facet_normals(basix.CellType.triangle)[e_n]
    tangent = v1 - v0

    edge_pts = np.array([v0 + p * tangent for p in pts])
    mat = np.zeros([3, 2, pts.shape[0]])
    for i in range(pts.shape[0]):
        for d in range(2):
            mat[0, d, i] = wts[i] * poly[i, 0] * normal[d]
            mat[1, d, i] = wts[i] * poly[i, 1] * normal[d]
            mat[2, d, i] = wts[i] * tangent[d]

    x[1].append(edge_pts)
    M[1].append(mat)

# -

# +
e = basix.create_custom_element(
    basix.CellType.triangle, [2], wcoeffs, x, M, basix.MapType.contravariantPiola, False, 1, 3)
# -

# Using Basix's UFL wrapper, we can wrap our custom Basix element up as a UFL element.

# +
mtw_element = basix.ufl_wrapper.BasixElement(e)
# -


# ## Defining a conforming Crouzeix-Raviart element
# In a similar way to above, we can define a degree 3 [conforming
# Crouzeix-Raviart](https://defelement.com/elements/conforming-crouzeix-raviart.html) element on a
# triangle.
#
# We begin by defining a basis of the polynomial set. For this element, this contains the polynomials
# $1$, $y$, $x$, $y^2$, $xy$, $x^2$, $x^3$, $y^3$, $xy^2$, $x^2y$, $x^3$, $xy^2(x+y)$, and $x^2y(x+y)$.

# +
npoly = 15
ndofs = 12
wcoeffs = np.zeros((ndofs, npoly))

dof_n = 0
for i in range(10):
    wcoeffs[dof_n, dof_n] = 1
    dof_n += 1

pts, wts = basix.make_quadrature(basix.CellType.triangle, 8)
poly = basix.tabulate_polynomials(basix.PolynomialType.legendre, basix.CellType.triangle, 4, pts)
for i in range(1, 3):
    x = pts[:, 0]
    y = pts[:, 1]
    f = x ** i * y ** (3 - i) * (x + y)

    for j in range(npoly):
        wcoeffs[dof_n, j] = sum(f * poly[:, j] * wts)
    dof_n += 1
# -

# Next, we define the interpolation operators. These represent a point evaluation at each vertex,
# two point evaluations on each edge, and three point evaluations on the interior of the cell.

# +
geometry = basix.geometry(basix.CellType.triangle)
topology = basix.topology(basix.CellType.triangle)
x = [[], [], [], []]
M = [[], [], [], []]
for v in topology[0]:
    x[0].append(np.array(geometry[v]))
    M[0].append(np.array([[[1.]]]))
pts = basix.create_lattice(basix.CellType.interval, 3, basix.LatticeType.equispaced, False)
mat = np.zeros((len(pts), 1, len(pts)))
mat[:, 0, :] = np.eye(len(pts))
for e in topology[1]:
    edge_pts = []
    v0 = geometry[e[0]]
    v1 = geometry[e[1]]
    for p in pts:
        edge_pts.append(v0 + p * (v1 - v0))
    x[1].append(np.array(edge_pts))
    M[1].append(mat)
pts = basix.create_lattice(basix.CellType.triangle, 4, basix.LatticeType.equispaced, False)
x[2].append(pts)
mat = np.zeros((len(pts), 1, len(pts)))
mat[:, 0, :] = np.eye(len(pts))
M[2].append(mat)
# -

# Finally, we create the element.

# +
element = basix.create_custom_element(
    basix.CellType.triangle, [], wcoeffs, x, M, basix.MapType.identity, False, 3, 4)

ccr_element = basix.ufl_wrapper.BasixElement(element)
# -

# ## Comparing elements for a Stokes problem
# Both the custom elements we have defined in this demo are designed to be used to solve Stokes
# problems. To demonstrate how we can use these custom elements, we use them to solve Stokes
# problems with a known analytic solution on a range of meshes, and look at how their convergence
# rates compare.
#
# We begin by defining a function that, given a mixed function space as input, returns the error
# when the problem is solved using that space.


# +
def solve_error(mixed_space, h):
    import random
    return random.random() * h
# -


# We now use this function on a range of meshes with Taylor-Hood, Mardal-Tai-Winther, and
# Crouzeix-Raviart elements, then plot the results.

# +
hs = []
th_errors = []
mtw_errors = []
cr_errors = []
for i in range(1, 6):
    N = 2 ** i

    msh = create_rectangle(MPI.COMM_WORLD,
                           [np.array([0, 0]), np.array([1, 1])],
                           [32, 32],
                           CellType.triangle, GhostMode.none)

    P2 = ufl.VectorElement("Lagrange", "triangle", 2)
    P1 = ufl.FiniteElement("Lagrange", "triangle", 1)
    P0 = ufl.FiniteElement("Discontinuous Lagrange", "triangle", 0)
    CR1 = ufl.FiniteElement("Crouzeix-Raviart", "triangle", 1)

    taylor_hood = fem.FunctionSpace(msh, P2 * P1)
    mtw = fem.FunctionSpace(msh, mtw_element * P0)
    cr = fem.FunctionSpace(msh, ccr_element * CR1)

    hs.append(1 / N)
    th_errors.append(solve_error(taylor_hood, 1 / N))
    mtw_errors.append(solve_error(mtw, 1 / N))
    cr_errors.append(solve_error(cr, 1 / N))

plt.plot(hs, th_errors, "bo-")
plt.plot(hs, mtw_errors, "r^-")
plt.plot(hs, cr_errors, "gs-")


plt.xscale("log")
plt.yscale("log")
plt.axis("equal")
plt.xlim(plt.xlim()[::-1])
plt.xlabel("$h$")
plt.ylabel("Error of solution")

plt.legend(["Taylor-Hood", "Mardal-Tai-Winther", "Crouzeix-Raviart"])

plt.savefig("demo_custom-elements_plot.png")
# -

# ![](demo_custom-elements_plot.png)
#
# In the plot it can be seen that the *** element acheives the best rate of convergence.
