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

from dolfinx import mesh
from ufl import (TrialFunction, TestFunction, inner, grad, div, dx,
                 as_vector, SpatialCoordinate, sin, cos, pi)
from petsc4py import PETSc
# -

# ## Defining a conforming Crouzeix-Raviart element
# Basix supports a range of finite elements, but there are many other possible elements that users may
# want to use. In this demo, we look at how Basix's custom element interface can be used to define
# elements. More detailed information about the inputs needed to create a custom element can be found
# in [the Basix documentation](https://docs.fenicsproject.org/basix/main/python/demo/demo_custom_element.py.html).
#
# As an example, we will define a [conforming
# Crouzeix-Raviart](https://defelement.com/elements/conforming-crouzeix-raviart.html) element on a
# triangle, as defined in [Conforming and nonconforming finite element methods for solving the
# stationary Stokes equations (Crouzeix, Raviart, 1973)](https://doi.org/10.1051/m2an/197307R300331).
#
# ### The polynomial set
# We begin by defining a basis of the polynomial space that this element spans. This is defined in terms
# of the orthogonal Legendre polynomials on the cell. The first six members of this basis are the
# polynomials $1$, $y$, $x$, $y^2$, $xy$, $x^2$, $x^3$, $y^3$, $xy^2$, $x^2y$, and $x^3$.

# +
npoly = 15
ndofs = 12
wcoeffs = np.zeros((ndofs, npoly))

dof_n = 0
for i in range(10):
    wcoeffs[dof_n, dof_n] = 1
    dof_n += 1
# -

# The final two elements of the basis are $xy^2(x+y)$ and $x^2y(x+y)$. The coefficients of these in terms
# of the Legendre polynomials are computed using integration.

# +
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

# ### The interpolation operators
# Next, we define functionalsthat represent interpolation into the finite element space. For this element,
# These operators represent integrals along each edge of the normal component times linear polynomials,
# and integrals along each edge of the tangential component.
#
# The functionals for this element are a point evaluation at each vertex, two point evaluations on each
# edge, and three point evaluations on the interior of the cell.

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

# Finally, we create the element. Using Basix's UFL wrapper, we can wrap our custom Basix element up as
# a UFL element.

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
def norm_L2(comm, v):
    """Compute the L2(Ω)-norm of v"""
    return np.sqrt(comm.allreduce(
        fem.assemble_scalar(fem.form(inner(v, v) * dx)), op=MPI.SUM))


def domain_average(msh, v):
    """Compute the average of a function over the domain"""
    vol = msh.comm.allreduce(
        fem.assemble_scalar(fem.form(
            fem.Constant(msh, PETSc.ScalarType(1.0)) * dx)), op=MPI.SUM)
    return 1 / vol * msh.comm.allreduce(
        fem.assemble_scalar(fem.form(v * dx)), op=MPI.SUM)


def solve_stokes(V, Q, k, boundary_marker, f, u_bc_expr):
    msh = V.mesh
    u = TrialFunction(V)
    v = TestFunction(V)
    p = TrialFunction(Q)
    q = TestFunction(Q)
    a_00 = inner(grad(u), grad(v)) * dx
    a_01 = - inner(p, div(v)) * dx
    a_10 = - inner(div(u), q) * dx
    a_11 = fem.Constant(msh, PETSc.ScalarType(0.0)) * inner(p, q) * dx
    a = fem.form([[a_00, a_01],
                  [a_10, a_11]])
    L = fem.form([inner(f, v) * dx,
                  inner(fem.Constant(msh, PETSc.ScalarType(0.0)), q) * dx])
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_bc_expr, V.element.interpolation_points))
    boundary_facets = mesh.locate_entities_boundary(
        msh, msh.topology.dim - 1, boundary_marker)
    boundary_vel_dofs = fem.locate_dofs_topological(
        V, msh.topology.dim - 1, boundary_facets)
    bc_u = fem.dirichletbc(u_bc, boundary_vel_dofs)
    pressure_dof = fem.locate_dofs_geometrical(
        Q, lambda x: np.logical_and(np.isclose(x[0], 0.0),
                                    np.isclose(x[1], 0.0)))
    bc_p = fem.dirichletbc(PETSc.ScalarType(0.0), pressure_dof, Q)
    bcs = [bc_u, bc_p]
    A = fem.petsc.assemble_matrix_block(a, bcs=bcs)
    A.assemble()
    b = fem.petsc.assemble_vector_block(L, a, bcs=bcs)
    # Create and configure solver
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("superlu_dist")
    # Compute solution
    x = A.createVecLeft()
    ksp.solve(b, x)
    # Create Functions and scatter x solution
    u, p = fem.Function(V), fem.Function(Q)
    offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    u.x.array[:offset] = x.array_r[:offset]
    u.x.scatter_forward()
    p.x.array[:(len(x.array_r) - offset)] = x.array_r[offset:]
    p.x.scatter_forward()
    return u, p


def boundary_marker(x):
    return np.logical_or(np.logical_or(np.isclose(x[0], 0.0),
                                       np.isclose(x[0], 1.0)),
                         np.logical_or(np.isclose(x[1], 0.0),
                                       np.isclose(x[1], 1.0)))


def solve_error(msh, e1, e2):
    V = fem.FunctionSpace(msh, e1)
    Q = fem.FunctionSpace(msh, e2)
    k = 2
    x = SpatialCoordinate(msh)
    # NOTE u_e must be divergence free
    u_e = as_vector((sin(pi * x[0]) * cos(pi * x[1]),
                     - sin(pi * x[1]) * cos(pi * x[0])))
    p_e = sin(pi * x[0]) * cos(pi * x[1])
    f = - div(grad(u_e)) + grad(p_e)
    u_h, p_h = solve_stokes(V, Q, k, boundary_marker, f, u_e)
    e_u = norm_L2(msh.comm, u_h - u_e)
    e_div_u = norm_L2(msh.comm, div(u_h))
    p_h_avg = domain_average(msh, p_h)
    p_e_avg = domain_average(msh, p_e)
    e_p = norm_L2(msh.comm, (p_h - p_h_avg) - (p_e - p_e_avg))
    return e_u, e_div_u, e_p
# -


# We now use this function on a range of meshes with Taylor-Hood, Mardal-Tai-Winther, and
# Crouzeix-Raviart elements, then plot the results.

# +
P2 = ufl.VectorElement("Lagrange", "triangle", 2)
P1 = ufl.FiniteElement("Lagrange", "triangle", 1)
P0 = ufl.FiniteElement("Discontinuous Lagrange", "triangle", 0)
CR1 = ufl.FiniteElement("Crouzeix-Raviart", "triangle", 1)
CCR = ufl.VectorElement(ccr_element)

hs = []
th_errors = []
cr_errors = []
for i in range(1, 6):
    N = 2 ** i

    msh = create_rectangle(MPI.COMM_WORLD,
                           [np.array([0, 0]), np.array([1, 1])],
                           [N, N],
                           CellType.triangle, GhostMode.none)

    hs.append(1 / N)
    th_errors.append(solve_error(msh, P2, P1))
    cr_errors.append(solve_error(msh, CCR, CR1))

plt.figure(figsize=(17, 8))
for i, ylabel in enumerate(["u-u_h", "\\operatorname{div}(u-u_h)", "p-p_h"]):
    plt.subplot(1, 3, i + 1)
    plt.plot(hs, [j[i] for j in th_errors], "bo-")
    plt.plot(hs, [j[i] for j in cr_errors], "gs-")

    plt.xscale("log")
    plt.yscale("log")
    plt.axis("equal")
    plt.xlim(plt.xlim()[::-1])
    plt.xlabel("$h$")
    plt.ylabel(f"$\\|{ylabel}\\|_2$")

    plt.legend(["Taylor-Hood", "Crouzeix-Raviart"])

plt.savefig("demo_custom-elements_plot.png")
# -

# ![](demo_custom-elements_plot.png)
#
# In the plot it can be seen that the *** element acheives the best rate of convergence.
