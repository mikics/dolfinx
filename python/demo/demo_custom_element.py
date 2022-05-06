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
# This demo ({download}`demo_custom_element.py`) illustrates how to:
#
# - Define custom elements using Basix
#
# ... create a degree 3 cronforming Crouzeix-Raviart element.
#
# TODO: describe what's going on here.

# +
import basix
import basix.ufl_wrapper
import numpy as np
from dolfinx import fem
from dolfinx.mesh import create_rectangle, CellType
from mpi4py import MPI


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

element = basix.create_custom_element(
    basix.CellType.triangle, [], wcoeffs, x, M, basix.MapType.identity, False, 3, 4)
# -

# +
ufl_element = basix.ufl_wrapper.BasixElement(element)

mesh = create_rectangle(comm=MPI.COMM_WORLD,
                        points=((0.0, 0.0), (2.0, 1.0)), n=(32, 16),
                        cell_type=CellType.triangle)

V = fem.FunctionSpace(mesh, ufl_element)
# -
