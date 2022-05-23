
#include "poisson.h"
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/common/sort.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/io/XDMFFile.h>
#include <dolfinx/mesh/utils.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtl/xspan.hpp>
using namespace dolfinx;
using T = PetscScalar;

int main(int argc, char* argv[])
{
  dolfinx::init_logging(argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);

  {
    double H = 2;
    // Create mesh and function space
    auto mesh = std::make_shared<mesh::Mesh>(mesh::create_rectangle(
        MPI_COMM_WORLD, {{{0.0, 0.0}, {1.0, H}}}, {32, 16},
        mesh::CellType::triangle, mesh::GhostMode::shared_facet));

    // Locate facets in the middle of the mesh
    std::vector<std::int32_t> facets
        = mesh::locate_entities(*mesh, mesh->topology().dim() - 1,
                                [](auto&& x) -> xt::xtensor<bool, 1>
                                { return xt::isclose(xt::row(x, 0), 0.5); });
    // Sort facets
    radix_sort(xtl::span(facets));

    // Create meshtag for facets
    std::vector<std::int32_t> facet_marker(facets.size(), 3);
    mesh::MeshTags<std::int32_t> facet_tag(mesh, mesh->topology().dim() - 1,
                                           facets, facet_marker);

    std::vector<std::int32_t> left_cells;
    left_cells.reserve(facets.size());

    std::vector<std::int32_t> right_cells;
    right_cells.reserve(facets.size());

    {
      // This scope is used to compute all cells connected to the facets, and
      // split them into two groups, those left and right of the interface
      mesh->topology_mutable().create_connectivity(mesh->topology().dim() - 1,
                                                   mesh->topology().dim());
      auto f_to_c = mesh->topology().connectivity(mesh->topology().dim() - 1,
                                                  mesh->topology().dim());

      // Find all cells connected to facet
      std::vector<std::int32_t> all_cells;
      all_cells.reserve(2 * facets.size());
      std::for_each(facets.cbegin(), facets.cend(),
                    [&f_to_c, &all_cells](auto f)
                    {
                      auto cells = f_to_c->links(f);
                      assert(cells.size() == 2);
                      all_cells.push_back(cells[0]);
                      all_cells.push_back(cells[1]);
                    });

      // Compute midpoint of each cell connected to the facet
      std::vector<double> midpoints
          = mesh::compute_midpoints(*mesh, mesh->topology().dim(), all_cells);

      // Split into cells left and right of the boundary

      for (std::size_t i = 0; i < all_cells.size(); ++i)
      {
        if (midpoints[3 * i] < 0.5)
          left_cells.push_back(all_cells[i]);
        else
          right_cells.push_back(all_cells[i]);
      }
    }

    auto W = std::make_shared<fem::FunctionSpace>(fem::create_functionspace(
        functionspace_form_poisson_J, "weight", mesh));
    // Create weight function deactivating all contributions from the left side
    // of the boundary
    auto w = std::make_shared<fem::Function<T>>(W);
    w->interpolate(
        [](auto& x)
        {
          std::array<std::size_t, 1> shape = {x.shape(1)};
          xt::xtensor<T, 1> values(shape);
          std::fill(values.begin(), values.end(), 0);
          return values;
        },
        left_cells);
    w->interpolate(
        [](auto& x)
        {
          std::array<std::size_t, 1> shape = {x.shape(1)};
          xt::xtensor<T, 1> values(shape);
          std::fill(values.begin(), values.end(), 1);
          return values;
        },
        right_cells);
    w->x()->scatter_fwd();

    // Define variational form
    // As we have defined w to be 1 on the left side, the integral will be
    // 1*inner((-1,0), (1, 0))*dS = -1 * dS = -H
    auto J = std::make_shared<fem::Form<T>>(fem::create_form<T>(
        *form_poisson_J, {}, {{"weight", w}}, {},
        {{fem::IntegralType::interior_facet, &facet_tag}}, mesh));
    const T contribution = fem::assemble_scalar(*J);
    T result;
    MPI_Allreduce(&contribution, &result, 1, dolfinx::MPI::mpi_type<T>(),
                  MPI_SUM, mesh->comm());

    std::cout << "Weighted result " << result << "\n";

    if (result != -H)
      throw std::runtime_error("Wrong scalar value obtained");

    // Compute the contribution without weights, to illustrate that they do not
    // make sense
    const std::map<std::string, std::shared_ptr<const fem::Function<T>>>
        _coeffs;
    auto Jp = std::make_shared<fem::Form<T>>(fem::create_form<T>(
        *form_poisson_J_plus, {}, _coeffs, {},
        {{fem::IntegralType::interior_facet, &facet_tag}}, mesh));
    T p_val = fem::assemble_scalar(*Jp);
    MPI_Allreduce(&p_val, &result, 1, dolfinx::MPI::mpi_type<T>(), MPI_SUM,
                  mesh->comm());

    std::cout << "n('+')[0]*dS " << p_val << "\n";

    auto Jm = std::make_shared<fem::Form<T>>(fem::create_form<T>(
        *form_poisson_J_minus, {}, _coeffs, {},
        {{fem::IntegralType::interior_facet, &facet_tag}}, mesh));
    T m_val = fem::assemble_scalar(*Jm);
    MPI_Allreduce(&m_val, &result, 1, dolfinx::MPI::mpi_type<T>(), MPI_SUM,
                  mesh->comm());

    std::cout << "n('-')[0]*dS " << m_val << "\n";

    // Debugging output
    dolfinx::io::XDMFFile xdmf(mesh->comm(), "weight.xdmf", "w");
    xdmf.write_mesh(*mesh);
    xdmf.write_meshtags(facet_tag, "/Xdmf/Domain/Grid/Geometry");
    xdmf.write_function(*w, 0.0);
    xdmf.close();
  }

  PetscFinalize();

  return 0;
}
