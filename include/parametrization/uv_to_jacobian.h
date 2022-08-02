#ifndef PARAMETRIZATION_UV_TO_JACOBIAN_H
#define PARAMETRIZATION_UV_TO_JACOBIAN_H

#include <Eigen/Core>
#include <Eigen/Sparse>

namespace parametrization {

// Compute the sparse matrix which maps the UV coordinates into the Jacobian
//  matrices that describe the deformation of the mesh into this UV map.
// The coordinate system for the Jacobian matrix is intrinsic to the triangle:
//  the x coordinate runs along th edge 12 of the triangle, the y coordinate is
//  perpendicular to it.
//
// Inputs:
//  V, F  input surface mesh as triangle soup. Is not allowed to contain
//        unreferenced vertices or degenerate triangles (where two vertices are
//        closer to each other than numerics allows).
// Outputs:
//  G  sparse matrix that maps UV coordinates W = [U V] into Jacobian matrices
//     in the format
//      [J1(0,0) J1(0,1); J2(0,0) J2(0,1); J1(1,0) J1(1,1); J2(1,0) J2(1,1)]
//
template <typename DerivedV, typename DerivedF, typename ScalarG>
void
uv_to_jacobian(const Eigen::MatrixBase<DerivedV>& V,
                   const Eigen::MatrixBase<DerivedF>& F,
                   Eigen::SparseMatrix<ScalarG>& G);

}

#endif
