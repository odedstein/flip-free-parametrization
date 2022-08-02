#ifndef PARAMETRIZATION_BFF_H
#define PARAMETRIZATION_BFF_H

#include <Eigen/Core>

namespace parametrization {

// Compute the bff parametrization of a mesh.
// This uses the implementation of "Boundary First Flattening" by Rohan Sawhney
//  and Keenan Crane, 2017, ACM TOG
//
// Input:
//  V, F  input surface mesh as triangle soup. Is not allowed to contain
//        unreferenced vertices.
// Output:
//  return value  whether the linear solve was successful
//  W  UV coordinates
//
template <typename DerivedV, typename DerivedF, typename DerivedW>
bool
bff(const Eigen::MatrixBase<DerivedV>& V,
    const Eigen::MatrixBase<DerivedF>& F,
    Eigen::PlainObjectBase<DerivedW>& W);

}

#endif
