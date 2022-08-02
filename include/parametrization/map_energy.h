#ifndef PARAMETRIZATION_MAP_ENERGY_H
#define PARAMETRIZATION_MAP_ENERGY_H

#include <Eigen/Core>

#include "energy.h"


namespace parametrization {

// Compute the mapping energy, given a map from V to W
//
// Inputs:
//  V, F  an input triangle mesh
//  W  the mapped vertex positions
//  template argument energy  which energy to compute
// Outputs:
//  return value  the mapping energy
//
template <EnergyType energy=EnergyType::SymmetricGradient,
typename DerivedV, typename DerivedF, typename DerivedW>
typename DerivedW::Scalar
map_energy(const Eigen::MatrixBase<DerivedV>& V,
           const Eigen::MatrixBase<DerivedF>& F,
           const Eigen::MatrixBase<DerivedW>& W);

}

#endif
