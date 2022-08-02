#ifndef PARAMETRIZATION_ENERGY_H
#define PARAMETRIZATION_ENERGY_H

#include <Eigen/Core>

namespace parametrization {

enum class EnergyType {
    SymmetricGradient, //E(P) = 0.5*||P||^2 - log(det(P))
    SymmetricDirichlet //E(P) = 0.5*||P||^2 + 0.5*||inv(P)||^2
};

// Compute parametrization energy, given the SPD matrices stacked in P
//
// Inputs:
//  P  n symmetric matrices, provided in the format
//      [P1(0,0); P1(0,1); P1(1,1); P2(0,0); P2(0,1); P2(1,1)] (in 2d) or
//      [P1(0,0); P1(0,1); P1(0,2); P1(1,1); P1(1,2); P1(2,2); ...
//       P2(0,0); P2(0,1); P2(0,2); P2(1,1); P2(1,2); P2(2,2)] (in 3d)
//     If P is not strictly positive definite, behavior is undefined.
//  w  n triangle areas / tet volumes
//  template argument energy  which energy to compute
//  template argument idim  choose between 2d and 3d variant of the method.
//                          If set to -1, will infer dimension from GW.
//
template <EnergyType energyT=EnergyType::SymmetricGradient, int idim=-1,
typename DerivedP, typename Derivedw>
typename DerivedP::Scalar
energy_from_P(const Eigen::MatrixBase<DerivedP>& P,
              const Eigen::MatrixBase<Derivedw>& w);

// Compute parametrization energy, given the Jacobians in GW
//
// Inputs:
//  GW  n matrices in the format
//      [GW1(0,0) GW1(0,1); GW2(0,0) GW2(0,1); ...
//       GW1(1,0) GW1(1,1); GW2(1,0) GW2(1,1)] (or analogous in 3D)
//  w  n triangle areas / tet volumes
//  template argument energy  which energy to compute
//  template argument idim  choose between 2d and 3d variant of the method.
//                          If set to -1, will infer dimension from GW.
//
template <EnergyType energyT=EnergyType::SymmetricGradient, int idim=-1,
typename DerivedGW, typename Derivedw>
typename DerivedGW::Scalar
energy_from_GW(const Eigen::MatrixBase<DerivedGW>& GW,
               const Eigen::MatrixBase<Derivedw>& w);

}

#endif
