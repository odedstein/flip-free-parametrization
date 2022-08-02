#ifndef PARAMETRIZATION_LAGRANGIAN_H
#define PARAMETRIZATION_LAGRANGIAN_H

#include <Eigen/Core>

#include "energy.h"


namespace parametrization {

// Compute the value of the augmented Lagrangian, the objective of the energy
//  optimization.
//
// Inputs:
//  U  stacked vector of n rotations as unit complex numbers in the format
//     [reU(0); imU(0); reU(1); imU(1)] (in 2d) or unit quaternions in the
//     format [wU(0); xU(0); yU(0); zU(0); wU(1); xU(1); yU(1); zU(1)] (in 3d)
//  P  n symmetric matrices, provided in the format
//      [P1(0,0); P1(0,1); P1(1,1); P2(0,0); P2(0,1); P2(1,1)] (in 2d) or
//      [P1(0,0); P1(0,1); P1(0,2); P1(1,1); P1(1,2); P1(2,2); ...
//       P2(0,0); P2(0,1); P2(0,2); P2(1,1); P2(1,2); P2(2,2)] (in 3d)
//     If P is not strictly positive definite, behavior is undefined.
//  GW  n matrices in the format
//      [GW1(0,0) GW1(0,1); GW2(0,0) GW2(0,1); ...
//       GW1(1,0) GW1(1,1); GW2(1,0) GW2(1,1)] (or analogous in 3d)
//  Lambda  n matrices in the format
//          [Lambda1(0,0) Lambda1(0,1); Lambda2(0,0) Lambda2(0,1); ...
//           Lambda1(1,0) Lambda1(1,1); Lambda2(1,0) Lambda2(1,1)]
//          (or analogous in 3d)
//  w  n triangle areas
//  mu  n penalty terms
//  template argument parallelize  if this is set to true, will parallelize
//                                 this operation
//  template argument energy  which energy to compute
//  template argument idim  choose between 2d and 3d variant of the method.
//                          If set to -1, will infer dimension from GW.
//
// Outputs:
//  return value  the Lagrangian of this configuration
//
template <bool parallelize=false,
EnergyType energy=EnergyType::SymmetricGradient,
int idim=-1,
typename DerivedU, typename DerivedP, typename DerivedGW,
typename DerivedLambda, typename Derivedw, typename Derivedmu>
typename DerivedU::Scalar
lagrangian(const Eigen::MatrixBase<DerivedU>& U,
           const Eigen::MatrixBase<DerivedP>& P,
           const Eigen::MatrixBase<DerivedGW>& GW,
           const Eigen::MatrixBase<DerivedLambda>& Lambda,
           const Eigen::MatrixBase<Derivedw>& w,
           const Eigen::MatrixBase<Derivedmu>& mu);

}

#endif
