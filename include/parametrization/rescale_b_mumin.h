#ifndef PARAMETRIZATION_RESCALE_B_MUMIN_H
#define PARAMETRIZATION_RESCALE_B_MUMIN_H

#include <Eigen/Core>

#include "energy.h"


namespace parametrization {


// Given the per-triangle values of P, rescale the gradient bounds b, the
//  minimum Augmented Lagrangian penalty values muMin, and the value of the
//  proximal constants h
//
// Inputs:
//  P  n symmetric matrices, provided in the format
//      [P1(0,0); P1(0,1); P1(1,1); P2(0,0); P2(0,1); P2(1,1)] (in 2d) or
//      [P1(0,0); P1(0,1); P1(0,2); P1(1,1); P1(1,2); P1(2,2); ...
//       P2(0,0); P2(0,1); P2(0,2); P2(1,1); P2(1,2); P2(2,2)] (in 3d)
//     If P is not strictly positive definite, behavior is undefined.
//  w  n triangle areas
//  bMargin  when the squared gradient bound b is rescaled it is set to
//           b = bMargin*(1+||grad||^2)
//  template argument parallelize  if this is set to true, will parallelize
//                                 this operation
//  template argument energy  which energy to compute
//  template argument dim  choose between 2d and 3d variant of the method.
// Outputs:
//  b  squared per-triangle gradient bounds
//  muMin  per-triangle minimum Augmented Lagrangian penalty values
//
template <bool parallelize=false,
EnergyType energy=EnergyType::SymmetricGradient,
int dim,
typename DerivedP, typename Derivedw, typename ScalarbMargin, typename Derivedb,
typename DerivedmuMin>
void
rescale_b_mumin(const Eigen::MatrixBase<DerivedP>& P,
                const Eigen::MatrixBase<Derivedw>& w,
                const ScalarbMargin bMargin,
                Eigen::PlainObjectBase<Derivedb>& b,
                Eigen::PlainObjectBase<DerivedmuMin>& muMin);

}

#endif
