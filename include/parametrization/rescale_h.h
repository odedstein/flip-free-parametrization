#ifndef PARAMETRIZATION_RESCALE_H_H
#define PARAMETRIZATION_RESCALE_H_H

#include <Eigen/Core>

namespace parametrization {


// Given the per-triangle values of P, the gradient bounds b, and the
//  minimum Augmented Lagrangian penalty values muMin, compute the value of the
//  proximal constants h
//
// Inputs:
//  w  n triangle areas
//  mu  n penalty terms
//  b  squared per-triangle gradient bounds
//  template argument parallelize  if this is set to true, will parallelize
//                                 this operation
// Outputs:
//  h  per-triangle proximal constants. Attention! These are a scaled version of
//     the parameter h in the paper, for implementation reasons.
//     code ht = mut*(paper ht)
//
template <bool parallelize=false,
typename Derivedw, typename Derivedmu, typename Derivedb, typename Derivedh>
void
rescale_h(const Eigen::MatrixBase<Derivedw>& w,
          const Eigen::MatrixBase<Derivedmu>& mu,
          const Eigen::MatrixBase<Derivedb>& b,
          Eigen::PlainObjectBase<Derivedh>& h);

}

#endif
