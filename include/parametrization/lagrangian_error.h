#ifndef PARAMETRIZATION_LAGRANGIAN_ERROR_H
#define PARAMETRIZATION_LAGRANGIAN_ERROR_H

#include <Eigen/Core>
#include <Eigen/Sparse>

namespace parametrization {

// Compute the Lagrangian primal and dual errors (per face) as well as the total
//  primal and dual errors.
// Careful: this computes squared quantities!
//
// Inputs:
//  U  stacked vector of n rotations as unit complex numbers in the format
//     [reU(0); imU(0); reU(1); imU(1)] (in 2d) or unit quaternions in the
//     format [wU(0); xU(0); yU(0); zU(0); wU(1); xU(1); yU(1); zU(1)] (in 3d)
//  U0  stacked vector of n rotations from the previous step as unit complex
//      numbers in the format [reU(0); imU(0); reU(1); imU(1)] (in 2d) or unit
//      quaternions in the format
//      [wU(0); xU(0); yU(0); zU(0); wU(1); xU(1); yU(1); zU(1)] (in 3d)
//  P  n symmetric matrices, provided in the format
//     [P1(0,0); P2(0,0); P1(0,1); P2(0,1); P1(1,1); P2(1,1)].
//     If P is not strictly positive definite, behavior is undefined.
//  P0  n symmetric matrices from the previous step, provided in the format
//     [P1(0,0); P2(0,0); P1(0,1); P2(0,1); P1(1,1); P2(1,1)].
//     If P is not strictly positive definite, behavior is undefined.
//  GW  (current step) n matrices in the format
//      [GW1(0,0) GW1(0,1); GW2(0,0) GW2(0,1); ...
//       GW1(1,0) GW1(1,1); GW2(1,0) GW2(1,1)]
//  GtLambda  the uv_to_jacobian matrix, transposed, multiplied with the
//            Lagrangian variable Lambda
//  mu  n penalty terms
//  template argument parallelize  if this is set to true, will parallelize
//                                 this operation
//  template argument idim  choose between 2d and 3d variant of the method.
//                          If set to -1, will infer dimension from GW.
// Outputs:
//  eP  squared total primal error
//  eD  squared total dual error
//  ePs  squared per-face primal error
//  eDs  squared per-face dual_error
//
template <bool parallelize=false, int idim=-1,
typename DerivedU, typename DerivedU0, typename DerivedP, typename DerivedP0,
typename DerivedGW, typename DerivedGW0, typename Derivedmu,
typename Derivedh, typename ScalareP, typename ScalareD, typename DerivedePs,
typename DerivedeDs>
void
lagrangian_errors(const Eigen::MatrixBase<DerivedU>& U,
                  const Eigen::MatrixBase<DerivedU0>& U0,
                  const Eigen::MatrixBase<DerivedP>& P,
                  const Eigen::MatrixBase<DerivedP0>& P0,
                  const Eigen::MatrixBase<DerivedGW>& GW,
                  const Eigen::MatrixBase<DerivedGW0>& GW0,
                  const Eigen::MatrixBase<Derivedmu>& mu,
                  const Eigen::MatrixBase<Derivedh>& h,
                  ScalareP& eP,
                  ScalareD& eD,
                  Eigen::PlainObjectBase<DerivedePs>& ePs,
                  Eigen::PlainObjectBase<DerivedeDs>& eDs);


// Compute the total Lagrangian primal and dual error.
// Careful: this computes squared quantities!
//
// Inputs:
//  U  stacked vector of n rotations as unit complex numbers in the format
//     [reU(0); imU(0); reU(1); imU(1)] (in 2d) or unit quaternions in the
//     format [wU(0); xU(0); yU(0); zU(0); wU(1); xU(1); yU(1); zU(1)] (in 3d)
//  U0  stacked vector of n rotations from the previous step as unit complex
//      numbers in the format [reU(0); imU(0); reU(1); imU(1)] (in 2d) or unit
//      quaternions in the format
//      [wU(0); xU(0); yU(0); zU(0); wU(1); xU(1); yU(1); zU(1)] (in 3d)
//  P  n symmetric matrices, provided in the format
//     [P1(0,0); P2(0,0); P1(0,1); P2(0,1); P1(1,1); P2(1,1)].
//     If P is not strictly positive definite, behavior is undefined.
//  P0  n symmetric matrices from the previous step, provided in the format
//     [P1(0,0); P2(0,0); P1(0,1); P2(0,1); P1(1,1); P2(1,1)].
//     If P is not strictly positive definite, behavior is undefined.
//  GW  (current step) n matrices in the format
//      [GW1(0,0) GW1(0,1); GW2(0,0) GW2(0,1); ...
//       GW1(1,0) GW1(1,1); GW2(1,0) GW2(1,1)]
//  GtLambda  the uv_to_jacobian matrix, transposed, multiplied with the
//            Lagrangian variable Lambda
//  mu  n penalty terms
//  template argument parallelize  if this is set to true, will parallelize
//                                 this operation
//  template argument idim  choose between 2d and 3d variant of the method.
//                          If set to -1, will infer dimension from GW.
// Outputs:
//  eP  squared total primal error
//  eD  squared total dual error
//
template <bool parallelize=false, int idim=-1,
typename DerivedU, typename DerivedU0, typename DerivedP, typename DerivedP0,
typename DerivedGW, typename DerivedGW0, typename Derivedmu,
typename Derivedh, typename ScalareP, typename ScalareD>
void
lagrangian_error(const Eigen::MatrixBase<DerivedU>& U,
                 const Eigen::MatrixBase<DerivedU0>& U0,
                 const Eigen::MatrixBase<DerivedP>& P,
                 const Eigen::MatrixBase<DerivedP0>& P0,
                 const Eigen::MatrixBase<DerivedGW>& GW,
                 const Eigen::MatrixBase<DerivedGW0>& GW0,
                 const Eigen::MatrixBase<Derivedmu>& mu,
                 const Eigen::MatrixBase<Derivedh>& h,
                 ScalareP& eP,
                 ScalareD& eD);

}

#endif
