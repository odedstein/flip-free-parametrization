#ifndef PARAMETRIZATION_ARGMIN_U_H
#define PARAMETRIZATION_ARGMIN_U_H

#include <Eigen/Core>
#include <Eigen/Sparse>

namespace parametrization {

// Optimize the Augmented Lagrangian with respect to U,
//  \sum_t ||GW_t - U_t*P_t||^2
//
// Inputs:
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
//  template argument parallelize  if this is set to true, will parallelize
//                                 this operation
//  template argument idim  choose between 2d and 3d variant of the method.
//                          If set to -1, will infer dimension from GW.
// Outputs:
//  U  stacked vector of n rotations as unit complex numbers in the format
//     [reU(0); imU(0); reU(1); imU(1)] (in 2d) or unit quaternions in the
//     format [wU(0); xU(0); yU(0); zU(0); wU(1); xU(1); yU(1); zU(1)] (in 3d)
//
template <bool parallelize=false, int idim=-1,
typename DerivedP, typename DerivedGW, typename DerivedLambda,
typename DerivedU>
void
argmin_U(const Eigen::MatrixBase<DerivedP>& P,
         const Eigen::MatrixBase<DerivedGW>& GW,
         const Eigen::MatrixBase<DerivedLambda>& Lambda,
         Eigen::PlainObjectBase<DerivedU>& U);


// Optimize the Augmented Lagrangian and a proximal term with respect to U,
//  \sum_t ||GW_t - U_t*P_t||^2 + h/2*||U_t - U_t^(k-1)||^2
//
// Inputs:
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
//  h  n proximal constants.
//  U  n rotation matrices, given as n rotation angles (in 2d). This will be
//     used for the proximal term, if one is specified.
//  template argument parallelize  if this is set to true, will parallelize
//                                 this operation
//  template argument idim  choose between 2d and 3d variant of the method.
//                          If set to -1, will infer dimension from GW.
// Outputs:
//  U  stacked vector of n rotations as unit complex numbers in the format
//     [reU(0); imU(0); reU(1); imU(1)] (in 2d) or unit quaternions in the
//     format [wU(0); xU(0); yU(0); zU(0); wU(1); xU(1); yU(1); zU(1)] (in 3d)
//
template <bool parallelize=false, int idim=-1,
typename DerivedP, typename DerivedGW, typename DerivedLambda,
typename Derivedh, typename DerivedU>
void
argmin_U(const Eigen::MatrixBase<DerivedP>& P,
         const Eigen::MatrixBase<DerivedGW>& GW,
         const Eigen::MatrixBase<DerivedLambda>& Lambda,
         const Eigen::MatrixBase<Derivedh>& h,
         Eigen::PlainObjectBase<DerivedU>& U);

}

#endif
