#ifndef PARAMETRIZATION_STEP_LAMBDA_H
#define PARAMETRIZATION_STEP_LAMBDA_H

#include <Eigen/Core>

namespace parametrization {

// Perform the Lambda step of the Augmented Lagrangian optimization.
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
//  template argument parallelize  if this is set to true, will parallelize
//                                 this operation
//  template argument idim  choose between 2d and 3d variant of the method.
//                          If set to -1, will infer dimension from GW.
// Outputs:
//  Lambda  the updated Lambda variable
//
//
template <bool parallelize=false, int idim=-1,
typename DerivedU, typename DerivedP, typename DerivedGW,
typename DerivedLambda>
void
step_Lambda(const Eigen::MatrixBase<DerivedU>& U,
            const Eigen::MatrixBase<DerivedP>& P,
            const Eigen::MatrixBase<DerivedGW>& GW,
            Eigen::PlainObjectBase<DerivedLambda>& Lambda);

}

#endif
