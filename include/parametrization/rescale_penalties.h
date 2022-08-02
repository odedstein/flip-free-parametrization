#ifndef PARAMETRIZATION_RESCALE_PENALTIES_H
#define PARAMETRIZATION_RESCALE_PENALTIES_H

#include <Eigen/Core>

namespace parametrization {


// Rescale the Lagrangian penalties based on per-face primal and dual errors.
//
// Inputs:
//  ePs  squared per-face primal error
//  eDs  squared per-face dual error
//  penaltyIncrease  how much to increase a penalty if increased
//  penaltyDecrease  how much to decrease a penalty if decreased
//  differenceToRescale  how large the discrepancy between primal and dual has
//                       to be to trigger a rescale
//  mu  n penalty terms
//  Lambda  n matrices in the format
//          [Lambda1(0,0) Lambda1(0,1); Lambda2(0,0) Lambda2(0,1); ...
//           Lambda1(1,0) Lambda1(1,1); Lambda2(1,0) Lambda2(1,1)]
//          (or analogous in 3d)
//  muMin  mu can never be lower than this.
//  template argument parallelize  if this is set to true, will parallelize
//                                 this operation
//  template argument idim  choose between 2d and 3d variant of the method.
//                          If set to -1, will infer dimension from GW.
// Outputs:
//  mu  n penalty terms
//  Lambda  n matrices in the format
//          [Lambda1(0,0) Lambda1(0,1); Lambda2(0,0) Lambda2(0,1); ...
//           Lambda1(1,0) Lambda1(1,1); Lambda2(1,0) Lambda2(1,1)]
//  return value  whether any rescaling happened or not
//
template <bool parallelize=false, int idim=-1,
typename DerivedePs, typename DerivedeDs, typename ScalarpenaltyIncrease,
typename ScalarpenaltyDecrease, typename ScalardifferenceToRescale,
typename Derivedmu, typename DerivedLambda, typename DerivedmuMin=
Eigen::Matrix<typename Derivedmu::Scalar, Eigen::Dynamic, 1> >
bool
rescale_penalties(const Eigen::MatrixBase<DerivedePs>& ePs,
                  const Eigen::MatrixBase<DerivedeDs>& eDs,
                  const ScalarpenaltyIncrease penaltyIncrease,
                  const ScalarpenaltyDecrease penaltyDecrease,
                  const ScalardifferenceToRescale differenceToRescale,
                  Eigen::PlainObjectBase<Derivedmu>& mu,
                  Eigen::PlainObjectBase<DerivedLambda>& Lambda,
                  const Eigen::MatrixBase<DerivedmuMin>& muMin=DerivedmuMin());

}

#endif
