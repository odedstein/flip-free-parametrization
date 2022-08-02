#ifndef PARAMETRIZATION_ARGMIN_W_H
#define PARAMETRIZATION_ARGMIN_W_H

#include <Eigen/Core>
#include <Eigen/Sparse>

namespace parametrization {

// Optimize the Augmented Lagrangian with respect to W, fix the fixed vertices
//  to fixedTo, and apply the linear constraint beq.
// No additional constraints will be applied here, so make sure that the problem
//  is actually invertible.
//
// Inputs:
//  GtMG  precomputed solver for the matrix G'*M*G
//        if beq is not trivial, make sure this solver can handle non-spd
//        problems.
//  fixedTo  inhomogeneous part of fixed constraint on W.
//  beq  inhomogeneous part of linear constraint on W.
//  GtM  sparse matrix precomputing G'*M
//  U  stacked vector of n rotations as unit complex numbers in the format
//     [reU(0); imU(0); reU(1); imU(1)] (in 2d) or unit quaternions in the
//     format [wU(0); xU(0); yU(0); zU(0); wU(1); xU(1); yU(1); zU(1)] (in 3d)
//  P  n symmetric matrices, provided in the format
//      [P1(0,0); P1(0,1); P1(1,1); P2(0,0); P2(0,1); P2(1,1)] (in 2d) or
//      [P1(0,0); P1(0,1); P1(0,2); P1(1,1); P1(1,2); P1(2,2); ...
//       P2(0,0); P2(0,1); P2(0,2); P2(1,1); P2(1,2); P2(2,2)] (in 3d)
//     If P is not strictly positive definite, behavior is undefined.
//  Lambda  n matrices in the format
//          [Lambda1(0,0) Lambda1(0,1); Lambda2(0,0) Lambda2(0,1); ...
//           Lambda1(1,0) Lambda1(1,1); Lambda2(1,0) Lambda2(1,1)]
//          (or analogous in 3d)
//  template argument parallelize  if this is set to true, will parallelize
//                                 this operation
//  template argument idim  choose between 2d and 3d variant of the method.
//                          If set to -1, will infer dimension from GW.
// Outputs:
//  W  output UV coordinates
//  return value  whether the solve was successful
//
template <bool parallelize=false, int idim=-1,
typename LinearSolver, typename DerivedGtM, typename DerivedfixedTo,
typename Derivedbeq, typename DerivedU, typename DerivedP,
typename DerivedLambda, typename DerivedW>
bool
argmin_W(const LinearSolver& GtMG,
         const Eigen::SparseMatrixBase<DerivedGtM>& GtM,
         const Eigen::MatrixBase<DerivedfixedTo>& fixedTo,
         const Eigen::MatrixBase<Derivedbeq>& beq,
         const Eigen::MatrixBase<DerivedU>& U,
         const Eigen::MatrixBase<DerivedP>& P,
         const Eigen::MatrixBase<DerivedLambda>& Lambda,
         Eigen::PlainObjectBase<DerivedW>& W);

}

#endif
