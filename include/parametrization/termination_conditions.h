#ifndef PARAMETRIZATION_TERMINATION_CONDITIONS_H
#define PARAMETRIZATION_TERMINATION_CONDITIONS_H

#include <Eigen/Core>
#include <Eigen/Sparse>

#include "energy.h"


namespace parametrization {

// All available termination conditions for the Lagrangian optimization
//
enum class TerminationCondition {
    PrimalDualError, //primal and dual error are below tolerances
    Noflips, //no triangles are inverted
    PrimalDualErrorNoflips, //PrimalDualError and no flipped triangles
    Progress, //Vertex progress, scaled by number of vertices, less than
    ProgressNoflips, //Progress and no flipped triangles
    TargetEnergy, //Optimize until a target energy is reached
    TargetEnergyNoflips, //Target energy and no flips
    NumberOfIterations, //Terminate after a fixed number of iterations
    NumberOfIterationsNoflips, //number of iterations and no flips
    None //Never terminate. Ignore maxIter. Combine with custom callback to exit
};

//
// Primal / dual error termination condition.
// Returns true if the condition is fulfilled and optimization should be
//  terminated, returns false otherwise.
//
// Inputs:
//  U  n rotation matrices, given as n rotation angles (in 2d)
//  P  n symmetric matrices, provided in the format
//      [P1(0,0); P1(0,1); P1(1,1); P2(0,0); P2(0,1); P2(1,1)] (in 2d) or
//      [P1(0,0); P1(0,1); P1(0,2); P1(1,1); P1(1,2); P1(2,2); ...
//       P2(0,0); P2(0,1); P2(0,2); P2(1,1); P2(1,2); P2(2,2)] (in 3d)
//     If P is not strictly positive definite, behavior is undefined.
//  GW  n matrices in the format
//      [GW1(0,0) GW1(0,1); GW2(0,0) GW2(0,1); ...
//       GW1(1,0) GW1(1,1); GW2(1,0) GW2(1,1)] (or analogous in 3d)
//  GtLambda  the uv_to_jacobian matrix, transposed, multiplied with the
//            Lagrangian variable Lambda
//  mu  n penalty terms
//  eP  squared primal error
//  eD  squared dual error
//  pAbsTol  primal absolute tolerance
//  pRelTol  primal relative tolerance
//  dAbsTol  dual absolute tolerance
//  dRelTol  dual relative tolerance
//  template argument idim  choose between 2d and 3d variant of the method.
//                          If set to -1, will infer dimension from GW.
//
// Outputs:
//  return value  whether to terminate the optimization or not
//

template <int idim=-1,
typename DerivedU, typename DerivedP, typename DerivedGW,
typename DerivedGtLambda, typename Derivedmu, typename ScalareP,
typename ScalareD, typename ScalarpAbsTol, typename ScalarpRelTol,
typename ScalardAbsTol, typename ScalardRelTol>
bool
primal_dual_termination_condition
 (const Eigen::MatrixBase<DerivedU>& U,
 const Eigen::MatrixBase<DerivedP>& P,
 const Eigen::MatrixBase<DerivedGW>& GW,
 const Eigen::MatrixBase<DerivedGtLambda>& GtLambda,
 const Eigen::MatrixBase<Derivedmu>& mu,
 const ScalareP eP,
 const ScalareD eD,
 const ScalarpAbsTol pAbsTol,
 const ScalarpRelTol pRelTol,
 const ScalardAbsTol dAbsTol,
 const ScalardRelTol dRelTol);


//
// No flipped triangles termination condition (based on Jacobian determinants
//  GW).
// Returns true if the condition is fulfilled and optimization should be
//  terminated, returns false otherwise.
//
// Inputs:
//  GW  n matrices in the format
//      [GW1(0,0) GW1(0,1); GW2(0,0) GW2(0,1); ...
//       GW1(1,0) GW1(1,1); GW2(1,0) GW2(1,1)] (or analogous in 3d)
//  template argument idim  choose between 2d and 3d variant of the method.
//                          If set to -1, will infer dimension from GW.
//
// Outputs:
//  return value  whether to terminate the optimization or not
//

template <int idim=-1,
typename DerivedGW>
bool
noflips_termination_condition
 (const Eigen::MatrixBase<DerivedGW>& GW);


//
// Primal / dual error termination condition with no flipped triangles
//  guarantee (based on Jacobian determinants GW).
// Returns true if the condition is fulfilled and optimization should be
//  terminated, returns false otherwise.
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
//  GtLambda  the uv_to_jacobian matrix, transposed, multiplied with the
//            Lagrangian variable Lambda
//  mu  n penalty terms
//  eP  squared primal error
//  eD  squared dual error
//  pAbsTol  primal absolute tolerance
//  pRelTol  primal relative tolerance
//  dAbsTol  dual absolute tolerance
//  dRelTol  dual relative tolerance
//  template argument idim  choose between 2d and 3d variant of the method.
//                          If set to -1, will infer dimension from GW.
//
// Outputs:
//  return value  whether to terminate the optimization or not
//

template <int idim=-1,
typename DerivedU, typename DerivedP, typename DerivedGW,
typename DerivedGtLambda, typename Derivedmu,
typename ScalareP, typename ScalareD, typename ScalarpAbsTol,
typename ScalarpRelTol, typename ScalardAbsTol, typename ScalardRelTol>
bool
primal_dual_noflips_termination_condition
 (const Eigen::MatrixBase<DerivedU>& U,
 const Eigen::MatrixBase<DerivedP>& P,
 const Eigen::MatrixBase<DerivedGW>& GW,
 const Eigen::MatrixBase<DerivedGtLambda>& GtLambda,
 const Eigen::MatrixBase<Derivedmu>& mu,
 const ScalareP eP,
 const ScalareD eD,
 const ScalarpAbsTol pAbsTol,
 const ScalarpRelTol pRelTol,
 const ScalardAbsTol dAbsTol,
 const ScalardRelTol dRelTol);


//
// Progress termination condition.
// Returns true if the progress in vertex positions W (scaler by number of
//  vertices) is less than a specified tolerance.
//
// Inputs:
//  W0  vertex positions at the last iteration
//  W  vertex positions at the current iteration
//  tol  tolerance below which we terminate
//  template argument idim  choose between 2d and 3d variant of the method.
//                          If set to -1, will infer dimension from GW.
//
// Outputs:
//  return value  whether to terminate the optimization or not
//

template <int idim=-1,
typename DerivedW0, typename DerivedW, typename Scalartol>
bool
progress_termination_condition
 (const Eigen::MatrixBase<DerivedW0>& W0,
 const Eigen::MatrixBase<DerivedW>& W,
 const Scalartol tol);


//
// Progress termination condition with no flipped triangles guarantee (based on
//  Jacobian determinants GW).
// Returns true if the progress in vertex positions W (scaler by number of
//  vertices) is less than a specified tolerance, and there are no inverted
//  triangles.
//
// Inputs:
//  GW  n matrices in the format
//      [GW1(0,0) GW1(0,1); GW2(0,0) GW2(0,1); ...
//       GW1(1,0) GW1(1,1); GW2(1,0) GW2(1,1)] (or analogous in 3d)
//  W0  vertex positions at the last iteration
//  W  vertex positions at the current iteration
//  tol  tolerance below which we terminate
//  template argument idim  choose between 2d and 3d variant of the method.
//                          If set to -1, will infer dimension from GW.
//
// Outputs:
//  return value  whether to terminate the optimization or not
//

template <int idim=-1,
typename DerivedGW, typename DerivedW0, typename DerivedW, typename Scalartol>
bool
progress_noflips_termination_condition
 (const Eigen::MatrixBase<DerivedGW>& GW,
  const Eigen::MatrixBase<DerivedW0>& W0,
  const Eigen::MatrixBase<DerivedW>& W,
  const Scalartol tol);


//
// Target energy termination condition.
// Returns true if the condition is fulfilled and optimization should be
//  terminated, returns false otherwise.
//
// Inputs:
//  GW  n matrices in the format
//      [GW1(0,0) GW1(0,1); GW2(0,0) GW2(0,1); ...
//       GW1(1,0) GW1(1,1); GW2(1,0) GW2(1,1)] (or analogous in 3d)
//  w  n triangle areas / tet volumes
//  tol  threshold for the target energy
//  template argument energy  which energy to compute
//  template argument idim  choose between 2d and 3d variant of the method.
//                          If set to -1, will infer dimension from GW.
//
// Outputs:
//  return value  whether to terminate the optimization or not
//

template <EnergyType energy=EnergyType::SymmetricGradient, int idim=-1,
typename DerivedGW, typename Derivedw, typename Scalartol>
bool
target_energy_termination_condition
 (const Eigen::MatrixBase<DerivedGW>& GW,
  const Eigen::MatrixBase<Derivedw>& w,
  const Scalartol tol);


//
// Target energy termination condition with noflip.
// Returns true if the condition is fulfilled with no flipped triangles and
//  optimization should be terminated, returns false otherwise.
//
// Inputs:
//  GW  n matrices in the format
//      [GW1(0,0) GW1(0,1); GW2(0,0) GW2(0,1); ...
//       GW1(1,0) GW1(1,1); GW2(1,0) GW2(1,1)] (or analogous in 3d)
//  w  n triangle areas / tet volumes
//  tol  threshold for the target energy
//  template argument energy  which energy to compute
//  template argument idim  choose between 2d and 3d variant of the method.
//                          If set to -1, will infer dimension from GW.
//
// Outputs:
//  return value  whether to terminate the optimization or not
//

template <EnergyType energy=EnergyType::SymmetricGradient, int idim=-1,
typename DerivedGW, typename Derivedw, typename Scalartol>
bool
target_energy_noflips_termination_condition
(const Eigen::MatrixBase<DerivedGW>& GW,
 const Eigen::MatrixBase<Derivedw>& w,
 const Scalartol tol);


}

#endif
