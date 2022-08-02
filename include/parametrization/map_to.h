#ifndef PARAMETRIZATION_MAP_TO_H
#define PARAMETRIZATION_MAP_TO_H

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <functional>

#include "termination_conditions.h"
#include "OptimizationOptions.h"
#include "CallbackFunction.h"
#include "energy.h"


namespace parametrization {

// Compute a map for the provided mesh using Augmented Lagrangian
//  optimization of the energy with three independent variables, U, P, W.
// Surfaces will be mapped into the plane, volumes will be mapped into 3D space.
//
// Inputs:
// Inputs:
//  V, F  input surface mesh as triangle soup. Is not allowed to contain
//        unreferenced vertices or degenerate triangles (where two vertices are
//        closer to each other than numerics allows).
//        The scalar type used should be the same as for W.
//  W  initial UV map. This needs to be initialized to a valid guess with
//     correct dimensions.
//     This should not contain degenerate triangles.
//     The better the guess, the better this method will perform.
//  opts  all kinds of options that can be passed to the optimization routine.
//        If you want to provide your own, create a OptimizationOptions struct and
//        customize it.
//  f  a callback function, see CallbackFunction.h
//  template argument parallelize  if this is set to true, will parallelize
//                                 this operation
// Outputs:
//  W  UV coordinates
//  return value  whether the optimization succeeded in finding a uv map or not
//
template <bool parallelize=false,
EnergyType energy=EnergyType::SymmetricGradient,
typename DerivedV, typename DerivedF,
typename DerivedW, typename OptsScalar=typename DerivedV::Scalar,
typename Tf = CallbackFunction<DerivedW, typename DerivedF::Scalar> >
bool
map_to(const Eigen::MatrixBase<DerivedV>& V,
       const Eigen::MatrixBase<DerivedF>& F,
       Eigen::PlainObjectBase<DerivedW>& W,
       const OptimizationOptions<OptsScalar>& opts
       = OptimizationOptions<typename DerivedW::Scalar>(),
       const Tf& f = emptyCallbackF<DerivedW, typename DerivedF::Scalar>());

// Compute a UV map for the provided mesh using Augmented Lagrangian
//  optimization of the energy with three independent variables, U, P, W.
// Surfaces will be mapped into the plane, volumes will be mapped into 3D space.
// This version admits linear constraints on the UV coordinates W
//  of the form A*W = b. These constraints must make the linear system G*W
//  uniquely solvable.
//
// Inputs:
//  V, F  input surface mesh as triangle soup. Is not allowed to contain
//        unreferenced vertices or degenerate triangles (where two vertices are
//        closer to each other than numerics allows).
//        The scalar type used should be the same as for W.
//  fixed  a vector of indices into the rows of W explicitly fixing vertex
//         positions to W(fixed,:) = fixedTo.
//         The Int type used should be the same as for F.
//  fixedTo  a fixed.rows() x 2 matrix such that W(fixed,:) = fixedTo are the
//           explicitly fixed vectors.
//          The scalar type used should be the same as for W.
//  Aeq  A dense Aeq.rows() x W.cols() matrix such that A*W = b are the linear
//       constraints on W.
//       The scalar type used should be the same as for W.
//       These constraints have to be linearly independent from fixed, fixedTo.
//  beq  A Aeq.rows() x 2 matrix such that A*W = b are the linear constraints
//       on W.
//       The scalar type used should be the same as for W.
//       These constraints have to be linearly independent from fixed, fixedTo.
//  W  initial UV map. This needs to be initialized to a valid guess with
//     correct dimensions.
//     This should not contain degenerate triangles.
//     The better the guess, the better this method will perform.
//  opts  all kinds of options that can be passed to the optimization routine.
//        If you want to provide your own, create a OptimizationOptions struct and
//        customize it.
//  f  a callback function, see CallbackFunction.h
//  template argument parallelize  if this is set to true, will parallelize
//                                 this operation
// Outputs:
//  W  UV coordinates
//  return value  whether the optimization succeeded in finding a uv map or not
//
template <bool parallelize=false,
EnergyType energy=EnergyType::SymmetricGradient,
typename DerivedV, typename DerivedF, typename Derivedfixed,
typename DerivedfixedTo, typename DerivedW, typename ScalarAeq,
typename Derivedbeq, typename OptsScalar=typename DerivedV::Scalar,
typename Tf = CallbackFunction<DerivedW, typename DerivedF::Scalar> >
bool
map_to(const Eigen::MatrixBase<DerivedV>& V,
       const Eigen::MatrixBase<DerivedF>& F,
       const Eigen::MatrixBase<Derivedfixed>& fixed,
       const Eigen::MatrixBase<DerivedfixedTo>& fixedTo,
       const Eigen::SparseMatrix<ScalarAeq>& Aeq,
       const Eigen::MatrixBase<Derivedbeq>& beq,
       Eigen::PlainObjectBase<DerivedW>& W,
       const OptimizationOptions<OptsScalar>& opts
       = OptimizationOptions<typename DerivedW::Scalar>(),
       const Tf& f = emptyCallbackF<DerivedW, typename DerivedF::Scalar>());


// Compute a map for the provided mesh using Augmented Lagrangian
//  optimization of the energy with three independent variables, U, P, W.
// Surfaces will be mapped into the plane, volumes will be mapped into 3D space.
// The dimension is supplied at compile time.
//
// Inputs:
// Inputs:
//  V, F  input surface mesh as triangle soup. Is not allowed to contain
//        unreferenced vertices or degenerate triangles (where two vertices are
//        closer to each other than numerics allows).
//        The scalar type used should be the same as for W.
//  W  initial UV map. This needs to be initialized to a valid guess with
//     correct dimensions.
//     This should not contain degenerate triangles.
//     The better the guess, the better this method will perform.
//  opts  all kinds of options that can be passed to the optimization routine.
//        If you want to provide your own, create a OptimizationOptions struct and
//        customize it.
//  f  a callback function, see CallbackFunction.h
//  template argument dim  choose between 2d and 3d variant of the method.
//  template argument parallelize  if this is set to true, will parallelize
//                                 this operation
// Outputs:
//  W  UV coordinates
//  return value  whether the optimization succeeded in finding a uv map or not
//
template <int dim, bool parallelize=false,
EnergyType energy=EnergyType::SymmetricGradient,
typename DerivedV, typename DerivedF,
typename DerivedW, typename OptsScalar=typename DerivedV::Scalar,
typename Tf = CallbackFunction<DerivedW, typename DerivedF::Scalar> >
bool
map_to_dim(const Eigen::MatrixBase<DerivedV>& V,
           const Eigen::MatrixBase<DerivedF>& F,
           Eigen::PlainObjectBase<DerivedW>& W,
           const OptimizationOptions<OptsScalar>& opts
           = OptimizationOptions<typename DerivedW::Scalar>(),
           const Tf& f = emptyCallbackF<DerivedW, typename DerivedF::Scalar>());

// Compute a UV map for the provided mesh using Augmented Lagrangian
//  optimization of the energy with three independent variables, U, P, W.
// Surfaces will be mapped into the plane, volumes will be mapped into 3D space.
// This version admits linear constraints on the UV coordinates W
//  of the form A*W = b. These constraints must make the linear system G*W
//  uniquely solvable.
// The dimension is supplied at compile time.
//
// Inputs:
//  V, F  input surface mesh as triangle soup. Is not allowed to contain
//        unreferenced vertices or degenerate triangles (where two vertices are
//        closer to each other than numerics allows).
//        The scalar type used should be the same as for W.
//  fixed  a vector of indices into the rows of W explicitly fixing vertex
//         positions to W(fixed,:) = fixedTo.
//         The Int type used should be the same as for F.
//  fixedTo  a fixed.rows() x 2 matrix such that W(fixed,:) = fixedTo are the
//           explicitly fixed vectors.
//          The scalar type used should be the same as for W.
//  Aeq  A dense Aeq.rows() x W.cols() matrix such that A*W = b are the linear
//       constraints on W.
//       The scalar type used should be the same as for W.
//       These constraints have to be linearly independent from fixed, fixedTo.
//  beq  A Aeq.rows() x 2 matrix such that A*W = b are the linear constraints
//       on W.
//       The scalar type used should be the same as for W.
//       These constraints have to be linearly independent from fixed, fixedTo.
//  W  initial UV map. This needs to be initialized to a valid guess with
//     correct dimensions.
//     This should not contain degenerate triangles.
//     The better the guess, the better this method will perform.
//  opts  all kinds of options that can be passed to the optimization routine.
//        If you want to provide your own, create a OptimizationOptions struct and
//        customize it.
//  f  a callback function, see CallbackFunction.h
//  template argument dim  choose between 2d and 3d variant of the method.
//  template argument parallelize  if this is set to true, will parallelize
//                                 this operation
// Outputs:
//  W  UV coordinates
//  return value  whether the optimization succeeded in finding a uv map or not
//
template <int dim, bool parallelize=false,
EnergyType energy=EnergyType::SymmetricGradient,
typename DerivedV, typename DerivedF, typename Derivedfixed,
typename DerivedfixedTo, typename DerivedW, typename ScalarAeq,
typename Derivedbeq, typename OptsScalar=typename DerivedV::Scalar,
typename Tf = CallbackFunction<DerivedW, typename DerivedF::Scalar> >
bool
map_to_dim(const Eigen::MatrixBase<DerivedV>& V,
           const Eigen::MatrixBase<DerivedF>& F,
           const Eigen::MatrixBase<Derivedfixed>& fixed,
           const Eigen::MatrixBase<DerivedfixedTo>& fixedTo,
           const Eigen::SparseMatrix<ScalarAeq>& Aeq,
           const Eigen::MatrixBase<Derivedbeq>& beq,
           Eigen::PlainObjectBase<DerivedW>& W,
           const OptimizationOptions<OptsScalar>& opts
           = OptimizationOptions<typename DerivedW::Scalar>(),
           const Tf& f = emptyCallbackF<DerivedW, typename DerivedF::Scalar>());

}

#endif
