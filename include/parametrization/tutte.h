#ifndef PARAMETRIZATION_TUTTE_H
#define PARAMETRIZATION_TUTTE_H

#include <Eigen/Core>
#include <Eigen/Sparse>


namespace parametrization {

// Compute the tutte embedding of a 2d mesh into a circle.
//
// Input:
//  V, F  input surface mesh as triangle soup. Is not allowed to contain
//        unreferenced vertices or degenerate triangles (where two vertices are
//        closer to each other than numerics allows).
//  template parameter uniformScaling  if set to true, will weigh all edges of
//                                     the mesh equally, and not by their
//                                     length.
// Output:
//  return value  whether the linear solve was successful
//  W  UV coordinates
//
template <bool uniformScaling=false,
typename DerivedV, typename DerivedF, typename DerivedW>
bool
tutte(const Eigen::MatrixBase<DerivedV>& V,
      const Eigen::MatrixBase<DerivedF>& F,
      Eigen::PlainObjectBase<DerivedW>& W);


// Compute the tutte embedding of a 2d mesh given custom constraints.
// The constraints have to be tight enough to guarantee a unique solution.
//
// Input:
//  V, F  input surface mesh as triangle soup. Is not allowed to contain
//        unreferenced vertices or degenerate triangles (where two vertices are
//        closer to each other than numerics allows).
//  fixed  a vector of indices into the rows of W explicitly fixing vertex
//         positions to W(fixed,:) = fixedTo.
//  fixedTo  a fixed.rows() x 2 matrix such that W(fixed,:) = fixedTo are the
//           explicitly fixed vectors.
//  Aeq  A dense Aeq.rows() x W.cols() matrix such that A*W = b are the linear
//       constraints on W.
//       These constraints have to be linearly independent from fixed, fixedTo.
//  beq  A Aeq.rows() x 2 matrix such that A*W = b are the linear constraints
//       on W.
//  template parameter uniformScaling  if set to true, will weigh all edges of
//                                     the mesh equally, and not by their
//                                     length.
// Output:
//  return value  whether the linear solve was successful
//  W  UV coordinates
//
template <bool uniformScaling=false,
typename DerivedV, typename DerivedF, typename DerivedW,
typename Derivedfixed, typename DerivedfixedTo, typename ScalarAeq,
typename Derivedbeq>
bool
tutte(const Eigen::MatrixBase<DerivedV>& V,
      const Eigen::MatrixBase<DerivedF>& F,
      const Eigen::MatrixBase<Derivedfixed>& fixed,
      const Eigen::MatrixBase<DerivedfixedTo>& fixedTo,
      const Eigen::SparseMatrix<ScalarAeq>& Aeq,
      const Eigen::MatrixBase<Derivedbeq>& beq,
      Eigen::PlainObjectBase<DerivedW>& W);

}

#endif
