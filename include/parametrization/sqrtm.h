#ifndef PARAMETRIZATION_SQRTM_H
#define PARAMETRIZATION_SQRTM_H

#include <Eigen/Core>

namespace parametrization {

// Methods for computing the matrix square root of symmetric, positive
// definite matrices

// Compute the positive square root of a symmetric positive definite 2x2 matrix.
// Make sure that the input matrix C is symmetric positive definite - the
//  output will be undefined otherwise.
//
// Inputs:
//  C  symmetric positive definite 2x2 matrix
// Outputs:
//  return value  matrix square root of C
//
template <typename DerivedC>
Eigen::Matrix<typename DerivedC::Scalar, 2, 2>
sqrtm_2x2(const Eigen::MatrixBase<DerivedC>& C);

// Compute the positive square root of a symmetric positive definite 3x3 matrix.
// Make sure that the input matrix C is symmetric positive definite - the
//  output will be undefined otherwise.
//
// Inputs:
//  C  symmetric positive definite 3x3 matrix
// Outputs:
//  return value  matrix square root of C
//
template <typename DerivedC>
Eigen::Matrix<typename DerivedC::Scalar, 3, 3>
sqrtm_3x3(const Eigen::MatrixBase<DerivedC>& C);


// Compute the positive square root of a symmetric positive definite nxn matrix.
// Make sure that the input matrix C is symmetric positive definite - the
//  output will be undefined otherwise.
// This is an extra stable version of the algorithm that uses
//  eigendecomposition.
//
// Inputs:
//  C  symmetric positive definite nxn matrix
// Outputs:
//  return value  matrix square root of C
//
template <typename DerivedC>
Eigen::Matrix<typename DerivedC::Scalar,
DerivedC::RowsAtCompileTime, DerivedC::ColsAtCompileTime>
stable_sqrtm(const Eigen::MatrixBase<DerivedC>& C);

}

#endif
