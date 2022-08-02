#ifndef PARAMETRIZATION_POLAR_DECOMPOSITION_H
#define PARAMETRIZATION_POLAR_DECOMPOSITION_H

#include <Eigen/Core>

namespace parametrization {

// Compute the polar decomposition of n 2x2 or 3x3 matrices into rotation
//  matrices and symmetric semipositive definite matrices. If the determinant of
//  the input matrix is negative, such a decomposition is not possible, and this
//  function will return an arbitrary U and small P.
// If one of the eigenvalues of the input matrix is 0, this function will add
//  the identity to it until we reach a certain tolerance.
//
// Input:
//  J  n matrices in the format
//      [J1(0,0) J1(0,1); J2(0,0) J2(0,1); J1(1,0) J1(1,1); J2(1,0) J2(1,1)]
//      (for 2x2), or
//      [J1(0,0) J1(0,1) J1(0,2); J2(0,0) J2(0,1) J2(0,2); ...
//      J1(1,0) J1(1,1) J1(1,2); J2(1,0) J2(1,1) J2(1,2); ...
//      J1(2,0) J1(2,1) J1(2,2); J2(2,0) J2(2,1) J2(2,2)] (for 3x3)
//  tol  threshold to determine whether an eigenvalue is zero.
// Output:
//  U  stacked vector of n rotations as unit complex numbers in the format
//     [reU(0); imU(0); reU(1); imU(1)] (in 2d) or unit quaternions in the
//     format [wU(0); xU(0); yU(0); zU(0); wU(1); xU(1); yU(1); zU(1)] (in 3d)
//  P  n symmetric matrices, provided in the format
//      [P1(0,0); P2(0,0); P1(0,1); P2(0,1); P1(1,1); P2(1,1)] (in 2d) or
//      [P1(0,0); P2(0,0); P1(0,1); P2(0,1); P1(0,2); P2(0,2); ...
//      P1(1,1); P2(1,1); P1(1,2); P2(1,2); P1(2,2); P2(2,2)] (in 3d).
//
template <typename DerivedJ, typename DerivedU, typename DerivedP,
typename Scalartol = typename DerivedJ::Scalar>
void
polar_decomposition(const Eigen::MatrixBase<DerivedJ>& J,
                    Eigen::PlainObjectBase<DerivedU>& U,
                    Eigen::PlainObjectBase<DerivedP>& P,
                    const Scalartol tol=
                    sqrt(std::numeric_limits<typename DerivedJ::Scalar>::epsilon()));

// Compute the polar decomposition of a 2x2 matrix into a rotation matrix and
//  a symmetrix semipositive definite matrix. If the determinant of the input
//  matrix is negative, such a decomposition is not possible, and this
//  function will return an arbitrary U and small P.
// If one of the eigenvalues of the output matrix P is 0, this function will add
//  the identity to it until we reach a certain tolerance.
//
// Input:
//  J  a 2x2 matrix
//  tol  threshold to determine whether an eigenvalue is zero.
// Output:
//  U  a 2d rotation matrix
//  P  a 2d symmetric semipositive definite matrix
//
template <typename DerivedJ, typename DerivedU, typename DerivedP,
typename Scalartol = typename DerivedJ::Scalar>
void
polar_decomposition_2x2(const Eigen::MatrixBase<DerivedJ>& J,
                        Eigen::PlainObjectBase<DerivedU>& U,
                        Eigen::PlainObjectBase<DerivedP>& P,
                        const Scalartol tol=
                        sqrt(std::numeric_limits<typename DerivedJ::Scalar>::epsilon()));


// Compute the polar decomposition of a 3x3 matrix into a rotation matrix and
//  a symmetrix semipositive definite matrix. If the determinant of the input
//  matrix is negative, such a decomposition is not possible, and this function
//  will flip one of the eigenvalues of the input matrix.
// If one of the eigenvalues of the output matrix P is 0, this function will add
//  the identity to it until we reach a certain tolerance.
//
// Input:
//  J  a 3x3 matrix
//  tol  threshold to determine whether an eigenvalue is zero.
// Output:
//  U  a 3d rotation matrix
//  P  a 3d symmetric semipositive definite matrix
//
template <typename DerivedJ, typename DerivedU, typename DerivedP,
typename Scalartol = typename DerivedJ::Scalar>
void
polar_decomposition_3x3(const Eigen::MatrixBase<DerivedJ>& J,
                        Eigen::PlainObjectBase<DerivedU>& U,
                        Eigen::PlainObjectBase<DerivedP>& P,
                        const Scalartol tol=
                        sqrt(std::numeric_limits<typename DerivedJ::Scalar>::epsilon()));

}

#endif
