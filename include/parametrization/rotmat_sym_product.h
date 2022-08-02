#ifndef PARAMETRIZATION_ROTMAT_SYM_PRODUCT_H
#define PARAMETRIZATION_ROTMAT_SYM_PRODUCT_H

#include <Eigen/Core>

namespace parametrization {

// Compute the product of n rotation matrices and n symmetric matrices in 2d
//
// Inputs:
//  U  stacked vector of n rotations as unit complex numbers in the format
//     [reU(0); imU(0); reU(1); imU(1)]
//  P  n symmetric matrices, provided in the format
//      [P1(0,0); P2(0,0); P1(0,1); P2(0,1); P1(1,1); P2(1,1)].
// Outputs:
//  J  n matrices in the format
//      [J1(0,0) J1(0,1); J2(0,0) J2(0,1); J1(1,0) J1(1,1); J2(1,0) J2(1,1)].
//
template <typename DerivedU, typename DerivedP, typename DerivedJ>
void
rotmat_sym_product_2d(const Eigen::MatrixBase<DerivedU>& U,
                      const Eigen::MatrixBase<DerivedP>& P,
                      Eigen::PlainObjectBase<DerivedJ>& J);

// Version that returns J instead of taking it as an input parameter
template <typename DerivedU, typename DerivedP>
Eigen::Matrix<typename DerivedU::Scalar, Eigen::Dynamic, 2>
rotmat_sym_product_2d(const Eigen::MatrixBase<DerivedU>& U,
                      const Eigen::MatrixBase<DerivedP>& P);


// Compute the product of n rotation matrices and n symmetric matrices in 3d
//
// Inputs:
//  U  n rotations, given as unit quaternions in the format
//     [wU(0); xU(0); yU(0); zU(0); wU(1); xU(1); yU(1); zU(1)]
//  P  n symmetric matrices, provided in the format
//      [P1(0,0); P2(0,0); P1(0,1); P2(0,1); P1(0,2); P2(0,2); ...
//      P1(1,1); P2(1,1); P1(1,2); P2(1,2); P1(2,2); P2(2,2)].
// Outputs:
//  J  n matrices in the format
//      [J1(0,0) J1(0,1) J1(0,2); J2(0,0) J2(0,1) J2(0,2); ...
//      J1(1,0) J1(1,1) J1(1,2); J2(1,0) J2(1,1) J2(1,2); ...
//      J1(2,0) J1(2,1) J1(2,2); J2(2,0) J2(2,1) J2(2,2)].
//
template <typename DerivedU, typename DerivedP, typename DerivedJ>
void
rotmat_sym_product_3d(const Eigen::MatrixBase<DerivedU>& U,
                      const Eigen::MatrixBase<DerivedP>& P,
                      Eigen::PlainObjectBase<DerivedJ>& J);

// Version that returns J instead of taking it as an input parameter
template <typename DerivedU, typename DerivedP>
Eigen::Matrix<typename DerivedU::Scalar, Eigen::Dynamic, 3>
rotmat_sym_product_3d(const Eigen::MatrixBase<DerivedU>& U,
                      const Eigen::MatrixBase<DerivedP>& P);


}

#endif
