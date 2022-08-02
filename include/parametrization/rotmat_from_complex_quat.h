#ifndef ROTMAT_FROM_COMPLEX_QUAT_H
#define ROTMAT_FROM_COMPLEX_QUAT_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "parametrization_assert.h"


namespace parametrization {

// Given a unit complex number representing a rotation, compute the 2d rotation
//  matrix it corresponds to.
//
// Inputs:
//  re, im  real and complex part of the rotation
// Outputs:
//  return value  2x2 rotation matrix corresponding to U
//
template <typename Scalar>
inline Eigen::Matrix<Scalar, 2, 2>
rotmat_from_complex(const Scalar re, const Scalar im)
{
    parametrization_assert(std::abs(re*re+im*im-1.) <
                           sqrt(std::numeric_limits<Scalar>::epsilon()) &&
                           "not a unit complex number");
    
    Eigen::Matrix<Scalar, 2, 2> mat;
    mat << re, -im, im, re;
    return mat;
}


// Given a vector of stacked rotations in 2d, extract the rotation corresponding
//  to the i-th index
//
// Inputs:
//  U  stacked vector of n rotations as unit complex numbers in the format
//     [reU(0); imU(0); reU(1); imU(1)]
//  i  index of rotation to extract
//  n  number of total rotations (i.e., U.size() / 2)
// Outputs:
//  return value  2x2 rotation matrix corresponding to the i-th rotation in U
//
template <typename DerivedU, typename Int>
inline Eigen::Matrix<typename DerivedU::Scalar, 2, 2>
rotmat_2d_from_index(const Eigen::MatrixBase<DerivedU>& U,
                     const Int i,
                     const Int n)
{
    parametrization_assert(i < n && "Invalid index");
    parametrization_assert(2*n == U.size() && "Invalid number of rotations");
    return rotmat_from_complex(U(2*i), U(2*i+1));
}


// Given a unit quaternion representing a rotation, compute the 3d rotation
//  matrix it corresponds to.
//
// Inputs:
//  w, x, y, z  components of the quaternion
// Outputs:
//  return value  3x3 rotation matrix corresponding to U
//
template <typename Scalar>
inline Eigen::Matrix<Scalar, 3, 3>
rotmat_from_quat(const Scalar w,
                 const Scalar x,
                 const Scalar y,
                 const Scalar z)
{
    parametrization_assert(std::abs(w*w+x*x+y*y+z*z-1.) <
                           sqrt(std::numeric_limits<Scalar>::epsilon()) &&
                           "not a unit quaternion");
    
    return Eigen::Quaternion<Scalar>(w, x, y, z).toRotationMatrix();
}


// Given a vector of stacked rotations in 3d, extract the rotation corresponding
//  to the i-th index
//
// Inputs:
//  U  n rotations, given as unit quaternions in the format
//     [wU(0); xU(0); yU(0); zU(0); wU(1); xU(1); yU(1); zU(1)]
//  i  index of rotation to extract
//  n  number of total rotations (i.e., U.size() / 4)
// Outputs:
//  return value  3x3 rotation matrix corresponding to the i-th rotation in U
//
template <typename DerivedU, typename Int>
inline Eigen::Matrix<typename DerivedU::Scalar, 3, 3>
rotmat_3d_from_index(const Eigen::MatrixBase<DerivedU>& U,
                     const Int i,
                     const Int n)
{
    parametrization_assert(i < n && "Invalid index");
    parametrization_assert(4*n == U.size() && "Invalid number of rotations");
    return rotmat_from_quat(U(4*i), U(4*i+1), U(4*i+2), U(4*i+3));
}


// Given a vector of stacked rotations in 2d or 3d, extract the rotation
//  corresponding to the i-th index
//
// Inputs:
//  U  stacked vector of n rotations as unit complex numbers in the format
//     [reU(0); imU(0); reU(1); imU(1)] (in 2d) or unit quaternions in the
//     format [wU(0); xU(0); yU(0); zU(0); wU(1); xU(1); yU(1); zU(1)] (in 3d)
//  i  index of rotation to extract
//  n  number of total rotations
// Outputs:
//  return value  2x2 or 3x3 rotation matrix corresponding to the i-th rotation
//                in U
//
template <typename DerivedU, typename Int>
inline Eigen::Matrix<typename DerivedU::Scalar, Eigen::Dynamic, Eigen::Dynamic>
rotmat_from_index(const Eigen::MatrixBase<DerivedU>& U,
                  const Int i,
                  const Int n)
{
    if(2*n==U.size()) {
        return rotmat_2d_from_index(U, i, n);
    } else {
        parametrization_assert(4*n==U.size() && "U does not correspond to n.");
        return rotmat_3d_from_index(U, i, n);
    }
}


// Given a 2x2 rotation matrix and a vector of stacked rotations in 2d, extract
//  write the rotation into the i-th index of the stacked matrix
//
// Inputs:
//  Ul  2x2 rotation matrix
//  U  stacked vector of n rotations as unit complex numbers in the format
//     [reU(0); imU(0); reU(1); imU(1)]
//  i  index of rotation
//  n  number of total rotations
//
template <typename DerivedUl, typename DerivedU, typename Int>
inline void
index_into_rotmat_2d(const Eigen::MatrixBase<DerivedUl>& Ul,
                     Eigen::PlainObjectBase<DerivedU>& U,
                     const Int i,
                     const Int n)
{
    using Scalar = typename DerivedUl::Scalar;
    
    parametrization_assert(i < n && "Invalid index");
    parametrization_assert(2*n==U.size() && "U does not correspond to n.");
    
    parametrization_assert(Ul.cols()==2 && Ul.rows()==2 &&
                           "Ul is not a 2x2 matrix.");
    parametrization_assert(std::abs(Ul.determinant() - 1.) <
                           sqrt(std::numeric_limits<Scalar>::epsilon()) &&
                           "U is not a rotation.");
    parametrization_assert
    ((Ul.transpose()*Ul - Eigen::Matrix<Scalar,2,2>::Identity())
     .squaredNorm() < sqrt(std::numeric_limits<Scalar>::epsilon()) &&
     "Ul is not a rotation.");
    
    U(2*i) = Ul(0,0);
    U(2*i+1) = Ul(1,0);
}


// Given a 3x3 rotation matrix and a vector of stacked rotations in 3d, extract
//  write the rotation into the i-th index of the stacked matrix
//
// Inputs:
//  Ul  3x3 rotation matrix
//  U  n rotations, given as unit quaternions in the format
//     [wU(0); xU(0); yU(0); zU(0); wU(1); xU(1); yU(1); zU(1)]
//  i  index of rotation to extract
//  n  number of total rotations (i.e., U.size() / 4)
//
template <typename DerivedUl, typename DerivedU, typename Int>
inline void
index_into_rotmat_3d(const Eigen::MatrixBase<DerivedUl>& Ul,
                     Eigen::PlainObjectBase<DerivedU>& U,
                     const Int i,
                     const Int n)
{
    using Scalar = typename DerivedUl::Scalar;
    
    parametrization_assert(i < n && "Invalid index");
    parametrization_assert(4*n == U.size() && "U does not correspond to n.");
    
    parametrization_assert(Ul.cols()==3 && Ul.rows()==3 &&
                           "Ul is not a 3x3 matrix.");
    parametrization_assert(std::abs(Ul.determinant() - 1) <
                           sqrt(std::numeric_limits<Scalar>::epsilon()) &&
                           "U is not a rotation.");
    parametrization_assert
    ((Ul.transpose()*Ul - Eigen::Matrix<Scalar,3,3>::Identity())
     .squaredNorm() < sqrt(std::numeric_limits<Scalar>::epsilon()) &&
     "Ul is not a rotation.");
    
    Eigen::Quaternion<Scalar> quat(Ul);
    U(4*i) = quat.w();
    U(4*i+1) = quat.x();
    U(4*i+2) = quat.y();
    U(4*i+3) = quat.z();
    
}


}

#endif
