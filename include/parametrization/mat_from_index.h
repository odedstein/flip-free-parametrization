#ifndef PARAMETRIZATION_MAT_FROM_INDEX_H
#define PARAMETRIZATION_MAT_FROM_INDEX_H

#include <Eigen/Core>

#include "parametrization_assert.h"

namespace parametrization {

template <typename DerivedJ, typename Int>
inline Eigen::Matrix<typename DerivedJ::Scalar, 2, 2>
mat_from_index_2x2(const Eigen::MatrixBase<DerivedJ>& J,
                  const Int i,
                  const Int n)
{
    parametrization_assert(i < n && "Invalid index");
    parametrization_assert(2*n == J.rows() && "Invalid number of matrices");
    parametrization_assert(J.cols()==2 && "J is not a 2x2 stacked matrix.");
    
    Eigen::Matrix<typename DerivedJ::Scalar, 2, 2> Jr;
    Jr << J(i,0), J(i,1), J(i+n,0), J(i+n,1);
    return Jr;
}


template <typename DerivedJ, typename Int>
inline Eigen::Matrix<typename DerivedJ::Scalar, 3, 3>
mat_from_index_3x3(const Eigen::MatrixBase<DerivedJ>& J,
                  const Int i,
                  const Int n)
{
    parametrization_assert(i < n && "Invalid index");
    parametrization_assert(3*n == J.rows() && "Invalid number of matrices");
    parametrization_assert(J.cols()==3 && "J is not a 2x2 stacked matrix.");
    
    Eigen::Matrix<typename DerivedJ::Scalar, 3, 3> Jr;
    Jr << J(i,0), J(i,1), J(i,2), J(i+n,0), J(i+n,1), J(i+n,2),
    J(i+2*n,0), J(i+2*n,1), J(i+2*n,2);
    return Jr;
}


// Given n 2x2 or 3x3 matrices in stacked format, extract the i-th matrix.
//
// Inputs:
//  J  n matrices in the format
//      [J1(0,0) J1(0,1); J2(0,0) J2(0,1); J1(1,0) J1(1,1); J2(1,0) J2(1,1)]
//      (for 2x2), or
//      [J1(0,0) J1(0,1) J1(0,2); J2(0,0) J2(0,1) J2(0,2); ...
//      J1(1,0) J1(1,1) J1(1,2); J2(1,0) J2(1,1) J2(1,2); ...
//      J1(2,0) J1(2,1) J1(2,2); J2(2,0) J2(2,1) J2(2,2)] (for 3x3)
//  i  index of matrix to extract
//  n  number of total matrices (i.e., J.rows() / 2 for 2D or
//      J.rows() / 3 for 3D)
// Outputs:
//  return value  2x2 symmetrix matrix corresponding to the i-th matrix in P
//
template <typename DerivedJ, typename Int>
inline Eigen::Matrix<typename DerivedJ::Scalar, Eigen::Dynamic, Eigen::Dynamic>
mat_from_index(const Eigen::MatrixBase<DerivedJ>& J,
                  const Int i,
                  const Int n)
{
    if(2*n == J.rows()) {
        return mat_from_index_2x2(J, i, n);
    } else {
        parametrization_assert(3*n == J.rows() && "Invalid number of matrices");
        return mat_from_index_3x3(J, i, n);
    }
}


template <typename DerivedJl, typename DerivedJ, typename Int>
inline void
index_into_mat_2x2(const Eigen::MatrixBase<DerivedJl>& Jl,
                   Eigen::PlainObjectBase<DerivedJ>& J,
                   const Int i,
                   const Int n)
{
    parametrization_assert(i < n && "Invalid index");
    parametrization_assert(Jl.rows()==2 && Jl.cols()==2 &&
                           "Jl is not a 2x2 matrix.");
    parametrization_assert(2*n == J.rows() && "Invalid number of matrices");
    parametrization_assert(J.cols()==2 && "J is not a 2x2 stacked matrix.");
    
    for(int j=0; j<2; ++j) {
        for(int k=0; k<2; ++k) {
            J(i+j*n,k) = Jl(j,k);
        }
    }
}


template <typename DerivedJl, typename DerivedJ, typename Int>
inline void
index_into_mat_3x3(const Eigen::MatrixBase<DerivedJl>& Jl,
                   Eigen::PlainObjectBase<DerivedJ>& J,
                   const Int i,
                   const Int n)
{
    parametrization_assert(i < n && "Invalid index");
    parametrization_assert(Jl.rows()==3 && Jl.cols()==3 &&
                           "Jl is not a 3x3 matrix.");
    parametrization_assert(3*n == J.rows() && "Invalid number of matrices");
    parametrization_assert(J.cols()==3 && "J is not a 3x3 stacked matrix.");
    
    for(int j=0; j<3; ++j) {
        for(int k=0; k<3; ++k) {
            J(i+j*n,k) = Jl(j,k);
        }
    }
}


// Given a 2x2 or 3x3 matrix and an index, write it into a stacked
//  representation of n matrices.
//
// Inputs:
//  Jl  matrix to be written into J
//  J  n matrices in the format
//      [J1(0,0) J1(0,1); J2(0,0) J2(0,1); J1(1,0) J1(1,1); J2(1,0) J2(1,1)]
//      (for 2x2), or
//      [J1(0,0) J1(0,1) J1(0,2); J2(0,0) J2(0,1) J2(0,2); ...
//      J1(1,0) J1(1,1) J1(1,2); J2(1,0) J2(1,1) J2(1,2); ...
//      J1(2,0) J1(2,1) J1(2,2); J2(2,0) J2(2,1) J2(2,2)] (for 3x3)
//  i  index of matrix to write into
//  n  number of total matrices (i.e., J.rows() / 2 for 2D or
//      J.rows() / 3 for 3D)
// Outputs:
//  J  stacked matrix with Jl written into it
//
template <typename DerivedJl, typename DerivedJ, typename Int>
inline void
index_into_mat(const Eigen::MatrixBase<DerivedJl>& Jl,
                   Eigen::PlainObjectBase<DerivedJ>& J,
                   Int i,
                   Int n)
{
    if(2*n == J.rows()) {
        return index_into_mat_2x2(Jl, J, i, n);
    } else {
        parametrization_assert(3*n == J.rows() && "Invalid number of matrices");
        return index_into_mat_3x3(Jl, J, i, n);
    }
}


template <typename DerivedJl, typename DerivedJ, typename Int>
inline void
add_into_mat_2x2(const Eigen::MatrixBase<DerivedJl>& Jl,
                   Eigen::PlainObjectBase<DerivedJ>& J,
                   const Int i,
                   const Int n)
{
    parametrization_assert(i < n && "Invalid index");
    parametrization_assert(Jl.rows()==2 && Jl.cols()==2 &&
                           "Jl is not a 2x2 matrix.");
    parametrization_assert(2*n == J.rows() && "Invalid number of matrices");
    parametrization_assert(J.cols()==2 && "J is not a 2x2 stacked matrix.");
    
    for(int j=0; j<2; ++j) {
        for(int k=0; k<2; ++k) {
            J(i+j*n,k) += Jl(j,k);
        }
    }
}


template <typename DerivedJl, typename DerivedJ, typename Int>
inline void
add_into_mat_3x3(const Eigen::MatrixBase<DerivedJl>& Jl,
                   Eigen::PlainObjectBase<DerivedJ>& J,
                   const Int i,
                   const Int n)
{
    parametrization_assert(i < n && "Invalid index");
    parametrization_assert(Jl.rows()==3 && Jl.cols()==3 &&
                           "Jl is not a 3x3 matrix.");
    parametrization_assert(3*n == J.rows() && "Invalid number of matrices");
    parametrization_assert(J.cols()==3 && "J is not a 3x3 stacked matrix.");
    
    for(int j=0; j<3; ++j) {
        for(int k=0; k<3; ++k) {
            J(i+j*n,k) += Jl(j,k);
        }
    }
}


// Given a 2x2 or 3x3 matrix and an index, add it into a stacked
//  representation of n matrices.
//
// Inputs:
//  Jl  matrix to be added onto J
//  J  n matrices in the format
//      [J1(0,0) J1(0,1); J2(0,0) J2(0,1); J1(1,0) J1(1,1); J2(1,0) J2(1,1)]
//      (for 2x2), or
//      [J1(0,0) J1(0,1) J1(0,2); J2(0,0) J2(0,1) J2(0,2); ...
//      J1(1,0) J1(1,1) J1(1,2); J2(1,0) J2(1,1) J2(1,2); ...
//      J1(2,0) J1(2,1) J1(2,2); J2(2,0) J2(2,1) J2(2,2)] (for 3x3)
//  i  index of matrix to add to
//  n  number of total matrices (i.e., J.rows() / 2 for 2D or
//      J.rows() / 3 for 3D)
// Outputs:
//  J  stacked matrix with Jl written into it
//
template <typename DerivedJl, typename DerivedJ, typename Int>
inline void
add_into_mat(const Eigen::MatrixBase<DerivedJl>& Jl,
                   Eigen::PlainObjectBase<DerivedJ>& J,
                   Int i,
                   Int n)
{
    if(2*n == J.rows()) {
        return add_into_mat_2x2(Jl, J, i, n);
    } else {
        parametrization_assert(3*n == J.rows() && "Invalid number of matrices");
        return add_into_mat_3x3(Jl, J, i, n);
    }
}


template <typename Scalarx, typename DerivedJ, typename Int>
inline void
multiply_into_mat_2x2(const Scalarx x,
                   Eigen::PlainObjectBase<DerivedJ>& J,
                   const Int i,
                   const Int n)
{
    parametrization_assert(i < n && "Invalid index");
    parametrization_assert(2*n == J.rows() && "Invalid number of matrices");
    parametrization_assert(J.cols()==2 && "J is not a 2x2 stacked matrix.");
    
    for(int j=0; j<2; ++j) {
        for(int k=0; k<2; ++k) {
            J(i+j*n,k) *= x;
        }
    }
}


template <typename Scalarx, typename DerivedJ, typename Int>
inline void
multiply_into_mat_3x3(const Scalarx x,
                   Eigen::PlainObjectBase<DerivedJ>& J,
                   const Int i,
                   const Int n)
{
    parametrization_assert(i < n && "Invalid index");
    parametrization_assert(3*n == J.rows() && "Invalid number of matrices");
    parametrization_assert(J.cols()==3 && "J is not a 3x3 stacked matrix.");
    
    for(int j=0; j<3; ++j) {
        for(int k=0; k<3; ++k) {
            J(i+j*n,k) *= x;
        }
    }
}


template <typename Scalarx, typename DerivedJ, typename Int>
inline void
divide_into_mat_2x2(const Scalarx x,
                   Eigen::PlainObjectBase<DerivedJ>& J,
                   const Int i,
                   const Int n)
{
    parametrization_assert(i < n && "Invalid index");
    parametrization_assert(2*n == J.rows() && "Invalid number of matrices");
    parametrization_assert(J.cols()==2 && "J is not a 2x2 stacked matrix.");
    
    for(int j=0; j<2; ++j) {
        for(int k=0; k<2; ++k) {
            J(i+j*n,k) /= x;
        }
    }
}


template <typename Scalarx, typename DerivedJ, typename Int>
inline void
divide_into_mat_3x3(const Scalarx x,
                   Eigen::PlainObjectBase<DerivedJ>& J,
                   const Int i,
                   const Int n)
{
    parametrization_assert(i < n && "Invalid index");
    parametrization_assert(3*n == J.rows() && "Invalid number of matrices");
    parametrization_assert(J.cols()==3 && "J is not a 3x3 stacked matrix.");
    
    for(int j=0; j<3; ++j) {
        for(int k=0; k<3; ++k) {
            J(i+j*n,k) /= x;
        }
    }
}


}

#endif
