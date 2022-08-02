#ifndef PARAMETRIZATION_SYM_FROM_TRIU_H
#define PARAMETRIZATION_SYM_FROM_TRIU_H

#include <Eigen/Dense>

#include "parametrization_assert.h"


namespace parametrization {

// Given the upper triangular part of a 2x2 symmetric matrix, compute the
//  matrix.
//
// Inputs:
//  P00, P01, P11  upper triangular part of the symmetric matrix
// Outputs:
//  return value  2x2 symmetric matrix
//
template <typename Scalar>
inline Eigen::Matrix<Scalar, 2, 2>
sym_from_triu(const Scalar P00,
                   const Scalar P01,
                   const Scalar P11)
{
    Eigen::Matrix<Scalar, 2, 2> P;
    P << P00, P01, P01, P11;
    return P;
}


// Given a vector of stacked upper triangular parts of symmetrix 2x2 matrices,
//  extract the matrix corresponding to the i-th index
//
// Inputs:
//  P  stacked vector of n symmetric matrices' upper triangular parts
//      provided in the format
//      [P1(0,0); P1(0,1); P1(1,1); P2(0,0); P2(0,1); P2(1,1)]
//  i  index of matrix to extract
//  n  number of total matrices (i.e., P.size() / 3)
// Outputs:
//  return value  2x2 symmetrix matrix corresponding to the i-th matrix in P
//
template <typename DerivedP, typename Int>
inline Eigen::Matrix<typename DerivedP::Scalar, 2, 2>
sym_2d_from_index(const Eigen::MatrixBase<DerivedP>& P,
                  const Int i,
                  const Int n)
{
    parametrization_assert(i < n && "Invalid index");
    parametrization_assert(3*n == P.size() && "Invalid number of matrices");
    return sym_from_triu(P(3*i), P(3*i+1), P(3*i+2));
}


// Given the upper triangular part of a 3x3 symmetric matrix, compute the
//  matrix.
//
// Inputs:
//  P00, P01, P02, P11, P12, P22  upper triangular part of the symmetric matrix
// Outputs:
//  return value  3x3 symmetric matrix
//
template <typename Scalar>
inline Eigen::Matrix<Scalar, 3, 3>
sym_from_triu(const Scalar P00,
                   const Scalar P01,
                   const Scalar P02,
                   const Scalar P11,
                   const Scalar P12,
                   const Scalar P22)
{
    Eigen::Matrix<Scalar, 3, 3> P;
    P << P00, P01, P02, P01, P11, P12, P02, P12, P22;
    return P;
}


// Given a vector of stacked upper triangular parts of symmetrix 3x3 matrices,
//  extract the matrix corresponding to the i-th index
//
// Inputs:
//  P  stacked vector of n symmetric matrices' upper triangular parts
//      provided in the format
//      [P1(0,0); P1(0,1); P1(0,2); P1(1,1); P1(1,2); P1(2,2); ...
//       P2(0,0); P2(0,1); P2(0,2); P2(1,1); P2(1,2); P2(2,2)] (in 3d)
//  i  index of matrix to extract
//  n  number of total matrices (i.e., P.size() / 2 or P.size() / 6)
// Outputs:
//  return value  3x3 symmetrix matrix corresponding to the i-th matrix in P
//
template <typename DerivedP, typename Int>
inline Eigen::Matrix<typename DerivedP::Scalar, 3, 3>
sym_3d_from_index(const Eigen::MatrixBase<DerivedP>& P,
                     const Int i,
                     const int n)
{
    parametrization_assert(i < n && "Invalid index");
    parametrization_assert(6*n == P.size() && "Invalid number of rotations");
    return sym_from_triu(P(6*i), P(6*i+1), P(6*i+2), P(6*i+3), P(6*i+4),
                         P(6*i+5));
}


// Given a vector of stacked upper triangular parts of symmetrix 2x2 or 3x3
//  matrices, extract the matrix corresponding to the i-th index
//
// Inputs:
//  P  stacked vector of n symmetric matrices' upper triangular parts
//      provided in the format
//      [P1(0,0); P1(0,1); P1(1,1); P2(0,0); P2(0,1); P2(1,1)] (in 2d) or
//      [P1(0,0); P1(0,1); P1(0,2); P1(1,1); P1(1,2); P1(2,2); ...
//       P2(0,0); P2(0,1); P2(0,2); P2(1,1); P2(1,2); P2(2,2)] (in 3d)
//  i  index of matrix to extract
//  n  number of total matrices (i.e., P.size() / 6)
// Outputs:
//  return value  2x2 or 3x3 symmetrix matrix corresponding to the i-th matrix
//                in P
//
template <typename DerivedP, typename Int>
inline Eigen::Matrix<typename DerivedP::Scalar, Eigen::Dynamic, Eigen::Dynamic>
sym_from_index(const Eigen::MatrixBase<DerivedP>& P,
                     const Int i,
                     const int n)
{
    if(3*n == P.rows()) {
        return sym_2d_from_index(P, i, n);
    } else {
        parametrization_assert(6*n == P.rows() &&
                               "P does not correspond to n.");
        return sym_3d_from_index(P, i, n);
    }
}


// Given a symmetrix 2x2 matrix, write it into a vector of stacked upper
//  triangular parts of symmetrix 2x2 matrices at the i-th index
//
// Inputs:
//  Pl  symmetric 2x2 matrix. Will only use the upper triangular part.
//  P  stacked vector of n symmetric matrices' upper triangular parts
//      provided in the format
//      [P1(0,0); P1(0,1); P1(1,1); P2(0,0); P2(0,1); P2(1,1)]
//  i  index of matrix to extract
//  n  number of total matrices (i.e., P.size() / 3)
//

template <typename DerivedPl, typename DerivedP, typename Int>
inline void
index_into_sym_2x2(const Eigen::MatrixBase<DerivedPl>& Pl,
                   Eigen::PlainObjectBase<DerivedP>& P,
                   const Int i,
                   const Int n)
{
    parametrization_assert(i < n && "Invalid index");
    parametrization_assert(Pl.rows()==2 && Pl.cols()==2 &&
                           "Pl is not a 2x2 matrix.");
    parametrization_assert(Pl.determinant()>0 && Pl.trace()>0);
    parametrization_assert(Pl(0,1)==Pl(1,0) && "Pl is not symmetric.");
    parametrization_assert(3*n == P.size() && "Invalid number of matrices");
    
    int offs = 0;
    for(int j=0; j<2; ++j) {
        for(int k=j; k<2; ++k) {
            P(3*i + (offs++)) = Pl(j,k);
        }
    }
}


// Given a symmetrix 3x3 matrix, write it into a vector of stacked upper
//  triangular parts of symmetrix 3x3 matrices at the i-th index
//
// Inputs:
//  Pl  symmetric 3x3 matrix. Will only use the upper triangular part.
//  P  stacked vector of n symmetric matrices' upper triangular parts
//      provided in the format
//      [P1(0,0); P1(0,1); P1(0,2); P1(1,1); P1(1,2); P1(2,2); ...
//       P2(0,0); P2(0,1); P2(0,2); P2(1,1); P2(1,2); P2(2,2)]
//  i  index of matrix to extract
//  n  number of total matrices (i.e., P.size() / 6)
//

template <typename DerivedPl, typename DerivedP, typename Int>
inline void
index_into_sym_3x3(const Eigen::MatrixBase<DerivedPl>& Pl,
                   Eigen::PlainObjectBase<DerivedP>& P,
                   const Int i,
                   const Int n)
{
    parametrization_assert(i < n && "Invalid index");
    parametrization_assert(Pl.rows()==3 && Pl.cols()==3 &&
                           "Pl is not a 3x3 matrix.");
    parametrization_assert(Pl.determinant()>0 && Pl.trace()>0);
    parametrization_assert(Pl(0,1)==Pl(1,0) && Pl(0,2)==Pl(2,0) &&
                           Pl(1,2)==Pl(2,1) && "Pl is not symmetric.");
    parametrization_assert(6*n == P.size() && "Invalid number of matrices");
    
    int offs = 0;
    for(int j=0; j<3; ++j) {
        for(int k=j; k<3; ++k) {
            P(6*i + (offs++)) = Pl(j,k);
        }
    }
}


// Given a symmetrix 2x2 or 3x3 matrix, write it into a vector of stacked upper
//  triangular parts of symmetrix 2x2 or 3x3 matrices at the i-th index
//
// Inputs:
//  Pl  symmetric 2x2 or 3x3 matrix. Will only use the upper triangular part.
//  P  stacked vector of n symmetric matrices' upper triangular parts
//      provided in the format
//      [P1(0,0); P1(0,1); P1(1,1); P2(0,0); P2(0,1); P2(1,1)] (in 2d) or
//      [P1(0,0); P1(0,1); P1(0,2); P1(1,1); P1(1,2); P1(2,2); ...
//       P2(0,0); P2(0,1); P2(0,2); P2(1,1); P2(1,2); P2(2,2)] (in 3d)
//  i  index of matrix to extract
//  n  number of total matrices (i.e., P.size() / 3)
//

template <typename DerivedPl, typename DerivedP, typename Int>
inline void
index_into_sym(const Eigen::MatrixBase<DerivedPl>& Pl,
                   Eigen::PlainObjectBase<DerivedP>& P,
                   const Int i,
                   const Int n)
{
    const Eigen::Index dim = Pl.rows();
    if(dim==2) {
        index_into_sym_2x2(Pl, P, i, n);
    } else {
        parametrization_assert(dim==3 && "P is neither 2x2 nor 3x3.");
        index_into_sym_3x3(Pl, P, i, n);
    }
}


}

#endif
