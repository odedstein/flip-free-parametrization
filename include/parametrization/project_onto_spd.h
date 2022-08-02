#ifndef PARAMETRIZATION_PROJECT_ONTO_SPD_H
#define PARAMETRIZATION_PROJECT_ONTO_SPD_H

#include <Eigen/Dense>

#include "parametrization_assert.h"
#include "symmetrize.h"


namespace parametrization {

// Project a symmetric matrix onto spd matrices
//
// Inputs:
//  P  symmetric matrix
//  tol  tolerance to determine whether a number is zero
// Outputs:
//  return value  the matrix P, projected onto spd matrices
//
template <typename DerivedP, typename Scalartol>
inline Eigen::Matrix<typename DerivedP::Scalar, DerivedP::RowsAtCompileTime,
DerivedP::ColsAtCompileTime>
project_onto_spd(const Eigen::MatrixBase<DerivedP>& P,
                 const Scalartol tol)
{
    using Scalar = typename DerivedP::Scalar;
    using HScalar = long double;
    using Vec = Eigen::Matrix<HScalar, DerivedP::RowsAtCompileTime, 1>;
    using Mat = Eigen::Matrix<HScalar, DerivedP::RowsAtCompileTime,
    DerivedP::ColsAtCompileTime>;
    using EigenSolver = Eigen::SelfAdjointEigenSolver<Mat>;
    
    parametrization_assert(tol>0 && "tol must be a small positive number.");
    parametrization_assert(P==P.transpose() && "P must be symmetric.");
    
    EigenSolver eigenSolver(P.template cast<HScalar>());
    Vec d = eigenSolver.eigenvalues();
    for(int i=0; i<d.size(); ++i) {
        if(d(i) < tol) {
            d(i) = tol;
        }
    }
    const Mat& V = eigenSolver.eigenvectors();
    
    Mat Q = V * d.asDiagonal() * V.transpose();
    symmetrize(Q);
    
    return Q.template cast<Scalar>();
}

}

#endif
