#ifndef PARAMETRIZATION_GRAD_F_H
#define PARAMETRIZATION_GRAD_F_H

#include <Eigen/Dense>

#include "energy.h"


namespace parametrization {

// Compute the gradient of the function f(P) for a spd dim x dim matrix P,
//  where dim==2 or 3
//
// Inputs:
//  P  2x2 spd matrix
// Outputs:
//  return value  grad_f(P)
//
template <EnergyType energy, typename DerivedP>
inline Eigen::Matrix<typename DerivedP::Scalar, DerivedP::RowsAtCompileTime,
DerivedP::ColsAtCompileTime>
grad_f(const Eigen::MatrixBase<DerivedP>& P)
{
    using Scalar = typename DerivedP::Scalar;
    constexpr int dim = DerivedP::RowsAtCompileTime;
    using Mat = Eigen::Matrix<Scalar, dim, dim>;
    static_assert(dim==2 || dim==3, "This function is for dim x dim matrices.");
    
    parametrization_assert(P.array().isFinite().all() && "Invalid values in P");
    parametrization_assert(P.rows()==dim && P.cols()==dim &&
                           "This function is for dim x dim matrices.");
    parametrization_assert(P.transpose()==P &&
                           "This function is for symmetric matrices.");
    parametrization_assert(P.determinant()>0 && P.trace()>0 &&
                           "This function is for positive definite matrices.");
    
    const Mat Pinv = P.inverse();
    switch(energy) {
        case EnergyType::SymmetricGradient:
        {
            return P - Pinv;
        }
        case EnergyType::SymmetricDirichlet:
        {
            return P - Pinv*Pinv*Pinv;
        }
    }
}


// Compute the squared norm of the gradient of the function f(P) for a spd
//  dim x dim matrix P, where dim==2 or 3
//
// Inputs:
//  P  2x2 spd matrix
// Outputs:
//  return value  ||grad_f(P)||^2
//
template <EnergyType energy, typename DerivedP>
inline typename DerivedP::Scalar
sq_norm_grad_f(const Eigen::MatrixBase<DerivedP>& P)
{
    return grad_f<energy>(P).squaredNorm();
}


}

#endif
