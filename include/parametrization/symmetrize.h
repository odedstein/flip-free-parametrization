#ifndef PARAMETRIZATION_SYMMETRIZE_H
#define PARAMETRIZATION_SYMMETRIZE_H

#include <Eigen/Core>

#include "parametrization_assert.h"


namespace parametrization {

// Symmetrize a square matrix in-place.
//
// Input:
//  X  square matrix
// Output:
//  X  symmetrized square matrix
//
template <typename DerivedX>
inline void
symmetrize(Eigen::PlainObjectBase<DerivedX>& X)
{
    using Scalar = typename DerivedX::Scalar;
    
    const Eigen::Index n = X.rows();
    parametrization_assert(X.cols()==n && "X must be square");
    
    for(Eigen::Index i=0; i<n; ++i) {
        for(Eigen::Index j=i+1; j<n; ++j) {
            const Scalar sym = 0.5*(X(i,j) + X(j,i));
            X(i,j) = sym;
            X(j,i) = sym;
        }
    }
}

}

#endif
