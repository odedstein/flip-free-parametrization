#ifndef PARAMETRIZATION_SPD_QUARTIC_POLYNOMIAL_H
#define PARAMETRIZATION_SPD_QUARTIC_POLYNOMIAL_H

#include <Eigen/Core>

namespace parametrization {

// Compute the spd matrix solving the matrix polynomial P^4 + B*P^3 + c*Id = 0
//
// Inputs:
//  B  a symmetric matrix
//  c  a negative scalar (otherwise an spd result might not exist!)
//
// Outputs:
//  return value  whether an spd solution to the polynomial was found or not
//  P  the spd solution to the polynomial
//
template <typename DerivedB, typename Scalarc, typename DerivedP>
void
spd_quartic_polynomial(const Eigen::MatrixBase<DerivedB>& B,
                       const Scalarc c,
                       Eigen::PlainObjectBase<DerivedP>& P);

}

#endif
