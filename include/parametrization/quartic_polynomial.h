#ifndef PARAMETRIZATION_QUARTIC_POLYNOMIAL_H
#define PARAMETRIZATION_QUARTIC_POLYNOMIAL_H


namespace parametrization {

// Compute a positive root of a convex scalar quartic polynomial
//  x^4 + a*x^3 + b*x^2 + c*x + d = 0.
//
// This function uses the public domain code of Khashin S.I.
//  http://math.ivanovo.ac.ru/dalgebra/Khashin/index.html
//
// Inputs:
//  a, b, c, d  polynomial coefficients
//
// Outputs:
//  return value  positive real polynomial root. If multiple exist, will return
//                an arbitrary one. If none exists, returns NaN.
//
template <typename Scalar>
Scalar
quartic_polynomial(const Scalar a,
                   const Scalar b,
                   const Scalar c,
                   const Scalar d);

}

#endif
