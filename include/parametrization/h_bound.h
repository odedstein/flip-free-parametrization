#ifndef PARAMETRIZATION_H_BOUND_H
#define PARAMETRIZATION_H_BOUND_H

#include <Eigen/Core>

#include "parametrization_assert.h"

namespace parametrization {

// Computes the theoretical lower bound on the proximal penalty parameter h,
//  given the Augmented Lagrangian penalty parameter mu, triangle area w and
//  gradient bound b.
// Attention! This is a scaled version of the parameter h in the paper, for
//  implementation reasons. code h = (paper h)/mu
//
// Inputs:
//  w  triangle area
//  mu  Augmented Lagrangian penalty parameter
//  b  squared gradient bound for that triangle
// Outputs
//  return value  theoretical lower bound on h

template <bool lessAggressiveHeuristic=true,
typename Scalarw, typename Scalarmu, typename Scalarb>
inline Scalarmu
h_bound(const Scalarw w, const Scalarmu mu, const Scalarb b)
{
    parametrization_assert(mu>0 && "mu is positive");
    parametrization_assert(w>0 && "triangle area is positive");
    parametrization_assert(b>0 && "gradient bound is positive");
    
    const Scalarmu tol = std::numeric_limits<Scalarmu>::epsilon();
    
    Scalarmu denom = mu * mu;
    if(denom < tol) {
        //Due to numerical troubles this might happen.
        denom = tol;
    }
    
    Scalarmu hBound;
    if(lessAggressiveHeuristic) {
        hBound = 4. * w*w * b / denom;
    } else {
        hBound = 4. * w*w * b / denom;
    }
    
    parametrization_assert(std::isfinite(hBound));
    parametrization_assert(hBound >= 0);
    
    return hBound;
}



}

#endif
