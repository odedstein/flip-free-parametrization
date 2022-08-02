#ifndef PARAMETRIZATION_MU_BOUND_H
#define PARAMETRIZATION_MU_BOUND_H

#include <Eigen/Core>

#include "parametrization_assert.h"
#include "quartic_polynomial.h"
#include "energy.h"


namespace parametrization {

// Computes the theoretical lower bound on the Augmented Lagrangian penalty
//  parameter mu, given triangle area w and gradient bound b.
//
// Inputs:
//  w  triangle area
//  b  gradient bound for that triangle (*not* squared)
//  template argument energy  which energy to compute
//  template argument dim  choose between 2d and 3d variant of the method.
//  template argument lessAggressiveHeuristic  if this is set to true, will
//                                             choose a lower bound than the
//                                             theoretical optimum based on a
//                                             heuristic.
// Outputs
//  return value  theoretical lower bound on mu

template <EnergyType energy, int dim,
bool lessAggressiveHeuristic=true,
typename Scalarw, typename Scalarb>
inline Scalarw
mu_bound(const Scalarw w, const Scalarb b)
{
    using Scalar = Scalarw;
    
    static_assert(dim==2 || dim==3, "only dimensions 2 and 3 supported.");
    parametrization_assert(w>0 && "triangle area is positive");
    parametrization_assert(b>0 && "gradient bound is positive");
    
    const Scalar tol = std::numeric_limits<Scalarb>::epsilon();
    const Scalar one=1., zero=0.;
    
    Scalar muBound;
    switch(energy) {
        case EnergyType::SymmetricGradient:
        {
            //c is the positive root of c^2 + b*c - 1
            const Scalar c = 0.5*(-b + sqrt(4. + b*b));
            Scalar c2 = c*c;
            if(c2 < tol) {
                //Due to numerical troubles this might happen.
                c2 = tol;
            }
            const Scalar f = 1. + sqrt(dim)/c2;
            if(lessAggressiveHeuristic) {
                const Scalar fsq = (std::min)(std::pow(tol,-0.25), f*f);
                muBound = 0.25*w * (sqrt(1.+16.*fsq) - 1.);
            } else {
                muBound = 0.5*w * (sqrt(1.+16.*f*f) - 1.);
            }
            break;
        }
        case EnergyType::SymmetricDirichlet:
        {
            //c is the positive root of c^4 + b*c^3 - 1
            Scalar c = quartic_polynomial(b, zero, zero, -one);
            if(!std::isfinite(c)) {
                c = 0.5*(-b + sqrt(4. + b*b));
            }
            Scalar c4 = (c*c)*(c*c);
            if(c4 < tol) {
                //Due to numerical troubles this might happen.
                c4 = tol;
            }
            const Scalar f = 1. + 3.*sqrt(dim)/c4;
            if(lessAggressiveHeuristic) {
                const Scalar fsq = (std::min)(std::pow(tol,-0.25), f*f);
                muBound = 0.25*w * (sqrt(1.+16.*fsq) - 1.);
            } else {
                muBound = 0.5*w * (sqrt(1.+16.*f*f) - 1.);
            }
            break;
        }
    }
    
    //Due to numerical issues, mu might be very small and negative.
    if(muBound < 0.) {
        muBound = 0.;
    }
    
    parametrization_assert(std::isfinite(muBound));
    
    return muBound;
}



}

#endif
