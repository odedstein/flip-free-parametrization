
#include "rescale_h.h"

#include "parametrization_assert.h"
#include "h_bound.h"
#include "mat_from_index.h"

#include <igl/PI.h>
#include <iostream>

#include <Eigen/Dense>


template <bool parallelize,
typename Derivedw, typename Derivedmu, typename Derivedb, typename Derivedh>
void
parametrization::rescale_h(const Eigen::MatrixBase<Derivedw>& w,
                           const Eigen::MatrixBase<Derivedmu>& mu,
                           const Eigen::MatrixBase<Derivedb>& b,
                           Eigen::PlainObjectBase<Derivedh>& h)
{
    using Scalar = typename Derivedw::Scalar;
    
    const Eigen::Index n = w.size();
    parametrization_assert(n==mu.size() && "mu does not correspond to w.");
    parametrization_assert(n==b.size() && "b does not correspond to w.");
    
    parametrization_assert(w.array().isFinite().all() && "invalid values in w");
    parametrization_assert(mu.array().isFinite().all() && "invalid values in w");
    parametrization_assert(b.array().isFinite().all() && "invalid values in w");
    
    //Make sure OpenMP exists if parallelize is set to true
#ifndef OPENMP_AVAILABLE
    static_assert(!parallelize, "Can't parallelize without OpenMP.");
#endif
    
    //Should be if constexpr, but libigl doesn't support C++17
    if(parallelize) {
        //Turn off Eigen's parallelization.
        Eigen::setNbThreads(1);
    }
    
    //Resize output matrices if they have the wrong size.
    if(h.size() != n) {
        h.resize(n,1);
    }
    
    //Rescale
    const auto rescale = [&] (const Eigen::Index i) {
        h(i) = h_bound(w(i), mu(i), b(i));
    };
    
    //Should be if constexpr, but libigl doesn't support C++17
    if(parallelize) {
#ifdef OPENMP_AVAILABLE
#pragma omp parallel for
        for(Eigen::Index i=0; i<n; ++i) {
            rescale(i);
        }
#endif
    } else {
        for(Eigen::Index i=0; i<n; ++i) {
            rescale(i);
        }
    }
    
    //Should be if constexpr, but libigl doesn't support C++17
    if(parallelize) {
        //Turn on Eigen's parallelization.
        Eigen::setNbThreads(0);
    }
    
    parametrization_assert(b.array().isFinite().all() &&
                           "Rescaling gradient bounds failed.");
}


// Explicit template instantiation
template void parametrization::rescale_h<false, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&);
#ifdef OPENMP_AVAILABLE
template void parametrization::rescale_h<true, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&);
#endif
