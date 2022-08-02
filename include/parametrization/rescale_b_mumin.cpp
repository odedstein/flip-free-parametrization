
#include "rescale_b_mumin.h"

#include "parametrization_assert.h"
#include "sym_from_triu.h"
#include "mu_bound.h"
#include "grad_f.h"


template <bool parallelize, parametrization::EnergyType energy, int dim,
typename DerivedP, typename Derivedw, typename ScalarbMargin, typename Derivedb,
typename DerivedmuMin>
void
parametrization::rescale_b_mumin(const Eigen::MatrixBase<DerivedP>& P,
                                   const Eigen::MatrixBase<Derivedw>& w,
                                   const ScalarbMargin bMargin,
                                   Eigen::PlainObjectBase<Derivedb>& b,
                                   Eigen::PlainObjectBase<DerivedmuMin>& muMin)
{
    using Scalar = typename DerivedP::Scalar;
    using Mat22 = Eigen::Matrix<Scalar, 2, 2>;
    using Mat33 = Eigen::Matrix<Scalar, 3, 3>;
    
    static_assert((dim==2 || dim==3), "dim must be 2 or 3.");
    
    const Eigen::Index n = w.size();
    parametrization_assert(((dim==2 && 3*n==P.size()) ||
                            (dim==3 && 6*n==P.size())) &&
                           "P does not correspond to U.");
    parametrization_assert(bMargin>=0 && "bMargin must be nonnegative.");
    parametrization_assert(P.array().isFinite().all() && "invalid values in P");
    parametrization_assert(w.array().isFinite().all() && "invalid values in w");
    
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
    if(b.size() != n) {
        b.resize(n,1);
    }
    if(muMin.size() != n) {
        muMin.resize(n,1);
    }
    
    //Rescale
    const auto rescale = [&] (const Eigen::Index i) {
        if(dim==2) {
            const Mat22 Pl = sym_2d_from_index(P, i, n);
            const Scalar gradNormSq = sq_norm_grad_f<energy>(Pl);
            
            b(i) = bMargin*(1. + gradNormSq);
            muMin(i) = mu_bound<energy,2>(w(i), sqrt(b(i)));
        } else {
            const Mat33 Pl = sym_3d_from_index(P, i, n);
            const Scalar gradNormSq = sq_norm_grad_f<energy>(Pl);
            
            b(i) = bMargin*(1. + gradNormSq);
            muMin(i) = mu_bound<energy,3>(w(i), sqrt(b(i)));
        }
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
}


// Explicit template instantiation
template void parametrization::rescale_b_mumin<false, (parametrization::EnergyType)0, 2, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, double, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, double, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&);
template void parametrization::rescale_b_mumin<false, (parametrization::EnergyType)0, 3, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, double, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, double, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&);
template void parametrization::rescale_b_mumin<false, (parametrization::EnergyType)1, 2, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, double, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, double, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&);
template void parametrization::rescale_b_mumin<false, (parametrization::EnergyType)1, 3, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, double, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, double, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&);
#ifdef OPENMP_AVAILABLE
template void parametrization::rescale_b_mumin<true, (parametrization::EnergyType)0, 2, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, double, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, double, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&);
template void parametrization::rescale_b_mumin<true, (parametrization::EnergyType)0, 3, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, double, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, double, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&);
template void parametrization::rescale_b_mumin<true, (parametrization::EnergyType)1, 2, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, double, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, double, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&);
template void parametrization::rescale_b_mumin<true, (parametrization::EnergyType)1, 3, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, double, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, double, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&);
#endif
