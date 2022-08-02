
#include "rescale_penalties.h"

#include "parametrization_assert.h"
#include "mat_from_index.h"


template <bool parallelize, int idim,
typename DerivedePs, typename DerivedeDs, typename ScalarpenaltyIncrease,
typename ScalarpenaltyDecrease, typename ScalardifferenceToRescale,
typename Derivedmu, typename DerivedLambda, typename DerivedmuMin>
bool
parametrization::rescale_penalties(const Eigen::MatrixBase<DerivedePs>& ePs,
                  const Eigen::MatrixBase<DerivedeDs>& eDs,
                  const ScalarpenaltyIncrease penaltyIncrease,
                  const ScalarpenaltyDecrease penaltyDecrease,
                  const ScalardifferenceToRescale differenceToRescale,
                  Eigen::PlainObjectBase<Derivedmu>& mu,
                  Eigen::PlainObjectBase<DerivedLambda>& Lambda,
                  const Eigen::MatrixBase<DerivedmuMin>& muMin)
{
    using Scalar = typename Derivedmu::Scalar;
    const Scalar tol = std::numeric_limits<Scalar>::epsilon();
    
    parametrization_assert(ePs.array().isFinite().all() &&
                           "illegal entry in ePs");
    parametrization_assert(eDs.array().isFinite().all() &&
                           "illegal entry in eDs");
    
    const int dim = idim>0 ? idim : Lambda.cols();
    parametrization_assert((dim==2 || dim==3) && "dim must be 2 or 3.");
    
    const Eigen::Index n = mu.size();
    parametrization_assert(dim*n==Lambda.rows() && dim==Lambda.cols() &&
                           "Lambda does not correspond to mu.");
    
    parametrization_assert(n==ePs.size() && "ePs does not correspond to mu.");
    parametrization_assert(n==eDs.size() && "eDs does not correspond to mu.");
    parametrization_assert(penaltyIncrease>=1. &&
                           "penaltyIncrease is decreasing.");
    parametrization_assert(penaltyDecrease>=1. &&
                           "penaltyDecrease is increasing.");
    parametrization_assert(differenceToRescale>1. &&
                           "differenceToRescale must be >1.");
    parametrization_assert(ePs.minCoeff()>=0 && eDs.minCoeff()>=0 &&
                           "Errors can't be negative.");
    
    parametrization_assert((muMin.size()==0 || muMin.size()==n) &&
                           "muMin must be empty or correspond to mu.");
    parametrization_assert((muMin.size()==0 || muMin.minCoeff()>=0) &&
                           "muMin must be empty or nonnegative.");
    
    //Make sure OpenMP exists if parallelize is set to true
#ifndef OPENMP_AVAILABLE
    static_assert(!parallelize, "Can't parallelize without OpenMP.");
#endif
    
    //Should be if constexpr, but libigl doesn't support C++17
    if(parallelize) {
        //Turn off Eigen's parallelization.
        Eigen::setNbThreads(1);
    }
    
    //The multiply/divide into functions depends on dimension, so we use
    // helper functions.
    const auto multiply_into_Lambda = [&]
    (const Scalar x, const Eigen::Index i) {
        if(dim==2) {
            multiply_into_mat_2x2(x, Lambda, i, n);
        } else {
            multiply_into_mat_3x3(x, Lambda, i, n);
        }
    };
    const auto divide_into_Lambda = [&]
    (const Scalar x, const Eigen::Index i) {
        if(dim==2) {
            divide_into_mat_2x2(x, Lambda, i, n);
        } else {
            divide_into_mat_3x3(x, Lambda, i, n);
        }
    };
    
    //Need to square the rescale parameter, since the errors are squared.
    const Scalar d = differenceToRescale*differenceToRescale;
    
    //Rescale
    const auto rescale = [&] (const Eigen::Index i, bool& changed) {
        const Scalar muMinl = muMin.size()==0 ? 0. : muMin(i);
        if(mu(i)<muMinl) {
            mu(i) = muMinl;
            multiply_into_Lambda(mu(i)/muMinl, i);
            changed = changed || true;
        } else if(ePs(i) > d*eDs(i)) {
            mu(i) *= penaltyIncrease;
            divide_into_Lambda(penaltyIncrease, i);
            changed = changed || true;
        } else if(eDs(i) > d*ePs(i)) {
            if(mu(i)==muMinl) {
                //This mu is equal to muMin and can not decrease.
                changed = changed || false;
                return;
            }
            const Scalar decreasedMu = mu(i)/penaltyDecrease;
            if(decreasedMu<muMinl) {
                mu(i) = muMinl;
                multiply_into_Lambda(mu(i)/muMinl, i);
            } else {
                mu(i) = decreasedMu;
                multiply_into_Lambda(penaltyDecrease, i);
            }
            changed = changed || true;
        }
    };
    
    //Should be if constexpr, but libigl doesn't support C++17
    bool changed = false;
    if(parallelize) {
#ifdef OPENMP_AVAILABLE
#pragma omp parallel for reduction(||:changed)
        for(Eigen::Index i=0; i<n; ++i) {
            rescale(i, changed);
        }
#endif
    } else {
        for(Eigen::Index i=0; i<n; ++i) {
            rescale(i, changed);
        }
    }
    
    //Should be if constexpr, but libigl doesn't support C++17
    if(parallelize) {
        //Turn on Eigen's parallelization.
        Eigen::setNbThreads(0);
    }
    
    parametrization_assert(mu.array().isFinite().all() &&
                           "mu increased/decreased too much.");
    parametrization_assert(Lambda.array().isFinite().all() &&
                           "Lambda increased/decreased too much.");
    
    return changed;
}


//Explicit template instantiation
template bool parametrization::rescale_penalties<false, 2, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, double, double, double, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 2, 0, -1, 2>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, double, double, double, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 2, 0, -1, 2> >&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&);
template bool parametrization::rescale_penalties<false, 2, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, double, double, double, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, double, double, double, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&);
template bool parametrization::rescale_penalties<false, 3, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, double, double, double, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 2, 0, -1, 2>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, double, double, double, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 2, 0, -1, 2> >&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&);
template bool parametrization::rescale_penalties<false, 3, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, double, double, double, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, double, double, double, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&);
#ifdef OPENMP_AVAILABLE
template bool parametrization::rescale_penalties<true, 2, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, double, double, double, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 2, 0, -1, 2>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, double, double, double, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 2, 0, -1, 2> >&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&);
template bool parametrization::rescale_penalties<true, 2, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, double, double, double, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, double, double, double, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&);
template bool parametrization::rescale_penalties<true, 3, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, double, double, double, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 2, 0, -1, 2>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, double, double, double, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 2, 0, -1, 2> >&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&);
template bool parametrization::rescale_penalties<true, 3, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, double, double, double, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, double, double, double, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&);
#endif
