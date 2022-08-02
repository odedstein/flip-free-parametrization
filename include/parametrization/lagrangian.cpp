
#include "lagrangian.h"

#include "parametrization_assert.h"
#include "rotmat_from_complex_quat.h"
#include "sym_from_triu.h"
#include "mat_from_index.h"

#include <Eigen/Sparse>

#include <limits>


template <bool parallelize, parametrization::EnergyType energy, int idim,
typename DerivedU, typename DerivedP, typename DerivedGW,
typename DerivedLambda, typename Derivedw, typename Derivedmu>
typename DerivedU::Scalar
parametrization::lagrangian(const Eigen::MatrixBase<DerivedU>& U,
           const Eigen::MatrixBase<DerivedP>& P,
           const Eigen::MatrixBase<DerivedGW>& GW,
           const Eigen::MatrixBase<DerivedLambda>& Lambda,
           const Eigen::MatrixBase<Derivedw>& w,
           const Eigen::MatrixBase<Derivedmu>& mu)
{
    using Scalar = typename DerivedU::Scalar;
    using Mat22 = Eigen::Matrix<Scalar, 2, 2>;
    using Mat33 = Eigen::Matrix<Scalar, 3, 3>;
    const Scalar tol = std::numeric_limits<Scalar>::epsilon();
    
    parametrization_assert(U.array().isFinite().all() &&
                           "Invalid entries in U");
    parametrization_assert(P.array().isFinite().all() &&
                           "Invalid entries in P");
    parametrization_assert(GW.array().isFinite().all() &&
                           "Invalid entries in GW");
    parametrization_assert(Lambda.array().isFinite().all() &&
                           "Invalid entries in Lambda");
    parametrization_assert(w.array().isFinite().all() &&
                           "Invalid entries in w");
    parametrization_assert(mu.array().isFinite().all() &&
                           "Invalid entries in mu");
    
    const int dim = idim>0 ? idim : GW.cols();
    parametrization_assert((dim==2 || dim==3) && "dim must be 2 or 3.");
    
    const Eigen::Index n = dim==2 ? U.size()/2 : U.size()/4;
    parametrization_assert(((dim==2 && 2*n==U.size()) ||
                            (dim==3 && 4*n==U.size())) &&
                           "U has the wrong size");
    parametrization_assert(((dim==2 && 3*n==P.size()) ||
                            (dim==3 && 6*n==P.size())) &&
                           "P does not correspond to U.");
    parametrization_assert(dim*n==GW.rows() && dim==GW.cols() &&
                           "GW does not correspond to U.");
    parametrization_assert(dim*n==Lambda.rows() && dim==Lambda.cols() &&
                           "Lambda does not correspond to P.");
    parametrization_assert(n==mu.size() && "mu does not correspond to U.");
    parametrization_assert(n==w.size() && "w does not correspond to U.");
    parametrization_assert(w.minCoeff()>tol && "Weights need to be positive.");
    parametrization_assert(mu.minCoeff()>tol &&
                           "Penalties need to be positive.");
    
    
    //Make sure OpenMP exists if parallelize is set to true
#ifndef OPENMP_AVAILABLE
    static_assert(!parallelize, "Can't parallelize without OpenMP.");
#endif
    
    //Should be if constexpr, but libigl doesn't support C++17
    if(parallelize) {
        //Turn off Eigen's parallelization.
        Eigen::setNbThreads(1);
    }
    
    const auto lag_for_triangle = [&] (const Eigen::Index i, Scalar& E) {
        if(dim==2) {
            const Mat22 Pl = sym_2d_from_index(P, i, n);
            Scalar fP;
            switch(energy) {
                case EnergyType::SymmetricGradient:
                {
                    Scalar det = Pl.determinant();
                    //P is strictly positive definite, so if the determinant is 0 here,
                    // this must be a numerical error, and we can clamp.
                    if(det<=0) {
                        det = std::numeric_limits<Scalar>::min();
                    }
                    //E(P) = w * (0.5*||P||_F^2 - log(det(P)))
                    fP = 0.5*Pl.squaredNorm() - log(det);
                    break;
                }
                case EnergyType::SymmetricDirichlet:
                {
                    Scalar invNorm = Pl.inverse().squaredNorm();
                    //P is strictly positive definite, so if taking the inverse
                    // failed this must be a numerical error, and we can just
                    // set the energy to be very large.
                    if(!std::isfinite(invNorm)) {
                        invNorm = 1. / std::numeric_limits<Scalar>::min();
                    }
                    //E(P) = w/2 * (||P||_F^2 + ||inv(P)||^2)
                    fP = 0.5 * (Pl.squaredNorm() + invNorm);
                    break;
                }
            }
            
            const Mat22 Ul = rotmat_2d_from_index(U, i, n);
            const Mat22 GWl = mat_from_index_2x2(GW, i, n);
            const Mat22 Lambdal = mat_from_index_2x2(Lambda, i, n);
            //Penalty term 0.5*||GW - U*P + Lambda||_F^2
            const Scalar penalty = 0.5*(GWl - Ul*Pl + Lambdal).squaredNorm();
            
            //Phi = w*fP + mu*penalty
            E += w(i)*fP + mu(i)*penalty;
        } else {
            const Mat33 Pl = sym_3d_from_index(P, i, n);
            Scalar fP;
            switch(energy) {
                case EnergyType::SymmetricGradient:
                {
                    Scalar det = Pl.determinant();
                    //P is strictly positive definite, so if the determinant is 0 here,
                    // this must be a numerical error, and we can clamp.
                    if(det<=0) {
                        det = std::numeric_limits<Scalar>::min();
                    }
                    //E(P) = w * (0.5*||P||_F^2 - log(det(P)))
                    fP = 0.5*Pl.squaredNorm() - log(det);
                    break;
                }
                case EnergyType::SymmetricDirichlet:
                {
                    Scalar invNorm = Pl.inverse().squaredNorm();
                    //P is strictly positive definite, so if taking the inverse
                    // failed this must be a numerical error, and we can just
                    // set the energy to be very large.
                    if(!std::isfinite(invNorm)) {
                        invNorm = 1. / std::numeric_limits<Scalar>::min();
                    }
                    //E(P) = w/2 * (||P||_F^2 + ||inv(P)||^2)
                    fP = 0.5 * (Pl.squaredNorm() + invNorm);
                    break;
                }
            }
            
            const Mat33 Ul = rotmat_3d_from_index(U, i, n);
            const Mat33 GWl = mat_from_index_3x3(GW, i, n);
            const Mat33 Lambdal = mat_from_index_3x3(Lambda, i, n);
            //Penalty term 0.5*||GW - U*P + Lambda||_F^2
            const Scalar penalty = 0.5*(GWl - Ul*Pl + Lambdal).squaredNorm();
            
            //Phi = w*fP + mu*penalty
            E += w(i)*fP + mu(i)*penalty;
        }
    };
    
    Scalar E = 0;
    if(parallelize) {
#ifdef OPENMP_AVAILABLE
#pragma omp parallel for reduction(+:E)
        for(Eigen::Index i=0; i<n; ++i) {
            lag_for_triangle(i, E);
        }
#endif
    } else {
        for(Eigen::Index i=0; i<n; ++i) {
            lag_for_triangle(i, E);
        }
    }
    
    //Should be if constexpr, but libigl doesn't support C++17
    if(parallelize) {
        //Turn on Eigen's parallelization.
        Eigen::setNbThreads(0);
    }
    
    return E;
}


// Explicit template instanciation
template Eigen::Matrix<double, -1, 1, 0, -1, 1>::Scalar parametrization::lagrangian<false, (parametrization::EnergyType)0, -1, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&);
template Eigen::Matrix<double, -1, 1, 0, -1, 1>::Scalar parametrization::lagrangian<false, (parametrization::EnergyType)1, -1, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&);
template Eigen::Matrix<double, -1, 1, 0, -1, 1>::Scalar parametrization::lagrangian<false, (parametrization::EnergyType)0, 2, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&);
template Eigen::Matrix<double, -1, 1, 0, -1, 1>::Scalar parametrization::lagrangian<false, (parametrization::EnergyType)1, 2, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&);
template Eigen::Matrix<double, -1, 1, 0, -1, 1>::Scalar parametrization::lagrangian<false, (parametrization::EnergyType)0, 3, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&);
template Eigen::Matrix<double, -1, 1, 0, -1, 1>::Scalar parametrization::lagrangian<false, (parametrization::EnergyType)1, 3, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&);
#ifdef OPENMP_AVAILABLE
template Eigen::Matrix<double, -1, 1, 0, -1, 1>::Scalar parametrization::lagrangian<true, (parametrization::EnergyType)0, -1, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&);
template Eigen::Matrix<double, -1, 1, 0, -1, 1>::Scalar parametrization::lagrangian<true, (parametrization::EnergyType)1, -1, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&);
template Eigen::Matrix<double, -1, 1, 0, -1, 1>::Scalar parametrization::lagrangian<true, (parametrization::EnergyType)0, 2, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&);
template Eigen::Matrix<double, -1, 1, 0, -1, 1>::Scalar parametrization::lagrangian<true, (parametrization::EnergyType)1, 2, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&);
template Eigen::Matrix<double, -1, 1, 0, -1, 1>::Scalar parametrization::lagrangian<true, (parametrization::EnergyType)0, 3, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&);
template Eigen::Matrix<double, -1, 1, 0, -1, 1>::Scalar parametrization::lagrangian<true, (parametrization::EnergyType)1, 3, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&);
#endif
