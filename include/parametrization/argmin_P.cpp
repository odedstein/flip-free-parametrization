
#include "argmin_P.h"

#include "parametrization_assert.h"
#include "sym_from_triu.h"
#include "mat_from_index.h"
#include "rotmat_from_complex_quat.h"
#include "symmetrize.h"
#include "sqrtm.h"
#include "spd_quartic_polynomial.h"
#include "project_onto_spd.h"

#include <Eigen/Dense>

#include <iomanip>


template <bool parallelize, parametrization::EnergyType energy, int idim,
typename DerivedU, typename DerivedGW, typename DerivedLambda,
typename Derivedw, typename Derivedmu, typename DerivedP>
void
parametrization::argmin_P(const Eigen::MatrixBase<DerivedU>& U,
                          const Eigen::MatrixBase<DerivedGW>& GW,
                          const Eigen::MatrixBase<DerivedLambda>& Lambda,
                          const Eigen::MatrixBase<Derivedw>& w,
                          const Eigen::MatrixBase<Derivedmu>& mu,
                          Eigen::PlainObjectBase<DerivedP>& P)
{
    using Scalar = typename DerivedU::Scalar;
    using Mat22 = Eigen::Matrix<Scalar, 2, 2>;
    using Mat33 = Eigen::Matrix<Scalar, 3, 3>;
    
    const Mat22 Id2 = Mat22::Identity();
    const Mat33 Id3 = Mat33::Identity();
    const Scalar tol = sqrt(std::numeric_limits<Scalar>::epsilon());
    
    parametrization_assert(U.array().isFinite().all() &&
                           "Invalid entries in U");
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
    parametrization_assert(dim*n==GW.rows() && dim==GW.cols() &&
                           "GW does not correspond to U.");
    parametrization_assert(dim*n==Lambda.rows() && dim==Lambda.cols() &&
                           "Lambda does not correspond to U.");
    parametrization_assert(n==mu.size() && "mu does not correspond to U.");
    parametrization_assert(n==w.size() && "w does not correspond to U.");
    parametrization_assert(w.minCoeff()>tol && "Weights need to be positive.");
    parametrization_assert(mu.minCoeff()>tol && "Penalties need to be positive.");
    
    //Make sure OpenMP exists if parallelize is set to true
#ifndef OPENMP_AVAILABLE
    static_assert(!parallelize, "Can't parallelize without OpenMP.");
#endif
    
    //Should be if constexpr, but libigl doesn't support C++17
    if(parallelize) {
        //Turn off Eigen's parallelization.
        Eigen::setNbThreads(1);
    }
    
    //Construct the vector P.
    if(dim==2) {
        if(P.size()!=3*n) {
            P.resize(3*n,1);
        }
    } else if(P.size()!=6*n) {
        P.resize(6*n,1);
    }
    //Solve for P in each coordinate.
    const auto solve_for_P = [&] (const Eigen::Index i) {
        const Scalar wl = w(i);
        const Scalar mul = mu(i);
        const Scalar wlpmul = wl + mul;
        const Scalar wlBwlpmul = wl / wlpmul;
        const Scalar mulBwlpmul = mul / wlpmul;
        
        if(dim==2) {
            const Mat22 Ul = rotmat_2d_from_index(U, i, n);
            const Mat22 Lambdal = mat_from_index_2x2(Lambda, i, n);
            const Mat22 GWl = mat_from_index_2x2(GW, i, n);
            const Mat22 UltGWlpLl = Ul.transpose()*(GWl+Lambdal);
            Mat22 Q = mulBwlpmul * UltGWlpLl;
            symmetrize(Q);
            
            Mat22 Pl;
            if(Q.squaredNorm() < tol) {
                switch(energy) {
                    //Taylor expansion around Q = 0
                    case EnergyType::SymmetricGradient:
                        Pl = sqrt(wlBwlpmul)*Id2 + 0.5*Q;
                        break;
                    case EnergyType::SymmetricDirichlet:
                        Pl = sqrt(sqrt(wlBwlpmul))*Id2 + 0.25*Q;
                        break;
                }
            } else {
                switch(energy) {
                    case EnergyType::SymmetricGradient:
                    {
                        Mat22 discr = Q*Q + (4.*wlBwlpmul)*Id2;
                        symmetrize(discr); //for numerical reasons
                        Pl = 0.5 * (Q + sqrtm_2x2(discr));
                        break;
                    }
                    case EnergyType::SymmetricDirichlet:
                    {
                        const Mat22 B = - Q;
                        const Scalar c = - wlBwlpmul;
                        spd_quartic_polynomial(B, c, Pl);
                        break;
                    }
                }
            }
            
            if(Pl.determinant()<=tol || Pl.trace()<=tol) {
                //This can happen due to numerical reasons, even if the input
                // is good.
                Pl = project_onto_spd(Pl, tol);
            }
            index_into_sym_2x2(Pl, P, i, n);
        } else {
            const Mat33 Ul = rotmat_3d_from_index(U, i, n);
            const Mat33 Lambdal = mat_from_index_3x3(Lambda, i, n);
            const Mat33 GWl = mat_from_index_3x3(GW, i, n);
            const Mat33 UltGWlpLl = Ul.transpose()*(GWl+Lambdal);
            Mat33 Q = mulBwlpmul * UltGWlpLl;
            symmetrize(Q);
            
            Mat33 Pl;
            if(Q.squaredNorm() < tol*tol) {
                switch(energy) {
                    //Taylor expansion around Q = 0
                    case EnergyType::SymmetricGradient:
                        Pl = sqrt(wlBwlpmul)*Id3 + 0.5*Q;
                        break;
                    case EnergyType::SymmetricDirichlet:
                        Pl = sqrt(sqrt(wlBwlpmul))*Id3 + 0.25*Q;
                        break;
                }
            } else {
                switch(energy) {
                    case EnergyType::SymmetricGradient:
                    {
                        Mat33 discr = Q*Q + (4.*wlBwlpmul)*Id3;
                        symmetrize(discr); //for numerical reasons
                        Pl = 0.5 * (Q + sqrtm_3x3(discr));
                        break;
                    }
                    case EnergyType::SymmetricDirichlet:
                    {
                        const Mat33 B = - Q;
                        const Scalar c = - wlBwlpmul;
                        spd_quartic_polynomial(B, c, Pl);
                        break;
                    }
                }
            }
            
            if(Pl.determinant()<=tol || Pl.trace()<=tol) {
                //This can happen due to numerical reasons, even if the input
                // is good.
                Pl = project_onto_spd(Pl, tol);
            }
            index_into_sym_3x3(Pl, P, i, n);
        }
    };
    
    //Should be if constexpr, but libigl doesn't support C++17
    if(parallelize) {
#ifdef OPENMP_AVAILABLE
#pragma omp parallel for
        for(Eigen::Index i=0; i<n; ++i) {
            solve_for_P(i);
        }
#endif
    } else {
        for(Eigen::Index i=0; i<n; ++i) {
            solve_for_P(i);
        }
    }
    
    //Should be if constexpr, but libigl doesn't support C++17
    if(parallelize) {
        //Turn on Eigen's parallelization.
        Eigen::setNbThreads(0);
    }
    
    parametrization_assert(P.array().isFinite().all());
}


// Explicit template instantiation
template void parametrization::argmin_P<false, (parametrization::EnergyType)0, 2, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&);
template void parametrization::argmin_P<false, (parametrization::EnergyType)0, 3, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&);
template void parametrization::argmin_P<false, (parametrization::EnergyType)1, 2, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&);
template void parametrization::argmin_P<false, (parametrization::EnergyType)1, 3, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&);
#ifdef OPENMP_AVAILABLE
template void parametrization::argmin_P<true, (parametrization::EnergyType)0, 2, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&);
template void parametrization::argmin_P<true, (parametrization::EnergyType)0, 3, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&);
template void parametrization::argmin_P<true, (parametrization::EnergyType)1, 2, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&);
template void parametrization::argmin_P<true, (parametrization::EnergyType)1, 3, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&);
#endif
