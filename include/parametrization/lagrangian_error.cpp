
#include "lagrangian_error.h"

#include "parametrization_assert.h"
#include "rotmat_from_complex_quat.h"
#include "sym_from_triu.h"
#include "mat_from_index.h"

#include <iostream>


template <bool parallelize, int idim,
typename DerivedU, typename DerivedU0, typename DerivedP, typename DerivedP0,
typename DerivedGW, typename DerivedGW0, typename Derivedmu,
typename Derivedh, typename ScalareP, typename ScalareD, typename DerivedePs,
typename DerivedeDs>
void
parametrization::lagrangian_errors(const Eigen::MatrixBase<DerivedU>& U,
                  const Eigen::MatrixBase<DerivedU0>& U0,
                  const Eigen::MatrixBase<DerivedP>& P,
                  const Eigen::MatrixBase<DerivedP0>& P0,
                  const Eigen::MatrixBase<DerivedGW>& GW,
                  const Eigen::MatrixBase<DerivedGW0>& GW0,
                  const Eigen::MatrixBase<Derivedmu>& mu,
                  const Eigen::MatrixBase<Derivedh>& h,
                  ScalareP& eP,
                  ScalareD& eD,
                  Eigen::PlainObjectBase<DerivedePs>& ePs,
                  Eigen::PlainObjectBase<DerivedeDs>& eDs)
{
    using Scalar = typename DerivedU::Scalar;
    using Mat22 = Eigen::Matrix<Scalar, 2, 2>;
    using Mat33 = Eigen::Matrix<Scalar, 3, 3>;
    const Scalar tol = std::numeric_limits<Scalar>::epsilon();
    
    parametrization_assert(U.array().isFinite().all() &&
                           "Invalid entries in U");
    parametrization_assert(U0.array().isFinite().all() &&
                           "Invalid entries in U0");
    parametrization_assert(P.array().isFinite().all() &&
                           "Invalid entries in P");
    parametrization_assert(P0.array().isFinite().all() &&
                           "Invalid entries in P0");
    parametrization_assert(GW.array().isFinite().all() &&
                           "Invalid entries in GW");
    parametrization_assert(GW0.array().isFinite().all() &&
                           "Invalid entries in GW0");
    parametrization_assert(mu.array().isFinite().all() &&
                           "Invalid entries in mu");
    parametrization_assert(h.array().isFinite().all() &&
                           "Invalid entries in h");
    
    const int dim = idim>0 ? idim : GW.cols();
    parametrization_assert((dim==2 || dim==3) && "dim must be 2 or 3.");
    
    const Eigen::Index n = dim==2 ? U.size()/2 : U.size()/4;
    parametrization_assert(((dim==2 && 2*n==U.size()) ||
                            (dim==3 && 4*n==U.size())) &&
                           "U has the wrong size");
    parametrization_assert(((dim==2 && 2*n==U0.size()) ||
                            (dim==3 && 4*n==U0.size())) &&
                           "U0 has the wrong size");
    parametrization_assert(((dim==2 && 3*n==P.size()) ||
                            (dim==3 && 6*n==P.size())) &&
                           "P does not correspond to U.");
    parametrization_assert(((dim==2 && 3*n==P0.size()) ||
                            (dim==3 && 6*n==P0.size())) &&
                           "P0 does not correspond to U.");
    parametrization_assert(dim*n==GW.rows() && dim==GW.cols() &&
                           "GW does not correspond to U.");
    parametrization_assert(dim*n==GW0.rows() && dim==GW0.cols() &&
                           "GW0 does not correspond to U.");
    parametrization_assert(n==mu.size() && "mu does not correspond to U.");
    parametrization_assert(n==h.size() && "h does not correspond to U.");
    parametrization_assert(mu.minCoeff()>tol &&
                           "Penalties need to be positive.");
    parametrization_assert(h.minCoeff()>tol &&
                           "Proximal penalties need to be positive.");
    
    
    //Make sure OpenMP exists if parallelize is set to true
#ifndef OPENMP_AVAILABLE
    static_assert(!parallelize, "Can't parallelize without OpenMP.");
#endif
    
    //Should be if constexpr, but libigl doesn't support C++17
    if(parallelize) {
        //Turn off Eigen's parallelization.
        Eigen::setNbThreads(1);
    }
    
    //Construct the output
    if(ePs.size() != n) {
        ePs.resize(n,1);
    }
    if(eDs.size() != n) {
        eDs.resize(n,1);
    }
    
    //Compute the primal and dual errors
    const auto compute_errors = [&]
    (const Eigen::Index i, ScalareP& eP, ScalareD& eD) {
        if(dim==2) {
            const Mat22 U0l = rotmat_2d_from_index(U0, i, n);
            const Mat22 Ul = rotmat_2d_from_index(U, i, n);
            const Mat22 P0l = sym_2d_from_index(P0, i, n);
            const Mat22 Pl = sym_2d_from_index(P, i, n);
            const Mat22 GW0l = mat_from_index_2x2(GW0, i, n);
            const Mat22 GWl = mat_from_index_2x2(GW, i, n);
            
            //It is ok to add to eP and eD here, even in a parallel loop, because
            // of the reduction later
            const ScalareP ePl = (GWl - Ul*Pl).squaredNorm();
            eP += ePl;
            ePs(i) = ePl;
            const ScalareD eDl = (mu(i)*mu(i)) *
            ((GW0l - GWl).squaredNorm()
             //+ (P0l - Pl).squaredNorm()
             //+ (h(i)*h(i))*(U0l - Ul).squaredNorm()
             );
            eD += eDl;
            eDs(i) = eDl;
        } else {
            const Mat33 U0l = rotmat_3d_from_index(U0, i, n);
            const Mat33 Ul = rotmat_3d_from_index(U, i, n);
            const Mat33 P0l = sym_3d_from_index(P0, i, n);
            const Mat33 Pl = sym_3d_from_index(P, i, n);
            const Mat33 GW0l = mat_from_index_3x3(GW0, i, n);
            const Mat33 GWl = mat_from_index_3x3(GW, i, n);
            
            //It is ok to add to eP and eD here, even in a parallel loop, because
            // of the reduction later
            const ScalareP ePl = (GWl - Ul*Pl).squaredNorm();
            eP += ePl;
            ePs(i) = ePl;
            const ScalareD eDl = (mu(i)*mu(i)) *
            ((GW0l - GWl).squaredNorm()
             //+ (P0l - Pl).squaredNorm()
             //+ (h(i)*h(i))*(U0l - Ul).squaredNorm()
             );
            eD += eDl;
            eDs(i) = eDl;
        }
    };
    
    //Should be if constexpr, but libigl doesn't support C++17
    eP = 0.;
    eD = 0.;
    if(parallelize) {
#ifdef OPENMP_AVAILABLE
#pragma omp parallel for reduction(+:eP,eD)
        for(Eigen::Index i=0; i<n; ++i) {
            compute_errors(i, eP, eD);
        }
#endif
    } else {
        for(Eigen::Index i=0; i<n; ++i) {
            compute_errors(i, eP, eD);
        }
    }
    
    //Should be if constexpr, but libigl doesn't support C++17
    if(parallelize) {
        //Turn on Eigen's parallelization.
        Eigen::setNbThreads(0);
    }
}


template <bool parallelize, int idim,
typename DerivedU, typename DerivedU0, typename DerivedP, typename DerivedP0,
typename DerivedGW, typename DerivedGW0, typename Derivedmu,
typename Derivedh, typename ScalareP, typename ScalareD>
void
parametrization::lagrangian_error(const Eigen::MatrixBase<DerivedU>& U,
                                  const Eigen::MatrixBase<DerivedU0>& U0,
                                  const Eigen::MatrixBase<DerivedP>& P,
                                  const Eigen::MatrixBase<DerivedP0>& P0,
                                  const Eigen::MatrixBase<DerivedGW>& GW,
                                  const Eigen::MatrixBase<DerivedGW0>& GW0,
                                  const Eigen::MatrixBase<Derivedmu>& mu,
                                  const Eigen::MatrixBase<Derivedh>& h,
                                  ScalareP& eP,
                                  ScalareD& eD)
{
    using Scalar = typename DerivedU::Scalar;
    using Mat22 = Eigen::Matrix<Scalar, 2, 2>;
    using Mat33 = Eigen::Matrix<Scalar, 3, 3>;
    const Scalar tol = std::numeric_limits<Scalar>::epsilon();
    
    parametrization_assert(U.array().isFinite().all() &&
                           "Invalid entries in U");
    parametrization_assert(U0.array().isFinite().all() &&
                           "Invalid entries in U0");
    parametrization_assert(P.array().isFinite().all() &&
                           "Invalid entries in P");
    parametrization_assert(P0.array().isFinite().all() &&
                           "Invalid entries in P0");
    parametrization_assert(GW.array().isFinite().all() &&
                           "Invalid entries in GW");
    parametrization_assert(GW0.array().isFinite().all() &&
                           "Invalid entries in GW0");
    parametrization_assert(mu.array().isFinite().all() &&
                           "Invalid entries in mu");
    parametrization_assert(h.array().isFinite().all() &&
                           "Invalid entries in h");
    
    const int dim = idim>0 ? idim : GW.cols();
    parametrization_assert((dim==2 || dim==3) && "dim must be 2 or 3.");
    
    const Eigen::Index n = dim==2 ? U.size()/2 : U.size()/4;
    parametrization_assert(((dim==2 && 2*n==U.size()) ||
                            (dim==3 && 4*n==U.size())) &&
                           "U has the wrong size");
    parametrization_assert(((dim==2 && 2*n==U0.size()) ||
                            (dim==3 && 4*n==U0.size())) &&
                           "U0 has the wrong size");
    parametrization_assert(((dim==2 && 3*n==P.size()) ||
                            (dim==3 && 6*n==P.size())) &&
                           "P does not correspond to U.");
    parametrization_assert(((dim==2 && 3*n==P0.size()) ||
                            (dim==3 && 6*n==P0.size())) &&
                           "P0 does not correspond to U.");
    parametrization_assert(dim*n==GW.rows() && dim==GW.cols() &&
                           "GW does not correspond to U.");
    parametrization_assert(dim*n==GW0.rows() && dim==GW0.cols() &&
                           "GW0 does not correspond to U.");
    parametrization_assert(n==mu.size() && "mu does not correspond to U.");
    parametrization_assert(n==h.size() && "h does not correspond to U.");
    parametrization_assert(mu.minCoeff()>tol &&
                           "Penalties need to be positive.");
    parametrization_assert(h.minCoeff()>tol &&
                           "Proximal penalties need to be positive.");
    
    //Make sure OpenMP exists if parallelize is set to true
#ifndef OPENMP_AVAILABLE
    static_assert(!parallelize, "Can't parallelize without OpenMP.");
#endif
    
    //Should be if constexpr, but libigl doesn't support C++17
    if(parallelize) {
        //Turn off Eigen's parallelization.
        Eigen::setNbThreads(1);
    }
    
    //Compute the primal and dual errors
    const auto compute_errors = [&]
    (const Eigen::Index i, ScalareP& eP, ScalareD& eD) {
        if(dim==2) {
            const Mat22 U0l = rotmat_2d_from_index(U0, i, n);
            const Mat22 Ul = rotmat_2d_from_index(U, i, n);
            const Mat22 P0l = sym_2d_from_index(P0, i, n);
            const Mat22 Pl = sym_2d_from_index(P, i, n);
            const Mat22 GW0l = mat_from_index_2x2(GW0, i, n);
            const Mat22 GWl = mat_from_index_2x2(GW, i, n);
            
            //It is ok to add to eP and eD here, even in a parallel loop, because
            // of the reduction later
            eP += (GWl - Ul*Pl).squaredNorm();
            eD += (mu(i)*mu(i)) * ((GW0l - GWl).squaredNorm()
                                   //+ (P0l - Pl).squaredNorm()
                                   //+ (h(i))*(U0l - Ul).squaredNorm()
                                   );
        } else {
            const Mat33 U0l = rotmat_3d_from_index(U0, i, n);
            const Mat33 Ul = rotmat_3d_from_index(U, i, n);
            const Mat33 P0l = sym_3d_from_index(P0, i, n);
            const Mat33 Pl = sym_3d_from_index(P, i, n);
            const Mat33 GW0l = mat_from_index_3x3(GW0, i, n);
            const Mat33 GWl = mat_from_index_3x3(GW, i, n);
            
            //It is ok to add to eP and eD here, even in a parallel loop, because
            // of the reduction later
            eP += (GWl - Ul*Pl).squaredNorm();
            eD += (mu(i)*mu(i)) * ((GW0l - GWl).squaredNorm()
                                   //+ (P0l - Pl).squaredNorm()
                                   //+ (h(i)*h(i))*(U0l - Ul).squaredNorm()
                                   );
        }
    };
    
    //Should be if constexpr, but libigl doesn't support C++17
    eP = 0.;
    eD = 0.;
    if(parallelize) {
#ifdef OPENMP_AVAILABLE
#pragma omp parallel for reduction(+:eP,eD)
        for(Eigen::Index i=0; i<n; ++i) {
            compute_errors(i, eP, eD);
        }
#endif
    } else {
        for(Eigen::Index i=0; i<n; ++i) {
            compute_errors(i, eP, eD);
        }
    }
    
    //Should be if constexpr, but libigl doesn't support C++17
    if(parallelize) {
        //Turn on Eigen's parallelization.
        Eigen::setNbThreads(0);
    }
    
    parametrization_assert(std::isfinite(eP));
    parametrization_assert(std::isfinite(eD));
}


// Explicit template instantiation
template void parametrization::lagrangian_errors<false, 2, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, double, double, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, double&, double&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&);
template void parametrization::lagrangian_error<false, 2, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, double, double>(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, double&, double&);
template void parametrization::lagrangian_errors<false, 3, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, double, double, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, double&, double&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&);
template void parametrization::lagrangian_error<false, 3, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, double, double>(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, double&, double&);
#ifdef OPENMP_AVAILABLE
template void parametrization::lagrangian_errors<true, 2, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, double, double, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, double&, double&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&);
template void parametrization::lagrangian_error<true, 2, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, double, double>(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, double&, double&);
template void parametrization::lagrangian_errors<true, 3, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, double, double, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, double&, double&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&);
template void parametrization::lagrangian_error<true, 3, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, double, double>(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, double&, double&);
#endif
