
#include "argmin_U.h"

#include "parametrization_assert.h"
#include "sym_from_triu.h"
#include "mat_from_index.h"
#include "rotmat_from_complex_quat.h"

#include <igl/polar_svd.h>


template <bool parallelize, int idim,
typename DerivedP, typename DerivedGW, typename DerivedLambda,
typename DerivedU>
void
parametrization::argmin_U(const Eigen::MatrixBase<DerivedP>& P,
         const Eigen::MatrixBase<DerivedGW>& GW,
         const Eigen::MatrixBase<DerivedLambda>& Lambda,
         Eigen::PlainObjectBase<DerivedU>& U)
{
    using Scalar = typename DerivedU::Scalar;
    using Mat22 = Eigen::Matrix<Scalar, 2, 2>;
    using Mat33 = Eigen::Matrix<Scalar, 3, 3>;
    
    parametrization_assert(P.array().isFinite().all() &&
                           "Invalid entries in P");
    parametrization_assert(GW.array().isFinite().all() &&
                           "Invalid entries in GW");
    parametrization_assert(Lambda.array().isFinite().all() &&
                           "Invalid entries in Lambda");
    
    const int dim = idim>0 ? idim : GW.cols();
    parametrization_assert((dim==2 || dim==3) && "dim must be 2 or 3.");
    
    const Eigen::Index n = dim==2 ? P.rows()/3 : P.rows()/6;
    parametrization_assert(((dim==2 && 3*n==P.size()) ||
                            (dim==3 && 6*n==P.size())) &&
                           "P has the wrong size.");
    parametrization_assert(dim*n==GW.rows() && dim==GW.cols() &&
                           "GW does not correspond to P.");
    parametrization_assert(dim*n==Lambda.rows() && dim==Lambda.cols() &&
                           "Lambda does not correspond to P.");
    
    //Make sure OpenMP exists if parallelize is set to true
#ifndef OPENMP_AVAILABLE
    static_assert(!parallelize, "Can't parallelize without OpenMP.");
#endif
    
    //Should be if constexpr, but libigl doesn't support C++17
    if(parallelize) {
        //Turn off Eigen's parallelization.
        Eigen::setNbThreads(1);
    }
    
    //Construct the vector for U
    if(dim==2) {
        if(U.size()!=2*n) {
            U.resize(2*n,1);
        }
    } else if(U.size()!=4*n) {
        U.resize(4*n,1);
    }
    //Solve for U in each coordinate
    const auto solve_for_U = [&] (const Eigen::Index i) {
        if(dim==2) {
            //custom 2x2 procrustes solver
            const Mat22 Pl = sym_2d_from_index(P, i, n);
            const Mat22 Lambdal = mat_from_index_2x2(Lambda, i, n);
            const Mat22 GWl = mat_from_index_2x2(GW, i, n);
            const Mat22 GWpLl = GWl + Lambdal;
            
            //This function evaluates (2/mu*) the Lagrangian objective value.
            const auto obj = [&Pl, &GWpLl] (const Scalar re, const Scalar im) {
                const Mat22 Ul = rotmat_from_complex(re, im);
                return (GWpLl - Ul*Pl).squaredNorm();
            };
            
            //Construct the two critical points
            const Mat22 A = GWpLl * Pl;
            const Scalar y=A(0,1)-A(1,0), x=A(0,0)+A(1,1);
            const Scalar norm = sqrt(x*x + y*y);
            //This can go wrong if x and y are 0, account for this.
            const Scalar reCrit1 = norm==0 ? 0. : x/norm;
            const Scalar reCrit2 = -reCrit1;
            const Scalar imCrit1 = norm==0 ? 0. : -y/norm;
            const Scalar imCrit2 = -imCrit1;
            
            //Whichever critical point has the smaller function value is it.
            if(obj(reCrit1,imCrit1) < obj(reCrit2,imCrit2)) {
                U(2*i) = reCrit1;
                U(2*i+1) = imCrit1;
            } else {
                U(2*i) = reCrit2;
                U(2*i+1) = imCrit2;
            }
        } else {
            //igl 3x3 procrustes
            const Mat33 Pl = sym_3d_from_index(P, i, n);
            const Mat33 Lambdal = mat_from_index_3x3(Lambda, i, n);
            const Mat33 GWl = mat_from_index_3x3(GW, i, n);
            
            const Mat33 A = (GWl + Lambdal)*Pl;
            Mat33 Ul, dummy;
            igl::polar_svd(A, Ul, dummy);
            index_into_rotmat_3d(Ul, U, i, n);
        }
    };
    
    //Should be if constexpr, but libigl doesn't support C++17
    if(parallelize) {
#ifdef OPENMP_AVAILABLE
#pragma omp parallel for
        for(Eigen::Index i=0; i<n; ++i) {
            solve_for_U(i);
        }
#endif
    } else {
        for(Eigen::Index i=0; i<n; ++i) {
            solve_for_U(i);
        }
    }
    
    //Should be if constexpr, but libigl doesn't support C++17
    if(parallelize) {
        //Turn on Eigen's parallelization.
        Eigen::setNbThreads(0);
    }
    
    parametrization_assert(U.array().isFinite().all());
}


template <bool parallelize, int idim,
typename DerivedP, typename DerivedGW, typename DerivedLambda,
typename Derivedh, typename DerivedU>
void
parametrization::argmin_U(const Eigen::MatrixBase<DerivedP>& P,
         const Eigen::MatrixBase<DerivedGW>& GW,
         const Eigen::MatrixBase<DerivedLambda>& Lambda,
         const Eigen::MatrixBase<Derivedh>& h,
         Eigen::PlainObjectBase<DerivedU>& U)
{
    using Scalar = typename DerivedU::Scalar;
    using Mat22 = Eigen::Matrix<Scalar, 2, 2>;
    using Mat33 = Eigen::Matrix<Scalar, 3, 3>;
    
    parametrization_assert(P.array().isFinite().all() &&
                           "Invalid entries in P");
    parametrization_assert(GW.array().isFinite().all() &&
                           "Invalid entries in GW");
    parametrization_assert(Lambda.array().isFinite().all() &&
                           "Invalid entries in Lambda");
    parametrization_assert(U.array().isFinite().all() &&
                           "Invalid entries in U");
    
    const int dim = idim>0 ? idim : GW.cols();
    parametrization_assert((dim==2 || dim==3) && "dim must be 2 or 3.");
    
    const Eigen::Index n = dim==2 ? P.rows()/3 : P.rows()/6;
    parametrization_assert(((dim==2 && 3*n==P.size()) ||
                            (dim==3 && 6*n==P.size())) &&
                           "P has the wrong size.");
    parametrization_assert(dim*n==GW.rows() && dim==GW.cols() &&
                           "GW does not correspond to P.");
    parametrization_assert(dim*n==Lambda.rows() && dim==Lambda.cols() &&
                           "Lambda does not correspond to P.");
    parametrization_assert(((dim==2 && 2*n==U.size()) ||
                            (dim==3 && 4*n==U.size())) &&
                           "U does not correspond to P.");
    parametrization_assert(h.size()==n && "One proximal constant per face");
    parametrization_assert(h.minCoeff()>=0 &&
                           "Proximal constants are nonnegative.");
    
    //Make sure OpenMP exists if parallelize is set to true
#ifndef OPENMP_AVAILABLE
    static_assert(!parallelize, "Can't parallelize without OpenMP.");
#endif
    
    //Should be if constexpr, but libigl doesn't support C++17
    if(parallelize) {
        //Turn off Eigen's parallelization.
        Eigen::setNbThreads(1);
    }
    
    //Solve for U in each coordinate
    const auto solve_for_U = [&] (const Eigen::Index i) {
        if(dim==2) {
            //custom 2x2 procrustes solver
            const Mat22 Ulold = rotmat_2d_from_index(U, i, n);
            const Scalar hl = h(i);
            const Mat22 Pl = sym_2d_from_index(P, i, n);
            const Mat22 Lambdal = mat_from_index_2x2(Lambda, i, n);
            const Mat22 GWl = mat_from_index_2x2(GW, i, n);
            const Mat22 GWpLl = GWl + Lambdal;
            
            //This function evaluates the second derivative of the objective
            // function
            const auto d2 = [&Pl, &GWpLl, &Ulold, &hl]
            (const Scalar re, const Scalar im) {
                Mat22 Ulrot;
                Ulrot << -re, im, -im, -re;
                return -(Ulrot.array()*(
                                        (GWpLl*Pl) + hl*Ulold
                                        ).array()).sum();
            };
            
            //Construct the critical point
            const Mat22 A = GWpLl*Pl + hl*Ulold;
            const Scalar y=A(0,1)-A(1,0), x=A(0,0)+A(1,1);
            const Scalar norm = sqrt(x*x + y*y);
            Scalar reCrit = norm==0 ? 0. : x/norm;
            Scalar imCrit = norm==0 ? 0. : -y/norm;
            
            //Check whether the second derivative is positive (then it's a min),
            // otherwise negate reCrit and imCrit.
            if(d2(reCrit,imCrit) < 0) {
                reCrit = -reCrit;
                imCrit = -imCrit;
            }
            
            //Assign
            U(2*i) = reCrit;
            U(2*i+1) = imCrit;
        } else {
            //igl 3x3 procrustes
            const Mat33 Ulold = rotmat_3d_from_index(U, i, n);
            const Scalar hl = h(i);
            const Mat33 Pl = sym_3d_from_index(P, i, n);
            const Mat33 Lambdal = mat_from_index_3x3(Lambda, i, n);
            const Mat33 GWl = mat_from_index_3x3(GW, i, n);
            
            const Mat33 A = (GWl + Lambdal)*Pl + hl*Ulold;
            Mat33 Ul, dummy;
            igl::polar_svd(A, Ul, dummy);
            index_into_rotmat_3d(Ul, U, i, n);
        }
    };
    
    //Should be if constexpr, but libigl doesn't support C++17
    if(parallelize) {
#ifdef OPENMP_AVAILABLE
#pragma omp parallel for
        for(Eigen::Index i=0; i<n; ++i) {
            solve_for_U(i);
        }
#endif
    } else {
        for(Eigen::Index i=0; i<n; ++i) {
            solve_for_U(i);
        }
    }
    
    //Should be if constexpr, but libigl doesn't support C++17
    if(parallelize) {
        //Turn on Eigen's parallelization.
        Eigen::setNbThreads(0);
    }
    
    parametrization_assert(U.array().isFinite().all());
}


// Explicit template instantiation
template void parametrization::argmin_U<false, 2, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&);
template void parametrization::argmin_U<false, 2, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 2, 0, -1, 2>, Eigen::Matrix<double, -1, 2, 0, -1, 2>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 2, 0, -1, 2> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 2, 0, -1, 2> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&);
template void parametrization::argmin_U<false, 2, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&);
template void parametrization::argmin_U<false, 2, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 2, 0, -1, 2>, Eigen::Matrix<double, -1, 2, 0, -1, 2>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 2, 0, -1, 2> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 2, 0, -1, 2> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&);
template void parametrization::argmin_U<false, 3, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&);
template void parametrization::argmin_U<false, 3, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&);
#ifdef OPENMP_AVAILABLE
template void parametrization::argmin_U<true, 3, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&);
template void parametrization::argmin_U<true, 2, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&);
template void parametrization::argmin_U<true, 2, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 2, 0, -1, 2>, Eigen::Matrix<double, -1, 2, 0, -1, 2>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 2, 0, -1, 2> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 2, 0, -1, 2> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&);
template void parametrization::argmin_U<true, 2, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&);
template void parametrization::argmin_U<true, 2, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 2, 0, -1, 2>, Eigen::Matrix<double, -1, 2, 0, -1, 2>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 2, 0, -1, 2> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 2, 0, -1, 2> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&);
template void parametrization::argmin_U<true, 3, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&);
#endif
