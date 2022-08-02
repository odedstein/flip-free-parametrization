
#include "step_Lambda.h"

#include "parametrization_assert.h"
#include "rotmat_from_complex_quat.h"
#include "sym_from_triu.h"
#include "mat_from_index.h"

#include <Eigen/SparseCholesky>
#ifdef SUITESPARSE_AVAILABLE
#include <Eigen/CholmodSupport>
#endif


template <bool parallelize, int idim,
typename DerivedU, typename DerivedP, typename DerivedGW,
typename DerivedLambda>
void
parametrization::step_Lambda(const Eigen::MatrixBase<DerivedU>& U,
                             const Eigen::MatrixBase<DerivedP>& P,
                             const Eigen::MatrixBase<DerivedGW>& GW,
                             Eigen::PlainObjectBase<DerivedLambda>& Lambda)
{
    using Scalar = typename DerivedLambda::Scalar;
    using Mat22 = Eigen::Matrix<Scalar, 2, 2>;
    using Mat33 = Eigen::Matrix<Scalar, 3, 3>;
    
    
    parametrization_assert(U.array().isFinite().all() &&
                           "Invalid entries in U");
    parametrization_assert(P.array().isFinite().all() &&
                           "Invalid entries in P");
    parametrization_assert(GW.array().isFinite().all() &&
                           "Invalid entries in GW");
    
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
    
    //Make sure OpenMP exists if parallelize is set to true
#ifndef OPENMP_AVAILABLE
    static_assert(!parallelize, "Can't parallelize without OpenMP.");
#endif
    
    //Should be if constexpr, but libigl doesn't support C++17
    if(parallelize) {
        //Turn off Eigen's parallelization.
        Eigen::setNbThreads(1);
    }
    
    //The function which is looped to update Lambda
    const auto update_face = [&] (const Eigen::Index i) {
        if(dim==2) {
            const Mat22 Ul = rotmat_2d_from_index(U, i, n);
            const Mat22 Pl = sym_2d_from_index(P, i, n);
            const Mat22 GWl = mat_from_index_2x2(GW, i, n);
            //Is eval() here faster than not?
            add_into_mat_2x2((GWl - Ul*Pl).eval(), Lambda, i, n);
        } else {
            const Mat33 Ul = rotmat_3d_from_index(U, i, n);
            const Mat33 Pl = sym_3d_from_index(P, i, n);
            const Mat33 GWl = mat_from_index_3x3(GW, i, n);
            //Is eval() here faster than not?
            add_into_mat_3x3((GWl - Ul*Pl).eval(), Lambda, i, n);
        }
    };
    
    //Should be if constexpr, but libigl doesn't support C++17
    if(parallelize) {
#ifdef OPENMP_AVAILABLE
#pragma omp parallel for
        for(Eigen::Index i=0; i<n; ++i) {
            update_face(i);
        }
#endif
    } else {
        for(Eigen::Index i=0; i<n; ++i) {
            update_face(i);
        }
    }
    
    //Should be if constexpr, but libigl doesn't support C++17
    if(parallelize) {
        //Turn on Eigen's parallelization.
        Eigen::setNbThreads(0);
    }
    
    parametrization_assert(Lambda.array().isFinite().all());
}


//Explicit template instantiation
template void parametrization::step_Lambda<false, 2, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template void parametrization::step_Lambda<false, 3, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
#ifdef OPENMP_AVAILABLE
template void parametrization::step_Lambda<true, 2, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template void parametrization::step_Lambda<true, 3, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
#endif
