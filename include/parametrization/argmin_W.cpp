
#include "argmin_W.h"

#include "parametrization_assert.h"
#include "rotmat_from_complex_quat.h"
#include "sym_from_triu.h"
#include "mat_from_index.h"
#include "constrained_qp.h"

#include <Eigen/SparseLU>
#include <Eigen/SparseCholesky>
#ifdef SUITESPARSE_AVAILABLE
#include <Eigen/UmfPackSupport>
#include <Eigen/CholmodSupport>
#endif


template <bool parallelize, int idim,
typename LinearSolver, typename DerivedGtM, typename DerivedfixedTo,
typename Derivedbeq, typename DerivedU, typename DerivedP,
typename DerivedLambda, typename DerivedW>
bool
parametrization::argmin_W(const LinearSolver& GtMG,
         const Eigen::SparseMatrixBase<DerivedGtM>& GtM,
         const Eigen::MatrixBase<DerivedfixedTo>& fixedTo,
         const Eigen::MatrixBase<Derivedbeq>& beq,
         const Eigen::MatrixBase<DerivedU>& U,
         const Eigen::MatrixBase<DerivedP>& P,
         const Eigen::MatrixBase<DerivedLambda>& Lambda,
         Eigen::PlainObjectBase<DerivedW>& W)
{
    using Scalar = typename DerivedW::Scalar;
    using Mat22 = Eigen::Matrix<Scalar, 2, 2>;
    using Mat33 = Eigen::Matrix<Scalar, 3, 3>;
    using MatXDim = Eigen::Matrix<Scalar, Eigen::Dynamic, idim>;
    using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    
    
    parametrization_assert(U.array().isFinite().all() &&
                           "Invalid entries in U");
    parametrization_assert(P.array().isFinite().all() &&
                           "Invalid entries in P");
    parametrization_assert(Lambda.array().isFinite().all() &&
                           "Invalid entries in Lambda");
    
    const int dim = idim>0 ? idim : Lambda.cols();
    parametrization_assert((dim==2 || dim==3) && "dim must be 2 or 3.");
    
    const Eigen::Index n = dim==2 ? U.size()/2 : U.size()/4;
    parametrization_assert(((dim==2 && 2*n==U.size()) ||
                            (dim==3 && 4*n==U.size())) &&
                           "U has the wrong size");
    parametrization_assert(((dim==2 && 3*n==P.size()) ||
                            (dim==3 && 6*n==P.size())) &&
                           "P does not correspond to U.");
    parametrization_assert(dim*n==GtM.cols() && "GtM does not correspond to U.");
    parametrization_assert(dim*n==Lambda.rows() && dim==Lambda.cols() &&
                           "Lambda does not correspond to P.");
    parametrization_assert((fixedTo.size()==0 || fixedTo.cols()==dim)
                           && "fixedTo has the wrong dimensions.");
    parametrization_assert((beq.size()==0 || beq.cols()==dim)
                           && "beq has the wrong dimensions.");
    
    //Make sure OpenMP exists if parallelize is set to true
#ifndef OPENMP_AVAILABLE
    static_assert(!parallelize, "Can't parallelize without OpenMP.");
#endif
    
    //Should be if constexpr, but libigl doesn't support C++17
    if(parallelize) {
        //Turn off Eigen's parallelization.
        Eigen::setNbThreads(1);
    }
    
    //Construct a right-hand side
    MatXDim rhs;
    if(dim==2) {
        rhs.resize(2*n, 2);
    } else {
        rhs.resize(3*n, 3);
    }
    const auto construct_rhs = [&] (const Eigen::Index i) {
        if(dim==2) {
            const Mat22 Ul = rotmat_2d_from_index(U, i, n);
            const Mat22 Pl = sym_2d_from_index(P, i, n);
            const Mat22 Lambdal = mat_from_index_2x2(Lambda, i, n);
            
            const Mat22 rhsl = Lambdal - Ul*Pl;
            index_into_mat_2x2(rhsl, rhs, i, n);
        } else {
            const Mat33 Ul = rotmat_3d_from_index(U, i, n);
            const Mat33 Pl = sym_3d_from_index(P, i, n);
            const Mat33 Lambdal = mat_from_index_3x3(Lambda, i, n);
            
            const Mat33 rhsl = Lambdal - Ul*Pl;
            index_into_mat_3x3(rhsl, rhs, i, n);
        }
    };
    
    //Should be if constexpr, but libigl doesn't support C++17
    if(parallelize) {
#ifdef OPENMP_AVAILABLE
#pragma omp parallel for
        for(Eigen::Index i=0; i<n; ++i) {
            construct_rhs(i);
        }
#endif
    } else {
        for(Eigen::Index i=0; i<n; ++i) {
            construct_rhs(i);
        }
    }
    
    //Should be if constexpr, but libigl doesn't support C++17
    if(parallelize) {
        //Turn Eigen's parallelization back on again.
        Eigen::setNbThreads(0);
    }
    
    //Solve optimization problem
    const MatXDim c = GtM*rhs;
    return constrained_qp_solve(GtMG, c, fixedTo, beq, W);
}


// Explicit template instantiation
template bool parametrization::argmin_W<false, 2, parametrization::ConstrainedQPPrecomputed<double, int, Eigen::SimplicialLLT<Eigen::SparseMatrix<double, 0, int>, 1, Eigen::AMDOrdering<int> > >, Eigen::SparseMatrix<double, 1, int>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(parametrization::ConstrainedQPPrecomputed<double, int, Eigen::SimplicialLLT<Eigen::SparseMatrix<double, 0, int>, 1, Eigen::AMDOrdering<int> > > const&, Eigen::SparseMatrixBase<Eigen::SparseMatrix<double, 1, int> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template bool parametrization::argmin_W<false, 2, parametrization::ConstrainedQPPrecomputed<double, int, Eigen::SparseLU<Eigen::SparseMatrix<double, 0, int>, Eigen::COLAMDOrdering<int> > >, Eigen::SparseMatrix<double, 1, int>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(parametrization::ConstrainedQPPrecomputed<double, int, Eigen::SparseLU<Eigen::SparseMatrix<double, 0, int>, Eigen::COLAMDOrdering<int> > > const&, Eigen::SparseMatrixBase<Eigen::SparseMatrix<double, 1, int> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template bool parametrization::argmin_W<false, 3, parametrization::ConstrainedQPPrecomputed<double, int, Eigen::SimplicialLLT<Eigen::SparseMatrix<double, 0, int>, 1, Eigen::AMDOrdering<int> > >, Eigen::SparseMatrix<double, 1, int>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(parametrization::ConstrainedQPPrecomputed<double, int, Eigen::SimplicialLLT<Eigen::SparseMatrix<double, 0, int>, 1, Eigen::AMDOrdering<int> > > const&, Eigen::SparseMatrixBase<Eigen::SparseMatrix<double, 1, int> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template bool parametrization::argmin_W<false, 3, parametrization::ConstrainedQPPrecomputed<double, int, Eigen::SparseLU<Eigen::SparseMatrix<double, 0, int>, Eigen::COLAMDOrdering<int> > >, Eigen::SparseMatrix<double, 1, int>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(parametrization::ConstrainedQPPrecomputed<double, int, Eigen::SparseLU<Eigen::SparseMatrix<double, 0, int>, Eigen::COLAMDOrdering<int> > > const&, Eigen::SparseMatrixBase<Eigen::SparseMatrix<double, 1, int> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template bool parametrization::argmin_W<false, 2, parametrization::ConstrainedQPPrecomputed<double, int, Eigen::SparseLU<Eigen::SparseMatrix<double, 0, int>, Eigen::COLAMDOrdering<int> > >, Eigen::SparseMatrix<double, 0, int>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(parametrization::ConstrainedQPPrecomputed<double, int, Eigen::SparseLU<Eigen::SparseMatrix<double, 0, int>, Eigen::COLAMDOrdering<int> > > const&, Eigen::SparseMatrixBase<Eigen::SparseMatrix<double, 0, int> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template bool parametrization::argmin_W<false, 3, parametrization::ConstrainedQPPrecomputed<double, int, Eigen::SparseLU<Eigen::SparseMatrix<double, 0, int>, Eigen::COLAMDOrdering<int> > >, Eigen::SparseMatrix<double, 0, int>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(parametrization::ConstrainedQPPrecomputed<double, int, Eigen::SparseLU<Eigen::SparseMatrix<double, 0, int>, Eigen::COLAMDOrdering<int> > > const&, Eigen::SparseMatrixBase<Eigen::SparseMatrix<double, 0, int> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
#ifdef OPENMP_AVAILABLE
template bool parametrization::argmin_W<true, 2, parametrization::ConstrainedQPPrecomputed<double, int, Eigen::SimplicialLLT<Eigen::SparseMatrix<double, 0, int>, 1, Eigen::AMDOrdering<int> > >, Eigen::SparseMatrix<double, 1, int>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(parametrization::ConstrainedQPPrecomputed<double, int, Eigen::SimplicialLLT<Eigen::SparseMatrix<double, 0, int>, 1, Eigen::AMDOrdering<int> > > const&, Eigen::SparseMatrixBase<Eigen::SparseMatrix<double, 1, int> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template bool parametrization::argmin_W<true, 2, parametrization::ConstrainedQPPrecomputed<double, int, Eigen::SparseLU<Eigen::SparseMatrix<double, 0, int>, Eigen::COLAMDOrdering<int> > >, Eigen::SparseMatrix<double, 1, int>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(parametrization::ConstrainedQPPrecomputed<double, int, Eigen::SparseLU<Eigen::SparseMatrix<double, 0, int>, Eigen::COLAMDOrdering<int> > > const&, Eigen::SparseMatrixBase<Eigen::SparseMatrix<double, 1, int> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template bool parametrization::argmin_W<true, 3, parametrization::ConstrainedQPPrecomputed<double, int, Eigen::SimplicialLLT<Eigen::SparseMatrix<double, 0, int>, 1, Eigen::AMDOrdering<int> > >, Eigen::SparseMatrix<double, 1, int>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(parametrization::ConstrainedQPPrecomputed<double, int, Eigen::SimplicialLLT<Eigen::SparseMatrix<double, 0, int>, 1, Eigen::AMDOrdering<int> > > const&, Eigen::SparseMatrixBase<Eigen::SparseMatrix<double, 1, int> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template bool parametrization::argmin_W<true, 3, parametrization::ConstrainedQPPrecomputed<double, int, Eigen::SparseLU<Eigen::SparseMatrix<double, 0, int>, Eigen::COLAMDOrdering<int> > >, Eigen::SparseMatrix<double, 1, int>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(parametrization::ConstrainedQPPrecomputed<double, int, Eigen::SparseLU<Eigen::SparseMatrix<double, 0, int>, Eigen::COLAMDOrdering<int> > > const&, Eigen::SparseMatrixBase<Eigen::SparseMatrix<double, 1, int> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template bool parametrization::argmin_W<true, 2, parametrization::ConstrainedQPPrecomputed<double, int, Eigen::SparseLU<Eigen::SparseMatrix<double, 0, int>, Eigen::COLAMDOrdering<int> > >, Eigen::SparseMatrix<double, 0, int>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(parametrization::ConstrainedQPPrecomputed<double, int, Eigen::SparseLU<Eigen::SparseMatrix<double, 0, int>, Eigen::COLAMDOrdering<int> > > const&, Eigen::SparseMatrixBase<Eigen::SparseMatrix<double, 0, int> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template bool parametrization::argmin_W<true, 3, parametrization::ConstrainedQPPrecomputed<double, int, Eigen::SparseLU<Eigen::SparseMatrix<double, 0, int>, Eigen::COLAMDOrdering<int> > >, Eigen::SparseMatrix<double, 0, int>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(parametrization::ConstrainedQPPrecomputed<double, int, Eigen::SparseLU<Eigen::SparseMatrix<double, 0, int>, Eigen::COLAMDOrdering<int> > > const&, Eigen::SparseMatrixBase<Eigen::SparseMatrix<double, 0, int> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
#endif
#ifdef SUITESPARSE_AVAILABLE
template bool parametrization::argmin_W<false, 2, parametrization::ConstrainedQPPrecomputed<double, int, Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double, 0, int>, 1> >, Eigen::SparseMatrix<double, 1, int>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(parametrization::ConstrainedQPPrecomputed<double, int, Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double, 0, int>, 1> > const&, Eigen::SparseMatrixBase<Eigen::SparseMatrix<double, 1, int> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template bool parametrization::argmin_W<false, 2, parametrization::ConstrainedQPPrecomputed<double, int, Eigen::UmfPackLU<Eigen::SparseMatrix<double, 0, int> > >, Eigen::SparseMatrix<double, 1, int>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(parametrization::ConstrainedQPPrecomputed<double, int, Eigen::UmfPackLU<Eigen::SparseMatrix<double, 0, int> > > const&, Eigen::SparseMatrixBase<Eigen::SparseMatrix<double, 1, int> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template bool parametrization::argmin_W<false, 3, parametrization::ConstrainedQPPrecomputed<double, int, Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double, 0, int>, 1> >, Eigen::SparseMatrix<double, 1, int>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(parametrization::ConstrainedQPPrecomputed<double, int, Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double, 0, int>, 1> > const&, Eigen::SparseMatrixBase<Eigen::SparseMatrix<double, 1, int> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template bool parametrization::argmin_W<false, 3, parametrization::ConstrainedQPPrecomputed<double, int, Eigen::UmfPackLU<Eigen::SparseMatrix<double, 0, int> > >, Eigen::SparseMatrix<double, 1, int>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(parametrization::ConstrainedQPPrecomputed<double, int, Eigen::UmfPackLU<Eigen::SparseMatrix<double, 0, int> > > const&, Eigen::SparseMatrixBase<Eigen::SparseMatrix<double, 1, int> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
#ifdef OPENMP_AVAILABLE
template bool parametrization::argmin_W<true, 2, parametrization::ConstrainedQPPrecomputed<double, int, Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double, 0, int>, 1> >, Eigen::SparseMatrix<double, 1, int>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(parametrization::ConstrainedQPPrecomputed<double, int, Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double, 0, int>, 1> > const&, Eigen::SparseMatrixBase<Eigen::SparseMatrix<double, 1, int> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template bool parametrization::argmin_W<true, 2, parametrization::ConstrainedQPPrecomputed<double, int, Eigen::UmfPackLU<Eigen::SparseMatrix<double, 0, int> > >, Eigen::SparseMatrix<double, 1, int>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(parametrization::ConstrainedQPPrecomputed<double, int, Eigen::UmfPackLU<Eigen::SparseMatrix<double, 0, int> > > const&, Eigen::SparseMatrixBase<Eigen::SparseMatrix<double, 1, int> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template bool parametrization::argmin_W<true, 3, parametrization::ConstrainedQPPrecomputed<double, int, Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double, 0, int>, 1> >, Eigen::SparseMatrix<double, 1, int>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(parametrization::ConstrainedQPPrecomputed<double, int, Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double, 0, int>, 1> > const&, Eigen::SparseMatrixBase<Eigen::SparseMatrix<double, 1, int> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template bool parametrization::argmin_W<true, 3, parametrization::ConstrainedQPPrecomputed<double, int, Eigen::UmfPackLU<Eigen::SparseMatrix<double, 0, int> > >, Eigen::SparseMatrix<double, 1, int>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(parametrization::ConstrainedQPPrecomputed<double, int, Eigen::UmfPackLU<Eigen::SparseMatrix<double, 0, int> > > const&, Eigen::SparseMatrixBase<Eigen::SparseMatrix<double, 1, int> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
#endif
#endif
