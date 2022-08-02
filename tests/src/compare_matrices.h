#ifndef PARAMETRIZATION_TESTS_COMPARE_MATRICES_H
#define PARAMETRIZATION_TESTS_COMPARE_MATRICES_H

#include <Eigen/Core>
#include <Eigen/Sparse>

//Compare two Eigen matrices with a specified relative tolerance
template <typename DerivedA, typename DerivedB>
inline bool matrix_equal(const Eigen::MatrixBase<DerivedA>& A,
                  const Eigen::MatrixBase<DerivedB>& B,
                  const typename DerivedA::Scalar tol)
{
    using Scalar = typename DerivedA::Scalar;
    return ((A-B).squaredNorm() / static_cast<Scalar>(A.size())) < tol*tol;
}
template <typename ScalarA, typename ScalarB>
inline bool matrix_equal(const Eigen::SparseMatrix<ScalarA>& A,
                  const Eigen::SparseMatrix<ScalarB>& B,
                  const ScalarA tol)
{
    return ((A-B).squaredNorm() / static_cast<ScalarA>(A.size())) < tol*tol;
}

#endif
