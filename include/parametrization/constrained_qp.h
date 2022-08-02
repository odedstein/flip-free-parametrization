#ifndef PARAMETRIZATION_CONSTRAINED_QP_H
#define PARAMETRIZATION_CONSTRAINED_QP_H

#include <Eigen/Core>
#include <Eigen/SparseLU>

namespace parametrization {

// This is a solver for quadratic programming problems with linear equality
//  constraints comparable to igl's min_quad_with_fixed
//
//
// Solve a quadratic problem with linear equality constraints
// argmin_z  0.5*z'*Q*z + c'*z
//           z(f) = g
//           A*z = b
// This supports solving multiple problems at the same time that have the same
//  Q, c, f, A but different g, b.
//
// Inputs:
//  template parameter Solver  be sure to specify what solver you want this
//                             function to use - the default only works for
//                             sparse double matrices.
//  Q  the sparse symmetric QP matrix (*must* be symmetric)
//  c  the linear part of the QP, a vector with either size 0 or the size of
//     the number of rows in Q, or a matrix with the number of rows of Q as
//     number of rows and as many columns as problems are solved at the same
//     time
//  f  fixed degrees of freedom in the solution as indices into z, an integer
//     vector. Can not contain duplicate entries.
//  g  what the degrees of freedom are fixed to, can be either a vector the size
//     of f, or a matrix with the size of f as number of rows and as many columns
//     as problems are solved at the same time
//  A  the sparse linear constraint matrix. All rows have to be linearly
//     independent (as well as independent from f).
//  b  what the sparse linear constraints should be equal to, can be either a
//     vector the size of f, or a matrix with the number of rows of A as number
//     of rows and as many columns as problems are solved at the same time
// Outputs:
//  return value  indicates optimization success or failure
//  z  solution to the quadratic problem
//
template <typename Solver = Eigen::SparseLU<Eigen::SparseMatrix<double>,
Eigen::COLAMDOrdering<int> >,
typename ScalarQ, typename Derivedc, typename Derivedf, typename Derivedg,
typename ScalarA, typename Derivedb, typename Derivedz>
bool
constrained_qp(const Eigen::SparseMatrix<ScalarQ>& Q,
               const Eigen::MatrixBase<Derivedc>& c,
               const Eigen::MatrixBase<Derivedf>& f,
               const Eigen::MatrixBase<Derivedg>& g,
               const Eigen::SparseMatrix<ScalarA>& A,
               const Eigen::MatrixBase<Derivedb>& b,
               Eigen::PlainObjectBase<Derivedz>& z);

//
// This is a structure that stores all precomputed data. It is templated by the
//  scalar type as well as the the solver chosen.
//  template parameter Scalar  the scalar type used for computation
//  template parameter Solver  if you plan to supply linear equality
//                             constraints, or your matrix is not positive def,
//                             make sure that this solver can handle indefinite
//                             problems.
//                             be sure to specify what solver you want this
//                             function to use - the default only works for
//                             sparse double matrices.
template <typename Scalar = double, typename Int = int,
typename Solver = Eigen::SparseLU<Eigen::SparseMatrix<double>,
Eigen::COLAMDOrdering<int> > >
struct ConstrainedQPPrecomputed;

//
// Precompute all data needed to then later solve the problem with varying data.
//
// Inputs:
//  Q  the sparse symmetric QP matrix (*must* be symmetric)
//  f  fixed degrees of freedom in the solution as indices into z, an integer
//     vector. Can not contain duplicate entries.
//  A  the sparse linear constraint matrix. All rows have to be linearly
//     independent (as well as independent from f).
// Outputs:
//  return value  indicates optimization success or failure
//  precomputed  a precomputed struct that can be used in constrained_qp_solve
//
template <typename Solver, typename ScalarQ, typename Derivedf,
typename ScalarA, typename ScalarPrec, typename IntPrec>
bool
constrained_qp_precompute(const Eigen::SparseMatrix<ScalarQ>& Q,
               const Eigen::MatrixBase<Derivedf>& f,
               const Eigen::SparseMatrix<ScalarA>& A,
               ConstrainedQPPrecomputed<ScalarPrec, IntPrec, Solver>& precomputed);

//
// Given precomputed data from constrained_qp_precompute, solve the problem
//
// Inputs:
//  precomputed  a precomputed struct returned by constrained_qp_precompute
//  c  the linear part of the QP, a vector with either size 0 or the size of
//     the number of rows in Q, or a matrix with the number of rows of Q as
//     number of rows and as many columns as problems are solved at the same
//     time
//  g  what the degrees of freedom are fixed to, can be either a vector the size
//     of f, or a matrix with the size of f as number of rows and as many columns
//     as problems are solved at the same time
//  b  what the sparse linear constraints should be equal to, can be either a
//     vector the size of f, or a matrix with the number of rows of A as number
//     of rows and as many columns as problems are solved at the same time
// Outputs:
//  return value  indicates optimization success or failure
//  z  solution to the quadratic problem
//
template <typename Solver, typename ScalarPrec, typename IntPrec,
typename Derivedc, typename Derivedg, typename Derivedb, typename Derivedz>
bool
constrained_qp_solve(const ConstrainedQPPrecomputed<ScalarPrec, IntPrec, Solver>& precomputed,
                   const Eigen::MatrixBase<Derivedc>& c,
                   const Eigen::MatrixBase<Derivedg>& g,
                   const Eigen::MatrixBase<Derivedb>& b,
                   Eigen::PlainObjectBase<Derivedz>& z);

//
// Actual implementaton of ConstrainedQPPrecomputed. A user should not need to
//  touch this ever beyond constructing an empty version for
//  constrained_qp_precompute.
//
template <typename Scalar, typename Int, typename Solver>
struct ConstrainedQPPrecomputed
{
    Solver solver;
    Eigen::Matrix<Int, Eigen::Dynamic, 1> f, fu; //known, unknown
    Eigen::SparseMatrix<Scalar> Y; //the matrix to multiply the RHS with
    Int n, m, l; //dofs in solution, fixed dofs, linear equality constraints.
};

}

#endif
