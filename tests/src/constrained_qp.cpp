
// Tests for the function in parametrization/constrained_qp.h
#include <catch2/catch.hpp>

#include "compare_matrices.h"

#include <parametrization/constrained_qp.h>

#include <igl/min_quad_with_fixed.h>

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#ifdef SUITESPARSE_AVAILABLE
#include <Eigen/CholmodSupport>
#endif

#include <iostream>


TEST_CASE("constrained_qp")
{
    //Solve the same problem with constrained_qp and min_quad_with fixed, the
    // two should approximately agree.
    srand(0);
    const auto rand_scalar = [] () { //Pseudorandom double between -1 and 1
        return
        -1. + static_cast<double>(rand())/(static_cast<double>(RAND_MAX)/2.);
    };
    const auto rand_pos_int = [] () {
        return (std::max)(0,rand());
    };
    
    std::vector<std::function<bool(Eigen::SparseMatrix<double>&,
                                         Eigen::MatrixXd&,
                                         Eigen::VectorXi&,
                                         Eigen::MatrixXd&,
                                         Eigen::SparseMatrix<double>&,
                                         Eigen::MatrixXd&,
                                         Eigen::MatrixXd&)> > solvers = {
        [] (auto& Q, auto& c, auto& f, auto& g, auto& A, auto& b, auto& zQp) {
            return parametrization::constrained_qp(Q, c, f, g, A, b, zQp);
        },
        [] (auto& Q, auto& c, auto& f, auto& g, auto& A, auto& b, auto& zQp) {
            return parametrization::constrained_qp
            <Eigen::SparseLU<Eigen::SparseMatrix<double>,
            Eigen::COLAMDOrdering<int> > >(Q, c, f, g, A, b, zQp);
        },
        [] (auto& Q, auto& c, auto& f, auto& g, auto& A, auto& b, auto& zQp) {
            return parametrization::constrained_qp
            <Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > >
            (Q, c, f, g, A, b, zQp);
        },
        [] (auto& Q, auto& c, auto& f, auto& g, auto& A, auto& b, auto& zQp) {
            return parametrization::constrained_qp
            <Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > >
            (Q, c, f, g, A, b, zQp);
        },
#ifdef SUITESPARSE_AVAILABLE
        [] (auto& Q, auto& c, auto& f, auto& g, auto& A, auto& b, auto& zQp) {
            return parametrization::constrained_qp
            <Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double> > >
            (Q, c, f, g, A, b, zQp);
        },
#endif
    };
    
    auto perform_comparison = [&rand_scalar, &rand_pos_int, &solvers]
    (const int solver, const int n, const int probs, const int m, const int l) {
        std::vector<Eigen::Triplet<double> > tripletListQ;
        for(int j=0; j<2*n; ++j) {
            const int row=rand_pos_int()%n, col=rand_pos_int()%n;
            const double val = rand_scalar();
            tripletListQ.emplace_back(row, col, val);
            tripletListQ.emplace_back(col, row, val);
        }
        for(int j=0; j<n; ++j) {
            tripletListQ.emplace_back(j, j, 3.*static_cast<double>(n));
        }
        Eigen::SparseMatrix<double> Q(n, n);
        Q.setFromTriplets(tripletListQ.begin(), tripletListQ.end());
        
        Eigen::MatrixXd c = Eigen::MatrixXd::Random(n,probs);
        
        Eigen::VectorXi f(m);
        for(int j=0; j<m; ++j) {
            f(j) = rand_pos_int() % n;
            for(int k=0; k<j; ++k) {
                if(f(k)==f(j)) {
                    --j;
                    break;
                }
            }
        }
        
        Eigen::MatrixXd g = Eigen::MatrixXd::Random(m,probs);
        
        std::vector<Eigen::Triplet<double> > tripletListA;
        for(int j=0; j<l; ++j) {
            std::vector<int> oldCols;
            for(int k=0; k<3; ++k) {
                //Make sure we never have a col repeat itself, for linear
                // independence.
                //Make sure not to constrain something already constrained by
                // f.
                int col = rand_pos_int()%n;
                if(std::find
                   (oldCols.begin(), oldCols.end(), col)!=oldCols.end() ||
                   std::find(f.data(), f.data()+m, col)!=f.data()+m) {
                    --k;
                    continue;
                }
                
                oldCols.push_back(col);
                tripletListA.emplace_back(j, col, rand_scalar());
            }
        }
        Eigen::SparseMatrix<double> A(l, n);
        A.setFromTriplets(tripletListA.begin(), tripletListA.end());
        
        Eigen::MatrixXd b = Eigen::MatrixXd::Random(l,probs);
        
        
        //Solve with constrained_qp and igl::min_quad_with_fixed, and compare.
        Eigen::MatrixXd zQp, zMqwf;
        //libigl's function can sometimes fail, no idea why. In this case, just
        // skip...
        REQUIRE(igl::min_quad_with_fixed(Q, c, f, g, A, b, l>0, zMqwf));
        REQUIRE(solvers[solver](Q, c, f, g, A, b, zQp));
        REQUIRE(matrix_equal(zQp, zMqwf, 1e-12));
    };
    
    for(int solver=0; solver<solvers.size(); ++solver) {
        for(int i=0; i<50; ++i) {
            if(solver<2) {
                //Linear equality constraints make the problem indefinite.
                //Thus, we only solve with the first solvers which can handle
                // this.
                perform_comparison(solver, 50, 3, 10, 10);
            }
        }
        for(int i=0; i<10; ++i) {
            if(solver<2) {
                //Linear equality constraints make the problem indefinite.
                //Thus, we only solve with the first solvers which can handle
                // this.
                perform_comparison(solver, 50, 2, 0, 10);
            }
        }
        for(int i=0; i<10; ++i) {
            perform_comparison(solver, 50, 2, 10, 0);
        }
        for(int i=0; i<10; ++i) {
            perform_comparison(solver, 50, 2, 0, 0);
        }
        for(int i=0; i<10; ++i) {
            perform_comparison(solver, 50, 1, 0, 0);
        }
    }
}

