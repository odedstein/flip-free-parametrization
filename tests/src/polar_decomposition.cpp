
// Tests for the functions in parametrization/rotmat_sym_product.h

#include <catch2/catch.hpp>

#include "compare_matrices.h"

#include <parametrization/polar_decomposition.h>
#include <parametrization/rotmat_sym_product.h>
#include <parametrization/mat_from_index.h>
#include <parametrization/rotmat_from_complex_quat.h>
#include <parametrization/sym_from_triu.h>
#include <Eigen/Eigenvalues>


TEST_CASE("polar_decomposition 2x2")
{
    //Generate a few random examples of J, compute the polar decomposition, and
    // check whether the decomposed matrices are ok and recombine into J.
    srand(0);
    for(int i=0; i<500; ++i) {
        const int n=50;
        const Eigen::MatrixXd J = Eigen::MatrixXd::Random(2*n,2);
        Eigen::VectorXd U, P;
        parametrization::polar_decomposition(J, U, P);
        
        for(int j=0; j<n; ++j) {
            //Is U a rotation matrix?
            const Eigen::Matrix2d Ul =
            parametrization::rotmat_from_index(U, j, n);
            REQUIRE(matrix_equal(Ul*Ul.transpose(), Eigen::Matrix2d::Identity(),
                                 1e-12));
            REQUIRE(std::abs(Ul.determinant()-1.) < 1e-12);
            
            //Is P spd?
            const Eigen::Matrix2d Pl = parametrization::sym_from_index(P, j, n);
            REQUIRE(Pl == Pl.transpose());
            Eigen::Vector2cd evals = Pl.eigenvalues();
            REQUIRE(((std::real(evals(0))>=0) && (std::real(evals(1))>=0)));
            
            //If J has positive determinant, is J == U*P?
            const Eigen::Matrix2d Jl = parametrization::mat_from_index(J, j, n);
            if(Jl.determinant() > 0) {
                REQUIRE(matrix_equal(Ul*Pl, Jl, 1e-12));
            }
        }
    }
}


TEST_CASE("polar_decomposition 3x3")
{
    //Generate a few random examples of J, compute the polar decomposition, and
    // check whether the decomposed matrices are ok and recombine into J.
    srand(0);
    for(int i=0; i<500; ++i) {
        const int n=50;
        const Eigen::MatrixXd J = Eigen::MatrixXd::Random(3*n,3);
        Eigen::VectorXd U, P;
        parametrization::polar_decomposition(J, U, P);
        
        for(int j=0; j<n; ++j) {
            //Is U a rotation matrix?
            const Eigen::Matrix3d Ul =
            parametrization::rotmat_from_index(U, j, n);
            REQUIRE(matrix_equal(Ul*Ul.transpose(), Eigen::Matrix3d::Identity(),
                                 1e-12));
            REQUIRE(std::abs(Ul.determinant()-1.) < 1e-12);
            
            //Is P spd?
            const Eigen::Matrix3d Pl = parametrization::sym_from_index(P, j, n);
            REQUIRE(Pl == Pl.transpose());
            Eigen::Vector3cd evals = Pl.eigenvalues();
            REQUIRE(((std::real(evals(0))>=0) && (std::real(evals(1))>=0) &&
                     (std::real(evals(2))>=0)));
            
            //If J has positive determinant, is J == U*P?
            const Eigen::Matrix3d Jl = parametrization::mat_from_index(J, j, n);
            if(Jl.determinant() > 0) {
                REQUIRE(matrix_equal(Ul*Pl, Jl, 1e-12));
            }
        }
    }
}

