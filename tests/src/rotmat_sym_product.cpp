
// Tests for the functions in parametrization/rotmat_sym_product.h

#include <catch2/catch.hpp>

#include "compare_matrices.h"

#include <parametrization/rotmat_sym_product.h>
#include <parametrization/sym_from_triu.h>
#include <parametrization/rotmat_from_complex_quat.h>
#include <parametrization/mat_from_index.h>

TEST_CASE("rotmat_sym_product 2x2")
{
    //Generate a few random examples of U and P, and check if their sym product
    // is correct
    srand(0);
    for(int i=0; i<500; ++i) {
        const int n=50;
        Eigen::VectorXd U = Eigen::VectorXd::Random(2*n);
        for(int i=0; i<n; ++i) {
            U.segment(2*i,2).normalize();
        }
        const Eigen::VectorXd P = Eigen::VectorXd::Random(3*n);
        const Eigen::MatrixXd J = parametrization::rotmat_sym_product_2d(U,P);
        
        for(int j=0; j<n; ++j) {
            const Eigen::Matrix2d Jl = parametrization::mat_from_index(J, j, n);
            const Eigen::Matrix2d Ul = parametrization::rotmat_from_index(U, j,
                                                                          n);
            const Eigen::Matrix2d Pl = parametrization::sym_from_index(P, j, n);
            
            REQUIRE(matrix_equal(Ul*Pl, Jl, 1e-12));
        }
    }
}


TEST_CASE("rotmat_sym_product 3x3")
{
    //Generate a few random examples of U and P, and check if their sym product
    // is correct
    srand(0);
    for(int i=0; i<500; ++i) {
        const int n=50;
        Eigen::VectorXd U = Eigen::VectorXd::Random(4*n);
        for(int i=0; i<n; ++i) {
            U.segment(4*i,4).normalize();
        }
        const Eigen::VectorXd P = Eigen::VectorXd::Random(6*n);
        const Eigen::MatrixXd J = parametrization::rotmat_sym_product_3d(U,P);
        
        for(int j=0; j<n; ++j) {
            const Eigen::Matrix3d Jl = parametrization::mat_from_index(J, j, n);
            const Eigen::Matrix3d Ul = parametrization::rotmat_from_index(U, j,
                                                                          n);
            const Eigen::Matrix3d Pl = parametrization::sym_from_index(P, j, n);
            
            REQUIRE(matrix_equal(Ul*Pl, Jl, 1e-12));
        }
    }
}
