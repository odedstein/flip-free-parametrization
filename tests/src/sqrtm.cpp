
// Tests for the functions in parametrization/sqrtm.h

#include <catch2/catch.hpp>
#include <parametrization/sqrtm.h>
#include <Eigen/Eigenvalues>

#include "compare_matrices.h"


TEST_CASE("sqrtm_2x2")
{
    //Generate a number of random 2-by-2 matrices, make sure they are SPD,
    // compute the square root, and see if they match.
    srand(0);
    for(int i=0; i<15000; ++i) {
        Eigen::Matrix2d C = Eigen::Matrix2d::Random().array();
        C = C.transpose().eval() + C;
        Eigen::Vector2cd evals = C.eigenvalues();
        if(std::imag(evals(0))!=0 || std::real(evals(0))<=0 ||
           std::imag(evals(1))!=0 || std::real(evals(1))<=0) {
            //Only want spd matrices
            continue;
        }
        
        const Eigen::Matrix2d U = parametrization::sqrtm_2x2(C);
        
        //U should be symmetric
        REQUIRE(U == U.transpose().eval());
        
        //U square should be C
        REQUIRE(matrix_equal(U*U, C, 1e-15));
    }
}


TEST_CASE("sqrtm_3x3")
{
    //Generate a number of random 3-by-3 matrices, make sure they are SPD,
    // compute the square root, and see if they match.
    srand(0);
    for(int i=0; i<15000; ++i) {
        Eigen::Matrix3d C = Eigen::Matrix3d::Random().array();
        C = C.transpose().eval() + C;
        Eigen::Vector3cd evals = C.eigenvalues();
        if(std::imag(evals(0))!=0 || std::real(evals(0))<=0 ||
           std::imag(evals(1))!=0 || std::real(evals(1))<=0 ||
           std::imag(evals(2))!=0 || std::real(evals(2))<=0) {
            //Only want spd matrices
            continue;
        }
        
        const Eigen::Matrix3d U = parametrization::sqrtm_3x3(C);
        
        //U should be symmetric
        REQUIRE(U == U.transpose().eval());
        
        //U square should be C
        REQUIRE(matrix_equal(U*U, C, 1e-14));
    }
}
