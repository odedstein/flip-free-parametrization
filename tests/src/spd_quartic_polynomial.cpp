
// Tests for the functions in parametrization/sqrtm.h

#include <catch2/catch.hpp>
#include <parametrization/spd_quartic_polynomial.h>
#include <parametrization/symmetrize.h>
#include "compare_matrices.h"
#include <Eigen/Dense>


TEST_CASE("quartic polynomial 2d")
{
    srand(0);
    for(int i=0; i<15000; ++i) {
        Eigen::Matrix2d B = Eigen::Matrix2d::Random();
        parametrization::symmetrize(B);
        double c = Eigen::Vector2d::Random()(0) - 1.;
        //The c small, B large case should never happen, argmin_P prevents it.
        if(i<2500) {
            c *= 1e-6;
        }
        if(i>=0 && i<5000) {
            B *= 1e-6;
        }
        
        Eigen::Matrix2d P;
        parametrization::spd_quartic_polynomial(B, c, P);
        
        REQUIRE(matrix_equal(P*P*P*P + B*P*P*P +
                             c*Eigen::Matrix2d::Identity(),
                             Eigen::Matrix2d::Zero(), 1e-12));
    }
}

TEST_CASE("quartic polynomial 3d")
{
    srand(0);
    for(int i=0; i<15000; ++i) {
        Eigen::Matrix3d B = Eigen::Matrix3d::Random();
        parametrization::symmetrize(B);
        double c = Eigen::Vector2d::Random()(0) - 1.;
        //The c small, B large case should never happen, argmin_P prevents it.
        if(i<2500) {
            c *= 1e-6;
        }
        if(i>=0 && i<5000) {
            B *= 1e-6;
        }
        
        Eigen::Matrix3d P;
        parametrization::spd_quartic_polynomial(B, c, P);
        REQUIRE(matrix_equal(P*P*P*P + B*P*P*P +
                             c*Eigen::Matrix3d::Identity(),
                             Eigen::Matrix3d::Zero(), 1e-12));
    }
}
