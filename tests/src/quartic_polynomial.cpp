
// Tests for the functions in parametrization/sqrtm.h

#include <catch2/catch.hpp>
#include <parametrization/quartic_polynomial.h>
#include <Eigen/Dense>


TEST_CASE("quartic polynomial")
{
    //Generate a random polynomial, and check if the returned solution is
    // actualla a solution.
    //This does NOT test for missed solutions!
    srand(0);
    for(int i=0; i<15000; ++i) {
        const Eigen::Vector4d c = Eigen::Vector4d::Random();
        const double x = parametrization::quartic_polynomial(c(0), c(1), c(2),
                                                             c(3));
        if(std::isfinite(x)) {
            const double pol = x*x*x*x + c(0)*x*x*x + c(1)*x*x + c(2)*x + c(3);
            REQUIRE(std::abs(pol) < 1e-14);
        }
    }
}
