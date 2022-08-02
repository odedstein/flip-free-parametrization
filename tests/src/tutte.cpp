
// Tests for the function in parametrization/tutte.h

#include <catch2/catch.hpp>

#include "test_meshes.h"

#include <igl/readOBJ.h>

#include <parametrization/tutte.h>


TEST_CASE("Tutte")
{
    //The tutte function has its own two asserts checking for flipped
    // triangles and checking for optimization success at the end of the
    // function, so we don't need to do anything but run the function itself.
    const std::vector<std::string> testMeshes = disk_topology_meshes();
    for(const auto& mesh : testMeshes) {
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        igl::readOBJ(mesh, V, F);
        
        Eigen::MatrixXd W;
        REQUIRE(parametrization::tutte(V, F, W));
        
        //A few general asserts
        REQUIRE(W.rows()==V.rows());
        REQUIRE(W.cols()==2);
        REQUIRE(W.array().isFinite().all());
        REQUIRE((W.rowwise().norm().array() < 1. + 1e-12).all());
    }
}
