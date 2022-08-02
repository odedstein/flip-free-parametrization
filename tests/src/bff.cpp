
// Tests for the function in parametrization/tutte.h

#include <catch2/catch.hpp>

#include "test_meshes.h"

#include <igl/readOBJ.h>

#include <parametrization/bff.h>


TEST_CASE("BFF")
{
    // BFF only works if SuiteSparse is available.
#ifdef SUITESPARSE_AVAILABLE
    const std::vector<std::string> testMeshes = disk_topology_meshes();
    for(const auto& mesh : testMeshes) {
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        igl::readOBJ(mesh, V, F);
        
        Eigen::MatrixXd W;
        REQUIRE(parametrization::bff(V, F, W));
        
        //A few general asserts
        REQUIRE(W.rows()==V.rows());
        REQUIRE(W.cols()==2);
        REQUIRE(W.array().isFinite().all());
    }
#endif
}
