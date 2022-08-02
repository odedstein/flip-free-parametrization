
// Tests for the function in parametrization/energy.h
#include <catch2/catch.hpp>

#include "test_meshes.h"

#include <parametrization/tutte.h>
#include <parametrization/energy.h>
#include <parametrization/uv_to_jacobian.h>
#include <parametrization/polar_decomposition.h>

#include <igl/readOBJ.h>
#include <igl/doublearea.h>


TEST_CASE("energy symmgrad 2D")
{
    //The known parametrization energy for the test meshes' Tutte
    // parametrization.
    const std::vector<double> knownEnergy = {3.93850094434697606260443,
        5.53784741345229392095462,
        6262993.51004086062312126};
    const std::vector<std::string> testMeshes = disk_topology_meshes();
    assert(knownEnergy.size()==testMeshes.size() &&
           "Number of known energies does not match number of meshes.");
    
    for(int i=0; i<testMeshes.size(); ++i) {
        const std::string& mesh = testMeshes[i];
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        igl::readOBJ(mesh, V, F);
        
        //Compute the Tutte parametrization.
        Eigen::MatrixXd W;
        parametrization::tutte(V, F, W);
        
        //Compute the Jacobians of the Tutte parametrization.
        Eigen::SparseMatrix<double> G;
        parametrization::uv_to_jacobian(V, F, G);
        Eigen::MatrixXd J = G*W;
        
        //Compute the triangle areas
        Eigen::VectorXd A;
        igl::doublearea(V, F, A);
        A *= 0.5;
        
        //Compute the parametrization energy and check with the known value.
        Eigen::VectorXd U, P;
        parametrization::polar_decomposition(J, U, P);
        const double E = parametrization::energy_from_P
        <parametrization::EnergyType::SymmetricGradient,2>(P, A);
//        std::cout << "kE: " << std::setprecision(24) << knownEnergy[i] << std::endl;
//        std::cout << "E: " << std::setprecision(24) << E << std::endl << std::endl;
        REQUIRE(std::abs(E - knownEnergy[i]) < 1e-10*knownEnergy[i]);
        const double E2 = parametrization::energy_from_GW
        <parametrization::EnergyType::SymmetricGradient,2>(J, A);
        REQUIRE(std::abs(E2 - knownEnergy[i]) < 1e-10*knownEnergy[i]);
    }
}


TEST_CASE("energy symmdir 2D")
{
    //The known parametrization energy for the test meshes' Tutte
    // parametrization.
    const std::vector<double> knownEnergy = {17.4898339581867148240235,
        63.558385950758214733014,
        1854284276831539.5};
    const std::vector<std::string> testMeshes = disk_topology_meshes();
    assert(knownEnergy.size()==testMeshes.size() &&
           "Number of known energies does not match number of meshes.");
    
    for(int i=0; i<testMeshes.size(); ++i) {
        const std::string& mesh = testMeshes[i];
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        igl::readOBJ(mesh, V, F);
        
        //Compute the Tutte parametrization.
        Eigen::MatrixXd W;
        parametrization::tutte(V, F, W);
        
        //Compute the Jacobians of the Tutte parametrization.
        Eigen::SparseMatrix<double> G;
        parametrization::uv_to_jacobian(V, F, G);
        Eigen::MatrixXd J = G*W;
        
        //Compute the triangle areas
        Eigen::VectorXd A;
        igl::doublearea(V, F, A);
        A *= 0.5;
        
        //Compute the parametrization energy and check with the known value.
        Eigen::VectorXd U, P;
        parametrization::polar_decomposition(J, U, P);
        const double E = parametrization::energy_from_P
        <parametrization::EnergyType::SymmetricDirichlet,2>(P, A);
//        std::cout << "kE: " << std::setprecision(24) << knownEnergy[i] << std::endl;
//        std::cout << "E: " << std::setprecision(24) << E << std::endl << std::endl;
        REQUIRE(std::abs(E - knownEnergy[i]) < 1e-10*knownEnergy[i]);
        const double E2 = parametrization::energy_from_GW
        <parametrization::EnergyType::SymmetricDirichlet,2>(J, A);
        REQUIRE(std::abs(E2 - knownEnergy[i]) < 1e-10*knownEnergy[i]);
    }
}

