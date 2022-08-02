
#include <parametrization/tutte.h>
#include <parametrization/map_to.h>

#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/flipped_triangles.h>
#include <igl/get_seconds.h>
#include <igl/arap.h>

#include <iostream>
#include <fstream>
#include <vector>

//This function computes a number of UV maps, with ARAP (initialized with
// Tutte), and the unflips any remaining triangles with map_to.

int main(int argc, char *argv[]) {
    namespace par = parametrization;
#ifdef OPENMP_AVAILABLE
    constexpr bool parallelize = true;
#else
    constexpr bool parallelize = false;
#endif
    
    std::vector<std::string> meshes = {"suzanne.obj", "turbine.obj"};
    
    for(const auto& mesh : meshes) {
        std::cout << "Computing UV maps for mesh " << mesh << "." << std::endl;
        
        //Create log file
        std::ofstream log;
        log.open(mesh + std::string(".log.txt"));
        log << mesh << std::endl;
        
        //Load mesh
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        const std::string meshPath = std::string(MESH_DIRECTORY) + mesh;
        igl::readOBJ(meshPath, V, F);
        
        //Mesh area
        Eigen::VectorXd A;
        igl::doublearea(V, F, A);
        A *= 0.5;
        
        //Helper function
        const auto inverted_triangles = [&] (const auto& W) {
            return igl::flipped_triangles(W,F).size();
        };
        const auto inverted_triangles_mesh = [&] (const auto& W) {
            const Eigen::VectorXi flip = igl::flipped_triangles(W,F);
            Eigen::MatrixXi flipF(flip.size(), 3);
            for(int i=0; i<flip.size(); ++i) {
                flipF.row(i) = F.row(flip(i));
            }
            return flipF;
        };
        
        //Compute Tutte parametrization.
        Eigen::MatrixXd W0t;
        const double tutte_t0 = igl::get_seconds();
        par::tutte(V, F, W0t);
        const double tutte_t = igl::get_seconds() - tutte_t0;
        log << " Tutte time: " << tutte_t << "s, inverted triangles: " <<
        inverted_triangles(W0t) << "." << std::endl;
        igl::writeOBJ(std::string("tutte_")+mesh, V, F, Eigen::MatrixXd(),
                      Eigen::MatrixXi(), W0t, F);
        igl::writeOBJ(std::string("tutte_UV_")+mesh, W0t, F);
        igl::writeOBJ(std::string("tutte_UV_flipped")+mesh, W0t,
                      inverted_triangles_mesh(W0t));
        
        //Compute UV map using ARAP, initialized with Tutte.
        Eigen::VectorXi fixed = Eigen::VectorXi::Ones(1);
        Eigen::MatrixXd fixedTo = Eigen::MatrixXd::Zero(1,2);
        Eigen::MatrixXd Wa = W0t;
        igl::ARAPData arap_data;
        arap_data.energy = igl::ARAP_ENERGY_TYPE_ELEMENTS;
        const double arap_t0 = igl::get_seconds();
        bool arapSuccess = igl::arap_precomputation(V, F, 2, fixed, arap_data);
        const double arapTutte_t0 = igl::get_seconds();
        const int arapMaxIter = 100000;
        for(int aIter=0; aIter<arapMaxIter; ++aIter) {
            Eigen::MatrixXd Wa0 = Wa;
            arapSuccess = igl::arap_solve(fixedTo, arap_data, Wa);
            if(!arapSuccess ||
               sqrt((Wa-Wa0).squaredNorm()/Wa.rows()) < 1e-6) {
                break;
            }
            if(aIter==arapMaxIter-1) {
                arapSuccess = false;
            }
        }
        const double arapTutte_t = igl::get_seconds() - arapTutte_t0;
        log << " ARAP, initialized with Tutte success: " <<
        arapSuccess << ", time: " << arapTutte_t << "s, inverted triangles: " <<
        inverted_triangles(Wa) << "." << std::endl;
        igl::writeOBJ(std::string("arap_")+mesh,
                      V, F, Eigen::MatrixXd(), Eigen::MatrixXi(), Wa, F);
        igl::writeOBJ(std::string("arap_UV_")+mesh, Wa, F);
        igl::writeOBJ(std::string("arap_UV_flipped")+mesh, Wa,
                      inverted_triangles_mesh(Wa));
        
        //Use map_to to unflip ARAP triangles.
        par::OptimizationOptions<double> opts;
        opts.terminationCondition = par::TerminationCondition::Noflips;
        opts.proximalBoundsActive = false;
        Eigen::MatrixXd Wsg = Wa;
        const double sg_t0 = igl::get_seconds();
        const bool sgParam = par::map_to
        <parallelize, par::EnergyType::SymmetricGradient>(V, F, Wsg, opts);
        const double sg_t = igl::get_seconds() - sg_t0;
        log << " Symmetric gradient, initialized with ARAP success: " <<
        sgParam << ", time: " << sg_t << "s, inverted triangles: " <<
        inverted_triangles(Wsg) << "." << std::endl;
        igl::writeOBJ(std::string("sg_")+mesh,
                      V, F, Eigen::MatrixXd(), Eigen::MatrixXi(), Wsg, F);
        igl::writeOBJ(std::string("sg_UV_")+mesh, Wsg, F);
        igl::writeOBJ(std::string("sg_UV_flipped")+mesh, Wsg,
                      inverted_triangles_mesh(Wsg));
        
        log << std::endl;
        log.close();
    }
    
    return 0;
}

