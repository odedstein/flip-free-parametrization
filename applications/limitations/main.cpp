#include <parametrization/tutte.h>
#include <parametrization/map_to.h>
#include <parametrization/map_energy.h>

#include <igl/readOBJ.h>
#include <igl/readMESH.h>
#include <igl/writeOBJ.h>
#include <igl/writeMESH.h>
#include <igl/doublearea.h>
#include <igl/flipped_triangles.h>
#include <igl/get_seconds.h>
#include <igl/snap_points.h>
#include <igl/boundary_facets.h>
#include <parametrization/sym_from_triu.h>
#include <parametrization/grad_f.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>

#define IGL_VIEWER_VIEWER_QUIET
#include <igl/opengl/glfw/Viewer.h>

//This function computes three examples of limitations of our method:
// - double covers
// - difficult to fulfill constraints
// - bad initial data

int main(int argc, char *argv[]) {
    namespace par = parametrization;
#ifdef OPENMP_AVAILABLE
    constexpr bool parallelize = true;
#else
    constexpr bool parallelize = false;
#endif
    
    //Double covers: there are valid solutions that are flip-free, but not
    // bijective.
    {
        //Create log file
        std::ofstream log;
        log.open(std::string("clowncollar.log.txt"));
        log << "clowncollar.obj" << std::endl;
        
        //Load clowncollar, embedded parametrization, and cover parametrization.
        Eigen::MatrixXd V, embW, coverW;
        Eigen::MatrixXi F;
        const std::string mesh = "clowncollar.obj",
        meshPath = std::string(MESH_DIRECTORY "clowncollar.obj"),
        embMeshPath = std::string(MESH_DIRECTORY "clowncollar_embedded.obj"),
        coverMeshPath = std::string(MESH_DIRECTORY "clowncollar_cover.obj");
        igl::readOBJ(meshPath, V, F);
        igl::readOBJ(embMeshPath, embW, F);
        igl::readOBJ(coverMeshPath, coverW, F);
        embW = embW.block(0, 0, embW.rows(), 2).eval();
        coverW = coverW.block(0, 0, coverW.rows(), 2).eval();
        
        //Mesh area
        Eigen::VectorXd A;
        igl::doublearea(V, F, A);
        A *= 0.5;
        
        //Helper functions for statistics
        const auto l2_area_distortion = [&] (const auto& W) {
            Eigen::VectorXd AW;
            igl::doublearea(W, F, AW);
            AW *= 0.5;
            return sqrt((A.array() * (A-AW).array().abs().pow(2)).sum());
        };
        const auto linf_area_distortion = [&] (const auto& W) {
            Eigen::VectorXd AW;
            igl::doublearea(W, F, AW);
            AW *= 0.5;
            return (A-AW).array().abs().maxCoeff();
        };
        const auto inverted_triangles = [&] (const auto& W) {
            return igl::flipped_triangles(W,F).size();
        };
        
        //Save UV-mapped mesh with embedded and cover parametrization.
        igl::writeOBJ(std::string("embedded_")+mesh, V, F,
                      Eigen::MatrixXd(), Eigen::MatrixXi(), embW, F);
        igl::writeOBJ(std::string("cover_")+mesh, V, F,
                      Eigen::MatrixXd(), Eigen::MatrixXi(), coverW, F);
        
        //Run our method, starting with an embedded parametrization.
        const double oursWithEmbedded_t0 = igl::get_seconds();
        const bool oursWithEmbeddedSuccess = par::map_to
        <parallelize, par::EnergyType::SymmetricGradient>(V, F, embW);
        const double oursWithEmbedded_t = igl::get_seconds() -
        oursWithEmbedded_t0;
        igl::writeOBJ(std::string("oursEmbedded_")+mesh, V, F,
                      Eigen::MatrixXd(), Eigen::MatrixXi(), embW, F);
        igl::writeOBJ(std::string("oursEmbedded_UV_")+mesh, embW, F);
        log << "  Ours, initializing with embedded, success: " <<
        oursWithEmbeddedSuccess << ", time: " << oursWithEmbedded_t <<
        "s, inverted triangles: " << inverted_triangles(embW) <<
        ", L2 distortion: " << l2_area_distortion(embW) <<
        ", Linf area distortion: " << linf_area_distortion(embW) <<
        ", map energy: " <<
        par::map_energy<par::EnergyType::SymmetricGradient>(V,F,embW) << "."
        << std::endl;
        
        //Run our method, starting with a cover parametrization.
        const double oursWithCover_t0 = igl::get_seconds();
        const bool oursWithCoverSuccess = par::map_to
        <parallelize, par::EnergyType::SymmetricGradient>(V, F, coverW);
        const double oursWithCover_t = igl::get_seconds() - oursWithCover_t0;
        igl::writeOBJ(std::string("oursCover_")+mesh, V, F,
                      Eigen::MatrixXd(), Eigen::MatrixXi(), coverW, F);
        igl::writeOBJ(std::string("oursCover_UV_")+mesh, coverW, F);
        log << "  Ours, initializing with cover, success: " <<
        oursWithCoverSuccess << ", time: " << oursWithCover_t <<
        "s, inverted triangles: " << inverted_triangles(coverW) <<
        ", L2 distortion: " << l2_area_distortion(coverW) <<
        ", Linf area distortion: " << linf_area_distortion(coverW) <<
        ", map energy: " <<
        par::map_energy<par::EnergyType::SymmetricGradient>(V,F,coverW) << "."
        << std::endl;
        
        log << std::endl;
        log.close();
    }
    
    //Difficult to fulfill initial constraints: a 3D deformation with many
    // constraints set randomly makes it impossible for our method to find a
    // solution.
    {
        const std::string mesh = "monkey.mesh";
        const std::string meshOBJ = mesh + std::string(".obj");
        
        //Read mesh
        Eigen::MatrixXd V;
        Eigen::MatrixXi F, tF, bdF;
        const std::string meshPath = std::string(MESH_DIRECTORY) + mesh;
        igl::readMESH(meshPath, V, F, tF);
        igl::boundary_facets(F, bdF);
        Eigen::VectorXi tmp = bdF.col(2);
        bdF.col(2) = bdF.col(1);
        bdF.col(1) = tmp;
        igl::writeOBJ(meshOBJ, V, bdF);
        
        //Create log file
        std::ofstream log;
        log.open(mesh + std::string(".log.txt"));
        log << mesh << std::endl;
        
        //Boundary conditions
        Eigen::MatrixXd fixedCoords, fixedTo;
        fixedCoords.resize(24,3);
        fixedCoords << -0.0932486, -0.141815, -0.303175,
        0.0850076, -0.152054, -0.315685,
        0.121852, 0.311024, -0.310157,
        -0.11444, 0.299227, -0.340067,
        0.0971955, 0.306997, -0.134926,
        -0.0992074, 0.299635, -0.122251,
        -0.0968845, 0.377968, 0.0724707,
        0.109897, 0.25914, 0.0759923,
        0.000135686, 0.512893, 0.21358,
        0.0146582, 0.454414, 0.146849,
        0.117117, -0.0810753, -0.0443019,
        -0.115462, -0.0735894, -0.026346,
        -0.165057, -0.114303, 0.155975,
        0.181025, -0.127228, 0.171933,
        -0.0102017, -0.0656066, 0.288051,
        -0.0059037, 0.0698861, 0.203186,
        -0.0129042, 0.288365, 0.208499,
        -0.0241481, -0.385072, 0.410487,
        -0.00197233, -0.265157, 0.312693,
        -0.00342699, -0.438631, 0.201942,
        0.023853, 0.123764, 0.0718019,
        -0.109141, -0.325096, 0.357585,
        0.107885, -0.323017, 0.358152,
        0.0114169, -0.354148, 0.271542;
        srand(19);
        fixedTo.resize(24,3);
        fixedTo.setRandom();
        
        //Fix vertices from fixedCoords
        Eigen::VectorXi fixed;
        igl::snap_points(fixedCoords, V, fixed);
        log << " fixed vertex IDs: ";
        for(int j=0; j<fixed.size(); ++j) {
            log << "  " << fixed(j) << " to " << fixedTo.row(j) << std::endl;
        }
        
        //Variables needed for map_to
        Eigen::SparseMatrix<double> Aeq;
        Eigen::MatrixXd beq;
        par::OptimizationOptions<double> opts;
        opts.maxIter = 10000;
        
        //Compute deformation using our symmetric gradient.
        Eigen::MatrixXd Wsg = V;
        const double sg_t0 = igl::get_seconds();
        const bool sgSuccess = par::map_to
        <parallelize, par::EnergyType::SymmetricGradient>
        (V, F, fixed, fixedTo, Aeq, beq, Wsg, opts);
        const double sg_t = igl::get_seconds() - sg_t0;
        log << " Symmetric gradient success: " << sgSuccess <<
        ", time: " << sg_t << "s." << std::endl;
        igl::writeOBJ(std::string("sg_deformed_")+meshOBJ, Wsg, bdF);
        igl::writeMESH(std::string("sg_deformed_")+mesh, Wsg, F, tF);
        
        //Compute deformation using our symmetric Dirichlet.
        Eigen::MatrixXd Wsd = V;
        const double sd_t0 = igl::get_seconds();
        const bool sdSuccess = par::map_to
        <parallelize, par::EnergyType::SymmetricDirichlet>
        (V, F, fixed, fixedTo, Aeq, beq, Wsd, opts);
        const double sd_t = igl::get_seconds() - sd_t0;
        log << " Symmetric Dirichlet success: " << sdSuccess <<
        ", time: " << sd_t << "s." << std::endl;
        igl::writeOBJ(std::string("sd_deformed_")+meshOBJ, Wsd, bdF);
        igl::writeMESH(std::string("sd_deformed_")+mesh, Wsd, F, tF);
        
        log << std::endl;
        log.close();
    }
    
    //Bad initial data: try to parametrize a mesh with random data as input.
    {
        std::string mesh = "helmet.obj";
        
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
        
        //Helper functions for statistics
        const auto l2_area_distortion = [&] (const auto& W) {
            Eigen::VectorXd AW;
            igl::doublearea(W, F, AW);
            AW *= 0.5;
            return sqrt((A.array() * (A-AW).array().abs().pow(2)).sum());
        };
        const auto linf_area_distortion = [&] (const auto& W) {
            Eigen::VectorXd AW;
            igl::doublearea(W, F, AW);
            AW *= 0.5;
            return (A-AW).array().abs().maxCoeff();
        };
        const auto inverted_triangles = [&] (const auto& W) {
            return igl::flipped_triangles(W,F).size();
        };
        const int m = F.rows();
        std::vector<double> largestGradf;
        par::CallbackFunction<Eigen::MatrixXd, int> f = [&]
        (auto& state, auto&, auto&) {
            double maxGradf = 0;
            for(int i=0; i<m; ++i) {
                const Eigen::Matrix2d Pl = par::sym_2d_from_index(state.P, i, m);
                const double normf =
                par::grad_f<par::EnergyType::SymmetricGradient>(Pl).norm();
                if(normf > maxGradf) {
                    maxGradf = normf;
                }
            }
            largestGradf.push_back(maxGradf);
            
            return false;
        };
        const auto reset_f = [&] () {
            largestGradf.clear();
        };
        
        //Compute Tutte parametrization.
        Eigen::MatrixXd Wt;
        bool tutteOk = par::tutte<false>(V, F, Wt);
        if(!tutteOk || igl::flipped_triangles(Wt,F).size()>0) {
            tutteOk = par::tutte<true>(V, F, Wt);
        }
        igl::writeOBJ(std::string("tutte_")+mesh, V, F,
                      Eigen::MatrixXd(), Eigen::MatrixXi(), Wt, F);
        
        //Variables needed for map_to
        par::OptimizationOptions<double> opts;
        opts.maxIter = 10000;
        
        //Compute parametrization using Tutte initialization
        const double oursWithTutte_t0 = igl::get_seconds();
        const bool oursWithTutteSuccess = par::map_to
        <parallelize, par::EnergyType::SymmetricGradient>(V, F, Wt, opts, f);
        const double oursWithTutte_t = igl::get_seconds() - oursWithTutte_t0;
        igl::writeOBJ(std::string("oursTutte_")+mesh, V, F,
                      Eigen::MatrixXd(), Eigen::MatrixXi(), Wt, F);
        igl::writeOBJ(std::string("oursTutte_UV_")+mesh, Wt, F);
        log << "  Ours, initializing with Tutte, success: " <<
        oursWithTutteSuccess << ", time: " << oursWithTutte_t <<
        "s, inverted triangles: " << inverted_triangles(Wt) <<
        ", L2 distortion: " << l2_area_distortion(Wt) <<
        ", Linf area distortion: " << linf_area_distortion(Wt) << "." <<
        std::endl;
        log << "Largest gradient norm log:" <<
        std::endl << "largestGradf = [";
        for(int i=0; i<largestGradf.size(); ++i) {
            log << largestGradf[i];
            if(i != largestGradf.size()-1) {
                log << ", ";
            }
        }
        log << "];" << std::endl;
        reset_f();
        
        //Random initialization
        srand(36);
        Eigen::MatrixXd Wrand = Eigen::MatrixXd::Random(V.rows(), 2);
        igl::writeOBJ(std::string("rand_")+mesh, V, F,
                      Eigen::MatrixXd(), Eigen::MatrixXi(), Wrand, F);
        
        //Compute parametrization using random initialization
        const double oursWithRand_t0 = igl::get_seconds();
        const bool oursWithRandSuccess = par::map_to
        <parallelize, par::EnergyType::SymmetricGradient>(V, F, Wrand, opts, f);
        const double oursWithRand_t = igl::get_seconds() - oursWithRand_t0;
        igl::writeOBJ(std::string("oursRand_")+mesh, V, F,
                      Eigen::MatrixXd(), Eigen::MatrixXi(), Wrand, F);
        igl::writeOBJ(std::string("oursRand_UV_")+mesh, Wrand, F);
        log << "  Ours, initializing with random, success: " <<
        oursWithRandSuccess << ", time: " << oursWithRand_t <<
        "s, inverted triangles: " << inverted_triangles(Wrand) <<
        ", L2 distortion: " << l2_area_distortion(Wrand) <<
        ", Linf area distortion: " << linf_area_distortion(Wrand) << "." <<
        std::endl;
        log << "Largest gradient norm log:" <<
        std::endl << "largestGradf = [";
        for(int i=0; i<largestGradf.size(); ++i) {
            log << largestGradf[i];
            if(i != largestGradf.size()-1) {
                log << ", ";
            }
        }
        log << "];" << std::endl;
        reset_f();
        
        log << std::endl;
        log.close();
    }
    
        
    return 0;
}
