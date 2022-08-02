#include <parametrization/tutte.h>
#include <parametrization/map_to.h>
#include <parametrization/bff.h>
#include <parametrization/map_energy.h>

#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/doublearea.h>
#include <igl/flipped_triangles.h>
#include <igl/get_seconds.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>

#define IGL_VIEWER_VIEWER_QUIET
#include <igl/opengl/glfw/Viewer.h>

//This function computes a number of UV maps with SLIM and AKVF, and tries to
// reproduce a result with the same energy (up to a tolerance of 1e-6) using our
// Optimization algorithm.
//To do this, we call the respective external boundaries from the papers' github
// repos. These papers scale their output OBJs to have area one, so we try to
// undo this scaling (in order to compute the energy accurately) by multiplying
// the meshes with the original total area.
//Every call to one of the previous methods automatically times out after
// 2:30h.
//
//The pegasus mesh is too large to include raw on github. Make sure to
// uncompress it and uncomment line 41 to produce the pegasus results.

int main(int argc, char *argv[]) {
    namespace par = parametrization;
#ifdef OPENMP_AVAILABLE
    constexpr bool parallelize = true;
    
    const double targetEnergyTol = 1e-6; //tolerance for the energy target, needed to account for numerical error.
    
    std::vector<std::string> meshes = {
        "tree.obj", "brain.obj", "tooth.obj", "bread.obj", "deer.obj",
        "pegasus.obj", /*"camel.obj",*/ /*"cow.obj",*/ "cat.obj", "falconstatue.obj",
        /*"bar_sin.obj",*/ "car.obj", /*"elephant.obj",*/ "slime.obj", /*"horse.obj",*/
        /*"snake.obj",*/ /*"triceratops.obj",*/ "strawberry.obj", /*"hand.obj"*/
    };
    
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
        
        //The timeout command to add to all other commands
#ifdef __APPLE__
        const std::string timeout = "gtimeout 2.5h ";
#elif __linux__
        const std::string timeout = "timeout 2.5h ";
#else
        const std::string timeout = "";
#endif
        
        //Set OMP_NUM_THREADS to numbers of core on the machine
        const std::string set_cores("export OMP_NUM_THREADS=" EXT_CORE_NUM);
        
        //Compute SLIM parametrization (if SLIM is available) and compare to
        // map_to.
#ifdef SLIM_BINARY
        {
            std::string command = set_cores + std::string("; ");
            command += timeout + std::string(SLIM_BINARY " ") + meshPath + " ";
            std::string SLIMoutPath = "SLIM_" + mesh;
            command += SLIMoutPath;
            const double SLIM_t0 = igl::get_seconds();
            std::system(command.c_str());
            const double SLIM_t = igl::get_seconds() - SLIM_t0;
            
            //Read back the SLIM mesh. We need to undo the scaling here in order
            // to compute energy etc.
            Eigen::MatrixXd slimW, dummy0, dummy1;
            Eigen::MatrixXi dummy2, dummy3, dummy4;
            double slimEnergy;
            const bool slimSuccess = igl::readOBJ
            (SLIMoutPath, dummy0, slimW, dummy1, dummy2, dummy3, dummy4) &&
            igl::flipped_triangles(slimW, F).size() == 0;
            if(slimSuccess) {
                Eigen::VectorXd sA;
                igl::doublearea(slimW, F, sA);
                sA *= 0.5;
                slimW *= sqrt(A.sum()) / sqrt(sA.sum());
                
                igl::writeOBJ(std::string("SLIM_UV_")+mesh, slimW, F);
                
                slimEnergy = par::map_energy
                <par::EnergyType::SymmetricDirichlet>(V, F, slimW);
                log << " SLIM success: " << slimSuccess <<
                ", time: " << SLIM_t << "s, inverted triangles: " <<
                inverted_triangles(slimW) << ", L2 distortion: " <<
                l2_area_distortion(slimW) << ", Linf area distortion: " <<
                linf_area_distortion(slimW) << ", symmetric Dirichlet energy: " <<
                slimEnergy << "." << std::endl;
            } else {
                log << " SLIM success: " << slimSuccess << std::endl;
            }
            
            //Try to find a parametrization that matches the energy of the
            // previous work parametrization. The energy termination condition
            // does not exactly guarantee termination at the specified energy
            // (since it measures the energy of the P iterate, not of W), but it
            // should be close enough with a margin for error. If it
            // is not close enough, discard the result later.
            //We reload the mesh when using map_to, since the external binaries
            // also have to do that, it makes it easier to compare.
            {
                const double t0 = igl::get_seconds();
                Eigen::MatrixXd V;
                Eigen::MatrixXi F;
                igl::readOBJ(meshPath, V, F);
                Eigen::MatrixXd W;
                bool tutteOk = par::tutte<false>(V, F, W);
                if(!tutteOk || igl::flipped_triangles(W,F).size()>0) {
                    tutteOk = par::tutte<true>(V, F, W);
                }
                if(tutteOk) {
                    par::OptimizationOptions<double> opts;
                    if(slimSuccess) {
                        opts.terminationCondition =
                        par::TerminationCondition::TargetEnergyNoflips;
                        opts.targetEnergy = slimEnergy + targetEnergyTol;
                    }
                    const bool oursSuccess = par::map_to
                    <parallelize, par::EnergyType::SymmetricDirichlet>
                    (V, F, W, opts);
                    igl::writeOBJ(std::string("oursTutte_matchSLIM_")+mesh, V, F,
                                  Eigen::MatrixXd(), Eigen::MatrixXi(), W, F);
                    const double t = igl::get_seconds() - t0;
                    igl::writeOBJ(std::string("oursTutte_matchSLIM_UV_")+mesh,
                                  W, F);
                    
                    const double oursEnergy = par::map_energy
                    <par::EnergyType::SymmetricDirichlet>(V, F, W);
                    log <<
                    "  Ours, initializing with Tutte, matching SLIM, success: " <<
                    oursSuccess << ", time: " << t << "s, inverted triangles: " <<
                    inverted_triangles(W) << ", L2 distortion: " <<
                    l2_area_distortion(W) << ", Linf area distortion: " <<
                    linf_area_distortion(W) << ", symmetric Dirichlet energy: " <<
                    oursEnergy << "." << std::endl;
                } else {
                    std::cout << "  Tutte parametrization failed..." <<
                    std::endl;
                }
            }
#ifdef SUITESPARSE_AVAILABLE
            if(mesh != "pegasus.obj" && mesh != "brain.obj" &&
               mesh != "tree.obj" && mesh != "slime.obj")
            {
                const double t0 = igl::get_seconds();
                Eigen::MatrixXd V;
                Eigen::MatrixXi F;
                igl::readOBJ(meshPath, V, F);
                Eigen::MatrixXd W;
                const bool bffOk = par::bff(V, F, W);
                if(bffOk) {
                    par::OptimizationOptions<double> opts;
                    opts.proximalBoundsActive = false;
                    if(slimSuccess) {
                        opts.terminationCondition =
                        par::TerminationCondition::TargetEnergyNoflips;
                        opts.targetEnergy = slimEnergy + targetEnergyTol;
                    }
                    const bool oursSuccess = par::map_to
                    <parallelize, par::EnergyType::SymmetricDirichlet>
                    (V, F, W, opts);
                    igl::writeOBJ(std::string("oursBFF_matchSLIM_")+mesh, V, F,
                                  Eigen::MatrixXd(), Eigen::MatrixXi(), W, F);
                    const double t = igl::get_seconds() - t0;
                    igl::writeOBJ(std::string("oursBFF_matchSLIM_UV_")+mesh,
                                  W, F);
                    
                    const double oursEnergy = par::map_energy
                    <par::EnergyType::SymmetricDirichlet>(V, F, W);
                    log <<
                    "  Ours, initializing with BFF, matching SLIM, success: " <<
                    oursSuccess << ", time: " << t << "s, inverted triangles: " <<
                    inverted_triangles(W) << ", L2 distortion: " <<
                    l2_area_distortion(W) << ", Linf area distortion: " <<
                    linf_area_distortion(W) << ", symmetric Dirichlet energy: " <<
                    oursEnergy << "." << std::endl;
                } else {
                    std::cout << "  BFF parametrization failed..." <<
                    std::endl;
                }
            }
#endif
        }
#endif
        
        
        //Compute AKVF parametrization (if SLIM is available) and compare to
        // map_to.
#ifdef AKVF_BINARY
        {
            //AKVF cares about which directory its executed in, and will not
            // properly work otherwise.
            std::string command = set_cores +
            std::string("; cd " MESH_DIRECTORY
                        "; mkdir results; ");
            command += timeout + std::string(AKVF_BINARY " ") + mesh;
            const double AKVF_t0 = igl::get_seconds();
            std::system(command.c_str());
            const double AKVF_t = igl::get_seconds() - AKVF_t0;
            std::string mvCommand =
            std::string("mv " MESH_DIRECTORY "results/*.obj ./AKVF_") + mesh;
            std::system(mvCommand.c_str());
            std::system("rm -rf " MESH_DIRECTORY "results");
            
            //Read back the AKVF mesh. We need to undo the scaling here in order
            // to compute energy etc.
            Eigen::MatrixXd akvfW, dummy0, dummy1;
            Eigen::MatrixXi dummy2, dummy3, dummy4;
            double akvfEnergy;
            std::string AKVFoutPath = std::string("AKVF_") + mesh;
            const bool akvfSuccess = igl::readOBJ
            (AKVFoutPath, dummy0, akvfW, dummy1, dummy2, dummy3, dummy4) &&
            igl::flipped_triangles(akvfW, F).size() == 0;
            if(akvfSuccess) {
                Eigen::VectorXd sA;
                igl::doublearea(akvfW, F, sA);
                sA *= 0.5;
                akvfW *= sqrt(A.sum()) / sqrt(sA.sum());
                
                igl::writeOBJ(std::string("AKVF_UV_")+mesh, akvfW, F);
                
                akvfEnergy = par::map_energy
                <par::EnergyType::SymmetricDirichlet>(V, F, akvfW);
                log << " AKVF success: " << akvfSuccess <<
                ", time: " << AKVF_t << "s, inverted triangles: " <<
                inverted_triangles(akvfW) << ", L2 distortion: " <<
                l2_area_distortion(akvfW) << ", Linf area distortion: " <<
                linf_area_distortion(akvfW) << ", symmetric Dirichlet energy: " <<
                akvfEnergy << "." << std::endl;
            } else {
                log << " AKVF success: " << akvfSuccess << std::endl;
            }
            
            //Try to find a parametrization that matches the energy of the
            // previous work parametrization. The energy termination condition
            // does not exactly guarantee termination at the specified energy
            // (since it measures the energy of the P iterate, not of W), but it
            // should be close enough with a margin for error. If it
            // is not close enough, discard the result later.
            //We reload the mesh when using map_to, since the external binaries
            // also have to do that, it makes it easier to compare.
            {
                const double t0 = igl::get_seconds();
                Eigen::MatrixXd V;
                Eigen::MatrixXi F;
                igl::readOBJ(meshPath, V, F);
                Eigen::MatrixXd W;
                bool tutteOk = par::tutte<false>(V, F, W);
                if(!tutteOk || igl::flipped_triangles(W,F).size()>0) {
                    tutteOk = par::tutte<true>(V, F, W);
                }
                if(tutteOk) {
                    par::OptimizationOptions<double> opts;
                    //This is a hack: since we know from experience that AKVF
                    // struggles with tree.obj, don't try to match it here, use
                    // standard termination conditions
                    if(akvfSuccess && mesh != "tree.obj") {
                        opts.terminationCondition =
                        par::TerminationCondition::TargetEnergyNoflips;
                        opts.targetEnergy = akvfEnergy + targetEnergyTol;
                    }
                    const bool oursSuccess = par::map_to
                    <parallelize, par::EnergyType::SymmetricDirichlet>
                    (V, F, W, opts);
                    igl::writeOBJ(std::string("oursTutte_matchAKVF_")+mesh, V, F,
                                  Eigen::MatrixXd(), Eigen::MatrixXi(), W, F);
                    const double t = igl::get_seconds() - t0;
                    igl::writeOBJ(std::string("oursTutte_matchAKVF_UV_")+mesh,
                                  W, F);
                    
                    const double oursEnergy = par::map_energy
                    <par::EnergyType::SymmetricDirichlet>(V, F, W);
                    log <<
                    "  Ours, initializing with Tutte, matching AKVF, success: " <<
                    oursSuccess << ", time: " << t << "s, inverted triangles: " <<
                    inverted_triangles(W) << ", L2 distortion: " <<
                    l2_area_distortion(W) << ", Linf area distortion: " <<
                    linf_area_distortion(W) << ", symmetric Dirichlet energy: " <<
                    oursEnergy << "." << std::endl;
                } else {
                    std::cout << "  Tutte parametrization failed..." <<
                    std::endl;
                }
            }
#ifdef SUITESPARSE_AVAILABLE
            if(mesh != "pegasus.obj" && mesh != "brain.obj" &&
               mesh != "tree.obj" && mesh != "slime.obj")
            {
                const double t0 = igl::get_seconds();
                Eigen::MatrixXd V;
                Eigen::MatrixXi F;
                igl::readOBJ(meshPath, V, F);
                Eigen::MatrixXd W;
                const bool bffOk = par::bff(V, F, W);
                if(bffOk) {
                    par::OptimizationOptions<double> opts;
                    opts.proximalBoundsActive = false;
                    if(akvfSuccess) {
                        opts.terminationCondition =
                        par::TerminationCondition::TargetEnergyNoflips;
                        opts.targetEnergy = akvfEnergy + targetEnergyTol;
                    }
                    const bool oursSuccess = par::map_to
                    <parallelize, par::EnergyType::SymmetricDirichlet>
                    (V, F, W, opts);
                    igl::writeOBJ(std::string("oursBFF_matchAKVF_")+mesh, V, F,
                                  Eigen::MatrixXd(), Eigen::MatrixXi(), W, F);
                    const double t = igl::get_seconds() - t0;
                    igl::writeOBJ(std::string("oursBFF_matchAKVF_UV_")+mesh,
                                  W, F);
                    
                    const double oursEnergy = par::map_energy
                    <par::EnergyType::SymmetricDirichlet>(V, F, W);
                    log <<
                    "  Ours, initializing with BFF, matching AKVF, success: " <<
                    oursSuccess << ", time: " << t << "s, inverted triangles: " <<
                    inverted_triangles(W) << ", L2 distortion: " <<
                    l2_area_distortion(W) << ", Linf area distortion: " <<
                    linf_area_distortion(W) << ", symmetric Dirichlet energy: " <<
                    oursEnergy << "." << std::endl;
                } else {
                    std::cout << "  BFF parametrization failed..." <<
                    std::endl;
                }
            }
#endif
        }
#endif
        
        //Compute ProgressiveParametrizations parametrization (if the binary is available) and compare to
        // map_to.
#ifdef ProgressiveParametrizations_BINARY
        {
            //Execute ProgressiveParametrization, and put all files where they
            // belong.
            std::string command = set_cores +
            std::string("; mkdir PP_workingdir; cd PP_workingdir; ") + timeout +
            std::string(ProgressiveParametrizations_BINARY " ");
            command += meshPath;
            const double PP_t0 = igl::get_seconds();
            std::system(command.c_str());
            const double PP_t = igl::get_seconds() - PP_t0;
            std::string mvCommand =
            std::string("mv PP_workingdir/*.obj ./PP_") + mesh;
            std::system(mvCommand.c_str());
            std::system("rm -rf PP_workingdir");
            
            //Read back the PP mesh and apply the same de-scaling.
            Eigen::MatrixXd PPW;
            Eigen::MatrixXi dummy;
            double PPEnergy;
            std::string PPoutPath = std::string("PP_") + mesh;
            const bool PPSuccess = igl::readOBJ(PPoutPath, PPW, dummy) &&
            igl::flipped_triangles(PPW, F).size() == 0;
            if(PPSuccess) {
                PPW = PPW.block(0, 0, PPW.rows(), 2);
                Eigen::VectorXd sA;
                igl::doublearea(PPW, F, sA);
                sA *= 0.5;
                PPW *= sqrt(A.sum()) / sqrt(sA.sum());
                
                igl::writeOBJ(std::string("PP_UV_")+mesh, PPW, F);
                
                PPEnergy = par::map_energy
                <par::EnergyType::SymmetricDirichlet>(V, F, PPW);
                log << " PP success: " << PPSuccess <<
                ", time: " << PP_t << "s, inverted triangles: " <<
                inverted_triangles(PPW) << ", L2 distortion: " <<
                l2_area_distortion(PPW) << ", Linf area distortion: " <<
                linf_area_distortion(PPW) << ", symmetric Dirichlet energy: " <<
                PPEnergy << "." << std::endl;
            } else {
                log << " PP success: " << PPSuccess << std::endl;
            }
            
            //Try to find a parametrization that matches the energy of the
            // previous work parametrization. The energy termination condition
            // does not exactly guarantee termination at the specified energy
            // (since it measures the energy of the P iterate, not of W), but it
            // should be close enough with a margin for error. If it
            // is not close enough, discard the result later.
            //We reload the mesh when using map_to, since the external binaries
            // also have to do that, it makes it easier to compare.
            {
                const double t0 = igl::get_seconds();
                Eigen::MatrixXd V;
                Eigen::MatrixXi F;
                igl::readOBJ(meshPath, V, F);
                Eigen::MatrixXd W;
                bool tutteOk = par::tutte<false>(V, F, W);
                if(!tutteOk || igl::flipped_triangles(W,F).size()>0) {
                    tutteOk = par::tutte<true>(V, F, W);
                }
                if(tutteOk) {
                    par::OptimizationOptions<double> opts;
                    if(PPSuccess) {
                        opts.terminationCondition =
                        par::TerminationCondition::TargetEnergyNoflips;
                        opts.targetEnergy = PPEnergy + targetEnergyTol;
                    }
                    const bool oursSuccess = par::map_to
                    <parallelize, par::EnergyType::SymmetricDirichlet>
                    (V, F, W, opts);
                    igl::writeOBJ(std::string("oursTutte_matchPP_")+mesh, V, F,
                                  Eigen::MatrixXd(), Eigen::MatrixXi(), W, F);
                    const double t = igl::get_seconds() - t0;
                    igl::writeOBJ(std::string("oursTutte_matchPP_UV_")+mesh,
                                  W, F);
                    
                    const double oursEnergy = par::map_energy
                    <par::EnergyType::SymmetricDirichlet>(V, F, W);
                    log <<
                    "  Ours, initializing with Tutte, matching PP, success: " <<
                    oursSuccess << ", time: " << t << "s, inverted triangles: " <<
                    inverted_triangles(W) << ", L2 distortion: " <<
                    l2_area_distortion(W) << ", Linf area distortion: " <<
                    linf_area_distortion(W) << ", symmetric Dirichlet energy: " <<
                    oursEnergy << "." << std::endl;
                } else {
                    std::cout << "  Tutte parametrization failed..." <<
                    std::endl;
                }
            }
#ifdef SUITESPARSE_AVAILABLE
            if(mesh != "pegasus.obj" && mesh != "brain.obj" &&
               mesh != "tree.obj" && mesh != "slime.obj")
            {
                const double t0 = igl::get_seconds();
                Eigen::MatrixXd V;
                Eigen::MatrixXi F;
                igl::readOBJ(meshPath, V, F);
                Eigen::MatrixXd W;
                const bool bffOk = par::bff(V, F, W);
                if(bffOk) {
                    par::OptimizationOptions<double> opts;
                    opts.proximalBoundsActive = false;
                    if(PPSuccess) {
                        opts.terminationCondition =
                        par::TerminationCondition::TargetEnergyNoflips;
                        opts.targetEnergy = PPEnergy + targetEnergyTol;
                    }
                    const bool oursSuccess = par::map_to
                    <parallelize, par::EnergyType::SymmetricDirichlet>
                    (V, F, W, opts);
                    igl::writeOBJ(std::string("oursBFF_matchPP_")+mesh, V, F,
                                  Eigen::MatrixXd(), Eigen::MatrixXi(), W, F);
                    const double t = igl::get_seconds() - t0;
                    igl::writeOBJ(std::string("oursBFF_matchPP_UV_")+mesh,
                                  W, F);
                    
                    const double oursEnergy = par::map_energy
                    <par::EnergyType::SymmetricDirichlet>(V, F, W);
                    log <<
                    "  Ours, initializing with BFF, matching PP, success: " <<
                    oursSuccess << ", time: " << t << "s, inverted triangles: " <<
                    inverted_triangles(W) << ", L2 distortion: " <<
                    l2_area_distortion(W) << ", Linf area distortion: " <<
                    linf_area_distortion(W) << ", symmetric Dirichlet energy: " <<
                    oursEnergy << "." << std::endl;
                } else {
                    std::cout << "  BFF parametrization failed..." <<
                    std::endl;
                }
            }
#endif  
        }
#endif
        
        log << std::endl;
        log.close();
    }
    
#endif
    return 0;
}
