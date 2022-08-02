
#include <parametrization/tutte.h>
#include <parametrization/map_to.h>
#include <parametrization/bff.h>
#include <parametrization/mat_from_index.h>
#include <parametrization/rotmat_from_complex_quat.h>
#include <parametrization/sym_from_triu.h>
#include <parametrization/grad_f.h>

#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/doublearea.h>
#include <igl/flipped_triangles.h>
#include <igl/get_seconds.h>
#include <igl/arap.h>

#include <iostream>
#include <fstream>
#include <vector>

//This function computes a number of UV maps, initializing with Tutte and BFF.
// alternately.
//The function outputs the UV maps and a variety of statistics.
//
//This function also computes an ARAP parametrization. However, do *not* use
// this to compare timings with the map_to maps - they use completely
// different termination conditions.
//ARAP is terminated after doing less than 1e-6 progress, or after 2:30h.

int main(int argc, char *argv[]) {
    namespace par = parametrization;
#ifdef OPENMP_AVAILABLE
    constexpr bool parallelize = true;
#else
    constexpr bool parallelize = false;
#endif
    
    std::vector<std::string> meshes = {
        "camelhead.obj", "bunny.obj",
        "goathead.obj", "mushroom.obj", "nefertiti.obj", "strawberry.obj",
        "tree.obj", "brucewick.obj", "mountain.obj", "cactus.obj", "brain.obj"
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
        const auto inverted_triangles_as_elements = [&] (const auto& W) {
            const Eigen::VectorXi flippedInds = igl::flipped_triangles(W,F);
            Eigen::MatrixXi flippedF(flippedInds.size(), 3);
            for(int i=0; i<flippedInds.size(); ++i) {
                flippedF.row(i) = F.row(flippedInds(i));
            }
            return flippedF;
        };
        const auto inverted_triangles = [&] (const auto& W) -> int {
            return inverted_triangles_as_elements(W).rows();
        };
        
        //Generic opts and callback
        par::OptimizationOptions<double> opts;
        Eigen::MatrixXd Lambda0;
        Eigen::VectorXd U0;
        std::vector<double> LambdaDiffNormByPreciseBound, largestGradf, ePs,
        eDs;
        const int m = F.rows();
        par::CallbackFunction<Eigen::MatrixXd, int> f = [&]
        (auto& state, auto&, auto&) {
            const double time = igl::get_seconds();
            
            double largestLambdaDiffNormByPreciseBound = 0;
            for(int i=0; i<m; ++i) {
                const double mul = state.mu(i);
                const Eigen::Matrix2d Lambdal = par::mat_from_index_2x2(state.Lambda, i, m) * mul;
                
                if(Lambda0.rows()>0 && U0.rows()>0) {
                    const Eigen::Matrix2d Lambda0l = par::mat_from_index_2x2(Lambda0, i, m);
                    const Eigen::Matrix2d Ul = par::rotmat_2d_from_index(state.U, i, m);
                    const Eigen::Matrix2d U0l = par::rotmat_2d_from_index(U0, i, m);
                    const double LambdalByPreciseBoundl =
                    (Lambdal - Lambda0l).norm() /
                    (0.5*(Lambdal + Ul*Lambdal.transpose()*Ul
                          - Lambda0l - U0l*Lambda0l.transpose()*U0l)).norm();
                    if(LambdalByPreciseBoundl >
                       largestLambdaDiffNormByPreciseBound) {
                        largestLambdaDiffNormByPreciseBound =
                        LambdalByPreciseBoundl;
                    }
                }
            }
            
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
            
            if(Lambda0.rows()>0 && U0.rows()>0) {
                LambdaDiffNormByPreciseBound.push_back
                (largestLambdaDiffNormByPreciseBound);
            } else {
                LambdaDiffNormByPreciseBound.push_back(0.);
            }
            ePs.push_back(sqrt(state.primalErr));
            eDs.push_back(sqrt(state.dualErr));
            
            Lambda0.resize(state.Lambda.rows(), state.Lambda.cols());
            for(int i=0; i<m; ++i) {
                const double mul = state.mu(i);
                const Eigen::Matrix2d Lambdal = par::mat_from_index_2x2(state.Lambda, i, m) * mul;
                par::index_into_mat_2x2(Lambdal, Lambda0, i, m);
            }
            U0 = state.U;
            
            return false;
        };
        const auto reset_f = [&] () {
            Lambda0.resize(0,0);
            U0.resize(0,0);
            LambdaDiffNormByPreciseBound.clear();
            ePs.clear();
            eDs.clear();
            largestGradf.clear();
        };
        const auto save_iter_data_to = [&] (const std::string& file) {
            std::ofstream log;
            log.open(file + std::string(".stats_log.txt"));
            log << file << "Primal error log:" << std::endl << "primal_err = [";
            for(int i=0; i<LambdaDiffNormByPreciseBound.size(); ++i) {
                log << ePs[i];
                if(i != ePs.size()-1) {
                    log << ", ";
                }
            }
            log << "];" << std::endl;
            log << file << "Dual error log:" << std::endl << "dual_err = [";
            for(int i=0; i<LambdaDiffNormByPreciseBound.size(); ++i) {
                log << eDs[i];
                if(i != eDs.size()-1) {
                    log << ", ";
                }
            }
            log << "];" << std::endl;
            log << file << "Largest ratio of LambdaDiff by PreciseBound log:" <<
            std::endl << "LambdaDiffByPreciseBound = [";
            for(int i=0; i<LambdaDiffNormByPreciseBound.size(); ++i) {
                log << LambdaDiffNormByPreciseBound[i];
                if(i != LambdaDiffNormByPreciseBound.size()-1) {
                    log << ", ";
                }
            }
            log << "];" << std::endl;
            log << file << "Largest gradient norm log:" <<
            std::endl << "largestGradf = [";
            for(int i=0; i<largestGradf.size(); ++i) {
                log << largestGradf[i];
                if(i != largestGradf.size()-1) {
                    log << ", ";
                }
            }
            log << "];" << std::endl;
            log.close();
            reset_f();
        };
        
        //Compute Tutte parametrization.
        Eigen::MatrixXd W0t;
        const double tutte_t0 = igl::get_seconds();
        par::tutte(V, F, W0t);
        const double tutte_t = igl::get_seconds() - tutte_t0;
        log << " Tutte time: " << tutte_t << "s, inverted triangles: " <<
        inverted_triangles(W0t) << ", L2 distortion: " <<
        l2_area_distortion(W0t) << ", Linf area distortion: " <<
        linf_area_distortion(W0t) << "." << std::endl;
        igl::writeOBJ(std::string("tutte_")+mesh, V, F, Eigen::MatrixXd(),
                      Eigen::MatrixXi(), W0t, F);
        igl::writeOBJ(std::string("tutte_UV_")+mesh, W0t, F);
        igl::writeOBJ(std::string("tutte_flipped_")+mesh, W0t,
                      inverted_triangles_as_elements(W0t));
        
        //Compute UV map using our symmetric gradient, initialized with Tutte.
        Eigen::MatrixXd Wsgt = W0t;
        reset_f();
        const double sgTutte_t0 = igl::get_seconds();
        const bool sgTutteParam = par::map_to
        <parallelize, par::EnergyType::SymmetricGradient>(V, F, Wsgt, opts, f);
        const double sgTutte_t = igl::get_seconds() - sgTutte_t0;
        log << " Symmetric gradient, initialized with Tutte success: " <<
        sgTutteParam << ", time: " << sgTutte_t << "s, inverted triangles: " <<
        inverted_triangles(Wsgt) << ", L2 distortion: " <<
        l2_area_distortion(Wsgt) << ", Linf area distortion: " <<
        linf_area_distortion(Wsgt) << "." << std::endl;
        igl::writeOBJ(std::string("sgTutteInitialization_")+mesh,
                      V, F, Eigen::MatrixXd(), Eigen::MatrixXi(), Wsgt, F);
        igl::writeOBJ(std::string("sgTutteInitialization_UV_")+mesh, Wsgt, F);
        igl::writeOBJ(std::string("sgTutteInitialization_flipped_")+mesh, Wsgt,
                      inverted_triangles_as_elements(Wsgt));
        save_iter_data_to(std::string("sgTutteInitialization_")+mesh);
        
        Eigen::MatrixXd Wsdt = W0t;
        const double sdTutte_t0 = igl::get_seconds();
        const bool sdTutteParam = par::map_to
        <parallelize, par::EnergyType::SymmetricDirichlet>(V, F, Wsdt, opts);
        const double sdTutte_t = igl::get_seconds() - sdTutte_t0;
        log << " Symmetric Dirichlet, initialized with Tutte success: " <<
        sdTutteParam << ", time: " << sdTutte_t << "s, inverted triangles: " <<
        inverted_triangles(Wsdt) << ", L2 distortion: " <<
        l2_area_distortion(Wsdt) << ", Linf area distortion: " <<
        linf_area_distortion(Wsdt) << "." << std::endl;
        igl::writeOBJ(std::string("sdTutteInitialization_")+mesh,
                      V, F, Eigen::MatrixXd(), Eigen::MatrixXi(), Wsdt, F);
        igl::writeOBJ(std::string("sdTutteInitialization_UV_")+mesh, Wsdt, F);
        igl::writeOBJ(std::string("sdTutteInitialization_flipped_")+mesh, Wsdt,
                      inverted_triangles_as_elements(Wsdt));
        
#ifdef SUITESPARSE_AVAILABLE
        Eigen::MatrixXd W0bff;
        if(mesh != "brain.obj" && mesh != "tree.obj")
        {
            //Compute BFF parametrization.
            par::OptimizationOptions<double> bffOpts;
            bffOpts.proximalBoundsActive = false;
            const double bff_t0 = igl::get_seconds();
            par::bff(V, F, W0bff);
            const double bff_t = igl::get_seconds() - bff_t0;
            log << " BFF time: " << bff_t << "s, inverted triangles: " <<
            inverted_triangles(W0bff) << ", L2 distortion: " <<
            l2_area_distortion(W0bff) << ", Linf area distortion: " <<
            linf_area_distortion(W0bff) << "." << std::endl;
            igl::writeOBJ(std::string("bff_")+mesh, V, F, Eigen::MatrixXd(),
                          Eigen::MatrixXi(), W0bff, F);
            igl::writeOBJ(std::string("bff_UV_")+mesh, W0bff, F);
            igl::writeOBJ(std::string("bff_flipped_")+mesh, W0bff,
                          inverted_triangles_as_elements(W0bff));
            
            //Compute UV map using our symmetric gradient, initialized with BFF.
            Eigen::MatrixXd Wsgbff = W0bff;
            reset_f();
            const double sgBff_t0 = igl::get_seconds();
            const bool sgBffParam = par::map_to
            <parallelize, par::EnergyType::SymmetricGradient>
            (V, F, Wsgbff, bffOpts, f);
            const double sgBff_t = igl::get_seconds() - sgBff_t0;
            log << " Symmetric Gradient, initialized with BFF success: " <<
            sgBffParam << ", time: " << sgBff_t << "s, inverted triangles: " <<
            inverted_triangles(Wsgbff) << ", L2 distortion: " <<
            l2_area_distortion(Wsgbff) << ", Linf area distortion: " <<
            linf_area_distortion(Wsgbff) << "." << std::endl;
            igl::writeOBJ(std::string("sgBffInitialization_")+mesh,
                          V, F, Eigen::MatrixXd(), Eigen::MatrixXi(), Wsgbff, F);
            igl::writeOBJ(std::string("sgBffInitialization_UV_")+mesh, Wsgbff, F);
            igl::writeOBJ(std::string("sgBffInitialization_flipped_")+mesh, Wsgbff,
                          inverted_triangles_as_elements(Wsgbff));
            save_iter_data_to(std::string("sgBffInitialization_")+mesh);
            
            //Compute UV map using our symmetric Dirichlet, initialized with BFF.
            Eigen::MatrixXd Wsdbff = W0bff;
            const double sdBff_t0 = igl::get_seconds();
            const bool sdBffParam = par::map_to
            <parallelize, par::EnergyType::SymmetricDirichlet>
            (V, F, Wsdbff, bffOpts);
            const double sdBff_t = igl::get_seconds() - sdBff_t0;
            log << " Symmetric Dirichlet, initialized with BFF success: " <<
            sdBffParam << ", time: " << sdBff_t << "s, inverted triangles: " <<
            inverted_triangles(Wsdbff) << ", L2 distortion: " <<
            l2_area_distortion(Wsdbff) << ", Linf area distortion: " <<
            linf_area_distortion(Wsdbff) << "." << std::endl;
            igl::writeOBJ(std::string("sdBffInitialization_")+mesh,
                          V, F, Eigen::MatrixXd(), Eigen::MatrixXi(), Wsdbff, F);
            igl::writeOBJ(std::string("sdBffInitialization_UV_")+mesh, Wsdbff, F);
            igl::writeOBJ(std::string("sdBffInitialization_flipped_")+mesh, Wsdbff,
                          inverted_triangles_as_elements(Wsdbff));
        }
#endif
        
        //Compute UV map using ARAP, initialized with Tutte.
        Eigen::VectorXi fixed = Eigen::VectorXi::Ones(1);
        Eigen::MatrixXd fixedTo = Eigen::MatrixXd::Zero(1,2);
        Eigen::MatrixXd Wat = W0t;
        igl::ARAPData arapTutte_data;
        arapTutte_data.energy = igl::ARAP_ENERGY_TYPE_ELEMENTS;
        const double arapTutte_t0 = igl::get_seconds();
        bool arapTutteSuccess = igl::arap_precomputation(V, F, 2, fixed,
                                                         arapTutte_data);
        const int arapTutteMaxIter = 50000;
        for(int aIter=0; aIter<arapTutteMaxIter; ++aIter) {
            Eigen::MatrixXd Wa0 = Wat;
            arapTutteSuccess = igl::arap_solve(fixedTo, arapTutte_data, Wat);
            if(!arapTutteSuccess ||
               sqrt((Wat-Wa0).squaredNorm()/Wat.rows()) < 1e-6) {
                break;
            }
            if(aIter==arapTutteMaxIter-1 ||
               igl::get_seconds()-arapTutte_t0 > 9000) {
                arapTutteSuccess = false;
                break;
            }
        }
        const double arapTutte_t = igl::get_seconds() - arapTutte_t0;
        log << " ARAP, initialized with Tutte success: " <<
        arapTutteSuccess << ", time: " << arapTutte_t <<
        "s, inverted triangles: " <<
        inverted_triangles(Wat) << ", L2 distortion: " <<
        l2_area_distortion(Wat) << ", Linf area distortion: " <<
        linf_area_distortion(Wat) << "." << std::endl;
        igl::writeOBJ(std::string("arapTutteInitialization_")+mesh,
                      V, F, Eigen::MatrixXd(), Eigen::MatrixXi(), Wat, F);
        igl::writeOBJ(std::string("arapTutteInitialization_UV_")+mesh, Wat, F);
        igl::writeOBJ(std::string("arapTutteInitialization_flipped_")+mesh, Wat,
                      inverted_triangles_as_elements(Wat));
        
#ifdef SUITESPARSE_AVAILABLE
        if(mesh != "brain.obj" && mesh != "tree.obj")
        {
            //Compute UV map using ARAP, initialized with BFF.
            Eigen::MatrixXd Wabff = W0bff;
            igl::ARAPData arapBff_data;
            arapBff_data.energy = igl::ARAP_ENERGY_TYPE_ELEMENTS;
            const double arapBff_t0 = igl::get_seconds();
            bool arapBffSuccess = igl::arap_precomputation(V, F, 2, fixed,
                                                           arapBff_data);
            const int arapBffMaxIter = 50000;
            for(int aIter=0; aIter<arapBffMaxIter; ++aIter) {
                Eigen::MatrixXd Wa0 = Wabff;
                arapBffSuccess = igl::arap_solve(fixedTo, arapBff_data, Wabff);
                if(!arapBffSuccess ||
                   sqrt((Wabff-Wa0).squaredNorm()/Wabff.rows()) < 1e-6) {
                    break;
                }
                if(aIter==arapBffMaxIter-1 ||
                   igl::get_seconds()-arapBff_t0 > 9000) {
                    arapBffSuccess = false;
                    break;
                }
            }
            const double arapBff_t = igl::get_seconds() - arapBff_t0;
            log << " ARAP, initialized with BFF success: " <<
            arapBffSuccess << ", time: " << arapBff_t << "s, inverted triangles: " <<
            inverted_triangles(Wabff) << ", L2 distortion: " <<
            l2_area_distortion(Wabff) << ", Linf area distortion: " <<
            linf_area_distortion(Wabff) << "." << std::endl;
            igl::writeOBJ(std::string("arapBFFInitialization_")+mesh,
                          V, F, Eigen::MatrixXd(), Eigen::MatrixXi(), Wabff, F);
            igl::writeOBJ(std::string("arapBFFInitialization_UV_")+mesh, Wabff, F);
            igl::writeOBJ(std::string("arapBFFInitialization_flipped_")+mesh, Wabff,
                          inverted_triangles_as_elements(Wabff));
        }
#endif
        
        log << std::endl;
        log.close();
    }
    
    return 0;
}
