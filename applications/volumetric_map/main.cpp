
#include <parametrization/map_to.h>
#include <parametrization/map_energy.h>
#include <parametrization/tutte.h>

#include <igl/readOBJ.h>
#include <igl/readMESH.h>
#include <igl/writeOBJ.h>
#include <igl/writeMESH.h>
#include <igl/doublearea.h>
#include <igl/volume.h>
#include <igl/flipped_triangles.h>
#include <igl/get_seconds.h>
#include <igl/arap.h>
#include <igl/boundary_facets.h>
#include <igl/unique.h>

#include <iostream>
#include <fstream>
#include <vector>

//This function computes a number of (partial) volumetric maps from a source
// tet mesh to a variety of pose tetmeshes.
//Do *not* use this to judge timing with respect to ARAP - ARAP uses a different
// termination criterion.

int main(int argc, char *argv[]) {
    namespace par = parametrization;
#ifdef OPENMP_AVAILABLE
    constexpr bool parallelize = true;
#else
    constexpr bool parallelize = false;
#endif
    
    for(int i=0; i<2; ++i) {
        std::string tetMesh;
        int nPoses;
        switch(i) {
            case 0:
                tetMesh = "human.mesh";
                nPoses = 10;
                break;
            case 1:
                tetMesh = "bear.mesh";
                nPoses = 2;
        }
        
        std::cout << "Computing volumetric map for mesh " << tetMesh << "." << std::endl;
        
        //Create log file
        std::ofstream log;
        log.open(tetMesh + std::string(".log.txt"));
        log << tetMesh << std::endl;
        
        //Load mesh
        Eigen::MatrixXd V;
        Eigen::MatrixXi F, tF, bdF;
        const std::string meshPath = std::string(MESH_DIRECTORY) + tetMesh;
        const std::string meshOBJ = tetMesh + std::string(".obj");
        igl::readMESH(meshPath, V, F, tF);
        igl::boundary_facets(F, bdF);
        Eigen::VectorXi tmp = bdF.col(2);
        bdF.col(2) = bdF.col(1);
        bdF.col(1) = tmp;
        igl::writeOBJ(meshOBJ, V, bdF);
        
        //Statistics for mesh
        Eigen::VectorXd A;
        igl::volume(V, F, A);
        
        //Compute boundary and create fixed vertex ids.
        Eigen::VectorXi bdFDuplicated(3*bdF.rows());
        bdFDuplicated.segment(0,bdF.rows()) = bdF.col(0);
        bdFDuplicated.segment(bdF.rows(),bdF.rows()) = bdF.col(1);
        bdFDuplicated.segment(2*bdF.rows(),bdF.rows()) = bdF.col(2);
        Eigen::VectorXi fixed;
        igl::unique(bdFDuplicated, fixed);
        
        //Go through all poses
        for(int j=0; j<nPoses; ++j) {
            //Variables needed for map_to
            Eigen::SparseMatrix<double> Aeq;
            Eigen::MatrixXd beq;
            par::OptimizationOptions<double> opts;
            opts.proximalBoundsActive = false;
            
            std::string poseMesh;
            switch(i) {
                case 0:
                    poseMesh = std::string("human-") + std::to_string(j) +
                    std::string(".obj");
                    break;
                case 1:
                    poseMesh = std::string("bear-") + std::to_string(j) +
                    std::string(".obj");
                    break;
            }
            
            log << " pose " << j << std::endl;
            
            //Load pose surface
            Eigen::MatrixXd poseV;
            Eigen::MatrixXi poseF;
            igl::readOBJ(std::string(MESH_DIRECTORY)+poseMesh, poseV, poseF);
            
            //Construct fixedTo using correspondence between the vertices in
            // idV,idF and poseV,poseF.
            Eigen::MatrixXd fixedTo(fixed.size(), 3);
            for(int k=0; k<fixed.size(); ++k) {
                fixedTo.row(k) = poseV.row(fixed(k));
            }
            
            //Helper functions for statistics
            const auto l2_area_distortion = [&] (const auto& W) {
                Eigen::VectorXd AW;
                igl::volume(W, F, AW);
                return sqrt((A.array() * (A-AW).array().abs().pow(2)).sum());
            };
            const auto linf_area_distortion = [&] (const auto& W) {
                Eigen::VectorXd AW;
                igl::volume(W, F, AW);
                return (A-AW).array().abs().maxCoeff();
            };
            const auto inverted_tets_as_elements = [&] (const auto& W) {
                    Eigen::MatrixXi flippedF(0, 4);
                    for(int i=0; i<F.rows(); ++i) {
                        const Eigen::Vector3d v0=W.row(F(i,0)), v1=W.row(F(i,1)),
                        v2=W.row(F(i,2)), v3=W.row(F(i,3));
                        Eigen::Matrix3d T;
                        T.col(0) = v1 - v0;
                        T.col(1) = v2 - v0;
                        T.col(2) = v3 - v0;
                        if(T.determinant()<0) {
                            flippedF.conservativeResize(flippedF.rows()+1, 4);
                            flippedF.row(flippedF.rows()-1) = F.row(i);
                        }
                    }
                    return flippedF;
            };
            const auto inverted_tets = [&] (const auto& W) -> int {
                return inverted_tets_as_elements(W).rows();
            };
            const auto flipped_tet_boundaries = [&] (const auto& W) {
                const Eigen::MatrixXi F = inverted_tets_as_elements(W);
                Eigen::MatrixXi bdF;
                igl::boundary_facets(F, bdF);
                Eigen::VectorXi tmp = bdF.col(2);
                bdF.col(2) = bdF.col(1);
                bdF.col(1) = tmp;
                return bdF;
            };
            
            //Helper functions to write tet slices
            const auto slice_coord = [&]
            (const auto& W, const std::string s, const int coord) {
                const double max = W.col(coord).maxCoeff()+1e-4,
                min = W.col(coord).minCoeff()-1e-4;
                for(int i=0; i<10; ++i) {
                    const double thresh = double(i)/double(10)*(max-min) + min;
                    Eigen::MatrixXi slicedF(0,4);
                    for(int j=0; j<F.rows(); ++j) {
                        if(W(F(j,0),coord)>=thresh && W(F(j,1),coord)>=thresh &&
                           W(F(j,2),coord)>=thresh && W(F(j,3),coord)>=thresh) {
                            slicedF.conservativeResize(slicedF.rows()+1,4);
                            slicedF.row(slicedF.rows()-1) = F.row(j);
                        }
                    }
                    Eigen::MatrixXi bF;
                    igl::boundary_facets(slicedF, bF);
                    Eigen::VectorXi tmp = bF.col(2);
                    bF.col(2) = bF.col(1);
                    bF.col(1) = tmp;
                    igl::writeOBJ(s+std::string("_slice_") + std::to_string(i)
                                  + std::string(".obj"), W, bF);
                }
            };
            const auto slice = [&] (const auto& W, const std::string s) {
                for(int i=0; i<3; ++i) {
                    slice_coord
                    (W, s+std::string("_coord_")+std::to_string(i), i);
                }
            };
            
            //Compute Tutte embedding of given fixed data.
            Eigen::MatrixXd Wt;
            const double tutte_t0 = igl::get_seconds();
            const bool tutteSuccess = par::tutte(V, F, fixed, fixedTo, Aeq, beq,
                                                 Wt);
            const double tutte_t = igl::get_seconds() - tutte_t0;
            log << "  Tutte success: " << tutteSuccess <<
            ", time: " << tutte_t << "s, inverted elements: " <<
            inverted_tets(Wt) << ", L2 distortion: " <<
            l2_area_distortion(Wt) << ", Linf distortion: " <<
            linf_area_distortion(Wt) << ", sg map energy: " <<
            par::map_energy<par::EnergyType::SymmetricGradient>(V,F,Wt) <<
            ", sd map energy: " <<
            par::map_energy<par::EnergyType::SymmetricDirichlet>(V,F,Wt) <<
            "." << std::endl;
            if(!tutteSuccess) {
                continue;
            }
            const std::string tuttePath = std::string("tutte_") + tetMesh +
            std::string("_pose_") + std::to_string(j);
            igl::writeMESH(tuttePath + std::string(".mesh"), Wt, F, tF);
            slice(Wt, tuttePath);
            igl::writeOBJ(tuttePath + std::string("_flipped.obj"), Wt,
                          flipped_tet_boundaries(Wt));
            
            
            //Compute Symmetric gradient embedding.
            Eigen::MatrixXd Wsg = Wt;
            const double sg_t0 = igl::get_seconds();
            const bool sgSuccess = par::map_to
            <parallelize, par::EnergyType::SymmetricGradient>
            (V, F, fixed, fixedTo, Aeq, beq, Wsg, opts);
            const double sg_t = igl::get_seconds() - sg_t0;
            log << "  Symmetric gradient success: " << sgSuccess <<
            ", time: " << sg_t << "s, inverted elements: " <<
            inverted_tets(Wsg) << ", L2 distortion: " <<
            l2_area_distortion(Wsg) << ", Linf distortion: " <<
            linf_area_distortion(Wsg) << ", sg map energy: " <<
            par::map_energy<par::EnergyType::SymmetricGradient>(V,F,Wsg) <<
            ", sd map energy: " <<
            par::map_energy<par::EnergyType::SymmetricDirichlet>(V,F,Wsg) <<
            "." << std::endl;
            const std::string sgPath = std::string("sg_") + tetMesh +
            std::string("_pose_") + std::to_string(j);
            igl::writeMESH(sgPath + std::string(".mesh"), Wsg, F, tF);
            slice(Wsg, sgPath);
            igl::writeOBJ(sgPath + std::string("_flipped.obj"), Wsg,
                          flipped_tet_boundaries(Wsg));
            
            //Compute ARAP embedding.
            Eigen::MatrixXd Wa = Wt;
            igl::ARAPData arap_data;
            arap_data.energy = igl::ARAP_ENERGY_TYPE_ELEMENTS;
            const double arap_t0 = igl::get_seconds();
            bool arapSuccess = igl::arap_precomputation(V, F, Wa.cols(), fixed,
                                                        arap_data);
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
            const double arap_t = igl::get_seconds() - arap_t0;
            log << "  ARAP success: " << arapSuccess <<
            ", time: " << arap_t << "s, inverted elements: " <<
            inverted_tets(Wa) << ", L2 distortion: " <<
            l2_area_distortion(Wa) << ", Linf distortion: " <<
            linf_area_distortion(Wa) << ", sg map energy: " <<
            par::map_energy<par::EnergyType::SymmetricGradient>(V,F,Wa) <<
            ", sd map energy: " <<
            par::map_energy<par::EnergyType::SymmetricDirichlet>(V,F,Wa)
            << "." << std::endl;
            const std::string arapPath = std::string("ARAP_") + tetMesh +
            std::string("_pose_") + std::to_string(j);
            igl::writeMESH(arapPath + std::string(".mesh"), Wa, F, tF);
            slice(Wa, arapPath);
            igl::writeOBJ(arapPath + std::string("_flipped.obj"), Wa,
                          flipped_tet_boundaries(Wa));
        }
        
        log << std::endl;
        log.close();
    }
    
    return 0;
}
