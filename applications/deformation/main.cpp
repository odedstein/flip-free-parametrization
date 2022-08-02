
#include <parametrization/map_to.h>
#include <parametrization/map_energy.h>

#include <igl/readOBJ.h>
#include <igl/readMESH.h>
#include <igl/writeOBJ.h>
#include <igl/writeMESH.h>
#include <igl/doublearea.h>
#include <igl/volume.h>
#include <igl/flipped_triangles.h>
#include <igl/get_seconds.h>
#include <igl/arap.h>
#include <igl/snap_points.h>
#include <igl/boundary_facets.h>

#include <iostream>
#include <fstream>
#include <vector>

//#define IGL_VIEWER_VIEWER_QUIET
//#include <igl/opengl/glfw/Viewer.h>

//This function computes a number of deformations for meshes.
//Do *not* use this to judge timing with respect to ARAP - ARAP uses a different
// termination criterion.

int main(int argc, char *argv[]) {
    namespace par = parametrization;
#ifdef OPENMP_AVAILABLE
    constexpr bool parallelize = true;
#else
    constexpr bool parallelize = false;
#endif
    
    for(int i=0; i<10; ++i) {
        std::string mesh;
        bool tetMesh = false;
        Eigen::MatrixXd fixedCoords, fixedTo;
        
        switch(i) {
            case 0:
                mesh = "worm.obj";
                fixedCoords.resize(4,3);
                fixedCoords << -0.3858, -0.033695, 0,
                -0.34129, -0.068956, 0,
                0.51824, -0.01719, 0,
                0.51899, -0.042698, 0;
                fixedTo.resize(4,2);
                fixedTo << -0.27864, -0.033695,
                -0.24103, -0.068956,
                0.39459, -0.01719,
                0.37519, -0.042698;
                break;
            case 1:
            {
                mesh = "octopus.obj";
                fixedCoords.resize(8,3);
                fixedCoords << -0.36426, -0.10893, 0,
                -0.29666, -0.063601, 0,
                -0.015116, 0.26884, 0,
                0.018288, 0.22192, 0,
                0.39368, 0.0079776, 0,
                0.36187, -0.079508, 0,
                0.19962, -0.20437, 0,
                -0.090672, -0.24334, 0;
                fixedTo.resize(8,2);
                fixedTo << -0.36539, -0.1598,
                -0.40293, 0.19076,
                -0.17806, 0.45589,
                0.2858, 0.42626,
                0.49736, 0.20448,
                0.47049, -0.18517,
                0.25233, -0.36103,
                -0.17495, -0.38941;
                break;
            }
            case 2:
            {
                mesh = "pants.obj";
                fixedCoords.resize(6,3);
                fixedCoords << -0.13264, 0.33818, 0,
                0.12512, 0.33718, 0,
                -0.13064, 0.01848, 0,
                0.15509, 0.010488, 0,
                -0.19158, -0.32819, 0,
                0.23202, -0.3272, 0;
                fixedTo.resize(6,2);
                fixedTo << -0.091254, 0.32035,
                0.08208, 0.34083,
                -0.11466, 0.015483,
                0.14103, 0.005466,
                -0.47594, -0.065472,
                0.50505, -0.078897;
                break;
            }
            case 3:
            {
                mesh = "arm.obj";
                fixedCoords.resize(3,3);
                fixedCoords << 0.1632, 0.3123, 0,
                0.15387, -0.10669, 0,
                -0.37623, -0.11941, 0;
                fixedTo.resize(3,2);
                fixedTo << 0.30592, 0.00020562,
                -0.036967, 0.083299,
                -0.61536, -0.00028471;
                break;
            }
            case 4:
            {
                mesh = "crocodile.obj";
                fixedCoords.resize(12,3);
                fixedCoords << -0.41197, 0.012295, 0,
                -0.40441, -0.026277, 0,
                -0.21004, 0.030446, 0,
                -0.21004, -0.014933, 0,
                -0.06785, -0.028546, 0,
                0.18854, -0.023252, 0,
                -0.077682, -0.076194, 0,
                0.18854, -0.07922, 0,
                -0.066337, 0.031203, 0,
                0.19005, 0.022883, 0,
                0.33905, 0.013051, 0,
                0.52586, -0.0088821, 0;
                fixedTo.resize(12,2);
                fixedTo << -0.3848, 0.12535,
                -0.40743, -0.095616,
                -0.19496, 0.018387,
                -0.20627, -0.026238,
                -0.050515, -0.024778,
                0.1501, -0.033804,
                -0.028692, -0.067904,
                0.14031, -0.074697,
                -0.066337, 0.031203,
                0.16744, 0.0085632,
                0.28554, 0.072592,
                0.42939, -0.011897;
                break;
            }
            case 5:
            {
                tetMesh = true;
                mesh = "hand.mesh";
                fixedCoords.resize(19,3);
                fixedCoords << -0.231413, 0.344762, 0.118337,
                -0.100013, 0.523663, 0.0636985,
                -0.0151988, 0.527535, 0.0563735,
                0.0683396, 0.48567, 0.0702074,
                0.140719, 0.425267, 0.0919583,
                -0.126413, 0.667596, 0.0523681,
                -0.13133, 0.786928, 0.0567782,
                -0.127036, 0.873781, 0.0584105,
                -0.304659, 0.443623, 0.140268,
                -0.378721, 0.527083, 0.143177,
                0.0064905, 0.682575, 0.0473417,
                0.0368988, 0.904401, 0.0485683,
                0.12128, 0.644307, 0.0640443,
                0.151384, 0.754502, 0.0667948,
                0.166307, 0.824097, 0.0649974,
                0.0296643, 0.810993, 0.0558126,
                0.229144, 0.536029, 0.0940584,
                0.266305, 0.60478, 0.101968,
                0.294675, 0.659962, 0.10702;
                fixedTo.resize(19,3);
                fixedTo << -0.231413, 0.344762, 0.118337,
                -0.100013, 0.523663, 0.0636985,
                -0.0151988, 0.527535, 0.0563735,
                0.0683396, 0.48567, 0.0702074,
                0.140719, 0.425267, 0.0919583,
                -0.125951, 0.66947, -0.00580951,
                -0.136614, 0.758288, 0.101968,
                -0.14856, 0.716009, 0.213067,
                -0.312316, 0.436557, 0.121986,
                -0.208358, 0.511161, 0.192034,
                0.0104992, 0.684705, -0.0188853,
                -0.00636762, 0.740082, 0.27455,
                0.136288, 0.640137, -0.00323181,
                0.13043, 0.720798, 0.12678,
                0.120913, 0.684477, 0.264911,
                0.00902367, 0.768794, 0.132834,
                0.242577, 0.524916, 0.0465791,
                0.236224, 0.590074, 0.147995,
                0.177656, 0.558437, 0.232307;
                break;
            }
            case 6:
            {
                tetMesh = true;
                mesh = "armadillo.mesh";
                fixedCoords.resize(15,3);
                fixedCoords << -0.138455, -0.526708, -0.149503,
                -0.209294, -0.544444, -0.0542189,
                0.205483, -0.520584, -0.0942188,
                0.253515, -0.541037, -0.0428974,
                0.200815, -0.283211, -0.0504314,
                -0.180924, -0.302315, -0.0630498,
                0.00112989, -0.110042, -0.0851767,
                -0.00754953, 0.268488, 0.055071,
                -0.383779, 0.285708, 0.161649,
                -0.170311, 0.153514, -0.0659577,
                0.159383, 0.163966, -0.0465278,
                -0.361434, 0.225589, 0.0819021,
                0.334726, 0.204425, 0.150611,
                0.364609, 0.240136, 0.265444,
                -0.00920906, 0.196405, 0.203676;
                fixedTo.resize(15,3);
                fixedTo << -0.128887, -0.406052, -0.156451,
                -0.223954, -0.4088, -0.0441587,
                0.171375, -0.390557, -0.127062,
                0.27535, -0.400485, -0.0531622,
                0.211306, -0.211337, -0.0387405,
                -0.187379, -0.220258, -0.0558686,
                0.00112989, -0.110042, -0.0851767,
                -0.000735164, 0.213083, 0.0267864,
                -0.107849, 0.383502, -0.0153556,
                -0.164616, 0.189593, -0.0582529,
                0.167005, 0.193006, -0.038654,
                -0.248988, 0.315881, 0.0320041,
                0.252982, 0.299028, 0.0556022,
                0.124448, 0.377738, 0.00306424,
                -0.00439107, 0.140396, 0.162033;
                break;
            }
            case 7:
            {
                tetMesh = true;
                mesh = "octopus-vol.mesh";
                fixedCoords.resize(11,3);
                fixedCoords << 0.0266997, -0.0228399, 0.0232984,
                0.0165755, -0.10201, -0.130327,
                0.142534, 0.386267, -0.128028,
                -0.188315, 0.144619, -0.0742034,
                0.0305459, 0.00175212, 0.0372242,
                -0.458731, 0.0818953, 0.0565088,
                0.179321, -0.0521324, 0.155486,
                0.530846, -0.0810217, 0.0853634,
                -0.257624, -0.149311, 0.379855,
                0.102436, -0.299553, 0.252436,
                -0.129705, 0.403408, -0.0932289;
                fixedTo.resize(11,3);
                fixedTo << 0.0446894, -0.0548279, -0.00642269,
                0.128472, -0.0701505, -0.162038,
                0.267128, 0.225692, 0.753398,
                0.036476, 0.162771, 0.430022,
                0.03253, -0.0021171, 0.044963,
                0.0820029, 0.207604, 0.756762,
                0.0709422, 0.0669447, 0.462737,
                0.296053, 0.0909071, 0.774987,
                0.144613, 0.128841, 0.824391,
                0.179199, 0.0872231, 0.780522,
                0.148748, 0.296122, 0.719084;
                break;
            }
            case 8:
            {
                tetMesh = true;
                mesh = "stuffedtoy.mesh";
                fixedCoords.resize(14,3);
                fixedCoords << -0.103604, 0.231932, -0.0157562,
                -0.050943, 0.235356, -0.00420699,
                0.086027, 0.237526, 0.00173399,
                0.0173508, 0.233932, -0.0137561,
                0.0701672, 0.0541778, 0.101669,
                -0.112029, 0.0526436, 0.0908521,
                -0.104364, 0.377044, 0.00552275,
                0.0973127, 0.37595, -0.00993531,
                -0.153461, 0.653483, 0.00863374,
                0.122721, 0.659915, 0.0119336,
                0.165332, 0.303507, -0.016481,
                -0.200786, 0.308624, -0.0141658,
                0.186595, 0.226377, 0.0102259,
                -0.24266, 0.243994, -0.00210259;
                fixedTo.resize(14,3);
                fixedTo << -0.103604, 0.231932, -0.0157562,
                -0.050943, 0.235356, -0.00420699,
                0.086027, 0.237526, 0.00173399,
                0.0173508, 0.233932, -0.0137561,
                0.0674277, 0.29763, 0.156811,
                -0.10696, 0.297148, 0.169426,
                -0.10159, 0.399348, 0.04023,
                0.0874224, 0.402838, 0.0135538,
                -0.0948747, 0.630545, 0.0843448,
                0.0554023, 0.691216, -0.101008,
                0.161991, 0.320423, -0.0149208,
                -0.198701, 0.333951, 0.0586142,
                0.150444, 0.27216, 0.0508868,
                -0.177015, 0.348547, 0.114475;
                break;
            }
            case 9:
            {
                tetMesh = true;
                mesh = "slug.mesh";
                fixedCoords.resize(8,3);
                fixedCoords << 0.140085, 0.19713, -0.010251,
                0.168262, 0.0578936, -0.0138428,
                0.0833628, 0.080183, -0.0884307,
                0.0846907, 0.112862, 0.0776426,
                -0.265895, 0.128475, -0.0262649,
                -0.260717, 0.0377528, 0.00975559,
                -0.555763, 0.0182125, 0.0723817,
                -0.555752, 0.0329755, -0.0682109;
                fixedTo.resize(8,3);
                fixedTo << 0.144934, 0.1749, -0.00947702,
                0.14501, 0.0640153, -0.0132461,
                0.0880017, 0.0870818, -0.13179,
                0.0782843, 0.113635, 0.102504,
                -0.0491678, 0.142624, -0.00956392,
                -0.0575573, 0.029567, 0.00301173,
                -0.415451, 0.0358429, 0.0349758,
                -0.411847, 0.0376907, -0.100439;
                break;
            }
            default:
                abort();
        }
        
        std::string meshOBJ = mesh + std::string(".obj");
        
        std::cout << "Computing deformation for mesh " << mesh << "." << std::endl;
        
        //Create log file
        std::ofstream log;
        log.open(mesh + std::string(".log.txt"));
        log << mesh << std::endl;
        
        //Load mesh
        Eigen::MatrixXd V;
        Eigen::MatrixXi F, tF, bdF;
        const std::string meshPath = std::string(MESH_DIRECTORY) + mesh;
        if(tetMesh) {
            igl::readMESH(meshPath, V, F, tF);
            igl::boundary_facets(F, bdF);
            Eigen::VectorXi tmp = bdF.col(2);
            bdF.col(2) = bdF.col(1);
            bdF.col(1) = tmp;
            igl::writeOBJ(meshOBJ, V, bdF);
        } else {
            igl::readOBJ(meshPath, V, F);
        }
        
        //Fix vertices from fixedCoords
        Eigen::VectorXi fixed;
        igl::snap_points(fixedCoords, V, fixed);
        log << " fixed vertex IDs: ";
        for(int j=0; j<fixed.size(); ++j) {
            log << "  " << fixed(j) << " to " << fixedTo.row(j) << std::endl;
        }
        
        //Mesh area
        Eigen::VectorXd A;
        if(tetMesh) {
            igl::volume(V, F, A);
        } else {
            igl::doublearea(V, F, A);
            A *= 0.5;
        }
        
        //Helper functions for statistics
        const auto l2_area_distortion = [&] (const auto& W) {
            Eigen::VectorXd AW;
            if(tetMesh) {
                igl::volume(W, F, AW);
            } else {
                igl::doublearea(W, F, AW);
                AW *= 0.5;
            }
            return sqrt((A.array() * (A-AW).array().abs().pow(2)).sum());
        };
        const auto linf_area_distortion = [&] (const auto& W) {
            Eigen::VectorXd AW;
            if(tetMesh) {
                igl::volume(W, F, AW);
            } else {
                igl::doublearea(W, F, AW);
                AW *= 0.5;
            }
            return (A-AW).array().abs().maxCoeff();
        };
        const auto inverted_triangles_as_elements = [&] (const auto& W) {
            if(tetMesh) {
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
            } else {
                const Eigen::VectorXi flippedInds = igl::flipped_triangles(W,F);
                Eigen::MatrixXi flippedF(flippedInds.size(), 3);
                for(int i=0; i<flippedInds.size(); ++i) {
                    flippedF.row(i) = F.row(flippedInds(i));
                }
                return flippedF;
            }
        };
        const auto inverted_triangles = [&] (const auto& W) -> int {
            return inverted_triangles_as_elements(W).rows();
        };
        const auto flipped_tet_boundaries = [&] (const auto& W) {
            const Eigen::MatrixXi F = inverted_triangles_as_elements(W);
            Eigen::MatrixXi bdF;
            igl::boundary_facets(F, bdF);
            Eigen::VectorXi tmp = bdF.col(2);
            bdF.col(2) = bdF.col(1);
            bdF.col(1) = tmp;
            return bdF;
        };
        
        
        //Variables needed for map_to
        Eigen::SparseMatrix<double> Aeq;
        Eigen::MatrixXd beq;
        par::OptimizationOptions<double> opts;
        opts.pAbsTol = 5e-10;
        opts.pRelTol = 5e-9;
        opts.dAbsTol = 5e-10;
        opts.dRelTol = 5e-9;
        opts.maxIter = 1000000;
        
        
        //Compute deformation using ARAP.
        Eigen::MatrixXd Wa;
        if(tetMesh) {
            Wa = V;
        } else {
            Wa = V.block(0,0,V.rows(),2);
        }
        igl::ARAPData arap_data;
        arap_data.energy = igl::ARAP_ENERGY_TYPE_ELEMENTS;
        const double arap_t0 = igl::get_seconds();
        bool arapSuccess = igl::arap_precomputation(V, F, Wa.cols(), fixed,
                                                    arap_data);
        const int arapMaxIter = opts.maxIter;
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
        log << " ARAP success: " << arapSuccess <<
        ", time: " << arap_t << "s, inverted elements: " <<
        inverted_triangles(Wa) << ", L2 distortion: " <<
        l2_area_distortion(Wa) << ", Linf distortion: " <<
        linf_area_distortion(Wa) << ", sg map energy: " <<
        par::map_energy<par::EnergyType::SymmetricGradient>(V,F,Wa) <<
        ", sd map energy: " <<
        par::map_energy<par::EnergyType::SymmetricDirichlet>(V,F,Wa)
        << "." << std::endl;
        if(tetMesh) {
            igl::writeOBJ(std::string("ARAP_deformed_")+meshOBJ, Wa, bdF);
            igl::writeMESH(std::string("ARAP_deformed_")+mesh, Wa, F, tF);
            igl::writeOBJ(std::string("ARAP_deformed_flipped_")+mesh+std::string(".obj"), Wa,
                          flipped_tet_boundaries(Wa));
        } else {
            igl::writeOBJ(std::string("ARAP_deformed_")+mesh, Wa, F);
            igl::writeOBJ(std::string("ARAP_deformed_flipped_")+mesh, Wa,
                          inverted_triangles_as_elements(Wa));
        }
        
        
        //Compute deformation using our symmetric gradient.
        Eigen::MatrixXd Wsg;
        if(tetMesh) {
            Wsg = V;
        } else {
            Wsg = V.block(0,0,V.rows(),2);
        }
        
        const double sg_t0 = igl::get_seconds();
        bool sgSuccess = par::map_to
        <parallelize, par::EnergyType::SymmetricGradient>
        (V, F, fixed, fixedTo, Aeq, beq, Wsg, opts);
        const double sg_t = igl::get_seconds() - sg_t0;
        log << " Symmetric gradient success: " << sgSuccess <<
        ", time: " << sg_t << "s, inverted elements: " <<
        inverted_triangles(Wsg) << ", L2 distortion: " <<
        l2_area_distortion(Wsg) << ", Linf distortion: " <<
        linf_area_distortion(Wsg) << ", sg map energy: " <<
        par::map_energy<par::EnergyType::SymmetricGradient>(V,F,Wsg) <<
        ", sd map energy: " <<
        par::map_energy<par::EnergyType::SymmetricDirichlet>(V,F,Wsg) <<
        "." << std::endl;
        if(tetMesh) {
            igl::writeOBJ(std::string("sg_deformed_")+meshOBJ, Wsg, bdF);
            igl::writeMESH(std::string("sg_deformed_")+mesh, Wsg, F, tF);
            igl::writeOBJ(std::string("sg_deformed_flipped_")+mesh+std::string(".obj"), Wsg,
                          flipped_tet_boundaries(Wsg));
        } else {
            igl::writeOBJ(std::string("sg_deformed_")+mesh, Wsg, F);
            igl::writeOBJ(std::string("sg_deformed_flipped_")+mesh, Wsg,
                          inverted_triangles_as_elements(Wsg));
        }
        
        
        //Compute deformation using our symmetric Dirichlet.
        Eigen::MatrixXd Wsd;
        if(tetMesh) {
            Wsd = V;
        } else {
            Wsd = V.block(0,0,V.rows(),2);
        }
        
        const double sd_t0 = igl::get_seconds();
        const bool sdSuccess = par::map_to
        <parallelize, par::EnergyType::SymmetricDirichlet>
        (V, F, fixed, fixedTo, Aeq, beq, Wsd, opts);
        const double sd_t = igl::get_seconds() - sd_t0;
        log << " Symmetric Dirichlet success: " << sdSuccess <<
        ", time: " << sd_t << "s, inverted elements: " <<
        inverted_triangles(Wsd) << ", L2 distortion: " <<
        l2_area_distortion(Wsd) << ", Linf distortion: " <<
        linf_area_distortion(Wsd) << ", sg map energy: " <<
        par::map_energy<par::EnergyType::SymmetricGradient>(V,F,Wsd) <<
        ", sd map energy: " <<
        par::map_energy<par::EnergyType::SymmetricDirichlet>(V,F,Wsd) <<
        "." << std::endl;
        if(tetMesh) {
            igl::writeOBJ(std::string("sd_deformed_")+meshOBJ, Wsd, bdF);
            igl::writeMESH(std::string("sd_deformed_")+mesh, Wsd, F, tF);
            igl::writeOBJ(std::string("sd_deformed_flipped_")+mesh+std::string(".obj"), Wsd,
                          flipped_tet_boundaries(Wsg));
        } else {
            igl::writeOBJ(std::string("sd_deformed_")+mesh, Wsg, F);
            igl::writeOBJ(std::string("sd_deformed_flipped_")+mesh, Wsd,
                          inverted_triangles_as_elements(Wsd));
        }
        
//        {
//            using Viewer = igl::opengl::glfw::Viewer;
//            Viewer viewer;
//            if(tetMesh) {
//                viewer.data().set_mesh(V, bdF);
//            } else {
//                viewer.data().set_mesh(V, F);
//            }
//            viewer.launch();
//        }
//        {
//            using Viewer = igl::opengl::glfw::Viewer;
//            Viewer viewer;
//            if(tetMesh) {
//                viewer.data().set_mesh(Wsg, bdF);
//            } else {
//                viewer.data().set_mesh(Wsg, F);
//            }
//            viewer.launch();
//        }
//        {
//            using Viewer = igl::opengl::glfw::Viewer;
//            Viewer viewer;
//            if(tetMesh) {
//                viewer.data().set_mesh(Wsd, bdF);
//            } else {
//                viewer.data().set_mesh(Wsd, F);
//            }
//            viewer.launch();
//        }
//        {
//            using Viewer = igl::opengl::glfw::Viewer;
//            Viewer viewer;
//            if(tetMesh) {
//                viewer.data().set_mesh(Wa, bdF);
//            } else {
//                viewer.data().set_mesh(Wa, F);
//            }
//            viewer.launch();
//        }
        
        log << std::endl;
        log.close();
    }
    
    return 0;
}
