
#include <parametrization/tutte.h>
#include <parametrization/bff.h>
#include <parametrization/map_to.h>

#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/flipped_triangles.h>
#include <igl/remove_unreferenced.h>
#include <igl/euler_characteristic.h>
#include <igl/get_seconds.h>
#include <igl/doublearea.h>

#include <iostream>


//This is a command-line utility to compute the UV map for an arbitrary disk
// topology mesh. Only use with disk topology meshes!
//Use as follows: ./uv_map input output
//If no output path is provided, will save the output to out.obj

int main(int argc, char *argv[]) {
    namespace par = parametrization;
#ifdef OPENMP_AVAILABLE
    constexpr bool parallelize = true;
#else
    constexpr bool parallelize = false;
#endif
    
    if(argc<2) {
        std::cerr << "No input mesh supplied." << std::endl;
        return -2;
    }
    std::string mesh = std::string(argv[1]);
    std::string out = "out.obj";
    if(argc>2) {
        out = std::string(argv[2]);
    }
    
    //Load mesh
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    if(!igl::readOBJ(mesh, V, F)) {
        std::cerr << "Error reading mesh " << mesh << "." << std::endl;
        return -3;
    }
    
    if(F.size() == 0) {
        std::cerr << "No mesh provided."
        << std::endl;
        return -4;
    }
    
    if(F.cols() != 3) {
        std::cerr << "This application only works for triangle meshes."
        << std::endl;
        return -5;
    }
    
    if(igl::euler_characteristic(F) != 1) {
        std::cerr << "This application only works for disk topology surfaces."
        << std::endl;
        return -6;
    }
    
    Eigen::VectorXd A;
    igl::doublearea(V, F, A);
    A *= 0.5;
    for(int i=0; i<A.size(); ++i) {
        if(A(i) < 1e-8) {
            std::cerr <<
            "This mesh contains a degenerate triangle (area <1e-8)." <<
            std::endl;
            return -7;
        }
    }
    
    //Remove unreferenced vertices
    Eigen::VectorXi dummy;
    igl::remove_unreferenced(Eigen::MatrixXd(V), Eigen::MatrixXi(F),
                             V, F, dummy);
    
    //Compute Tutte parametrization.
    Eigen::MatrixXd Wt;
    if(par::tutte(V, F, Wt)) {
        //Compute UV map using our symmetric gradient, initialized with Tutte.
        const double t0 = igl::get_seconds();
        if(par::map_to
           <parallelize, par::EnergyType::SymmetricGradient>(V, F, Wt)) {
            const double t = igl::get_seconds() - t0;
            if(Wt.array().isFinite().all() &&
               igl::flipped_triangles(Wt,F).size()==0) {
                std::cout <<
                "Computed parametrization by initializing with Tutte in "
                << t << "s." << std::endl;
                if(igl::writeOBJ(out, V, F,
                                 Eigen::MatrixXd(), Eigen::MatrixXi(),
                                 Wt, F)) {
                    return 0;
                } else {
                    std::cerr << "Error writing to " << out << "." << std::endl;
                    return -8;
                }
            }
        }
    }
    
    std::cerr <<
    "Unable to compute a flip-free parametrization that minimizes the energy."
    << std::endl;
    return -10;
}
