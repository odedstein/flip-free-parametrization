
#include "InteractiveDeformer.h"

#include <igl/readOBJ.h>
#include <igl/readMESH.h>
#include <igl/remove_unreferenced.h>
#include <igl/doublearea.h>
#include <igl/volume.h>
#include <igl/png/readPNG.h>

#include <iostream>


//This is a GUI utility to deform surfaces and volumes.
//Use as follows: ./interactive_deformation input [texture] for normal use, or
// ./interactive_deformation -v input [texture] for verbose use.
//Provide 2D surfaces in the OBJ format (file name must end in ".obj",
// z-coordinate is ignored), and 3D volumes in the MESH format
// (file name must end in ".mesh").
//For 2D surfaces, you can optionally provide a png file as the texture
// argument, it will be used to texture the mesh during deformation.
//
//Controls for the interactive deformer:
// Left-click and drag to rotate the object (only in 3D).
// Right-click and drag to translate the object.
// Shift + Left-click to move pinned vertices.
// Shift + Right-click to pin and unpin vertices.
// Press l to show / hide wireframe.
// Press w to save the current mesh to out.obj / out.mesh.
// Press esc to exit the application.

int main(int argc, char *argv[]) {
    std::string howToUse = R"(This is a GUI utility to deform surfaces and volumes.
Use as follows: ./interactive_deformation input [texture] for normal use, or ./interactive_deformation -v input [texture] for verbose use.
Provide 2D surfaces in the OBJ format (file name must end in ".obj", z-coordinate is ignored), and 3D volumes in the MESH format (file name must end in ".mesh").
For 2D surfaces, you can optionally provide a png file as the texture argument, it will be used to texture the mesh during deformation.
    
For example: try ./interactive_deformation meshes/robot.obj meshes/robot.png)";
    if(argc<2) {
        std::cerr << "No input mesh supplied." << std::endl;
        std::cout << howToUse << std::endl;
        return -2;
    }
    
    std::string mesh;
    bool verbose = false;
    if(std::string(argv[1]) == std::string("-v")) {
        verbose = true;
        if(argc<3) {
            std::cerr << "No input mesh supplied." << std::endl;
            std::cout << howToUse << std::endl;
            return -2;
        }
        mesh = std::string(argv[2]);
    } else {
        mesh = std::string(argv[1]);
    }
    
    //Check for texture
    const int textureIndex = verbose ? 3 : 2;
    bool textureAvailable = false;
    Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> R, G, B, A;
    if(argc>textureIndex) {
        textureAvailable = igl::png::readPNG(std::string(argv[textureIndex]),
                                             R, G, B, A);
    }
    
    //Load mesh
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    if(mesh.size()>3 && (mesh.substr(mesh.size()-4,4)==".obj" ||
                         mesh.substr(mesh.size()-4,4)==".OBJ")) {
        if(!igl::readOBJ(mesh, V, F)) {
            std::cerr << "Error reading mesh " << mesh << "." << std::endl;
            return -3;
        }
        V = V.block(0,0,V.rows(),2);
    } else if(mesh.size()>4 && (mesh.substr(mesh.size()-5,5)==".mesh" ||
                                mesh.substr(mesh.size()-5,5)==".MESH")) {
        Eigen::MatrixXi dummy;
        if(!igl::readMESH(mesh, V, F, dummy)) {
            std::cerr << "Error reading mesh " << mesh << "." << std::endl;
            return -3;
        }
    } else {
        std::cerr << "Input does not seem to be obj or mesh." << std::endl;
        return -4;
    }
    
    if(F.size() == 0) {
        std::cerr << "No mesh provided."
        << std::endl;
        return -5;
    }
    
    if(F.cols()!=3 && F.cols()!=4) {
        std::cerr << "This application only works for triangle and tet  meshes."
        << std::endl;
        return -6;
    }
    
    Eigen::VectorXd AV;
    if(F.cols()==3) {
        igl::doublearea(V, F, AV);
        AV *= 0.5;
    } else {
        igl::volume(V, F, AV);
    }
    for(int i=0; i<AV.size(); ++i) {
        if(AV(i) < 1e-8) {
            std::cerr <<
            "This mesh contains a degenerate triangle or tet (area/vol <1e-8)."
            << std::endl;
            //return -7;
        }
    }
    
    //Remove unreferenced vertices
    Eigen::VectorXi dummy;
    igl::remove_unreferenced(Eigen::MatrixXd(V), Eigen::MatrixXi(F),
                             V, F, dummy);
    
    //Initialize InteractiveDeformer and launch it.
    InteractiveDeformer<double> interactiveDeformer(V, F, verbose);
    if(textureAvailable) {
        interactiveDeformer.set_texture(R, G, B, A);
    }
    std::cout << R"(Controls for the interactive deformer:
    Left-click and drag to rotate the object (only in 3D).
    Right-click and drag to translate the object.
    Shift + Left-click to move pinned vertices.
    Shift + Right-click to pin and unpin vertices.
    Press l to show / hide wireframe.
    Press w to save the current mesh to out.obj / out.mesh.
    Press esc to exit the application.)" << std::endl;
    interactiveDeformer.launch();

    return 0;
}
