#ifndef APPLICATION_INTERACTIVEDEFORMER_H
#define APPLICATION_INTERACTIVEDEFORMER_H


#include <Eigen/Core>

#include <igl/writeOBJ.h>
#include <igl/writeMESH.h>
#include <igl/boundary_facets.h>
#include <igl/project.h>
#include <igl/get_seconds.h>
#include <igl/unproject_ray.h>
#include <igl/ray_mesh_intersect.h>
#include <igl/bounding_box_diagonal.h>
#include <igl/png/readPNG.h>
#define IGL_VIEWER_VIEWER_QUIET
#include <igl/opengl/glfw/Viewer.h>

#include <parametrization/map_to.h>

#include <iostream>
#include <atomic>
#include <mutex>
#include <thread>


template<typename _Scalar>
class InteractiveDeformer {
public:
    using Scalar = _Scalar;
    using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
    using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
    using Vec4 = Eigen::Matrix<Scalar, 4, 1>;
    using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using MatX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VecI = Eigen::Matrix<int, Eigen::Dynamic, 1>;
    using MatI = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;
    using MatUC = Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>;
    
    //Constructor supplying the mesh to be deformed, as well as the verbosity.
    template<typename DerivedV, typename DerivedF>
    InteractiveDeformer(const Eigen::MatrixBase<DerivedV>& _V,
                        const Eigen::MatrixBase<DerivedF>& _F,
                        const bool _verbose);
    
    //Launch the deformer
    void launch();
    
    //On 2d images, set a texture instead of default shading.
    void set_texture(const MatUC& _R, const MatUC& _G, const MatUC& _B,
                     const MatUC& _A);
    
    //Color of the bg
    Vec4 bgColor{217./255., 217./255., 217./255., 1.};
    
    //Color of fixed vertices and dragged vertices
    Vec3 fixedColor{255./255.,255./255.,204./255.};
    Vec3 draggedColor{254./255.,178./255.,76./255.};
    
    //Radius of displayed points (in screen space)
    Scalar r = 20;
    
    //If a vertex is dragged by more than this ratio of the bounding box
    // diagonal of the original mesh, will try to stagger the dragging.
    Scalar staggeredDraggingRatio = 0.1;
    
private:
    const int dim; //the dimension of the surface
    const MatX oV; //original vertex positions
    const MatI F;
    const MatI bF; //if surface, bF==f. If volume, bF==boundary faces.
    const Scalar diag; //Diagonal size of oV;
    const bool verbose;
    
    //Texture data
    MatUC R, G, B, A;
    
    //Colors of the surface (when not in texture mode).
    const Vec3 surfaceColor{185./255., 123./255., 98./255.};
    
    //Matcap texture
    MatUC mcR, mcG, mcB, mcA;
    
    MatX cV; //current vertex positions
    MatX mapToV; //mesh for map_to to operate on
    MatX uvV; //if texture mapping, mesh with UV coords
    
    using Viewer = igl::opengl::glfw::Viewer;
    Viewer viewer; //the libigl viewer where everything happens
    
    //Do meshes / fixed vertices need to be redrawn?
    bool redraw = true;
    //Draw meshes and fixed vertices
    void draw();
    
    //Libigl viewer callbacks
    bool pre_draw();
    bool mouse_down(int button, int modifier);
    bool mouse_up(int button, int modifier);
    bool mouse_move(int mouse_x, int mouse_y);
    bool key_pressed(unsigned int key, int modifiers);
    
    //Stores info about pressed buttons.
    bool buttonPressed = false;
    bool buttonPressedWithShift = false;
    int lastButtonPressed = -1;
    int mouseX=-1, mouseY=-1;
    
    //Write current mesh to an output file
    void write();
    
    //Stores info about fixed and dragged vertices
    VecI fixed;
    MatX fixedTo;
    int draggedVertex = -1; //indexes fixed if positive
    bool fixedChanged=false, fixedToChanged=false;
    //Fix / unfix the vertex at the current position.
    void fix_unfix();
    //Select (to drag) the vertex at the current position.
    void select();
    //Drag the vertex to the mouse position
    void dragVertex();
    //Utility function to find index of fixed vertex beneath mouse (or -1)
    int fixedAtMouse();
    //Mutex to lock access to
    
    //Callback function for map_to
    bool map_to_callback(parametrization::CallbackFunctionState
                         <InteractiveDeformer<Scalar>::MatX, int>& state,
                         bool& changedPrec,
                         bool& changed);
    
    //Variables to exchange information from map_to thread to main thread,
    // and a mutex to lock these variables
    MatX exchangeV, exchangeFixedTo;
    VecI exchangeFixed;
    bool exchangeFixedChanged=false, exchangeFixedToChanged=false;
    std::mutex exchangeMutex;
    
    //How often to try to exchange, and when was the last time we exchanged.
    const double exchangeEvery = 0.01;
    double lastExchangedOnMapto=igl::get_seconds(),
    lastExchangedOnMain=igl::get_seconds();
    
    //Check if it is time and perform all exchanging necessary on the main
    // thread (read from exchangeV, write into exchangeFixed, exchangeFixedTo).
    void exchangeOnMain();
    
    //Check if it is time and perform all exchanging necessary on the map_to
    // thread (write into exchangeV, read from exchangeFixed, exchangeFixedTo).
    void exchangeOnMapTo(const MatX& V, VecI& fixed, MatX& fixedTo,
                         bool& changedPrec, bool& changed);
    
    //Which parametrization energy to choose
    static constexpr parametrization::EnergyType energy =
    parametrization::EnergyType::SymmetricDirichlet;
    
    //Is the application about to end?
    std::atomic<bool> ending{false};
};



template<typename Scalar>
template<typename DerivedV, typename DerivedF>
InteractiveDeformer<Scalar>::InteractiveDeformer
(const Eigen::MatrixBase<DerivedV>& _V,
 const Eigen::MatrixBase<DerivedF>& _F,
 const bool _verbose) :
oV(_V.template cast<Scalar>()), cV(_V.template cast<Scalar>()),
exchangeV(_V.template cast<Scalar>()),
F(_F.template cast<int>()),
bF(_V.cols()==3 ? [&]() {
    MatI tF;
    igl::boundary_facets(F, tF);
    VecI temp = tF.col(2);
    tF.col(2) = tF.col(1);
    tF.col(1) = temp;
    return tF;
}() :
   F.template cast<int>()),
dim(_V.cols()), verbose(_verbose),
diag(igl::bounding_box_diagonal(_V))
{
    //Check input data.
    if(F.cols()-1 != dim) {
        std::cerr << "For 2D, the input needs to be a triangle mesh. For 3D, the input needs to be a tet mesh." << std::endl;
        abort();
    }
    if(dim!=2 && dim!=3) {
        std::cerr << "Only 2D and 3D are supported." << std::endl;
        abort();
    }
    
    //Read in matcap texture data, if possible.
    igl::png::readPNG(MATCAPS_DIRECTORY "clay.png", mcR, mcG, mcB, mcA);
}


template<typename Scalar>
void
InteractiveDeformer<Scalar>::write()
{
    if(dim==2) {
        if(igl::writeOBJ("out.obj", oV, F)) {
            if(verbose) {
                std::cout << "Wrote output to out.obj." << std::endl;
            }
        } else {
            std::cerr << "Could not write to out.obj." << std::endl;
        }
    } else {
        //dim==3
        if(igl::writeOBJ("out.obj", oV, bF) &&
           igl::writeMESH("out.mesh", oV, F, bF)) {
            if(verbose) {
                std::cout << "Wrote output to out.obj and out.mesh." <<
                std::endl;
            }
        } else {
            std::cerr << "Could not write to oub.obj / out.mesh." << std::endl;
        }
    }
}


template<typename Scalar>
void
InteractiveDeformer<Scalar>::launch()
{
    //Start map_to on a background thread
    std::thread threadMapTo([this]() {
        //Callback function and options
        parametrization::OptimizationOptions<Scalar> opts;
        opts.terminationCondition = parametrization::TerminationCondition::None;
        parametrization::CallbackFunction<MatX,int> f =
        [this] (auto& state, auto& changedPrec, auto& changed) {
            return map_to_callback(state, changedPrec, changed);
        };
        
#ifdef OPENMP_AVAILABLE
        constexpr bool parallelize = true;
#else
        constexpr bool parallelize = false;
#endif
        if(verbose) {
            std::cout << "Starting background thread running map_to." <<
            std::endl;
        }
        
        bool success = false;
        while(!success) {
            //Set up working variable
            mapToV = exchangeV;
            
            //We need to make sure at least one point is fixed.
            VecI fixed = exchangeFixed;
            MatX fixedTo = exchangeFixedTo;
            if(fixed.size()==0) {
                fixed = VecI::Zero(1);
                fixedTo = mapToV.row(0);
            }
            Eigen::SparseMatrix<Scalar> Aeq;
            MatX beq;
            
            success = parametrization::map_to<parallelize, energy>
            (oV, F, fixed, fixedTo, Aeq, beq, mapToV, opts, f);
            
            if(verbose && !success) {
                std::cout << "Optimization will restart." << std::endl;
            }
        }
        
        if(verbose) {
            std::cout << "Background thread running map_to quit." << std::endl;
        }
    });
    
    //Start the viewer
    viewer.callback_pre_draw = [this] (Viewer& viewer) {
        return pre_draw();
    };
    viewer.callback_mouse_down = [this]
    (Viewer& viewer, int button, int modifier) {
        return mouse_down(button, modifier);
    };
    viewer.callback_mouse_up = [this]
    (Viewer& viewer, int button, int modifier) {
        return mouse_up(button, modifier);
    };
    viewer.callback_mouse_move = [this]
    (Viewer& viewer, int mouse_x, int mouse_y) {
        return mouse_move(mouse_x, mouse_y);
    };
    viewer.callback_key_pressed = [this]
    (Viewer& viewer, unsigned int key, int modifiers) {
        return key_pressed(key, modifiers);
    };
    viewer.core().is_animating = true;
    viewer.data().show_lines = false;
    viewer.data().show_overlay_depth = false;
    viewer.launch();
    
    //The libigl viewer has quit, this application is about to end.
    ending = true;
    
    //Join the background map_to thread.
    threadMapTo.join();
}


template<typename Scalar>
void
InteractiveDeformer<Scalar>::set_texture(const MatUC& _R,
                                         const MatUC& _G,
                                         const MatUC& _B,
                                         const MatUC& _A)
{
    //This only works for 2D meshes.
    if(dim==2) {
        //Make sure default lighting does not interfere with texture
        viewer.core().lighting_factor = 0.;
        
        //UV map is trivial for a flat 2D mesh, but we must move it into
        // the rectangle where texture lookup happens.
        const Vec2 max=oV.colwise().maxCoeff(), min=oV.colwise().minCoeff();
        const Vec2 wh = (max-min).array().abs().matrix();
        uvV = oV;
        for(int i=0; i<uvV.rows(); ++i) {
            uvV.row(i) -= min;
            uvV.row(i).array() /= wh.array();
        }
        
        //Copy texture data into the right place.
        R = _R;
        G = _G;
        B = _B;
        A = _A;
        
        A.setOnes();
        A *= 255;
    }
}


template<typename Scalar>
void
InteractiveDeformer<Scalar>::draw()
{
    viewer.core().background_color = bgColor.template cast<float>();
    
    const bool firstTimeSet = viewer.data().V.rows()==0;
    viewer.data().set_mesh(cV, bF);
    if(uvV.size() != 0) {
        //Texture mapping has turned off lighting, set material to white.
        viewer.data().set_colors(Eigen::RowVector3d(1., 1., 1.));
        //Load texture to display if this is the first time the mesh is set.
        if(firstTimeSet) {
            viewer.data().set_uv(uvV);
            viewer.data().set_texture(R, G, B, A);
            viewer.data().show_texture = true;
        }
    } else {
        //If matcap was successfully read, use it. Otherwise, use normal surface
        // colors.
        if(mcR.size()>0) {
            if(firstTimeSet) {
                viewer.data().set_texture(mcR, mcG, mcB, mcA);
                viewer.data().use_matcap = true;
            }
        } else {
            viewer.data().set_colors(surfaceColor.transpose());
        }
    }
    
    MatX pointsColors(fixed.size(), 3);
    for(int i=0; i<fixed.size(); ++i) {
        if(i==draggedVertex) {
            pointsColors.row(i) = draggedColor;
        } else {
            pointsColors.row(i) = fixedColor;
        }
    }
    viewer.data().set_points(fixedTo, pointsColors);
    viewer.data().point_size = r;
}


template<typename Scalar>
bool
InteractiveDeformer<Scalar>::map_to_callback
(parametrization::CallbackFunctionState
 <InteractiveDeformer<Scalar>::MatX, int>& state,
 bool& changedPrec,
 bool& changed)
{
    //Exchange with main thread
    exchangeOnMapTo(state.W, state.fixed, state.fixedTo, changedPrec, changed);
    
    //We need to make sure at least one point is fixed.
    if(state.fixed.size()==0) {
        state.fixed = VecI::Zero(1);
        state.fixedTo = state.W.row(0);
    }
    
    if(changedPrec) {
        state.Lambda.setZero();
    }
    
    //Has the InteractiveDeformer ended?
    return ending;
}


template<typename Scalar>
void
InteractiveDeformer<Scalar>::exchangeOnMain()
{
    const double t = igl::get_seconds();
    if(t-lastExchangedOnMain > exchangeEvery) {
        std::lock_guard<std::mutex> guard(exchangeMutex);
        
        if((exchangeV-cV).squaredNorm() > 1e-10*exchangeV.rows()) {
            cV = exchangeV;
            redraw = true;
        }
        if(fixedChanged) {
            exchangeFixed = fixed;
            exchangeFixedChanged = true;
            fixedChanged = false;
        }
        if(fixedToChanged) {
            exchangeFixedTo = fixedTo;
            exchangeFixedToChanged = true;
            fixedToChanged = false;
        }
        
        lastExchangedOnMain = t;
    }
}

template<typename Scalar>
void
InteractiveDeformer<Scalar>::exchangeOnMapTo(const MatX& V,
                                             VecI& fixed,
                                             MatX& fixedTo,
                                             bool& changedPrec,
                                             bool& changed)
{
    const double t = igl::get_seconds();
    if(t-lastExchangedOnMapto > exchangeEvery) {
        std::lock_guard<std::mutex> guard(exchangeMutex);
        
        exchangeV = V;
        
        changedPrec = exchangeFixedChanged;
        if(exchangeFixedChanged) {
            fixed = exchangeFixed;
        }
        exchangeFixedChanged = false;
        
        changed = exchangeFixedToChanged;
        if(exchangeFixedToChanged) {
            fixedTo = exchangeFixedTo;
        }
        exchangeFixedToChanged = false;
        
        lastExchangedOnMapto = t;
    } else {
        changedPrec = false;
        changed = false;
    }
    
}


template<typename Scalar>
void
InteractiveDeformer<Scalar>::fix_unfix()
{
    //Transform mouse coords into screen space.
    Eigen::Vector2f screenMouse(mouseX, viewer.core().viewport(3)-mouseY);
    
    //If we are above a fixed vertex, unfix that vertex.
    int vertexToUnfix = fixedAtMouse();
    if(vertexToUnfix >= 0) {
        for(int i=vertexToUnfix; i<fixed.size()-1; ++i) {
            fixed(i) = fixed(i+1);
            fixedTo.row(i) = fixedTo.row(i+1);
        }
        fixed.conservativeResize(fixed.size()-1);
        fixedTo.conservativeResize(fixedTo.rows()-1, fixedTo.cols());
        
        if(verbose) {
            std::cout << "Unfixed vertex " << vertexToUnfix << "." << std::endl;
        }
        redraw = true;
        fixedChanged = true;
        fixedToChanged = true;
        return;
    }
    
    //Fix a new vertex
    //Find view ray
    Eigen::Vector3f s, dir;
    igl::unproject_ray(screenMouse, viewer.core().view, viewer.core().proj,
                       viewer.core().viewport, s, dir);
    dir.normalize();
    
    //Intersect the view ray with the mesh to find whether there is a new
    // vertex to fix.
    std::vector<igl::Hit> hits;
    if(dim == 2) {
        MatX cV3(cV.rows(), 3);
        cV3.block(0, 0, cV.rows(), 2) = cV;
        cV3.col(2).setZero();
        igl::ray_mesh_intersect(s, dir, cV3, bF, hits);
    } else {
        igl::ray_mesh_intersect(s, dir, cV, bF, hits);
    }
    
    //Helper function to get hit coord
    const auto hit_to_coord = [&] (const igl::Hit& hit) {
        return cV.row(bF(hit.id,0))*(1-hit.u-hit.v) +
        cV.row(bF(hit.id,1))*hit.u +
        cV.row(bF(hit.id,2))*hit.v;
    };
    
    if(hits.size()>0) {
        int closestId;
        
        const VecX hit0 = hit_to_coord(hits[0]);
        if(dim==2 || hits.size()<2) {
            //2-dim, or hit at a very weird angle, or inside mesh.
            //Fix the closest vertex to the hit.
            closestId = bF(hits[0].id, 0);
            Scalar closestDist =
            (hit0.transpose() - cV.row(closestId)).squaredNorm();
            for(int i=1; i<3; ++i) {
                const int id = bF(hits[0].id, i);
                const Scalar d = (hit0.transpose() - cV.row(id)).squaredNorm();
                if(d < closestDist) {
                    closestId = id;
                    closestDist = d;
                }
            }
        } else {
            //Find the first two hits and fix the vertex closest to the middle.
            const Vec3 mid = 0.5*(hit0 + hit_to_coord(hits[1]).transpose());
            
            closestId = 0;
            Scalar closestDist =
            (mid.transpose() - cV.row(closestId)).squaredNorm();
            for(int i=1; i<cV.rows(); ++i) {
                const Scalar d = (mid.transpose() - cV.row(i)).squaredNorm();
                if(d < closestDist) {
                    closestId = i;
                    closestDist = d;
                }
            }
        }
        
        //Change fixed and fixedTo to include the new vertex, unless it is
        // already fixed.
        for(int i=0; i<fixed.size(); ++i) {
            if(fixed(i) == closestId) {
                return;
            }
        }
        fixed.conservativeResize(fixed.size()+1);
        fixed(fixed.size()-1) = closestId;
        fixedTo.conservativeResize(fixedTo.rows()+1, dim);
        fixedTo.row(fixedTo.rows()-1) = cV.row(closestId);
        
        if(verbose) {
            std::cout << "Fixed vertex " << closestId << " at " <<
            fixedTo.row(fixedTo.rows()-1) << "." << std::endl;
        }
        redraw = true;
        fixedChanged = true;
        fixedToChanged = true;
        return;
    }
}


template<typename Scalar>
void
InteractiveDeformer<Scalar>::select()
{
    //Find index of vertex beneath mouse position.
    draggedVertex = fixedAtMouse();
    
    //If a vertex was selected, handle it.
    if(draggedVertex >= 0) {
        redraw = true;
        if(verbose) {
            std::cout << "Selected vertex " << fixed(draggedVertex) << "." <<
            std::endl;
        }
    }
}


template<typename Scalar>
void
InteractiveDeformer<Scalar>::dragVertex()
{
    if(draggedVertex<0) {
        //No vertex is selected.
        return;
    }
    
    //Current coordinates of the selected vertex
    Eigen::Vector3f sel;
    if(dim == 2) {
        sel.template topRows<2>() =
        cV.row(fixed(draggedVertex)).template cast<float>();
        sel(2) = 0;
    } else {
        sel = cV.row(fixed(draggedVertex)).template cast<float>();
    }
    
    //Transform mouse coords into screen space.
    Eigen::Vector2f screenMouse(mouseX, viewer.core().viewport(3)-mouseY);
    
    //Find view ray
    Eigen::Vector3f s, dir;
    igl::unproject_ray(screenMouse, viewer.core().view, viewer.core().proj,
                       viewer.core().viewport, s, dir);
    dir.normalize();
    
    //Project s onto selected's plane.
    s -= (s-sel).dot(dir)*dir;
    
    //Stagger dragging.
    const double sSelDist = (s-sel).norm();
    const double staggerDist = diag * staggeredDraggingRatio;
    if(sSelDist > staggerDist) {
        const double ratio = staggerDist / sSelDist;
        s = (1.-ratio)*sel + ratio*s;
    }
    
    //Fix new position
    if(dim == 2) {
        fixedTo.row(draggedVertex) =
        s.template topRows<2>().template cast<Scalar>();
    } else {
        fixedTo.row(draggedVertex) = s.template cast<Scalar>();
    }
    cV.row(fixed(draggedVertex)) = fixedTo.row(draggedVertex);
    
    //State variables
    redraw = true;
    fixedToChanged = true;
}


template<typename Scalar>
int
InteractiveDeformer<Scalar>::fixedAtMouse()
{
    //Transform mouse coords into screen space.
    Eigen::Vector2f screenMouse(mouseX, viewer.core().viewport(3)-mouseY);
    
    //For every fixed point, project into screen space and see whether this is
    // close enough to mouseX, mouseY.
    for(int i=0; i<fixed.size(); ++i) {
        Eigen::Vector3f fixedCoords;
        if(dim==2) {
            fixedCoords.template topRows<2>() =
            fixedTo.row(i).template cast<float>();
            fixedCoords(2) = 0.;
        } else {
            fixedCoords = fixedTo.row(i).template cast<float>();
        }
        const Eigen::Vector2f screenFixedCoords = igl::project
        (fixedCoords, viewer.core().view, viewer.core().proj,
         viewer.core().viewport).template topRows<2>();
        if((screenFixedCoords - screenMouse).squaredNorm() < 0.25*r*r) {
            //Only the first hit matters
            return i;
        }
    }
    
    return -1;
}


template<typename Scalar>
bool
InteractiveDeformer<Scalar>::pre_draw()
{
    //When dragging, call drag function
    if(buttonPressed && buttonPressedWithShift && lastButtonPressed==0) {
        dragVertex();
    }
    
    //Exchange data with the other thread.
    exchangeOnMain();
    
    //Do we need to redraw the scene?
    if(redraw) {
        draw();
        redraw = false;
    }
    
    return false;
}


template<typename Scalar>
bool
InteractiveDeformer<Scalar>::mouse_down(int button, int modifier)
{
    lastButtonPressed = button;
    buttonPressed = true;
    buttonPressedWithShift = modifier==1;
    
    if(button==0 && modifier==1) {
        if(mouseX>=0 && mouseY>=0) {
            select();
        }
    }
    if(button==2 && modifier==1) {
        if(mouseX>=0 && mouseY>=0) {
            fix_unfix();
        }
    }
    
    return false;
}


template<typename Scalar>
bool
InteractiveDeformer<Scalar>::mouse_up(int button, int modifier)
{
    buttonPressed = false;
    if(draggedVertex >= 0) {
        if(verbose) {
            std::cout << "Dragged vertex " << fixed(draggedVertex) << " to " <<
            fixedTo.row(draggedVertex) << "." << std::endl;
        }
        draggedVertex = -1;
        redraw = true;
    }
    
    return false;
}


template<typename Scalar>
bool
InteractiveDeformer<Scalar>::mouse_move(int mouse_x, int mouse_y)
{
    mouseX = mouse_x;
    mouseY = mouse_y;
    
    //When dragging, do not rotate or translate view
    if(buttonPressed && buttonPressedWithShift) {
        return true;
    }
    
    //In 2D, can not rotate, only translate.
    if(dim==2 && buttonPressed && lastButtonPressed==0) {
        return true;
    }
    
    return false;
}


template<typename Scalar>
bool
InteractiveDeformer<Scalar>::key_pressed(unsigned int key, int modifiers)
{
    switch(key) {
        case 'w':
        case 'W':
            write();
            break;
        case 'l':
        case 'L':
            return false;
        case 'o':
        case 'O':
            return false;
    }
    return true;
}


#endif
