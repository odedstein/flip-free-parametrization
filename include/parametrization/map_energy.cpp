
#include "map_energy.h"

#include "uv_to_jacobian.h"
#include "energy.h"
#include "parametrization_assert.h"
#include "polar_decomposition.h"

#include <igl/grad.h>
#include <igl/flipped_triangles.h>

#include <Eigen/Sparse>


template <parametrization::EnergyType energy,
typename DerivedV, typename DerivedF, typename DerivedW>
typename DerivedW::Scalar
parametrization::map_energy(const Eigen::MatrixBase<DerivedV>& V,
           const Eigen::MatrixBase<DerivedF>& F,
           const Eigen::MatrixBase<DerivedW>& W)
{
    using Int = typename DerivedF::Scalar;
    using Scalar = typename DerivedW::Scalar;
    using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
    using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Mat33 = Eigen::Matrix<Scalar, 3, 3>;
    using MatX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using SparseMat = Eigen::SparseMatrix<Scalar>;
    
    parametrization_assert(V.array().isFinite().all() &&
                           "Invalid entries in V");
    parametrization_assert(F.array().isFinite().all() &&
                           "Invalid entries in F");
    parametrization_assert(W.array().isFinite().all() &&
                           "Invalid entries in W");
    
    const Int n=V.rows(), m=F.rows();
    parametrization_assert((V.cols()==2 || V.cols()==3) &&
                           "V,F must be a mesh in 2d or 3d.");
    const Int dim = F.cols()-1;
    parametrization_assert((dim==2 || dim==3) &&
                           "V,F must be a triangle or tet mesh.");
    parametrization_assert(F.maxCoeff()<V.rows() &&
                           "F does not correspond to V.");
    parametrization_assert(F.minCoeff()>=0 && "F contains an illegal vertex.");
    parametrization_assert([&](){
        VecX A;
        if(dim==2) {
            igl::doublearea(V, F, A);
            A *= 0.5;
        } else {
            igl::volume(V, F, A);
        }
        return A;
    }().minCoeff() > 1e-12 && "Degenerate face/tet in V input.");
    parametrization_assert(W.rows()==V.rows() && "W does not correspond to V.");
    parametrization_assert(W.cols()==dim && "W has the wrong dimensions.");
    
    //Compute the Jacobian of the map and then the energy.
    VecX w;
    SparseMat G;
    if(dim==2) {
        //Areas of triangles / intrinsic gradient
        igl::doublearea(V, F, w);
        w *= 0.5;
        uv_to_jacobian(V, F, G);
    } else {
        //Volumes of tets / normal gradient
        igl::volume(V, F, w);
        igl::grad(V, F, G);
    }
    MatX J = G*W;
    
    return dim==2 ? parametrization::energy_from_GW<energy,2>(J,w) :
    parametrization::energy_from_GW<energy,3>(J,w);
}


// Explicit template instantiation
template Eigen::Matrix<double, -1, -1, 0, -1, -1>::Scalar parametrization::map_energy<(parametrization::EnergyType)0, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&);
template Eigen::Matrix<double, -1, -1, 0, -1, -1>::Scalar parametrization::map_energy<(parametrization::EnergyType)1, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&);
