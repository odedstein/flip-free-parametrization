
#include "uv_to_jacobian.h"

#include <igl/grad_intrinsic.h>
#include <igl/edge_lengths.h>
#include <igl/doublearea.h>

#include "parametrization_assert.h"

#include <Eigen/Sparse>


template <typename DerivedV, typename DerivedF, typename ScalarG>
void
parametrization::uv_to_jacobian(const Eigen::MatrixBase<DerivedV>& V,
                                const Eigen::MatrixBase<DerivedF>& F,
                                Eigen::SparseMatrix<ScalarG>& G)
{
    using Scalar = typename DerivedV::Scalar;
    using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using MatX3 = Eigen::Matrix<Scalar, Eigen::Dynamic, 3>;
    using SparseMat = Eigen::SparseMatrix<ScalarG>;
    
    parametrization_assert(V.array().isFinite().all() &&
                           "Invalid entries in V");
    parametrization_assert(F.array().isFinite().all() &&
                           "Invalid entries in F");
    parametrization_assert((V.cols()==2 || V.cols()==3) &&
                           "V,F must be a surface mesh in 2d or 3d.");
    parametrization_assert(F.cols()==3 && "V,F must be a triangle mesh.");
    parametrization_assert(F.maxCoeff()<V.rows() &&
                           "F does not correspond to V.");
    parametrization_assert(F.minCoeff()>=0 && "F contains an illegal vertex.");
    parametrization_assert([&](){
        VecX A;
        igl::doublearea(V, F, A);
        return A;
    }().minCoeff() > std::numeric_limits<Scalar>::min() &&
                           "Degenerate face in V input.");
    
    MatX3 l;
    igl::edge_lengths(V, F, l);
    parametrization_assert(l.array().isFinite().all() &&
                           l.minCoeff() > std::numeric_limits<Scalar>::min() &&
                           "degenerate faces");
    parametrization_assert
    ((l.array().col(0)+l.array().col(1)>l.array().col(2)).all() &&
     (l.array().col(1)+l.array().col(2)>l.array().col(0)).all() &&
     (l.array().col(2)+l.array().col(0)>l.array().col(1)).all() &&
     "degenerate faces");
    igl::grad_intrinsic(l, F, G);
    
    for(int i=0; i<G.outerSize(); ++i) {
        for(typename SparseMat::InnerIterator it(G,i); it; ++it) {
            parametrization_assert(std::isfinite(it.value()) &&
                                   "degenerate faces");
        }
    }
}


// Explicit template instantiation
template void parametrization::uv_to_jacobian<Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, double>(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const&, Eigen::SparseMatrix<double, 0, int>&);
