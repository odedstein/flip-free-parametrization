
#include "tutte.h"

#include <igl/boundary_loop.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/flipped_triangles.h>
#include <igl/edges.h>
#include <igl/euler_characteristic.h>

#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#ifdef SUITESPARSE_AVAILABLE
#include <Eigen/CholmodSupport>
#include <Eigen/UmfPackSupport>
#endif
#include <Eigen/IterativeLinearSolvers>

#include "parametrization_assert.h"
#include "constrained_qp.h"


template <bool uniformScaling,
typename DerivedV, typename DerivedF, typename DerivedW>
bool
parametrization::tutte(const Eigen::MatrixBase<DerivedV>& V,
      const Eigen::MatrixBase<DerivedF>& F,
      Eigen::PlainObjectBase<DerivedW>& W)
{
    using Int = typename DerivedF::Scalar;
    using VecI = Eigen::Matrix<Int, Eigen::Dynamic, 1>;
    using Scalar = typename DerivedV::Scalar;
    using MatX2 = Eigen::Matrix<Scalar, Eigen::Dynamic, 2>;
    using SparseMat = Eigen::SparseMatrix<Scalar>;
    
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
    parametrization_assert(igl::euler_characteristic(F)==1 &&
                           "Mesh must be disk topology.");
    
    const Int n = V.rows();
    
    //Extract the boundary loop and map it to a circle.
    VecI fixed;
    MatX2 fixedTo;
    igl::boundary_loop(F, fixed);
    parametrization_assert(fixed.size()>0 && "mesh has no boundary");
    {
        ///TODO
        ///libigl's map_vertices_to circle only works with MatrixXd, this should be
        /// replaced by our own function in the future.
        Eigen::MatrixXd hackFixedTo;
        igl::map_vertices_to_circle(V.template cast<double>(),
                                    fixed.template cast<int>(), hackFixedTo);
        fixedTo = hackFixedTo.template cast<Scalar>();
    }
    
    //Handle the special case where all vertices are boundary vertices.
    if(fixed.size()==n) {
        if(W.rows()!=n || W.cols()!=2) {
            W.resize(n, 2);
        }
        for(Int i=0; i<n; ++i) {
            W.row(fixed(i)) = fixedTo.row(i);
        }
        return true;
    }
    
    return tutte<uniformScaling>(V, F, fixed, fixedTo, SparseMat(), MatX2(), W);
}


template <bool uniformScaling,
typename DerivedV, typename DerivedF, typename DerivedW,
typename Derivedfixed, typename DerivedfixedTo, typename ScalarAeq,
typename Derivedbeq>
bool
parametrization::tutte(const Eigen::MatrixBase<DerivedV>& V,
      const Eigen::MatrixBase<DerivedF>& F,
      const Eigen::MatrixBase<Derivedfixed>& fixed,
      const Eigen::MatrixBase<DerivedfixedTo>& fixedTo,
      const Eigen::SparseMatrix<ScalarAeq>& Aeq,
      const Eigen::MatrixBase<Derivedbeq>& beq,
      Eigen::PlainObjectBase<DerivedW>& W)
{
    using Int = typename DerivedF::Scalar;
    using MatI2 = Eigen::Matrix<Int, Eigen::Dynamic, 2>;
    using Scalar = typename DerivedV::Scalar;
    using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using SparseMat = Eigen::SparseMatrix<Scalar>;
    
    const int dim = F.cols()-1;
    
    parametrization_assert(V.array().isFinite().all() &&
                           "Invalid entries in V");
    parametrization_assert(F.array().isFinite().all() &&
                           "Invalid entries in F");
    parametrization_assert((V.cols()==2 || V.cols()==3) &&
                           "V must have points in 2d or 3d.");
    parametrization_assert((dim==2 || dim==3) &&
                           "V,F must be a triangle or tet mesh.");
    parametrization_assert(F.maxCoeff()<V.rows() &&
                           "F does not correspond to V.");
    parametrization_assert(F.minCoeff()>=0 && "F contains an illegal vertex.");
    
    parametrization_assert((Aeq.size()==0)==(beq.size()==0) &&
                           "Aeq is specified iff beq is specified.");
    parametrization_assert((Aeq.size()==0 || Aeq.cols()==V.rows()) &&
                           "Aeq does not correspond to V.");
    parametrization_assert((beq.size()==0 || beq.cols()==dim) &&
                           "beq has the wrong dimension.");
    parametrization_assert((beq.size()==0 || beq.rows()==Aeq.rows()) &&
                           "beq does not correspond to A.");
    parametrization_assert((fixed.size()==0 || fixed.size()==fixedTo.rows()) &&
                           "fixed does not correspond to fixedTo.");
    parametrization_assert((fixedTo.size()==0 || fixedTo.cols()==dim) &&
                           "fixedTo has the wrong dimension.");
    parametrization_assert((fixed.size()==0)==(fixedTo.size()==0) &&
                           "fixed is specified iff fixedTo is specified.");
    
    
#ifdef SUITESPARSE_AVAILABLE
    using SPDSolver = Eigen::CholmodSupernodalLLT<SparseMat>;
    using Solver = Eigen::UmfPackLU<SparseMat>;
#else
    using SPDSolver = Eigen::SimplicialLLT<SparseMat>;
    using Solver = Eigen::SparseLU<SparseMat>;
#endif
    using RobustSolver = Eigen::ConjugateGradient
    <SparseMat, Eigen::Lower|Eigen::Upper>;
    
    const Int n = V.rows();
    
    //Set up the sparse matrix for the Tutte mass-spring system.
    MatI2 E;
    igl::edges(F, E);
    const Int nE = E.rows();
    std::vector<Eigen::Triplet<Scalar> > tripletList;
    tripletList.reserve(4*nE);
    for(Int e=0; e<nE; ++e) {
        const Int i1=E(e,0), i2=E(e,1);
        parametrization_assert(i1!=i2 && "degenerate face");
        
        const auto& v1=V.row(i1), &v2=V.row(i2);
        parametrization_assert((v1-v2).squaredNorm()>0 && "degenerate face");
        
        const Scalar w = uniformScaling ? 1. : 1./(v1-v2).norm();
        
        tripletList.emplace_back(i1, i1, w);
        tripletList.emplace_back(i1, i2, -w);
        tripletList.emplace_back(i2, i1, -w);
        tripletList.emplace_back(i2, i2, w);
    }
    SparseMat L(n, n);
    L.setFromTriplets(tripletList.begin(), tripletList.end());
    
    //Optimize the mass-spring system with fixed boundary.
    const auto initial_solve = [&]() {
        if (Aeq.size()==0) {
            return constrained_qp<SPDSolver>
            (L, VecX(), fixed, fixedTo, Aeq, beq, W);
        } else {
            return constrained_qp<Solver>
            (L, VecX(), fixed, fixedTo, Aeq, beq, W);
        }
    };
    if(initial_solve()) {
        return W.array().isFinite().all();
    } else {
        return constrained_qp<RobustSolver>
        (L, VecX(), fixed, fixedTo, Aeq, beq, W) &&
        W.array().isFinite().all();
    }
}


// Explicit template instantiation
template bool parametrization::tutte<false, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template bool parametrization::tutte<true, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template bool parametrization::tutte<false, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, double, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::SparseMatrix<double, 0, int> const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template bool parametrization::tutte<true, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, double, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::SparseMatrix<double, 0, int> const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
