
#ifdef SUITESPARSE_AVAILABLE

#include "bff.h"

#include "parametrization_assert.h"

#include <MeshIO.h>
#include <Bff.h>
#include <DenseMatrix.h>


template <typename DerivedV, typename DerivedF, typename DerivedW>
bool
parametrization::bff(const Eigen::MatrixBase<DerivedV>& V,
                     const Eigen::MatrixBase<DerivedF>& F,
                     Eigen::PlainObjectBase<DerivedW>& W)
{
    using Int = typename DerivedF::Scalar;
    using Scalar = typename DerivedV::Scalar;
    
    parametrization_assert(V.array().isFinite().all() &&
                           "Invalid entries in V");
    parametrization_assert(F.array().isFinite().all() &&
                           "Invalid entries in F");
    parametrization_assert(V.cols()==3 && "V,F must be a surface mesh in 3d.");
    parametrization_assert(F.cols()==3 && "V,F must be a surface mesh in 3d.");
    parametrization_assert(F.maxCoeff()<V.rows() &&
                           "F does not correspond to V.");
    parametrization_assert(F.minCoeff()>=0 && "F contains an illegal vertex.");
    
    const Int n=V.rows(), m=F.rows();
    
    //Copy vertices and faces
    bff::PolygonSoup soup;
    soup.positions.reserve(n);
    for(Int i=0; i<n; ++i) {
        soup.positions.emplace_back(bff::Vector(V(i,0), V(i,1), V(i,2)));
    }
    soup.indices.reserve(3*m);
    for(Int i=0; i<m; ++i) {
        for(int j=0; j<3; ++j) {
            soup.indices.emplace_back(F(i,j));
        }
    }
    soup.table.construct(n, soup.indices);
    std::vector<int> isCuttableEdge(soup.table.getSize(), 0);
    
    //Construct mesh
    bff::Mesh mesh;
    std::string error;
    bff::MeshIO::buildMesh(soup, isCuttableEdge, mesh, error);
    if(!error.empty()) {
        return false;
    }
    
    //Compute BFF
    bff::BFF bffComputer(mesh);
    bff::DenseMatrix boundaryData(bffComputer.data->bN);
    if(!bffComputer.flatten(boundaryData, true)) {
        return false;
    }
//    bffComputer.flattenToDisk();
    
    //Extract UV
    W.resize(n,2);
    for(const auto& corner : mesh.corners) {
        const auto& vert = corner.vertex();
        const int v = corner.vertex()->index;
        W(v,0) = corner.uv.x;
        W(v,1) = corner.uv.y;
    }
    
    return W.rows()==V.rows() && W.cols()==2 && W.array().isFinite().all();
}


//Explicit template instantiation
template bool parametrization::bff<Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);

#endif
