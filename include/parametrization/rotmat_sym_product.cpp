
#include "rotmat_sym_product.h"

#include "parametrization_assert.h"
#include "rotmat_from_complex_quat.h"
#include "sym_from_triu.h"
#include "mat_from_index.h"


template <typename DerivedU, typename DerivedP, typename DerivedJ>
void
parametrization::rotmat_sym_product_2d(const Eigen::MatrixBase<DerivedU>& U,
                                       const Eigen::MatrixBase<DerivedP>& P,
                                       Eigen::PlainObjectBase<DerivedJ>& J)
{
    const Eigen::Index n = U.size()/2;
    parametrization_assert(U.size() == 2*n && "Dimensions of U are wrong.");
    parametrization_assert(P.size() == 3*n && "P does not correspond to U.");
    
    if(J.rows()!=2*n || J.cols()!=2) {
        J.resize(2*n, 2);
    }
    
    for(Eigen::Index i=0; i<n; ++i) {
        const auto Ul = rotmat_2d_from_index(U, i, n);
        const auto Pl = sym_2d_from_index(P, i, n);
        index_into_mat_2x2(Ul*Pl, J, i, n);
    }
}


template <typename DerivedU, typename DerivedP>
Eigen::Matrix<typename DerivedU::Scalar, Eigen::Dynamic, 2>
parametrization::rotmat_sym_product_2d(const Eigen::MatrixBase<DerivedU>& U,
                                       const Eigen::MatrixBase<DerivedP>& P)
{
    Eigen::Matrix<typename DerivedU::Scalar, Eigen::Dynamic, 2> J;
    rotmat_sym_product_2d(U, P, J);
    return J;
}


template <typename DerivedU, typename DerivedP, typename DerivedJ>
void
parametrization::rotmat_sym_product_3d(const Eigen::MatrixBase<DerivedU>& U,
                                       const Eigen::MatrixBase<DerivedP>& P,
                                       Eigen::PlainObjectBase<DerivedJ>& J)
{
    const Eigen::Index n = U.size()/4;
    parametrization_assert(U.size() == 4*n && "The size of U is invalid.");
    parametrization_assert(P.size() == 6*n && "P does not correspond to U.");
    
    if(J.rows()!=3*n || J.cols()!=3) {
        J.resize(3*n, 3);
    }
    
    for(Eigen::Index i=0; i<n; ++i) {
        const auto Ul = rotmat_3d_from_index(U, i, n);
        const auto Pl = sym_3d_from_index(P, i, n);
        index_into_mat_3x3(Ul*Pl, J, i, n);
    }
}


template <typename DerivedU, typename DerivedP>
Eigen::Matrix<typename DerivedU::Scalar, Eigen::Dynamic, 3>
parametrization::rotmat_sym_product_3d(const Eigen::MatrixBase<DerivedU>& U,
                                       const Eigen::MatrixBase<DerivedP>& P)
{
    Eigen::Matrix<typename DerivedU::Scalar, Eigen::Dynamic, 3> J;
    rotmat_sym_product_3d(U, P, J);
    return J;
}


// Explicit template instantiation
template Eigen::Matrix<Eigen::Matrix<double, -1, 1, 0, -1, 1>::Scalar, -1, 2, 0, -1, 2> parametrization::rotmat_sym_product_2d<Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&);
template Eigen::Matrix<Eigen::Matrix<double, -1, 1, 0, -1, 1>::Scalar, -1, 3, 0, -1, 3> parametrization::rotmat_sym_product_3d<Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&);
