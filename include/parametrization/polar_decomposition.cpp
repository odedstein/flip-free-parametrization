
#include "polar_decomposition.h"

#include "parametrization_assert.h"
#include "mat_from_index.h"
#include "rotmat_from_complex_quat.h"
#include "sym_from_triu.h"

#include <Eigen/SVD>
#include <Eigen/Eigenvalues>


template <typename DerivedJ, typename DerivedU, typename DerivedP,
typename Scalartol>
void
parametrization::polar_decomposition(const Eigen::MatrixBase<DerivedJ>& J,
                                     Eigen::PlainObjectBase<DerivedU>& U,
                                     Eigen::PlainObjectBase<DerivedP>& P,
                                     const Scalartol tol)
{
    using Scalar = typename DerivedJ::Scalar;
    using Mat2 = Eigen::Matrix<Scalar, 2, 2>;
    using Mat3 = Eigen::Matrix<Scalar, 3, 3>;
    
    const int dim = J.cols();
    parametrization_assert((dim==2 || dim==3) &&
                           "Only 2x2 or 3x3 matrices are supported.");
    parametrization_assert(J.array().isFinite().all() &&
                           "Invalid entries in J");
    parametrization_assert(tol>0 && "tol must be positive.");
    
    const Eigen::Index n = J.rows() / dim;
    parametrization_assert(dim*n == J.rows() && "J is not stacked correctly.");
    
    if(dim==2) {
        if(U.size()!=2*n) {
            U.resize(2*n,1);
        }
        if(P.size()!=3*n) {
            P.resize(3*n,1);
        }
        for(Eigen::Index i=0; i<n; ++i) {
            Mat2 Ul, Pl;
            polar_decomposition_2x2(mat_from_index_2x2(J,i,n), Ul, Pl);
            index_into_rotmat_2d(Ul, U, i, n);
            index_into_sym_2x2(Pl, P, i, n);
        }
    } else if(dim==3) {
        if(U.size()!=4*n) {
            U.resize(4*n,1);
        }
        if(P.size()!=6*n) {
            P.resize(6*n,1);
        }
        for(Eigen::Index i=0; i<n; ++i) {
            Mat3 Ul, Pl;
            polar_decomposition_3x3(mat_from_index_3x3(J,i,n), Ul, Pl);
            index_into_rotmat_3d(Ul, U, i, n);
            index_into_sym_3x3(Pl, P, i, n);
        }
    }
}


template <typename DerivedJ, typename DerivedU, typename DerivedP,
typename Scalartol>
void
parametrization::polar_decomposition_2x2(const Eigen::MatrixBase<DerivedJ>& J,
                                         Eigen::PlainObjectBase<DerivedU>& U,
                                         Eigen::PlainObjectBase<DerivedP>& P,
                                         const Scalartol tol)
{
    using Scalar = typename DerivedJ::Scalar;
    using Mat = Eigen::Matrix<Scalar, 2, 2>;
    using Vec = Eigen::Matrix<Scalar, 2, 1>;
    
    parametrization_assert(J.rows()==2 && J.cols()==2 && "J must be 2x2.");
    parametrization_assert(tol>0 && "tol must be positive.");
    
    if(U.rows()!=2 || U.cols()!=2) {
        U.resize(2,2);
    }
    if(P.rows()!=2 || P.cols()!=2) {
        P.resize(2,2);
    }
    
    //In the future this can potentially do something smart, at the moment just
    // call Eigen's generic SVD.
    Eigen::JacobiSVD<Mat> svd;
    svd.compute(J, Eigen::ComputeFullU | Eigen::ComputeFullV);
    
    U = svd.matrixU() * svd.matrixV().transpose();
    Vec d;
    bool negativeDet = false;
    if(U.determinant() < 0) {
        U.col(1) *= -1.;
        d = Vec::Constant(tol);
        negativeDet = true;
    } else {
        d = svd.singularValues();
        for(int i=0; i<d.size(); ++i) {
            if(d(i) < tol) {
                d(i) = tol;
            }
        }
    }
    P = svd.matrixV() * d.asDiagonal() * svd.matrixV().transpose();
    
    //Make sure P is symmetric (doesn't have to be based on numerical error).
    P(0,1) = P(1,0);
    
    parametrization_assert(std::abs(U.determinant() - 1.) <
        sqrt(std::numeric_limits<Scalar>::epsilon()) &&
        "U is not a rotation.");
    parametrization_assert
    ((U.transpose()*U - Eigen::Matrix<Scalar,2,2>::Identity())
     .squaredNorm() < sqrt(std::numeric_limits<Scalar>::epsilon()) &&
     "U is not a rotation.");
    parametrization_assert(P.determinant()>0 && P.trace()>0);
    parametrization_assert(negativeDet || (U*P-J).norm() < 10*tol);
}


template <typename DerivedJ, typename DerivedU, typename DerivedP,
typename Scalartol>
void
parametrization::polar_decomposition_3x3(const Eigen::MatrixBase<DerivedJ>& J,
                                         Eigen::PlainObjectBase<DerivedU>& U,
                                         Eigen::PlainObjectBase<DerivedP>& P,
                                         const Scalartol tol)
{
    using Scalar = typename DerivedJ::Scalar;
    using Mat = Eigen::Matrix<Scalar, 3, 3>;
    using Vec = Eigen::Matrix<Scalar, 3, 1>;
    
    parametrization_assert(J.rows()==3 && J.cols()==3 && "J must be 3x3.");
    parametrization_assert(tol>0 && "tol must be positive.");
    
    if(U.rows()!=3 || U.cols()!=3) {
        U.resize(3,3);
    }
    if(P.rows()!=3 || P.cols()!=3) {
        P.resize(3,3);
    }
    
    //In the future this can potentially do something smart, at the moment just
    // call Eigen's generic SVD.
    Eigen::JacobiSVD<Mat> svd;
    svd.compute(J, Eigen::ComputeFullU | Eigen::ComputeFullV);
    
    U = svd.matrixU() * svd.matrixV().transpose();
    Vec d;
    bool negativeDet = false;
    if(U.determinant() < 0) {
        U.col(2) *= -1.;
        d = Vec::Constant(tol);
        negativeDet = true;
    } else {
        d = svd.singularValues();
        for(int i=0; i<d.size(); ++i) {
            if(d(i) < tol) {
                d(i) = tol;
            }
        }
    }
    P = svd.matrixV() * d.asDiagonal() * svd.matrixV().transpose();
    
    //Make sure P is symmetric (doesn't have to be based on numerical error).
    P(0,1) = P(1,0);
    P(0,2) = P(2,0);
    P(1,2) = P(2,1);
    
    parametrization_assert(std::abs(U.determinant() - 1.) <
        sqrt(std::numeric_limits<Scalar>::epsilon()) &&
        "U is not a rotation.");
    parametrization_assert
    ((U.transpose()*U - Eigen::Matrix<Scalar,3,3>::Identity())
     .squaredNorm() < sqrt(std::numeric_limits<Scalar>::epsilon()) &&
     "U is not a rotation.");
    parametrization_assert(P.determinant()>0 && P.trace()>0);
    parametrization_assert(negativeDet || (U*P-J).norm() < 10*tol);
}


// Explicit template instantiation
template void parametrization::polar_decomposition<Eigen::Matrix<double, -1, 2, 0, -1, 2>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, double>(Eigen::MatrixBase<Eigen::Matrix<double, -1, 2, 0, -1, 2> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&, double);
template void parametrization::polar_decomposition<Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, double>(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&, double);
