
#include "spd_quartic_polynomial.h"

#include "symmetrize.h"
#include "quartic_polynomial.h"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>


template <typename DerivedB, typename Scalarc, typename DerivedP>
void
parametrization::spd_quartic_polynomial(const Eigen::MatrixBase<DerivedB>& B,
                                        const Scalarc c,
                                        Eigen::PlainObjectBase<DerivedP>& P)
{
    using Scalar = typename DerivedP::Scalar;
    using Mat = Eigen::Matrix<Scalar, DerivedB::RowsAtCompileTime,
    DerivedB::ColsAtCompileTime>;
    using Vec = Eigen::Matrix<Scalar, DerivedB::RowsAtCompileTime, 1>;
    using EigenSolver = Eigen::SelfAdjointEigenSolver<DerivedB>;
    
    const Scalar tol = std::numeric_limits<Scalar>::epsilon();
    const Scalar rescaleIfSmaller = sqrt(sqrt(tol));
    
    const int dim = DerivedB::RowsAtCompileTime>=0 ?
    DerivedB::RowsAtCompileTime : B.rows();
    const Scalar z = 0.;
    Scalar cc = static_cast<Scalar>(c);
    
    parametrization_assert(B.array().isFinite().all() &&
                           "illegal entries in B");
    parametrization_assert(c<0 && "c must be negative.");
    parametrization_assert((dim==B.rows() && dim==B.cols()) &&
                           "B does not have the right dimension.");
    parametrization_assert(B==B.transpose() && "B not symmetric");
    
    EigenSolver eigenSolver(B);
    Vec diagB = eigenSolver.eigenvalues().template cast<Scalar>();
    // B = V*diagB*V'
    Mat V = eigenSolver.eigenvectors().template cast<Scalar>();
    parametrization_assert((V.transpose()*V - Mat::Identity(dim,dim))
                           .squaredNorm() < tol);
    parametrization_assert
    ((V*diagB.asDiagonal()*V.transpose() - B).squaredNorm() < tol);
    
    //Solve one guartic polynomial in each dimension, store result in diagP.
    Vec diagP;
    if(DerivedB::RowsAtCompileTime < 0) {
        diagP.resize(dim);
    }
    bool numericalFix = false;
    for(int i=0; i<dim; ++i) {
        Scalar rescaledBy = 1;
        //Rescale if coefficients are very small, for numerical reasons.
        if(std::abs(diagB(i)) < rescaleIfSmaller &&
           std::abs(cc)<rescaleIfSmaller) {
            rescaledBy = -sqrt(sqrt(-cc));
            diagP(i) = quartic_polynomial(diagB(i)/rescaledBy, z, z,
                                          static_cast<Scalar>(-1.));
        } else {
            diagP(i) = quartic_polynomial(diagB(i), z, z, cc);
        }
        if(!std::isfinite(diagP(i)) || diagP(i)<tol) {
            //No positive root found. This means that there is a numerical
            // issue, and the eigenvalue must be very small.
            diagP(i) = tol;
            numericalFix = true;
        }
        if(rescaledBy!=1 && !numericalFix) {
            diagP(i) *= rescaledBy;
        }
    }
    
    //Un-diagonalize P
    P = V * diagP.asDiagonal() * V.transpose();
    
    //For numerical reasons
    symmetrize(P);
    
    parametrization_assert
    (numericalFix ||
     (P*P*P*P + B*P*P*P + c*Mat::Identity(dim,dim)).squaredNorm() < tol);
}


// Explicit template instantiation
template void parametrization::spd_quartic_polynomial<Eigen::Matrix<double, 2, 2, 0, 2, 2>, double, Eigen::Matrix<double, 2, 2, 0, 2, 2> >(Eigen::MatrixBase<Eigen::Matrix<double, 2, 2, 0, 2, 2> > const&, double, Eigen::PlainObjectBase<Eigen::Matrix<double, 2, 2, 0, 2, 2> >&);
template void parametrization::spd_quartic_polynomial<Eigen::Matrix<double, 3, 3, 0, 3, 3>, double, Eigen::Matrix<double, 3, 3, 0, 3, 3> >(Eigen::MatrixBase<Eigen::Matrix<double, 3, 3, 0, 3, 3> > const&, double, Eigen::PlainObjectBase<Eigen::Matrix<double, 3, 3, 0, 3, 3> >&);
