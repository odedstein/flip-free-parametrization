
#include "sqrtm.h"

#include <Eigen/Dense>

#include "parametrization_assert.h"
#include "symmetrize.h"


template <typename DerivedC>
Eigen::Matrix<typename DerivedC::Scalar, 2, 2>
parametrization::sqrtm_2x2(const Eigen::MatrixBase<DerivedC>& C)
{
    parametrization_assert(C.array().isFinite().all() && "Invalid values in C");
    parametrization_assert(C.rows()==2 && C.cols()==2 &&
                           "This function is for 2x2 matrices.");
    parametrization_assert(C(1,0)==C(0,1) &&
                           "This function is for symmetric matrices.");
    parametrization_assert(C.determinant()>0 && C.trace()>0 &&
           "This function is for positive definite matrices.");
    
    using Scalar = typename DerivedC::Scalar;
    using Mat2 = Eigen::Matrix<typename DerivedC::Scalar, 2, 2>;
    //If the discriminant is below this tol, fallback.
    const Scalar tol = sqrt(std::numeric_limits<Scalar>::epsilon());
    
    //Algorithm from Franca, 1988. "An Algorithm to Compute the Square Root of
    // a 3x3 Positive Definite Matrix.
    //const Scalar Ic = C.trace();
    const Scalar Ic = C(0,0) + C(1,1);
    //Scalar IIc = C.determinant();
    Scalar IIc = C(0,0)*C(1,1) - C(1,0)*C(0,1);
    //More numerically stable (avoids cancellation), but not sure if needed.
    //Scalar IIc = (C(0,0)-C(1,0))*C(0,1) - (C(1,1)-C(0,1))*C(1,0);
    if(IIc<0) {
        //Matrix is not positive definite, fix.
        IIc = 0;
    }
    const Scalar IIu = sqrt(IIc);
    Scalar discr = Ic + 2.*IIu;
    if(discr<tol) {
        //Matrix is not positive definite enough to divide by safely.
        //Since we only allow spd input, this must be numerical noise. Set discr
        // to a safe value.
        //discr = tol;
        return stable_sqrtm(C);
    }
    const Scalar Iu = sqrt(discr);
    Mat2 U = C;
    U(0,0) += IIu;
    U(1,1) += IIu;
    U /= Iu;
    parametrization_assert(U.eigenvalues().real().minCoeff()>0);
    
    return U;
}


template <typename DerivedC>
Eigen::Matrix<typename DerivedC::Scalar, 3, 3>
parametrization::sqrtm_3x3(const Eigen::MatrixBase<DerivedC>& C)
{
    parametrization_assert(C.array().isFinite().all() && "Invalid values in C");
    parametrization_assert(C.rows()==3 && C.cols()==3 &&
                           "This function is for 3x3 matrices.");
    parametrization_assert(C(1,0)==C(0,1) && C(2,0)==C(0,2) && C(1,2)==C(2,1) &&
                           "This function is for symmetric matrices.");
    parametrization_assert(C.eigenvalues().real().minCoeff()>0 &&
                           "This funtion is for positive definite matrices");
    
    using Scalar = typename DerivedC::Scalar;
    using Mat3 = Eigen::Matrix<typename DerivedC::Scalar, 3, 3>;
    //If the discriminant is below this tol, fallback.
    const Scalar tol = sqrt(std::numeric_limits<Scalar>::epsilon());
    
    const auto trace = [] (auto& X) {
        return X(0,0) + X(1,1) + X(2,2);
    };
    const auto det = [] (auto& X) {
        return Eigen::internal::determinant_impl<DerivedC, 3>::run(X);
    };
    
    //Algorithm from Franca, 1988. "An Algorithm to Compute the Square Root of
    // a 3x3 Positive Definite Matrix.
    //There is a typo in the algorithm, see comment below.
    //const Scalar Ic = C.trace();
    const Scalar Ic = trace(C);
    //const Scalar IIc = 0.5*(Ic*Ic - (C*C).trace());
    const Scalar IIc = 0.5*(Ic*Ic - trace(C*C));
    //const Scalar IIIc = C.determinant();
    const Scalar IIIc = det(C);
    const Scalar k = Ic*Ic - 3.*IIc;
    if(k<tol) {
        //const Scalar lambda = sqrt(Ic/3.);
        //return lambda * Mat3::Identity();
        return stable_sqrtm(C);
    }
    //This is a typo in Franca 1988, where the article says Ic*Ic*(Ic-9./2.*IIc)
    const Scalar l = Ic*(Ic*Ic-9./2.*IIc) + 27./2.*IIIc;
    const Scalar phi = acos((std::min)(1., (std::max)(-1., l/std::pow(k,1.5))));
    const Scalar lambda_sq = (Ic + 2.*sqrt(k)*cos(phi/3.)) / 3.;
    //lambda_sq is positive by construction, phi<=pi so cos(phi/3)>0, and k>0.
    // If it is below tolerance, it must be a numerical error, and we
    // can clamp it to that tolerance.
    const Scalar lambda = lambda_sq<tol*tol ? tol : sqrt(lambda_sq);
    //Same argument with IIIc: it must be positive, since C is positive
    // definite. So, if it isn't, it must be some numerical error and we can
    // clamp it to zero.
    const Scalar IIIu = IIIc<0. ? 0. : sqrt(IIIc);
    const Scalar tsq = -lambda_sq + Ic + 2.*IIIu/lambda;
    //The usual argument. If this is not positive, we can clamp it to zero, as
    // it must be a numerical error.
    const Scalar Iu = tsq<0 ? lambda : lambda + sqrt(tsq);
    const Scalar Iu_sq = Iu*Iu;
    const Scalar IIu = (Iu_sq - Ic) / 2.;
    Scalar discr = Iu*IIu - IIIu;
    //We must be able to divide by this discriminant, which is always possible
    // for a positive definite matrix. If we cannot, this must be a numerical
    // error, and we can clamp to tolerance.
    if(std::abs(discr) < tol) {
        //discr = discr>0 ? tol : -tol;
        return stable_sqrtm(C);
    }
    //If C is symmetric, then C*C is also symmetric. However, because of
    // floating point properties this is not always true, so we need to ensure
    // that it is.
    Mat3 Csq = C*C;
    Csq(1,0) = Csq(0,1);
    Csq(2,0) = Csq(0,2);
    Csq(2,1) = Csq(1,2);
    Mat3 U = Iu*IIIu*Mat3::Identity() + (Iu_sq-IIu)*C - Csq;
    U /= discr;
    parametrization_assert(U.eigenvalues().real().minCoeff()>0);
    
    return U;
}


template <typename DerivedC>
Eigen::Matrix<typename DerivedC::Scalar,
DerivedC::RowsAtCompileTime, DerivedC::ColsAtCompileTime>
parametrization::stable_sqrtm(const Eigen::MatrixBase<DerivedC>& C)
{
    using Scalar = typename DerivedC::Scalar;
    using Mat = Eigen::Matrix<Scalar, DerivedC::RowsAtCompileTime,
    DerivedC::ColsAtCompileTime>;
    using Vec = Eigen::Matrix<Scalar, DerivedC::RowsAtCompileTime, 1>;
    using EigenSolver = Eigen::SelfAdjointEigenSolver<DerivedC>;
    
    const Scalar tol = std::numeric_limits<Scalar>::epsilon();
    
    parametrization_assert(C.array().isFinite().all() && "Invalid values in C");
    parametrization_assert(C==C.transpose() &&
                           "This function is for symmetric matrices.");
    parametrization_assert(C.eigenvalues().real().minCoeff()>0 &&
                           "This funtion is for positive definite matrices");
    
    const int dim = DerivedC::RowsAtCompileTime>=0 ?
    DerivedC::RowsAtCompileTime : C.rows();
    
    EigenSolver eigenSolver(C);
    Vec diag = eigenSolver.eigenvalues();
    // C = V*diag*V'
    Mat V = eigenSolver.eigenvectors();
    parametrization_assert((V.transpose()*V - Mat::Identity(dim,dim))
                           .squaredNorm() < tol);
    parametrization_assert
    ((V*diag.asDiagonal()*V.transpose() - C).squaredNorm() < tol);
    
    Vec diagSqrt = diag.array().sqrt().matrix();
    for(int i=0; i<dim; ++i) {
        if(!std::isfinite(diagSqrt(i)) || diagSqrt(i)<=0) {
            diagSqrt(i) = std::numeric_limits<Scalar>::min();
        }
    }
    
    Mat sqrtC = V * diagSqrt.asDiagonal() * V.transpose();
    symmetrize(sqrtC);
    parametrization_assert(C.eigenvalues().real().minCoeff()>0);
    parametrization_assert((sqrtC*sqrtC - C).squaredNorm() < tol);
    
    return sqrtC;
}


// Explicit template instantiation
template Eigen::Matrix<Eigen::Matrix<double, 2, 2, 0, 2, 2>::Scalar, 2, 2, 0, 2, 2> parametrization::sqrtm_2x2<Eigen::Matrix<double, 2, 2, 0, 2, 2> >(Eigen::MatrixBase<Eigen::Matrix<double, 2, 2, 0, 2, 2> > const&);
template Eigen::Matrix<Eigen::Matrix<double, 3, 3, 0, 3, 3>::Scalar, 3, 3, 0, 3, 3> parametrization::sqrtm_3x3<Eigen::Matrix<double, 3, 3, 0, 3, 3> >(Eigen::MatrixBase<Eigen::Matrix<double, 3, 3, 0, 3, 3> > const&);
template Eigen::Matrix<Eigen::Matrix<float, 2, 2, 0, 2, 2>::Scalar, 2, 2, 0, 2, 2> parametrization::sqrtm_2x2<Eigen::Matrix<float, 2, 2, 0, 2, 2> >(Eigen::MatrixBase<Eigen::Matrix<float, 2, 2, 0, 2, 2> > const&);
template Eigen::Matrix<Eigen::Matrix<float, 3, 3, 0, 3, 3>::Scalar, 3, 3, 0, 3, 3> parametrization::sqrtm_3x3<Eigen::Matrix<float, 3, 3, 0, 3, 3> >(Eigen::MatrixBase<Eigen::Matrix<float, 3, 3, 0, 3, 3> > const&);
template Eigen::Matrix<Eigen::Matrix<double, -1, -1, 0, -1, -1>::Scalar, 2, 2, 0, 2, 2> parametrization::sqrtm_2x2<Eigen::Matrix<double, -1, -1, 0, -1, -1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&);
template Eigen::Matrix<Eigen::Matrix<double, -1, -1, 0, -1, -1>::Scalar, 3, 3, 0, 3, 3> parametrization::sqrtm_3x3<Eigen::Matrix<double, -1, -1, 0, -1, -1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&);
template Eigen::Matrix<Eigen::Matrix<float, -1, -1, 0, -1, -1>::Scalar, 2, 2, 0, 2, 2> parametrization::sqrtm_2x2<Eigen::Matrix<float, -1, -1, 0, -1, -1> >(Eigen::MatrixBase<Eigen::Matrix<float, -1, -1, 0, -1, -1> > const&);
template Eigen::Matrix<Eigen::Matrix<float, -1, -1, 0, -1, -1>::Scalar, 3, 3, 0, 3, 3> parametrization::sqrtm_3x3<Eigen::Matrix<float, -1, -1, 0, -1, -1> >(Eigen::MatrixBase<Eigen::Matrix<float, -1, -1, 0, -1, -1> > const&);

