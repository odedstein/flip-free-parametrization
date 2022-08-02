
#include "energy.h"

#include "parametrization_assert.h"
#include "sym_from_triu.h"
#include "mat_from_index.h"

#include <limits>

#include <Eigen/Dense>


namespace parametrization {
namespace parametrization_helper_int {
    //Energy from SPD helper function to compute the per-element energy of a nxn
    // matrix, without weighing by area.
    //This function does not check for inversion, check before calling this.
    //Only for internal use
    template <parametrization::EnergyType energyT, int dim, typename DerivedX>
    static inline
    typename DerivedX::Scalar
    energy_for_element(const Eigen::MatrixBase<DerivedX>& X) {
        using Scalar = typename DerivedX::Scalar;
        const Scalar tol = std::numeric_limits<Scalar>::epsilon();
        
        static_assert(dim==2 || dim==3, "Only dims 2 and 3 are supported.");
        parametrization_assert(X.array().isFinite().all() &&
                               "Invalid entries in X.");
        
        switch(energyT) {
            case EnergyType::SymmetricGradient:
            {
                Scalar det = X.determinant();
                //P is strictly positive definite, so if the determinant is 0 here,
                // this must be a numerical error, and we can clamp.
                if(det<=0) {
                    det = std::numeric_limits<Scalar>::min();
                }
                //f(P) = 0.5*||P||_F^2 - log(det(P))
                return 0.5*X.squaredNorm() - log(det);
            }
            case EnergyType::SymmetricDirichlet:
            {
                Scalar invNorm = X.inverse().squaredNorm();
                //P is strictly positive definite, so if taking the inverse
                // failed this must be a numerical error, and we can just
                // set the energy to be very large.
                if(!std::isfinite(invNorm)) {
                    invNorm = 1. / tol;
                }
                //f(P) = 0.5*(||P||_F^2 + ||inv(P)||^2)
                return 0.5 * (X.squaredNorm() + invNorm);
            }
        }
    }
}
}


template <parametrization::EnergyType energyT, int idim,
typename DerivedP, typename Derivedw>
typename DerivedP::Scalar
parametrization::energy_from_P(const Eigen::MatrixBase<DerivedP>& P,
       const Eigen::MatrixBase<Derivedw>& w)
{
    using Scalar = typename DerivedP::Scalar;
    using Mat22 = Eigen::Matrix<Scalar, 2, 2>;
    using Mat33 = Eigen::Matrix<Scalar, 3, 3>;
    const Scalar tol = std::numeric_limits<Scalar>::epsilon();
    
    const int dim = idim>0 ? idim : (P.size()/w.size()==3 ? 2 : 3);
    parametrization_assert((dim==2 || dim==3) && "dim must be 2 or 3.");
    
    parametrization_assert(P.array().isFinite().all() &&
                           "Invalid entries in P");
    parametrization_assert(w.array().isFinite().all() &&
                           "Invalid entries in w");
    
    const Eigen::Index n = w.size();
    parametrization_assert(((dim==2 && 3*n==P.size()) ||
                            (dim==3 && 6*n==P.size())) &&
                           "P has the wrong size.");
    parametrization_assert(n==w.size() && "w does not correspond to P.");
    
    Scalar E = 0;
    for(Eigen::Index i=0; i<n; ++i) {
        if(dim==2) {
            const Mat22 Pl = sym_2d_from_index(P, i, n);
            E += w(i) * parametrization_helper_int::energy_for_element
            <energyT,2>(Pl);
        } else {
            const Mat33 Pl = sym_3d_from_index(P, i, n);
            E += w(i) * parametrization_helper_int::energy_for_element
            <energyT,3>(Pl);
        }
    }
    
    return E;
}


template <parametrization::EnergyType energyT, int idim,
typename DerivedGW, typename Derivedw>
typename DerivedGW::Scalar
parametrization::energy_from_GW(const Eigen::MatrixBase<DerivedGW>& GW,
                                const Eigen::MatrixBase<Derivedw>& w)
{
    using Scalar = typename DerivedGW::Scalar;
    using Mat22 = Eigen::Matrix<Scalar, 2, 2>;
    using Mat33 = Eigen::Matrix<Scalar, 3, 3>;
    const Scalar tol = std::numeric_limits<Scalar>::epsilon();
    
    const int dim = idim>0 ? idim : GW.cols();
    parametrization_assert((dim==2 || dim==3) && "dim must be 2 or 3.");
    
    parametrization_assert(GW.array().isFinite().all() &&
                           "Invalid entries in GW");
    parametrization_assert(w.array().isFinite().all() &&
                           "Invalid entries in w");
    
    const Eigen::Index n = w.size();
    parametrization_assert(n*dim == GW.rows() && dim == GW.cols() &&
                           "GW has the wrong size.");
    parametrization_assert(n==w.size() && "w does not correspond to GW.");
    
    Scalar E = 0;
    for(Eigen::Index i=0; i<n; ++i) {
        if(dim==2) {
            const Mat22 GWl = mat_from_index_2x2(GW, i, n);
            if(GWl.determinant()<=0) {
                return std::numeric_limits<Scalar>::infinity();
            }
            E += w(i) * parametrization_helper_int::energy_for_element
            <energyT,2>(GWl);
        } else {
            const Mat33 GWl = mat_from_index_3x3(GW, i, n);
            if(GWl.determinant()<=0) {
                return std::numeric_limits<Scalar>::infinity();
            }
            E += w(i) * parametrization_helper_int::energy_for_element
            <energyT,3>(GWl);
        }
    }
    
    return E;
}


// Explicit template instantiation
template Eigen::Matrix<double, -1, 1, 0, -1, 1>::Scalar parametrization::energy_from_P<(parametrization::EnergyType)0, 2, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&);
template Eigen::Matrix<double, -1, 1, 0, -1, 1>::Scalar parametrization::energy_from_P<(parametrization::EnergyType)0, 3, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&);
template Eigen::Matrix<double, -1, 1, 0, -1, 1>::Scalar parametrization::energy_from_P<(parametrization::EnergyType)1, 2, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&);
template Eigen::Matrix<double, -1, 1, 0, -1, 1>::Scalar parametrization::energy_from_P<(parametrization::EnergyType)1, 3, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&);
template Eigen::Matrix<double, -1, -1, 0, -1, -1>::Scalar parametrization::energy_from_GW<(parametrization::EnergyType)0, 2, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&);
template Eigen::Matrix<double, -1, -1, 0, -1, -1>::Scalar parametrization::energy_from_GW<(parametrization::EnergyType)1, 2, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&);
template Eigen::Matrix<double, -1, -1, 0, -1, -1>::Scalar parametrization::energy_from_GW<(parametrization::EnergyType)0, 3, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&);
template Eigen::Matrix<double, -1, -1, 0, -1, -1>::Scalar parametrization::energy_from_GW<(parametrization::EnergyType)1, 3, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&);
template Eigen::Matrix<double, -1, -1, 0, -1, -1>::Scalar parametrization::energy_from_GW<(parametrization::EnergyType)0, -1, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&);
template Eigen::Matrix<double, -1, -1, 0, -1, -1>::Scalar parametrization::energy_from_GW<(parametrization::EnergyType)1, -1, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&);
