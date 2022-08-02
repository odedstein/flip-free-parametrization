#undef DEBUG_PARAMETRIZATION_TERMINATION_CONDITIONS

#include "termination_conditions.h"

#include "parametrization_assert.h"
#include "mat_from_index.h"
#include "sym_from_triu.h"

#include <Eigen/Dense>
#ifdef DEBUG_PARAMETRIZATION_TERMINATION_CONDITIONS
#include <iostream>
#endif


template <int idim, typename DerivedU, typename DerivedP, typename DerivedGW,
typename DerivedGtLambda, typename Derivedmu, typename ScalareP,
typename ScalareD, typename ScalarpAbsTol, typename ScalarpRelTol,
typename ScalardAbsTol, typename ScalardRelTol>
bool
parametrization::primal_dual_termination_condition
 (const Eigen::MatrixBase<DerivedU>& U,
  const Eigen::MatrixBase<DerivedP>& P,
  const Eigen::MatrixBase<DerivedGW>& GW,
  const Eigen::MatrixBase<DerivedGtLambda>& GtLambda,
  const Eigen::MatrixBase<Derivedmu>& mu,
  const ScalareP eP,
  const ScalareD eD,
  const ScalarpAbsTol pAbsTol,
  const ScalarpRelTol pRelTol,
  const ScalardAbsTol dAbsTol,
  const ScalardRelTol dRelTol)
{
    using Scalar = typename DerivedU::Scalar;
    using Mat22 = Eigen::Matrix<Scalar, 2, 2>;
    using Mat33 = Eigen::Matrix<Scalar, 3, 3>;
    
    parametrization_assert(U.array().isFinite().all() &&
                           "Invalid entries in U");
    parametrization_assert(P.array().isFinite().all() &&
                           "Invalid entries in P");
    parametrization_assert(GW.array().isFinite().all() &&
                           "Invalid entries in GW");
    parametrization_assert(GtLambda.array().isFinite().all() &&
                           "Invalid entries in GtLambda");
    parametrization_assert(mu.array().isFinite().all() &&
                           "Invalid entries in mu");
    parametrization_assert(std::isfinite(eP) && std::isfinite(eD) &&
                           std::isfinite(pAbsTol) && std::isfinite(pRelTol) &&
                           std::isfinite(dAbsTol) && std::isfinite(dRelTol) &&
                           "Invalid errors or tolerances");
    
    const int dim = idim>0 ? idim : GW.cols();
    parametrization_assert((dim==2 || dim==3) && "dim must be 2 or 3.");
    
    const Eigen::Index n = dim==2 ? U.size()/2 : U.size()/4;
    parametrization_assert(((dim==2 && 2*n==U.size()) ||
                            (dim==3 && 4*n==U.size())) &&
                           "U has the wrong size");
    parametrization_assert(((dim==2 && 3*n==P.size()) ||
                            (dim==3 && 6*n==P.size())) &&
                           "P does not correspond to U.");
    parametrization_assert(dim*n==GW.rows() && dim==GW.cols() &&
                           "GW does not correspond to U.");
    parametrization_assert(dim==GtLambda.cols() &&
                           "GtLambda has the wrong dimensions.");
    parametrization_assert(n==mu.size() && "mu does not correspond to U.");
    parametrization_assert(eP>=0 && eD>=0 && "Errors can not be negative.");
    parametrization_assert(pAbsTol>0 && pRelTol>0 && dAbsTol>0 && dRelTol>0 &&
                           "Tolerances must be positive.");
    
    Scalar pNorm = 0;
    for(Eigen::Index i=0; i<n; ++i) {
        if(dim==2) {
            const Mat22 Pl = sym_2d_from_index(P, i, n);
            pNorm += Pl.squaredNorm();
        } else {
            const Mat33 Pl = sym_3d_from_index(P, i, n);
            pNorm += Pl.squaredNorm();
        }
    }
    pNorm = sqrt(pNorm);
    
    const Scalar primalThreshold = sqrt(GW.rows()) * pAbsTol +
    (std::max)(GW.norm(), pNorm) * pRelTol;
    const Scalar dualThreshold = sqrt(GW.rows()) * dAbsTol +
    GtLambda.norm() * dRelTol;
    
#ifdef DEBUG_PARAMETRIZATION_TERMINATION_CONDITIONS
    std::cout << "eP is " << eP << " of " << primalThreshold*primalThreshold << "; ";
    std::cout << "eD is " << eD << " of " << dualThreshold*dualThreshold << "; ";
    std::cout << std::endl;
#endif
    
    // Careful: eP and eD are squared quantities
    return eP<primalThreshold*primalThreshold && eD<dualThreshold*dualThreshold;
}

template <int idim,
typename DerivedGW>
bool
parametrization::noflips_termination_condition
 (const Eigen::MatrixBase<DerivedGW>& GW)
{
    using Scalar = typename DerivedGW::Scalar;
    using Mat22 = Eigen::Matrix<Scalar, 2, 2>;
    using Mat33 = Eigen::Matrix<Scalar, 3, 3>;
    
    const int dim = idim>0 ? idim : GW.cols();
    parametrization_assert((dim==2 || dim==3) && "dim must be 2 or 3.");
    
    const Eigen::Index n = GW.rows() / dim;
    for(Eigen::Index i=0; i<n; ++i) {
        //Check whether each determinant is positive
        if(dim==2) {
            const Mat22 GWl = mat_from_index_2x2(GW, i, n);
            if(GWl.determinant()<=0) {
#ifdef DEBUG_PARAMETRIZATION_TERMINATION_CONDITIONS
                std::cout << "flip " << GWl.determinant() << " in " << i << std::endl;
#endif
                return false;
            }
        } else {
            const Mat33 GWl = mat_from_index_3x3(GW, i, n);
            if(GWl.determinant()<=0) {
#ifdef DEBUG_PARAMETRIZATION_TERMINATION_CONDITIONS
                std::cout << "flip " << GWl.determinant() << " in " << i << std::endl;
#endif
                return false;
            }
        }
    }
    
    return true;
}


template <int idim, typename DerivedU, typename DerivedP, typename DerivedGW,
typename DerivedGtLambda, typename Derivedmu,
typename ScalareP, typename ScalareD, typename ScalarpAbsTol,
typename ScalarpRelTol, typename ScalardAbsTol, typename ScalardRelTol>
bool
parametrization::primal_dual_noflips_termination_condition
(const Eigen::MatrixBase<DerivedU>& U,
 const Eigen::MatrixBase<DerivedP>& P,
 const Eigen::MatrixBase<DerivedGW>& GW,
 const Eigen::MatrixBase<DerivedGtLambda>& GtLambda,
 const Eigen::MatrixBase<Derivedmu>& mu,
 const ScalareP eP,
 const ScalareD eD,
 const ScalarpAbsTol pAbsTol,
 const ScalarpRelTol pRelTol,
 const ScalardAbsTol dAbsTol,
 const ScalardRelTol dRelTol)
{
    return primal_dual_termination_condition<idim>
    (U, P, GW, GtLambda, mu, eP, eD, pAbsTol, pRelTol, dAbsTol, dRelTol) &&
    noflips_termination_condition<idim>(GW);
}


template <int idim,
typename DerivedW0, typename DerivedW, typename Scalartol>
bool
parametrization::progress_termination_condition
 (const Eigen::MatrixBase<DerivedW0>& W0,
 const Eigen::MatrixBase<DerivedW>& W,
 const Scalartol tol)
{
    const int dim = idim>0 ? idim : W.cols();
    parametrization_assert((dim==2 || dim==3) && "dim must be 2 or 3.");
    
    parametrization_assert(W.array().isFinite().all() &&
                           "Invalid entries in W");
    parametrization_assert(W0.array().isFinite().all() &&
                           "Invalid entries in W0");
    parametrization_assert(W.cols()==dim && "dimensions of W wrong.");
    parametrization_assert(W0.cols()==dim && "dimensions of W0 wrong.");
    parametrization_assert(W.rows()==W0.rows() &&
                           "W does not correspond to W0.");
    
#ifdef DEBUG_PARAMETRIZATION_TERMINATION_CONDITIONS
    std::cout << "progress " << (W-W0).squaredNorm() << " of " << W.rows() * tol*tol << std::endl;
#endif
    
    return (W-W0).squaredNorm() < W.rows() * tol*tol;
}


template <int idim,
typename DerivedGW, typename DerivedW0, typename DerivedW, typename Scalartol>
bool
parametrization::progress_noflips_termination_condition
 (const Eigen::MatrixBase<DerivedGW>& GW,
  const Eigen::MatrixBase<DerivedW0>& W0,
  const Eigen::MatrixBase<DerivedW>& W,
  const Scalartol tol)
{
    return progress_termination_condition<idim>(W0, W, tol) &&
    noflips_termination_condition<idim>(GW);
}


template <parametrization::EnergyType energy, int idim,
typename DerivedGW, typename Derivedw, typename Scalartol>
bool
parametrization::target_energy_termination_condition
 (const Eigen::MatrixBase<DerivedGW>& GW,
  const Eigen::MatrixBase<Derivedw>& w,
  const Scalartol tol)
{
#ifdef DEBUG_PARAMETRIZATION_TERMINATION_CONDITIONS
    std::cout << "en: " << parametrization::energy_from_GW<energy,idim>(GW, w) << std::endl;
#endif
    return parametrization::energy_from_GW<energy,idim>(GW, w) < tol;
}


template <parametrization::EnergyType energy, int idim,
typename DerivedGW, typename Derivedw, typename Scalartol>
bool
parametrization::target_energy_noflips_termination_condition
(const Eigen::MatrixBase<DerivedGW>& GW,
 const Eigen::MatrixBase<Derivedw>& w,
 const Scalartol tol)
{
    return target_energy_termination_condition<energy,idim>(GW, w, tol) &&
    noflips_termination_condition<idim>(GW);
}



// Explicit template instantiation
template bool parametrization::primal_dual_termination_condition<2, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Product<Eigen::Transpose<Eigen::SparseMatrix<double, 0, int> >, Eigen::Matrix<double, -1, -1, 0, -1, -1>, 0>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, double, double, double, double, double, double>(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Product<Eigen::Transpose<Eigen::SparseMatrix<double, 0, int> >, Eigen::Matrix<double, -1, -1, 0, -1, -1>, 0> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, double, double, double, double, double, double);
template bool parametrization::primal_dual_termination_condition<3, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Product<Eigen::Transpose<Eigen::SparseMatrix<double, 0, int> >, Eigen::Matrix<double, -1, -1, 0, -1, -1>, 0>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, double, double, double, double, double, double>(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Product<Eigen::Transpose<Eigen::SparseMatrix<double, 0, int> >, Eigen::Matrix<double, -1, -1, 0, -1, -1>, 0> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, double, double, double, double, double, double);
template bool parametrization::noflips_termination_condition<2, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&);
template bool parametrization::noflips_termination_condition<3, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&);
template bool parametrization::primal_dual_noflips_termination_condition<2, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Product<Eigen::Transpose<Eigen::SparseMatrix<double, 0, int> >, Eigen::Matrix<double, -1, -1, 0, -1, -1>, 0>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, double, double, double, double, double, double>(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Product<Eigen::Transpose<Eigen::SparseMatrix<double, 0, int> >, Eigen::Matrix<double, -1, -1, 0, -1, -1>, 0> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, double, double, double, double, double, double);
template bool parametrization::primal_dual_noflips_termination_condition<3, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Product<Eigen::Transpose<Eigen::SparseMatrix<double, 0, int> >, Eigen::Matrix<double, -1, -1, 0, -1, -1>, 0>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, double, double, double, double, double, double>(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Product<Eigen::Transpose<Eigen::SparseMatrix<double, 0, int> >, Eigen::Matrix<double, -1, -1, 0, -1, -1>, 0> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, double, double, double, double, double, double);
template bool parametrization::progress_termination_condition<2, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, double>(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, double);
template bool parametrization::progress_termination_condition<3, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, double>(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, double);
template bool parametrization::progress_noflips_termination_condition<2, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, double>(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, double);
template bool parametrization::progress_noflips_termination_condition<3, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, double>(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, double);
template bool parametrization::target_energy_termination_condition<(parametrization::EnergyType)0, 2, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, double>(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, double);
template bool parametrization::target_energy_termination_condition<(parametrization::EnergyType)1, 2, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, double>(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, double);
template bool parametrization::target_energy_termination_condition<(parametrization::EnergyType)0, 3, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, double>(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, double);
template bool parametrization::target_energy_termination_condition<(parametrization::EnergyType)1, 3, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, double>(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, double);
template bool parametrization::target_energy_noflips_termination_condition<(parametrization::EnergyType)0, 2, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, double>(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, double);
template bool parametrization::target_energy_noflips_termination_condition<(parametrization::EnergyType)1, 2, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, double>(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, double);
template bool parametrization::target_energy_noflips_termination_condition<(parametrization::EnergyType)0, 3, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, double>(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, double);
template bool parametrization::target_energy_noflips_termination_condition<(parametrization::EnergyType)1, 3, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, double>(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, double);

