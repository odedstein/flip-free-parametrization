
#include "map_to.h"

#include "parametrization_assert.h"
#include "uv_to_jacobian.h"
#include "polar_decomposition.h"
#include "constrained_qp.h"
#include "argmin_P.h"
#include "argmin_U.h"
#include "argmin_W.h"
#include "step_Lambda.h"
#include "lagrangian.h"
#include "lagrangian_error.h"
#include "rescale_penalties.h"
#include "rescale_b_mumin.h"
#include "rescale_h.h"
#include "termination_conditions.h"
#include "mat_from_index.h"
#include "sym_from_triu.h"
#include "grad_f.h"
#include "energy.h"
#include "map_energy.h"

#include <igl/doublearea.h>
#include <igl/volume.h>
#include <igl/edge_lengths.h>
#include <igl/grad.h>
#include <igl/flipped_triangles.h>
#include <igl/massmatrix.h>
#include <igl/cotmatrix.h>

#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#ifdef SUITESPARSE_AVAILABLE
#include <Eigen/CholmodSupport>
#include <Eigen/UmfPackSupport>
#endif


template <bool parallelize, parametrization::EnergyType energy,
typename DerivedV, typename DerivedF,
typename DerivedW, typename OptsScalar,
typename Tf>
bool
parametrization::map_to
(const Eigen::MatrixBase<DerivedV>& V,
 const Eigen::MatrixBase<DerivedF>& F,
 Eigen::PlainObjectBase<DerivedW>& W,
 const OptimizationOptions<OptsScalar>& opts,
 const Tf& f)
{
    const int dim = W.cols();
    parametrization_assert((dim==2 || dim==3) && "dim must be 2 or 3.");
    
    if(dim==2) {
        return map_to_dim<2,parallelize,energy>(V, F, W, opts, f);
    } else {
        return map_to_dim<3,parallelize,energy>(V, F, W, opts, f);
    }
}


template <bool parallelize, parametrization::EnergyType energy,
typename DerivedV, typename DerivedF, typename Derivedfixed,
typename DerivedfixedTo, typename DerivedW, typename ScalarAeq,
typename Derivedbeq, typename OptsScalar,
typename Tf>
bool
parametrization::map_to
(const Eigen::MatrixBase<DerivedV>& V,
 const Eigen::MatrixBase<DerivedF>& F,
 const Eigen::MatrixBase<Derivedfixed>& fixed,
 const Eigen::MatrixBase<DerivedfixedTo>& fixedTo,
 const Eigen::SparseMatrix<ScalarAeq>& Aeq,
 const Eigen::MatrixBase<Derivedbeq>& beq,
 Eigen::PlainObjectBase<DerivedW>& W,
 const OptimizationOptions<OptsScalar>& opts,
 const Tf& f)
{
    const int dim = W.cols();
    parametrization_assert((dim==2 || dim==3) && "dim must be 2 or 3.");
    
    if(dim==2) {
        return map_to_dim<2,parallelize,energy>
        (V, F, fixed, fixedTo, Aeq, beq, W, opts, f);
    } else {
        return map_to_dim<3,parallelize,energy>
        (V, F, fixed, fixedTo, Aeq, beq, W, opts, f);
    }
}


template <int dim, bool parallelize, parametrization::EnergyType energy,
typename DerivedV, typename DerivedF,
typename DerivedW, typename OptsScalar,
typename Tf>
bool
parametrization::map_to_dim
 (const Eigen::MatrixBase<DerivedV>& V,
  const Eigen::MatrixBase<DerivedF>& F,
  Eigen::PlainObjectBase<DerivedW>& W,
  const parametrization::OptimizationOptions<OptsScalar>& opts,
  const Tf& f)
{
    using Int = typename DerivedF::Scalar;
    using VecI1 = Eigen::Matrix<Int, 1, 1>;
    using Scalar = typename DerivedW::Scalar;
    using SparseMat = Eigen::SparseMatrix<Scalar>;
    using MatXDim =
    Eigen::Matrix<Scalar, Eigen::Dynamic, DerivedW::ColsAtCompileTime>;
    
    //Fix just one vertex
    VecI1 fixed;
    fixed << 0;
    SparseMat Aeq;
    MatXDim fixedTo, beq;
    fixedTo.resize(1,dim);
    fixedTo.setZero();
    
    return map_to_dim<dim,parallelize,energy>
    (V, F, fixed, fixedTo, Aeq, beq, W, opts, f);
}


template <int dim, bool parallelize, parametrization::EnergyType energy,
typename DerivedV, typename DerivedF, typename Derivedfixed,
typename DerivedfixedTo, typename DerivedW, typename ScalarAeq,
typename Derivedbeq, typename OptsScalar,
typename Tf>
bool
parametrization::map_to_dim
 (const Eigen::MatrixBase<DerivedV>& V,
  const Eigen::MatrixBase<DerivedF>& F,
  const Eigen::MatrixBase<Derivedfixed>& fixed,
  const Eigen::MatrixBase<DerivedfixedTo>& fixedTo,
  const Eigen::SparseMatrix<ScalarAeq>& Aeq,
  const Eigen::MatrixBase<Derivedbeq>& beq,
  Eigen::PlainObjectBase<DerivedW>& W,
  const OptimizationOptions<OptsScalar>& opts,
  const Tf& f)
{
    using Int = typename DerivedF::Scalar;
    using VecI1 = Eigen::Matrix<Int, 1, 1>;
    using VecI = Eigen::Matrix<Int, Eigen::Dynamic, 1>;
    using Scalar = typename DerivedW::Scalar;
    using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
    using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using MatX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using MatXDim = Eigen::Matrix<Scalar, Eigen::Dynamic,
    DerivedW::ColsAtCompileTime>;
    using SparseMat = Eigen::SparseMatrix<Scalar>;
    //Matrix-vector multiplications in Eigen are faster with RowMajor matrices.
    using RSparseMat = Eigen::SparseMatrix<Scalar, Eigen::RowMajor>;
    using DiagMat = Eigen::DiagonalMatrix<Scalar, Eigen::Dynamic>;
    
    static_assert(std::is_same<Scalar, typename DerivedV::Scalar>::value,
                  "W and V must have the same scalar type.");
    static_assert(std::is_same<Scalar, ScalarAeq>::value,
                  "W and Aeq must have the same scalar type.");
    static_assert(std::is_same<Scalar, typename Derivedbeq::Scalar>::value,
                  "W and beq must have the same scalar type.");
    static_assert(std::is_same<Scalar, typename DerivedfixedTo::Scalar>::value,
                  "W and fixedTo must have the same scalar type.");
    static_assert(std::is_same<Int, typename Derivedfixed::Scalar>::value,
                  "F and fixed must have the same scalar type.");
    
#ifdef SUITESPARSE_AVAILABLE
    using SPDSolver = Eigen::CholmodSupernodalLLT<SparseMat>;
    using Solver = Eigen::UmfPackLU<SparseMat>;
#else
    using SPDSolver = Eigen::SimplicialLLT<SparseMat>;
    using Solver = Eigen::SparseLU<SparseMat>;
#endif
    //The problem is SPD if Aeq, beq are not provided.
    const bool spd = Aeq.size()==0;
    
    //Make sure OpenMP exists if parallelize is set to true
#ifndef OPENMP_AVAILABLE
    static_assert(!parallelize, "Can't parallelize without OpenMP.");
#endif
    
    parametrization_assert(V.array().isFinite().all() &&
                           "Invalid entries in V");
    parametrization_assert(F.array().isFinite().all() &&
                           "Invalid entries in F");
    parametrization_assert(W.array().isFinite().all() &&
                           "Invalid entries in W");
    
    const Int n=V.rows(), m=F.rows();
    parametrization_assert((V.cols()==2 || V.cols()==3) &&
                           "V,F must be a surface mesh in 2d or 3d.");
    static_assert((dim==2 || dim==3), "V,F must be a triangle or tet mesh.");
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
    }().minCoeff() > sqrt(std::numeric_limits<Scalar>::min()) &&
                           "Degenerate face/tet in W input.");
    parametrization_assert(W.rows()==V.rows() && "W does not correspond to V.");
    parametrization_assert(W.cols()==dim && "W has the wrong dimensions.");
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
    
    //Mesh geometry things
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
    G.makeCompressed();
    const RSparseMat Gr(G);
    
    //Penalty weights
    VecX mu = opts.mu0 * w;
    
    //Set up and initialize Lagrangian variables.
    MatXDim Lambda(dim*m, dim), GW;
    Lambda.setZero();
    VecX U, P;
    GW.noalias() = Gr * W;
    Scalar polarTol;
    switch(energy) {
        case EnergyType::SymmetricGradient:
            polarTol = sqrt(sqrt(std::numeric_limits<Scalar>::epsilon()));
            break;
        case EnergyType::SymmetricDirichlet:
            polarTol = sqrt(sqrt(sqrt(std::numeric_limits<Scalar>::epsilon())));
            break;
    }
    polar_decomposition(GW, U, P, polarTol);
    
    //Bounds
    VecX b=VecX::Zero(m), muMin=VecX::Zero(m), h=VecX::Zero(m);
    if(opts.minPenaltyBoundsActive && opts.turnPenaltyBoundsOnAt<0) {
        rescale_b_mumin<parallelize,energy,dim>(P, w, opts.bMargin, b, muMin);
        for(Int i=0; i<m; ++i) {
            if(mu(i) < muMin(i)) {
                mu(i) = muMin(i);
            }
        }
    }
    if(opts.proximalBoundsActive && opts.turnProximalBoundsOnAt<0) {
        rescale_h<parallelize>(mu, w, b, h);
    }
    const auto rescaling_ok = [&opts] (const auto& b, const auto& muMin) {
        return b.array().isFinite().all() && muMin.array().isFinite().all();
    };
    
    //Things that need to be precomputed, and a helper function to do it
    DiagMat M(dim*m);
    RSparseMat GtM;
    SparseMat GtMGmat;
    VecI fixedL = fixed;
    SparseMat AeqL = Aeq;
    MatXDim fixedToL = fixedTo;
    MatXDim beqL = beq;
    ConstrainedQPPrecomputed<Scalar, Int, SPDSolver> spdGtMG;
    ConstrainedQPPrecomputed<Scalar, Int, Solver> GtMG;
    const auto precompute =
    [&G, &mu, &M, &GtM, &GtMGmat, &spdGtMG, &GtMG, &fixedL, &AeqL, m, spd] () {
        for(int j=0; j<dim; ++j) {
            M.diagonal().segment(j*m, m) = mu;
        }
        GtM = G.transpose() * M;
        GtM.makeCompressed();
        GtMGmat = (GtM * G).template selfadjointView<Eigen::Upper>();
        GtMGmat.makeCompressed();
        if(spd) {
            return constrained_qp_precompute(GtMGmat, fixedL, AeqL, spdGtMG);
        } else {
            return constrained_qp_precompute(GtMGmat, fixedL, AeqL, GtMG);
        }
    };
    if(!precompute()) {
        return false;
    }
    
    //Helper functions to do the argmin_W step choosing correct precomputation
    const auto c_argmin_W =
    [&spdGtMG, &GtMG, &GtM, &Gr, &fixedToL, &beqL, spd]
    (const auto& U, const auto& P, const auto& Lambda, auto& W, auto& GW) {
        bool success;
        if(spd) {
            success = argmin_W<parallelize,dim>
            (spdGtMG, GtM, fixedToL, beqL, U, P, Lambda, W);
        } else {
            success = argmin_W<parallelize,dim>
            (GtMG, GtM, fixedToL, beqL, U, P, Lambda, W);
        }
        
        GW.noalias() = Gr * W;
        return success;
    };
    
    //Main optimization loop
    MatXDim U0, P0, GW0, W0;
    VecX ePs, eDs;
    Scalar eP, eD;
    int rescaleFrequency = opts.rescaleFrequency;
    for(int iter=0;
        (opts.terminationCondition==TerminationCondition::None) ||
        (iter<opts.maxIter);
        iter<std::numeric_limits<int>::max() ? ++iter : 0) {
        
        //Cache old iteration results.
        if(opts.terminationCondition==TerminationCondition::Progress ||
           opts.terminationCondition==TerminationCondition::ProgressNoflips) {
            W0 = W;
        }
        GW0 = GW;
        U0 = U;
        P0 = P;
        
        //Perform one Augmented Lagrangian iteration.
        if(!c_argmin_W(U, P, Lambda, W, GW)) {
            return false;
        }
        argmin_U<parallelize,dim>(P, GW, Lambda, h, U);
        argmin_P<parallelize,energy,dim>(U, GW, Lambda, w, mu, P);
        step_Lambda<parallelize,dim>(U, P, GW, Lambda);
        
        //Compute errors & rescale if needed
        if((opts.rescaleFrequency>0 &&
            (iter-opts.initialRescale)%rescaleFrequency==0) ||
           iter<opts.initialRescale) {
            //Rescale gradient bounds.
            if(opts.minPenaltyBoundsActive && iter>=opts.turnPenaltyBoundsOnAt)
            {
                rescale_b_mumin<parallelize,energy,dim>
                (P, w, opts.bMargin, b, muMin);
                if(!rescaling_ok(b, muMin)) {
                    return false;
                }
            }
            
            //Compute primal and dual error, rescale mu if needed.
            lagrangian_errors<parallelize,dim>
            (U, U0, P, P0, GW, GW0, mu, h, eP, eD, ePs, eDs);
            if(rescale_penalties<parallelize,dim>
               (ePs, eDs, opts.penaltyIncrease, opts.penaltyDecrease,
                opts.differenceToRescale, mu, Lambda, muMin)) {
                if(!precompute()) {
                    return false;
                }
            }
            
            //Rescale proximal bounds.
            if(opts.proximalBoundsActive && iter>=opts.turnProximalBoundsOnAt) {
                rescale_h<parallelize>(w, mu, b, h);
            }
            
            //Rescale less and less over time.
            if(iter>=opts.initialRescale) {
                rescaleFrequency += 0.5*rescaleFrequency + 0.5;
            }
        } else {
            lagrangian_error<false,dim>(U, U0, P, P0, GW, GW0, mu, h, eP, eD);
        }
        
        //Execute callback function.
        bool changedPrec=false, changed=false;
        CallbackFunctionState<DerivedW, Int> state
        {eP, eD, G, U, P, W, Lambda, w, mu, fixedL, fixedToL, AeqL, beqL};
        if(f(state, changedPrec, changed)) {
            return true;
        }
        if(changedPrec) {
            if(!precompute()) {
                return false;
            }
        }
        if(changedPrec || changed) {
            rescaleFrequency = opts.rescaleFrequency;
        }
        
        //Make sure that all assumptions still hold after the callback function
        // might have changed something.
        parametrization_assert(((dim==2 && U.size()==2*m) ||
                                (dim==3 && U.size()==4*m)) &&
                               "U has the wrong size.");
        parametrization_assert(((dim==2 && P.size()==3*m) ||
                                (dim==3 && P.size()==6*m)) &&
                               "P has the wrong size.");
        parametrization_assert(W.rows() == n && W.cols() == dim &&
                               "W has the wrong size.");
        parametrization_assert(Lambda.rows() == dim*m &&
                               Lambda.cols() == dim &&
                               "Lambda has the wrong size.");
        parametrization_assert(w.size() == m && "w has the wrong size.");
        parametrization_assert(mu.size() == m && "mu has the wrong size.");
        parametrization_assert((AeqL.size()==0)==(beqL.size()==0) &&
                               "Aeq is specified iff beq is specified.");
        parametrization_assert((AeqL.size()==0 || AeqL.cols()==V.rows()) &&
                               "Aeq does not correspond to V.");
        parametrization_assert((beqL.size()==0 || beqL.cols()==dim) &&
                               "beq has the wrong dimension.");
        parametrization_assert((beqL.size()==0 || beqL.rows()==Aeq.rows())
                               && "beq does not correspond to A.");
        parametrization_assert((fixedL.size()==0 ||
                                fixedL.size()==fixedToL.rows()) &&
                               "fixed does not correspond to fixedTo.");
        parametrization_assert((fixedToL.size()==0 || fixedToL.cols()==dim)
                               && "fixedTo has the wrong dimension.");
        parametrization_assert((fixedL.size()==0)==(fixedToL.size()==0) &&
                               "fixed is specified iff fixedTo is specified.");
        
        //If there is a problem with any of the iteration variables, abort.
        if(!W.array().isFinite().all() || !U.array().isFinite().all() ||
           !P.array().isFinite().all() || !Lambda.array().isFinite().all()) {
            return false;
        }
        
        //Check termination condition.
        switch(opts.terminationCondition) {
            case TerminationCondition::PrimalDualError:
                if(primal_dual_termination_condition<dim>
                   (U, P, GW, G.transpose()*Lambda, mu, eP, eD, opts.pAbsTol,
                    opts.pRelTol, opts.dAbsTol, opts.dRelTol)) {
                    return true;
                }
                break;
            case TerminationCondition::Noflips:
                if(noflips_termination_condition<dim>(GW)) {
                    return true;
                }
                break;
            case TerminationCondition::PrimalDualErrorNoflips:
                if(primal_dual_noflips_termination_condition<dim>
                   (U, P, GW, G.transpose()*Lambda, mu, eP, eD, opts.pAbsTol,
                    opts.pRelTol, opts.dAbsTol, opts.dRelTol)) {
                    return true;
                }
                break;
            case TerminationCondition::Progress:
                if(progress_termination_condition<dim>
                   (W0, W, opts.progressTol)) {
                    return true;
                }
                break;
            case TerminationCondition::ProgressNoflips:
                if(progress_noflips_termination_condition<dim>
                   (GW, W0, W, opts.progressTol)) {
                    return true;
                }
                break;
            case TerminationCondition::TargetEnergy:
                if(iter>0 &&
                   target_energy_termination_condition<energy,dim>
                   (GW, w, opts.targetEnergy)) {
                    return true;
                }
                break;
            case TerminationCondition::TargetEnergyNoflips:
                if(target_energy_noflips_termination_condition<energy,dim>
                   (GW, w, opts.targetEnergy)) {
                    return true;
                }
                break;
            case TerminationCondition::NumberOfIterations:
                if(iter >= opts.terminationIter) {
                    return true;
                }
                break;
            case TerminationCondition::NumberOfIterationsNoflips:
                if(noflips_termination_condition<dim>(GW)) {
                    return true;
                }
                break;
            case TerminationCondition::None:
                break;
            default:
                parametrization_assert(false &&
                                       "Invalid termination condition");
        }
    }
    
    return false;
}


// Explicit template instantiation
template bool parametrization::map_to<false, (parametrization::EnergyType)0, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, double, std::function<bool (parametrization::CallbackFunctionState<Eigen::Matrix<double, -1, -1, 0, -1, -1>, int>&, bool&, bool&)> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&, parametrization::OptimizationOptions<double> const&, std::function<bool (parametrization::CallbackFunctionState<Eigen::Matrix<double, -1, -1, 0, -1, -1>, int>&, bool&, bool&)> const&);
template bool parametrization::map_to<false, (parametrization::EnergyType)0, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, double, Eigen::Matrix<double, -1, -1, 0, -1, -1>, double, std::function<bool (parametrization::CallbackFunctionState<Eigen::Matrix<double, -1, -1, 0, -1, -1>, int>&, bool&, bool&)> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::SparseMatrix<double, 0, int> const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&, parametrization::OptimizationOptions<double> const&, std::function<bool (parametrization::CallbackFunctionState<Eigen::Matrix<double, -1, -1, 0, -1, -1>, int>&, bool&, bool&)> const&);
template bool parametrization::map_to<false, (parametrization::EnergyType)1, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, double, std::function<bool (parametrization::CallbackFunctionState<Eigen::Matrix<double, -1, -1, 0, -1, -1>, int>&, bool&, bool&)> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&, parametrization::OptimizationOptions<double> const&, std::function<bool (parametrization::CallbackFunctionState<Eigen::Matrix<double, -1, -1, 0, -1, -1>, int>&, bool&, bool&)> const&);
template bool parametrization::map_to<false, (parametrization::EnergyType)1, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, double, Eigen::Matrix<double, -1, -1, 0, -1, -1>, double, std::function<bool (parametrization::CallbackFunctionState<Eigen::Matrix<double, -1, -1, 0, -1, -1>, int>&, bool&, bool&)> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::SparseMatrix<double, 0, int> const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&, parametrization::OptimizationOptions<double> const&, std::function<bool (parametrization::CallbackFunctionState<Eigen::Matrix<double, -1, -1, 0, -1, -1>, int>&, bool&, bool&)> const&);
#ifdef OPENMP_AVAILABLE
template bool parametrization::map_to<true, (parametrization::EnergyType)0, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, double, std::function<bool (parametrization::CallbackFunctionState<Eigen::Matrix<double, -1, -1, 0, -1, -1>, int>&, bool&, bool&)> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&, parametrization::OptimizationOptions<double> const&, std::function<bool (parametrization::CallbackFunctionState<Eigen::Matrix<double, -1, -1, 0, -1, -1>, int>&, bool&, bool&)> const&);
template bool parametrization::map_to<true, (parametrization::EnergyType)0, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, double, Eigen::Matrix<double, -1, -1, 0, -1, -1>, double, std::function<bool (parametrization::CallbackFunctionState<Eigen::Matrix<double, -1, -1, 0, -1, -1>, int>&, bool&, bool&)> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::SparseMatrix<double, 0, int> const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&, parametrization::OptimizationOptions<double> const&, std::function<bool (parametrization::CallbackFunctionState<Eigen::Matrix<double, -1, -1, 0, -1, -1>, int>&, bool&, bool&)> const&);
template bool parametrization::map_to<true, (parametrization::EnergyType)1, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, double, std::function<bool (parametrization::CallbackFunctionState<Eigen::Matrix<double, -1, -1, 0, -1, -1>, int>&, bool&, bool&)> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&, parametrization::OptimizationOptions<double> const&, std::function<bool (parametrization::CallbackFunctionState<Eigen::Matrix<double, -1, -1, 0, -1, -1>, int>&, bool&, bool&)> const&);
template bool parametrization::map_to<true, (parametrization::EnergyType)1, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, double, Eigen::Matrix<double, -1, -1, 0, -1, -1>, double, std::function<bool (parametrization::CallbackFunctionState<Eigen::Matrix<double, -1, -1, 0, -1, -1>, int>&, bool&, bool&)> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::SparseMatrix<double, 0, int> const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&, parametrization::OptimizationOptions<double> const&, std::function<bool (parametrization::CallbackFunctionState<Eigen::Matrix<double, -1, -1, 0, -1, -1>, int>&, bool&, bool&)> const&);
#endif
