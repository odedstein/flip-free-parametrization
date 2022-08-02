#ifndef PARAMETRIZATION_CALLBACKFUNCTION_H
#define PARAMETRIZATION_CALLBACKFUNCTION_H

#include <Eigen/Core>
#include <Eigen/Sparse>


namespace parametrization
{
//
// The state that is passed to a caller in the callback function during an
//  optimization.
// Look at the documentation of map_to to see what each argument stands for.
//
template <typename DerivedW, typename Int>
struct CallbackFunctionState {
    using VecI = Eigen::Matrix<Int, Eigen::Dynamic, 1>;
    using Scalar = typename DerivedW::Scalar;
    using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using MatXDim =
    Eigen::Matrix<Scalar, Eigen::Dynamic, DerivedW::ColsAtCompileTime>;
    using SparseMat = Eigen::SparseMatrix<Scalar>;
    
    const Scalar primalErr;
    const Scalar dualErr;
    const SparseMat& G;
    VecX& U;
    VecX& P;
    Eigen::PlainObjectBase<DerivedW>& W; //(This is the currently produced map!)
    MatXDim& Lambda;
    VecX& w;
    VecX& mu;
    VecI& fixed;
    MatXDim& fixedTo;
    SparseMat& Aeq;
    MatXDim& beq;
    
    CallbackFunctionState(const Scalar _primalErr,
                          const Scalar _dualErr,
                          const SparseMat& _G,
                          VecX& _U,
                          VecX& _P,
                          Eigen::PlainObjectBase<DerivedW>& _W,
                          MatXDim& _Lambda,
                          VecX& _w,
                          VecX& _mu,
                          VecI& _fixed,
                          MatXDim& _fixedTo,
                          SparseMat& _Aeq,
                          MatXDim& _beq) : primalErr(_primalErr),
    dualErr(_dualErr), G(_G), U(_U), P(_P), W(_W), Lambda(_Lambda), w(_w),
    mu(_mu), fixed(_fixed), fixedTo(_fixedTo), Aeq(_Aeq), beq(_beq) {}
};

// The generic type of a callback function that can be fed to map_to to be
//  executed at the end of each step.
// The function is templated by the type of W, the matrix that holds the UV
//  map
// Its arguments are references in the "state" struct.
// Set the first boolean changedPrec to true if you changed anything that
//  requires precomputation, so that all precomputations can happen again.
// Set the second boolean changed to true if you changed anything that needs
//  iteration behavior to reset (such as fixedTo and beq).
// The function returns a bool (if it returns true, abort the optimization
//     sucessfully, if it returns false, continue).
//
template <typename DerivedW=Eigen::MatrixXd, typename Int=int>
using CallbackFunction =
std::function<bool(CallbackFunctionState<DerivedW, Int>&, //state
                   bool&,                                 //changedPrec
                   bool&                                  //changed
                   )>;

// An empty callback function that does nothing.
template <typename DerivedW=Eigen::MatrixXd, typename Int=int>
inline
CallbackFunction<DerivedW, Int>
emptyCallbackF() {
    return CallbackFunction<DerivedW, Int>
    ([](auto&, auto&, auto&) {
        return false;
    });
}

}

#endif

