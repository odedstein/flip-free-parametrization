
#include "constrained_qp.h"

#include "parametrization_assert.h"

#include <vector>

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#ifdef SUITESPARSE_AVAILABLE
#include <Eigen/CholmodSupport>
#include <Eigen/UmfPackSupport>
#endif
#include <Eigen/IterativeLinearSolvers>


template <typename Solver,
typename ScalarQ, typename Derivedc, typename Derivedf, typename Derivedg,
typename ScalarA, typename Derivedb, typename Derivedz>
bool
parametrization::constrained_qp(const Eigen::SparseMatrix<ScalarQ>& Q,
                                const Eigen::MatrixBase<Derivedc>& c,
                                const Eigen::MatrixBase<Derivedf>& f,
                                const Eigen::MatrixBase<Derivedg>& g,
                                const Eigen::SparseMatrix<ScalarA>& A,
                                const Eigen::MatrixBase<Derivedb>& b,
                                Eigen::PlainObjectBase<Derivedz>& z)
{
    ConstrainedQPPrecomputed<ScalarQ, typename Derivedf::Scalar, Solver>
    precomputed;
    const bool success = constrained_qp_precompute(Q, f, A, precomputed);
    if(!success) {
        return false;
    }
    
    return constrained_qp_solve(precomputed, c, g, b, z);
}


template <typename Solver, typename ScalarQ, typename Derivedf,
typename ScalarA, typename ScalarPrec, typename IntPrec>
bool
parametrization::constrained_qp_precompute(const Eigen::SparseMatrix<ScalarQ>& Q,
                                           const Eigen::MatrixBase<Derivedf>& f,
                                           const Eigen::SparseMatrix<ScalarA>& A,
                                           ConstrainedQPPrecomputed<ScalarPrec, IntPrec, Solver>& precomputed)
{
    using Int = IntPrec;
    parametrization_assert(Q.rows() < std::numeric_limits<Int>::max() &&
           "Q is too large for the chosen integer type.");
    using VecI = Eigen::Matrix<Int, Eigen::Dynamic, 1>;
    
    //If the solver only uses the lower/upper triangular part of a symmetric
    // matrix,specify here, this will save time later.
    enum class TriangularType {Both, Lower, Upper};
    constexpr TriangularType triType = TriangularType::Both;
    
    //Every other scalar will be cast to this when precomputing
    using Scalar = ScalarPrec;
    
    //The sparse matrix type used and its triplet
    using SparseMat = Eigen::SparseMatrix<Scalar>;
    using Triplet = Eigen::Triplet<Scalar>;
    
    //Which parts of the problem are active, and with which dimension?
    precomputed.n = Q.rows();
    precomputed.m = f.size();
    precomputed.l = A.rows();
    const Int n = precomputed.n;
    const Int m = precomputed.m;
    const Int l = precomputed.l;
    
    //Verify preconditions
    parametrization_assert(n==Q.cols() && "Q must be square.");
    parametrization_assert((m==0 || (f.minCoeff()>=0 && f.maxCoeff()<n)) &&
    "Invalid indices in f.");
    parametrization_assert((l==0 || n==A.cols()) &&
                           "A does not correspond to Q.");
    
    //Store known and unknown, count how many are actually unknown
    SparseMat QU; //Q, but with only the unknown variables in it
    VecI fu_reverse, f_reverse; //reverse maps all->unknown, all->known if needed.
    std::vector<bool> isKnown; //bool vector determining known or not, if needed
    if(m>0) {
        precomputed.f = f.template cast<Int>();
        isKnown.resize(n, false);
        for(Int i=0; i<m; ++i) {
            isKnown[f(i)] = true;
        }
        Int mu = n; //Number of unknowns
        for(bool known : isKnown) {
            if(known) {
                --mu;
            }
        }
        precomputed.fu.resize(mu);
        fu_reverse.resize(n);
        f_reverse.resize(n);
        Int f_ind=0, fu_ind=0;
        for(Int i=0; i<n; ++i) {
            if(isKnown[i]) {
                fu_reverse(i) = -1;
                f_reverse(f(f_ind)) = f_ind;
                ++f_ind;
            } else {
                f_reverse(i) = -1;
                precomputed.fu(fu_ind) = i;
                fu_reverse(i) = fu_ind;
                ++fu_ind;
            }
        }
        parametrization_assert(mu>0 &&
                               "There has to be at least one unknown.");
        
        std::vector<Triplet> tripletListQU;
        //Should be if constexpr, but libigl does not work with C++17
        if(triType==TriangularType::Both) {
            tripletListQU.reserve(Q.nonZeros());
        } else {
            tripletListQU.reserve(Q.nonZeros());
        }
        
        for(Int k=0; k<Q.outerSize(); ++k) {
            for(typename SparseMat::InnerIterator it(Q,k); it; ++it) {
                //Should be if constexpr, but libigl does not work with C++17
                if(triType==TriangularType::Upper) {
                    if(it.row() > it.col()) {
                        continue;
                    }
                } else if(triType==TriangularType::Lower) {
                    if(it.row() < it.col()) {
                        continue;
                    }
                }
                const Int rrow=fu_reverse(it.row()), rcol=fu_reverse(it.col());
                if(rrow>=0 && rcol>=0) {
                    tripletListQU.emplace_back(rrow, rcol, it.value());
                }
            }
        }
        QU.resize(mu, mu);
        QU.setFromTriplets(tripletListQU.begin(), tripletListQU.end());
    }
    
    //Prepare the large system matrix into which we will put the information
    // from both Q and A.
    //This is only needed if A is actually supplied.
    SparseMat QA;
    if(l>0) {
        std::vector<Triplet> tripletListQA;
        //Should be if constexpr, but libigl does not work with C++17
        if(triType==TriangularType::Both) {
            tripletListQA.reserve(Q.nonZeros()+2*A.nonZeros());
        } else {
            tripletListQA.reserve(Q.nonZeros()+A.nonZeros());
        }
        
        //Which matrix are we building upon? If we removed some degrees of
        // freedom earlier, we now need to use QU.
        const auto& Qloc = (m==0 ? Q : QU);
        const Int nloc = Qloc.rows();
        for(Int k=0; k<Qloc.outerSize(); ++k) {
            for(typename SparseMat::InnerIterator it(Qloc,k); it; ++it) {
                //Should be if constexpr, but libigl does not work with C++17
                if(triType==TriangularType::Upper) {
                    if(it.row() > it.col()) {
                        continue;
                    }
                } else if(triType==TriangularType::Lower) {
                    if(it.row() < it.col()) {
                        continue;
                    }
                }
                tripletListQA.emplace_back(it.row(), it.col(), it.value());
            }
        }
        
        //Here we could technically remove linearly dependent constraints, but
        // we require that rows of Aeq are linearly independent as a
        // precondition.
        //This lets us avoid a QR decomposition at this point.
        for(Int k=0; k<A.outerSize(); ++k) {
            for(typename SparseMat::InnerIterator it(A,k); it; ++it) {
                const Int col = (m==0 ? it.col() : fu_reverse(it.col()));
                if(col<0) {
                    continue;
                }
                //Should be if constexpr, but libigl does not work with C++17
                if(triType==TriangularType::Upper) {
                    tripletListQA.emplace_back(col,nloc+it.row(),it.value());
                } else if(triType==TriangularType::Lower) {
                    tripletListQA.emplace_back(nloc+it.row(),col,it.value());
                } else {
                    tripletListQA.emplace_back(nloc+it.row(),col,it.value());
                    tripletListQA.emplace_back(col,nloc+it.row(),it.value());
                }
            }
        }
        
        QA.resize(nloc+l, nloc+l);
        QA.setFromTriplets(tripletListQA.begin(), tripletListQA.end());
    }
    
    
    //Construct the matrix used to build the right-hand side
    if(m>0) {
        std::vector<Triplet> tripletListY;
        if(triType==TriangularType::Both) {
            tripletListY.reserve(Q.nonZeros()+A.nonZeros());
        } else {
            tripletListY.reserve(2*Q.nonZeros()+A.nonZeros());
        }
        
        //Constructing Q(unknown, known);
        for(Int k=0; k<Q.outerSize(); ++k) {
            for(typename SparseMat::InnerIterator it(Q,k); it; ++it) {
                const Int row=it.row(), col=it.col();
                //Should be if constexpr, but libigl does not work with C++17
                if(triType==TriangularType::Upper) {
                    if(col>=row) {
                        if(!isKnown[row] && isKnown[col]) {
                            tripletListY.emplace_back
                            (fu_reverse(row), f_reverse(col), it.value());
                        }
                    } else {
                        if(!isKnown[col] && isKnown[row]) {
                            tripletListY.emplace_back
                            (fu_reverse(col), f_reverse(row), it.value());
                        }
                    }
                } else if(triType==TriangularType::Lower) {
                    if(row>=col) {
                        if(!isKnown[row] && isKnown[col]) {
                            tripletListY.emplace_back
                            (fu_reverse(row), f_reverse(col), it.value());
                        }
                    } else {
                        if(!isKnown[col] && isKnown[row]) {
                            tripletListY.emplace_back
                            (fu_reverse(col), f_reverse(row), it.value());
                        }
                    }
                } else {
                    if(!isKnown[row] && isKnown[col]) {
                        tripletListY.emplace_back
                        (fu_reverse(row), f_reverse(col), it.value());
                    }
                }
            }
        }
        
        //Appending A(:, known)
        const Int mu = precomputed.fu.size();
        for(Int k=0; k<A.outerSize(); ++k) {
            for(typename SparseMat::InnerIterator it(A,k); it; ++it) {
                const Int row=it.row(), col=it.col();
                if(isKnown[col]) {
                    tripletListY.emplace_back
                    (mu+row, f_reverse(col), it.value());
                }
            }
        }
        
        precomputed.Y.resize(mu+l, m);
        precomputed.Y.setFromTriplets(tripletListY.begin(), tripletListY.end());
    }
    
    //Call on the solver to precompute
    if(l>0) {
        QA.makeCompressed();
        precomputed.solver.compute(QA);
    } else if(m>0) {
        QU.makeCompressed();
        precomputed.solver.compute(QU);
    } else {
        if(Q.isCompressed()) {
            precomputed.solver.compute(Q);
        } else {
            SparseMat Qc = Q.template cast<Scalar>();
            Qc.makeCompressed();
            precomputed.solver.compute(Qc);
        }
    }
    if(precomputed.solver.info() != Eigen::Success) {
        return false;
    }
    
    return true;
}


template <typename Solver, typename ScalarPrec, typename IntPrec,
typename Derivedc, typename Derivedg, typename Derivedb, typename Derivedz>
bool
parametrization::constrained_qp_solve(
                                      const ConstrainedQPPrecomputed<ScalarPrec, IntPrec, Solver>& precomputed,
                                      const Eigen::MatrixBase<Derivedc>& c,
                                      const Eigen::MatrixBase<Derivedg>& g,
                                      const Eigen::MatrixBase<Derivedb>& b,
                                      Eigen::PlainObjectBase<Derivedz>& z)
{
    using Int = IntPrec;
    parametrization_assert(c.rows() < std::numeric_limits<Int>::max() &&
           precomputed.Y.rows() < std::numeric_limits<Int>::max() &&
           "Q is too large for the chosen integer type.");
    using VecI = Eigen::Matrix<Int, Eigen::Dynamic, 1>;
    
    //If the solver only uses the lower/upper triangular part of a symmetric
    // matrix,specify here, this will save time later.
    enum class TriangularType {Both, Lower, Upper};
    constexpr TriangularType triType = TriangularType::Both;
    
    //Every other scalar will be cast to this when precomputing.
    using Scalar = ScalarPrec;
    using MatX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    
    //Reject completely empty problems.
    parametrization_assert((c.size()>0 || g.size()>0 || b.size()>0) &&
           "Neither c, nor g, nor b provided.");
    
    //Which parts of the problem are active, and with which dimension?
    const Int n = precomputed.n;
    const Int m = precomputed.m;
    const Int l = precomputed.l;
    
    //Make sure the input corresponds with what is known from precomputed.
    parametrization_assert(precomputed.Y.cols()==m &&
           "Number of knowns does not match with precomputed.");
    parametrization_assert((c.rows()==0 || c.rows()==n) &&
                           "c has the wrong number of rows.");
    parametrization_assert(g.rows()==m && "g has the wrong number of rows.");
    parametrization_assert(b.rows()==l && "b has the wrong number of rows.");
    
    //Determine dimension.
    const Int dim = c.cols()<2 ? (g.cols()<2 ? (b.cols()<2 ? 1 : b.cols())
                                   : g.cols()) : c.cols();
    parametrization_assert(dim>0 &&
                           "At least one of c, g, b must have one column.");
    parametrization_assert((c.cols()<2 || c.cols()==dim) &&
                           "c has the wrong number of cols.");
    parametrization_assert((g.cols()<2 || g.cols()==dim) &&
                           "g has the wrong number of cols.");
    parametrization_assert((b.cols()<2 || b.cols()==dim) &&
                           "b has the wrong number of cols.");
    //Do c,g,b have more than one column (or do we need to stretch it to dim)?
    const bool fullc = c.cols()>1;
    const bool fullg = g.cols()>1;
    const bool fullb = b.cols()>1;
    
    //Deal with homogeneous g special case
    const bool homIng = (g.array()==0).all();
    
    
    //If there are no constraints at all, just invert and be done with it.
    if(m==0 && l==0) {
        z = -precomputed.solver.solve(c);
        if(precomputed.solver.info() != Eigen::Success) {
            return false;
        }
        return true;
    }
    
    //Construct right-hand side (if needed): cu = c(unknown)
    MatX ca;
    Int mu;
    if(m>0) {
        mu = precomputed.fu.size();
        ca.resize(mu,dim);
        if(c.size()>0) {
#pragma omp parallel for
            for(Int i=0; i<mu; ++i) {
                const Int pi = precomputed.fu(i);
//                for(Int j=0; j<dim; ++j) {
//                    ca(i,j) = (fullc ? -c(pi,j) : -c(pi));
//                }
                if(fullc) {
                    ca.row(i) = -c.row(pi);
                } else {
                    ca.row(i).array() = -c(pi);
                }
            }
        } else {
            ca.setZero();
        }
    } else {
        if(c.size()==0) {
            ca.resize(n,dim);
            ca.setZero();
        }
    }
    
    //Append b to right-hand side (if needed): ca = [c(unknown); b]
    if(l>0) {
        const Int nloc = m>0 ? mu : n;
        if(m>0 || c.size()==0) {
            //ca has already been initialized, enlarge it.
            ca.conservativeResize(nloc+l,dim);
        } else{
            //ca has not yet been initialized, do so and load c into it.
            ca.resize(n+l,dim);
#pragma omp parallel for
            for(Int i=0; i<n; ++i) {
//                for(Int j=0; j<dim; ++j) {
//                    ca(i,j) = (fullc ? -c(i,j) : -c(i));
//                }
                if(fullc) {
                    ca.row(i) = -c.row(i);
                } else {
                    ca.row(i).array() = -c(i);
                }
            }
        }
        
        //Append b into now enlarged ca.
        for(Int i=0; i<l; ++i) {
//            for(Int j=0; j<dim; ++j) {
//                ca(nloc+i,j) = (fullb ? b(i,j) : b(i));
//            }
            if(fullb) {
                ca.row(nloc+i) = b.row(i);
            } else {
                ca.row(nloc+i).array() = b(i);
            }
        }
    }
    //At this point, ca has the right dimensions.
    
    //Need to account for the knowns in the right-hand side now
    if(m>0 && !homIng) {
        if(fullg) {
            ca -= precomputed.Y * g;
        } else {
            const MatX Yg = precomputed.Y * g;
            for(Int i=0; i<dim; ++i) {
                ca.col(i) -= Yg;
            }
        }
    }
    
    //Prepare result
    if(z.rows()!=n || z.cols()!=dim) {
        z.resize(n,dim);
    }
    
    //Fill in knowns
    if(m>0) {
        for(Int i=0; i<m; ++i) {
            const Int pi = precomputed.f(i);
//            for(Int j=0; j<dim; ++j) {
//                z(pi,j) = (homIng ? 0. : (fullg ? g(i,j) : g(i)));
//            }
            if(homIng) {
                z.row(pi).setZero();
            } else {
                if(fullg) {
                    z.row(pi) = g.row(i);
                } else {
                    z.row(pi).array() = g(i);
                }
            }
        }
    }
    
    //Solve the system
    MatX za = precomputed.solver.solve(ca);
    if(precomputed.solver.info() != Eigen::Success) {
        return false;
    }
    
    //Copy the right entries from za to z
    if(m>0) {
        for(Int i=0; i<mu; ++i) {
            const Int pi = precomputed.fu(i);
//            for(Int j=0; j<dim; ++j) {
//                z(pi,j) = za(i,j);
//            }
            z.row(pi) = za.row(i);
        }
    } else {
//        for(Int i=0; i<n; ++i) {
//            for(Int j=0; j<dim; ++j) {
//                z(i,j) = za(i,j);
//            }
//        }
        z = za.topRows(n);
    }
    
    return true;
}


// Explicit template instantiation
template bool parametrization::constrained_qp<Eigen::SimplicialLLT<Eigen::SparseMatrix<double, 0, int>, 1, Eigen::AMDOrdering<int> >, double, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, double, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(Eigen::SparseMatrix<double, 0, int> const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::SparseMatrix<double, 0, int> const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template bool parametrization::constrained_qp<Eigen::SimplicialLLT<Eigen::SparseMatrix<double, 0, int>, 1, Eigen::AMDOrdering<int> >, double, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 2, 0, -1, 2>, double, Eigen::Matrix<double, -1, 2, 0, -1, 2>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(Eigen::SparseMatrix<double, 0, int> const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 2, 0, -1, 2> > const&, Eigen::SparseMatrix<double, 0, int> const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 2, 0, -1, 2> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template bool parametrization::constrained_qp<Eigen::SimplicialLDLT<Eigen::SparseMatrix<double, 0, int>, 1, Eigen::AMDOrdering<int> >, double, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, double, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(Eigen::SparseMatrix<double, 0, int> const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::SparseMatrix<double, 0, int> const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template bool parametrization::constrained_qp<Eigen::SparseLU<Eigen::SparseMatrix<double, 0, int>, Eigen::COLAMDOrdering<int> >, double, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, double, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(Eigen::SparseMatrix<double, 0, int> const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::SparseMatrix<double, 0, int> const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template bool parametrization::constrained_qp<Eigen::SimplicialLLT<Eigen::SparseMatrix<double, 0, int>, 1, Eigen::AMDOrdering<int> >, double, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, double, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(Eigen::SparseMatrix<double, 0, int> const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::SparseMatrix<double, 0, int> const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template bool parametrization::constrained_qp<Eigen::SparseLU<Eigen::SparseMatrix<double, 0, int>, Eigen::COLAMDOrdering<int> >, double, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 2, 0, -1, 2>, double, Eigen::Matrix<double, -1, 2, 0, -1, 2>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(Eigen::SparseMatrix<double, 0, int> const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 2, 0, -1, 2> > const&, Eigen::SparseMatrix<double, 0, int> const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 2, 0, -1, 2> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template bool parametrization::constrained_qp<Eigen::SparseLU<Eigen::SparseMatrix<double, 0, int>, Eigen::COLAMDOrdering<int> >, double, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, double, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(Eigen::SparseMatrix<double, 0, int> const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::SparseMatrix<double, 0, int> const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template bool parametrization::constrained_qp<Eigen::ConjugateGradient<Eigen::SparseMatrix<double, 0, int>, 3, Eigen::DiagonalPreconditioner<double> >, double, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 2, 0, -1, 2>, double, Eigen::Matrix<double, -1, 2, 0, -1, 2>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(Eigen::SparseMatrix<double, 0, int> const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 2, 0, -1, 2> > const&, Eigen::SparseMatrix<double, 0, int> const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 2, 0, -1, 2> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template bool parametrization::constrained_qp<Eigen::ConjugateGradient<Eigen::SparseMatrix<double, 0, int>, 3, Eigen::DiagonalPreconditioner<double> >, double, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, double, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(Eigen::SparseMatrix<double, 0, int> const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::SparseMatrix<double, 0, int> const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template bool parametrization::constrained_qp_solve<Eigen::SimplicialLLT<Eigen::SparseMatrix<double, 0, int>, 1, Eigen::AMDOrdering<int> >, double, int, Eigen::Matrix<double, -1, 2, 0, -1, 2>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(parametrization::ConstrainedQPPrecomputed<double, int, Eigen::SimplicialLLT<Eigen::SparseMatrix<double, 0, int>, 1, Eigen::AMDOrdering<int> > > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 2, 0, -1, 2> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template bool parametrization::constrained_qp_solve<Eigen::SimplicialLLT<Eigen::SparseMatrix<double, 0, int>, 1, Eigen::AMDOrdering<int> >, double, int, Eigen::Matrix<double, -1, 3, 0, -1, 3>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(parametrization::ConstrainedQPPrecomputed<double, int, Eigen::SimplicialLLT<Eigen::SparseMatrix<double, 0, int>, 1, Eigen::AMDOrdering<int> > > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 3, 0, -1, 3> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template bool parametrization::constrained_qp_solve<Eigen::SparseLU<Eigen::SparseMatrix<double, 0, int>, Eigen::COLAMDOrdering<int> >, double, int, Eigen::Matrix<double, -1, 2, 0, -1, 2>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(parametrization::ConstrainedQPPrecomputed<double, int, Eigen::SparseLU<Eigen::SparseMatrix<double, 0, int>, Eigen::COLAMDOrdering<int> > > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 2, 0, -1, 2> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template bool parametrization::constrained_qp_solve<Eigen::SparseLU<Eigen::SparseMatrix<double, 0, int>, Eigen::COLAMDOrdering<int> >, double, int, Eigen::Matrix<double, -1, 3, 0, -1, 3>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(parametrization::ConstrainedQPPrecomputed<double, int, Eigen::SparseLU<Eigen::SparseMatrix<double, 0, int>, Eigen::COLAMDOrdering<int> > > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 3, 0, -1, 3> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
#ifdef SUITESPARSE_AVAILABLE
template bool parametrization::constrained_qp<Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double, 0, int>, 1>, double, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, double, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(Eigen::SparseMatrix<double, 0, int> const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::SparseMatrix<double, 0, int> const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template bool parametrization::constrained_qp<Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double, 0, int>, 1>, double, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, double, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(Eigen::SparseMatrix<double, 0, int> const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::SparseMatrix<double, 0, int> const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template bool parametrization::constrained_qp<Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double, 0, int>, 1>, double, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 2, 0, -1, 2>, double, Eigen::Matrix<double, -1, 2, 0, -1, 2>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(Eigen::SparseMatrix<double, 0, int> const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 2, 0, -1, 2> > const&, Eigen::SparseMatrix<double, 0, int> const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 2, 0, -1, 2> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template bool parametrization::constrained_qp<Eigen::UmfPackLU<Eigen::SparseMatrix<double, 0, int> >, double, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, double, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(Eigen::SparseMatrix<double, 0, int> const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::SparseMatrix<double, 0, int> const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template bool parametrization::constrained_qp<Eigen::UmfPackLU<Eigen::SparseMatrix<double, 0, int> >, double, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 2, 0, -1, 2>, double, Eigen::Matrix<double, -1, 2, 0, -1, 2>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(Eigen::SparseMatrix<double, 0, int> const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 2, 0, -1, 2> > const&, Eigen::SparseMatrix<double, 0, int> const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 2, 0, -1, 2> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template bool parametrization::constrained_qp_solve<Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double, 0, int>, 1>, double, int, Eigen::Matrix<double, -1, 2, 0, -1, 2>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(parametrization::ConstrainedQPPrecomputed<double, int, Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double, 0, int>, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 2, 0, -1, 2> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template bool parametrization::constrained_qp_solve<Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double, 0, int>, 1>, double, int, Eigen::Matrix<double, -1, 3, 0, -1, 3>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(parametrization::ConstrainedQPPrecomputed<double, int, Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double, 0, int>, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 3, 0, -1, 3> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template bool parametrization::constrained_qp_solve<Eigen::UmfPackLU<Eigen::SparseMatrix<double, 0, int> >, double, int, Eigen::Matrix<double, -1, 2, 0, -1, 2>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(parametrization::ConstrainedQPPrecomputed<double, int, Eigen::UmfPackLU<Eigen::SparseMatrix<double, 0, int> > > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 2, 0, -1, 2> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template bool parametrization::constrained_qp_solve<Eigen::UmfPackLU<Eigen::SparseMatrix<double, 0, int> >, double, int, Eigen::Matrix<double, -1, 3, 0, -1, 3>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(parametrization::ConstrainedQPPrecomputed<double, int, Eigen::UmfPackLU<Eigen::SparseMatrix<double, 0, int> > > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 3, 0, -1, 3> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
#endif
