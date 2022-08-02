
// Tests for the function in parametrization/lagrangian.h
#include <catch2/catch.hpp>

#include <parametrization/constrained_qp.h>
#include <parametrization/lagrangian.h>
#include <parametrization/argmin_W.h>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>


TEST_CASE("argmin_W 2D")
{
    //n randomly generated examples for which we check the argmin
    //const int n=20, meshdim=20, dim=30;
    const int n=20, meshdim=100, dim=300;
    
    srand(0);
    for(int i=0; i<n; ++i) {
        Eigen::VectorXd U = Eigen::VectorXd::Random(2*dim);
        for(int i=0; i<dim; ++i) {
            U.segment(2*i,2).normalize();
        }
        
        Eigen::VectorXd P = Eigen::VectorXd::Random(3*dim);
        for(int i=0; i<dim; ++i) {
            P(3*i) += 1.;
            P(3*i+2) += 1.;
        }
        
        Eigen::MatrixXd Gfull = Eigen::MatrixXd::Random(2*dim, meshdim);
        Gfull.row(0).setZero();
        Eigen::SparseMatrix<double> G = Gfull.sparseView();
        
        Eigen::MatrixXd Lambda = Eigen::MatrixXd::Random(2*dim, 2);
        
        Eigen::VectorXd mu = Eigen::VectorXd::Random(dim).array() + 1.;
        Eigen::DiagonalMatrix<double, Eigen::Dynamic> M(2*dim);
        M.diagonal() << mu, mu;
        Eigen::SparseMatrix<double> GtM = G.transpose()*M;
        
        Eigen::VectorXd w = Eigen::VectorXd::Random(dim).array() + 1.;
        
        parametrization::ConstrainedQPPrecomputed<double, int,
        Eigen::SparseLU<Eigen::SparseMatrix<double> > > GtMG;
        Eigen::VectorXi fixed(3);
        fixed << 2%meshdim, 5%meshdim, 14%meshdim;
        Eigen::SparseMatrix<double> Aeq;
        if(i < n/2) {
            Aeq.resize(2,meshdim);
            Aeq.coeffRef(0,5%meshdim) = rand();
            Aeq.coeffRef(0,12%meshdim) = rand();
            Aeq.coeffRef(1,3%meshdim) = rand();
            Aeq.coeffRef(1,9%meshdim) = rand();
            Aeq.coeffRef(1,11%meshdim) = rand();
        }
        parametrization::constrained_qp_precompute((GtM*G).eval(), fixed, Aeq,
                                                   GtMG);
        
        //Compute argmin_W
        Eigen::MatrixXd fixedTo = Eigen::MatrixXd::Random(3,2);
        Eigen::MatrixXd beq;
        if(i < n/2) {
            beq.setRandom(2,2);
        }
        Eigen::MatrixXd W;
        REQUIRE(parametrization::argmin_W<false,2>
                (GtMG, GtM, fixedTo, beq, U, P, Lambda, W));
        REQUIRE(W.array().isFinite().all());
        
        //Do the parallel and serial versions agree?
#ifdef OPENMP_AVAILABLE
        Eigen::MatrixXd Wpar;
        REQUIRE(parametrization::argmin_W<true,2>
                (GtMG, GtM, fixedTo, beq, U, P, Lambda, Wpar));
        REQUIRE(W == Wpar);
        //Reset Eigen parallelization, the function might have changed it.
        Eigen::setNbThreads(0);
#endif
    }
}


TEST_CASE("argmin_W 3D")
{
    //n randomly generated examples for which we check the argmin
    //const int n=20, meshdim=20, dim=30;
    const int n=20, meshdim=100, dim=300;
    
    srand(0);
    for(int i=0; i<n; ++i) {
        Eigen::VectorXd U = Eigen::VectorXd::Random(4*dim);
        for(int i=0; i<dim; ++i) {
            U.segment(4*i,4).normalize();
        }

        Eigen::VectorXd P = Eigen::VectorXd::Random(6*dim);
        for(int i=0; i<dim; ++i) {
            P(6*i) += 1.;
            P(6*i+3) += 1.;
            P(6*i+5) += 1.;
        }
        
        Eigen::MatrixXd Gfull = Eigen::MatrixXd::Random(3*dim, meshdim);
        Gfull.row(0).setZero();
        Eigen::SparseMatrix<double> G = Gfull.sparseView();

        Eigen::MatrixXd Lambda = Eigen::MatrixXd::Random(3*dim, 3);

        Eigen::VectorXd mu = Eigen::VectorXd::Random(dim).array() + 1.;
        Eigen::DiagonalMatrix<double, Eigen::Dynamic> M(3*dim);
        M.diagonal() << mu, mu, mu;
        Eigen::SparseMatrix<double> GtM = G.transpose()*M;

        Eigen::VectorXd w = Eigen::VectorXd::Random(dim).array() + 1.;
        
        parametrization::ConstrainedQPPrecomputed<double, int,
        Eigen::SparseLU<Eigen::SparseMatrix<double> > > GtMG;
        Eigen::VectorXi fixed(3);
        fixed << 2%meshdim, 5%meshdim, 14%meshdim;
        Eigen::SparseMatrix<double> Aeq;
        if(i < n/2) {
            Aeq.resize(2,meshdim);
            Aeq.coeffRef(0,5%meshdim) = rand();
            Aeq.coeffRef(0,12%meshdim) = rand();
            Aeq.coeffRef(1,3%meshdim) = rand();
            Aeq.coeffRef(1,9%meshdim) = rand();
            Aeq.coeffRef(1,11%meshdim) = rand();
        }
        parametrization::constrained_qp_precompute((GtM*G).eval(), fixed, Aeq,
                                                   GtMG);
        
        //Compute argmin_W
        Eigen::MatrixXd fixedTo = Eigen::MatrixXd::Random(3,3);
        Eigen::MatrixXd beq;
        if(i < n/2) {
            beq.setRandom(2,3);
        }
        Eigen::MatrixXd W;
        REQUIRE(parametrization::argmin_W<false,3>
                (GtMG, GtM, fixedTo, beq, U, P, Lambda, W));
        REQUIRE(W.array().isFinite().all());
        
        //Do the parallel and serial versions agree?
#ifdef OPENMP_AVAILABLE
        Eigen::MatrixXd Wpar;
        REQUIRE(parametrization::argmin_W<true,3>
                (GtMG, GtM, fixedTo, beq, U, P, Lambda, Wpar));
        REQUIRE(W == Wpar);
        //Reset Eigen parallelization, the function might have changed it.
        Eigen::setNbThreads(0);
#endif
    }
}
