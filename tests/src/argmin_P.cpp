
// Tests for the function in parametrization/lagrangian.h
#include <catch2/catch.hpp>

#include <parametrization/lagrangian.h>
#include <parametrization/argmin_P.h>
#include <parametrization/sym_from_triu.h>


TEST_CASE("argmin_P 2D")
{
    //n randomly generated examples for which we check the argmin
    const int n=50, dim=300;
    
    srand(0);
    for(int i=0; i<n; ++i) {
        Eigen::VectorXd U = Eigen::VectorXd::Random(2*dim);
        for(int i=0; i<dim; ++i) {
            U.segment(2*i,2).normalize();
        }
        
        //        Eigen::VectorXd P = Eigen::VectorXd::Random(3*dim);
        //        for(int i=0; i<dim; ++i) {
        //            P(3*i) += 1.;
        //            P(3*i+2) += 1.;
        //        }
        
        Eigen::MatrixXd GW = Eigen::MatrixXd::Random(2*dim, 2);
        
        Eigen::MatrixXd Lambda = Eigen::MatrixXd::Random(2*dim, 2);
        
        Eigen::VectorXd mu = Eigen::VectorXd::Random(dim).array() + 1.;
        if(i<20) {
            //Check for extreme values
            mu *= 1e-6;
        }
        
        Eigen::VectorXd w = Eigen::VectorXd::Random(dim).array() + 1.;
        if(i>=10 && i<30) {
            //Check for extreme values
            w *= 1e-6;
        }
        
        for(int i=0; i<mu.size(); ++i) {
            if(mu(i) <= sqrt(std::numeric_limits<double>::epsilon())) {
                mu(i) = 1.01*sqrt(std::numeric_limits<double>::epsilon());
            }
            if(w(i) <= sqrt(std::numeric_limits<double>::epsilon())) {
                w(i) = 1.01*sqrt(std::numeric_limits<double>::epsilon());
            }
        }
        
        //Symmetric gradient
        {
            //Compute argmin_P
            Eigen::VectorXd P;
            parametrization::argmin_P
            <false,parametrization::EnergyType::SymmetricGradient,2>
            (U, GW, Lambda, w, mu, P);
            REQUIRE(P.array().isFinite().all());
            
            //Do the parallel and serial versions agree?
#ifdef OPENMP_AVAILABLE
            Eigen::VectorXd Ppar;
            parametrization::argmin_P
            <true,parametrization::EnergyType::SymmetricGradient,2>
            (U, GW, Lambda, w, mu, Ppar);
            REQUIRE(P == Ppar);
            //Reset Eigen parallelization, the function might have changed it.
            Eigen::setNbThreads(0);
#endif
            
            //Compute a few perturbations and check whether their energy is larger.
            const double E_min = parametrization::lagrangian
            <false,parametrization::EnergyType::SymmetricGradient>
            (U,P,GW,Lambda,w,mu);
            //Can produce invalid examples due to numerical errors if a P with
            // low determinant is present leading to high energy. Can not test on these.
            if(E_min > 1e8) {
                --i;
                continue;
            }

            for(int j=0; j<dim; ++j) {
                Eigen::VectorXd Ppert = P +
                0.01*Eigen::VectorXd::Random(3*dim);
                for(int k=0; k<dim; ++k) {
                    const Eigen::Matrix2d Pl =
                    parametrization::sym_2d_from_index(Ppert, k, dim);
                    if((Pl.eigenvalues().real().array()<=0).any()) {
                        parametrization::index_into_sym_2x2
                        (parametrization::sym_2d_from_index
                         (P, k, dim), Ppert, k, dim);
                    }
                }
                const double E_pert =
                parametrization::lagrangian
                <false,parametrization::EnergyType::SymmetricGradient>
                (U,Ppert,GW,Lambda,w,mu);
                REQUIRE(E_min <= E_pert);
            }
        }
        
        //Symmetric Dirichlet
        {
            //Compute argmin_P
            Eigen::VectorXd P;
            parametrization::argmin_P
            <false,parametrization::EnergyType::SymmetricDirichlet,2>
            (U, GW, Lambda, w, mu, P);
            REQUIRE(P.array().isFinite().all());
            
            //Do the parallel and serial versions agree?
#ifdef OPENMP_AVAILABLE
            Eigen::VectorXd Ppar;
            parametrization::argmin_P
            <true,parametrization::EnergyType::SymmetricDirichlet,2>
            (U, GW, Lambda, w, mu, Ppar);
            REQUIRE(P == Ppar);
            //Reset Eigen parallelization, the function might have changed it.
            Eigen::setNbThreads(0);
#endif
            
            //Compute a few perturbations and check whether their energy is larger.
            const double E_min = parametrization::lagrangian
            <false,parametrization::EnergyType::SymmetricDirichlet>
            (U,P,GW,Lambda,w,mu);
            //Can produce invalid examples due to numerical errors if a P with
            // low determinant is present leading to high energy. Can not test on these.
            if(E_min > 1e8) {
                --i;
                continue;
            }

            for(int j=0; j<dim; ++j) {
                Eigen::VectorXd Ppert = P +
                0.01*Eigen::VectorXd::Random(3*dim);
                for(int k=0; k<dim; ++k) {
                    const Eigen::Matrix2d Pl =
                    parametrization::sym_2d_from_index(Ppert, k, dim);
                    if((Pl.eigenvalues().real().array()<=0).any()) {
                        parametrization::index_into_sym_2x2
                        (parametrization::sym_2d_from_index
                         (P, k, dim), Ppert, k, dim);
                    }
                }
                const double E_pert =
                parametrization::lagrangian
                <false,parametrization::EnergyType::SymmetricDirichlet>
                (U,Ppert,GW,Lambda,w,mu);
                REQUIRE(E_min <= E_pert);
            }
        }
    }
}


TEST_CASE("argmin_P 3D")
{
    //n randomly generated examples for which we check the argmin
    const int n=50, dim=300;
    
    srand(0);
    for(int i=0; i<n; ++i) {
        Eigen::VectorXd U = Eigen::VectorXd::Random(4*dim);
        for(int i=0; i<dim; ++i) {
            U.segment(4*i,4).normalize();
        }
        
        //        Eigen::VectorXd P = Eigen::VectorXd::Random(6*dim);
        //        for(int i=0; i<dim; ++i) {
        //            P(6*i) += 1.;
        //            P(6*i+3) += 1.;
        //            P(6*i+5) += 1.;
        //        }
        
        Eigen::MatrixXd GW = Eigen::MatrixXd::Random(3*dim, 3);
        
        Eigen::MatrixXd Lambda = Eigen::MatrixXd::Random(3*dim, 3);
        
        Eigen::VectorXd mu = Eigen::VectorXd::Random(dim).array() + 1.;
        if(i<20) {
            //Check for extreme values
            mu *= 1e-6;
        }
        
        Eigen::VectorXd w = Eigen::VectorXd::Random(dim).array() + 1.;
        if(i>=10 && i<30) {
            //Check for extreme values
            w *= 1e-6;
        }
        
        for(int i=0; i<mu.size(); ++i) {
            if(mu(i) <= sqrt(std::numeric_limits<double>::epsilon())) {
                mu(i) = 1.01*sqrt(std::numeric_limits<double>::epsilon());
            }
            if(w(i) <= sqrt(std::numeric_limits<double>::epsilon())) {
                w(i) = 1.01*sqrt(std::numeric_limits<double>::epsilon());
            }
        }
        
        //Symmetric gradient
        {
            //Compute argmin_P
            Eigen::VectorXd P;
            parametrization::argmin_P
            <false,parametrization::EnergyType::SymmetricGradient,3>
            (U, GW, Lambda, w, mu, P);
            REQUIRE(P.array().isFinite().all());
            
            //Do the parallel and serial versions agree?
#ifdef OPENMP_AVAILABLE
            Eigen::VectorXd Ppar;
            parametrization::argmin_P
            <true,parametrization::EnergyType::SymmetricGradient,3>
            (U, GW, Lambda, w, mu, Ppar);
            REQUIRE(P == Ppar);
            //Reset Eigen parallelization, the function might have changed it.
            Eigen::setNbThreads(0);
#endif
            
            //Compute a few perturbations and check whether their energy is larger.
            const double E_min = parametrization::lagrangian
            <false,parametrization::EnergyType::SymmetricGradient>
            (U,P,GW,Lambda,w,mu);
            //Can produce invalid examples due to numerical errors if a P with
            // low determinant is present leading to high energy. Can not test on these.
            if(E_min > 1e8) {
                --i;
                continue;
            }

            for(int j=0; j<dim; ++j) {
                Eigen::VectorXd Ppert = P +
                0.01*Eigen::VectorXd::Random(6*dim);
                for(int k=0; k<dim; ++k) {
                    const Eigen::Matrix3d Pl =
                    parametrization::sym_3d_from_index(Ppert, k, dim);
                    if((Pl.eigenvalues().real().array()<=0).any()) {
                        parametrization::index_into_sym_3x3
                        (parametrization::sym_3d_from_index
                         (P, k, dim), Ppert, k, dim);
                    }
                }
                const double E_pert =
                parametrization::lagrangian
                <false,parametrization::EnergyType::SymmetricGradient>
                (U,Ppert,GW,Lambda,w,mu);
                REQUIRE(E_min <= E_pert);
            }
        }
        
        //Symmetric Dirichlet
        {
            //Compute argmin_P
            Eigen::VectorXd P;
            parametrization::argmin_P
            <false,parametrization::EnergyType::SymmetricDirichlet,3>
            (U, GW, Lambda, w, mu, P);
            REQUIRE(P.array().isFinite().all());
            
            //Do the parallel and serial versions agree?
#ifdef OPENMP_AVAILABLE
            Eigen::VectorXd Ppar;
            parametrization::argmin_P
            <true,parametrization::EnergyType::SymmetricDirichlet,3>
            (U, GW, Lambda, w, mu, Ppar);
            REQUIRE(P == Ppar);
            //Reset Eigen parallelization, the function might have changed it.
            Eigen::setNbThreads(0);
#endif
            
            //Compute a few perturbations and check whether their energy is larger.
            const double E_min = parametrization::lagrangian
            <false,parametrization::EnergyType::SymmetricDirichlet>
            (U,P,GW,Lambda,w,mu);
            //Can produce invalid examples due to numerical errors if a P with
            // low determinant is present leading to high energy. Can not test on these.
            if(E_min > 1e8) {
                --i;
                continue;
            }

            for(int j=0; j<dim; ++j) {
                Eigen::VectorXd Ppert = P +
                0.01*Eigen::VectorXd::Random(6*dim);
                for(int k=0; k<dim; ++k) {
                    const Eigen::Matrix3d Pl =
                    parametrization::sym_3d_from_index(Ppert, k, dim);
                    if((Pl.eigenvalues().real().array()<=0).any()) {
                        parametrization::index_into_sym_3x3
                        (parametrization::sym_3d_from_index
                         (P, k, dim), Ppert, k, dim);
                    }
                }
                const double E_pert =
                parametrization::lagrangian
                <false,parametrization::EnergyType::SymmetricDirichlet>
                (U,Ppert,GW,Lambda,w,mu);
                REQUIRE(E_min <= E_pert);
            }
        }
    }
}

