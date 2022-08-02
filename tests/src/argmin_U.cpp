
// Tests for the function in parametrization/lagrangian.h
#include <catch2/catch.hpp>

#include <parametrization/lagrangian.h>
#include <parametrization/argmin_U.h>
#include <parametrization/rotmat_from_complex_quat.h>


TEST_CASE("argmin_U 2D")
{
    //n randomly generated examples for which we check the argmin
    //const int n=20, dim=30;
    const int n=20, dim=300;
    
    srand(0);
    for(int i=0; i<n; ++i) {
//        Eigen::VectorXd U = Eigen::VectorXd::Random(2*dim);
//        for(int i=0; i<dim; ++i) {
//            U.segment(2*i,2).normalize();
//        }
        
        Eigen::VectorXd P = Eigen::VectorXd::Random(3*dim);
        for(int i=0; i<dim; ++i) {
            P(3*i) += 1.;
            P(3*i+2) += 1.;
        }
        
        Eigen::MatrixXd GW = Eigen::MatrixXd::Random(2*dim, 2);
        
        Eigen::MatrixXd Lambda = Eigen::MatrixXd::Random(2*dim, 2);
        
        Eigen::VectorXd mu = Eigen::VectorXd::Random(dim).array() + 1.;
        
        Eigen::VectorXd w = Eigen::VectorXd::Random(dim).array() + 1.;
        
        //Compute argmin_U
        Eigen::VectorXd U;
        parametrization::argmin_U<false,2>(P, GW, Lambda, U);
        REQUIRE(U.array().isFinite().all());
        
        //Do the parallel and serial versions agree?
#ifdef OPENMP_AVAILABLE
        Eigen::VectorXd Upar;
        parametrization::argmin_U<true,2>(P, GW, Lambda, Upar);
        REQUIRE(U == Upar);
        //Reset Eigen parallelization, the function might have changed it.
        Eigen::setNbThreads(0);
#endif
        
        //Compute a few perturbations and check whether their energy is larger.
        double E_min = parametrization::lagrangian(U,P,GW,Lambda,w,mu);
        for(int j=0; j<100*dim; ++j) {
            Eigen::VectorXd Upert;
            if(j<50*dim) {
                Upert = Eigen::VectorXd::Random(2*dim);
            } else {
                Upert = U + 0.01*Eigen::VectorXd::Random(2*dim);
            }
            for(int i=0; i<dim; ++i) {
                Upert.segment(2*i,2).normalize();
            }
            const double E_pert =
            parametrization::lagrangian(Upert,P,GW,Lambda,w,mu);
            REQUIRE(E_min <= E_pert);
        }
        
        //Same thing, but now with the proximal version of the function.
        Eigen::VectorXd h = Eigen::VectorXd::Random(dim).array() + 1.;
        
        //Compute argmin_U.
        //Now we need to supply an old U for the proximal term.
        Eigen::VectorXd oldU = Eigen::VectorXd::Random(2*dim);
        for(int i=0; i<dim; ++i) {
            oldU.segment(2*i,2).normalize();
        }
        U = oldU;
        parametrization::argmin_U<false,2>(P, GW, Lambda, h, U);
        REQUIRE(U.array().isFinite().all());
        
        //Do the parallel and serial versions agree?
#ifdef OPENMP_AVAILABLE
        Upar = oldU;
        parametrization::argmin_U<true,2>(P, GW, Lambda, h, Upar);
        REQUIRE(U == Upar);
        //Reset Eigen parallelization, the function might have changed it.
        Eigen::setNbThreads(0);
#endif
        
        //Compute a few perturbations and check whether their energy is larger.
        E_min = parametrization::lagrangian(U,P,GW,Lambda,w,mu);
        for(int i=0; i<dim; ++i) {
            const Eigen::Matrix2d
            Uoldl = parametrization::rotmat_2d_from_index(oldU,i,dim),
            Ul = parametrization::rotmat_2d_from_index(U,i,dim);
            E_min += 0.5*mu(i)*h(i)*(Ul - Uoldl).squaredNorm();
        }
        for(int j=0; j<100*dim; ++j) {
            Eigen::VectorXd Upert;
            if(j<50*dim) {
                Upert = Eigen::VectorXd::Random(2*dim);
            } else {
                Upert = U + 0.01*Eigen::VectorXd::Random(2*dim);
            }
            for(int i=0; i<dim; ++i) {
                Upert.segment(2*i,2).normalize();
            }
            double E_pert =
            parametrization::lagrangian(Upert,P,GW,Lambda,w,mu);
            for(int i=0; i<dim; ++i) {
                const Eigen::Matrix2d
                Uoldl = parametrization::rotmat_2d_from_index(oldU,i,dim),
                Upertl = parametrization::rotmat_2d_from_index(Upert,i,dim);
                E_pert += 0.5*mu(i)*h(i)*(Upertl - Uoldl).squaredNorm();
            }
            REQUIRE(E_min <= E_pert);
        }
    }
}


TEST_CASE("argmin_U 3D")
{
    //n randomly generated examples for which we check the argmin
    //const int n=20, dim=30;
    const int n=20, dim=300;
    
    srand(0);
    for(int i=0; i<n; ++i) {
//        Eigen::VectorXd U = Eigen::VectorXd::Random(4*dim);
//        for(int i=0; i<dim; ++i) {
//            U.segment(4*i,4).normalize();
//        }
        
        Eigen::VectorXd P = Eigen::VectorXd::Random(6*dim);
        for(int i=0; i<dim; ++i) {
            P(6*i) += 1.;
            P(6*i+3) += 1.;
            P(6*i+5) += 1.;
        }
        
        Eigen::MatrixXd GW = Eigen::MatrixXd::Random(3*dim, 3);
        
        Eigen::MatrixXd Lambda = Eigen::MatrixXd::Random(3*dim, 3);
        
        Eigen::VectorXd mu = Eigen::VectorXd::Random(dim).array() + 1.;
        
        Eigen::VectorXd w = Eigen::VectorXd::Random(dim).array() + 1.;
        
        //Compute argmin_U
        Eigen::VectorXd U;
        parametrization::argmin_U<false,3>(P, GW, Lambda, U);
        REQUIRE(U.array().isFinite().all());
        
        //Do the parallel and serial versions agree?
#ifdef OPENMP_AVAILABLE
        Eigen::VectorXd Upar;
        parametrization::argmin_U<true,3>(P, GW, Lambda, Upar);
        REQUIRE(U == Upar);
        //Reset Eigen parallelization, the function might have changed it.
        Eigen::setNbThreads(0);
#endif
        
        //Compute a few perturbations and check whether their energy is larger.
        double E_min = parametrization::lagrangian(U,P,GW,Lambda,w,mu);
        for(int j=0; j<100*dim; ++j) {
            Eigen::VectorXd Upert;
            if(j<50*dim) {
                Upert = Eigen::VectorXd::Random(4*dim);
            } else {
                Upert = U + 0.01*Eigen::VectorXd::Random(4*dim);
            }
            for(int i=0; i<dim; ++i) {
                Upert.segment(4*i,4).normalize();
            }
            const double E_pert =
            parametrization::lagrangian(Upert,P,GW,Lambda,w,mu);
            REQUIRE(E_min <= E_pert);
        }
        
        //Same thing, but now with the proximal version of the function.
        Eigen::VectorXd h = Eigen::VectorXd::Random(dim).array() + 1.;
        
        //Compute argmin_U.
        //Now we need to supply an old U for the proximal term.
        Eigen::VectorXd oldU = Eigen::VectorXd::Random(4*dim);
        for(int i=0; i<dim; ++i) {
            oldU.segment(4*i,4).normalize();
        }
        U = oldU;
        parametrization::argmin_U<false,3>(P, GW, Lambda, h, U);
        REQUIRE(U.array().isFinite().all());
        
        //Do the parallel and serial versions agree?
#ifdef OPENMP_AVAILABLE
        Upar = oldU;
        parametrization::argmin_U<true,3>(P, GW, Lambda, h, Upar);
        REQUIRE(U == Upar);
        //Reset Eigen parallelization, the function might have changed it.
        Eigen::setNbThreads(0);
#endif
        
        //Compute a few perturbations and check whether their energy is larger.
        E_min = parametrization::lagrangian(U,P,GW,Lambda,w,mu);
        for(int i=0; i<dim; ++i) {
            const Eigen::Matrix3d
            Uoldl = parametrization::rotmat_3d_from_index(oldU,i,dim),
            Ul = parametrization::rotmat_3d_from_index(U,i,dim);
            E_min += 0.5*mu(i)*h(i)*(Ul - Uoldl).squaredNorm();
        }
        for(int j=0; j<100*dim; ++j) {
            Eigen::VectorXd Upert;
            if(j<50*dim) {
                Upert = Eigen::VectorXd::Random(4*dim);
            } else {
                Upert = U + 0.01*Eigen::VectorXd::Random(4*dim);
            }
            for(int i=0; i<dim; ++i) {
                Upert.segment(4*i,4).normalize();
            }
            double E_pert =
            parametrization::lagrangian(Upert,P,GW,Lambda,w,mu);
            for(int i=0; i<dim; ++i) {
                const Eigen::Matrix3d
                Uoldl = parametrization::rotmat_3d_from_index(oldU,i,dim),
                Upertl = parametrization::rotmat_3d_from_index(Upert,i,dim);
                E_pert += 0.5*mu(i)*h(i)*(Upertl - Uoldl).squaredNorm();
            }
            REQUIRE(E_min <= E_pert);
        }
    }
}

