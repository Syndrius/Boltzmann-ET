#ifndef ET_INTEGRATION_INIT_K_H
#define ET_INTEGRATION_INIT_K_H

#include <AMReX_REAL.H> 
#include <AMReX_Array4.H> 
#include <ET_Integration_Setup_K.H> 

AMREX_GPU_DEVICE 
inline 
void 
state_init(int i, int j, int k, 
        amrex::Array4<amrex::Real> const& state_fab, 
        amrex::Real time, const amrex::GeometryData& geom) noexcept 
{
        const auto domain_xlo = geom.ProbLo(); 
        
        amrex::Real x0 = (i + 0.5)*geom.CellSize(0) + domain_xlo[0]; 
        amrex::Real x1 = (j + 0.5)*geom.CellSize(1) + domain_xlo[1]; 
        amrex::Real x2 = (k + 0.5)*geom.CellSize(2) + domain_xlo[2]; 
        state_fab(i, j, k, Idx::bi1) = 0;

        state_fab(i, j, k, Idx::bi2) = 0;

        state_fab(i, j, k, Idx::bi3) = 0;

        state_fab(i, j, k, Idx::br1) = 0;

        state_fab(i, j, k, Idx::br2) = 0;

        state_fab(i, j, k, Idx::br3) = 0;

        state_fab(i, j, k, Idx::li1) = 0;

        state_fab(i, j, k, Idx::li2) = 0;

        state_fab(i, j, k, Idx::li3) = 0;

        state_fab(i, j, k, Idx::lr1) = 0;

        state_fab(i, j, k, Idx::lr2) = 0;

        state_fab(i, j, k, Idx::lr3) = 0;

        state_fab(i, j, k, Idx::vi1) = 0;

        state_fab(i, j, k, Idx::vi2) = 0;

        state_fab(i, j, k, Idx::vi3) = 0;

        state_fab(i, j, k, Idx::vr1) = 0;

        state_fab(i, j, k, Idx::vr2) = 0;

        state_fab(i, j, k, Idx::vr3) = 0;

        state_fab(i, j, k, Idx::wi1) = 0;

        state_fab(i, j, k, Idx::wi2) = 0;

        state_fab(i, j, k, Idx::wi3) = 0;

        state_fab(i, j, k, Idx::wr1) = 0;

        state_fab(i, j, k, Idx::wr2) = 0;

        state_fab(i, j, k, Idx::wr3) = 0;

        state_fab(i, j, k, Idx::Kn) = 0.10000000000000001;

        state_fab(i, j, k, Idx::T) = 0.10000000000000001;

        state_fab(i, j, k, Idx::c0) = -0.033333333333333333*std::pow(M_PI, -4.5)*(10.0*std::pow(M_PI, 4.5)*std::pow(x0, 5) - 10.0*std::pow(M_PI, 4.5)*std::pow(x0, 4)*std::sqrt(std::pow(x0, 2) - 1) - 10.0*std::pow(M_PI, 4.5)*std::pow(x0, 3) - 20.0*std::pow(M_PI, 3.5)*std::pow(x0, 3) + 20.0*std::pow(M_PI, 3.5)*x0)/std::pow(x0, 3);

        state_fab(i, j, k, Idx::c1) = -0.033333333333333333*std::pow(M_PI, -4.5)*(10.0*std::pow(M_PI, 4.0)*std::pow(x0, 5) - 10.0*std::pow(M_PI, 4.0)*std::pow(x0, 4)*std::sqrt(std::pow(x0, 2) - 1) - 10.0*std::pow(M_PI, 4.0)*std::pow(x0, 3) + 20.0*std::pow(M_PI, 4.0)*std::pow(x0, 2)*std::sqrt(std::pow(x0, 2) - 1) - 10.0*std::pow(M_PI, 4.0)*std::sqrt(std::pow(x0, 2) - 1))/std::pow(x0, 3);

        state_fab(i, j, k, Idx::c2) = 0;

        state_fab(i, j, k, Idx::c3) = -0.033333333333333333*std::pow(M_PI, -4.5)*(10.0*std::pow(M_PI, 4.5)*std::pow(x0, 3) - 20.0*std::pow(M_PI, 3.5)*x0)/std::pow(x0, 3);

        state_fab(i, j, k, Idx::c4) = -0.033333333333333333*std::pow(M_PI, -4.5)*(20.0*std::pow(M_PI, 4.5)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 4) - 20.0*std::pow(M_PI, 4.5)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 3)*std::sqrt(std::pow(x0, 2) - 1) - 10.0*std::pow(M_PI, 4.5)*state_fab(i,j,k,Idx::Kn)*x0*std::sqrt(std::pow(x0, 2) - 1) - 20.0*std::pow(M_PI, 4.0)*std::pow(x0, 2)*std::sqrt(std::pow(x0, 2) - 1) + 10.0*std::pow(M_PI, 4.0)*std::sqrt(std::pow(x0, 2) - 1))/std::pow(x0, 3);

        state_fab(i, j, k, Idx::c5) = -0.033333333333333333*std::pow(M_PI, -4.5)*(20.0*std::pow(M_PI, 4.5)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 4) - 20.0*std::pow(M_PI, 4.5)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 3)*std::sqrt(std::pow(x0, 2) - 1) - 10.0*std::pow(M_PI, 4.5)*state_fab(i,j,k,Idx::Kn)*x0*std::sqrt(std::pow(x0, 2) - 1))/std::pow(x0, 3);

        state_fab(i, j, k, Idx::c6) = -0.033333333333333333*std::pow(M_PI, -4.5)*(20.0*std::pow(M_PI, 4.5)*state_fab(i,j,k,Idx::T)*std::pow(x0, 5) - 20.0*std::pow(M_PI, 4.5)*state_fab(i,j,k,Idx::T)*std::pow(x0, 4)*std::sqrt(std::pow(x0, 2) - 1))/std::pow(x0, 3);

        state_fab(i, j, k, Idx::c7) = 0.66666666666666663*std::pow(M_PI, -0.5)*state_fab(i,j,k,Idx::T);

        state_fab(i, j, k, Idx::c8) = -0.033333333333333333*std::pow(M_PI, -4.5)*(-20.0*std::pow(M_PI, 4.5)*std::pow(x0, 5) + 20.0*std::pow(M_PI, 4.5)*std::pow(x0, 4)*std::sqrt(std::pow(x0, 2) - 1) + 10.0*std::pow(M_PI, 4.5)*std::pow(x0, 3) + 20.0*std::pow(M_PI, 3.5)*x0)/std::pow(x0, 3);

        state_fab(i, j, k, Idx::c9) = -0.33333333333333331*std::pow(M_PI, -0.5)*std::sqrt(std::pow(x0, 2) - 1)/std::pow(x0, 3);

        state_fab(i, j, k, Idx::c10) = -0.033333333333333333*std::pow(M_PI, -4.5)*(10.0*std::pow(M_PI, 4.5)*std::pow(x0, 5) - 10.0*std::pow(M_PI, 4.5)*std::pow(x0, 4)*std::sqrt(std::pow(x0, 2) - 1) - 10.0*std::pow(M_PI, 4.5)*std::pow(x0, 3) + 20.0*std::pow(M_PI, 3.5)*std::pow(x0, 3) - 20.0*std::pow(M_PI, 3.5)*x0)/std::pow(x0, 3);

        state_fab(i, j, k, Idx::c11) = -0.033333333333333333*std::pow(M_PI, -4.5)*(-10.0*std::pow(M_PI, 4.0)*std::pow(x0, 5) + 10.0*std::pow(M_PI, 4.0)*std::pow(x0, 4)*std::sqrt(std::pow(x0, 2) - 1) + 10.0*std::pow(M_PI, 4.0)*std::pow(x0, 3) - 10.0*std::pow(M_PI, 4.0)*std::sqrt(std::pow(x0, 2) - 1))/std::pow(x0, 3);

        state_fab(i, j, k, Idx::c12) = -0.0055555555555555558*std::pow(M_PI, -6.0)*(-60.0*std::pow(M_PI, 5.5)*std::pow(x0, 6) + 60.0*std::pow(M_PI, 5.5)*std::pow(x0, 5)*std::sqrt(std::pow(x0, 2) - 1) + 60.0*std::pow(M_PI, 5.5)*std::pow(x0, 4) - 180.0*std::pow(M_PI, 5.5)*std::pow(x0, 3)*std::sqrt(std::pow(x0, 2) - 1) + 60.0*std::pow(M_PI, 5.5)*x0*std::sqrt(std::pow(x0, 2) - 1))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::c13) = -0.0055555555555555558*std::pow(M_PI, -6.0)*(-30.0*std::pow(M_PI, 6.0)*std::pow(x0, 6) + 30.0*std::pow(M_PI, 6.0)*std::pow(x0, 5)*std::sqrt(std::pow(x0, 2) - 1) + 69.999999999999986*std::pow(M_PI, 5.0)*std::pow(x0, 4) + 90.0*std::pow(M_PI, 6.0)*std::pow(x0, 4) - 30.0*std::pow(M_PI, 6.0)*std::pow(x0, 3)*std::sqrt(std::pow(x0, 2) - 1) - 90.0*std::pow(M_PI, 6.0)*std::pow(x0, 2) - 69.999999999999986*std::pow(M_PI, 5.0)*std::pow(x0, 2) + 30.0*std::pow(M_PI, 6.0))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::c14) = -0.0055555555555555558*std::pow(M_PI, -6.0)*(120.0*std::pow(M_PI, 5.5)*std::pow(x0, 3)*std::sqrt(std::pow(x0, 2) - 1) - 60.0*std::pow(M_PI, 5.5)*x0*std::sqrt(std::pow(x0, 2) - 1))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::c15) = -0.0055555555555555558*std::pow(M_PI, -6.0)*(120.0*std::pow(M_PI, 5.5)*state_fab(i,j,k,Idx::Kn)*x0 - 90.0*std::pow(M_PI, 6.0)*std::pow(x0, 4) + 69.999999999999986*std::pow(M_PI, 5.0)*std::pow(x0, 2) + 90.0*std::pow(M_PI, 6.0)*std::pow(x0, 2) - 30.0*std::pow(M_PI, 6.0))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::c16) = -0.66666666666666674*std::pow(M_PI, -0.5)*state_fab(i,j,k,Idx::Kn)/std::pow(x0, 3);

        state_fab(i, j, k, Idx::c17) = -0.66666666666666674*std::pow(M_PI, -0.5)*state_fab(i,j,k,Idx::T);

        state_fab(i, j, k, Idx::c18) = -0.0055555555555555558*std::pow(M_PI, -6.0)*(-60.0*std::pow(M_PI, 6.0)*state_fab(i,j,k,Idx::T)*std::pow(x0, 6) + 60.0*std::pow(M_PI, 6.0)*state_fab(i,j,k,Idx::T)*std::pow(x0, 5)*std::sqrt(std::pow(x0, 2) - 1) - 60.0*std::pow(M_PI, 6.0)*state_fab(i,j,k,Idx::T)*std::pow(x0, 3)*std::sqrt(std::pow(x0, 2) - 1))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::c19) = -0.33333333333333337*std::pow(M_PI, -0.5)*std::sqrt(std::pow(x0, 2) - 1)/std::pow(x0, 3);

        state_fab(i, j, k, Idx::c20) = -0.0055555555555555558*std::pow(M_PI, -6.0)*(60.0*std::pow(M_PI, 6.0)*std::pow(x0, 6) - 60.0*std::pow(M_PI, 6.0)*std::pow(x0, 5)*std::sqrt(std::pow(x0, 2) - 1) - 90.0*std::pow(M_PI, 6.0)*std::pow(x0, 4) + 60.0*std::pow(M_PI, 6.0)*std::pow(x0, 3)*std::sqrt(std::pow(x0, 2) - 1) - 69.999999999999986*std::pow(M_PI, 5.0)*std::pow(x0, 2) + 90.0*std::pow(M_PI, 6.0)*std::pow(x0, 2) - 30.0*std::pow(M_PI, 6.0))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::c21) = -0.0055555555555555558*std::pow(M_PI, -6.0)*(60.0*std::pow(M_PI, 5.5)*std::pow(x0, 6) - 60.0*std::pow(M_PI, 5.5)*std::pow(x0, 5)*std::sqrt(std::pow(x0, 2) - 1) - 60.0*std::pow(M_PI, 5.5)*std::pow(x0, 4) + 60.0*std::pow(M_PI, 5.5)*std::pow(x0, 3)*std::sqrt(std::pow(x0, 2) - 1) - 60.0*std::pow(M_PI, 5.5)*x0*std::sqrt(std::pow(x0, 2) - 1))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::c22) = -0.0055555555555555558*std::pow(M_PI, -6.0)*(-30.0*std::pow(M_PI, 6.0)*std::pow(x0, 6) + 30.0*std::pow(M_PI, 6.0)*std::pow(x0, 5)*std::sqrt(std::pow(x0, 2) - 1) - 69.999999999999986*std::pow(M_PI, 5.0)*std::pow(x0, 4) + 90.0*std::pow(M_PI, 6.0)*std::pow(x0, 4) - 30.0*std::pow(M_PI, 6.0)*std::pow(x0, 3)*std::sqrt(std::pow(x0, 2) - 1) - 90.0*std::pow(M_PI, 6.0)*std::pow(x0, 2) + 69.999999999999986*std::pow(M_PI, 5.0)*std::pow(x0, 2) + 30.0*std::pow(M_PI, 6.0))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::c23) = -1.0/12.0*std::pow(M_PI, -1.5)*(-4*std::pow(M_PI, 1.5)*std::pow(x0, 6) + 4*std::pow(M_PI, 1.5)*std::pow(x0, 5)*std::sqrt(std::pow(x0, 2) - 1) + 2*std::pow(M_PI, 1.5)*std::pow(x0, 3)*std::sqrt(std::pow(x0, 2) - 1) + 3*std::pow(M_PI, 1.5)*std::pow(x0, 2) + std::pow(M_PI, 1.5))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::c24) = -1.0/12.0*std::pow(M_PI, -1.5)*(8*std::pow(M_PI, 1.5)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 5) - 8*std::pow(M_PI, 1.5)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 4)*std::sqrt(std::pow(x0, 2) - 1) - 4*std::pow(M_PI, 1.5)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 2)*std::sqrt(std::pow(x0, 2) - 1))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::c25) = (2.0/3.0)*std::pow(M_PI, -0.5)*state_fab(i,j,k,Idx::Kn)/std::pow(x0, 3);

        state_fab(i, j, k, Idx::c26) = -1.0/12.0*std::pow(M_PI, -1.5)*(-8*std::pow(M_PI, 1.0)*state_fab(i,j,k,Idx::Kn)*x0 - 3*std::pow(M_PI, 1.5)*std::pow(x0, 2) - std::pow(M_PI, 1.5))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::c27) = -1.0/12.0*std::pow(M_PI, -1.5)*(-8*std::pow(M_PI, 1.5)*state_fab(i,j,k,Idx::T)*std::pow(x0, 6) + 8*std::pow(M_PI, 1.5)*state_fab(i,j,k,Idx::T)*std::pow(x0, 5)*std::sqrt(std::pow(x0, 2) - 1) + 4*std::pow(M_PI, 1.5)*state_fab(i,j,k,Idx::T)*std::pow(x0, 3)*std::sqrt(std::pow(x0, 2) - 1))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::c28) = -1.0/12.0*std::pow(M_PI, -1.5)*(8*std::pow(M_PI, 1.5)*std::pow(x0, 6) - 8*std::pow(M_PI, 1.5)*std::pow(x0, 5)*std::sqrt(std::pow(x0, 2) - 1) - 4*std::pow(M_PI, 1.5)*std::pow(x0, 3)*std::sqrt(std::pow(x0, 2) - 1) - 3*std::pow(M_PI, 1.5)*std::pow(x0, 2) - std::pow(M_PI, 1.5))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::c29) = -0.033333333333333333*std::pow(M_PI, -5.0)*(-10.0*std::pow(M_PI, 5.0)*std::pow(x0, 5) - 20.0*std::pow(M_PI, 4.0)*std::pow(x0, 5) + 20.0*std::pow(M_PI, 5.0)*std::pow(x0, 4)*std::sqrt(std::pow(x0, 2) - 1) + 10.0*std::pow(M_PI, 5.0)*std::pow(x0, 3) + 40.0*std::pow(M_PI, 4.0)*std::pow(x0, 3) - 20.0*std::pow(M_PI, 4.0)*x0)/std::pow(x0, 3);

        state_fab(i, j, k, Idx::c30) = -0.033333333333333333*std::pow(M_PI, -5.0)*(20.0*std::pow(M_PI, 5.0)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 3)*std::sqrt(std::pow(x0, 2) - 1) + 10.0*std::pow(M_PI, 5.0)*state_fab(i,j,k,Idx::Kn)*x0*std::sqrt(std::pow(x0, 2) - 1) - 20.0*std::pow(M_PI, 4.5)*std::pow(x0, 5) + 20.0*std::pow(M_PI, 4.5)*std::pow(x0, 4)*std::sqrt(std::pow(x0, 2) - 1) + 20.0*std::pow(M_PI, 4.5)*std::pow(x0, 3) - 30.0*std::pow(M_PI, 4.5)*std::pow(x0, 2)*std::sqrt(std::pow(x0, 2) - 1) + 10.0*std::pow(M_PI, 4.5)*std::sqrt(std::pow(x0, 2) - 1))/std::pow(x0, 3);

        state_fab(i, j, k, Idx::c31) = -0.033333333333333333*std::pow(M_PI, -5.0)*(20.0*std::pow(M_PI, 5.0)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 3)*std::sqrt(std::pow(x0, 2) - 1) + 10.0*std::pow(M_PI, 5.0)*state_fab(i,j,k,Idx::Kn)*x0*std::sqrt(std::pow(x0, 2) - 1))/std::pow(x0, 3);

        state_fab(i, j, k, Idx::c32) = -0.033333333333333333*std::pow(M_PI, -5.0)*(10.0*std::pow(M_PI, 5.0)*std::pow(x0, 5) - 10.0*std::pow(M_PI, 5.0)*std::pow(x0, 4)*std::sqrt(std::pow(x0, 2) - 1) - 10.0*std::pow(M_PI, 5.0)*std::pow(x0, 3) - 20.0*std::pow(M_PI, 4.0)*std::pow(x0, 3) + 20.0*std::pow(M_PI, 4.0)*x0)/std::pow(x0, 3);

        state_fab(i, j, k, Idx::c33) = -0.033333333333333333*std::pow(M_PI, -5.0)*(10.0*std::pow(M_PI, 4.5)*std::pow(x0, 5) - 10.0*std::pow(M_PI, 4.5)*std::pow(x0, 4)*std::sqrt(std::pow(x0, 2) - 1) - 10.0*std::pow(M_PI, 4.5)*std::pow(x0, 3) + 30.0*std::pow(M_PI, 4.5)*std::pow(x0, 2)*std::sqrt(std::pow(x0, 2) - 1) - 10.0*std::pow(M_PI, 4.5)*std::sqrt(std::pow(x0, 2) - 1))/std::pow(x0, 3);

        state_fab(i, j, k, Idx::c34) = -0.66666666666666663*state_fab(i,j,k,Idx::T)*x0*std::sqrt(std::pow(x0, 2) - 1);

        state_fab(i, j, k, Idx::c35) = -0.033333333333333333*std::pow(M_PI, -5.0)*(-20.0*std::pow(M_PI, 4.5)*state_fab(i,j,k,Idx::T)*std::pow(x0, 5) + 20.0*std::pow(M_PI, 4.5)*state_fab(i,j,k,Idx::T)*std::pow(x0, 3))/std::pow(x0, 3);

        state_fab(i, j, k, Idx::c36) = -0.033333333333333333*std::pow(M_PI, -5.0)*(10.0*std::pow(M_PI, 5.0)*std::pow(x0, 5) - 10.0*std::pow(M_PI, 5.0)*std::pow(x0, 4)*std::sqrt(std::pow(x0, 2) - 1) - 10.0*std::pow(M_PI, 5.0)*std::pow(x0, 3) + 20.0*std::pow(M_PI, 4.0)*std::pow(x0, 3) - 20.0*std::pow(M_PI, 4.0)*x0)/std::pow(x0, 3);

        state_fab(i, j, k, Idx::c37) = -0.033333333333333333*std::pow(M_PI, -5.0)*(10.0*std::pow(M_PI, 4.5)*std::pow(x0, 5) - 10.0*std::pow(M_PI, 4.5)*std::pow(x0, 4)*std::sqrt(std::pow(x0, 2) - 1) - 10.0*std::pow(M_PI, 4.5)*std::pow(x0, 3) + 10.0*std::pow(M_PI, 4.5)*std::pow(x0, 2)*std::sqrt(std::pow(x0, 2) - 1) - 10.0*std::pow(M_PI, 4.5)*std::sqrt(std::pow(x0, 2) - 1))/std::pow(x0, 3);

        state_fab(i, j, k, Idx::c38) = -0.033333333333333333*std::pow(M_PI, -5.0)*(-10.0*std::pow(M_PI, 5.0)*std::pow(x0, 5) + 20.0*std::pow(M_PI, 4.0)*std::pow(x0, 5) - 40.0*std::pow(M_PI, 4.0)*std::pow(x0, 3) + 10.0*std::pow(M_PI, 5.0)*std::pow(x0, 3) + 20.0*std::pow(M_PI, 4.0)*x0)/std::pow(x0, 3);

        state_fab(i, j, k, Idx::c39) = -0.033333333333333333*std::pow(M_PI, -5.0)*(-10.0*std::pow(M_PI, 4.5)*std::pow(x0, 2)*std::sqrt(std::pow(x0, 2) - 1) + 10.0*std::pow(M_PI, 4.5)*std::sqrt(std::pow(x0, 2) - 1))/std::pow(x0, 3);

        state_fab(i, j, k, Idx::c40) = -0.0055555555555555558*std::pow(M_PI, -6.0)*(120.0*std::pow(M_PI, 5.5)*std::pow(x0, 6) - 120.0*std::pow(M_PI, 5.5)*std::pow(x0, 5)*std::sqrt(std::pow(x0, 2) - 1) - 120.0*std::pow(M_PI, 5.5)*std::pow(x0, 4) + 180.0*std::pow(M_PI, 5.5)*std::pow(x0, 3)*std::sqrt(std::pow(x0, 2) - 1) - 60.0*std::pow(M_PI, 5.5)*x0*std::sqrt(std::pow(x0, 2) - 1))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::c41) = -0.0055555555555555558*std::pow(M_PI, -6.0)*(120.0*std::pow(M_PI, 5.5)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 5) - 120.0*std::pow(M_PI, 5.5)*state_fab(i,j,k,Idx::Kn)*x0 + 69.999999999999986*std::pow(M_PI, 5.0)*std::pow(x0, 6) + 30.0*std::pow(M_PI, 6.0)*std::pow(x0, 6) - 60.0*std::pow(M_PI, 6.0)*std::pow(x0, 5)*std::sqrt(std::pow(x0, 2) - 1) - 90.0*std::pow(M_PI, 6.0)*std::pow(x0, 4) - 139.99999999999997*std::pow(M_PI, 5.0)*std::pow(x0, 4) + 60.0*std::pow(M_PI, 6.0)*std::pow(x0, 3)*std::sqrt(std::pow(x0, 2) - 1) + 69.999999999999986*std::pow(M_PI, 5.0)*std::pow(x0, 2) + 90.0*std::pow(M_PI, 6.0)*std::pow(x0, 2) - 30.0*std::pow(M_PI, 6.0))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::c42) = -0.0055555555555555558*std::pow(M_PI, -6.0)*(120.0*std::pow(M_PI, 5.5)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 5) - 120.0*std::pow(M_PI, 5.5)*state_fab(i,j,k,Idx::Kn)*x0)/std::pow(x0, 4);

        state_fab(i, j, k, Idx::c43) = -0.0055555555555555558*std::pow(M_PI, -6.0)*(-60.0*std::pow(M_PI, 5.5)*std::pow(x0, 6) + 60.0*std::pow(M_PI, 5.5)*std::pow(x0, 5)*std::sqrt(std::pow(x0, 2) - 1) + 60.0*std::pow(M_PI, 5.5)*std::pow(x0, 4) - 120.0*std::pow(M_PI, 5.5)*std::pow(x0, 3)*std::sqrt(std::pow(x0, 2) - 1) + 60.0*std::pow(M_PI, 5.5)*x0*std::sqrt(std::pow(x0, 2) - 1))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::c44) = -0.0055555555555555558*std::pow(M_PI, -6.0)*(120.0*std::pow(M_PI, 5.5)*state_fab(i,j,k,Idx::T)*std::pow(x0, 6) - 120.0*std::pow(M_PI, 5.5)*state_fab(i,j,k,Idx::T)*std::pow(x0, 4))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::c45) = -0.0055555555555555558*std::pow(M_PI, -6.0)*(-60.0*std::pow(M_PI, 6.0)*state_fab(i,j,k,Idx::T)*std::pow(x0, 5)*std::sqrt(std::pow(x0, 2) - 1) + 60.0*std::pow(M_PI, 6.0)*state_fab(i,j,k,Idx::T)*std::pow(x0, 3)*std::sqrt(std::pow(x0, 2) - 1))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::c46) = -0.0055555555555555558*std::pow(M_PI, -6.0)*(-60.0*std::pow(M_PI, 5.5)*std::pow(x0, 6) + 60.0*std::pow(M_PI, 5.5)*std::pow(x0, 5)*std::sqrt(std::pow(x0, 2) - 1) + 60.0*std::pow(M_PI, 5.5)*std::pow(x0, 4) - 60.0*std::pow(M_PI, 5.5)*x0*std::sqrt(std::pow(x0, 2) - 1))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::c47) = -0.0055555555555555558*std::pow(M_PI, -6.0)*(-60.0*std::pow(M_PI, 5.5)*std::pow(x0, 3)*std::sqrt(std::pow(x0, 2) - 1) + 60.0*std::pow(M_PI, 5.5)*x0*std::sqrt(std::pow(x0, 2) - 1))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::c48) = -0.0055555555555555558*std::pow(M_PI, -6.0)*(-69.999999999999986*std::pow(M_PI, 5.0)*std::pow(x0, 6) + 30.0*std::pow(M_PI, 6.0)*std::pow(x0, 6) - 90.0*std::pow(M_PI, 6.0)*std::pow(x0, 4) + 139.99999999999997*std::pow(M_PI, 5.0)*std::pow(x0, 4) - 69.999999999999986*std::pow(M_PI, 5.0)*std::pow(x0, 2) + 90.0*std::pow(M_PI, 6.0)*std::pow(x0, 2) - 30.0*std::pow(M_PI, 6.0))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::c49) = -1.0/12.0*std::pow(M_PI, -2.5)*(8*std::pow(M_PI, 2.5)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 4)*std::sqrt(std::pow(x0, 2) - 1) + 4*std::pow(M_PI, 2.5)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 2)*std::sqrt(std::pow(x0, 2) - 1))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::c50) = -1.0/12.0*std::pow(M_PI, -2.5)*(-8*std::pow(M_PI, 2.0)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 5) + 8*std::pow(M_PI, 2.0)*state_fab(i,j,k,Idx::Kn)*x0)/std::pow(x0, 4);

        state_fab(i, j, k, Idx::c51) = -1.0/12.0*std::pow(M_PI, -2.5)*(-8*std::pow(M_PI, 2.0)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 5) + 8*std::pow(M_PI, 2.0)*state_fab(i,j,k,Idx::Kn)*x0 + 4*std::pow(M_PI, 2.5)*std::pow(x0, 6) - 8*std::pow(M_PI, 2.5)*std::pow(x0, 5)*std::sqrt(std::pow(x0, 2) - 1) - 4*std::pow(M_PI, 2.5)*std::pow(x0, 3)*std::sqrt(std::pow(x0, 2) - 1) - 3*std::pow(M_PI, 2.5)*std::pow(x0, 2) - std::pow(M_PI, 2.5))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::c52) = -1.0/12.0*std::pow(M_PI, -2.5)*(-4*std::pow(M_PI, 2.5)*std::pow(x0, 6) + 4*std::pow(M_PI, 2.5)*std::pow(x0, 5)*std::sqrt(std::pow(x0, 2) - 1) + 2*std::pow(M_PI, 2.5)*std::pow(x0, 3)*std::sqrt(std::pow(x0, 2) - 1) + 3*std::pow(M_PI, 2.5)*std::pow(x0, 2) + std::pow(M_PI, 2.5))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::c53) = -1.0/12.0*std::pow(M_PI, -2.5)*(-8*std::pow(M_PI, 2.5)*state_fab(i,j,k,Idx::T)*std::pow(x0, 5)*std::sqrt(std::pow(x0, 2) - 1) - 4*std::pow(M_PI, 2.5)*state_fab(i,j,k,Idx::T)*std::pow(x0, 3)*std::sqrt(std::pow(x0, 2) - 1))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::c54) = -1.0/12.0*std::pow(M_PI, -2.5)*(4*std::pow(M_PI, 2.5)*std::pow(x0, 6) - 3*std::pow(M_PI, 2.5)*std::pow(x0, 2) - std::pow(M_PI, 2.5))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::c55) = -0.033333333333333333*std::pow(M_PI, -5.0)*(-40.0*std::pow(M_PI, 4.5)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 4) - 10.0*std::pow(M_PI, 5.0)*std::pow(x0, 5) + 20.0*std::pow(M_PI, 4.0)*std::pow(x0, 5) - 40.0*std::pow(M_PI, 4.0)*std::pow(x0, 3) + 10.0*std::pow(M_PI, 5.0)*std::pow(x0, 3) + 20.0*std::pow(M_PI, 4.0)*x0)/std::pow(x0, 3);

        state_fab(i, j, k, Idx::c56) = -0.033333333333333333*std::pow(M_PI, -5.0)*(10.0*std::pow(M_PI, 4.5)*std::pow(x0, 2)*std::sqrt(std::pow(x0, 2) - 1) - 10.0*std::pow(M_PI, 4.5)*std::sqrt(std::pow(x0, 2) - 1))/std::pow(x0, 3);

        state_fab(i, j, k, Idx::c57) = -0.033333333333333333*std::pow(M_PI, -5.0)*(-10.0*std::pow(M_PI, 4.5)*std::pow(x0, 5) + 10.0*std::pow(M_PI, 4.5)*std::pow(x0, 4)*std::sqrt(std::pow(x0, 2) - 1) + 10.0*std::pow(M_PI, 4.5)*std::pow(x0, 3) - 10.0*std::pow(M_PI, 4.5)*std::pow(x0, 2)*std::sqrt(std::pow(x0, 2) - 1) + 10.0*std::pow(M_PI, 4.5)*std::sqrt(std::pow(x0, 2) - 1))/std::pow(x0, 3);

        state_fab(i, j, k, Idx::c58) = -0.033333333333333333*std::pow(M_PI, -5.0)*(-10.0*std::pow(M_PI, 4.5)*std::pow(x0, 5) + 10.0*std::pow(M_PI, 4.5)*std::pow(x0, 4)*std::sqrt(std::pow(x0, 2) - 1) + 10.0*std::pow(M_PI, 4.5)*std::pow(x0, 3) - 30.0*std::pow(M_PI, 4.5)*std::pow(x0, 2)*std::sqrt(std::pow(x0, 2) - 1) + 10.0*std::pow(M_PI, 4.5)*std::sqrt(std::pow(x0, 2) - 1))/std::pow(x0, 3);

        state_fab(i, j, k, Idx::c59) = -0.033333333333333333*std::pow(M_PI, -5.0)*(40.0*std::pow(M_PI, 4.5)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 4) - 10.0*std::pow(M_PI, 5.0)*std::pow(x0, 5) - 20.0*std::pow(M_PI, 4.0)*std::pow(x0, 5) + 20.0*std::pow(M_PI, 5.0)*std::pow(x0, 4)*std::sqrt(std::pow(x0, 2) - 1) + 10.0*std::pow(M_PI, 5.0)*std::pow(x0, 3) + 40.0*std::pow(M_PI, 4.0)*std::pow(x0, 3) - 20.0*std::pow(M_PI, 4.0)*x0)/std::pow(x0, 3);

        state_fab(i, j, k, Idx::c60) = -0.033333333333333333*std::pow(M_PI, -5.0)*(20.0*std::pow(M_PI, 5.0)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 3)*std::sqrt(std::pow(x0, 2) - 1) + 10.0*std::pow(M_PI, 5.0)*state_fab(i,j,k,Idx::Kn)*x0*std::sqrt(std::pow(x0, 2) - 1) + 20.0*std::pow(M_PI, 4.5)*std::pow(x0, 5) - 20.0*std::pow(M_PI, 4.5)*std::pow(x0, 4)*std::sqrt(std::pow(x0, 2) - 1) - 20.0*std::pow(M_PI, 4.5)*std::pow(x0, 3) + 30.0*std::pow(M_PI, 4.5)*std::pow(x0, 2)*std::sqrt(std::pow(x0, 2) - 1) - 10.0*std::pow(M_PI, 4.5)*std::sqrt(std::pow(x0, 2) - 1))/std::pow(x0, 3);

        state_fab(i, j, k, Idx::c61) = -0.033333333333333333*std::pow(M_PI, -5.0)*(20.0*std::pow(M_PI, 4.5)*state_fab(i,j,k,Idx::T)*std::pow(x0, 5) - 20.0*std::pow(M_PI, 4.5)*state_fab(i,j,k,Idx::T)*std::pow(x0, 3))/std::pow(x0, 3);

        state_fab(i, j, k, Idx::c62) = -0.0055555555555555558*std::pow(M_PI, -6.0)*(60.0*std::pow(M_PI, 5.5)*std::pow(x0, 3)*std::sqrt(std::pow(x0, 2) - 1) - 60.0*std::pow(M_PI, 5.5)*x0*std::sqrt(std::pow(x0, 2) - 1))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::c63) = -0.0055555555555555558*std::pow(M_PI, -6.0)*(60.0*std::pow(M_PI, 5.5)*std::pow(x0, 6) - 60.0*std::pow(M_PI, 5.5)*std::pow(x0, 5)*std::sqrt(std::pow(x0, 2) - 1) - 60.0*std::pow(M_PI, 5.5)*std::pow(x0, 4) + 60.0*std::pow(M_PI, 5.5)*x0*std::sqrt(std::pow(x0, 2) - 1))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::c64) = -0.0055555555555555558*std::pow(M_PI, -6.0)*(60.0*std::pow(M_PI, 5.5)*std::pow(x0, 6) - 60.0*std::pow(M_PI, 5.5)*std::pow(x0, 5)*std::sqrt(std::pow(x0, 2) - 1) - 60.0*std::pow(M_PI, 5.5)*std::pow(x0, 4) + 120.0*std::pow(M_PI, 5.5)*std::pow(x0, 3)*std::sqrt(std::pow(x0, 2) - 1) - 60.0*std::pow(M_PI, 5.5)*x0*std::sqrt(std::pow(x0, 2) - 1))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::c65) = -0.0055555555555555558*std::pow(M_PI, -6.0)*(-120.0*std::pow(M_PI, 5.5)*std::pow(x0, 6) + 120.0*std::pow(M_PI, 5.5)*std::pow(x0, 5)*std::sqrt(std::pow(x0, 2) - 1) + 120.0*std::pow(M_PI, 5.5)*std::pow(x0, 4) - 180.0*std::pow(M_PI, 5.5)*std::pow(x0, 3)*std::sqrt(std::pow(x0, 2) - 1) + 60.0*std::pow(M_PI, 5.5)*x0*std::sqrt(std::pow(x0, 2) - 1))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::c66) = -0.0055555555555555558*std::pow(M_PI, -6.0)*(-120.0*std::pow(M_PI, 5.5)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 5) + 120.0*std::pow(M_PI, 5.5)*state_fab(i,j,k,Idx::Kn)*x0 + 69.999999999999986*std::pow(M_PI, 5.0)*std::pow(x0, 6) + 30.0*std::pow(M_PI, 6.0)*std::pow(x0, 6) - 60.0*std::pow(M_PI, 6.0)*std::pow(x0, 5)*std::sqrt(std::pow(x0, 2) - 1) - 90.0*std::pow(M_PI, 6.0)*std::pow(x0, 4) - 139.99999999999997*std::pow(M_PI, 5.0)*std::pow(x0, 4) + 60.0*std::pow(M_PI, 6.0)*std::pow(x0, 3)*std::sqrt(std::pow(x0, 2) - 1) + 69.999999999999986*std::pow(M_PI, 5.0)*std::pow(x0, 2) + 90.0*std::pow(M_PI, 6.0)*std::pow(x0, 2) - 30.0*std::pow(M_PI, 6.0))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::c67) = -0.0055555555555555558*std::pow(M_PI, -6.0)*(-120.0*std::pow(M_PI, 5.5)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 5) + 120.0*std::pow(M_PI, 5.5)*state_fab(i,j,k,Idx::Kn)*x0)/std::pow(x0, 4);

        state_fab(i, j, k, Idx::c68) = -0.0055555555555555558*std::pow(M_PI, -6.0)*(-120.0*std::pow(M_PI, 5.5)*state_fab(i,j,k,Idx::T)*std::pow(x0, 6) + 120.0*std::pow(M_PI, 5.5)*state_fab(i,j,k,Idx::T)*std::pow(x0, 4))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::c69) = -1.0/12.0*std::pow(M_PI, -2.5)*(32*std::pow(M_PI, 2.0)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 5) + 4*std::pow(M_PI, 2.5)*std::pow(x0, 6) - 3*std::pow(M_PI, 2.5)*std::pow(x0, 2) - std::pow(M_PI, 2.5))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::c70) = -1.0/12.0*std::pow(M_PI, -2.5)*(8*std::pow(M_PI, 2.0)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 5) - 8*std::pow(M_PI, 2.0)*state_fab(i,j,k,Idx::Kn)*x0)/std::pow(x0, 4);

        state_fab(i, j, k, Idx::c71) = -1.0/12.0*std::pow(M_PI, -2.5)*(-24*std::pow(M_PI, 2.0)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 5) - 8*std::pow(M_PI, 2.0)*state_fab(i,j,k,Idx::Kn)*x0 + 4*std::pow(M_PI, 2.5)*std::pow(x0, 6) - 8*std::pow(M_PI, 2.5)*std::pow(x0, 5)*std::sqrt(std::pow(x0, 2) - 1) - 4*std::pow(M_PI, 2.5)*std::pow(x0, 3)*std::sqrt(std::pow(x0, 2) - 1) - 3*std::pow(M_PI, 2.5)*std::pow(x0, 2) - std::pow(M_PI, 2.5))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::c72) = -0.033333333333333333*std::pow(M_PI, -4.5)*(10.0*std::pow(M_PI, 4.0)*std::pow(x0, 5) - 10.0*std::pow(M_PI, 4.0)*std::pow(x0, 4)*std::sqrt(std::pow(x0, 2) - 1) - 10.0*std::pow(M_PI, 4.0)*std::pow(x0, 3) + 10.0*std::pow(M_PI, 4.0)*std::sqrt(std::pow(x0, 2) - 1))/std::pow(x0, 3);

        state_fab(i, j, k, Idx::c73) = -0.033333333333333333*std::pow(M_PI, -4.5)*(20.0*std::pow(M_PI, 4.5)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 4) - 20.0*std::pow(M_PI, 4.5)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 3)*std::sqrt(std::pow(x0, 2) - 1) - 10.0*std::pow(M_PI, 4.5)*state_fab(i,j,k,Idx::Kn)*x0*std::sqrt(std::pow(x0, 2) - 1) + 20.0*std::pow(M_PI, 4.0)*std::pow(x0, 2)*std::sqrt(std::pow(x0, 2) - 1) - 10.0*std::pow(M_PI, 4.0)*std::sqrt(std::pow(x0, 2) - 1))/std::pow(x0, 3);

        state_fab(i, j, k, Idx::c74) = -0.033333333333333333*std::pow(M_PI, -4.5)*(-10.0*std::pow(M_PI, 4.0)*std::pow(x0, 5) + 10.0*std::pow(M_PI, 4.0)*std::pow(x0, 4)*std::sqrt(std::pow(x0, 2) - 1) + 10.0*std::pow(M_PI, 4.0)*std::pow(x0, 3) - 20.0*std::pow(M_PI, 4.0)*std::pow(x0, 2)*std::sqrt(std::pow(x0, 2) - 1) + 10.0*std::pow(M_PI, 4.0)*std::sqrt(std::pow(x0, 2) - 1))/std::pow(x0, 3);

        state_fab(i, j, k, Idx::c75) = -0.0055555555555555558*std::pow(M_PI, -6.0)*(-60.0*std::pow(M_PI, 5.5)*std::pow(x0, 6) + 60.0*std::pow(M_PI, 5.5)*std::pow(x0, 5)*std::sqrt(std::pow(x0, 2) - 1) + 60.0*std::pow(M_PI, 5.5)*std::pow(x0, 4) - 60.0*std::pow(M_PI, 5.5)*std::pow(x0, 3)*std::sqrt(std::pow(x0, 2) - 1) + 60.0*std::pow(M_PI, 5.5)*x0*std::sqrt(std::pow(x0, 2) - 1))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::c76) = -0.0055555555555555558*std::pow(M_PI, -6.0)*(-120.0*std::pow(M_PI, 5.5)*std::pow(x0, 3)*std::sqrt(std::pow(x0, 2) - 1) + 60.0*std::pow(M_PI, 5.5)*x0*std::sqrt(std::pow(x0, 2) - 1))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::c77) = -0.0055555555555555558*std::pow(M_PI, -6.0)*(-120.0*std::pow(M_PI, 5.5)*state_fab(i,j,k,Idx::Kn)*x0 - 90.0*std::pow(M_PI, 6.0)*std::pow(x0, 4) + 69.999999999999986*std::pow(M_PI, 5.0)*std::pow(x0, 2) + 90.0*std::pow(M_PI, 6.0)*std::pow(x0, 2) - 30.0*std::pow(M_PI, 6.0))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::c78) = -0.0055555555555555558*std::pow(M_PI, -6.0)*(60.0*std::pow(M_PI, 5.5)*std::pow(x0, 6) - 60.0*std::pow(M_PI, 5.5)*std::pow(x0, 5)*std::sqrt(std::pow(x0, 2) - 1) - 60.0*std::pow(M_PI, 5.5)*std::pow(x0, 4) + 180.0*std::pow(M_PI, 5.5)*std::pow(x0, 3)*std::sqrt(std::pow(x0, 2) - 1) - 60.0*std::pow(M_PI, 5.5)*x0*std::sqrt(std::pow(x0, 2) - 1))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::c79) = -1.0/12.0*std::pow(M_PI, -1.5)*(8*std::pow(M_PI, 1.0)*state_fab(i,j,k,Idx::Kn)*x0 - 3*std::pow(M_PI, 1.5)*std::pow(x0, 2) - std::pow(M_PI, 1.5))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::c80) = -0.033333333333333333*std::pow(M_PI, -4.5)*(-20.0*std::pow(M_PI, 4.5)*state_fab(i,j,k,Idx::T)*std::pow(x0, 5) + 20.0*std::pow(M_PI, 4.5)*state_fab(i,j,k,Idx::T)*std::pow(x0, 4)*std::sqrt(std::pow(x0, 2) - 1))/std::pow(x0, 3);

        state_fab(i, j, k, Idx::c81) = -0.0055555555555555558*std::pow(M_PI, -6.0)*(60.0*std::pow(M_PI, 6.0)*state_fab(i,j,k,Idx::T)*std::pow(x0, 6) - 60.0*std::pow(M_PI, 6.0)*state_fab(i,j,k,Idx::T)*std::pow(x0, 5)*std::sqrt(std::pow(x0, 2) - 1) + 60.0*std::pow(M_PI, 6.0)*state_fab(i,j,k,Idx::T)*std::pow(x0, 3)*std::sqrt(std::pow(x0, 2) - 1))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::c82) = -1.0/12.0*std::pow(M_PI, -1.5)*(8*std::pow(M_PI, 1.5)*state_fab(i,j,k,Idx::T)*std::pow(x0, 6) - 8*std::pow(M_PI, 1.5)*state_fab(i,j,k,Idx::T)*std::pow(x0, 5)*std::sqrt(std::pow(x0, 2) - 1) - 4*std::pow(M_PI, 1.5)*state_fab(i,j,k,Idx::T)*std::pow(x0, 3)*std::sqrt(std::pow(x0, 2) - 1))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::c83) = -0.0055555555555555558*std::pow(M_PI, -6.0)*(60.0*std::pow(M_PI, 6.0)*state_fab(i,j,k,Idx::T)*std::pow(x0, 5)*std::sqrt(std::pow(x0, 2) - 1) - 60.0*std::pow(M_PI, 6.0)*state_fab(i,j,k,Idx::T)*std::pow(x0, 3)*std::sqrt(std::pow(x0, 2) - 1))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::c84) = -1.0/12.0*std::pow(M_PI, -2.5)*(8*std::pow(M_PI, 2.5)*state_fab(i,j,k,Idx::T)*std::pow(x0, 5)*std::sqrt(std::pow(x0, 2) - 1) + 4*std::pow(M_PI, 2.5)*state_fab(i,j,k,Idx::T)*std::pow(x0, 3)*std::sqrt(std::pow(x0, 2) - 1))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::g0) = -0.66666666666666663*std::pow(M_PI, -0.5)*state_fab(i,j,k,Idx::Kn);

        state_fab(i, j, k, Idx::g1) = 0.033333333333333333*std::pow(M_PI, -4.5)*(10.0*std::pow(M_PI, 4.5)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 5) - 10.0*std::pow(M_PI, 4.5)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 4)*std::sqrt(std::pow(x0, 2) - 1) + 10.0*std::pow(M_PI, 4.5)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 2)*std::sqrt(std::pow(x0, 2) - 1))/std::pow(x0, 3);

        state_fab(i, j, k, Idx::g2) = 0.0055555555555555558*std::pow(M_PI, -6.0)*(-60.0*std::pow(M_PI, 6.0)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 6) + 60.0*std::pow(M_PI, 6.0)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 5)*std::sqrt(std::pow(x0, 2) - 1) - 60.0*std::pow(M_PI, 6.0)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 3)*std::sqrt(std::pow(x0, 2) - 1))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::g3) = 0.0055555555555555558*std::pow(M_PI, -6.0)*(240.0*std::pow(M_PI, 5.5)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 4) - 120.0*std::pow(M_PI, 5.5)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 2))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::g4) = (2.0/3.0)*std::pow(M_PI, -0.5)*state_fab(i,j,k,Idx::Kn)/std::pow(x0, 2);

        state_fab(i, j, k, Idx::g5) = 0.033333333333333333*std::pow(M_PI, -5.0)*(-20.0*std::pow(M_PI, 4.5)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 5) + 20.0*std::pow(M_PI, 4.5)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 3))/std::pow(x0, 3);

        state_fab(i, j, k, Idx::g6) = 0.033333333333333333*std::pow(M_PI, -5.0)*(10.0*std::pow(M_PI, 5.0)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 4)*std::sqrt(std::pow(x0, 2) - 1) - 10.0*std::pow(M_PI, 5.0)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 2)*std::sqrt(std::pow(x0, 2) - 1))/std::pow(x0, 3);

        state_fab(i, j, k, Idx::g7) = 0.0055555555555555558*std::pow(M_PI, -6.0)*(-60.0*std::pow(M_PI, 6.0)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 5)*std::sqrt(std::pow(x0, 2) - 1) + 60.0*std::pow(M_PI, 6.0)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 3)*std::sqrt(std::pow(x0, 2) - 1))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::g8) = 0.0055555555555555558*std::pow(M_PI, -6.0)*(120.0*std::pow(M_PI, 5.5)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 6) - 240.0*std::pow(M_PI, 5.5)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 4) + 120.0*std::pow(M_PI, 5.5)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 2))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::g9) = (1.0/12.0)*std::pow(M_PI, -2.5)*(8*std::pow(M_PI, 2.0)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 6) - 8*std::pow(M_PI, 2.0)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 2))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::g10) = 0.033333333333333333*std::pow(M_PI, -5.0)*(20.0*std::pow(M_PI, 4.5)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 5) - 20.0*std::pow(M_PI, 4.5)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 3))/std::pow(x0, 3);

        state_fab(i, j, k, Idx::g11) = 0.0055555555555555558*std::pow(M_PI, -6.0)*(-120.0*std::pow(M_PI, 5.5)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 6) + 240.0*std::pow(M_PI, 5.5)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 4) - 120.0*std::pow(M_PI, 5.5)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 2))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::g12) = (1.0/12.0)*std::pow(M_PI, -2.5)*(-8*std::pow(M_PI, 2.0)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 6) + 8*std::pow(M_PI, 2.0)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 2))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::g13) = 0.0055555555555555558*std::pow(M_PI, -6.0)*(-240.0*std::pow(M_PI, 5.5)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 4) + 120.0*std::pow(M_PI, 5.5)*state_fab(i,j,k,Idx::Kn)*std::pow(x0, 2))/std::pow(x0, 4);

        state_fab(i, j, k, Idx::q0) = 1.0/state_fab(i,j,k,Idx::g10);

        state_fab(i, j, k, Idx::q1) = state_fab(i,j,k,Idx::g7)*state_fab(i,j,k,Idx::q0);

        state_fab(i, j, k, Idx::q2) = 1.0/(state_fab(i,j,k,Idx::g11) - state_fab(i,j,k,Idx::g6)*state_fab(i,j,k,Idx::q1));

        state_fab(i, j, k, Idx::q3) = 1.0/state_fab(i,j,k,Idx::g12);

        state_fab(i, j, k, Idx::q4) = 1.0/state_fab(i,j,k,Idx::g5);

        state_fab(i, j, k, Idx::q5) = state_fab(i,j,k,Idx::g7)*state_fab(i,j,k,Idx::q4);

        state_fab(i, j, k, Idx::q6) = 1.0/(-state_fab(i,j,k,Idx::g6)*state_fab(i,j,k,Idx::q5) + state_fab(i,j,k,Idx::g8));

        state_fab(i, j, k, Idx::q7) = 1.0/state_fab(i,j,k,Idx::g9);

        state_fab(i, j, k, Idx::q8) = 1.0/state_fab(i,j,k,Idx::g0);

        state_fab(i, j, k, Idx::q9) = state_fab(i,j,k,Idx::g2)*state_fab(i,j,k,Idx::q8);

        state_fab(i, j, k, Idx::q10) = state_fab(i,j,k,Idx::g1)*state_fab(i,j,k,Idx::q9);

        state_fab(i, j, k, Idx::q11) = 1.0/(state_fab(i,j,k,Idx::g13) + state_fab(i,j,k,Idx::q10));

        state_fab(i, j, k, Idx::q12) = 1.0/state_fab(i,j,k,Idx::g4);

        state_fab(i, j, k, Idx::q13) = 1.0/(state_fab(i,j,k,Idx::g3) - state_fab(i,j,k,Idx::q10));

}
#endif