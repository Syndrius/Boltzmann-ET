#ifndef ET_INTEGRATION_DIAG_K_H
#define ET_INTEGRATION_DIAG_K_H

#include <AMReX_REAL.H> 
#include <AMReX_Array4.H> 
#include <ET_Integration_Setup_K.H> 

AMREX_GPU_DEVICE 
inline 
void 
state_diagnostics(int i, int j, int k, 
        amrex::Array4<amrex::Real> const& diag, 
        amrex::Array4<amrex::Real const> const& state_fab,
        const amrex::Real time_lev,
        amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx, 
        const amrex::GeometryData& geom) noexcept 
{
        const auto domain_xlo = geom.ProbLo();
        
        amrex::Real x0 = (i + 0.5)*geom.CellSize(0) + domain_xlo[0]; 
        amrex::Real x1 = (j + 0.5)*geom.CellSize(1) + domain_xlo[1]; 
        amrex::Real bi1 = state_fab(i, j, k, Idx::bi1);

        amrex::Real bi2 = state_fab(i, j, k, Idx::bi2);

        amrex::Real bi3 = state_fab(i, j, k, Idx::bi3);

        amrex::Real br1 = state_fab(i, j, k, Idx::br1);

        amrex::Real br2 = state_fab(i, j, k, Idx::br2);

        amrex::Real br3 = state_fab(i, j, k, Idx::br3);

        amrex::Real li1 = state_fab(i, j, k, Idx::li1);

        amrex::Real li2 = state_fab(i, j, k, Idx::li2);

        amrex::Real li3 = state_fab(i, j, k, Idx::li3);

        amrex::Real lr1 = state_fab(i, j, k, Idx::lr1);

        amrex::Real lr2 = state_fab(i, j, k, Idx::lr2);

        amrex::Real lr3 = state_fab(i, j, k, Idx::lr3);

        amrex::Real vi1 = state_fab(i, j, k, Idx::vi1);

        amrex::Real vi2 = state_fab(i, j, k, Idx::vi2);

        amrex::Real vi3 = state_fab(i, j, k, Idx::vi3);

        amrex::Real vr1 = state_fab(i, j, k, Idx::vr1);

        amrex::Real vr2 = state_fab(i, j, k, Idx::vr2);

        amrex::Real vr3 = state_fab(i, j, k, Idx::vr3);

        amrex::Real wi1 = state_fab(i, j, k, Idx::wi1);

        amrex::Real wi2 = state_fab(i, j, k, Idx::wi2);

        amrex::Real wi3 = state_fab(i, j, k, Idx::wi3);

        amrex::Real wr1 = state_fab(i, j, k, Idx::wr1);

        amrex::Real wr2 = state_fab(i, j, k, Idx::wr2);

        amrex::Real wr3 = state_fab(i, j, k, Idx::wr3);

        diag(i, j, k, Diag::u_r_re) = (1.0/2.0)*std::pow(M_PI, -0.5)*br1 - 1.0/2.0*std::pow(M_PI, -0.5)*br1/std::pow(x0, 2) + (1.0/4.0)*br2*std::sqrt(1 - 1/std::pow(x0, 2)) - 1.0/4.0*br2*std::sqrt(1 - 1/std::pow(x0, 2))/std::pow(x0, 2) - 1.0/2.0*std::pow(M_PI, -0.5)*lr1 + (1.0/2.0)*std::pow(M_PI, -0.5)*lr1/std::pow(x0, 2) + (1.0/4.0)*lr2*std::sqrt(1 - 1/std::pow(x0, 2)) - 1.0/4.0*lr2*std::sqrt(1 - 1/std::pow(x0, 2))/std::pow(x0, 2) + (1.0/2.0)*std::pow(M_PI, -0.5)*vr1/std::pow(x0, 2) - 1.0/4.0*vr2*std::sqrt(1 - 1/std::pow(x0, 2)) + (1.0/4.0)*vr2 + (1.0/4.0)*vr2*std::sqrt(1 - 1/std::pow(x0, 2))/std::pow(x0, 2) - 1.0/2.0*std::pow(M_PI, -0.5)*wr1/std::pow(x0, 2) - 1.0/4.0*wr2*std::sqrt(1 - 1/std::pow(x0, 2)) + (1.0/4.0)*wr2 + (1.0/4.0)*wr2*std::sqrt(1 - 1/std::pow(x0, 2))/std::pow(x0, 2);

        diag(i, j, k, Diag::u_al_re) = (1.0/4.0)*br3*std::sqrt(1 - 1/std::pow(x0, 2)) + (1.0/8.0)*br3*std::sqrt(1 - 1/std::pow(x0, 2))/std::pow(x0, 2) + (1.0/4.0)*lr3*std::sqrt(1 - 1/std::pow(x0, 2)) + (1.0/8.0)*lr3*std::sqrt(1 - 1/std::pow(x0, 2))/std::pow(x0, 2) - 1.0/4.0*vr3*std::sqrt(1 - 1/std::pow(x0, 2)) + (1.0/4.0)*vr3 - 1.0/8.0*vr3*std::sqrt(1 - 1/std::pow(x0, 2))/std::pow(x0, 2) - 1.0/4.0*wr3*std::sqrt(1 - 1/std::pow(x0, 2)) + (1.0/4.0)*wr3 - 1.0/8.0*wr3*std::sqrt(1 - 1/std::pow(x0, 2))/std::pow(x0, 2);

        diag(i, j, k, Diag::sigma_re) = (1.0/2.0)*br1*std::sqrt(1 - 1/std::pow(x0, 2)) + (1.0/2.0)*std::pow(M_PI, -0.5)*br2 - 1.0/2.0*std::pow(M_PI, -0.5)*br2/std::pow(x0, 2) + (1.0/2.0)*lr1*std::sqrt(1 - 1/std::pow(x0, 2)) - 1.0/2.0*std::pow(M_PI, -0.5)*lr2 + (1.0/2.0)*std::pow(M_PI, -0.5)*lr2/std::pow(x0, 2) - 1.0/2.0*vr1*std::sqrt(1 - 1/std::pow(x0, 2)) + (1.0/2.0)*vr1 + (1.0/2.0)*std::pow(M_PI, -0.5)*vr2/std::pow(x0, 2) - 1.0/2.0*wr1*std::sqrt(1 - 1/std::pow(x0, 2)) + (1.0/2.0)*wr1 - 1.0/2.0*std::pow(M_PI, -0.5)*wr2/std::pow(x0, 2);

        diag(i, j, k, Diag::kappa_re) = 0.16666666666666663*std::pow(M_PI, -0.5)*br2 - 0.16666666666666663*std::pow(M_PI, -0.5)*br2/std::pow(x0, 2) - 0.16666666666666663*std::pow(M_PI, -0.5)*lr2 + 0.16666666666666663*std::pow(M_PI, -0.5)*lr2/std::pow(x0, 2) + 0.16666666666666663*std::pow(M_PI, -0.5)*vr2/std::pow(x0, 2) - 0.16666666666666663*std::pow(M_PI, -0.5)*wr2/std::pow(x0, 2);

        diag(i, j, k, Diag::u_r_im) = (1.0/2.0)*std::pow(M_PI, -0.5)*bi1 - 1.0/2.0*std::pow(M_PI, -0.5)*bi1/std::pow(x0, 2) + (1.0/4.0)*bi2*std::sqrt(1 - 1/std::pow(x0, 2)) - 1.0/4.0*bi2*std::sqrt(1 - 1/std::pow(x0, 2))/std::pow(x0, 2) - 1.0/2.0*std::pow(M_PI, -0.5)*li1 + (1.0/2.0)*std::pow(M_PI, -0.5)*li1/std::pow(x0, 2) + (1.0/4.0)*li2*std::sqrt(1 - 1/std::pow(x0, 2)) - 1.0/4.0*li2*std::sqrt(1 - 1/std::pow(x0, 2))/std::pow(x0, 2) + (1.0/2.0)*std::pow(M_PI, -0.5)*vi1/std::pow(x0, 2) - 1.0/4.0*vi2*std::sqrt(1 - 1/std::pow(x0, 2)) + (1.0/4.0)*vi2 + (1.0/4.0)*vi2*std::sqrt(1 - 1/std::pow(x0, 2))/std::pow(x0, 2) - 1.0/2.0*std::pow(M_PI, -0.5)*wi1/std::pow(x0, 2) - 1.0/4.0*wi2*std::sqrt(1 - 1/std::pow(x0, 2)) + (1.0/4.0)*wi2 + (1.0/4.0)*wi2*std::sqrt(1 - 1/std::pow(x0, 2))/std::pow(x0, 2);

        diag(i, j, k, Diag::u_al_im) = (1.0/4.0)*bi3*std::sqrt(1 - 1/std::pow(x0, 2)) + (1.0/8.0)*bi3*std::sqrt(1 - 1/std::pow(x0, 2))/std::pow(x0, 2) + (1.0/4.0)*li3*std::sqrt(1 - 1/std::pow(x0, 2)) + (1.0/8.0)*li3*std::sqrt(1 - 1/std::pow(x0, 2))/std::pow(x0, 2) - 1.0/4.0*vi3*std::sqrt(1 - 1/std::pow(x0, 2)) + (1.0/4.0)*vi3 - 1.0/8.0*vi3*std::sqrt(1 - 1/std::pow(x0, 2))/std::pow(x0, 2) - 1.0/4.0*wi3*std::sqrt(1 - 1/std::pow(x0, 2)) + (1.0/4.0)*wi3 - 1.0/8.0*wi3*std::sqrt(1 - 1/std::pow(x0, 2))/std::pow(x0, 2);

        diag(i, j, k, Diag::sigma_im) = (1.0/2.0)*bi1*std::sqrt(1 - 1/std::pow(x0, 2)) + (1.0/2.0)*std::pow(M_PI, -0.5)*bi2 - 1.0/2.0*std::pow(M_PI, -0.5)*bi2/std::pow(x0, 2) + (1.0/2.0)*li1*std::sqrt(1 - 1/std::pow(x0, 2)) - 1.0/2.0*std::pow(M_PI, -0.5)*li2 + (1.0/2.0)*std::pow(M_PI, -0.5)*li2/std::pow(x0, 2) - 1.0/2.0*vi1*std::sqrt(1 - 1/std::pow(x0, 2)) + (1.0/2.0)*vi1 + (1.0/2.0)*std::pow(M_PI, -0.5)*vi2/std::pow(x0, 2) - 1.0/2.0*wi1*std::sqrt(1 - 1/std::pow(x0, 2)) + (1.0/2.0)*wi1 - 1.0/2.0*std::pow(M_PI, -0.5)*wi2/std::pow(x0, 2);

        diag(i, j, k, Diag::kappa_im) = 0.16666666666666663*std::pow(M_PI, -0.5)*bi2 - 0.16666666666666663*std::pow(M_PI, -0.5)*bi2/std::pow(x0, 2) - 0.16666666666666663*std::pow(M_PI, -0.5)*li2 + 0.16666666666666663*std::pow(M_PI, -0.5)*li2/std::pow(x0, 2) + 0.16666666666666663*std::pow(M_PI, -0.5)*vi2/std::pow(x0, 2) - 0.16666666666666663*std::pow(M_PI, -0.5)*wi2/std::pow(x0, 2);

}
#endif