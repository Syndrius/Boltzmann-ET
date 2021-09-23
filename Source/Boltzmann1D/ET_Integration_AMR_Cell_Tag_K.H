#ifndef ET_INTEGRATION_AMR_CELL_TAG_K_H
#define ET_INTEGRATION_AMR_CELL_TAG_K_H

#include <AMReX_REAL.H> 
#include <AMReX_Array4.H> 
#include <ET_Integration_Setup_K.H> 

AMREX_GPU_DEVICE 
inline 
bool
state_is_tagged(int i, int j, int k, 
        amrex::Array4<amrex::Real const> const& state_fab,
        amrex::Real error_threshold,
        const amrex::Real time,
        amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx,
        const amrex::GeometryData& geom) noexcept 
{

        const auto domain_xlo = geom.ProbLo(); 

        amrex::Real x0 = (i + 0.5)*geom.CellSize(0) + domain_xlo[0]; 
        amrex::Real x1 = (j + 0.5)*geom.CellSize(1) + domain_xlo[1]; 
        amrex::Real x2 = (k + 0.5)*geom.CellSize(2) + domain_xlo[2]; 
        amrex::Real bi1 = state_fab(i, j, k, Idx::bi1);

        amrex::Real dbi10 = (-2*state_fab(i - 1, j, k, Idx::bi1) + (1.0/2.0)*state_fab(i - 2, j, k, Idx::bi1) + (3.0/2.0)*state_fab(i, j, k, Idx::bi1))/dx[0];
        amrex::Real bi2 = state_fab(i, j, k, Idx::bi2);

        amrex::Real dbi20 = (-2*state_fab(i - 1, j, k, Idx::bi2) + (1.0/2.0)*state_fab(i - 2, j, k, Idx::bi2) + (3.0/2.0)*state_fab(i, j, k, Idx::bi2))/dx[0];
        amrex::Real bi3 = state_fab(i, j, k, Idx::bi3);

        amrex::Real dbi30 = (-2*state_fab(i - 1, j, k, Idx::bi3) + (1.0/2.0)*state_fab(i - 2, j, k, Idx::bi3) + (3.0/2.0)*state_fab(i, j, k, Idx::bi3))/dx[0];
        amrex::Real br1 = state_fab(i, j, k, Idx::br1);

        amrex::Real dbr10 = (-2*state_fab(i - 1, j, k, Idx::br1) + (1.0/2.0)*state_fab(i - 2, j, k, Idx::br1) + (3.0/2.0)*state_fab(i, j, k, Idx::br1))/dx[0];
        amrex::Real br2 = state_fab(i, j, k, Idx::br2);

        amrex::Real dbr20 = (-2*state_fab(i - 1, j, k, Idx::br2) + (1.0/2.0)*state_fab(i - 2, j, k, Idx::br2) + (3.0/2.0)*state_fab(i, j, k, Idx::br2))/dx[0];
        amrex::Real br3 = state_fab(i, j, k, Idx::br3);

        amrex::Real dbr30 = (-2*state_fab(i - 1, j, k, Idx::br3) + (1.0/2.0)*state_fab(i - 2, j, k, Idx::br3) + (3.0/2.0)*state_fab(i, j, k, Idx::br3))/dx[0];
        amrex::Real li1 = state_fab(i, j, k, Idx::li1);

        amrex::Real dli10 = (2*state_fab(i + 1, j, k, Idx::li1) - 1.0/2.0*state_fab(i + 2, j, k, Idx::li1) - 3.0/2.0*state_fab(i, j, k, Idx::li1))/dx[0];
        amrex::Real li2 = state_fab(i, j, k, Idx::li2);

        amrex::Real dli20 = (2*state_fab(i + 1, j, k, Idx::li2) - 1.0/2.0*state_fab(i + 2, j, k, Idx::li2) - 3.0/2.0*state_fab(i, j, k, Idx::li2))/dx[0];
        amrex::Real li3 = state_fab(i, j, k, Idx::li3);

        amrex::Real dli30 = (2*state_fab(i + 1, j, k, Idx::li3) - 1.0/2.0*state_fab(i + 2, j, k, Idx::li3) - 3.0/2.0*state_fab(i, j, k, Idx::li3))/dx[0];
        amrex::Real lr1 = state_fab(i, j, k, Idx::lr1);

        amrex::Real dlr10 = (2*state_fab(i + 1, j, k, Idx::lr1) - 1.0/2.0*state_fab(i + 2, j, k, Idx::lr1) - 3.0/2.0*state_fab(i, j, k, Idx::lr1))/dx[0];
        amrex::Real lr2 = state_fab(i, j, k, Idx::lr2);

        amrex::Real dlr20 = (2*state_fab(i + 1, j, k, Idx::lr2) - 1.0/2.0*state_fab(i + 2, j, k, Idx::lr2) - 3.0/2.0*state_fab(i, j, k, Idx::lr2))/dx[0];
        amrex::Real lr3 = state_fab(i, j, k, Idx::lr3);

        amrex::Real dlr30 = (2*state_fab(i + 1, j, k, Idx::lr3) - 1.0/2.0*state_fab(i + 2, j, k, Idx::lr3) - 3.0/2.0*state_fab(i, j, k, Idx::lr3))/dx[0];
        amrex::Real vi1 = state_fab(i, j, k, Idx::vi1);

        amrex::Real dvi10 = (-2*state_fab(i - 1, j, k, Idx::vi1) + (1.0/2.0)*state_fab(i - 2, j, k, Idx::vi1) + (3.0/2.0)*state_fab(i, j, k, Idx::vi1))/dx[0];
        amrex::Real vi2 = state_fab(i, j, k, Idx::vi2);

        amrex::Real dvi20 = (-2*state_fab(i - 1, j, k, Idx::vi2) + (1.0/2.0)*state_fab(i - 2, j, k, Idx::vi2) + (3.0/2.0)*state_fab(i, j, k, Idx::vi2))/dx[0];
        amrex::Real vi3 = state_fab(i, j, k, Idx::vi3);

        amrex::Real dvi30 = (-2*state_fab(i - 1, j, k, Idx::vi3) + (1.0/2.0)*state_fab(i - 2, j, k, Idx::vi3) + (3.0/2.0)*state_fab(i, j, k, Idx::vi3))/dx[0];
        amrex::Real vr1 = state_fab(i, j, k, Idx::vr1);

        amrex::Real dvr10 = (-2*state_fab(i - 1, j, k, Idx::vr1) + (1.0/2.0)*state_fab(i - 2, j, k, Idx::vr1) + (3.0/2.0)*state_fab(i, j, k, Idx::vr1))/dx[0];
        amrex::Real vr2 = state_fab(i, j, k, Idx::vr2);

        amrex::Real dvr20 = (-2*state_fab(i - 1, j, k, Idx::vr2) + (1.0/2.0)*state_fab(i - 2, j, k, Idx::vr2) + (3.0/2.0)*state_fab(i, j, k, Idx::vr2))/dx[0];
        amrex::Real vr3 = state_fab(i, j, k, Idx::vr3);

        amrex::Real dvr30 = (-2*state_fab(i - 1, j, k, Idx::vr3) + (1.0/2.0)*state_fab(i - 2, j, k, Idx::vr3) + (3.0/2.0)*state_fab(i, j, k, Idx::vr3))/dx[0];
        amrex::Real wi1 = state_fab(i, j, k, Idx::wi1);

        amrex::Real dwi10 = (2*state_fab(i + 1, j, k, Idx::wi1) - 1.0/2.0*state_fab(i + 2, j, k, Idx::wi1) - 3.0/2.0*state_fab(i, j, k, Idx::wi1))/dx[0];
        amrex::Real wi2 = state_fab(i, j, k, Idx::wi2);

        amrex::Real dwi20 = (2*state_fab(i + 1, j, k, Idx::wi2) - 1.0/2.0*state_fab(i + 2, j, k, Idx::wi2) - 3.0/2.0*state_fab(i, j, k, Idx::wi2))/dx[0];
        amrex::Real wi3 = state_fab(i, j, k, Idx::wi3);

        amrex::Real dwi30 = (2*state_fab(i + 1, j, k, Idx::wi3) - 1.0/2.0*state_fab(i + 2, j, k, Idx::wi3) - 3.0/2.0*state_fab(i, j, k, Idx::wi3))/dx[0];
        amrex::Real wr1 = state_fab(i, j, k, Idx::wr1);

        amrex::Real dwr10 = (2*state_fab(i + 1, j, k, Idx::wr1) - 1.0/2.0*state_fab(i + 2, j, k, Idx::wr1) - 3.0/2.0*state_fab(i, j, k, Idx::wr1))/dx[0];
        amrex::Real wr2 = state_fab(i, j, k, Idx::wr2);

        amrex::Real dwr20 = (2*state_fab(i + 1, j, k, Idx::wr2) - 1.0/2.0*state_fab(i + 2, j, k, Idx::wr2) - 3.0/2.0*state_fab(i, j, k, Idx::wr2))/dx[0];
        amrex::Real wr3 = state_fab(i, j, k, Idx::wr3);

        amrex::Real dwr30 = (2*state_fab(i + 1, j, k, Idx::wr3) - 1.0/2.0*state_fab(i + 2, j, k, Idx::wr3) - 3.0/2.0*state_fab(i, j, k, Idx::wr3))/dx[0];
        amrex::Real abs_dbi10 = std::fabs(dbi10);

        if (abs_dbi10 > error_threshold) {
            return 1;
        }
        amrex::Real abs_dbi20 = std::fabs(dbi20);

        if (abs_dbi20 > error_threshold) {
            return 1;
        }
        amrex::Real abs_dbi30 = std::fabs(dbi30);

        if (abs_dbi30 > error_threshold) {
            return 1;
        }
        amrex::Real abs_dbr10 = std::fabs(dbr10);

        if (abs_dbr10 > error_threshold) {
            return 1;
        }
        amrex::Real abs_dbr20 = std::fabs(dbr20);

        if (abs_dbr20 > error_threshold) {
            return 1;
        }
        amrex::Real abs_dbr30 = std::fabs(dbr30);

        if (abs_dbr30 > error_threshold) {
            return 1;
        }
        amrex::Real abs_dli10 = std::fabs(dli10);

        if (abs_dli10 > error_threshold) {
            return 1;
        }
        amrex::Real abs_dli20 = std::fabs(dli20);

        if (abs_dli20 > error_threshold) {
            return 1;
        }
        amrex::Real abs_dli30 = std::fabs(dli30);

        if (abs_dli30 > error_threshold) {
            return 1;
        }
        amrex::Real abs_dlr10 = std::fabs(dlr10);

        if (abs_dlr10 > error_threshold) {
            return 1;
        }
        amrex::Real abs_dlr20 = std::fabs(dlr20);

        if (abs_dlr20 > error_threshold) {
            return 1;
        }
        amrex::Real abs_dlr30 = std::fabs(dlr30);

        if (abs_dlr30 > error_threshold) {
            return 1;
        }
        amrex::Real abs_dvi10 = std::fabs(dvi10);

        if (abs_dvi10 > error_threshold) {
            return 1;
        }
        amrex::Real abs_dvi20 = std::fabs(dvi20);

        if (abs_dvi20 > error_threshold) {
            return 1;
        }
        amrex::Real abs_dvi30 = std::fabs(dvi30);

        if (abs_dvi30 > error_threshold) {
            return 1;
        }
        amrex::Real abs_dvr10 = std::fabs(dvr10);

        if (abs_dvr10 > error_threshold) {
            return 1;
        }
        amrex::Real abs_dvr20 = std::fabs(dvr20);

        if (abs_dvr20 > error_threshold) {
            return 1;
        }
        amrex::Real abs_dvr30 = std::fabs(dvr30);

        if (abs_dvr30 > error_threshold) {
            return 1;
        }
        amrex::Real abs_dwi10 = std::fabs(dwi10);

        if (abs_dwi10 > error_threshold) {
            return 1;
        }
        amrex::Real abs_dwi20 = std::fabs(dwi20);

        if (abs_dwi20 > error_threshold) {
            return 1;
        }
        amrex::Real abs_dwi30 = std::fabs(dwi30);

        if (abs_dwi30 > error_threshold) {
            return 1;
        }
        amrex::Real abs_dwr10 = std::fabs(dwr10);

        if (abs_dwr10 > error_threshold) {
            return 1;
        }
        amrex::Real abs_dwr20 = std::fabs(dwr20);

        if (abs_dwr20 > error_threshold) {
            return 1;
        }
        amrex::Real abs_dwr30 = std::fabs(dwr30);

        if (abs_dwr30 > error_threshold) {
            return 1;
        }
        return 0;
}
#endif