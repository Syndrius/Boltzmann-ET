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
        // This is used to change where the grid refinement
        // first occurs, changing the location of the error
        if (i<20) {
            return 0;
        }

        const auto domain_xlo = geom.ProbLo(); 

        amrex::Real x0 = (i + 0.5)*geom.CellSize(0) + domain_xlo[0]; 
        amrex::Real x1 = (j + 0.5)*geom.CellSize(1) + domain_xlo[1]; 
        amrex::Real x2 = (k + 0.5)*geom.CellSize(2) + domain_xlo[2]; 
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

        amrex::Real abs_bi1 = std::fabs(bi1);

        if (abs_bi1 > error_threshold) {
            return 1;
        }
        amrex::Real abs_bi2 = std::fabs(bi2);

        if (abs_bi2 > error_threshold) {
            return 1;
        }
        amrex::Real abs_bi3 = std::fabs(bi3);

        if (abs_bi3 > error_threshold) {
            return 1;
        }
        amrex::Real abs_br1 = std::fabs(br1);

        if (abs_br1 > error_threshold) {
            return 1;
        }
        amrex::Real abs_br2 = std::fabs(br2);

        if (abs_br2 > error_threshold) {
            return 1;
        }
        amrex::Real abs_br3 = std::fabs(br3);

        if (abs_br3 > error_threshold) {
            return 1;
        }
        amrex::Real abs_li1 = std::fabs(li1);

        if (abs_li1 > error_threshold) {
            return 1;
        }
        amrex::Real abs_li2 = std::fabs(li2);

        if (abs_li2 > error_threshold) {
            return 1;
        }
        amrex::Real abs_li3 = std::fabs(li3);

        if (abs_li3 > error_threshold) {
            return 1;
        }
        amrex::Real abs_lr1 = std::fabs(lr1);

        if (abs_lr1 > error_threshold) {
            return 1;
        }
        amrex::Real abs_lr2 = std::fabs(lr2);

        if (abs_lr2 > error_threshold) {
            return 1;
        }
        amrex::Real abs_lr3 = std::fabs(lr3);

        if (abs_lr3 > error_threshold) {
            return 1;
        }
        amrex::Real abs_vi1 = std::fabs(vi1);

        if (abs_vi1 > error_threshold) {
            return 1;
        }
        amrex::Real abs_vi2 = std::fabs(vi2);

        if (abs_vi2 > error_threshold) {
            return 1;
        }
        amrex::Real abs_vi3 = std::fabs(vi3);

        if (abs_vi3 > error_threshold) {
            return 1;
        }
        amrex::Real abs_vr1 = std::fabs(vr1);

        if (abs_vr1 > error_threshold) {
            return 1;
        }
        amrex::Real abs_vr2 = std::fabs(vr2);

        if (abs_vr2 > error_threshold) {
            return 1;
        }
        amrex::Real abs_vr3 = std::fabs(vr3);

        if (abs_vr3 > error_threshold) {
            return 1;
        }
        amrex::Real abs_wi1 = std::fabs(wi1);

        if (abs_wi1 > error_threshold) {
            return 1;
        }
        amrex::Real abs_wi2 = std::fabs(wi2);

        if (abs_wi2 > error_threshold) {
            return 1;
        }
        amrex::Real abs_wi3 = std::fabs(wi3);

        if (abs_wi3 > error_threshold) {
            return 1;
        }
        amrex::Real abs_wr1 = std::fabs(wr1);

        if (abs_wr1 > error_threshold) {
            return 1;
        }
        amrex::Real abs_wr2 = std::fabs(wr2);

        if (abs_wr2 > error_threshold) {
            return 1;
        }
        amrex::Real abs_wr3 = std::fabs(wr3);

        if (abs_wr3 > error_threshold) {
            return 1;
        }
        return 0;
}
#endif
