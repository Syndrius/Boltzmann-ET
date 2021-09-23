
#include <bc_fill.H>
#include "ET_Integration.H"

using namespace amrex;

void AmrCoreFillCpu (Box const& bx, Array4<Real> const& data,
                     const int dcomp, const int numcomp,
                     GeometryData const& geom, const Real time,
                     const BCRec* bcr, const int bcomp,
                     const int orig_comp)
{
    // do something for external Dirichlet (BCType::ext_dir)
    const auto& domain = geom.Domain();
    const auto& dom_lo = amrex::lbound(domain);
    const auto& dom_hi = amrex::ubound(domain);

    Box dom(geom.Domain());

    int lox = dom.smallEnd(0);
    int hix = dom.bigEnd(0);
    int loy = dom.smallEnd(1);
    int hiy = dom.bigEnd(1);


    ParallelFor(bx, numcomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int mcomp) {
    
        if(i == lox-1)
        {
            data(i,j,k,Idx::vr1) = -2*data(i+1,j,k,Idx::vr1)+1.0/3*data(i+2,j,k,Idx::vr1)+8.0/3*(-2*(-1.0/2.0*M_PI*(15.0/8*data(i+1,j,k,Idx::wr1) - 5.0/4*data(i+2,j,k,Idx::wr1) + 3.0/8*data(i+3,j,k,Idx::wr1)) + (1.0/4.0)*std::pow(M_PI, 3.0/2.0)*(15.0/8*data(i+1,j,k,Idx::wr2) - 5.0/4*data(i+2,j,k,Idx::wr2) + 3.0/8*data(i+3,j,k,Idx::wr2)))/M_PI + std::sqrt(M_PI));
            data(i,j,k,Idx::vr2) = -2*data(i+1,j,k,Idx::vr2)+1.0/3*data(i+2,j,k,Idx::vr2)+8.0/3*(2);
            data(i,j,k,Idx::vr3) = -2*data(i+1,j,k,Idx::vr3)+1.0/3*data(i+2,j,k,Idx::vr3)+8.0/3*(-2);
            data(i,j,k,Idx::br1) = -2*data(i+1,j,k,Idx::br1)+1.0/3*data(i+2,j,k,Idx::br1)+8.0/3*(0);
            data(i,j,k,Idx::br2) = -2*data(i+1,j,k,Idx::br2)+1.0/3*data(i+2,j,k,Idx::br2)+8.0/3*(0);
            data(i,j,k,Idx::br3) = -2*data(i+1,j,k,Idx::br3)+1.0/3*data(i+2,j,k,Idx::br3)+8.0/3*(0);
            data(i,j,k,Idx::vi1) = -2*data(i+1,j,k,Idx::vi1)+1.0/3*data(i+2,j,k,Idx::vi1)+8.0/3*(-2*(-1.0/2.0*M_PI*(15.0/8*data(i+1,j,k,Idx::wi1) - 5.0/4*data(i+2,j,k,Idx::wi1) + 3.0/8*data(i+3,j,k,Idx::wi1)) + (1.0/4.0)*std::pow(M_PI, 3.0/2.0)*(15.0/8*data(i+1,j,k,Idx::wi2) - 5.0/4*data(i+2,j,k,Idx::wi2) + 3.0/8*data(i+3,j,k,Idx::wi2)))/M_PI);
            data(i,j,k,Idx::vi2) = -2*data(i+1,j,k,Idx::vi2)+1.0/3*data(i+2,j,k,Idx::vi2)+8.0/3*(0);
            data(i,j,k,Idx::vi3) = -2*data(i+1,j,k,Idx::vi3)+1.0/3*data(i+2,j,k,Idx::vi3)+8.0/3*(0);
            data(i,j,k,Idx::bi1) = -2*data(i+1,j,k,Idx::bi1)+1.0/3*data(i+2,j,k,Idx::bi1)+8.0/3*(0);
            data(i,j,k,Idx::bi2) = -2*data(i+1,j,k,Idx::bi2)+1.0/3*data(i+2,j,k,Idx::bi2)+8.0/3*(0);
            data(i,j,k,Idx::bi3) = -2*data(i+1,j,k,Idx::bi3)+1.0/3*data(i+2,j,k,Idx::bi3)+8.0/3*(0);
        }else if(i == lox-2)
        {
            data(i,j,k,Idx::vr1) = 3*data(i+2,j,k,Idx::vr1)+6*data(i+1,j,k,Idx::vr1)-8*(-2*(-1.0/2.0*M_PI*(15.0/8*data(i+2,j,k,Idx::wr1) - 5.0/4*data(i+3,j,k,Idx::wr1) + 3.0/8*data(i+4,j,k,Idx::wr1)) + (1.0/4.0)*std::pow(M_PI, 3.0/2.0)*(15.0/8*data(i+2,j,k,Idx::wr2) - 5.0/4*data(i+3,j,k,Idx::wr2) + 3.0/8*data(i+4,j,k,Idx::wr2)))/M_PI + std::sqrt(M_PI));
            data(i,j,k,Idx::vr2) = 3*data(i+2,j,k,Idx::vr2)+6*data(i+1,j,k,Idx::vr2)-8*(2);
            data(i,j,k,Idx::vr3) = 3*data(i+2,j,k,Idx::vr3)+6*data(i+1,j,k,Idx::vr3)-8*(-2);
            data(i,j,k,Idx::br1) = 3*data(i+2,j,k,Idx::br1)+6*data(i+1,j,k,Idx::br1)-8*(0);
            data(i,j,k,Idx::br2) = 3*data(i+2,j,k,Idx::br2)+6*data(i+1,j,k,Idx::br2)-8*(0);
            data(i,j,k,Idx::br3) = 3*data(i+2,j,k,Idx::br3)+6*data(i+1,j,k,Idx::br3)-8*(0);
            data(i,j,k,Idx::vi1) = 3*data(i+2,j,k,Idx::vi1)+6*data(i+1,j,k,Idx::vi1)-8*(-2*(-1.0/2.0*M_PI*(15.0/8*data(i+2,j,k,Idx::wi1) - 5.0/4*data(i+3,j,k,Idx::wi1) + 3.0/8*data(i+4,j,k,Idx::wi1)) + (1.0/4.0)*std::pow(M_PI, 3.0/2.0)*(15.0/8*data(i+2,j,k,Idx::wi2) - 5.0/4*data(i+3,j,k,Idx::wi2) + 3.0/8*data(i+4,j,k,Idx::wi2)))/M_PI);
            data(i,j,k,Idx::vi2) = 3*data(i+2,j,k,Idx::vi2)+6*data(i+1,j,k,Idx::vi2)-8*(0);
            data(i,j,k,Idx::vi3) = 3*data(i+2,j,k,Idx::vi3)+6*data(i+1,j,k,Idx::vi3)-8*(0);
            data(i,j,k,Idx::bi1) = 3*data(i+2,j,k,Idx::bi1)+6*data(i+1,j,k,Idx::bi1)-8*(0);
            data(i,j,k,Idx::bi2) = 3*data(i+2,j,k,Idx::bi2)+6*data(i+1,j,k,Idx::bi2)-8*(0);
            data(i,j,k,Idx::bi3) = 3*data(i+2,j,k,Idx::bi3)+6*data(i+1,j,k,Idx::bi3)-8*(0);
        }else if(i == hix+1)
        {
            data(i,j,k,Idx::lr1) = -2*data(i-1,j,k,Idx::lr1)+1.0/3*data(i-2,j,k,Idx::lr1)+8.0/3*(0);
            data(i,j,k,Idx::lr2) = -2*data(i-1,j,k,Idx::lr2)+1.0/3*data(i-2,j,k,Idx::lr2)+8.0/3*(0);
            data(i,j,k,Idx::lr3) = -2*data(i-1,j,k,Idx::lr3)+1.0/3*data(i-2,j,k,Idx::lr3)+8.0/3*(0);
            data(i,j,k,Idx::wr1) = -2*data(i-1,j,k,Idx::wr1)+1.0/3*data(i-2,j,k,Idx::wr1)+8.0/3*(0);
            data(i,j,k,Idx::wr2) = -2*data(i-1,j,k,Idx::wr2)+1.0/3*data(i-2,j,k,Idx::wr2)+8.0/3*(0);
            data(i,j,k,Idx::wr3) = -2*data(i-1,j,k,Idx::wr3)+1.0/3*data(i-2,j,k,Idx::wr3)+8.0/3*(0);
            data(i,j,k,Idx::li1) = -2*data(i-1,j,k,Idx::li1)+1.0/3*data(i-2,j,k,Idx::li1)+8.0/3*(0);
            data(i,j,k,Idx::li2) = -2*data(i-1,j,k,Idx::li2)+1.0/3*data(i-2,j,k,Idx::li2)+8.0/3*(0);
            data(i,j,k,Idx::li3) = -2*data(i-1,j,k,Idx::li3)+1.0/3*data(i-2,j,k,Idx::li3)+8.0/3*(0);
            data(i,j,k,Idx::wi1) = -2*data(i-1,j,k,Idx::wi1)+1.0/3*data(i-2,j,k,Idx::wi1)+8.0/3*(0);
            data(i,j,k,Idx::wi2) = -2*data(i-1,j,k,Idx::wi2)+1.0/3*data(i-2,j,k,Idx::wi2)+8.0/3*(0);
            data(i,j,k,Idx::wi3) = -2*data(i-1,j,k,Idx::wi3)+1.0/3*data(i-2,j,k,Idx::wi3)+8.0/3*(0);
        }else if(i == hix+2)
        {
            data(i,j,k,Idx::lr1) = 3*data(i-2,j,k,Idx::lr1)+6*data(i-1,j,k,Idx::lr1)-8*(0);
            data(i,j,k,Idx::lr2) = 3*data(i-2,j,k,Idx::lr2)+6*data(i-1,j,k,Idx::lr2)-8*(0);
            data(i,j,k,Idx::lr3) = 3*data(i-2,j,k,Idx::lr3)+6*data(i-1,j,k,Idx::lr3)-8*(0);
            data(i,j,k,Idx::wr1) = 3*data(i-2,j,k,Idx::wr1)+6*data(i-1,j,k,Idx::wr1)-8*(0);
            data(i,j,k,Idx::wr2) = 3*data(i-2,j,k,Idx::wr2)+6*data(i-1,j,k,Idx::wr2)-8*(0);
            data(i,j,k,Idx::wr3) = 3*data(i-2,j,k,Idx::wr3)+6*data(i-1,j,k,Idx::wr3)-8*(0);
            data(i,j,k,Idx::li1) = 3*data(i-2,j,k,Idx::li1)+6*data(i-1,j,k,Idx::li1)-8*(0);
            data(i,j,k,Idx::li2) = 3*data(i-2,j,k,Idx::li2)+6*data(i-1,j,k,Idx::li2)-8*(0);
            data(i,j,k,Idx::li3) = 3*data(i-2,j,k,Idx::li3)+6*data(i-1,j,k,Idx::li3)-8*(0);
            data(i,j,k,Idx::wi1) = 3*data(i-2,j,k,Idx::wi1)+6*data(i-1,j,k,Idx::wi1)-8*(0);
            data(i,j,k,Idx::wi2) = 3*data(i-2,j,k,Idx::wi2)+6*data(i-1,j,k,Idx::wi2)-8*(0);
            data(i,j,k,Idx::wi3) = 3*data(i-2,j,k,Idx::wi3)+6*data(i-1,j,k,Idx::wi3)-8*(0);
        }
    });
}
