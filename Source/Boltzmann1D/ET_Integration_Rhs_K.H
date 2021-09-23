#ifndef ET_INTEGRATION_RHS_K_H
#define ET_INTEGRATION_RHS_K_H

#include <AMReX_REAL.H> 
#include <AMReX_Array4.H> 
#include <ET_Integration_Setup_K.H> 

AMREX_GPU_DEVICE 
inline 
void 
state_rhs(int i, int j, int k, 
        amrex::Array4<amrex::Real> const& rhs_fab, 
        amrex::Array4<amrex::Real const> const& state_fab,
        const amrex::Real time,
        amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx, 
        const amrex::GeometryData& geom) noexcept 
{
        const auto domain_xlo = geom.ProbLo();
        
        amrex::Real x0 = (i + 0.5)*geom.CellSize(0) + domain_xlo[0]; 
        amrex::Real x1 = (j + 0.5)*geom.CellSize(1) + domain_xlo[1]; 
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
        amrex::Real Kn = state_fab(i,j,k,Idx::Kn);

        amrex::Real T = state_fab(i,j,k,Idx::T);

        amrex::Real c0 = state_fab(i,j,k,Idx::c0);

        amrex::Real c1 = state_fab(i,j,k,Idx::c1);

        amrex::Real c2 = state_fab(i,j,k,Idx::c2);

        amrex::Real c3 = state_fab(i,j,k,Idx::c3);

        amrex::Real c4 = state_fab(i,j,k,Idx::c4);

        amrex::Real c5 = state_fab(i,j,k,Idx::c5);

        amrex::Real c6 = state_fab(i,j,k,Idx::c6);

        amrex::Real c7 = state_fab(i,j,k,Idx::c7);

        amrex::Real c8 = state_fab(i,j,k,Idx::c8);

        amrex::Real c9 = state_fab(i,j,k,Idx::c9);

        amrex::Real c10 = state_fab(i,j,k,Idx::c10);

        amrex::Real c11 = state_fab(i,j,k,Idx::c11);

        amrex::Real c12 = state_fab(i,j,k,Idx::c12);

        amrex::Real c13 = state_fab(i,j,k,Idx::c13);

        amrex::Real c14 = state_fab(i,j,k,Idx::c14);

        amrex::Real c15 = state_fab(i,j,k,Idx::c15);

        amrex::Real c16 = state_fab(i,j,k,Idx::c16);

        amrex::Real c17 = state_fab(i,j,k,Idx::c17);

        amrex::Real c18 = state_fab(i,j,k,Idx::c18);

        amrex::Real c19 = state_fab(i,j,k,Idx::c19);

        amrex::Real c20 = state_fab(i,j,k,Idx::c20);

        amrex::Real c21 = state_fab(i,j,k,Idx::c21);

        amrex::Real c22 = state_fab(i,j,k,Idx::c22);

        amrex::Real c23 = state_fab(i,j,k,Idx::c23);

        amrex::Real c24 = state_fab(i,j,k,Idx::c24);

        amrex::Real c25 = state_fab(i,j,k,Idx::c25);

        amrex::Real c26 = state_fab(i,j,k,Idx::c26);

        amrex::Real c27 = state_fab(i,j,k,Idx::c27);

        amrex::Real c28 = state_fab(i,j,k,Idx::c28);

        amrex::Real c29 = state_fab(i,j,k,Idx::c29);

        amrex::Real c30 = state_fab(i,j,k,Idx::c30);

        amrex::Real c31 = state_fab(i,j,k,Idx::c31);

        amrex::Real c32 = state_fab(i,j,k,Idx::c32);

        amrex::Real c33 = state_fab(i,j,k,Idx::c33);

        amrex::Real c34 = state_fab(i,j,k,Idx::c34);

        amrex::Real c35 = state_fab(i,j,k,Idx::c35);

        amrex::Real c36 = state_fab(i,j,k,Idx::c36);

        amrex::Real c37 = state_fab(i,j,k,Idx::c37);

        amrex::Real c38 = state_fab(i,j,k,Idx::c38);

        amrex::Real c39 = state_fab(i,j,k,Idx::c39);

        amrex::Real c40 = state_fab(i,j,k,Idx::c40);

        amrex::Real c41 = state_fab(i,j,k,Idx::c41);

        amrex::Real c42 = state_fab(i,j,k,Idx::c42);

        amrex::Real c43 = state_fab(i,j,k,Idx::c43);

        amrex::Real c44 = state_fab(i,j,k,Idx::c44);

        amrex::Real c45 = state_fab(i,j,k,Idx::c45);

        amrex::Real c46 = state_fab(i,j,k,Idx::c46);

        amrex::Real c47 = state_fab(i,j,k,Idx::c47);

        amrex::Real c48 = state_fab(i,j,k,Idx::c48);

        amrex::Real c49 = state_fab(i,j,k,Idx::c49);

        amrex::Real c50 = state_fab(i,j,k,Idx::c50);

        amrex::Real c51 = state_fab(i,j,k,Idx::c51);

        amrex::Real c52 = state_fab(i,j,k,Idx::c52);

        amrex::Real c53 = state_fab(i,j,k,Idx::c53);

        amrex::Real c54 = state_fab(i,j,k,Idx::c54);

        amrex::Real c55 = state_fab(i,j,k,Idx::c55);

        amrex::Real c56 = state_fab(i,j,k,Idx::c56);

        amrex::Real c57 = state_fab(i,j,k,Idx::c57);

        amrex::Real c58 = state_fab(i,j,k,Idx::c58);

        amrex::Real c59 = state_fab(i,j,k,Idx::c59);

        amrex::Real c60 = state_fab(i,j,k,Idx::c60);

        amrex::Real c61 = state_fab(i,j,k,Idx::c61);

        amrex::Real c62 = state_fab(i,j,k,Idx::c62);

        amrex::Real c63 = state_fab(i,j,k,Idx::c63);

        amrex::Real c64 = state_fab(i,j,k,Idx::c64);

        amrex::Real c65 = state_fab(i,j,k,Idx::c65);

        amrex::Real c66 = state_fab(i,j,k,Idx::c66);

        amrex::Real c67 = state_fab(i,j,k,Idx::c67);

        amrex::Real c68 = state_fab(i,j,k,Idx::c68);

        amrex::Real c69 = state_fab(i,j,k,Idx::c69);

        amrex::Real c70 = state_fab(i,j,k,Idx::c70);

        amrex::Real c71 = state_fab(i,j,k,Idx::c71);

        amrex::Real c72 = state_fab(i,j,k,Idx::c72);

        amrex::Real c73 = state_fab(i,j,k,Idx::c73);

        amrex::Real c74 = state_fab(i,j,k,Idx::c74);

        amrex::Real c75 = state_fab(i,j,k,Idx::c75);

        amrex::Real c76 = state_fab(i,j,k,Idx::c76);

        amrex::Real c77 = state_fab(i,j,k,Idx::c77);

        amrex::Real c78 = state_fab(i,j,k,Idx::c78);

        amrex::Real c79 = state_fab(i,j,k,Idx::c79);

        amrex::Real c80 = state_fab(i,j,k,Idx::c80);

        amrex::Real c81 = state_fab(i,j,k,Idx::c81);

        amrex::Real c82 = state_fab(i,j,k,Idx::c82);

        amrex::Real c83 = state_fab(i,j,k,Idx::c83);

        amrex::Real c84 = state_fab(i,j,k,Idx::c84);

        amrex::Real g0 = state_fab(i,j,k,Idx::g0);

        amrex::Real g1 = state_fab(i,j,k,Idx::g1);

        amrex::Real g2 = state_fab(i,j,k,Idx::g2);

        amrex::Real g3 = state_fab(i,j,k,Idx::g3);

        amrex::Real g4 = state_fab(i,j,k,Idx::g4);

        amrex::Real g5 = state_fab(i,j,k,Idx::g5);

        amrex::Real g6 = state_fab(i,j,k,Idx::g6);

        amrex::Real g7 = state_fab(i,j,k,Idx::g7);

        amrex::Real g8 = state_fab(i,j,k,Idx::g8);

        amrex::Real g9 = state_fab(i,j,k,Idx::g9);

        amrex::Real g10 = state_fab(i,j,k,Idx::g10);

        amrex::Real g11 = state_fab(i,j,k,Idx::g11);

        amrex::Real g12 = state_fab(i,j,k,Idx::g12);

        amrex::Real g13 = state_fab(i,j,k,Idx::g13);

        amrex::Real q0 = state_fab(i,j,k,Idx::q0);

        amrex::Real q1 = state_fab(i,j,k,Idx::q1);

        amrex::Real q2 = state_fab(i,j,k,Idx::q2);

        amrex::Real q3 = state_fab(i,j,k,Idx::q3);

        amrex::Real q4 = state_fab(i,j,k,Idx::q4);

        amrex::Real q5 = state_fab(i,j,k,Idx::q5);

        amrex::Real q6 = state_fab(i,j,k,Idx::q6);

        amrex::Real q7 = state_fab(i,j,k,Idx::q7);

        amrex::Real q8 = state_fab(i,j,k,Idx::q8);

        amrex::Real q9 = state_fab(i,j,k,Idx::q9);

        amrex::Real q10 = state_fab(i,j,k,Idx::q10);

        amrex::Real q11 = state_fab(i,j,k,Idx::q11);

        amrex::Real q12 = state_fab(i,j,k,Idx::q12);

        amrex::Real q13 = state_fab(i,j,k,Idx::q13);

        rhs_fab(i, j, k, Idx::bi1) = -dbi10 + q0*(bi1*c59 + bi2*c60 + bi3*c31 - br1*c34 + br2*c35 + br3*c2 + c2*li3 + c2*lr1 + c2*lr2 + c2*lr3 + c2*vi3 + c2*vr1 + c2*vr2 + c2*vr3 + c2*wi3 + c2*wr1 + c2*wr2 + c2*wr3 + c32*vi1 + c36*wi1 + c55*li1 + c56*li2 + c57*wi2 + c58*vi2 - g6*q2*(bi1*c65 + bi2*c66 + bi3*c67 + br1*c44 + br2*c83 + br3*c2 + c13*vi2 + c2*li3 + c2*lr1 + c2*lr2 + c2*lr3 + c2*vi3 + c2*vr1 + c2*vr2 + c2*vr3 + c2*wi3 + c2*wr1 + c2*wr2 + c2*wr3 + c22*wi2 + c48*li2 + c62*li1 + c63*wi1 + c64*vi1 - q1*(bi1*c59 + bi2*c60 + bi3*c31 - br1*c34 + br2*c35 + br3*c2 + c2*li3 + c2*lr1 + c2*lr2 + c2*lr3 + c2*vi3 + c2*vr1 + c2*vr2 + c2*vr3 + c2*wi3 + c2*wr1 + c2*wr2 + c2*wr3 + c32*vi1 + c36*wi1 + c55*li1 + c56*li2 + c57*wi2 + c58*vi2)));

        rhs_fab(i, j, k, Idx::bi2) = -dbi20 + q2*(bi1*c65 + bi2*c66 + bi3*c67 + br1*c44 + br2*c83 + br3*c2 + c13*vi2 + c2*li3 + c2*lr1 + c2*lr2 + c2*lr3 + c2*vi3 + c2*vr1 + c2*vr2 + c2*vr3 + c2*wi3 + c2*wr1 + c2*wr2 + c2*wr3 + c22*wi2 + c48*li2 + c62*li1 + c63*wi1 + c64*vi1 - q1*(bi1*c59 + bi2*c60 + bi3*c31 - br1*c34 + br2*c35 + br3*c2 + c2*li3 + c2*lr1 + c2*lr2 + c2*lr3 + c2*vi3 + c2*vr1 + c2*vr2 + c2*vr3 + c2*wi3 + c2*wr1 + c2*wr2 + c2*wr3 + c32*vi1 + c36*wi1 + c55*li1 + c56*li2 + c57*wi2 + c58*vi2));

        rhs_fab(i, j, k, Idx::bi3) = -dbi30 + q3*(bi1*c49 + bi2*c70 + bi3*c71 + br1*c2 + br2*c2 + br3*c84 + c2*li1 + c2*li2 + c2*lr1 + c2*lr2 + c2*lr3 + c2*vi1 + c2*vi2 + c2*vr1 + c2*vr2 + c2*vr3 + c2*wi1 + c2*wi2 + c2*wr1 + c2*wr2 + c2*wr3 + c52*vi3 + c52*wi3 + c69*li3);

        rhs_fab(i, j, k, Idx::br1) = -dbr10 + q0*(bi1*c34 + bi2*c61 + bi3*c2 + br1*c59 + br2*c60 + br3*c31 + c2*li1 + c2*li2 + c2*li3 + c2*lr3 + c2*vi1 + c2*vi2 + c2*vi3 + c2*vr3 + c2*wi1 + c2*wi2 + c2*wi3 + c2*wr3 + c32*vr1 + c36*wr1 + c55*lr1 + c56*lr2 + c57*wr2 + c58*vr2 - g6*q2*(bi1*c68 + bi2*c45 + bi3*c2 + br1*c65 + br2*c66 + br3*c67 + c13*vr2 + c2*li1 + c2*li2 + c2*li3 + c2*lr3 + c2*vi1 + c2*vi2 + c2*vi3 + c2*vr3 + c2*wi1 + c2*wi2 + c2*wi3 + c2*wr3 + c22*wr2 + c48*lr2 + c62*lr1 + c63*wr1 + c64*vr1 - q1*(bi1*c34 + bi2*c61 + bi3*c2 + br1*c59 + br2*c60 + br3*c31 + c2*li1 + c2*li2 + c2*li3 + c2*lr3 + c2*vi1 + c2*vi2 + c2*vi3 + c2*vr3 + c2*wi1 + c2*wi2 + c2*wi3 + c2*wr3 + c32*vr1 + c36*wr1 + c55*lr1 + c56*lr2 + c57*wr2 + c58*vr2)));

        rhs_fab(i, j, k, Idx::br2) = -dbr20 + q2*(bi1*c68 + bi2*c45 + bi3*c2 + br1*c65 + br2*c66 + br3*c67 + c13*vr2 + c2*li1 + c2*li2 + c2*li3 + c2*lr3 + c2*vi1 + c2*vi2 + c2*vi3 + c2*vr3 + c2*wi1 + c2*wi2 + c2*wi3 + c2*wr3 + c22*wr2 + c48*lr2 + c62*lr1 + c63*wr1 + c64*vr1 - q1*(bi1*c34 + bi2*c61 + bi3*c2 + br1*c59 + br2*c60 + br3*c31 + c2*li1 + c2*li2 + c2*li3 + c2*lr3 + c2*vi1 + c2*vi2 + c2*vi3 + c2*vr3 + c2*wi1 + c2*wi2 + c2*wi3 + c2*wr3 + c32*vr1 + c36*wr1 + c55*lr1 + c56*lr2 + c57*wr2 + c58*vr2));

        rhs_fab(i, j, k, Idx::br3) = -dbr30 + q3*(bi1*c2 + bi2*c2 + bi3*c53 + br1*c49 + br2*c70 + br3*c71 + c2*li1 + c2*li2 + c2*li3 + c2*lr1 + c2*lr2 + c2*vi1 + c2*vi2 + c2*vi3 + c2*vr1 + c2*vr2 + c2*wi1 + c2*wi2 + c2*wi3 + c2*wr1 + c2*wr2 + c52*vr3 + c52*wr3 + c69*lr3);

        rhs_fab(i, j, k, Idx::li1) = dli10 - q4*(bi1*c38 + bi2*c39 + bi3*c2 + br1*c2 + br2*c2 + br3*c2 + c2*lr3 + c2*vi3 + c2*vr1 + c2*vr2 + c2*vr3 + c2*wi3 + c2*wr1 + c2*wr2 + c2*wr3 + c29*li1 + c30*li2 + c31*li3 + c32*wi1 + c33*wi2 - c34*lr1 + c36*vi1 + c37*vi2 + c61*lr2 - g6*q6*(bi1*c47 + bi2*c48 + bi3*c2 + br1*c2 + br2*c2 + br3*c2 + c13*wi2 + c2*lr3 + c2*vi3 + c2*vr1 + c2*vr2 + c2*vr3 + c2*wi3 + c2*wr1 + c2*wr2 + c2*wr3 + c22*vi2 + c40*li1 + c41*li2 + c42*li3 + c43*wi1 + c46*vi1 + c68*lr1 + c83*lr2 - q5*(bi1*c38 + bi2*c39 + bi3*c2 + br1*c2 + br2*c2 + br3*c2 + c2*lr3 + c2*vi3 + c2*vr1 + c2*vr2 + c2*vr3 + c2*wi3 + c2*wr1 + c2*wr2 + c2*wr3 + c29*li1 + c30*li2 + c31*li3 + c32*wi1 + c33*wi2 - c34*lr1 + c36*vi1 + c37*vi2 + c61*lr2)));

        rhs_fab(i, j, k, Idx::li2) = dli20 - q6*(bi1*c47 + bi2*c48 + bi3*c2 + br1*c2 + br2*c2 + br3*c2 + c13*wi2 + c2*lr3 + c2*vi3 + c2*vr1 + c2*vr2 + c2*vr3 + c2*wi3 + c2*wr1 + c2*wr2 + c2*wr3 + c22*vi2 + c40*li1 + c41*li2 + c42*li3 + c43*wi1 + c46*vi1 + c68*lr1 + c83*lr2 - q5*(bi1*c38 + bi2*c39 + bi3*c2 + br1*c2 + br2*c2 + br3*c2 + c2*lr3 + c2*vi3 + c2*vr1 + c2*vr2 + c2*vr3 + c2*wi3 + c2*wr1 + c2*wr2 + c2*wr3 + c29*li1 + c30*li2 + c31*li3 + c32*wi1 + c33*wi2 - c34*lr1 + c36*vi1 + c37*vi2 + c61*lr2));

        rhs_fab(i, j, k, Idx::li3) = dli30 - q7*(bi1*c2 + bi2*c2 + bi3*c54 + br1*c2 + br2*c2 + br3*c2 + c2*lr1 + c2*lr2 + c2*vi1 + c2*vi2 + c2*vr1 + c2*vr2 + c2*vr3 + c2*wi1 + c2*wi2 + c2*wr1 + c2*wr2 + c2*wr3 + c49*li1 + c50*li2 + c51*li3 + c52*vi3 + c52*wi3 + c84*lr3);

        rhs_fab(i, j, k, Idx::lr1) = dlr10 - q4*(bi1*c2 + bi2*c2 + bi3*c2 + br1*c38 + br2*c39 + br3*c2 + c2*li3 + c2*vi1 + c2*vi2 + c2*vi3 + c2*vr3 + c2*wi1 + c2*wi2 + c2*wi3 + c2*wr3 + c29*lr1 + c30*lr2 + c31*lr3 + c32*wr1 + c33*wr2 + c34*li1 + c35*li2 + c36*vr1 + c37*vr2 - g6*q6*(bi1*c2 + bi2*c2 + bi3*c2 + br1*c47 + br2*c48 + br3*c2 + c13*wr2 + c2*li3 + c2*vi1 + c2*vi2 + c2*vi3 + c2*vr3 + c2*wi1 + c2*wi2 + c2*wi3 + c2*wr3 + c22*vr2 + c40*lr1 + c41*lr2 + c42*lr3 + c43*wr1 + c44*li1 + c45*li2 + c46*vr1 - q5*(bi1*c2 + bi2*c2 + bi3*c2 + br1*c38 + br2*c39 + br3*c2 + c2*li3 + c2*vi1 + c2*vi2 + c2*vi3 + c2*vr3 + c2*wi1 + c2*wi2 + c2*wi3 + c2*wr3 + c29*lr1 + c30*lr2 + c31*lr3 + c32*wr1 + c33*wr2 + c34*li1 + c35*li2 + c36*vr1 + c37*vr2)));

        rhs_fab(i, j, k, Idx::lr2) = dlr20 - q6*(bi1*c2 + bi2*c2 + bi3*c2 + br1*c47 + br2*c48 + br3*c2 + c13*wr2 + c2*li3 + c2*vi1 + c2*vi2 + c2*vi3 + c2*vr3 + c2*wi1 + c2*wi2 + c2*wi3 + c2*wr3 + c22*vr2 + c40*lr1 + c41*lr2 + c42*lr3 + c43*wr1 + c44*li1 + c45*li2 + c46*vr1 - q5*(bi1*c2 + bi2*c2 + bi3*c2 + br1*c38 + br2*c39 + br3*c2 + c2*li3 + c2*vi1 + c2*vi2 + c2*vi3 + c2*vr3 + c2*wi1 + c2*wi2 + c2*wi3 + c2*wr3 + c29*lr1 + c30*lr2 + c31*lr3 + c32*wr1 + c33*wr2 + c34*li1 + c35*li2 + c36*vr1 + c37*vr2));

        rhs_fab(i, j, k, Idx::lr3) = dlr30 - q7*(bi1*c2 + bi2*c2 + bi3*c2 + br1*c2 + br2*c2 + br3*c54 + c2*li1 + c2*li2 + c2*vi1 + c2*vi2 + c2*vi3 + c2*vr1 + c2*vr2 + c2*wi1 + c2*wi2 + c2*wi3 + c2*wr1 + c2*wr2 + c49*lr1 + c50*lr2 + c51*lr3 + c52*vr3 + c52*wr3 + c53*li3);

        rhs_fab(i, j, k, Idx::vi1) = -dvi10 - q8*(bi1*c0 + bi2*c74 + bi3*c2 + br1*c2 + br2*c2 + br3*c2 + c10*li1 + c2*li3 + c2*lr1 + c2*lr2 + c2*lr3 + c2*vr3 + c2*wi3 + c2*wr1 + c2*wr2 + c2*wr3 + c3*vi1 + c5*vi3 + c7*vr2 + c72*li2 + c73*vi2 + c8*wi1 + c80*vr1 - c9*wi2 - g1*q11*(bi1*c78 + bi2*c13 + bi3*c2 + br1*c2 + br2*c2 + br3*c2 - c16*vi3 + c17*vr1 - c19*wi1 + c2*li3 + c2*lr1 + c2*lr2 + c2*lr3 + c2*vr3 + c2*wi3 + c2*wr1 + c2*wr2 + c2*wr3 + c20*wi2 + c22*li2 + c75*li1 + c76*vi1 + c77*vi2 + c81*vr2 + q9*(bi1*c0 + bi2*c74 + bi3*c2 + br1*c2 + br2*c2 + br3*c2 + c10*li1 + c2*li3 + c2*lr1 + c2*lr2 + c2*lr3 + c2*vr3 + c2*wi3 + c2*wr1 + c2*wr2 + c2*wr3 + c3*vi1 + c5*vi3 + c7*vr2 + c72*li2 + c73*vi2 + c8*wi1 + c80*vr1 - c9*wi2)));

        rhs_fab(i, j, k, Idx::vi2) = -dvi20 + q11*(bi1*c78 + bi2*c13 + bi3*c2 + br1*c2 + br2*c2 + br3*c2 - c16*vi3 + c17*vr1 - c19*wi1 + c2*li3 + c2*lr1 + c2*lr2 + c2*lr3 + c2*vr3 + c2*wi3 + c2*wr1 + c2*wr2 + c2*wr3 + c20*wi2 + c22*li2 + c75*li1 + c76*vi1 + c77*vi2 + c81*vr2 + q9*(bi1*c0 + bi2*c74 + bi3*c2 + br1*c2 + br2*c2 + br3*c2 + c10*li1 + c2*li3 + c2*lr1 + c2*lr2 + c2*lr3 + c2*vr3 + c2*wi3 + c2*wr1 + c2*wr2 + c2*wr3 + c3*vi1 + c5*vi3 + c7*vr2 + c72*li2 + c73*vi2 + c8*wi1 + c80*vr1 - c9*wi2));

        rhs_fab(i, j, k, Idx::vi3) = -dvi30 - q12*(bi1*c2 + bi2*c2 + bi3*c23 + br1*c2 + br2*c2 + br3*c2 + c2*li1 + c2*li2 + c2*lr1 + c2*lr2 + c2*lr3 + c2*vr1 + c2*vr2 + c2*wi1 + c2*wi2 + c2*wr1 + c2*wr2 + c2*wr3 + c23*li3 + c24*vi1 - c25*vi2 + c28*wi3 + c79*vi3 + c82*vr3);

        rhs_fab(i, j, k, Idx::vr1) = -dvr10 - q8*(bi1*c2 + bi2*c2 + bi3*c2 + br1*c0 + br2*c74 + br3*c2 + c10*lr1 + c2*li1 + c2*li2 + c2*li3 + c2*lr3 + c2*vi3 + c2*wi1 + c2*wi2 + c2*wi3 + c2*wr3 + c3*vr1 + c5*vr3 + c6*vi1 - c7*vi2 + c72*lr2 + c73*vr2 + c8*wr1 - c9*wr2 - g1*q11*(bi1*c2 + bi2*c2 + bi3*c2 + br1*c78 + br2*c13 + br3*c2 - c16*vr3 - c17*vi1 + c18*vi2 - c19*wr1 + c2*li1 + c2*li2 + c2*li3 + c2*lr3 + c2*vi3 + c2*wi1 + c2*wi2 + c2*wi3 + c2*wr3 + c20*wr2 + c22*lr2 + c75*lr1 + c76*vr1 + c77*vr2 + q9*(bi1*c2 + bi2*c2 + bi3*c2 + br1*c0 + br2*c74 + br3*c2 + c10*lr1 + c2*li1 + c2*li2 + c2*li3 + c2*lr3 + c2*vi3 + c2*wi1 + c2*wi2 + c2*wi3 + c2*wr3 + c3*vr1 + c5*vr3 + c6*vi1 - c7*vi2 + c72*lr2 + c73*vr2 + c8*wr1 - c9*wr2)));

        rhs_fab(i, j, k, Idx::vr2) = -dvr20 + q11*(bi1*c2 + bi2*c2 + bi3*c2 + br1*c78 + br2*c13 + br3*c2 - c16*vr3 - c17*vi1 + c18*vi2 - c19*wr1 + c2*li1 + c2*li2 + c2*li3 + c2*lr3 + c2*vi3 + c2*wi1 + c2*wi2 + c2*wi3 + c2*wr3 + c20*wr2 + c22*lr2 + c75*lr1 + c76*vr1 + c77*vr2 + q9*(bi1*c2 + bi2*c2 + bi3*c2 + br1*c0 + br2*c74 + br3*c2 + c10*lr1 + c2*li1 + c2*li2 + c2*li3 + c2*lr3 + c2*vi3 + c2*wi1 + c2*wi2 + c2*wi3 + c2*wr3 + c3*vr1 + c5*vr3 + c6*vi1 - c7*vi2 + c72*lr2 + c73*vr2 + c8*wr1 - c9*wr2));

        rhs_fab(i, j, k, Idx::vr3) = -dvr30 - q12*(bi1*c2 + bi2*c2 + bi3*c2 + br1*c2 + br2*c2 + br3*c23 + c2*li1 + c2*li2 + c2*li3 + c2*lr1 + c2*lr2 + c2*vi1 + c2*vi2 + c2*wi1 + c2*wi2 + c2*wi3 + c2*wr1 + c2*wr2 + c23*lr3 + c24*vr1 - c25*vr2 + c27*vi3 + c28*wr3 + c79*vr3);

        rhs_fab(i, j, k, Idx::wi1) = dwi10 - q8*(bi1*c10 + bi2*c11 + bi3*c2 + br1*c2 + br2*c2 + br3*c2 + c0*li1 + c1*li2 + c2*li3 + c2*lr1 + c2*lr2 + c2*lr3 + c2*vi3 + c2*vr1 + c2*vr2 + c2*vr3 + c2*wr3 + c3*wi1 + c4*wi2 + c5*wi3 - c7*wr2 + c8*vi1 + c80*wr1 + c9*vi2 - g1*q13*(bi1*c21 + bi2*c22 + bi3*c2 + br1*c2 + br2*c2 + br3*c2 + c12*li1 + c13*li2 + c14*wi1 + c15*wi2 + c16*wi3 - c17*wr1 + c19*vi1 + c2*li3 + c2*lr1 + c2*lr2 + c2*lr3 + c2*vi3 + c2*vr1 + c2*vr2 + c2*vr3 + c2*wr3 + c20*vi2 + c81*wr2 - q9*(bi1*c10 + bi2*c11 + bi3*c2 + br1*c2 + br2*c2 + br3*c2 + c0*li1 + c1*li2 + c2*li3 + c2*lr1 + c2*lr2 + c2*lr3 + c2*vi3 + c2*vr1 + c2*vr2 + c2*vr3 + c2*wr3 + c3*wi1 + c4*wi2 + c5*wi3 - c7*wr2 + c8*vi1 + c80*wr1 + c9*vi2)));

        rhs_fab(i, j, k, Idx::wi2) = dwi20 - q13*(bi1*c21 + bi2*c22 + bi3*c2 + br1*c2 + br2*c2 + br3*c2 + c12*li1 + c13*li2 + c14*wi1 + c15*wi2 + c16*wi3 - c17*wr1 + c19*vi1 + c2*li3 + c2*lr1 + c2*lr2 + c2*lr3 + c2*vi3 + c2*vr1 + c2*vr2 + c2*vr3 + c2*wr3 + c20*vi2 + c81*wr2 - q9*(bi1*c10 + bi2*c11 + bi3*c2 + br1*c2 + br2*c2 + br3*c2 + c0*li1 + c1*li2 + c2*li3 + c2*lr1 + c2*lr2 + c2*lr3 + c2*vi3 + c2*vr1 + c2*vr2 + c2*vr3 + c2*wr3 + c3*wi1 + c4*wi2 + c5*wi3 - c7*wr2 + c8*vi1 + c80*wr1 + c9*vi2));

        rhs_fab(i, j, k, Idx::wi3) = dwi30 - q12*(bi1*c2 + bi2*c2 + bi3*c23 + br1*c2 + br2*c2 + br3*c2 + c2*li1 + c2*li2 + c2*lr1 + c2*lr2 + c2*lr3 + c2*vi1 + c2*vi2 + c2*vr1 + c2*vr2 + c2*vr3 + c2*wr1 + c2*wr2 + c23*li3 + c24*wi1 + c25*wi2 + c26*wi3 + c28*vi3 + c82*wr3);

        rhs_fab(i, j, k, Idx::wr1) = dwr10 - q8*(bi1*c2 + bi2*c2 + bi3*c2 + br1*c10 + br2*c11 + br3*c2 + c0*lr1 + c1*lr2 + c2*li1 + c2*li2 + c2*li3 + c2*lr3 + c2*vi1 + c2*vi2 + c2*vi3 + c2*vr3 + c2*wi3 + c3*wr1 + c4*wr2 + c5*wr3 + c6*wi1 + c7*wi2 + c8*vr1 + c9*vr2 - g1*q13*(bi1*c2 + bi2*c2 + bi3*c2 + br1*c21 + br2*c22 + br3*c2 + c12*lr1 + c13*lr2 + c14*wr1 + c15*wr2 + c16*wr3 + c17*wi1 + c18*wi2 + c19*vr1 + c2*li1 + c2*li2 + c2*li3 + c2*lr3 + c2*vi1 + c2*vi2 + c2*vi3 + c2*vr3 + c2*wi3 + c20*vr2 - q9*(bi1*c2 + bi2*c2 + bi3*c2 + br1*c10 + br2*c11 + br3*c2 + c0*lr1 + c1*lr2 + c2*li1 + c2*li2 + c2*li3 + c2*lr3 + c2*vi1 + c2*vi2 + c2*vi3 + c2*vr3 + c2*wi3 + c3*wr1 + c4*wr2 + c5*wr3 + c6*wi1 + c7*wi2 + c8*vr1 + c9*vr2)));

        rhs_fab(i, j, k, Idx::wr2) = dwr20 - q13*(bi1*c2 + bi2*c2 + bi3*c2 + br1*c21 + br2*c22 + br3*c2 + c12*lr1 + c13*lr2 + c14*wr1 + c15*wr2 + c16*wr3 + c17*wi1 + c18*wi2 + c19*vr1 + c2*li1 + c2*li2 + c2*li3 + c2*lr3 + c2*vi1 + c2*vi2 + c2*vi3 + c2*vr3 + c2*wi3 + c20*vr2 - q9*(bi1*c2 + bi2*c2 + bi3*c2 + br1*c10 + br2*c11 + br3*c2 + c0*lr1 + c1*lr2 + c2*li1 + c2*li2 + c2*li3 + c2*lr3 + c2*vi1 + c2*vi2 + c2*vi3 + c2*vr3 + c2*wi3 + c3*wr1 + c4*wr2 + c5*wr3 + c6*wi1 + c7*wi2 + c8*vr1 + c9*vr2));

        rhs_fab(i, j, k, Idx::wr3) = dwr30 - q12*(bi1*c2 + bi2*c2 + bi3*c2 + br1*c2 + br2*c2 + br3*c23 + c2*li1 + c2*li2 + c2*li3 + c2*lr1 + c2*lr2 + c2*vi1 + c2*vi2 + c2*vi3 + c2*vr1 + c2*vr2 + c2*wi1 + c2*wi2 + c23*lr3 + c24*wr1 + c25*wr2 + c26*wr3 + c27*wi3 + c28*vr3);

}
#endif