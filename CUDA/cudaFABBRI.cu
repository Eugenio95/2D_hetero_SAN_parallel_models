///////////////////// code implemented by Chiara Campana ---- last update 07/22/20 ///////////////////////////////

// SAN Fabbri et al. tissue model 
// This program implements 1D and 2D models of SAN cell 
// Fabbri A, Fantini M, Wilders R, Severi S. 
// Computational analysis of the human sinus node action potential: model development and effects of mutations. 
// J Physiol. 2017;595(7):2365-2396. doi:10.1113/JP273259

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <fstream>
#include <string>
#include <stdio.h>
#include <sstream>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <cstdlib>
#include <errno.h>
#include <fstream>
#include <time.h>
#include <curand_kernel.h>

using namespace std;
#define M_PI 3.14159265358979323846  /* pi */


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////// max number of cells per block = 500, if more than 500 cells then change cellsThread and nSIMS /////////////////////////////////////
#define nSIMS 10//10			// number of Blocks = gridSize
#define nCELLS 2500		// = number of cells in a fiber/tissue 
#define cellsThread 250 //number of threads per block
#define sim2D 1	// set 0 for SC simulations, 1 for 2D tissue simulations
#define WIDTH 50
#define LENGTH 50		//must be length*width = nCELLS for 2D tissue simulations
#define nSTATES 33		// make sure it correponds to number of state variables in struct cell = states
#define saveCURRENTS 0
///////////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ double updateSingleState(double oldval, double cval, double tstep) //this function doesn't need to be global 
{
	if (!isinf(oldval + cval * tstep) && !isnan(oldval + cval * tstep)) {
		return oldval + cval*tstep;
	}
	else { return oldval ; }
}

typedef struct
{//define here the state variables
	double V, Casub, Nai, y, m, h, dL, fL, fCa, dT, fT, R, O, I, RI;
	double Cajsr, Cansr, Cai, fTMM, fCMi, fCMs, fTC, fTMC, fCQ;
	double rkur, skur, q, r, paS, paF, piy, n, a;
} cell;
typedef struct
{
	// all parameters to be varied to include heterogeneity should be part of this struct
	double P_CaL, P_CaT, g_KACh, g_Kr, g_Ks;
	double g_Kur, g_Na, g_f, g_to, i_NaK_max, K_NaCa;
} cellGs;
//__global__ void assignHeterogeneity(cellGs* Gs, float* gChannels) {
__global__ void assignHeterogeneity(cellGs* Gs, float* gChannels) {

		int idx = blockIdx.x*blockDim.x+threadIdx.x;
		Gs[idx].P_CaL = gChannels[idx + 0 * nCELLS];   // nanoA_per_millimolar(in i_CaL)
		Gs[idx].P_CaT = gChannels[idx + 1 * nCELLS];   // nanoA_per_millimolar(in i_CaT)
		Gs[idx].g_KACh = gChannels[idx + 2 * nCELLS];   // microS(in i_KACh)
		Gs[idx].g_Kr = gChannels[idx + 3 * nCELLS];   // microS(in i_Kr)
		Gs[idx].g_Ks = gChannels[idx + 4 * nCELLS];   // microS(in i_Ks)
		Gs[idx].g_Kur = gChannels[idx + 5 * nCELLS];   // microS(in i_Kur)
		Gs[idx].g_Na = gChannels[idx + 6 * nCELLS];   // microS(in i_Na)
		Gs[idx].g_f = gChannels[idx + 7 * nCELLS];   // microS(in i_f)
		Gs[idx].g_to = gChannels[idx + 8 * nCELLS];   // microS(in i_to)
		Gs[idx].i_NaK_max = gChannels[idx + 9 * nCELLS];   // nanoA(in i_NaK)
		Gs[idx].K_NaCa = gChannels[idx + 10 * nCELLS];   // nanoA(in i_NaCa)
		//printf("Gs[idx].P_CaL=%e\n", Gs[idx].P_CaL);
		//printf("Gs[idx].K_NaCa=%e\n", Gs[idx].K_NaCa);	
//	if (idx>=0&&idx<=5){
//	printf("blockIdx = %d, idx=%d, Gs[idx].K_NaCa=%f, Gs[idx].i_NaK_max=%e \n ", blockIdx.x,idx,Gs[idx].K_NaCa, Gs[idx].i_NaK_max);

//	}
	
	
}


//__global__ void initialConditions(cell* vars, float* uploadICs) {
__global__ void initialConditions(cell* vars, float* uploadICs) {

	//int idx = cellsThread*threadIdx.x;
	//printf("threadIdx=%d\n", threadIdx.x);
	//int maxIdx = idx + cellsThread;
	//for (;idx < maxIdx;idx++) {
	//	printf("idx=%d\n", idx);
//      
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
        vars[idx].I = uploadICs[idx+0*nCELLS];
        vars[idx].O = uploadICs[idx+1*nCELLS];
		vars[idx].R = uploadICs[idx+2*nCELLS]; //R1
        vars[idx].RI = uploadICs[idx+3*nCELLS];
        vars[idx].fCMi = uploadICs[idx+4*nCELLS];
        vars[idx].fCMs = uploadICs[idx+5*nCELLS];
        vars[idx].fCQ = uploadICs[idx+6*nCELLS];
        vars[idx].fTC = uploadICs[idx+7*nCELLS];
		vars[idx].fTMC = uploadICs[idx+8*nCELLS];
        vars[idx].fTMM = uploadICs[idx+9*nCELLS];
 //'Ca_jsr', 'Ca_nsr', 'Ca_sub', 'Cai', 'V_ode', 'Nai_', 'dL', 'fCa', 'fL',
        vars[idx].Cajsr = uploadICs[idx+10*nCELLS];
		vars[idx].Cansr = uploadICs[idx+11*nCELLS]; 
        vars[idx].Casub = uploadICs[idx+12*nCELLS]; 
        vars[idx].Cai = uploadICs[idx+13*nCELLS]; 
         
        vars[idx].V = uploadICs[idx+14*nCELLS];
		
		vars[idx].Nai = uploadICs[idx+15*nCELLS];
        vars[idx].dL = uploadICs[idx+16*nCELLS];
        vars[idx].fCa = uploadICs[idx+17*nCELLS];
        vars[idx].fL = uploadICs[idx+18*nCELLS];
//  'dT', 'fT', 'a', 'paF', 'paS', 'piy', 'n', 'r_Kur', 's_Kur', 'h', 'm', 'y', 'q', 'r'};
		vars[idx].dT = uploadICs[idx+19*nCELLS];
		vars[idx].fT = uploadICs[idx+20*nCELLS];
		vars[idx].a = uploadICs[idx+21*nCELLS];
		vars[idx].paF = uploadICs[idx+22*nCELLS];
		vars[idx].paS = uploadICs[idx+23*nCELLS];
        vars[idx].piy = uploadICs[idx+24*nCELLS];
		vars[idx].n = uploadICs[idx+25*nCELLS];
        vars[idx].rkur = uploadICs[idx+26*nCELLS];
		vars[idx].skur = uploadICs[idx+27*nCELLS];
        vars[idx].h = uploadICs[idx+28*nCELLS];
        vars[idx].m = uploadICs[idx+29*nCELLS];
		vars[idx].y = uploadICs[idx+30*nCELLS];
		vars[idx].q = uploadICs[idx+31*nCELLS];
		vars[idx].r = uploadICs[idx+32*nCELLS];
	//}
	//	if (idx>=0 && idx<=5) {
	//		printf("idx=%d, blockIdx=%d,vars[idx].r=%e,vars[idx].I=%e,vars[idx].Nai=%e\n", idx, blockIdx.x, vars[idx].r, vars[idx].I,vars[idx].Nai);
	//	}
	
}





__global__ void computeState(cell* x, double* ion_current, double step, cell* x_temp, cellGs* Gs) {

	//int idx = cellsThread*threadIdx.x;
	//int maxIdx = idx + cellsThread;
	int cell_num;
	//printf("maxIdx=%d\n", cell_num);
	//for (;idx < maxIdx;idx++) {
		int idx = blockIdx.x*blockDim.x + threadIdx.x;
		cell_num =  idx;//(blockIdx.x * blockDim.x) +

	//	printf("idx=%d,cellnum=%d\n", idx, cell_num);
		//int	cell_num = (blockIdx.x * blockDim.x) + threadIdx.x;
		//int idx = blockIdx.x * blockDim.x + threadIdx.x;
		//printf("cell_num=%d\n", cell_num);
		//printf("maxIdx=%d\n", time);
		/////////////////// define here all constants ///////////////////////////////////

		double EC50_SR = 0.45;   // millimolar(in Ca_SR_release)
		double HSR = 2.5;   // dimensionless(in Ca_SR_release)
		double MaxSR = 15.0;   // dimensionless(in Ca_SR_release)
		double MinSR = 1.0;   // dimensionless(in Ca_SR_release)
		double kiCa = 500.0;   // per_millimolar_second(in Ca_SR_release)
		double kim = 5.0;   // per_second(in Ca_SR_release)
		double koCa = 10000.0;   // per_millimolar2_second(in Ca_SR_release)
		double kom = 660.0;   // per_second(in Ca_SR_release)
		double ks = 148041085.1;   // per_second(in Ca_SR_release)
		double CM_tot = 0.045;   // millimolar(in Ca_buffering)
		double CQ_tot = 10.0;   // millimolar(in Ca_buffering)
		double Mgi = 2.5;   // millimolar(in Ca_buffering)
		double TC_tot = 0.031;   // millimolar(in Ca_buffering)
		double TMC_tot = 0.062;   // millimolar(in Ca_buffering)
		double kb_CM = 542.0;   // per_second(in Ca_buffering)
		double kb_CQ = 445.0;   // per_second(in Ca_buffering)
		double kb_TC = 446.0;   // per_second(in Ca_buffering)
		double kb_TMC = 7.51;   // per_second(in Ca_buffering)
		double kb_TMM = 751.0;   // per_second(in Ca_buffering)
		double kf_CM = 1.642e6;   // per_millimolar_second(in Ca_buffering)
		double kf_CQ = 175.4;   // per_millimolar_second(in Ca_buffering)
		double kf_TC = 88800.0;   // per_millimolar_second(in Ca_buffering)
		double kf_TMC = 227700.0;   // per_millimolar_second(in Ca_buffering)
		double kf_TMM = 2277.0;   // per_millimolar_second(in Ca_buffering)
		double K_up = 0.000286113;   // millimolar(in Ca_intracellular_fluxes)
		double P_up_basal = 5.0;   // millimolar_per_second(in Ca_intracellular_fluxes)
		double slope_up = 5.0e-5;   // millimolar(in Ca_intracellular_fluxes)
		double tau_dif_Ca = 5.469e-5;   // second(in Ca_intracellular_fluxes)
		double tau_tr = 0.04;   // second(in Ca_intracellular_fluxes)
		double L_cell = 67.0;   // micrometre(in Cell_parameters)
		double L_sub = 0.02;   // micrometre(in Cell_parameters)
		double R_cell = 3.9;   // micrometre(in Cell_parameters)
		double V_i_part = 0.46;   // dimensionless(in Cell_parameters)
		double V_jsr_part = 0.0012;   // dimensionless(in Cell_parameters)
		double V_nsr_part = 0.0116;   // dimensionless(in Cell_parameters)
		double Cao = 1.8;   // millimolar(in Ionic_values)
		double Ki = 140.0;   // millimolar(in Ionic_values)
		double Ko = 5.4;   // millimolar(in Ionic_values)
		double Nao = 140.0;   // millimolar(in Ionic_values)
		double C = 5.7e-5;   // microF(in Membrane)
		double F = 96485.3415;   // coulomb_per_mole(in Membrane)
		double R2 = 8314.472;   // joule_per_kilomole_kelvin(R in Membrane)
		double T = 310.0;   // kelvin(in Membrane)
		double clamp_mode = 0.0;   // dimensionless(in Membrane)
		double Nai_clamp = 1.0;   // dimensionless(in Nai_concentration)
		double ACh = 0.0;   // millimolar(in Rate_modulation_experiments)
		double Iso_1_uM = 0.0;   // dimensionless(in Rate_modulation_experiments)
		double V_holding = -45.0;   // millivolt(in Voltage_clamp)
		double V_test = -35.0;   // millivolt(in Voltage_clamp)
		double t_holding = 0.5;   // second(in Voltage_clamp)
		double t_test = 0.5;   // second(in Voltage_clamp)
		double V_dL = -16.4508;   // millivolt(in i_CaL_dL_gate)
		double k_dL = 4.3371;   // millivolt(in i_CaL_dL_gate)
		double Km_fCa = 0.000338;   // millimolar(in i_CaL_fCa_gate)
		double alpha_fCa = 0.0075;   // per_second(in i_CaL_fCa_gate)
		double k_fL = 0.0;   // millivolt(in i_CaL_fL_gate)
		double shift_fL = 0.0;   // millivolt(in i_CaL_fL_gate)
		
		double offset_fT = 0.0;   // second(in i_CaT_fT_gate)
		
		double ACh_on = 1.0;   // dimensionless(in i_KACh)
		
		
		double K1ni = 395.3;   // millimolar(in i_NaCa)
		double K1no = 1628.0;   // millimolar(in i_NaCa)
		double K2ni = 2.289;   // millimolar(in i_NaCa)
		double K2no = 561.4;   // millimolar(in i_NaCa)
		double K3ni = 26.44;   // millimolar(in i_NaCa)
		double K3no = 4.663;   // millimolar(in i_NaCa)
		
		double Kci = 0.0207;   // millimolar(in i_NaCa)
		double Kcni = 26.44;   // millimolar(in i_NaCa)
		double Kco = 3.663;   // millimolar(in i_NaCa)
		double Qci = 0.1369;   // dimensionless(in i_NaCa)
		double Qco = 0.0;   // dimensionless(in i_NaCa)
		double Qn = 0.4315;   // dimensionless(in i_NaCa)
		double blockade_NaCa = 0.0;   // dimensionless(in i_NaCa)
		double Km_Kp = 1.4;   // millimolar(in i_NaK)
		double Km_Nap = 14.0;   // millimolar(in i_NaK)
		
		double delta_m = 1.0e-5;   // millivolt(in i_Na_m_gate)
		
		double g_Na_L = 0.0;   // microS(in i_Na)
		double y_shift = 0.0;   // millivolt(in i_f_y_gate)
		double Km_f = 45.0;   // millimolar(in i_f)
		double alpha = 0.5927;   // dimensionless(in i_f)
		double blockade = 0.0;   // dimensionless(in i_f)
		
////////////////////////////////	assign heterogeneous channel conductivities		//////////////////////////////////////
	/*	double P_CaL = 0.4578;   // nanoA_per_millimolar(in i_CaL)
		double P_CaT = 0.04132;   // nanoA_per_millimolar(in i_CaT)
		double g_KACh = 0.00345;   // microS(in i_KACh)
		double g_Kr = 0.00424;   // microS(in i_Kr)
		double g_Ks_ = 0.00065;   // microS(in i_Ks)
		double g_Kur = 0.1539e-3;   // microS(in i_Kur)
		double g_Na = 0.0223;   // microS(in i_Na)
		double g_f = 0.00427;   // microS(in i_f)
		double g_to = 3.5e-3;   // microS(in i_to)
		double i_NaK_max = 0.08105;   // nanoA(in i_NaK)
		double K_NaCa = 3.343;   // nanoA(in i_NaCa)
		*/
		/////////////////////////////////////////////////////////////////////

		double P_CaL = Gs[cell_num].P_CaL;   // nanoA_per_millimolar(in i_CaL)
		//P_CaL = 0.5*P_CaL;

		////////////////////////////////////////////////////////////////////
		double P_CaT = Gs[cell_num].P_CaT;   // nanoA_per_millimolar(in i_CaT)
		double g_KACh = Gs[cell_num].g_KACh;   // microS(in i_KACh)
		/////////////////////////////////////////////////////////////////

		double g_Kr = Gs[cell_num].g_Kr;   // microS(in i_Kr)
		//g_Kr = 1.2*g_Kr;

		///////////////////////////////////////////////////////////////////////
		double g_Ks_ = Gs[cell_num].g_Ks;   // microS(in i_Ks)
		double g_Kur = Gs[cell_num].g_Kur;   // microS(in i_Kur)
		double g_Na = Gs[cell_num].g_Na;   // microS(in i_Na)
		double g_f = Gs[cell_num].g_f;   // microS(in i_f)
		double g_to = Gs[cell_num].g_to;   // microS(in i_to)
		double i_NaK_max = Gs[cell_num].i_NaK_max;   // nanoA(in i_NaK)
		double K_NaCa = Gs[cell_num].K_NaCa;   // nanoA(in i_NaCa)
////////////////////////////////////////////////////////////////////////////////////////////////
		//printf("K_NaCa=%e\n", K_NaCa);

			// ------------------------------------------------------------------------------ -
			// Computation
			//------------------------------------------------------------------------------ -

			// time(second)

		double j_SRCarel = ks*x[cell_num].O*(x[cell_num].Cajsr - x[cell_num].Casub);
		double diff = x[cell_num].Cajsr - x[cell_num].Casub;
		double kCaSR = MaxSR - (MaxSR - MinSR) / (1.0 + pow((EC50_SR / x[cell_num].Cajsr), HSR));
		double koSRCa = koCa / kCaSR;
		double kiSRCa = kiCa*kCaSR;
		//dY(3, 1) = kim*Y(4) - kiSRCa*x[cell_num].Casub*Y(3) - (koSRCa*x[cell_num].Casub ^ 2.0*Y(3) - kom*Y(2));
		//dY(2, 1) = koSRCa*x[cell_num].Casub ^ 2.0*Y(3) - kom*Y(2) - (kiSRCa*x[cell_num].Casub*Y(2) - kim*Y(1));
		//dY(1, 1) = kiSRCa*x[cell_num].Casub*Y(2) - kim*Y(1) - (kom*Y(1) - koSRCa*x[cell_num].Casub ^ 2.0*Y(4));
		//dY(4, 1) = kom*Y(1) - koSRCa*x[cell_num].Casub ^ 2.0*Y(4) - (kim*Y(4) - kiSRCa*x[cell_num].Casub*Y(3));
		//P_tot = Y(3) + Y(2) + Y(1) + Y(4);
		double delta_fTC = kf_TC*x[cell_num].Cai*(1.0 - x[cell_num].fTC) - kb_TC*x[cell_num].fTC;
		//dY(8, 1) = delta_fTC;
		double delta_fTMC = kf_TMC*x[cell_num].Cai*(1.0 - (x[cell_num].fTMC + x[cell_num].fTMM)) - kb_TMC*x[cell_num].fTMC;
		//dY(9, 1) = delta_fTMC;
		double delta_fTMM = kf_TMM*Mgi*(1.0 - (x[cell_num].fTMC + x[cell_num].fTMM)) - kb_TMM*x[cell_num].fTMM;
		//dY(10, 1) = delta_fTMM;
		double delta_fCMi = kf_CM*x[cell_num].Cai*(1.0 - x[cell_num].fCMi) - kb_CM*x[cell_num].fCMi;
		//dY(5, 1) = delta_fCMi;
		double delta_fCMs = kf_CM*x[cell_num].Casub*(1.0 - x[cell_num].fCMs) - kb_CM*x[cell_num].fCMs;
		//dY(6, 1) = delta_fCMs;
		double delta_fCQ = kf_CQ*x[cell_num].Cajsr*(1.0 - x[cell_num].fCQ) - kb_CQ*x[cell_num].fCQ;
		//dY(7, 1) = delta_fCQ;
		double j_Ca_dif = (x[cell_num].Casub - x[cell_num].Cai) / tau_dif_Ca;
		double V_sub = 0.000000001*2.0*M_PI*L_sub*(R_cell - L_sub / 2.0)*L_cell;

		double b_up;
		if (Iso_1_uM > 0.0) { b_up = -0.25; }
		else if (ACh > 0.0) { b_up = 0.7*ACh / (0.00009 + ACh); }
		else { b_up = 0.0; }

		double P_up = P_up_basal*(1.0 - b_up);
		double j_up = P_up / (1.0 + exp((-x[cell_num].Cai + K_up) / slope_up));
		double V_cell = 0.000000001*M_PI*pow(R_cell, 2.0)*L_cell;
		double V_nsr = V_nsr_part*V_cell;
		double V_i = V_i_part*V_cell - V_sub;
		//dY(14, 1) = 1.0*(j_Ca_dif*V_sub - j_up*V_nsr) / V_i - (CM_tot*delta_fCMi + TC_tot*delta_fTC + TMC_tot*delta_fTMC);
		double V_jsr = V_jsr_part*V_cell;

		//double V_clamp;
		//if ((time > t_holding) && (time < t_holding + t_test)) { V_clamp = V_test; }
		//else { V_clamp = V_holding; }
		//if (clamp_mode >= 1.0) { V = V_clamp; }
		//else { V = Y(15); }
		double RTONF = R2*T / F;
		double i_siCa = 2.0*P_CaL*(x[cell_num].V - 0.0) / (RTONF*(1.0 - exp(-1.0*(x[cell_num].V - 0.0)*2.0 / RTONF)))*(x[cell_num].Casub - Cao*exp(-2.0*(x[cell_num].V - 0.0) / RTONF))*x[cell_num].dL*x[cell_num].fL*x[cell_num].fCa;
		double i_CaT = 2.0*P_CaT*x[cell_num].V / (RTONF*(1.0 - exp(-1.0*x[cell_num].V*2.0 / RTONF)))*(x[cell_num].Casub - Cao*exp(-2.0*x[cell_num].V / RTONF))*x[cell_num].dT*x[cell_num].fT;
		double k32 = exp(Qn*x[cell_num].V / (2.0*RTONF));
		double Nai = x[cell_num].Nai;
		double k43 = Nai / (K3ni + Nai);
		double di = 1.0 + x[cell_num].Casub / Kci*(1.0 + exp(-Qci*x[cell_num].V / RTONF) + Nai / Kcni) + Nai / K1ni*(1.0 + Nai / K2ni*(1.0 + Nai / K3ni));
		double k14 = Nai / K1ni*Nai / K2ni*(1.0 + Nai / K3ni)*exp(Qn*x[cell_num].V / (2.0*RTONF)) / di;
		double k12 = x[cell_num].Casub / Kci*exp(-Qci*x[cell_num].V / RTONF) / di;
		double k41 = exp(-Qn*x[cell_num].V / (2.0*RTONF));
		double k34 = Nao / (K3no + Nao);
		double x2 = k32*k43*(k14 + k12) + k41*k12*(k34 + k32);
		double do_ = 1.0 + Cao / Kco*(1.0 + exp(Qco*x[cell_num].V / RTONF)) + Nao / K1no*(1.0 + Nao / K2no*(1.0 + Nao / K3no));
		double k21 = Cao / Kco*exp(Qco*x[cell_num].V / RTONF) / do_;
		double k23 = Nao / K1no*Nao / K2no*(1.0 + Nao / K3no)*exp(-Qn*x[cell_num].V / (2.0*RTONF)) / do_;
		double x1 = k41*k34*(k23 + k21) + k21*k32*(k43 + k41);
		double x3 = k14*k43*(k23 + k21) + k12*k23*(k43 + k41);
		double x4 = k23*k34*(k14 + k12) + k14*k21*(k34 + k32);
		double i_NaCa = (1.0 - blockade_NaCa)*K_NaCa*(x2*k21 - x1*k12) / (x1 + x2 + x3 + x4);
		//dY(13, 1) = j_SRCarel*V_jsr / V_sub - ((i_siCa + i_CaT - 2.0*i_NaCa) / (2.0*F*V_sub) + j_Ca_dif + CM_tot*delta_fCMs);
		double j_tr = (x[cell_num].Cansr - x[cell_num].Cajsr) / tau_tr;
		//dY(12, 1) = j_up - j_tr*V_jsr / V_nsr;
		//dY(11, 1) = j_tr - (j_SRCarel + CQ_tot*delta_fCQ);
		double E_Na = RTONF*log(Nao / Nai);
		//printf("Ena=%e\n", E_Na);

		double E_K = RTONF*log(Ko / Ki);
		double E_Ca = 0.5*RTONF*log(Cao / x[cell_num].Casub);
		double G_f = g_f / (Ko / (Ko + Km_f));
		double G_f_K = G_f / (alpha + 1.0);
		double G_f_Na = alpha*G_f_K;
		double g_f_Na = G_f_Na*Ko / (Ko + Km_f);
		double i_fNa = x[cell_num].y*g_f_Na*(x[cell_num].V - E_Na)*(1.0 - blockade);
		double g_f_K = G_f_K*Ko / (Ko + Km_f);
		double i_fK = x[cell_num].y*g_f_K*(x[cell_num].V - E_K)*(1.0 - blockade);
		double i_f = i_fNa + i_fK;
		double i_Kr = g_Kr*(x[cell_num].V - E_K)*(0.9*x[cell_num].paF + 0.1*x[cell_num].paS)*x[cell_num].piy;

		double g_Ks;
		if (Iso_1_uM > 0.0) { g_Ks = 1.2*g_Ks_; }
		else { g_Ks = g_Ks_; }

		double E_Ks = RTONF*log((Ko + 0.12*Nao) / (Ki + 0.12*Nai));
		double i_Ks = g_Ks*(x[cell_num].V - E_Ks)*pow(x[cell_num].n, 2.0);
		double i_to = g_to*(x[cell_num].V - E_K)*x[cell_num].q*x[cell_num].r;

		double Iso_increase_2;
		if (Iso_1_uM > 0.0) { Iso_increase_2 = 1.2; }
		else { Iso_increase_2 = 1.0; }

		double i_NaK = Iso_increase_2*i_NaK_max*pow((1.0 + pow((Km_Kp / Ko), 1.2)), -1.0)*pow((1.0 + pow((Km_Nap / Nai), 1.3)), -1.0)*pow((1.0 + exp(-(x[cell_num].V - E_Na + 110.0) / 20.0)), -1.0);
		//	INaK = INaK_max * pow((1 + pow((KmKp / Ko), 1.2)), -1) * pow((1 + pow((KmNap / x[cell_num].Nai), 1.3)), -1) * pow((1 + exp(-(x[cell_num].V - ENa + 110) / 20)), -1);
		double E_mh = RTONF*log((Nao + 0.12*Ko) / (Nai + 0.12*Ki));
		double i_Na_ = g_Na*pow(x[cell_num].m, 3.0)*x[cell_num].h*(x[cell_num].V - E_mh);
		double i_Na_L = g_Na_L*pow(x[cell_num].m, 3.0)*(x[cell_num].V - E_mh);
		double i_Na = i_Na_ + i_Na_L;
		double i_siK = 0.000365*P_CaL*(x[cell_num].V - 0.0) / (RTONF*(1.0 - exp(-1.0*(x[cell_num].V - 0.0) / RTONF)))*(Ki - Ko*exp(-1.0*(x[cell_num].V - 0.0) / RTONF))*x[cell_num].dL*x[cell_num].fL*x[cell_num].fCa;
		double i_siNa = 0.0000185*P_CaL*(x[cell_num].V - 0.0) / (RTONF*(1.0 - exp(-1.0*(x[cell_num].V - 0.0) / RTONF)))*(Nai - Nao*exp(-1.0*(x[cell_num].V - 0.0) / RTONF))*x[cell_num].dL*x[cell_num].fL*x[cell_num].fCa;
		double ACh_block = 0.31*ACh / (ACh + 0.00009);

		double Iso_increase_1;
		if (Iso_1_uM > 0.0) { Iso_increase_1 = 1.23; }
		else { Iso_increase_1 = 1.0; }
		double i_CaL = (i_siCa + i_siK + i_siNa)*(1.0 - ACh_block)*1.0*Iso_increase_1;
		double i_KACh;
		if (ACh > 0.0) { i_KACh = ACh_on*g_KACh*(x[cell_num].V - E_K)*(1.0 + exp((x[cell_num].V + 20.0) / 20.0))*x[cell_num].a; }
		else { i_KACh = 0.0; }
		double i_Kur = g_Kur*x[cell_num].rkur*x[cell_num].skur*(x[cell_num].V - E_K);
		///////////////////////////////////////////////////////////////////////////////////////////////////////
		double i_tot = i_f + i_Kr + i_Ks + i_to + i_NaK + i_NaCa + i_Na + i_CaL + i_CaT + i_KACh + i_Kur;
		//	printf("I_Na=%e,i_f=%e,i_NaK=%e\n", i_Na,i_f,i_NaK);
		//////////////////////////////////////////////////////////////////////////////////////////////////////////
		//dY(15, 1) = -i_tot / C;
		//dY(16, 1) = (1.0 - Nai_clamp)*-1.0*(i_Na + i_fNa + i_siNa + 3.0*i_NaK + 3.0*i_NaCa) / (1.0*(V_i + V_sub)*F);

		double Iso_shift_dL, Iso_slope_dL;

		if (Iso_1_uM > 0.0) { Iso_shift_dL = -8.0; }
		else { Iso_shift_dL = 0.0; }

		if (Iso_1_uM > 0.0) { Iso_slope_dL = -27.0; }
		else { Iso_slope_dL = 0.0; }

		double dL_infinity = 1.0 / (1.0 + exp(-(x[cell_num].V - V_dL - Iso_shift_dL) / (k_dL*(1.0 + Iso_slope_dL / 100.0))));
		double adVm;
		if (x[cell_num].V == -41.8) { adVm = -41.80001; }
		else if (x[cell_num].V == 0.0) { adVm = 0.0; }
		else if (x[cell_num].V == -6.8) { adVm = -6.80001; }
		else { adVm = x[cell_num].V; }

		double alpha_dL = -0.02839*(adVm + 41.8) / (exp(-(adVm + 41.8) / 2.5) - 1.0) - 0.0849*(adVm + 6.8) / (exp(-(adVm + 6.8) / 4.8) - 1.0);
		double bdVm;
		if (x[cell_num].V == -1.8) { bdVm = -1.80001; }
		else { bdVm = x[cell_num].V; }
		double beta_dL = 0.01143*(bdVm + 1.8) / (exp((bdVm + 1.8) / 2.5) - 1.0);
		double tau_dL = 0.001 / (alpha_dL + beta_dL);
		//dY(17, 1) = (dL_infinity - x[cell_num].dL) / tau_dL;
		double fCa_infinity = Km_fCa / (Km_fCa + x[cell_num].Casub);
		double tau_fCa = 0.001*fCa_infinity / alpha_fCa;
		//dY(18, 1) = (fCa_infinity - x[cell_num].fCa) / tau_fCa;
		double fL_infinity = 1.0 / (1.0 + exp((x[cell_num].V + 37.4 + shift_fL) / (5.3 + k_fL)));
		double tau_fL = 0.001*(44.3 + 230.0*exp(-pow(((x[cell_num].V + 36.0) / 10.0), 2.0)));
		//dY(19, 1) = (fL_infinity - x[cell_num].fL) / tau_fL;
		double dT_infinity = 1.0 / (1.0 + exp(-(x[cell_num].V + 38.3) / 5.5));
		double tau_dT = 0.001 / (1.068*exp((x[cell_num].V + 38.3) / 30.0) + 1.068*exp(-(x[cell_num].V + 38.3) / 30.0));
		//dY(20, 1) = (dT_infinity - x[cell_num].dT) / tau_dT;
		double fT_infinity = 1.0 / (1.0 + exp((x[cell_num].V + 58.7) / 3.8));
		double tau_fT = 1.0 / (16.67*exp(-(x[cell_num].V + 75.0) / 83.3) + 16.67*exp((x[cell_num].V + 75.0) / 15.38)) + offset_fT;
		//dY(21, 1) = (fT_infinity - x[cell_num].fT) / tau_fT;
		double alpha_a = (3.5988 - 0.025641) / (1.0 + 0.0000012155 / pow((1.0*ACh), 1.6951)) + 0.025641;
		double beta_a = 10.0*exp(0.0133*(x[cell_num].V + 40.0));
		double a_infinity = alpha_a / (alpha_a + beta_a);
		double tau_a = 1.0 / (alpha_a + beta_a);
		//dY(22, 1) = (a_infinity - x[cell_num].a) / tau_a;
		double alfapaF = 1.0 / (1.0 + exp(-(x[cell_num].V + 23.2) / 6.6)) / (0.84655354 / (37.2*exp(x[cell_num].V / 11.9) + 0.96*exp(-x[cell_num].V / 18.5)));
		double betapaF = 4.0*((37.2*exp(x[cell_num].V / 15.9) + 0.96*exp(-x[cell_num].V / 22.5)) / 0.84655354 - 1.0 / (1.0 + exp(-(x[cell_num].V + 23.2) / 10.6)) / (0.84655354 / (37.2*exp(x[cell_num].V / 15.9) + 0.96*exp(-x[cell_num].V / 22.5))));
		double pa_infinity = 1.0 / (1.0 + exp(-(x[cell_num].V + 10.0144) / 7.6607));
		double tau_paS = 0.84655354 / (4.2*exp(x[cell_num].V / 17.0) + 0.15*exp(-x[cell_num].V / 21.6));
		double tau_paF = 1.0 / (30.0*exp(x[cell_num].V / 10.0) + exp(-x[cell_num].V / 12.0));
		//dY(24, 1) = (pa_infinity - x[cell_num].paS) / tau_paS;
		//dY(23, 1) = (pa_infinity - x[cell_num].paF) / tau_paF;
		double tau_pi = 1.0 / (100.0*exp(-x[cell_num].V / 54.645) + 656.0*exp(x[cell_num].V / 106.157));
		double pi_infinity = 1.0 / (1.0 + exp((x[cell_num].V + 28.6) / 17.1));
		//dY(25, 1) = (pi_infinity - x[cell_num].piy) / tau_pi;
		double Iso_shift_1;
		if (Iso_1_uM > 0.0) { Iso_shift_1 = -14.0; }
		else { Iso_shift_1 = 0.0; }

		double n_infinity = sqrt(1.0 / (1.0 + exp(-(x[cell_num].V + 0.6383 - Iso_shift_1) / 10.7071)));
		double alpha_n = 28.0 / (1.0 + exp(-(x[cell_num].V - 40.0 - Iso_shift_1) / 3.0));
		double beta_n = 1.0*exp(-(x[cell_num].V - Iso_shift_1 - 5.0) / 25.0);
		double tau_n = 1.0 / (alpha_n + beta_n);
		//dY(26, 1) = (n_infinity - x[cell_num].n) / tau_n;
		double r_Kur_infinity = 1.0 / (1.0 + exp((x[cell_num].V + 6.0) / -8.6));
		double tau_r_Kur = 0.009 / (1.0 + exp((x[cell_num].V + 5.0) / 12.0)) + 0.0005;
		//dY(27, 1) = (r_Kur_infinity - x[cell_num].r_Kur) / tau_r_Kur;
		double s_Kur_infinity = 1.0 / (1.0 + exp((x[cell_num].V + 7.5) / 10.0));
		double tau_s_Kur = 0.59 / (1.0 + exp((x[cell_num].V + 60.0) / 10.0)) + 3.05;
		//dY(28, 1) = (s_Kur_infinity - x[cell_num].s_Kur) / tau_s_Kur;
		double h_infinity = 1.0 / (1.0 + exp((x[cell_num].V + 69.804) / 4.4565));
		double alpha_h = 20.0*exp(-0.125*(x[cell_num].V + 75.0));
		double beta_h = 2000.0 / (320.0*exp(-0.1*(x[cell_num].V + 75.0)) + 1.0);
		double tau_h = 1.0 / (alpha_h + beta_h);
		//dY(29, 1) = (h_infinity - x[cell_num].h) / tau_h;
		double m_infinity = 1.0 / (1.0 + exp(-(x[cell_num].V + 42.0504) / 8.3106));
		double E0_m = x[cell_num].V + 41.0;
		double alpha_m;
		if (abs(E0_m) < delta_m) { alpha_m = 2000.0; }
		else { alpha_m = 200.0*E0_m / (1.0 - exp(-0.1*E0_m)); }

		double beta_m = 8000.0*exp(-0.056*(x[cell_num].V + 66.0));
		double tau_m = 1.0 / (alpha_m + beta_m);
		//dY(30, 1) = (m_infinity - x[cell_num].m) / tau_m;
		double ACh_shift;
		if (ACh > 0.0) { ACh_shift = -1.0 - 9.898*pow((1.0*ACh), 0.618) / (pow((1.0*ACh), 0.618) + 0.00122423); }
		else { ACh_shift = 0.0; }
		double Iso_shift_2;
		if (Iso_1_uM > 0.0) { Iso_shift_2 = 7.5; }
		else { Iso_shift_2 = 0.0; }
		double tau_y = 1.0 / (0.36*(x[cell_num].V + 148.8 - ACh_shift - Iso_shift_2) / (exp(0.066*(x[cell_num].V + 148.8 - ACh_shift - Iso_shift_2)) - 1.0) + 0.1*(x[cell_num].V + 87.3 - ACh_shift - Iso_shift_2) / (1.0 - exp(-0.2*(x[cell_num].V + 87.3 - ACh_shift - Iso_shift_2)))) - 0.054;

		double y_infinity;
		if (x[cell_num].V < -(80.0 - ACh_shift - Iso_shift_2 - y_shift)) { y_infinity = 0.01329 + 0.99921 / (1.0 + exp((x[cell_num].V + 97.134 - ACh_shift - Iso_shift_2 - y_shift) / 8.1752)); }
		else { y_infinity = 0.0002501*exp(-(x[cell_num].V - ACh_shift - Iso_shift_2 - y_shift) / 12.861); }

		//dY(31, 1) = (y_infinity - x[cell_num].y) / tau_y;
		double q_infinity = 1.0 / (1.0 + exp((x[cell_num].V + 49.0) / 13.0));
		double tau_q = 0.001*0.6*(65.17 / (0.57*exp(-0.08*(x[cell_num].V + 44.0)) + 0.065*exp(0.1*(x[cell_num].V + 45.93))) + 10.1);
		//dY(32, 1) = (q_infinity - x[cell_num].q) / tau_q;
		double r_infinity = 1.0 / (1.0 + exp(-(x[cell_num].V - 19.3) / 15.0));
		double tau_r = 0.001*0.66*1.4*(15.59 / (1.037*exp(0.09*(x[cell_num].V + 30.61)) + 0.369*exp(-0.12*(x[cell_num].V + 23.84))) + 2.98);
		//dY(33, 1) = (r_infinity - x[cell_num].r) / tau_r;

		double Iion = i_tot;
		ion_current[cell_num] = Iion;
	//	printf("x[cell_num].V=%f,cell_num=%d,Iion=%f\n", x[cell_num].V,cell_num,ion_current[cell_num]);

		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////   COMPUTE STATES   ////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		x_temp[cell_num].I = updateSingleState(x[cell_num].I, kiSRCa*x[cell_num].Casub*x[cell_num].O - kim*x[cell_num].I - (kom*x[cell_num].I - koSRCa*pow(x[cell_num].Casub, 2)*x[cell_num].RI), step);//
		x_temp[cell_num].O = updateSingleState(x[cell_num].O, koSRCa*pow(x[cell_num].Casub, 2)*x[cell_num].R - kom*x[cell_num].O - (kiSRCa*x[cell_num].Casub*x[cell_num].O - kim*x[cell_num].I), step);//
		x_temp[cell_num].R = updateSingleState(x[cell_num].R, kim*x[cell_num].RI - kiSRCa*x[cell_num].Casub*x[cell_num].R - (koSRCa*pow(x[cell_num].Casub, 2)*x[cell_num].R - kom*x[cell_num].O), step);//
		x_temp[cell_num].RI = updateSingleState(x[cell_num].RI, kom*x[cell_num].I - koSRCa*pow(x[cell_num].Casub, 2)*x[cell_num].RI - (kim*x[cell_num].RI - kiSRCa*x[cell_num].Casub* x[cell_num].R), step); //
		x_temp[cell_num].fCMi = updateSingleState(x[cell_num].fCMi, delta_fCMi, step);//
		x_temp[cell_num].fCMs = updateSingleState(x[cell_num].fCMs, delta_fCMs, step);
		x_temp[cell_num].fCQ = updateSingleState(x[cell_num].fCQ, delta_fCQ, step); //
		x_temp[cell_num].fTC = updateSingleState(x[cell_num].fTC, delta_fTC, step);//
		x_temp[cell_num].fTMC = updateSingleState(x[cell_num].fTMC, delta_fTMC, step);//
		x_temp[cell_num].fTMM = updateSingleState(x[cell_num].fTMM, delta_fTMM, step); //
		x_temp[cell_num].Cajsr = updateSingleState(x[cell_num].Cajsr, j_tr - (j_SRCarel + CQ_tot*delta_fCQ), step);// corrected
		x_temp[cell_num].Cansr = updateSingleState(x[cell_num].Cansr, j_up - (j_tr*V_jsr / V_nsr), step);// 
		x_temp[cell_num].Casub = updateSingleState(x[cell_num].Casub, ((j_SRCarel*V_jsr) / V_sub) - (((i_siCa + i_CaT - 2 * i_NaCa) / (2 * F*V_sub)) + j_Ca_dif + CM_tot*delta_fCMs), step);//((JSRCarel*Vjsr) / Vsub) - (((IsiCa + ICaT - 2 * INaCa) / (2 * F*Vsub)) + JCadif + CMtot*d_fCMs)
		x_temp[cell_num].Cai = updateSingleState(x[cell_num].Cai, ((1 * (j_Ca_dif*V_sub - j_up*V_nsr)) / V_i) - (CM_tot*delta_fCMi + TC_tot*delta_fTC + TMC_tot*delta_fTMC), step);//							

			/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		x_temp[cell_num].Nai = updateSingleState(x[cell_num].Nai, ((1 - 1)* (-1)*(i_Na + i_fNa + i_siNa + 3 * i_NaK + 3 * i_NaCa)) / (1 * (V_i + V_sub)*F), step);
		//	double check that Nai is actually constant		( (1-1)* (-1)*(INa + IfNa + IsiNa + 3*INaK + 3*INaCa ))/(1*(Vi + Vsub)*F)
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		x_temp[cell_num].dL = updateSingleState(x[cell_num].dL, (dL_infinity - x[cell_num].dL) / tau_dL, step);//
		x_temp[cell_num].fCa = updateSingleState(x[cell_num].fCa, (fCa_infinity - x[cell_num].fCa) / tau_fCa, step);//
		x_temp[cell_num].fL = updateSingleState(x[cell_num].fL, (fL_infinity - x[cell_num].fL) / tau_fL, step);//
		x_temp[cell_num].dT = updateSingleState(x[cell_num].dT, (dT_infinity - x[cell_num].dT) / tau_dT, step);//				
		x_temp[cell_num].fT = updateSingleState(x[cell_num].fT, (fT_infinity - x[cell_num].fT) / tau_fT, step);//
		x_temp[cell_num].a = updateSingleState(x[cell_num].a, (a_infinity - x[cell_num].a) / tau_a, step);
		x_temp[cell_num].paF = updateSingleState(x[cell_num].paF, (pa_infinity - x[cell_num].paF) / tau_paF, step);
		x_temp[cell_num].paS = updateSingleState(x[cell_num].paS, (pa_infinity - x[cell_num].paS) / tau_paS, step);
		x_temp[cell_num].piy = updateSingleState(x[cell_num].piy, (pi_infinity - x[cell_num].piy) / tau_pi, step);
		x_temp[cell_num].n = updateSingleState(x[cell_num].n, (n_infinity - x[cell_num].n) / tau_n, step);
		x_temp[cell_num].rkur = updateSingleState(x[cell_num].rkur, (r_Kur_infinity - x[cell_num].rkur) / tau_r_Kur, step);
		x_temp[cell_num].skur = updateSingleState(x[cell_num].skur, (s_Kur_infinity - x[cell_num].skur) / tau_s_Kur, step);
		x_temp[cell_num].h = updateSingleState(x[cell_num].h, (h_infinity - x[cell_num].h) / tau_h, step);//
		x_temp[cell_num].m = updateSingleState(x[cell_num].m, (m_infinity - x[cell_num].m) / tau_m, step);//
		x_temp[cell_num].y = updateSingleState(x[cell_num].y, (y_infinity - x[cell_num].y) / tau_y, step);//
		x_temp[cell_num].q = updateSingleState(x[cell_num].q, (q_infinity - x[cell_num].q) / tau_q, step);
		x_temp[cell_num].r = updateSingleState(x[cell_num].r, (r_infinity - x[cell_num].r) / tau_r, step);

		//if ((x[cell_num].m+((m_inf - x[cell_num].m) / tau_m)) > 0 && (x[cell_num].m + ((m_inf - x[cell_num].m) / tau_m))<1) {

	//	}else {
		//		x_temp[cell_num].m = x[cell_num].m;/
			//}//					
		//printf("INa=%f, taum= %f,minf = %f, x_m= %f, cval_m=%f\n", INa, tau_m, m_inf, x[cell_num].m, (m_inf-x[cell_num].m)/tau_m);
		//printf("V= %f, tauh= %f,hinf = %f, x_h= %f, cval_h=%f\n", x[cell_num].V, tau_h, h_inf, x[cell_num].h, (h_inf - x[cell_num].h) / tau_h);
		//printf("x_temp.r=%e\n", x_temp[cell_num].r);
//	}
	//printf("x_temp.r=%e\n", x_temp[cell_num].r);
}


__global__ void updateState(cell* x, cell* x_temp) {
	int idx = (blockIdx.x*blockDim.x) + threadIdx.x;
//	int idx = cellsThread*threadIdx.x;
	//int maxIdx = idx + cellsThread;
	//for (;idx < maxIdx;idx++) {
		x[idx].Casub = x_temp[idx].Casub;
	//	printf("UpStateKernel=%f\n", x[idx].Casub);
		x[idx].Nai = x_temp[idx].Nai;
		x[idx].y = x_temp[idx].y;
		x[idx].m = x_temp[idx].m;
		x[idx].h = x_temp[idx].h;
		x[idx].dL = x_temp[idx].dL;
		x[idx].fL = x_temp[idx].fL;
		x[idx].fCa = x_temp[idx].fCa;
		x[idx].dT = x_temp[idx].dT;
		x[idx].fT = x_temp[idx].fT;
		x[idx].R = x_temp[idx].R;
		x[idx].O = x_temp[idx].O;
		x[idx].I = x_temp[idx].I;
		x[idx].RI = x_temp[idx].RI;
		x[idx].Cajsr = x_temp[idx].Cajsr;
		x[idx].Cansr = x_temp[idx].Cansr;
		x[idx].Cai = x_temp[idx].Cai;
		x[idx].fTMM = x_temp[idx].fTMM;
		x[idx].fCMi = x_temp[idx].fCMi;
		x[idx].fCMs = x_temp[idx].fCMs;
		x[idx].fTC = x_temp[idx].fTC;
		x[idx].fTMC = x_temp[idx].fTMC;
		x[idx].fCQ = x_temp[idx].fCQ;
		x[idx].rkur = x_temp[idx].rkur;
		x[idx].skur = x_temp[idx].skur;
		x[idx].q = x_temp[idx].q;
		x[idx].r = x_temp[idx].r;
		x[idx].paS = x_temp[idx].paS;
		x[idx].paF = x_temp[idx].paF;
		x[idx].piy = x_temp[idx].piy;
		x[idx].n = x_temp[idx].n;
		x[idx].a = x_temp[idx].a;
	//}
}


__global__ void computeVoltage(cell* x, double* V, double* Iion, double step, int time, double* Inet) {

	int idx = (blockIdx.x*blockDim.x) + threadIdx.x;
	double Cm = 5.7e-5;
		
	double	rGap = 10000 * 1e6; //% MOhm
	double	gapJunct = (57 * 1e-12)*rGap;
		
	//////////////////////////////////////////		Tissue model		////////////////////////////////////////////////////////
	if (sim2D == 1) {
		double Vnet_U, Vnet_D, Vnet_L, Vnet_R;
		Vnet_U = 0;Vnet_D = 0;Vnet_L = 0;Vnet_R = 0;
		if (idx >= WIDTH) { Vnet_U = x[idx - WIDTH].V - x[idx].V; }
		else { Vnet_U = 0; }
		if (idx < (LENGTH - 1)*WIDTH) { Vnet_D = x[idx + WIDTH].V - x[idx].V; }
		else { Vnet_D = 0; }
		if (idx%WIDTH == 0) { Vnet_L = 0; }
		else { Vnet_L = x[idx - 1].V - x[idx].V; }
		if (idx%WIDTH < (WIDTH - 1)) { Vnet_R = x[idx + 1].V - x[idx].V; }
		else { Vnet_R = 0; }




	/*	if (idx == 194 || idx == 195 || idx == 196 || idx == 294 || idx == 296 || idx == 345) { Vnet_D = 0; }
		if (idx == 244 || idx == 245 || idx == 246 || idx == 344 || idx == 346 || idx == 395) { Vnet_U = 0; }
		if (idx == 244 || idx == 294 || idx == 247 || idx == 297 || idx == 346 || idx == 345) { Vnet_L = 0; }
		if (idx == 243 || idx == 293 || idx == 246 || idx == 296 || idx == 345 || idx == 344) { Vnet_R = 0; }

		if (idx == 437 || idx == 438 || idx == 489 || idx == 587 || idx == 588 || idx == 589) { Vnet_D = 0; }
		if (idx == 487 || idx == 488 || idx == 539 || idx == 637 || idx == 638 || idx == 639) { Vnet_U = 0; }
		if (idx == 487 || idx == 537 || idx == 587 || idx == 489 || idx == 540 || idx == 590) { Vnet_L = 0; }
		if (idx == 486 || idx == 536 || idx == 586 || idx == 539 || idx == 589 || idx == 488) { Vnet_R = 0; }

		if (idx == 804 || idx == 805 || idx == 856 || idx == 906 || idx == 954 || idx == 955) { Vnet_D = 0; }
		if (idx == 854 || idx == 855 || idx == 906 || idx == 956 || idx == 1004 || idx == 1005) { Vnet_U = 0; }
		if (idx == 854 || idx == 904 || idx == 954 || idx == 856 || idx == 907 || idx == 956) { Vnet_L = 0; }
		if (idx == 853 || idx == 903 || idx == 953 || idx == 855 || idx == 906 || idx == 955) { Vnet_R = 0; }

		if (idx == 1213 || idx == 1214 || idx == 1265 || idx == 1363 || idx == 1364 || idx == 1365) { Vnet_D = 0; }
		if (idx == 1263 || idx == 1264 || idx == 1315 || idx == 1413 || idx == 1414 || idx == 1415) { Vnet_U = 0; }
		if (idx == 1263 || idx == 1313 || idx == 1363 || idx == 1265 || idx == 1316 || idx == 1366) { Vnet_L = 0; }
		if (idx == 1262 || idx == 1312 || idx == 1362 || idx == 1264 || idx == 1315 || idx == 1365) { Vnet_R = 0; }
		*/




		double Vnet = Vnet_R + Vnet_L + Vnet_U + Vnet_D;
		Inet[idx] = (1 / gapJunct)*(Vnet) ;
		if (x[idx].V <= -50 && Iion[idx]){
			Inet[idx] = 0;
			Vnet = 0;
		}

		V[idx] = updateSingleState(x[idx].V, ((1 / gapJunct)*(Vnet)) - (Iion[idx] / Cm), step);
		//printf("idx=%d, Vnet = %f, Inet[idx]= %f \n", idx, Vnet, Inet[idx]);
		//V[idx] = updateSingleState(x[idx].V,  - (Iion[idx] / Cm), step);
		//(x[idx].V) + step*((1 / gapJunct)*(Vnet_R + Vnet_L + Vnet_U + Vnet_D) - (Iion[idx] / Cm));
		//V[idx] = (x[idx].V) + step*( - (Iion[idx] / Cm)); //single cell?
	//	if (time == 0 && (idx<=100 ) ) {
			//== 2490 || idx == 1 || idx == 49 || idx == 50 || idx == 249 || idx == 250 || idx == 0 || idx == 2499
		//	printf("blockIdx = %d, threadIdx=%d, idx=%d, Vnet_L =%f, Vnet_R=%f, Vnet_U=%f ,Vnet_D = %f \n", blockIdx.x, threadIdx.x, idx, Vnet_L, Vnet_R, Vnet_U, Vnet_D);
		//}
	}
	else {
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		V[idx] = updateSingleState(x[idx].V, -(Iion[idx] / Cm), step);
		
	}


}

__global__ void updateVoltage(cell* x, double* V) {

	int idx = (blockIdx.x*blockDim.x) + threadIdx.x;

		x[idx].V = V[idx];
	
}

int main(int argc, const char* argv[])
{
	

	int time = 0;
	int N = nCELLS*nSIMS;
	int cellSize = sizeof(cell);
	int numBytes = N*cellSize;

	int blockSize = nCELLS;
	cell* hVars;
	cell* dVars;

	cellGs* Gs;
	cudaMalloc((void**)&Gs, N * sizeof(cellGs));
	//cellScales* scales;
	//cudaMalloc((void**)&scales, N * sizeof(cellScales));
	int size = nSIMS*nCELLS;
	hVars = (cell *)malloc(numBytes);
	cudaMalloc((void**)&dVars, numBytes);

	float* hgChannels;
	float* gChannels;
	//double* hgChannels;
	//double* gChannels;

	float* huploadICs;
	float* uploadICs;
	//double* huploadICs;
	//double* uploadICs;
	
	int nChannels = 11;
	hgChannels = (float *)malloc(sizeof(float)*nCELLS * nChannels);
	//hgChannels = (double *)malloc(sizeof(double)*nCELLS * nChannels);

    int nStates = 33;
    huploadICs = (float *)malloc(sizeof(float)*nCELLS * nStates);
	//huploadICs = (double *)malloc(sizeof(double)*nCELLS * nStates);

	
	FILE *f1 = fopen("C:/Users/sobie/Desktop/Chiara/SAN project_Fabbri_lv/cudasparkmodel2011/Gs_F_02_50x50.txt", "rb");//Gs_F_01_50x50.txt
	int errnum;
	if ((f1 = fopen("C:/Users/sobie/Desktop/Chiara/SAN project_Fabbri_lv/cudasparkmodel2011/Gs_F_02_50x50.txt", "rb")) == NULL)//Gs_F_01_50x50
		//if ((f1 = fopen("C:/Users/sobie/Desktop/chiara/simparall/cudasparkmodel2011/MscalesIna.txt", "rb")) == NULL)

	{
		errnum = errno;
		fprintf(stderr, "Value of errno: %d\n", errno);
		perror("Error printed by perror");
		fprintf(stderr, "Error opening file: %s\n", strerror(errnum));

	}
	for (int j = 0;j<nCELLS*nChannels;j++) {
		fscanf(f1, "%f \n ", &hgChannels[j]);
		//printf("%f \n", hgChannels[j]);

	}
	fclose(f1);

	cudaMalloc(&gChannels, sizeof(float)*nCELLS* nChannels);
	cudaMemcpy(gChannels, hgChannels, nCELLS* nChannels * sizeof(float), cudaMemcpyHostToDevice);
	//cudaMalloc(&gChannels, sizeof(double)*nCELLS* nChannels);
	//cudaMemcpy(gChannels, hgChannels, nCELLS* nChannels * sizeof(double), cudaMemcpyHostToDevice);

	
	FILE *f2 = fopen("C:/Users/sobie/Desktop/Chiara/SAN project_Fabbri_lv/cudasparkmodel2011/ICS_s02F.txt", "rb");//ICs_Fabbri_s01873_50x50ICs_F_s03_CA05_n100
	
	if ((f2 = fopen("C:/Users/sobie/Desktop/Chiara/SAN project_Fabbri_lv/cudasparkmodel2011/ICS_s02F.txt", "rb")) == NULL)//ICS_s02F
		//if ((f1 = fopen("C:/Users/sobie/Desktop/chiara/simparall/cudasparkmodel2011/MscalesIna.txt", "rb")) == NULL)

	{
		errnum = errno;
		fprintf(stderr, "Value of errno: %d\n", errno);
		perror("Error printed by perror");
		fprintf(stderr, "Error opening file: %s\n", strerror(errnum));

	}
	for (int j = 0;j<nCELLS*nStates;j++) {
		fscanf(f2, "%f \n ", &huploadICs[j]);
		//printf("%f \n", hgChannels[j]);

	}
	fclose(f2);

	cudaMalloc(&uploadICs, sizeof(float)*nCELLS* nStates);
	cudaMemcpy(uploadICs, huploadICs, nCELLS* nStates * sizeof(float), cudaMemcpyHostToDevice);
	//cudaMalloc(&uploadICs, sizeof(double)*nCELLS* nStates);
	//cudaMemcpy(uploadICs, huploadICs, nCELLS* nStates * sizeof(double), cudaMemcpyHostToDevice);
	

	//assignHeterogeneity << <nSIMS, nCELLS / cellsThread >> > (Gs, gChannels);
	assignHeterogeneity << <nSIMS, cellsThread >> > (Gs, gChannels);
//	initialConditions << <nSIMS, nCELLS / cellsThread >> >(dVars, uploadICs);
	initialConditions << <nSIMS, cellsThread >> >(dVars, uploadICs);
	//initialConditions << <nSIMS, nCELLS >> >(dVars);

	cudaMemcpy(hVars, dVars, numBytes, cudaMemcpyDeviceToHost);

	//for (int j = 0;j<nCELLS *nSIMS;j++) {
	//printf("V: %f, C1: %f, Ki: %f\n", hVars[j].V, hVars[j].C1_Na, hVars[j].Ki);
	//}

	FILE *fV = fopen("norm_F_V_02_R1e4_50x50_AP", "w");
	FILE *ft = fopen("norm_F_T_02_R1e4_50x50_AP", "w");
	FILE *fSimulation = fopen("F_SimTime_20s", "w");
	FILE *fIion = fopen("CA05_F_Iion_01_R1000_iso2", "w"); 
	FILE *fInet = fopen("CA05_F_Inet_01_R1000_iso2", "w");
	//FILE *fStates = fopen("ICsToAssign_s01", "w");

	int index = 0;

//	double* V_array;
//	double* t_array;


	double* dev_ion_currents;
	double* dev_Inet;

	cell* dev_x_temp;
	double* host_Vtemp;
	double* dev_Vtemp;

	double* host_ion_currents;
	double* host_Inet;

	cudaEvent_t start, stop;
	float elapsedTime;
	double begin_time;
	double end_time;
	// Time Step Variables
	double step = 0.00001;//0.000002;//s;
	double tend = 2 ;
	int iterations = tend / step;
	double skip_time_value = 0.00125;// s 0.00025
	int skip_timept = skip_time_value / step; // skipping time points in voltage array & time array
	int total_timepts = iterations / skip_timept;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	// results of the computeState kernel
//	V_array = (double*)malloc(sizeof(double)*(total_timepts*nCELLS*nSIMS));
//	t_array = (double*)malloc(sizeof(double)*(total_timepts*nSIMS));

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	cudaMalloc((void**)&dev_x_temp, numBytes);//cudaMalloc(&dev_x_temp, sizeof(cell)*size);
	//cudaMalloc((void**)&dev_ion_currents, numBytes);//cudaMalloc(&dev_ion_currents, sizeof(double)*nCELLS*nSIMS);
	//cudaMalloc((void**)&dev_Inet, numBytes);

	cell* host_x_temp;
	host_x_temp = (cell*)malloc(numBytes);
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	host_Vtemp = (double*)malloc(sizeof(double)*N);
	cudaMalloc((void**)&dev_Vtemp, sizeof(double)*N);


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////	to export ion currents	///////////////////////////////////////////////////////////////////
	host_ion_currents = (double*)malloc(sizeof(double)*N);
	cudaMalloc((void**)&dev_ion_currents, sizeof(double)*N);

	host_Inet = (double*)malloc(sizeof(double)*N);
	cudaMalloc((void**)&dev_Inet, sizeof(double)*N);
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////


	while (time<iterations) {


		//	while (time<iterations) {

				if (time == 0) {
					printf("\n");
					printf("\n");
					printf("\n");
					printf("\n");
					printf("                     ... Running Fabbri 2D simulation ...                       \n ");
					printf("\n");
					printf("\n");
					printf("\n");
					printf("\n");
				}
				if (time % 100000 < 5) {
					printf("				%d/%d (%f percent) done .... \n", time, iterations, (100.0*time) / iterations);
				}

		computeState << <nSIMS,  cellsThread >> > (dVars, dev_ion_currents, step, dev_x_temp, Gs);

		//computeState << <nSIMS, nCELLS/cellsThread >> >(dVars, dev_ion_currents, step, dev_x_temp, Gs);//time, 
		updateState << <nSIMS, cellsThread >> >(dVars, dev_x_temp);//, num_cells, cells_per_thread
		computeVoltage << <nSIMS, cellsThread >> >(dVars, dev_Vtemp, dev_ion_currents, step, time, dev_Inet);
		updateVoltage << <nSIMS, cellsThread >> >(dVars, dev_Vtemp);
//cellsThread



		//update Voltage and time arrays and write data to file
		////////////////////////////////////////////////////////////////////////////////////////////////////////////
		cudaMemcpy(host_x_temp, dev_x_temp, numBytes, cudaMemcpyDeviceToHost);
		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		cudaMemcpy(host_Vtemp, dev_Vtemp, N * sizeof(double), cudaMemcpyDeviceToHost);

		cudaMemcpy(host_ion_currents, dev_ion_currents, N * sizeof(double), cudaMemcpyDeviceToHost);

		cudaMemcpy(host_Inet, dev_Inet, N * sizeof(double), cudaMemcpyDeviceToHost);
		
		if (time%skip_timept == 0) {//&& (time*step)>=20
			for (int j = 0;j < nCELLS ;j++) {//*nSIMS
		//	printf("time: %d,V: %f,y: %f,Nai: %f, Casub: %f,fCa: %f, Cai: %f, m: %f, h: %f, dT: %f\n",time, host_Vtemp[j], host_x_temp[j].y, host_x_temp[j].Nai, host_x_temp[j].Casub,  host_x_temp[j].fCa, host_x_temp[j].Cai, host_x_temp[j].m, host_x_temp[j].h, host_x_temp[j].dT);
				//V_array[(j*(iterations / skip_timept)) + index] = host_Vtemp[j];

				fprintf(fV, "%.15f\t ", host_Vtemp[j]);

				//printf("time=%f,V=%f\n", time*step, host_Vtemp[j]);

				if (saveCURRENTS == 1) {
				//	printf("Iion: %f, Inet: %f \n ", host_ion_currents[j],host_Inet[j]);
					fprintf(fIion, "%.15f\t ", host_ion_currents[j]);
					fprintf(fInet, "%.15f\t ", host_Inet[j]);
					//fprintf(fCurrents, "%.15f %.15f\n ",
						//host_ion_currents[j],
						//host_Inet[j]);
				}
				

			}

			//cudaMemcpy(host_Vtemp, dev_Vtemp, num_cells*simulations * sizeof(double), cudaMemcpyDeviceToHost);
			//if (time%skip_timept == 0) {
			//for (i = 0;i<num_cells*simulations;i++) {
			//V_array[(i*(iterations / skip_timept)) + index] = host_Vtemp[i];
			//fprintf(fV, "%f\t ", host_Vtemp[i]);
			//}

			if (saveCURRENTS == 1) {
				fprintf(fIion, "\n");
				fprintf(fInet, "\n");
			}
			
			fprintf(fV, "\n");
			
			fprintf(ft, "%f \n", time*step);
			
			//for (int i = 0;i<nSIMS;i++) {
				//t_array[(index*nSIMS) + i] = time*step;
				
			//}
			//index++;
		}
		time++;
		//{'I', 'O', 'R1', 'RI', 'fCMi', 'fCMs', 'fCQ', 'fTC', 'fTMC', 'fTMM', 
		//'Ca_jsr', 'Ca_nsr', 'Ca_sub', 'Cai', 'V_ode', 'Nai_', 'dL', 'fCa', 'fL', 'dT', 'fT', 'a', 'paF', 'paS', 'piy', 'n', 'r_Kur', 's_Kur', 'h', 'm', 'y', 'q', 'r'};
	 	/*if (time == iterations-1) {
			for (int j = 0; j < nCELLS; j++) {
				printf("states: %f\n ", host_x_temp[j].Nai);
				fprintf(fStates, "%.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f\n ", 
					host_x_temp[j].I,
					host_x_temp[j].O,
					host_x_temp[j].R,
					host_x_temp[j].RI,
					host_x_temp[j].fCMi,
					host_x_temp[j].fCMs,
					host_x_temp[j].fCQ,
					host_x_temp[j].fTC,
					host_x_temp[j].fTMC,
					host_x_temp[j].fTMM,
					host_x_temp[j].Cajsr,
					host_x_temp[j].Cansr,
					host_x_temp[j].Casub,
					host_x_temp[j].Cai,

					host_Vtemp[j],

					host_x_temp[j].Nai,
					host_x_temp[j].dL,
					host_x_temp[j].fCa,
					host_x_temp[j].fL,
					host_x_temp[j].dT,
					host_x_temp[j].fT,
					host_x_temp[j].a,
					host_x_temp[j].paF,
					host_x_temp[j].paS,	
					host_x_temp[j].piy,
					host_x_temp[j].n,
					host_x_temp[j].rkur,
					host_x_temp[j].skur,
					host_x_temp[j].h,
					host_x_temp[j].m,
					host_x_temp[j].y,				
					host_x_temp[j].q,
					host_x_temp[j].r);
			}
		
			//fprintf(fStates,"%.15f\t",host_x_temp[j])
		} */
	}

	//fprintf(fV, "]; \n");

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cudaFree(dev_ion_currents);
	cudaFree(dev_Inet);
	cudaFree(dev_x_temp);
	free(host_Vtemp);
	cudaFree(dev_Vtemp);

	printf("Elapsed Time = %f s \n", elapsedTime / 1000);
	fprintf(fSimulation,"%f", elapsedTime / 1000);
	fprintf(fSimulation, "s\n");

	printf("\n");
	printf("Calculating Simulation outputs...\n");
	printf("\n");

	
	
	
	
	free(hVars);
	////////////////////////////////////////////////////////////////////////////
//	free(host_x_temp);
	/////////////////////////////////////////////////////////////////////////
	cudaFree(dVars);
	cudaFree(dev_x_temp);
	

	

}

