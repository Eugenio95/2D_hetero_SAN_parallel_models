///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////	code implemented by Chiara Campana ---- last update 11/12/20	///////////////////////////////////////////////////////////
///////////////////////////////////			 Severi model - multicellular implementation		//////////////////////////////////////////////////////////////////
/////////////////////////////////////	 This program implements 1D and 2D models of SAN cell	//////////////////////////////////////////////////////////////////
/////////////////////////////////////	Severi, S., Fantini, M., Charawi, L.A. and DiFrancesco, D. (2012)	/////////////////////////////////////////////////////
// An updated computational model of rabbit sinoatrial action potential to investigate the mechanisms of heart rate modulation. /////////////////////////////////
/////////////////////////////////////	The Journal of Physiology, 590: 4483-4499. doi:10.1113/jphysiol.2012.229435	 ///////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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

//#include <fenv.h>
//feenableexcept(FE_INVALID | FE_OVERFLOW);
using namespace std;
#define M_PI 3.14159265358979323846 // pi
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////// max number of cells per block = 500, if more than 500 cells then change cellsThread and nSIMS /////////////////////////////////////
#define nSIMS 10			// number of Blocks = gridSize
#define nCELLS 2500		// number of cells in a fiber/tissue - in this program = total number of threads. It should be set = (max nthreads per block=nSIMS) * cellsThread ;
#define cellsThread 250	// cells for each thread
#define sim2D 0		// 0 for SC simulations, 1 for 2D tissue simulations
#define WIDTH 50
#define LENGTH 50		// must be length*width = nCELLS for 2D tissue simulations
#define nSTATES 33		//31+1 for step implementation//34 w BAPTA simulation// make sure it correponds to number of state variables in struct cell = states
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ double updateSingleState(double oldval, double cval, double tstep) //this function doesn't need to be global 
{
	if (!isinf(oldval + cval * tstep) && !isnan(oldval + cval * tstep)) {
		return oldval + cval*tstep;
	}
	else { return oldval ; }
}


typedef struct
{
	double I, O, R1, RI, fCMi, fCMs, fCQ, fTC, fTMC, fTMM, Ca_jsr, Ca_nsr;
	double Ca_sub_, Cai_, V, Nai_, dL, fCa, fL, dT, fT, a, paF, paS, piy, n, h, m, y, q, r;
	double fBAPTA, fBAPTA_sub;
	//double max_state_change;
} cell;

typedef struct 
{	// all parameters to be varied to include heterogeneity should be part of this struct
	double g_CaL, g_CaT, g_Kr, g_Ks, g_to, g_if, i_NaK_max, K_NaCa; //, P_up_basal;
	double g_KACh,g_Na;		//11 channels
} cellGs;

__global__ void assignHeterogeneity(cellGs* Gs, float* gChannels) {		
	//upload vectors w heterogeneity from MATLAB 
	//int idx = cellsThread*threadIdx.x;
	//int maxIdx = idx + cellsThread;
	//for (;idx < maxIdx;idx++) {
        int idx = blockIdx.x*blockDim.x+threadIdx.x;
		Gs[idx].g_CaL = gChannels[idx + 0 * nCELLS];
		Gs[idx].g_CaT = gChannels[idx + 1 * nCELLS];
		Gs[idx].g_Kr = gChannels[idx + 2 * nCELLS];
		Gs[idx].g_Ks = gChannels[idx + 3 * nCELLS];
		Gs[idx].g_to = gChannels[idx + 4 * nCELLS];
		Gs[idx].g_if = gChannels[idx + 5 * nCELLS];
		Gs[idx].i_NaK_max = gChannels[idx + 6 * nCELLS];
		Gs[idx].K_NaCa = gChannels[idx + 7 * nCELLS];
		//Gs[idx].P_up_basal = gChannels[idx + 8 * nCELLS];
		Gs[idx].g_KACh = gChannels[idx + 8 * nCELLS];	//not used
		Gs[idx].g_Na = gChannels[idx + 9 * nCELLS];
//	}
	//printf("Gs[0].g_CaL=%e\n", Gs[0].g_CaL);
	//printf("Gs[1].g_CaL=%e\n", Gs[1].g_CaL);
	//printf("Gs[0].g_CaT=%e\n", Gs[0].g_CaT);
	//printf("Gs[1].g_CaT=%e\n", Gs[1].g_CaT);
}

__global__ void initialConditions(cell* vars, float* uploadICs) {

	//int idx = cellsThread*threadIdx.x;
//	int maxIdx = idx + cellsThread;
	//for (;idx < maxIdx;idx++) {
		//int idx = blockIdx.x*blockDim.x + cellsThread*threadIdx.x;
int idx = blockIdx.x*blockDim.x + threadIdx.x;
//	double I, O, R1, RI, fCMi, fCMs, fCQ, fTC, fTMC, fTMM, Ca_jsr, Ca_nsr;
//	double Ca_sub_, Cai_, V, Nai_, dL, fCa, fL, dT, fT, a, paF, paS, piy, n, h, m, y, q, r;
//	double fBAPTA, fBAPTA_sub;
		vars[idx].I = uploadICs[idx+0*nCELLS];//7.86181717518e-08;	
		vars[idx].O = uploadICs[idx+1*nCELLS];//1.7340201253e-07;
		vars[idx].R1 = uploadICs[idx+2*nCELLS];//0.912317231017262;	
		vars[idx].RI = uploadICs[idx+3*nCELLS];//0.211148145512825;	
		vars[idx].fCMi = uploadICs[idx+4*nCELLS];//0.0373817991524254;
		vars[idx].fCMs = uploadICs[idx+5*nCELLS];//0.054381370046;		
		vars[idx].fCQ = uploadICs[idx+6*nCELLS];//0.299624275428735;	
		vars[idx].fTC = uploadICs[idx+7*nCELLS];//0.0180519400676086;	
		vars[idx].fTMC = uploadICs[idx+8*nCELLS];//0.281244308217086;	
		vars[idx].fTMM = uploadICs[idx+9*nCELLS];//0.501049376634;
		vars[idx].Ca_jsr = uploadICs[idx+10*nCELLS];//0.316762674605;		
		vars[idx].Ca_nsr = uploadICs[idx+11*nCELLS];//1.05386465080816;
		vars[idx].Ca_sub_ = uploadICs[idx+12*nCELLS];//1.0e-05;		
		vars[idx].Cai_ = uploadICs[idx+13*nCELLS];//1.0e-05;

		vars[idx].fBAPTA = uploadICs[idx + 14 * nCELLS];
		vars[idx].fBAPTA_sub = uploadICs[idx + 15 * nCELLS];

		vars[idx].V = uploadICs[idx+16*nCELLS];//-52;

		vars[idx].Nai_ = uploadICs[idx+17*nCELLS];//7.5;
		vars[idx].dL = uploadICs[idx+18*nCELLS];//0.0;					
		vars[idx].fCa = uploadICs[idx+19*nCELLS];//0.697998543259722;	
		vars[idx].fL = uploadICs[idx+20*nCELLS];//0.497133507285601;	
		vars[idx].dT = uploadICs[idx+21*nCELLS];//0.0;					
		vars[idx].fT = uploadICs[idx+22*nCELLS];//0.0;
		vars[idx].a = uploadICs[idx+23*nCELLS];//0.0;						
		vars[idx].paF = uploadICs[idx+24*nCELLS];//0.0990510403258968;	
		vars[idx].paS = uploadICs[idx+25*nCELLS];//0.322999177802891;	
		vars[idx].piy = uploadICs[idx+26*nCELLS];//0.705410877258545;	
		vars[idx].n = uploadICs[idx+27*nCELLS];//0.0;
		vars[idx].h = uploadICs[idx+28*nCELLS];//1.3676940140066e-05;		
		vars[idx].m = uploadICs[idx+29*nCELLS];//0.440131579215766;	
		vars[idx].y = uploadICs[idx+30*nCELLS];//0.181334538702451;	
		vars[idx].q = uploadICs[idx+31*nCELLS];//0.506139850982478;	
		vars[idx].r = uploadICs[idx+32*nCELLS];//0.0144605370597924;

			

		//vars[idx].max_state_change = 0.0;
	//}

	// check that all variables are initialized properly 
	// printf("vars[0].V=%e\n", vars[0].V);
	// printf("vars[1].RI=%e\n", vars[1].RI);
	// printf("vars[999].RI=%e\n", vars[999].RI);
}

__global__ void computeState(cell* x, double* ion_current, double step, cell* x_temp,  cellGs* Gs, int time) {
//int time,
	//double cur_max_state_change = 0;

//	int idx = cellsThread*threadIdx.x;
	//int maxIdx = idx + cellsThread;
	int cell_num ;
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
		cell_num =  idx;//(blockIdx.x * blockDim.x) +
	//for (;idx < maxIdx;idx++) {
		//cell_num = (blockIdx.x * blockDim.x) + idx;
		//	int	cell_num = (blockIdx.x * blockDim.x) + cellsThread*threadIdx.x;
		//	int idx = blockIdx.x * blockDim.x + cellsThread*threadIdx.x;
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//	all constants first	////////////////////////////////////////////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		double MDDS, t0, Vm_fit, t_fit, CCC, i;

		double EC50_SR, HSR, MaxSR, MinSR, kiCa, kim, koCa, kom, ks, CM_tot, CQ_tot, Mgi, TC_tot;
		double TMC_tot, kb_CM, kb_CQ, kb_TC, kb_TMC, kb_TMM, kf_CM, kf_CQ, kf_TC, kf_TMC, kf_TMM;
		double T1, kbBAPTA, kfBAPTA, tauBAPTAdiff, K_up, P_up_basal, tau_dif_Ca, tau_tr, L_cell;
		double L_sub, R_cell, V_i_part, V_jsr_part, V_nsr_part, Cao, Ki, Ko, Nao, C, F, R2, T2;
		double clamp_mode, T3, ACh, Cs_5_mM, Iso_1_uM, Iva_3_uM, Rya_3_uM, V_holding, V_test;
		double t_holding, t_test, Km_fCa, alpha_fCa, g_CaL, g_CaT, g_KACh, g_Kr, shift, gg, K1ni;
		double K1no, K2ni, K2no, K3ni, K3no, K_NaCa, Kci, Kcni, Kco, Qci, Qco, Qn, Km_Kp, Km_Nap;
		double i_NaK_max, delta_m, g_Na, g_Cab, g_Kb, g_Nab, Blocco, Km_f, g_to, g_if;
		double g_Ks;
		//%double j_SRCarel;
		EC50_SR = 0.45;   // millimolar(in Ca_SR_release)
		HSR = 2.5;   // dimensionless(in Ca_SR_release)
		MaxSR = 15;   // dimensionless(in Ca_SR_release)
		MinSR = 1;   // dimensionless(in Ca_SR_release)
		kiCa = 500;   // per_millimolar_second(in Ca_SR_release)
		kim = 5;   // per_second(in Ca_SR_release)
		koCa = 10000;   // per_millimolar2_second(in Ca_SR_release)
		kom = 60;   // per_second(in Ca_SR_release)
		ks = 250000000;   // per_second(in Ca_SR_release)
		CM_tot = 0.045;   // millimolar(in Ca_buffering)
		CQ_tot = 10;   // millimolar(in Ca_buffering)
		Mgi = 2.5;   // millimolar(in Ca_buffering)
		TC_tot = 0.031;   // millimolar(in Ca_buffering)
		TMC_tot = 0.062;   // millimolar(in Ca_buffering)
		kb_CM = 542;   // per_second(in Ca_buffering)
		kb_CQ = 445;   // per_second(in Ca_buffering)
		kb_TC = 446;   // per_second(in Ca_buffering)
		kb_TMC = 7.51;   // per_second(in Ca_buffering)
		kb_TMM = 751;   // per_second(in Ca_buffering)
		kf_CM = 227700;   // per_millimolar_second(in Ca_buffering)
		kf_CQ = 534;   // per_millimolar_second(in Ca_buffering)
		kf_TC = 88800;   // per_millimolar_second(in Ca_buffering)
		kf_TMC = 227700;   // per_millimolar_second(in Ca_buffering)
		kf_TMM = 2277;   // per_millimolar_second(in Ca_buffering)
		T1 = 6.928;   // second(T in Ca_dynamics)
		kbBAPTA = 119.38;   // per_second(in Ca_dynamics)
		kfBAPTA = 940000;   // per_millimolar_second(in Ca_dynamics)
		tauBAPTAdiff = 0.6;   // second(in Ca_dynamics)
		K_up = 0.0006;   // millimolar(in Ca_intracellular_fluxes)
		
		tau_dif_Ca = 4.0e-05;   // second(in Ca_intracellular_fluxes)
		tau_tr = 0.04;   // second(in Ca_intracellular_fluxes)
		L_cell = 70;   // micrometre(in Cell_parameters)
		L_sub = 0.02;   // micrometre(in Cell_parameters)
		R_cell = 4;   // micrometre(in Cell_parameters)
		V_i_part = 0.46;   // dimensionless(in Cell_parameters)
		V_jsr_part = 0.0012;   // dimensionless(in Cell_parameters)
		V_nsr_part = 0.0116;   // dimensionless(in Cell_parameters)
		Cao = 1.8;   // millimolar(in Ionic_values)
		Ki = 140;   // millimolar(in Ionic_values)
		Ko = 5.4;   // millimolar(in Ionic_values)
		Nao = 140;   // millimolar(in Ionic_values)
		C = 3.2e-5;   // microF(in Membrane)
		F = 96485.3415;   // coulomb_per_mole(in Membrane)
		R2 = 8314.472;   // joule_per_kilomole_kelvin(R in Membrane)
		T2 = 310;   // kelvin(T in Membrane)
		clamp_mode = 0;   // dimensionless(in Membrane)
		T3 = 6.928;   // second(T in Nai_concentration)
		ACh = 0;   // millimolar(in Rate_modulation_experiments)
		Cs_5_mM = 0;   // dimensionless(in Rate_modulation_experiments)
		Iso_1_uM = 0;   // dimensionless(in Rate_modulation_experiments)
		Iva_3_uM = 0;   // dimensionless(in Rate_modulation_experiments)
		Rya_3_uM = 0;   // dimensionless(in Rate_modulation_experiments)
		V_holding = -45;   // millivolt(in Voltage_clamp)
		V_test = -35;   // millivolt(in Voltage_clamp)
		t_holding = 0.5;   // second(in Voltage_clamp)
		t_test = 0.5;   // second(in Voltage_clamp)
		Km_fCa = 0.00035;   // millimolar(in i_CaL_fCa_gate)
		alpha_fCa = 0.01;   // per_second(in i_CaL_fCa_gate)
		
		shift = 0;   // millivolt(in i_Ks_n_gate)
		gg = 0.2;   // dimensionless(in i_Ks)
		K1ni = 395.3;   // millimolar(in i_NaCa)
		K1no = 1628;   // millimolar(in i_NaCa)
		K2ni = 2.289;   // millimolar(in i_NaCa)
		K2no = 561.4;   // millimolar(in i_NaCa)
		K3ni = 26.44;   // millimolar(in i_NaCa)
		K3no = 4.663;   // millimolar(in i_NaCa)
		
		Kci = 0.0207;   // millimolar(in i_NaCa)
		Kcni = 26.44;   // millimolar(in i_NaCa)
		Kco = 3.663;   // millimolar(in i_NaCa)
		Qci = 0.1369;   // dimensionless(in i_NaCa)
		Qco = 0;   // dimensionless(in i_NaCa)
		Qn = 0.4315;   // dimensionless(in i_NaCa)
		Km_Kp = 1.4;   // millimolar(in i_NaK)
		Km_Nap = 14;   // millimolar(in i_NaK)
		
		delta_m = 1.0e-05;   // millivolt(in i_Na_m_gate)
		
		Blocco = 0;   // dimensionless(in i_f)
		Km_f = 45;   // millimolar(in i_f)
		
		double p_dL, p_fL, p_fCa, p_dT, p_fT, p_a, p_paS, p_paF, p_pi, p_n, p_y, p_q, p_r;
		p_dL = 1;
		p_fL = 1;
		p_fCa = 1;
		p_dT = 1;
		p_fT = 1;
		p_a = 1;
		p_paS = 1;
		p_paF = 1;
		p_pi = 1;
		p_n = 1;
		p_y = 1;
		p_q = 1;
		p_r = 1;

		double Vshift_dL, Vshift_fL, Vshift_dT, Vshift_fT, Vshift_pa, Vshift_pi;
		double Vshift_n, Vshift_y, Vshift_q, Vshift_r;
		Vshift_dL = 0;
		Vshift_fL = 0;
		Vshift_dT = 0;
		Vshift_fT = 0;
		Vshift_pa = 0;
		Vshift_pi = 0;
		Vshift_n = 0;
		Vshift_y = 0;
		Vshift_q = 0;
		Vshift_r = 0;
	/////////////////////////////////////////////////		heterogeneous channel conductivities		////////////////////////////////////////////
		/*g_CaL = 0.2;   // nanoA_per_millimolar(in i_CaL)
		g_CaT = 0.02;   // nanoA_per_millimolar(in i_CaT)
		g_Kr = 0.0011;//		0.0021637;   // microS(in i_Kr)
		g_Ks = 0.008288;
		g_to = 0.002;   // microS(in i_to)
		g_if = 0.03;  // microS(in i_f)
		i_NaK_max = 0.063;   // nanoA(in i_NaK)
		K_NaCa = 4;   // nanoA(in i_NaCa)
		
		g_KACh = 0.00864;   // microS(in i_KACh)
		g_Na = 0.0125;   // microS(in i_Na)
		*/
        P_up_basal = 12;   // millimolar_per_second(in Ca_intracellular_fluxes)
		//////////////////////////////////////////////////////////////////////////////////////////

		g_CaL = Gs[cell_num].g_CaL;//0.2;// 
	 	//g_CaL = g_CaL*0.5;
		
		//////////////////////////////////////////////////////////////////////////////////////////
		g_CaT =Gs[cell_num].g_CaT;// 0.02;// 
		////////////////////////////////////////

		g_Kr = Gs[cell_num].g_Kr;//0.0021637;// 
		//g_Kr = 1.2*g_Kr;

		/////////////////////////////////////
		g_Ks =  Gs[cell_num].g_Ks;//0.0016576;//
		g_to = Gs[cell_num].g_to;//0.002;// 
		g_if = Gs[cell_num].g_if;//0.06;// 
		i_NaK_max = Gs[cell_num].i_NaK_max;//0.063;// 
		K_NaCa = Gs[cell_num].K_NaCa;//4;// 
		//P_up_basal = Gs[cell_num].P_up_basal;//
		g_KACh = Gs[cell_num].g_KACh;//0.00864;// 
		g_Na = Gs[cell_num].g_Na;//0.0125;// 
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////	
		g_Cab = 0;   // microS(in i_b_Ca)
		g_Kb = 0;   // microS(in i_b_K)
		g_Nab = 0;   // microS(in i_b_Na)
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////	
	//////////////////////////////////////////////////////				all algebraic equations				//////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		double O_, Ca_sub;
		if (Rya_3_uM > 0) { O_ = 0.7; }
		else { O_ = x[cell_num].O; }
		if (time < -5.882) { Ca_sub = 0.000105; }
		else { Ca_sub = x[cell_num].Ca_sub_; }

		double j_SRCarel = ks*x[cell_num].O*(x[cell_num].Ca_jsr - x[cell_num].Ca_sub_);
		double kCaSR = MaxSR - (MaxSR - MinSR) / (1 + pow((EC50_SR / x[cell_num].Ca_jsr), HSR));
		double koSRCa = koCa / kCaSR;
		double kiSRCa = kiCa*kCaSR;
		/*dR1 = kim*RI - kiSRCa*Ca_sub*R1 - (koSRCa*Ca_sub^2*R1 - kom*O);
		dO = koSRCa*Ca_sub^2*R1 - kom*O - (kiSRCa*Ca_sub*O - kim*I);
		dI = kiSRCa*Ca_sub*O - kim*I - (kom*I - koSRCa*Ca_sub^2*RI);
		dRI = kom*I - koSRCa*Ca_sub^2*RI - (kim*RI - kiSRCa*Ca_sub*R1);*/
		double Cai;
		if (time < -5.882) { Cai = 0.00008; }
		else { Cai = x[cell_num].Cai_; }
		//printf("Cai=%f\n", Cai);
		double delta_fTC = kf_TC*x[cell_num].Cai_*(1.0 - x[cell_num].fTC) - kb_TC*x[cell_num].fTC;
		double delta_fTMC = kf_TMC*x[cell_num].Cai_*(1.0 - (x[cell_num].fTMC + x[cell_num].fTMM)) - kb_TMC*x[cell_num].fTMC;
		double delta_fTMM = kf_TMM*Mgi*(1.0 - (x[cell_num].fTMC + x[cell_num].fTMM)) - kb_TMM*x[cell_num].fTMM;
		double delta_fCMi = kf_CM*x[cell_num].Cai_*(1.0 - x[cell_num].fCMi) - kb_CM*x[cell_num].fCMi;
		double delta_fCMs = kf_CM*x[cell_num].Ca_sub_*(1.0 - x[cell_num].fCMs) - kb_CM*x[cell_num].fCMs;
		double delta_fCQ = kf_CQ*x[cell_num].Ca_jsr*(1 - x[cell_num].fCQ) - kb_CQ*x[cell_num].fCQ;
		//printf("delta_fTC=%f,delta_fTMC=%f,delta_fTMM=%f,delta_fCMi=%f,delta_fCMs=%f,delta_fCQ=%f\n", delta_fTC, delta_fTMC, delta_fTMM, delta_fCMi, delta_fCMs, delta_fCQ);
		/*if (time < -T1)
		BAPTA = 10;
		else
		BAPTA = 0;
		end;

		dBAPTAin = (BAPTA - BAPTAin) / tauBAPTAdiff;*/
		double j_Ca_dif = (x[cell_num].Ca_sub_ - x[cell_num].Cai_) / tau_dif_Ca;
		double V_sub = 0.000000001 * 2 * M_PI*L_sub*(R_cell - L_sub / 2.0)*L_cell;
		double b_up;
		if (Iso_1_uM > 0) { b_up = -0.25; }
		else if (ACh > 0) { b_up = 0.7*ACh / (0.00009 + ACh); }
		else { b_up = 0; }
		double P_up = P_up_basal*(1 - b_up);
		double j_up = P_up / (1 + K_up / x[cell_num].Cai_);
		double V_cell = 0.000000001*M_PI*pow(R_cell, 2)*L_cell;
		double V_nsr = V_nsr_part*V_cell;
		double V_i = V_i_part*V_cell - V_sub;
		//dCai_ = 1*(j_Ca_dif*V_sub - j_up*V_nsr) / V_i - (CM_tot*delta_fCMi + TC_tot*delta_fTC + TMC_tot*delta_fTMC) - (kfBAPTA*Cai*(BAPTAin - fBAPTA) - kbBAPTA*fBAPTA);
		//dfBAPTA = kfBAPTA*x[cell_num].Cai_*(BAPTAin - fBAPTA) - kbBAPTA*fBAPTA;
		double V_jsr = V_jsr_part*V_cell;
		//printf("V_jsr=%e\n", V_jsr);
		double V_clamp;
		if ((time > t_holding) && (time < t_holding + t_test)) { V_clamp = V_test; }
		else { V_clamp = V_holding; }
		//double V
		//if (clamp_mode >= 1) {V = V_clamp;}
		//else {V = x[cell_num].V;}


		double RTONF = R2*T2 / F;
		double i_siCa = 2 * g_CaL*(x[cell_num].V - 0.0) / (RTONF*(1.0 - exp(-1.0 * (x[cell_num].V - 0.0) * 2 / RTONF)))*(x[cell_num].Ca_sub_ - Cao*exp(-2.0 * (x[cell_num].V - 0.0) / RTONF))*x[cell_num].dL*x[cell_num].fL*x[cell_num].fCa;
		double i_CaT = 2 * g_CaT*x[cell_num].V / (RTONF*(1 - exp(-1 * x[cell_num].V * 2 / RTONF)))*(x[cell_num].Ca_sub_ - Cao*exp(-2 * x[cell_num].V / RTONF))*x[cell_num].dT*x[cell_num].fT;
		double E_Ca_2 = 0.5*RTONF*log(Cao / x[cell_num].Ca_sub_);
		//printf("ECa_2=%f,Cao=%f,Ca_sub=%f\n", log(Cao / Ca_sub),Cao,Ca_sub);
		double i_b_Ca = 0.0;//g_Cab*(x[cell_num].V - E_Ca_2);
		double k32 = exp(Qn*x[cell_num].V / (2 * RTONF));
		double Nai;
		if (time < -T3) { Nai = 7.5; }
		else { Nai = x[cell_num].Nai_; }


		double k43 = x[cell_num].Nai_ / (K3ni + x[cell_num].Nai_);
		double di = 1 + x[cell_num].Ca_sub_ / Kci*(1 + exp(-Qci*x[cell_num].V / RTONF) + x[cell_num].Nai_ / Kcni) + x[cell_num].Nai_ / K1ni*(1 + x[cell_num].Nai_ / K2ni*(1 + x[cell_num].Nai_ / K3ni));
		double k14 = x[cell_num].Nai_ / K1ni*x[cell_num].Nai_ / K2ni*(1 + x[cell_num].Nai_ / K3ni)*exp(Qn*x[cell_num].V / (2 * RTONF)) / di;
		double k12 = x[cell_num].Ca_sub_ / Kci*exp(-Qci*x[cell_num].V / RTONF) / di;
		double k41 = exp(-Qn*x[cell_num].V / (2 * RTONF));
		double k34 = Nao / (K3no + Nao);
		double x2 = k32*k43*(k14 + k12) + k41*k12*(k34 + k32);
		double do_ = 1 + Cao / Kco*(1 + exp(Qco*x[cell_num].V / RTONF)) + Nao / K1no*(1 + Nao / K2no*(1 + Nao / K3no));
		double k21 = Cao / Kco*exp(Qco*x[cell_num].V / RTONF) / do_;
		double k23 = Nao / K1no*Nao / K2no*(1 + Nao / K3no)*exp(-Qn*x[cell_num].V / (2 * RTONF)) / do_;
		double x1 = k41*k34*(k23 + k21) + k21*k32*(k43 + k41);
		double x3 = k14*k43*(k23 + k21) + k12*k23*(k43 + k41);
		double x4 = k23*k34*(k14 + k12) + k14*k21*(k34 + k32);

		//printf("time=%e\n,x[cell_num].Ca_sub_after_iter1=%e\n", time, x[cell_num].Ca_sub_);

		double i_NaCa = K_NaCa*(x2*k21 - x1*k12) / (x1 + x2 + x3 + x4);
		//dCa_sub_ = j_SRCarel*V_jsr / V_sub - ((i_siCa + i_CaT + i_b_Ca - 2*i_NaCa) / (2*F*V_sub) + j_Ca_dif + CM_tot*delta_fCMs) - (kfBAPTA*Ca_sub*(BAPTAin - fBAPTA_sub) - kbBAPTA*fBAPTA_sub);
		//dfBAPTA_sub = kfBAPTA*Ca_sub*(BAPTAin - fBAPTA_sub) - kbBAPTA*fBAPTA_sub;
		double j_tr = (x[cell_num].Ca_nsr - x[cell_num].Ca_jsr) / tau_tr;
		//dCa_nsr = j_up - j_tr*V_jsr / V_nsr;
		//dCa_jsr = j_tr - (j_SRCarel + CQ_tot*delta_fCQ);
		double E_Na = RTONF*log(Nao / x[cell_num].Nai_);
		double E_K = RTONF*log(Ko / Ki);
		double E_Ca_1 = 0.5*RTONF*log(Cao / x[cell_num].Ca_sub_);
		//printf("E_Na=%f,E_K=%f,E_Ca_1=%f\n", E_Na, E_K, E_Ca_1);
		if (Iva_3_uM >= 1) { g_if = g_if*(1 - 0.66); }

		double g_f_Na = 0.5*g_if;
		double ICs_on_Icontrol_3;
		if (Cs_5_mM >= 1) { ICs_on_Icontrol_3 = 10.6015 / 5 / (10.6015 / 5 + exp(-0.71*x[cell_num].V / 25)); }
		else { ICs_on_Icontrol_3 = 1; }
		double i_fNa = pow(x[cell_num].y, 2)*Ko / (Ko + Km_f)*g_f_Na*(x[cell_num].V - E_Na)*ICs_on_Icontrol_3*(1 - Blocco);

		double g_f_K = 0.5*g_if;

		double i_fK = pow(x[cell_num].y, 2)*Ko / (Ko + Km_f)*g_f_K*(x[cell_num].V - E_K)*ICs_on_Icontrol_3*(1 - Blocco);
		double i_f = i_fNa + i_fK;
		double i_b_Na = g_Nab*(x[cell_num].V - E_Na);
		double ICs_on_Icontrol_1;
		if (Cs_5_mM >= 2) { ICs_on_Icontrol_1 = 1 / (1 + 5 / (85 * exp(x[cell_num].V / RTONF))); }
		else { ICs_on_Icontrol_1 = 1.0; }


		double i_Kr = g_Kr*ICs_on_Icontrol_1*(x[cell_num].V - E_K)*(0.9*x[cell_num].paF + 0.1*x[cell_num].paS)*x[cell_num].piy;
		//printf("g_Kr=%f\n", g_Kr);
		//printf("(x[cell_num].V - E_K)*(0.9*x[cell_num].paF + 0.1*x[cell_num].paS)*x[cell_num].piy=%e\n", g_Kr*(x[cell_num].V - E_K)*(0.9*x[cell_num].paF + 0.1*x[cell_num].paS)*x[cell_num].piy);
		if (Iso_1_uM > 0) { g_Ks = 1.2*g_Ks; }



		double ICs_on_Icontrol_2;
		if (Cs_5_mM >= 2) { ICs_on_Icontrol_2 = 1 / (1 + 5 / (85 * exp(x[cell_num].V / RTONF))); }

		else { ICs_on_Icontrol_2 = 1; }


		double E_Ks = RTONF*log((Ko + 0 * Nao) / (Ki + 0 * x[cell_num].Nai_));
		//printf("E_Ks=%f\n", E_Ks);
		double i_Ks = gg*g_Ks*ICs_on_Icontrol_2*(x[cell_num].V - E_Ks)*pow(x[cell_num].n, 2);
		double i_to = g_to*(x[cell_num].V - E_K)*x[cell_num].q*x[cell_num].r;
		double Iso_aumento_g_2;
		if (Iso_1_uM > 0) { Iso_aumento_g_2 = 1.2; }
		else { Iso_aumento_g_2 = 1; }


		double i_NaK = Iso_aumento_g_2*i_NaK_max*(1 / (1 + pow((Km_Kp / Ko), 1.2)))*(1 / (1 + pow((Km_Nap / x[cell_num].Nai_), 1.3)))*(1 / (1 + exp(-(x[cell_num].V - E_Na + 110) / 20)));
		double E_mh = RTONF*log((Nao + 0.12*Ko) / (x[cell_num].Nai_ + 0.12*Ki));
		double i_Na = g_Na*pow(x[cell_num].m, 3)*x[cell_num].h*(x[cell_num].V - E_mh);
		double i_siK = 0.000365*g_CaL*(x[cell_num].V - 0) / (RTONF*(1 - exp(-1 * (x[cell_num].V - 0) / RTONF)))*(Ki - Ko*exp(-1 * (x[cell_num].V - 0) / RTONF))*x[cell_num].dL*x[cell_num].fL*x[cell_num].fCa;
		double i_siNa = 0.0000185*g_CaL*(x[cell_num].V - 0) / (RTONF*(1 - exp(-1 * (x[cell_num].V - 0) / RTONF)))*(x[cell_num].Nai_ - Nao*exp(-1 * (x[cell_num].V - 0) / RTONF))*x[cell_num].dL*x[cell_num].fL*x[cell_num].fCa;
		double ACh_block = 0.31*ACh / (ACh + 0.00009);
		double Iso_aumento_g_1;
		if (Iso_1_uM > 0) { Iso_aumento_g_1 = 1.23; }
		else { Iso_aumento_g_1 = 1; }
		double i_CaL = (i_siCa + i_siK + i_siNa)*(1 - ACh_block) * 1 * Iso_aumento_g_1;
		double i_b_K = g_Kb*(x[cell_num].V - E_K);
		double i_KACh;
		if (ACh > 0) { i_KACh = g_KACh*(x[cell_num].V - E_K)*(1 + exp((x[cell_num].V + 20) / 20))*x[cell_num].a; }
		else { i_KACh = 0; }
	
	
		double Iion = i_f + i_Kr + i_Ks + i_to + i_NaK + i_Na + i_NaCa + i_CaL + i_CaT + i_b_Na + i_b_Ca + i_b_K + i_KACh;
	
		//if (time==0) {
		/*	printf("i_siCa=%e,i_siNa=%e,i_siK=%e,i_f=%e\n",i_siCa, i_siNa, i_siK, i_f);
			printf("i_f_k =%e,i_f_na=%e,i_Kr=%e,iNa=%e,i_to=%e,iNak=%e,i_b_Na=%e,i_b_Ca=%e,i_NaCa=%e\n", i_fK,i_fNa, i_Kr, i_Na, i_to, i_NaK, i_b_Na, i_b_Ca, i_NaCa);
			printf("i_CaL =%e,i_CaT=%e,i_b_K=%e,i_KACh=%e,i_Ks=%e\n", i_CaL, i_CaT, i_b_K, i_KACh,i_Ks);
			*/
		//}

		//dV_ode = -(i_f + i_b_Na + i_b_Ca + i_Kr + i_Ks + i_to + i_NaK + i_NaCa + i_Na + i_CaL + i_CaT + i_b_K + i_KACh) / C;
		//dNai_ = -1*(i_Na + i_b_Na + i_fNa + i_siNa + 3*i_NaK + 3*i_NaCa) / (1*(V_i + V_sub)*F);
		/*
		if (Iso_1_uM > 0)
		Iso_shift_1 = -8;
		else
		Iso_shift_1 = 0;
		end;*/
		double Iso_shift_1 = 0;
		/*if (Iso_1_uM > 0)
		Iso_slope_1 = 0.69;
		else
		Iso_slope_1 = 1;
		end;*/
		double Iso_slope_1 = 1;
		//% V = Vold - Vshift_dL;
		double dL_infinity = 1 / (1 + exp(-(x[cell_num].V + 20.3 - Iso_shift_1) / (Iso_slope_1*4.2)));
		//% V = Vold;
		double adVm;
		if (x[cell_num].V == -41.8) { adVm = -41.80001; }

		else if (x[cell_num].V == 0) { adVm = 0; }

		else if (x[cell_num].V == -6.8) { adVm = -6.80001; }

		else { adVm = x[cell_num].V; }
		double alpha_dL = -0.02839*(adVm + 41.8 - Iso_shift_1) / (exp(-(adVm + 41.8 - Iso_shift_1) / 2.5) - 1) - 0.0849*(adVm + 6.8 - Iso_shift_1) / (exp(-(adVm + 6.8 - Iso_shift_1) / 4.8) - 1);
		double bdVm;
		if (x[cell_num].V == -1.8) { bdVm = -1.80001; }
		else { bdVm = x[cell_num].V; }



		double beta_dL = 0.01143*(bdVm + 1.8 - Iso_shift_1) / (exp((bdVm + 1.8 - Iso_shift_1) / 2.5) - 1);
		double tau_dL = p_dL*0.001 / (alpha_dL + beta_dL);
		//ddL = (dL_infinity - dL) / tau_dL;
		//% ? ? ?
		double 	fCa_infinity = Km_fCa / (Km_fCa + x[cell_num].Ca_sub_);

		double tau_fCa = p_fCa*0.001*fCa_infinity / alpha_fCa;
		//dfCa = (fCa_infinity - fCa) / tau_fCa;
		//% V = Vold - Vshift_fL;
		double fL_infinity = 1 / (1 + exp((x[cell_num].V + 37.4) / 5.3));
		//% V = Vold;
		double tau_fL = p_fL*0.001*(44.3 + 230 * exp(-pow(((x[cell_num].V + 36) / 10), 2)));
		//dfL = (fL_infinity - fL) / tau_fL;
		//% V = Vold - Vshift_dT;
		double dT_infinity = 1 / (1 + exp(-(x[cell_num].V + 38.3) / 5.5));
		//% V = Vold;
		double tau_dT = p_dT*0.001 / (1.068*exp((x[cell_num].V + 38.3) / 30) + 1.068*exp(-(x[cell_num].V + 38.3) / 30));
		//ddT = (dT_infinity - dT) / tau_dT;
		//% V = Vold - Vshift_fT;
		double fT_infinity = 1 / (1 + exp((x[cell_num].V + 58.7) / 3.8));
		//% V = Vold;
		double tau_fT = p_fT * 1 / (16.67*exp(-(x[cell_num].V + 75) / 83.3) + 16.67*exp((x[cell_num].V + 75) / 15.38));
		//dfT = (fT_infinity - fT) / tau_fT;
		double alpha_a = (3.5988 - 0.025641) / (1 + 0.0000012155 / pow((1 * ACh), 1.6951)) + 0.025641;
		//printf("alpha_a=%f\n", alpha_a);
		double beta_a = 10 * exp(0.0133*(x[cell_num].V + 40));
		//% ? ? ?
		double 	a_infinity = alpha_a / (alpha_a + beta_a);

		double tau_a = p_a * 1 / (alpha_a + beta_a);
		//da = (a_infinity - a) / tau_a;
		double alfapaF = 1 / (1 + exp(-(x[cell_num].V + 23.2) / 6.6)) / (0.84655354 / (37.2*exp(x[cell_num].V / 11.9) + 0.96*exp(-x[cell_num].V / 18.5)));
		double betapaF = 4 * ((37.2*exp(x[cell_num].V / 15.9) + 0.96*exp(-x[cell_num].V / 22.5)) / 0.84655354 - 1 / (1 + exp(-(x[cell_num].V + 23.2) / 10.6)) / (0.84655354 / (37.2*exp(x[cell_num].V / 15.9) + 0.96*exp(-x[cell_num].V / 22.5))));
		//% x[cell_num].V = x[cell_num].Vold - x[cell_num].Vshift_pa;
		double pa_infinity = 1 / (1 + exp(-(x[cell_num].V + 14.8) / 8.5));
		//% V = Vold;
		double tau_paS = p_paS*0.84655354 / (4.2*exp(x[cell_num].V / 17) + 0.15*exp(-x[cell_num].V / 21.6));
		double tau_paF = p_paF * 1 / (30 * exp(x[cell_num].V / 10) + exp(-x[cell_num].V / 12));
		//dpaS = (pa_infinity - paS) / tau_paS;
		//dpaF = (pa_infinity - paF) / tau_paF;
		double tau_pi = p_pi * 1 / (100 * exp(-x[cell_num].V / 54.645) + 656 * exp(x[cell_num].V / 106.157));
		//% V = Vold - Vshift_pi;
		double pi_infinity = 1 / (1 + exp((x[cell_num].V + 28.6) / 17.1));
		//% V = Vold;
		//dpiy = (pi_infinity - piy) / tau_pi;
		double Iso_slope_2, Iso_shift_2;
		if (Iso_1_uM > 0) { Iso_slope_2 = 0; }
		else { Iso_slope_2 = 0; }

		if (Iso_1_uM > 0) { Iso_shift_2 = -14; }
		else { Iso_shift_2 = 0; }


		//V = Vold - Vshift_n;
		double n_infinity = 14 / (1 + exp(-(x[cell_num].V - 40 - Iso_shift_2) / (12 + Iso_slope_2))) / (14 / (1 + exp(-(x[cell_num].V - 40 - Iso_shift_2) / (12 + Iso_slope_2))) + 1 * exp(-(x[cell_num].V - 0 - Iso_shift_2) / 45));
		//V = Vold;
		double alpha_n = 28 / (1 + exp(-(x[cell_num].V - 40 - Iso_shift_2) / 3));
		double beta_n = 1 * exp(-(x[cell_num].V - Iso_shift_2 - shift - 5) / 25);
		double tau_n = p_n * 1 / (alpha_n + beta_n);
		//dn = (n_infinity - n) / tau_n;
		double alpha_h = 20 * exp(-0.125*(x[cell_num].V + 75));
		double beta_h = 2000 / (320 * exp(-0.1*(x[cell_num].V + 75)) + 1);
		//dh = alpha_h*(1 - h) - beta_h*h;
		double E0_m = x[cell_num].V + 41;
		double alpha_m, ACh_shift, beta_m, Iso_shift_3;
		if (fabs(E0_m) < delta_m) { alpha_m = 2000; }
		else { alpha_m = 200 * E0_m / (1 - exp(-0.1*E0_m)); }
		//printf("alpha_m=%f\n", alpha_m);
		beta_m = 8000 * exp(-0.056*(x[cell_num].V + 66));
		double dm = alpha_m*(1 - x[cell_num].m) - beta_m*x[cell_num].m;

		if (ACh > 0) { ACh_shift = -1 - 9.898*pow((1 * ACh), 0.618) / (pow((1 * ACh), 0.618) + 0.00122423); }
		else { ACh_shift = 0.0; }

		if (Iso_1_uM > 0) { Iso_shift_3 = 7.5; }
		else { Iso_shift_3 = 0; }

		double tau_y = p_y*0.7166529 / (0.0708*exp(-(x[cell_num].V + 5 - ACh_shift - Iso_shift_3) / 20.2791) + 10.6*exp((x[cell_num].V - ACh_shift - Iso_shift_3) / 18));
		//V = Vold - Vshift_y;
		double y_infinity = 1 / (1 + exp((x[cell_num].V + 52.5 - ACh_shift - Iso_shift_3) / 9));
		//V = Vold;
		//dy = (y_infinity - y) / tau_y;
		//V = Vold - Vshift_q;
		double q_infinity = 1 / (1 + exp((x[cell_num].V + 49) / 13));
		//V = Vold;
		double tau_q = p_q*0.001*0.6*(65.17 / (0.57*exp(-0.08*(x[cell_num].V + 44)) + 0.065*exp(0.1*(x[cell_num].V + 45.93))) + 10.1);
		//dq = (q_infinity - q) / tau_q;
		//V = Vold - Vshift_r;
		double r_infinity = 1 / (1 + exp(-(x[cell_num].V - 19.3) / 15));
		//V = Vold;
		double tau_r = p_r*0.001*0.66*1.4*(15.59 / (1.037*exp(0.09*(x[cell_num].V + 30.61)) + 0.369*exp(-0.12*(x[cell_num].V + 23.84))) + 2.98);
		//dr = (r_infinity - r) / tau_r;


		//if (time == 0) { printf("m: %f\n", x[cell_num].m); }
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		ion_current[cell_num] = Iion;
	//	printf("Iion_pre=%e\n", Iion);
		//printf("k12: %f,x2: %f,k14: %f, x4: %f, di: %f,Casub: %f\n", k12,x2,k14,x4,di,x[cell_num].Ca_sub_);
		//printf("i_NaCa: %f,x2: %f, x4: %f, x1: %f, x3: ,k14: %f, k23: %f , k41 : %f , k21: %f,x[cell_num].V= %f\n",  i_NaCa, x2, x4,x1,x3,k14,k23,k41,k21,x[cell_num].V);
		//printf("i_siCa: %f,i_NaK: %f,i_Na: %f,i_NaCa: %f,i_CaT: %f, Iion: %f\n", i_siCa, i_NaK,i_Na,i_NaCa,i_CaT,Iion);
	
		////////////////////////////////////////////////////////////////////////////////////		COMPUTE STATES		//////////////////////////////////////////////////////////////////////////////////////////
		/*I, O, R1, RI, fCMi, fCMs, fCQ, fTC, fTMC, fTMM, BAPTAin, Ca_jsr, Ca_nsr, ...
		Ca_sub_, Cai_, fBAPTA, fBAPTA_sub, V_ode, Nai_, dL, fCa, fL, dT, fT, a, paF, paS, ...
		piy, n, h, m, y, q, r] = deal(outputcell{ : });*/
		/*double printResult;
		double ttt0 = 0.0;
		printResult = (3.5988 - 0.025641) / (1.0 + 0.0000012155 / pow(ttt0, 1.6951));
		printf("printResult=%e\n", printResult );*/


		x_temp[cell_num].I = updateSingleState(x[cell_num].I, kiSRCa*x[cell_num].Ca_sub_*x[cell_num].O - kim*x[cell_num].I - (kom*x[cell_num].I - koSRCa*pow(x[cell_num].Ca_sub_, 2)*x[cell_num].RI), step);//, &cur_max_state_change);
		x_temp[cell_num].R1 = updateSingleState(x[cell_num].R1, kim*x[cell_num].RI - kiSRCa*x[cell_num].Ca_sub_*x[cell_num].R1 - (koSRCa*pow(x[cell_num].Ca_sub_, 2)*x[cell_num].R1 - kom*x[cell_num].O), step);// &cur_max_state_change);
		x_temp[cell_num].O = updateSingleState(x[cell_num].O, koSRCa*pow(x[cell_num].Ca_sub_, 2)*x[cell_num].R1 - kom*x[cell_num].O - (kiSRCa*x[cell_num].Ca_sub_*x[cell_num].O - kim*x[cell_num].I), step);//&cur_max_state_change);
		x_temp[cell_num].RI = updateSingleState(x[cell_num].RI, kom*x[cell_num].I - koSRCa*pow(x[cell_num].Ca_sub_, 2)*x[cell_num].RI - (kim*x[cell_num].RI - kiSRCa*x[cell_num].Ca_sub_*x[cell_num].R1), step);// &cur_max_state_change);
		x_temp[cell_num].fCMi = updateSingleState(x[cell_num].fCMi, delta_fCMi, step);// &cur_max_state_change);
		x_temp[cell_num].fCMs = updateSingleState(x[cell_num].fCMs, delta_fCMs, step);//&cur_max_state_change);
		x_temp[cell_num].fCQ = updateSingleState(x[cell_num].fCQ, delta_fCQ, step);// &cur_max_state_change);
		x_temp[cell_num].fTC = updateSingleState(x[cell_num].fTC, delta_fTC, step);// &cur_max_state_change);
		x_temp[cell_num].fTMC = updateSingleState(x[cell_num].fTMC, delta_fTMC, step);// &cur_max_state_change);
		x_temp[cell_num].fTMM = updateSingleState(x[cell_num].fTMM, delta_fTMM, step);// &cur_max_state_change);
		x_temp[cell_num].Ca_jsr = updateSingleState(x[cell_num].Ca_jsr, (j_tr - (j_SRCarel + CQ_tot*delta_fCQ)), step);// &cur_max_state_change);
		x_temp[cell_num].Ca_nsr = updateSingleState(x[cell_num].Ca_nsr, (j_up - j_tr*V_jsr / V_nsr), step);// &cur_max_state_change);
		x_temp[cell_num].Ca_sub_ = updateSingleState(x[cell_num].Ca_sub_, ((j_SRCarel*V_jsr / V_sub - ((i_siCa + i_CaT + i_b_Ca - 2 * i_NaCa) / (2 * F*V_sub) + j_Ca_dif + CM_tot*delta_fCMs))),step);// &cur_max_state_change);//((j_SRCarel*V_jsr / V_sub - ((i_siCa + i_CaT + i_b_Ca - 2 * i_NaCa) / (2 * F*V_sub) + j_Ca_dif + CM_tot*delta_fCMs)))//- (kfBAPTA*x[cell_num].Ca_sub_*(x[cell_num].BAPTAin - x[cell_num].fBAPTA_sub) - kbBAPTA*x[cell_num].fBAPTA_sub)
	
	
		//if (((i_siCa + i_CaT + i_b_Ca - 2 * i_NaCa) / (2 * F*V_sub) + j_Ca_dif + CM_tot*delta_fCMs) > (j_SRCarel*V_jsr / V_sub)) {
		//	printf("j_SRCarel*V_jsr / V_sub=%f,p2=%f,curChange=%f\n", j_SRCarel*V_jsr / V_sub, ((i_siCa + i_CaT + i_b_Ca - 2 * i_NaCa) / (2 * F*V_sub) + j_Ca_dif + CM_tot*delta_fCMs), cur_max_state_change);
		//}
		x_temp[cell_num].Cai_ = updateSingleState(x[cell_num].Cai_, (j_Ca_dif*V_sub - j_up*V_nsr) / V_i - (CM_tot*delta_fCMi + TC_tot*delta_fTC + TMC_tot*delta_fTMC), step);//&cur_max_state_change);//- (kfBAPTA*x[cell_num].Cai_*(x[cell_num].BAPTAin - x[cell_num].fBAPTA) - kbBAPTA*x[cell_num].fBAPTA)
		x_temp[cell_num].Nai_ = updateSingleState(x[cell_num].Nai_, -1*(i_Na + i_b_Na + i_fNa + i_siNa + 3*i_NaK + 3*i_NaCa) / (1*(V_i + V_sub)*F), step);//&cur_max_state_change); // kept constant in Maltsev and in study by Wilders 
		//dNai_ = ;
		x_temp[cell_num].dL = updateSingleState(x[cell_num].dL, (dL_infinity - x[cell_num].dL) / tau_dL, step);// &cur_max_state_change);
		x_temp[cell_num].fCa = updateSingleState(x[cell_num].fCa, (fCa_infinity - x[cell_num].fCa) / tau_fCa, step);// &cur_max_state_change);
		x_temp[cell_num].fL = updateSingleState(x[cell_num].fL, (fL_infinity - x[cell_num].fL) / tau_fL, step);// &cur_max_state_change);
		x_temp[cell_num].fT = updateSingleState(x[cell_num].fT, (fT_infinity - x[cell_num].fT) / tau_fT,step);// &cur_max_state_change);
		x_temp[cell_num].dT = updateSingleState(x[cell_num].dT, (dT_infinity - x[cell_num].dT) / tau_dT, step);// &cur_max_state_change);
		x_temp[cell_num].a = updateSingleState(x[cell_num].a, (a_infinity - x[cell_num].a) / tau_a, step);//&cur_max_state_change);
		x_temp[cell_num].piy = updateSingleState(x[cell_num].piy, (pi_infinity - x[cell_num].piy) / tau_pi, step);// &cur_max_state_change);
		x_temp[cell_num].paS = updateSingleState(x[cell_num].paS, (pa_infinity - x[cell_num].paS) / tau_paS, step);// &cur_max_state_change);
		x_temp[cell_num].paF = updateSingleState(x[cell_num].paF, (pa_infinity - x[cell_num].paF) / tau_paF, step);// &cur_max_state_change);
		x_temp[cell_num].n = updateSingleState(x[cell_num].n, ((n_infinity - x[cell_num].n) / tau_n), step);// &cur_max_state_change);
		x_temp[cell_num].m = updateSingleState(x[cell_num].m, alpha_m*(1 - x[cell_num].m) - beta_m*x[cell_num].m, step);// &cur_max_state_change);
		x_temp[cell_num].h = updateSingleState(x[cell_num].h, alpha_h*(1 - x[cell_num].h) - beta_h*x[cell_num].h, step);// &cur_max_state_change);
		x_temp[cell_num].y = updateSingleState(x[cell_num].y, (y_infinity - x[cell_num].y) / tau_y, step);// &cur_max_state_change);
		x_temp[cell_num].q = updateSingleState(x[cell_num].q, ((q_infinity - x[cell_num].q) / tau_q), step);// &cur_max_state_change);
		x_temp[cell_num].r = updateSingleState(x[cell_num].r, ((r_infinity - x[cell_num].r) / tau_r), step);// &cur_max_state_change);
		//if (time == 20) {


		/*	printf("dI=%e\n", kiSRCa*x[cell_num].Ca_sub_*x[cell_num].O - kim*x[cell_num].I - (kom*x[cell_num].I - koSRCa*pow(x[cell_num].Ca_sub_, 2)*x[cell_num].RI));
			printf("dR1=%e\n", kim*x[cell_num].RI - kiSRCa*x[cell_num].Ca_sub_*x[cell_num].R1 - (koSRCa*pow(x[cell_num].Ca_sub_, 2)*x[cell_num].R1 - kom*x[cell_num].O));
			printf("dO=%e\n", koSRCa*pow(x[cell_num].Ca_sub_, 2)*x[cell_num].R1 - kom*x[cell_num].O - (kiSRCa*x[cell_num].Ca_sub_*x[cell_num].O - kim*x[cell_num].I));
			printf("dRI=%e\n", kom*x[cell_num].I - koSRCa*pow(x[cell_num].Ca_sub_, 2)*x[cell_num].RI - (kim*x[cell_num].RI - kiSRCa*x[cell_num].Ca_sub_*x[cell_num].R1));
			printf("dCajsr=%e\n", (j_tr - (j_SRCarel + CQ_tot*delta_fCQ)));
			printf("dCansr=%e\n", (j_up - j_tr*V_jsr / V_nsr));
			printf("dCasub=%e, Casub=%e\n", ((j_SRCarel*V_jsr / V_sub - ((i_siCa + i_CaT + i_b_Ca - 2 * i_NaCa) / (2 * F*V_sub) + j_Ca_dif + CM_tot*delta_fCMs))), x[cell_num].Ca_sub_);
			printf("dCai=%e\n", (j_Ca_dif*V_sub - j_up*V_nsr) / V_i - (CM_tot*delta_fCMi + TC_tot*delta_fTC + TMC_tot*delta_fTMC));
			printf("Nai=%e\n", x_temp[0].Nai_);
			printf("delta_fCMs=%e\n", delta_fCMs);
			//printf("i_part in Casub=%e\n", i_b_Ca);
			printf("delta_fCMi=%e, delta_fCQ=%e, delta_fTC=%e, delta_fTMC=%e, delta_fTMM=%e\n", delta_fCMi, delta_fCQ, delta_fTC, delta_fTMC, delta_fTMM);
			printf("dL=%e\n", (dL_infinity - x[cell_num].dL) / tau_dL);
			printf("dfCa=%e\n", (fCa_infinity - x[cell_num].fCa) / tau_fCa);
			printf("dfL=%e,dfT=%e,dT=%e\n",(fL_infinity - x[cell_num].fL) / tau_fL, (fT_infinity - x[cell_num].fT) / tau_fT, (dT_infinity - x[cell_num].dT) / tau_dT);
			printf("da=%e,dpiy=%e,dpaS=%e,dpaF=%e\n",(a_infinity - x[cell_num].a) / tau_a, (pi_infinity - x[cell_num].piy) / tau_pi, (pa_infinity - x[cell_num].paS) / tau_paS, (pa_infinity - x[cell_num].paF) / tau_paF);
			printf("dn=%e,dm=%e,dh=%e,dy=%e\n", ((n_infinity - x[cell_num].n) / tau_n), alpha_m*(1 - x[cell_num].m) - beta_m*x[cell_num].m, alpha_h*(1 - x[cell_num].h) - beta_h*x[cell_num].h, (y_infinity - x[cell_num].y) / tau_y);
			printf("dq=%e,dr=%e\n",((q_infinity - x[cell_num].q) / tau_q), ((r_infinity - x[cell_num].r) / tau_r));
			*/
		
			//}
	

		//x_temp[cell_num].max_state_change = cur_max_state_change;

		//printf("max state change for cell %d at t = %d = %f\n", cell_num, time, cur_max_state_change);
		//}
	//printf("xtemp[0].RI=%e\n", x_temp[0].RI);
	//printf("xtemp[1].RI=%e\n", x_temp[1].RI);
	//printf("xtemp[999].RI=%e\n", x_temp[999].RI);
}

__global__ void updateState(cell* x, cell* x_temp) {
	int idx = (blockIdx.x*blockDim.x) + threadIdx.x;
	//int idx = (blockIdx.x*blockDim.x) + cellsThread*threadIdx.x;
	//int idx = cellsThread*threadIdx.x;
	//int maxIdx = idx + cellsThread;
	//for (;idx < maxIdx;idx++) {

		//do not include Vm here!
		x[idx].I = x_temp[idx].I;
		//printf("x[idx.I]=%e\n", x[idx].I);
		x[idx].O = x_temp[idx].O;
		x[idx].R1 = x_temp[idx].R1;
		x[idx].RI = x_temp[idx].RI;
		x[idx].fCMi = x_temp[idx].fCMi;
		x[idx].fCMs = x_temp[idx].fCMs;
		x[idx].fCQ = x_temp[idx].fCQ;
		x[idx].fTC = x_temp[idx].fTC;
		x[idx].fTMC = x_temp[idx].fTMC;
		x[idx].fTMM = x_temp[idx].fTMM;
		x[idx].Ca_jsr = x_temp[idx].Ca_jsr;
		x[idx].Ca_nsr = x_temp[idx].Ca_nsr;
		x[idx].Ca_sub_ = x_temp[idx].Ca_sub_;
		//printf("x[idx.Ca_sub]=%e\n", x[idx].Ca_sub_);
		x[idx].Cai_ = x_temp[idx].Cai_;
		x[idx].Nai_ = x_temp[idx].Nai_;
		x[idx].dL = x_temp[idx].dL;
		x[idx].fCa = x_temp[idx].fCa;
		x[idx].fL = x_temp[idx].fL;
		x[idx].dT = x_temp[idx].dT;
		x[idx].fT = x_temp[idx].fT;
		x[idx].a = x_temp[idx].a;
		x[idx].paF = x_temp[idx].paF;
		x[idx].paS = x_temp[idx].paS;
		x[idx].piy = x_temp[idx].piy;
		x[idx].n = x_temp[idx].n;
		x[idx].h = x_temp[idx].h;
		x[idx].m = x_temp[idx].m;

		x[idx].y = x_temp[idx].y;
		x[idx].q = x_temp[idx].q;
		x[idx].r = x_temp[idx].r;

	//	x[idx].max_state_change = x_temp[idx].max_state_change;
	//}
	//printf("xup[0].RI=%e\n", x[0].RI);
	//printf("xup[1].RI=%e\n", x[1].RI);
	//printf("xup[999].RI=%e\n", x[999].RI);

}
__global__ void computeVoltage(cell* x, double* V, double* Iion, double step, int time){//, double stimDur, double stimAmp, int tstim) {

        
    int idx = (blockIdx.x*blockDim.x) + threadIdx.x;
	double Cm = 3.2e-5;// 5.7e-5;
		
	double	rGap = 10 * 1e6; //% MOhm
	double	gapJunct = (32 * 1e-12)*rGap;

	//int idx = (blockIdx.x*blockDim.x) + cellsThread*threadIdx.x;
	//int idx = cellsThread*threadIdx.x;
	//int maxIdx = idx + cellsThread;
	//for (;idx < maxIdx;idx++) {


	//	double stim = 0;
	//	double Istim1 = 0;
		//	double Vnet_R, Vnet_L, Vnet_U, Vnet_D;
	//	double rad = 0.0011;
	//	double deltx = 0.01;
	//	double rho;
	//	double Cm = 3.2e-5;
	//	double Rmyo;
	//	double gj;

	//	gj = 1.27;
	//	Rmyo = 526;
	//	rho = 3.14159*pow(rad, 2)*(Rmyo + 1000 / gj) / deltx; // total resistivity

	//	int stimInterval = 0;//1000;

							 //if (time%tstim > (stimDur / step)) { Istim1 = 0; }
							 //else { Istim1 = stimAmp; }
	//	if (time % (int)(stimInterval / step) <= stimDur / step) {	//&& time < endS1 / step
		//	Istim1 = stimAmp;
	//	}
	//	else { Istim1 = 0.0; }

		////////////////////////////////////////		Tissue model		/////////////////////////////////////////////////////
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

		V[idx] = updateSingleState(x[idx].V, ((1 / gapJunct)*(Vnet_R + Vnet_L + Vnet_U + Vnet_D)) - (Iion[idx] / Cm), step);
		//
	}
	else {
		////////////////////////////////////////////// SC - uncoupled cells simulations ////////////////////////////////////////////
		V[idx] = updateSingleState(x[idx].V, -(Iion[idx] / Cm), step);
		}
		

	//}
	//printf("Vcomp[0]=%e\n", V[0]);
	//printf("Vcomp[1]=%e\n", V[1]);
	//printf("Vcomp[999]=%e\n", V[999]);
}

__global__ void updateVoltage(cell* x, double* V) {

    int idx = (blockIdx.x*blockDim.x) + threadIdx.x;

	//int idx = (blockIdx.x*blockDim.x) + cellsThread*threadIdx.x;
	//int idx = cellsThread*threadIdx.x;
	//int maxIdx = idx + cellsThread;
	//for (;idx < maxIdx;idx++) {

		x[idx].V = V[idx];
	//}
	//printf("Vup[0]=%e\n", x[0].V);
	//printf("Vup[1]=%e\n", x[1].V);
	//printf("Vup[999]=%e\n", x[999].V);
	//printf("bldim=%d\n", blockDim.x);
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
	float* huploadICs;
	float* uploadICs;
	int nChannels = 10 ;
	hgChannels = (float *)malloc(sizeof(float)*nCELLS * nChannels);
	huploadICs = (float *)malloc(sizeof(float)*nCELLS * nSTATES);

	FILE *f1 = fopen("C:/Users/sobie/Desktop/Chiara/SAN project S/cudasparkmodel2011/nGs_S_02_50x50.txt", "rb");
	
	int errnum;
	if ((f1 = fopen("C:/Users/sobie/Desktop/Chiara/SAN project S/cudasparkmodel2011/nGs_S_02_50x50.txt", "rb")) == NULL)
	{
		errnum = errno;
		fprintf(stderr, "Value of errno: %d\n", errno);
		perror("Error printed by perror");
		fprintf(stderr, "---Error opening file: %s\n", strerror(errnum));

	}
	for (int j = 0;j<nCELLS*nChannels;j++) {
		fscanf(f1, "%f \n ", &hgChannels[j]);
		//printf("%f \n", hgChannels[j]);
	
		}
	fclose(f1);

	cudaMalloc(&gChannels, sizeof(float)*nCELLS* nChannels);
	cudaMemcpy(gChannels, hgChannels, nCELLS* nChannels * sizeof(float), cudaMemcpyHostToDevice);



	FILE *f2 = fopen("C:/Users/sobie/Desktop/Chiara/SAN project S/cudasparkmodel2011/YsGPUS.txt", "rb");//YsGPUSICS_s04S
	
	if ((f2 = fopen("C:/Users/sobie/Desktop/Chiara/SAN project S/cudasparkmodel2011/YsGPUS.txt", "rb")) == NULL)
		//if ((f1 = fopen("C:/Users/sobie/Desktop/chiara/simparall/cudasparkmodel2011/MscalesIna.txt", "rb")) == NULL)

	{
		errnum = errno;
		fprintf(stderr, "Value of errno: %d\n", errno);
		perror("Error printed by perror");
		fprintf(stderr, "Error opening file: %s\n", strerror(errnum));

	}
	for (int j = 0;j<nCELLS*nSTATES;j++) {
		fscanf(f2, "%f \n ", &huploadICs[j]);
	//	if (j>=40000 && j<=60000){
		//	printf("%f \n", huploadICs[j]);}

	}
	fclose(f2);

	cudaMalloc(&uploadICs, sizeof(float)*nCELLS*nSTATES);
	cudaMemcpy(uploadICs, huploadICs, nCELLS* nSTATES * sizeof(float), cudaMemcpyHostToDevice);


	assignHeterogeneity << <nSIMS, cellsThread >> > (Gs, gChannels);
	initialConditions << <nSIMS, cellsThread >> >(dVars, uploadICs);

	cudaMemcpy(hVars, dVars, numBytes, cudaMemcpyDeviceToHost);

	//for (int j = 0;j<nCELLS *nSIMS;j++) {
	//printf("V: %f, C1: %f, Ki: %f\n", hVars[j].V, hVars[j].C1_Na, hVars[j].Ki);
	//}

	FILE *fV = fopen("norm_S_V_02_SC_50x50", "w");
	FILE *ft = fopen("norm_S_T_02_SC_50x50", "w");
    FILE *fSimulation = fopen("S_SimTime", "w");    
	FILE *fStates = fopen("ICsToAssign_S_s02", "w");


	int index = 0;

	//double* V_array;
	//double* t_array;


	double* dev_ion_currents;
	cell* dev_x_temp;
	double* host_Vtemp;
	//double* host_Vtemp_old;

	double* dev_Vtemp;

	cudaEvent_t start, stop;
	float elapsedTime;
	float begin_time;
	float end_time;
	// Time Step Variables
	//double step_small = 0.01;
	double step = 0.00001;
	double tend = 20;
	int iterations = tend / step;
	double skip_time_value = 0.00125;//0.25 *1e-3;//0.5; //ms
	int skip_timept = skip_time_value / step; // skipping time points in voltage array & time array
	int total_timepts = iterations / skip_timept;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	

	cudaMalloc((void**)&dev_x_temp, numBytes);//cudaMalloc(&dev_x_temp, sizeof(cell)*size);
	cudaMalloc((void**)&dev_ion_currents, numBytes);//cudaMalloc(&dev_ion_currents, sizeof(double)*nCELLS*nSIMS);
	cell* host_x_temp;
	host_x_temp = (cell*)malloc(numBytes);
	host_Vtemp = (double*)malloc(sizeof(double)*N);
	cudaMalloc((void**)&dev_Vtemp, sizeof(double)*N);



	while (time<iterations) {
		
			if (time == 0) {
					printf("\n");
					printf("\n");
					printf("\n");
					printf("\n");
					printf("                     ... Running SEVERI 2D simulation ...                       \n ");
					printf("\n");
					printf("\n");
					printf("\n");
					printf("\n");
				}
				if (time % 100000 < 5) {
					printf("				%d/%d (%f percent) done .... \n", time, iterations, (100.0*time) / iterations);
				}

		//computeState << <nSIMS, nCELLS >> > (dVars, dev_ion_currents, step, dev_x_temp);//, scales
		computeState << <nSIMS, cellsThread >> > (dVars, dev_ion_currents, step, dev_x_temp, Gs,time);//time,
		updateState << <nSIMS, cellsThread >> >(dVars, dev_x_temp);//, num_cells, cells_per_thread
		computeVoltage << <nSIMS, cellsThread >> >(dVars, dev_Vtemp, dev_ion_currents, step, time);//, stimDur, stimAmp, tstim);
		updateVoltage << <nSIMS, cellsThread >> >(dVars, dev_Vtemp);

		//update Voltage and time arrays and write data to file
		cudaMemcpy(host_x_temp, dev_x_temp, numBytes, cudaMemcpyDeviceToHost);
		cudaMemcpy(host_Vtemp, dev_Vtemp, N * sizeof(double), cudaMemcpyDeviceToHost);

		if (time%skip_timept == 0) {
			for (int j = 0; j < nCELLS; j++) {//*nSIMS
			//	printf("V: %f,y: %f,Nai: %f, Casub: %f,fCa: %f, Cai: %f, m: %f, h: %f, dT: %f\n", host_Vtemp[j], host_x_temp[j].y, host_x_temp[j].Nai_, host_x_temp[j].Ca_sub_,  host_x_temp[j].fCa, host_x_temp[j].Cai_, host_x_temp[j].m, host_x_temp[j].h, host_x_temp[j].dT);
				//printf("V: %f, y; %f\n", host_Vtemp[j], host_x_temp[j].y);
				//V_array[(j*(iterations / skip_timept)) + index] = host_Vtemp[j];
				fprintf(fV, "%.15f\t ", host_Vtemp[j]);
			}

			//cudaMemcpy(host_Vtemp, dev_Vtemp, num_cells*simulations * sizeof(double), cudaMemcpyDeviceToHost);
			//if (time%skip_timept == 0) {
			//for (i = 0;i<num_cells*simulations;i++) {
			//V_array[(i*(iterations / skip_timept)) + index] = host_Vtemp[i];
			//fprintf(fV, "%f\t ", host_Vtemp[i]);
			//}
			fprintf(fV, "\n");
			fprintf(ft, "%f \n", time*step);
			//	for (int i = 0;i<nSIMS;i++) {
				//	t_array[(index*nSIMS) + i] = time*step;
				//}
			//	index++;
			//}
		}
		time++;
		if (time == iterations-1) {
		for (int j = 0; j < nCELLS; j++) {
		//	double I, O, R1, RI, fCMi, fCMs, fCQ, fTC, fTMC, fTMM, Ca_jsr, Ca_nsr;
		//	double Ca_sub_, Cai_, V, Nai_, dL, fCa, fL, dT, fT, a, paF, paS, piy, n, h, m, y, q, r;
		//	double fBAPTA, fBAPTA_sub;
		printf("states: %f\n ", host_x_temp[j].Nai_);
		fprintf(fStates, "%.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f\n ",
		host_x_temp[j].I,
		host_x_temp[j].O,
		host_x_temp[j].R1,
		host_x_temp[j].RI,
		host_x_temp[j].fCMi,
		host_x_temp[j].fCMs,
		host_x_temp[j].fCQ,
		host_x_temp[j].fTC,
		host_x_temp[j].fTMC,
		host_x_temp[j].fTMM,
		host_x_temp[j].Ca_jsr,
		host_x_temp[j].Ca_nsr,
		host_x_temp[j].Ca_sub_,
		host_x_temp[j].Cai_,
			
		host_x_temp[j].fBAPTA, 
		host_x_temp[j].fBAPTA_sub,
		
		host_Vtemp[j],

		host_x_temp[j].Nai_,
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
		host_x_temp[j].h,
		host_x_temp[j].m,
		host_x_temp[j].y,
		host_x_temp[j].q,
		host_x_temp[j].r );
		
		}

		//fprintf(fStates,"%.15f\t",host_x_temp[j])
		} 
	}

	//fprintf(fV, "]; \n");

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cudaFree(dev_ion_currents);
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
	free(host_x_temp);
	cudaFree(dVars);
	cudaFree(dev_x_temp);
	

	

}
