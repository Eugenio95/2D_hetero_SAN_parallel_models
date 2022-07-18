///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////	Code implemented by Chiara Campana ---- last update 11/12/20	///////////////////////////////////////////////////////////
///////////////////////////////////			 MaltsevLakatta Model - Multicellular Implementation		///////////////////////////////////////////////////////////
/////////////////////////////////////	 This program implements 1D and 2D models of SAN cell			///////////////////////////////////////////////////////////
/////////////////////////////////////					 Maltsev VA & Lakatta EG (2009).				///////////////////////////////////////////////////////////
// Synergism of coupled subsarcolemmal Ca2+ clocks and sarcolemmal voltage clocks confers robust and flexible pacemaker function in a novel pacemaker cell model.//
/////////////////////////////////////			Am J Physiol Heart Circ Physiol 296, H594–H615.			///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
#define M_PI 3.14159265358979323846 // pi

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////// max number of cells per block = 500, if more than 500 cells then change cellsThread and nSIMS /////////////////////////////////////
#define nSIMS 5			// number of Blocks = gridSize
#define nCELLS 2500		// number of cells in a fiber/tissue - in this program = total number of threads. It should be set = (max nthreads per block=nSIMS) * cellsThread ;
#define cellsThread 500	// cells for each thread
#define sim2D 1			// 0 for SC simulations, 1 for 2D tissue simulations
#define WIDTH 50
#define LENGTH 50		// must be length*width = nCELLS for 2D tissue simulations
#define nSTATES 29		// make sure it correponds to number of state variables in struct cell = states
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ double updateSingleState(double oldval, double cval, double tstep) //this function doesn't need to be global 
{
	if (!isinf(oldval + cval * tstep) && !isnan(oldval + cval * tstep)) {	return oldval + cval*tstep;	}
	else { return oldval; }
}

typedef struct
{	// all state variables here
	double Vm , q , r , fCMi, fCMs, fCQ, fTC, fTMC , fTMM , Ca_jsr,Ca_nsr , Ca_sub ;
	double Cai, dL, fCa, fL, dT, fT, paF, paS, pi_, n, y, qa, qi, I;
	double O, R1 ,RI;
} cell;

typedef struct 
{	// all parameters to be varied to include heterogeneity should be part of this struct
	double g_sus ,g_to , g_CaT , g_Kr, g_Ks ;
	double g_CaL , g_b_Ca , g_b_Na , g_if , g_st ;

	double i_NaK_max, kNaCa;
	//[g_sus, g_to, g_CaL, g_CaT, g_Kr, g_Ks, g_b_Ca, g_b_Na, g_if, g_st, i_NaK_max, kNaCa];
} cellGs;

__global__ void assignHeterogeneity(cellGs* Gs, float* gChannels) {		
	//upload vectors w heterogeneity from MATLAB 
	//int idx = cellsThread*threadIdx.x;
	//int maxIdx = idx + cellsThread;
	//for (;idx < maxIdx;idx++) {
        int idx = blockIdx.x*blockDim.x+threadIdx.x;
		Gs[idx].g_sus = gChannels[idx + 0 * nCELLS];
		Gs[idx].g_to = gChannels[idx + 1 * nCELLS];
		Gs[idx].g_CaL = gChannels[idx + 2 * nCELLS];
		Gs[idx].g_CaT = gChannels[idx + 3 * nCELLS];
		Gs[idx].g_Kr = gChannels[idx + 4 * nCELLS];
		Gs[idx].g_Ks = gChannels[idx + 5 * nCELLS];
		Gs[idx].g_b_Ca = gChannels[idx + 6 * nCELLS];
		Gs[idx].g_b_Na = gChannels[idx + 7 * nCELLS];
		Gs[idx].g_if = gChannels[idx + 8 * nCELLS];
		Gs[idx].g_st = gChannels[idx + 9 * nCELLS];
		Gs[idx].i_NaK_max = gChannels[idx + 10 * nCELLS];
		Gs[idx].kNaCa = gChannels[idx + 11 * nCELLS];
	//}
		//if(idx<=5){	printf("idx = %d, Gs.g_sus=%e\n", idx , Gs[idx].g_sus);	}
	//printf("Gs[1].g_sus=%e\n", Gs[1].g_sus);
	//printf("Gs[0].g_to=%e\n", Gs[0].g_to);
	//printf("Gs[1].g_to=%e\n", Gs[1].g_to);
}

__global__ void initialConditions(cell* vars, float* uploadICs) {

    int idx = blockIdx.x*blockDim.x + threadIdx.x;
	//int idx = cellsThread*threadIdx.x;
	//int maxIdx = idx + cellsThread;
	//for (;idx < maxIdx;idx++) {
		//int idx = blockIdx.x*blockDim.x + cellsThread*threadIdx.x;

	//% 1: q(dimensionless) (in AP_sensitive_currents_q_gate)
	//	% 2 : r(dimensionless) (in AP_sensitive_currents_r_gate)
	//	% 3 : Vm(millivolt) (in Vm)
	//	% 4 : fCMi(dimensionless) (in calcium_buffering)
	//	% 5 : fCMs(dimensionless) (in calcium_buffering)
	//	% 6 : fCQ(dimensionless) (in calcium_buffering)
	//	% 7 : fTC(dimensionless) (in calcium_buffering)
	//	% 8 : fTMC(dimensionless) (in calcium_buffering)
	//	% 9 : fTMM(dimensionless) (in calcium_buffering)
	//	% 10 : Ca_jsr(millimolar) (in calcium_dynamics)
	//	% 11 : Ca_nsr(millimolar) (in calcium_dynamics)
	//	% 12 : Ca_sub(millimolar) (in calcium_dynamics)
	//	% 13 : Cai(millimolar) (in calcium_dynamics)
	//	% 14 : dL(dimensionless) (in i_CaL_dL_gate)
	//	% 15 : fCa(dimensionless) (in i_CaL_fCa_gate)
	//	% 16 : fL(dimensionless) (in i_CaL_fL_gate)
	//	% 17 : dT(dimensionless) (in i_CaT_dT_gate)
	//	% 18 : fT(dimensionless) (in i_CaT_fT_gate)
	//	% 19 : paF(dimensionless) (in i_Kr_pa_gate)
	//	% 20 : paS(dimensionless) (in i_Kr_pa_gate)
	//	% 21 : pi_(dimensionless) (in i_Kr_pi_gate)
	//	% 22 : n(dimensionless) (in i_Ks_n_gate)
	//	% 23 : y(dimensionless) (in i_f_y_gate)
	//	% 24 : qa(dimensionless) (in i_st_qa_gate)
	//	% 25 : qi(dimensionless) (in i_st_qi_gate)
	//	% 26 : I(dimensionless) (in j_SRCarel)
	//	% 27 : O(dimensionless) (in j_SRCarel)
	//	% 28 : R1(dimensionless) (R in j_SRCarel)
	//	% 29 : RI(dimensionless) (in j_SRCarel)


		vars[idx].q = uploadICs[idx+0*nCELLS];//0.69424;
		vars[idx].r = uploadICs[idx+1*nCELLS];//0.0055813;
		vars[idx].Vm = uploadICs[idx+2*nCELLS];//-57.9639;
		vars[idx].fCMi = uploadICs[idx+3*nCELLS];//0.059488;
		vars[idx].fCMs = uploadICs[idx+4*nCELLS];//0.054381;
		vars[idx].fCQ = uploadICs[idx+5*nCELLS];//0.27321;
		vars[idx].fTC = uploadICs[idx+6*nCELLS];//0.029132;
		vars[idx].fTMC = uploadICs[idx+7*nCELLS];//0.43269;
		vars[idx].fTMM = uploadICs[idx+8*nCELLS];//0.50105;
		vars[idx].Ca_jsr = uploadICs[idx+9*nCELLS];//0.31676;
		vars[idx].Ca_nsr = uploadICs[idx+10*nCELLS];//1.4935;
		vars[idx].Ca_sub = uploadICs[idx+11*nCELLS];//0.00013811;
		vars[idx].Cai = uploadICs[idx+12*nCELLS];//0.00015002;
		vars[idx].dL = uploadICs[idx+13*nCELLS];//0.00058455;
		vars[idx].fCa = uploadICs[idx+14*nCELLS];//0.7114;
		vars[idx].fL = uploadICs[idx+15*nCELLS];//0.86238;
		vars[idx].dT = uploadICs[idx+16*nCELLS];//0.0050439;
		vars[idx].fT = uploadICs[idx+17*nCELLS];//0.42076;
		vars[idx].paF = uploadICs[idx+18*nCELLS];//0.14476;
		vars[idx].paS = uploadICs[idx+19*nCELLS];//0.4531;
		vars[idx].pi_ = uploadICs[idx+20*nCELLS];//0.84941;
		vars[idx].n = uploadICs[idx+21*nCELLS];//0.02646;
		vars[idx].y = uploadICs[idx+22*nCELLS];//0.11364;
		vars[idx].qa = uploadICs[idx+23*nCELLS];//0.4238;
		vars[idx].qi = uploadICs[idx+24*nCELLS];//0.44729;
		vars[idx].I = uploadICs[idx+25*nCELLS];//7.8618e-008;
		vars[idx].O = uploadICs[idx+26*nCELLS];//1.734e-007;
		vars[idx].R1 = uploadICs[idx+27*nCELLS];//0.68805;
		vars[idx].RI = uploadICs[idx+28*nCELLS];//0.31195;
	//}
	// check that all variables are initialized properly 
	// printf("vars[0].RI=%e\n", vars[0].RI);
	// printf("vars[1].RI=%e\n", vars[1].RI);
	// printf("vars[999].RI=%e\n", vars[999].RI);
}

__global__ void computeState(cell* x, double* ion_current, double step, cell* x_temp, cellGs* Gs) {

	//int idx = cellsThread*threadIdx.x;
	//int maxIdx = idx + cellsThread;
	int cell_num ;
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	//for (;idx < maxIdx;idx++) {
            
		cell_num = idx;//(blockIdx.x * blockDim.x) + 
		//	int	cell_num = (blockIdx.x * blockDim.x) + cellsThread*threadIdx.x;
		//	int idx = blockIdx.x * blockDim.x + cellsThread*threadIdx.x;
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//	all constants first	////////////////////////////////////////////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		double E_K, i_to, i_sus, q_infinity, tau_q;
		double r_infinity, tau_r, i_CaL, i_CaT, E_Na, i_f_Na, i_f_K, i_f;
		double i_st, i_Kr, E_Ks, i_Ks, i_NaK;
		double RTOnF, k32, k43, di, k14, k12, k41, k34, x2, Do, k21, k23, x1, x3, x4, i_NaCa;
		double i_b_Ca, i_b_Na, delta_fTC, delta_fTMC, delta_fTMM, delta_fCMi;
		double delta_fCMs, delta_fCQ, j_Ca_dif;
		double V_sub, j_up, V_cell, V_nsr, V_in, j_SRCarel, V_jsr, j_tr, dL_infinity;
		double adVm, alpha_dL, bdVm, beta_dL, tau_dL, fCa_infinity, tau_fCa;
		double fL_infinity, tau_fL, dT_infinity, tau_dT, fT_infinity, tau_fT;
		double pa_infinity, tau_paS, tau_paF, pi_infinity, tau_pi;
		double alpha_n, beta_n, n_infinity, tau_n;
		double y_infinity, tau_y, qa_infinity, alpha_qa, beta_qa, tau_qa;
		double alpha_qi, beta_qi, qi_infinity, tau_qi;
		double kCaSR, koSRCa, kiSRCa, Iion;

		double Cm = 32;
		double CM_tot = 0.045;
		double CQ_tot = 10.0;
		double TC_tot = 0.031;
		double TMC_tot = 0.062;
		double kb_CM = 0.542;
		double kb_CQ = 0.445;
		double kb_TC = 0.446;
		double kb_TMC = 0.00751;
		double kb_TMM = 0.751;
		double kf_CM = 227.7;
		double kf_CQ = 0.534;
		double kf_TC = 88.8;
		double kf_TMC = 227.7;
		double kf_TMM = 2.277;
		double Km_fCa = 0.00035;
		double alpha_fCa = 0.021;
		double E_CaL = 45.0;
		double E_CaT = 45.0;
		double K1ni = 395.3;
		double K1no = 1628.0;
		double K2ni = 2.289;
		double K2no = 561.4;
		double K3ni = 26.44;
		double K3no = 4.663;
		double Kci = 0.0207;
		double Kcni = 26.44;
		double Kco = 3.663;
		double Qci = 0.1369;
		double Qco = 0.0;
		double Qn = 0.4315;
		
		double Km_Kp = 1.4;
		double Km_Nap = 14.0;
		
		double VIf_half = -64.0;
		double E_st = 37.4;

		double K_up = 0.0006;
		double P_up = 0.012;
		double tau_dif_Ca = 0.04;
		double tau_tr = 40.0;
		double EC50_SR = 0.45;
		double HSR = 2.5;
		double MaxSR = 15.0;
		double MinSR = 1.0;
		double kiCa = 0.5;
		double kim = 0.005;
		double koCa = 10.0;
		double kom = 0.06;
		double ks = 250000.0;
		double Cao = 2.0;
		double F = 96485.0;
		double Ki = 140.0;
		double Ko = 5.4;
		double L_cell = 70.0;
		double L_sub = 0.02;
		double Mgi = 2.5;
		double Nai = 10.0;
		double Nao = 140.0;
		double R2 = 8314.4;
		double R_cell = 4.0;
		double T = 310.15;
		double V_in_part = 0.46;
		double V_jsr_part = 0.0012;
		double V_nsr_part = 0.0116;

		//////////////////////////////				heterogenous channel conductivities		//////////////////////////

		/*double g_sus = 0.02;
		double g_to = 0.252;
		double g_CaT = 0.1832;
		double g_Kr = 0.08113973;
		double g_Ks = 0.0259;
		double g_CaL = 0.464;
		double g_b_Ca = 0.0006;
		double g_b_Na = 0.00486;
		double g_if = 0.15;
		double g_st = 0.003;
		double i_NaK_max = 2.88;
		double kNaCa = 187.5;*/
		double g_sus = Gs[cell_num].g_sus;
		double g_to = Gs[cell_num].g_to;
		double g_CaT = Gs[cell_num].g_CaT;
		double g_Kr = Gs[cell_num].g_Kr;
		double g_Ks = Gs[cell_num].g_Ks;

		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		double g_CaL = Gs[cell_num].g_CaL;
		g_CaL = 0.7*g_CaL;

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		double g_b_Ca = Gs[cell_num].g_b_Ca;
		double g_b_Na = Gs[cell_num].g_b_Na;
		double g_if = Gs[cell_num].g_if;
		double g_st = Gs[cell_num].g_st;
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		double i_NaK_max = Gs[cell_num].i_NaK_max;
		//i_NaK_max = i_NaK_max*1.2;

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		double kNaCa = Gs[cell_num].kNaCa;
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////		all algebraic equations		/////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		E_K = R2*T / F*log(Ko / Ki);
		i_to = Cm*g_to*(x[cell_num].Vm - E_K)*(x[cell_num].q)*(x[cell_num].r);
		i_sus = Cm*g_sus*(x[cell_num].Vm - E_K)*(x[cell_num].r);

		q_infinity = 1 / (1 + exp((x[cell_num].Vm + 49) / 13));
		tau_q = 6.06 + 39.102 / (0.57*exp(-0.08*(x[cell_num].Vm + 44)) + 0.065*exp(0.1*(x[cell_num].Vm + 45.93)));

		r_infinity = 1 / (1 + exp(-(x[cell_num].Vm - 19.3) / 15));
		tau_r = 2.75352 + 14.40516 / (1.037*exp(0.09*(x[cell_num].Vm + 30.61)) + 0.369*exp(-0.12*(x[cell_num].Vm + 23.84)));

		i_CaL = Cm*g_CaL*(x[cell_num].Vm - E_CaL)*x[cell_num].dL * x[cell_num].fL * x[cell_num].fCa;
		i_CaT = Cm*g_CaT*(x[cell_num].Vm - E_CaT)* x[cell_num].dT * x[cell_num].fT;

		E_Na = R2*T / F*log(Nao / Nai);
		i_f_Na = Cm*0.3833*g_if*(x[cell_num].Vm - E_Na)*pow(x[cell_num].y, 2);
		i_f_K = Cm*0.6167*g_if*(x[cell_num].Vm - E_K)* pow(x[cell_num].y, 2);
		i_f = i_f_Na + i_f_K;

		i_st = Cm*g_st*(x[cell_num].Vm - E_st) *x[cell_num].qa * x[cell_num].qi;

		i_Kr = Cm*g_Kr*(x[cell_num].Vm - E_K) *(0.6* x[cell_num].paF + 0.4* x[cell_num].paS) * x[cell_num].pi_;

		E_Ks = R2*T / F*log((Ko + 0.12*Nao) / (Ki + 0.12*Nai));
		i_Ks = Cm*g_Ks*(x[cell_num].Vm - E_Ks) *pow(x[cell_num].n, 2);

		i_NaK = Cm*i_NaK_max / ((1 + pow((Km_Kp / Ko), 1.2))*(1 + pow((Km_Nap / Nai), 1.3))*(1 + exp(-(x[cell_num].Vm - E_Na + 120.0) / 30.0)));

		RTOnF = R2*T / F;
		k32 = exp(Qn* x[cell_num].Vm / (2 * RTOnF));
		k43 = Nai / (K3ni + Nai);
		di = 1 + x[cell_num].Ca_sub / Kci*(1 + exp(-Qci*x[cell_num].Vm / RTOnF) + Nai / Kcni) + Nai / K1ni*(1 + Nai / K2ni*(1 + Nai / K3ni));
		k14 = Nai / K1ni*Nai / K2ni*(1 + Nai / K3ni) *exp(Qn* x[cell_num].Vm / (2 * RTOnF)) / di;
		k12 = x[cell_num].Ca_sub / Kci*exp(-Qci* x[cell_num].Vm / RTOnF) / di;
		k41 = exp(-Qn* x[cell_num].Vm / (2 * RTOnF));
		k34 = Nao / (K3no + Nao);
		x2 = k32*k43*(k14 + k12) + k41*k12*(k34 + k32);
		Do = 1 + Cao / Kco*(1 + exp(Qco* x[cell_num].Vm / RTOnF)) + Nao / K1no*(1 + Nao / K2no*(1 + Nao / K3no));
		k21 = Cao / Kco*exp(Qco* x[cell_num].Vm / RTOnF) / Do;
		k23 = Nao / K1no*Nao / K2no*(1 + Nao / K3no) *exp(-Qn* x[cell_num].Vm / (2 * RTOnF)) / Do;
		x1 = k41*k34*(k23 + k21) + k21*k32*(k43 + k41);
		x3 = k14*k43*(k23 + k21) + k12*k23*(k43 + k41);
		x4 = k23*k34*(k14 + k12) + k14*k21*(k34 + k32);
		i_NaCa = Cm*kNaCa*(x2*k21 - x1*k12) / (x1 + x2 + x3 + x4);

		i_b_Ca = Cm*g_b_Ca*(x[cell_num].Vm - E_CaL);
		i_b_Na = Cm*g_b_Na*(x[cell_num].Vm - E_Na);

		delta_fTC = kf_TC* x[cell_num].Cai * (1 - x[cell_num].fTC) - kb_TC* x[cell_num].fTC;
		delta_fTMC = kf_TMC* x[cell_num].Cai* (1 - (x[cell_num].fTMC + x[cell_num].fTMM)) - kb_TMC* x[cell_num].fTMC;
		delta_fTMM = kf_TMM*Mgi*(1 - (x[cell_num].fTMC + x[cell_num].fTMM)) - kb_TMM* x[cell_num].fTMM;
		delta_fCMi = kf_CM*x[cell_num].Cai* (1 - x[cell_num].fCMi) - kb_CM*x[cell_num].fCMi;
		delta_fCMs = kf_CM* x[cell_num].Ca_sub * (1 - x[cell_num].fCMs) - kb_CM* x[cell_num].fCMs;
		delta_fCQ = kf_CQ* x[cell_num].Ca_jsr * (1 - x[cell_num].fCQ) - kb_CQ* x[cell_num].fCQ;

		j_Ca_dif = (x[cell_num].Ca_sub - x[cell_num].Cai) / tau_dif_Ca;

		V_sub = 0.001 * 2 * M_PI *L_sub*(R_cell - L_sub / 2)*L_cell;
		j_up = P_up / (1 + K_up / x[cell_num].Cai);
		V_cell = 0.001*M_PI *pow(R_cell, 2)*L_cell;
		V_nsr = V_nsr_part*V_cell;
		V_in = V_in_part*V_cell - V_sub;

		j_SRCarel = ks* x[cell_num].O * (x[cell_num].Ca_jsr - x[cell_num].Ca_sub);
		V_jsr = V_jsr_part*V_cell;
		j_tr = (x[cell_num].Ca_nsr - x[cell_num].Ca_jsr) / tau_tr;
		dL_infinity = 1 / (1 + exp(-(x[cell_num].Vm + 13.5) / 6));

		if (x[cell_num].Vm == -35) {
			adVm = -35.00001;
		}
		else if (x[cell_num].Vm == 0) {
			adVm = 0.00001;
		}
		else {
			adVm = x[cell_num].Vm;
		}

		alpha_dL = -0.02839*(adVm + 35) / (exp(-(adVm + 35) / 2.5) - 1) - 0.0849*adVm / (exp(-adVm / 4.8) - 1);
		if (x[cell_num].Vm == 5) {
			bdVm = 5.00001;
		}
		else {
			bdVm = x[cell_num].Vm;
		}
		beta_dL = 0.01143*(bdVm - 5) / (exp((bdVm - 5) / 2.5) - 1);
		tau_dL = 1 / (alpha_dL + beta_dL);
		fCa_infinity = Km_fCa / (Km_fCa + x[cell_num].Ca_sub);
		tau_fCa = fCa_infinity / alpha_fCa;
		fL_infinity = 1 / (1 + exp((x[cell_num].Vm + 35) / 7.3));
		tau_fL = 44.3 + 257.1*exp(-pow(((x[cell_num].Vm + 32.5) / 13.9), 2));

		dT_infinity = 1.0 / (1 + exp(-(x[cell_num].Vm + 26.3) / 6));
		tau_dT = 1 / (1.068*exp((x[cell_num].Vm + 26.3) / 30) + 1.068*exp(-(x[cell_num].Vm + 26.3) / 30));
		fT_infinity = 1 / (1 + exp((x[cell_num].Vm + 61.7) / 5.6));
		tau_fT = 1 / (0.0153*exp(-(x[cell_num].Vm + 61.7) / 83.3) + 0.015*exp((x[cell_num].Vm + 61.7) / 15.38));

		pa_infinity = 1 / (1 + exp(-(x[cell_num].Vm + 23.2) / 10.6));
		tau_paS = 0.84655354 / (0.0042*exp(x[cell_num].Vm / 17) + 0.00015*exp(-x[cell_num].Vm / 21.6));
		tau_paF = 0.84655354 / (0.0372*exp(x[cell_num].Vm / 15.9) + 0.00096*exp(-x[cell_num].Vm / 22.5));
		pi_infinity = 1 / (1 + exp((x[cell_num].Vm + 28.6) / 17.1));
		tau_pi = 1 / (0.1*exp(-x[cell_num].Vm / 54.645) + 0.656*exp(x[cell_num].Vm / 106.157));

		alpha_n = 0.014 / (1 + exp(-(x[cell_num].Vm - 40) / 9));
		beta_n = 0.001*exp(-x[cell_num].Vm / 45);
		n_infinity = alpha_n / (alpha_n + beta_n);
		tau_n = 1 / (alpha_n + beta_n);

		y_infinity = 1 / (1 + exp((x[cell_num].Vm - VIf_half) / 13.5));
		tau_y = 0.7166529 / (exp(-(x[cell_num].Vm + 386.9) / 45.302) + exp((x[cell_num].Vm - 73.08) / 19.231));

		qa_infinity = 1 / (1 + exp(-(x[cell_num].Vm + 57) / 5));
		alpha_qa = 1 / (0.15*exp(-x[cell_num].Vm / 11) + 0.2*exp(-x[cell_num].Vm / 700));
		beta_qa = 1 / (16 * exp(x[cell_num].Vm / 8) + 15 * exp(x[cell_num].Vm / 50));
		tau_qa = 1 / (alpha_qa + beta_qa);
		alpha_qi = 1 / (3100 * exp(x[cell_num].Vm / 13) + 700 * exp(x[cell_num].Vm / 70));
		beta_qi = 1 / (95 * exp(-x[cell_num].Vm / 10) + 50 * exp(-x[cell_num].Vm / 700)) + 0.000229 / (1 + exp(-x[cell_num].Vm / 5));
		qi_infinity = alpha_qi / (alpha_qi + beta_qi);
		tau_qi = 6.65 / (alpha_qi + beta_qi);

		kCaSR = MaxSR - (MaxSR - MinSR) / (1 + pow((EC50_SR / x[cell_num].Ca_jsr), HSR));
		koSRCa = koCa / kCaSR;
		kiSRCa = kiCa*kCaSR;
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		Iion = i_CaL + i_CaT + i_f + i_st + i_Kr + i_Ks + i_to + i_sus + i_NaK + i_NaCa + i_b_Ca + i_b_Na;
		ion_current[cell_num] = Iion;

		////////////////////////////////////////////////////////////////////////////////////		COMPUTE STATES		//////////////////////////////////////////////////////////////////////////////////////////

		x_temp[cell_num].q = updateSingleState(x[cell_num].q, ((q_infinity - x[cell_num].q) / tau_q), step);
		x_temp[cell_num].r = updateSingleState(x[cell_num].r, ((r_infinity - x[cell_num].r) / tau_r), step);

		x_temp[cell_num].fCMi = updateSingleState(x[cell_num].fCMi, delta_fCMi, step);
		x_temp[cell_num].fCMs = updateSingleState(x[cell_num].fCMs, delta_fCMs, step);
		x_temp[cell_num].fCQ = updateSingleState(x[cell_num].fCQ, delta_fCQ, step);
		x_temp[cell_num].fTC = updateSingleState(x[cell_num].fTC, delta_fTC, step);
		x_temp[cell_num].fTMC = updateSingleState(x[cell_num].fTMC, delta_fTMC, step);
		x_temp[cell_num].fTMM = updateSingleState(x[cell_num].fTMM, delta_fTMM, step);

		x_temp[cell_num].Ca_jsr = updateSingleState(x[cell_num].Ca_jsr, (j_tr - (j_SRCarel + CQ_tot*delta_fCQ)), step);
		x_temp[cell_num].Ca_nsr = updateSingleState(x[cell_num].Ca_nsr, (j_up - j_tr*V_jsr / V_nsr), step);
		x_temp[cell_num].Ca_sub = updateSingleState(x[cell_num].Ca_sub, ((j_SRCarel*V_jsr / V_sub - ((i_CaL + i_CaT + i_b_Ca - 2 * i_NaCa) / (2 * F*V_sub) + j_Ca_dif + CM_tot*delta_fCMs))), step);
		x_temp[cell_num].Cai = updateSingleState(x[cell_num].Cai, ((j_Ca_dif*V_sub - j_up*V_nsr) / V_in - (CM_tot*delta_fCMi + TC_tot*delta_fTC + TMC_tot*delta_fTMC)), step);

		x_temp[cell_num].dL = updateSingleState(x[cell_num].dL, ((dL_infinity - x[cell_num].dL) / tau_dL), step);
		x_temp[cell_num].fCa = updateSingleState(x[cell_num].fCa, ((fCa_infinity - x[cell_num].fCa) / tau_fCa), step);
		x_temp[cell_num].fL = updateSingleState(x[cell_num].fL, ((fL_infinity - x[cell_num].fL) / tau_fL), step);
		x_temp[cell_num].dT = updateSingleState(x[cell_num].dT, ((dT_infinity - x[cell_num].dT) / tau_dT), step);
		x_temp[cell_num].fT = updateSingleState(x[cell_num].fT, ((fT_infinity - x[cell_num].fT) / tau_fT), step);
		x_temp[cell_num].paF = updateSingleState(x[cell_num].paF, ((pa_infinity - x[cell_num].paF) / tau_paF), step);
		x_temp[cell_num].paS = updateSingleState(x[cell_num].paS, ((pa_infinity - x[cell_num].paS) / tau_paS), step);
		x_temp[cell_num].pi_ = updateSingleState(x[cell_num].pi_, ((pi_infinity - x[cell_num].pi_) / tau_pi), step);
		x_temp[cell_num].n = updateSingleState(x[cell_num].n, ((n_infinity - x[cell_num].n) / tau_n), step);
		x_temp[cell_num].y = updateSingleState(x[cell_num].y, ((y_infinity - x[cell_num].y) / tau_y), step);
		x_temp[cell_num].qa = updateSingleState(x[cell_num].qa, ((qa_infinity - x[cell_num].qa) / tau_qa), step);
		x_temp[cell_num].qi = updateSingleState(x[cell_num].qi, ((qi_infinity - x[cell_num].qi) / tau_qi), step);

		x_temp[cell_num].I = updateSingleState(x[cell_num].I, (kiSRCa*x[cell_num].Ca_sub * x[cell_num].O - kim*x[cell_num].I - (kom*x[cell_num].I - koSRCa*pow(x[cell_num].Ca_sub, 2)*x[cell_num].RI)), step);
		x_temp[cell_num].O = updateSingleState(x[cell_num].O, (koSRCa*pow(x[cell_num].Ca_sub, 2)*x[cell_num].R1 - kom*x[cell_num].O - (kiSRCa*x[cell_num].Ca_sub * x[cell_num].O - kim*x[cell_num].I)), step);
		x_temp[cell_num].R1 = updateSingleState(x[cell_num].R1, (kim*x[cell_num].RI - kiSRCa*x[cell_num].Ca_sub * x[cell_num].R1 - (koSRCa*pow(x[cell_num].Ca_sub, 2)*x[cell_num].R1 - kom*x[cell_num].O)), step);
		x_temp[cell_num].RI = updateSingleState(x[cell_num].RI, (kom*x[cell_num].I - koSRCa*pow(x[cell_num].Ca_sub, 2)*x[cell_num].RI - (kim*x[cell_num].RI - kiSRCa*x[cell_num].Ca_sub * x[cell_num].R1)), step);
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//printf("x_tempRI=%e\n", x_temp[cell_num].RI);
//	}
	//printf("xtemp[0].RI=%e\n", x_temp[0].RI);
	//printf("xtemp[1].RI=%e\n", x_temp[1].RI);
	//printf("xtemp[999].RI=%e\n", x_temp[999].RI);
}

__global__ void updateState(cell* x, cell* x_temp) {

	//int idx = (blockIdx.x*blockDim.x) + cellsThread*threadIdx.x;
	//int idx = cellsThread*threadIdx.x;
	//int maxIdx = idx + cellsThread;
	//for (;idx < maxIdx;idx++) {
        int idx = (blockIdx.x*blockDim.x) + threadIdx.x;
		//do not include Vm here!
		x[idx].q = x_temp[idx].q;
		x[idx].r = x_temp[idx].r;
		x[idx].fCMi = x_temp[idx].fCMi;
		x[idx].fCMs = x_temp[idx].fCMs;
		x[idx].fCQ = x_temp[idx].fCQ;
		x[idx].fTC = x_temp[idx].fTC;
		x[idx].fTMC = x_temp[idx].fTMC;
		x[idx].fTMM = x_temp[idx].fTMM;
		x[idx].Ca_jsr = x_temp[idx].Ca_jsr;
		x[idx].Ca_nsr = x_temp[idx].Ca_nsr;
		x[idx].Ca_sub = x_temp[idx].Ca_sub;
		x[idx].Cai = x_temp[idx].Cai;
		x[idx].dL = x_temp[idx].dL;
		x[idx].fCa = x_temp[idx].fCa;
		x[idx].fL = x_temp[idx].fL;
		x[idx].dT = x_temp[idx].dT;
		x[idx].fT = x_temp[idx].fT;
		x[idx].paF = x_temp[idx].paF;
		x[idx].paS = x_temp[idx].paS;
		x[idx].pi_ = x_temp[idx].pi_;
		x[idx].n = x_temp[idx].n;
		x[idx].y = x_temp[idx].y;
		x[idx].qa = x_temp[idx].qa;
		x[idx].qi = x_temp[idx].qi;
		x[idx].I = x_temp[idx].I;
		x[idx].O = x_temp[idx].O;
		x[idx].R1 = x_temp[idx].R1;
		x[idx].RI = x_temp[idx].RI;
		//printf("x[idx].RI=%e\n", x[idx].RI);

	//}

	//printf("xup[0].RI=%e\n", x[0].RI);
	//printf("xup[1].RI=%e\n", x[1].RI);
	//printf("xup[999].RI=%e\n", x[999].RI);
}
__global__ void computeVoltage(cell* x, double* V, double* Iion, double step, int time){//, double stimDur, double stimAmp, int tstim) {

	//int idx = (blockIdx.x*blockDim.x) + cellsThread*threadIdx.x;
	//int idx = cellsThread*threadIdx.x;
	//int maxIdx = idx + cellsThread;
	//for (;idx < maxIdx;idx++) {
        int idx = (blockIdx.x*blockDim.x) + threadIdx.x;
//	double Cm = 5.7e-5;
		
		double	rGap = 10000 * 1e9 ;// 10000 * 1e6; //% MOhm
	
		//double stim = 0.0;
		//double Istim1 = 0.0;
		//	double Vnet_R, Vnet_L, Vnet_U, Vnet_D;
		//double rad = 0.0011;
		//double deltx = 0.01;
		//double rho;
		double Cm = 32;// 32;
		//double Rmyo;
		//double gj;
        double	gapJunct = 32*1e-12*rGap;//(57 * 1e-12)*rGap;
	//	gj = 1.27;
	//	Rmyo = 526;
	//	rho = 3.14159*pow(rad, 2)*(Rmyo + 1000 / gj) / deltx; // total resistivity

		//int stimInterval = 0;	//1000;

		//if (time%tstim > (stimDur / step)) { Istim1 = 0.0; }
		//else { Istim1 = stimAmp; }
		//if (time % (int)(stimInterval / step) <= stimDur / step) {	//&& time < endS1 / step
		//	Istim1 = stimAmp;	//stimAmp is 0
		//}
		//else { Istim1 = 0.0; }
		
		//////////////////////////////////////////		Tissue model		////////////////////////////////////////////////////////
		if (sim2D == 1) {
			double Vnet_U, Vnet_D, Vnet_L, Vnet_R;
		Vnet_U = 0;Vnet_D = 0;Vnet_L = 0;Vnet_R = 0;
		if (idx >= WIDTH) { Vnet_U = x[idx - WIDTH].Vm - x[idx].Vm; }
		else { Vnet_U = 0; }
		if (idx < (LENGTH - 1)*WIDTH) { Vnet_D = x[idx + WIDTH].Vm - x[idx].Vm; }
		else { Vnet_D = 0; }
		if (idx%WIDTH == 0) { Vnet_L = 0; }
		else { Vnet_L = x[idx - 1].Vm - x[idx].Vm; }
		if (idx%WIDTH < (WIDTH - 1)) { Vnet_R = x[idx + 1].Vm - x[idx].Vm; }
		else { Vnet_R = 0; }

		V[idx] = updateSingleState(x[idx].Vm, ((1 / gapJunct)*(Vnet_R + Vnet_L + Vnet_U + Vnet_D)) - (Iion[idx] / Cm), step);
		
	}
	else {
		V[idx] = updateSingleState(x[idx].Vm, -(Iion[idx] / Cm), step);
		
	}
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//	}
	//printf("Vcomp[0]=%e\n", V[0]);
	//printf("Vcomp[1]=%e\n", V[1]);
	//printf("Vcomp[999]=%e\n", V[999]);

}

__global__ void updateVoltage(cell* x, double* V) {

	//int idx = (blockIdx.x*blockDim.x) + cellsThread*threadIdx.x;
	//int idx = cellsThread*threadIdx.x;
	//int maxIdx = idx + cellsThread;
	//for (;idx < maxIdx;idx++) {
    int idx = (blockIdx.x*blockDim.x) + threadIdx.x;
		x[idx].Vm = V[idx];
	//}
	//if (idx == 0) {			printf("Vm[0]=%e\n", x[0].Vm);		}
	//printf("Vup[0]=%e\n", x[0].Vm);
	//printf("Vup[1]=%e\n", x[1].Vm);
	//printf("Vup[999]=%e\n", x[999].Vm);
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
	//float* hMscalesIna;
	//float* MscalesIna;
	//float* Mtransmurality;
	//float* hMtransmurality;
	int nChannels = 12;
	hgChannels = (float *)malloc(sizeof(float)*nCELLS * nChannels);
	huploadICs = (float *)malloc(sizeof(float)*nCELLS * nSTATES);
	//hMscalesIna = (float *)malloc(sizeof(float)*nSIMS*nCELLS * 13);
	//hMtransmurality = (float *)malloc(sizeof(float)*nSIMS*nCELLS * 2);

	FILE *f1 = fopen("C:/Users/sobie/Desktop/chiara/chiara_codde/SAN project_Maltsev/cudasparkmodel2011/Gs_M_s04_50x50.txt", "rb");
	//FILE *f1 = fopen("C:/Users/sobie/Desktop/chiara/simparall/cudasparkmodel2011/MscalesIna.txt", "rb");
	//FILE *f2 = fopen("C:/Users/sobie/Desktop/chiara/simparall/cudasparkmodel2011/Mtranssmurality.txt", "rb");

	int errnum;
	if ((f1 = fopen("C:/Users/sobie/Desktop/chiara/chiara_codde/SAN project_Maltsev/cudasparkmodel2011/Gs_M_s04_50x50.txt", "rb")) == NULL)
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



	FILE *f2 = fopen("C:/Users/sobie/Desktop/chiara/chiara_codde/SAN project_Maltsev/cudasparkmodel2011/ICs_s04M.txt", "rb");
	
	if ((f2 = fopen("C:/Users/sobie/Desktop/chiara/chiara_codde/SAN project_Maltsev/cudasparkmodel2011/ICs_s04M.txt", "rb")) == NULL)
		//if ((f1 = fopen("C:/Users/sobie/Desktop/chiara/simparall/cudasparkmodel2011/MscalesIna.txt", "rb")) == NULL)

	{
		errnum = errno;
		fprintf(stderr, "Value of errno: %d\n", errno);
		perror("Error printed by perror");
		fprintf(stderr, "Error opening file: %s\n", strerror(errnum));

	}
	for (int j = 0;j<nCELLS*nSTATES;j++) {
		fscanf(f2, "%f \n ", &huploadICs[j]);
		//printf("%f \n", hgChannels[j]);

	}
	fclose(f2);

	cudaMalloc(&uploadICs, sizeof(float)*nCELLS* nSTATES);
	cudaMemcpy(uploadICs, huploadICs, nCELLS* nSTATES * sizeof(float), cudaMemcpyHostToDevice);


	assignHeterogeneity << <nSIMS, cellsThread >> > (Gs, gChannels);
	initialConditions << <nSIMS, cellsThread >> >(dVars,uploadICs);

	cudaMemcpy(hVars, dVars, numBytes, cudaMemcpyDeviceToHost);

	//for (int j = 0;j<nCELLS *nSIMS;j++) {
	//printf("V: %f, C1: %f, Ki: %f\n", hVars[j].V, hVars[j].C1_Na, hVars[j].Ki);
	//}

	FILE *fV = fopen("CA07_M_V_04_1e4_50x50", "w");
	FILE *ft = fopen("CA07_M_T_04_1e4_50x50", "w");
	FILE *fSimulation = fopen("M_SimTime_20s", "w");
	//FILE *fIion = fopen("Iion_s02_SC", "w");
	//FILE *fInet = fopen("Inet_s01_SC", "w");
 	//FILE *fStates = fopen("ICsToAssign_M_s05", "w");

	int index = 0;

	//double* V_array;
	//double* t_array;


	double* dev_ion_currents;
	cell* dev_x_temp;
	double* host_Vtemp;
	double* dev_Vtemp;

	cudaEvent_t start, stop;
	float elapsedTime;
	double begin_time;
	double end_time;
	// Time Step Variables
	double step = 0.005;// 1;// 0.002;
	double tend = 20000;// 20000;
	int iterations = tend / step;
	double skip_time_value = 1.25;// 0.5;// 1.25;// 0.125;//0.5; //ms
	int skip_timept = skip_time_value / step; // skipping time points in voltage array & time array
	int total_timepts = iterations / skip_timept;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	// results of the computeState kernel
	//V_array = (double*)malloc(sizeof(double)*(total_timepts*nCELLS*nSIMS));
	//t_array = (double*)malloc(sizeof(double)*(total_timepts*nSIMS));

	//double stimDur = 2.0;
	//double stimAmp = 0;//-80
	//double stimInterval = 0;
	//int tstim = stimInterval / step;

	cudaMalloc((void**)&dev_x_temp, numBytes);//cudaMalloc(&dev_x_temp, sizeof(cell)*size);
	cudaMalloc((void**)&dev_ion_currents, numBytes);//cudaMalloc(&dev_ion_currents, sizeof(double)*nCELLS*nSIMS);
	cell* host_x_temp;
	host_x_temp = (cell*)malloc(numBytes);
	host_Vtemp = (double*)malloc(sizeof(double)*N);
	cudaMalloc((void**)&dev_Vtemp, sizeof(double)*N);

	while (time<iterations) {

		//printf("Running Maltsev model multicellular simulation...\n");
		//if (time%100000<5){	printf("%d/%d (%f percent) done .... \n",time,iterations,(100.0*time)/iterations);	}
                if (time == 0) {
					printf("\n");
					printf("\n");
					printf("\n");
					printf("\n");
					printf("                     ... Running Maltsev 2D simulation ...                       \n ");
					printf("\n");
					printf("\n");
					printf("\n");
					printf("\n");
				}
				if (time % 100000 < 5) {
					printf("				%d/%d (%f percent) done .... \n", time, iterations, (100.0*time) / iterations);
				}
		
        computeState << <nSIMS, cellsThread >> > (dVars, dev_ion_currents, step, dev_x_temp,Gs);//, scales
		updateState << <nSIMS, cellsThread >> >(dVars, dev_x_temp);//, num_cells, cells_per_thread
		computeVoltage << <nSIMS, cellsThread >> >(dVars, dev_Vtemp, dev_ion_currents, step, time);//, stimDur, stimAmp, tstim);
		updateVoltage << <nSIMS, cellsThread >> >(dVars, dev_Vtemp);

		//update Voltage and time arrays and write data to file
		cudaMemcpy(host_x_temp, dev_x_temp, numBytes, cudaMemcpyDeviceToHost);
		cudaMemcpy(host_Vtemp, dev_Vtemp, N * sizeof(double), cudaMemcpyDeviceToHost);
		

		if (time%skip_timept == 0) {// && time>=(300/step)) {
			for (int j = 0; j < nCELLS; j++) {	//*nSIMS
				//int idx = (blockIdx.x*blockDim.x) + cellsThread*threadIdx.x;
			//	printf("V: %f,y: %f,Nai: %f, Casub: %f,fCa: %f, Cai: %f, m: %f, h: %f, dT: %f\n", host_Vtemp[j], host_x_temp[j].y, host_x_temp[j].Nai, host_x_temp[j].Casub,  host_x_temp[j].fCa, host_x_temp[j].Cai, host_x_temp[j].m, host_x_temp[j].h, host_x_temp[j].dT);
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
			//for (int i = 0;i<nSIMS;i++) {
			//	t_array[(index*nSIMS) + i] = time*step;
			//}
		//	index++;
		//}
		}
		time++;
		/*if (time == iterations-1) {
			for (int j = 0; j < nCELLS; j++) {
			printf("states: %f\n ", host_x_temp[j].q);
			fprintf(fStates, "%.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f\n ",
			host_x_temp[j].q ,
			host_x_temp[j].r ,
			host_x_temp[j].Vm ,
			host_x_temp[j].fCMi,
			host_x_temp[j].fCMs,
			host_x_temp[j].fCQ ,
			host_x_temp[j].fTC,
			host_x_temp[j].fTMC,
			host_x_temp[j].fTMM,
			host_x_temp[j].Ca_jsr ,
			host_x_temp[j].Ca_nsr,
			host_x_temp[j].Ca_sub,
			host_x_temp[j].Cai ,
			host_x_temp[j].dL ,
			host_x_temp[j].fCa ,
			host_x_temp[j].fL ,
			host_x_temp[j].dT ,
			host_x_temp[j].fT ,
			host_x_temp[j].paF,
			host_x_temp[j].paS,
			host_x_temp[j].pi_ ,
			host_x_temp[j].n,
			host_x_temp[j].y,
			host_x_temp[j].qa ,
			host_x_temp[j].qi ,
			host_x_temp[j].I,
			host_x_temp[j].O,
			host_x_temp[j].R1,
			host_x_temp[j].RI );
			}

		//fprintf(fStates,"%.15f\t",host_x_temp[j])
		}*/
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

