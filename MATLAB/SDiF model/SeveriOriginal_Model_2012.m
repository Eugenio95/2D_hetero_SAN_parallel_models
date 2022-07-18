%===============================================================================
% CellML file:   C:\Users\eucci\OneDrive\Desktop\Dottorato\SAN Project\Modelli\severi_fantini_charawi_difrancesco_2012.cellml
% CellML model:  severi_fantini_charawi_difrancesco_2012
% Date and time: 10/11/2020 at 08:55:07
%-------------------------------------------------------------------------------
% Conversion from CellML 1.0 to MATLAB (init) was done using COR (0.9.31.1409)
%    Copyright 2002-2020 Dr Alan Garny
%    http://cor.physiol.ox.ac.uk/ - cor@physiol.ox.ac.uk
%-------------------------------------------------------------------------------
% http://www.cellml.org/
%===============================================================================

function dY = SeveriModel2012(time, Y)

%-------------------------------------------------------------------------------
% Initial conditions
%-------------------------------------------------------------------------------

% Y = [7.86181717518e-8, 1.7340201253e-7, 0.912317231017262, 0.211148145512825, 0.0373817991524254, 0.054381370046, 0.299624275428735, 0.0180519400676086, 0.281244308217086, 0.501049376634, 0.316762674605, 1.05386465080816, 1.0e-5, 1.0e-5, 0.0, 0.0, -52.0, 7.5, 0.0, 0.697998543259722, 0.497133507285601, 0.0, 0.0, 0.0, 0.0990510403258968, 0.322999177802891, 0.705410877258545, 0.0, 1.3676940140066e-5, 0.440131579215766, 0.181334538702451, 0.506139850982478, 0.0144605370597924];

% YNames = {'I', 'O', 'R1', 'RI', 'fCMi', 'fCMs', 'fCQ', 'fTC', 'fTMC', 'fTMM', 'Ca_jsr', 'Ca_nsr', 'Ca_sub', 'Cai', 'fBAPTA', 'fBAPTA_sub', 'V_ode', 'Nai_', 'dL', 'fCa', 'fL', 'dT', 'fT', 'a', 'paF', 'paS', 'piy', 'n', 'h', 'm', 'y', 'q', 'r'};
% YUnits = {'dimensionless', 'dimensionless', 'dimensionless', 'dimensionless', 'dimensionless', 'dimensionless', 'dimensionless', 'dimensionless', 'dimensionless', 'dimensionless', 'millimolar', 'millimolar', 'millimolar', 'millimolar', 'millimolar', 'millimolar', 'millivolt', 'millimolar', 'dimensionless', 'dimensionless', 'dimensionless', 'dimensionless', 'dimensionless', 'dimensionless', 'dimensionless', 'dimensionless', 'dimensionless', 'dimensionless', 'dimensionless', 'dimensionless', 'dimensionless', 'dimensionless', 'dimensionless'};
% YComponents = {'Ca_SR_release', 'Ca_SR_release', 'Ca_SR_release', 'Ca_SR_release', 'Ca_buffering', 'Ca_buffering', 'Ca_buffering', 'Ca_buffering', 'Ca_buffering', 'Ca_buffering', 'Ca_dynamics', 'Ca_dynamics', 'Ca_dynamics', 'Ca_dynamics', 'Ca_dynamics', 'Ca_dynamics', 'Membrane', 'Nai_concentration', 'i_CaL_dL_gate', 'i_CaL_fCa_gate', 'i_CaL_fL_gate', 'i_CaT_dT_gate', 'i_CaT_fT_gate', 'i_KACh_a_gate', 'i_Kr_pa_gate', 'i_Kr_pa_gate', 'i_Kr_pi_gate', 'i_Ks_n_gate', 'i_Na_h_gate', 'i_Na_m_gate', 'i_f_y_gate', 'i_to_q_gate', 'i_to_r_gate'};

%-------------------------------------------------------------------------------
% State variables
%-------------------------------------------------------------------------------

% 1: I (dimensionless) (in Ca_SR_release)
% 2: O (dimensionless) (in Ca_SR_release)
% 3: R1 (dimensionless) (R in Ca_SR_release)
% 4: RI (dimensionless) (in Ca_SR_release)
% 5: fCMi (dimensionless) (in Ca_buffering)
% 6: fCMs (dimensionless) (in Ca_buffering)
% 7: fCQ (dimensionless) (in Ca_buffering)
% 8: fTC (dimensionless) (in Ca_buffering)
% 9: fTMC (dimensionless) (in Ca_buffering)
% 10: fTMM (dimensionless) (in Ca_buffering)
% 11: Ca_jsr (millimolar) (in Ca_dynamics)
% 12: Ca_nsr (millimolar) (in Ca_dynamics)
% 13: Ca_sub (millimolar) (in Ca_dynamics)
% 14: Cai (millimolar) (in Ca_dynamics)
% 15: fBAPTA (millimolar) (in Ca_dynamics)
% 16: fBAPTA_sub (millimolar) (in Ca_dynamics)
% 17: V_ode (millivolt) (in Membrane)
% 18: Nai_ (millimolar) (in Nai_concentration)
% 19: dL (dimensionless) (in i_CaL_dL_gate)
% 20: fCa (dimensionless) (in i_CaL_fCa_gate)
% 21: fL (dimensionless) (in i_CaL_fL_gate)
% 22: dT (dimensionless) (in i_CaT_dT_gate)
% 23: fT (dimensionless) (in i_CaT_fT_gate)
% 24: a (dimensionless) (in i_KACh_a_gate)
% 25: paF (dimensionless) (in i_Kr_pa_gate)
% 26: paS (dimensionless) (in i_Kr_pa_gate)
% 27: piy (dimensionless) (in i_Kr_pi_gate)
% 28: n (dimensionless) (in i_Ks_n_gate)
% 29: h (dimensionless) (in i_Na_h_gate)
% 30: m (dimensionless) (in i_Na_m_gate)
% 31: y (dimensionless) (in i_f_y_gate)
% 32: q (dimensionless) (in i_to_q_gate)
% 33: r (dimensionless) (in i_to_r_gate)

%-------------------------------------------------------------------------------
% Constants
%-------------------------------------------------------------------------------

EC50_SR = 0.45;   % millimolar (in Ca_SR_release)
HSR = 2.5;   % dimensionless (in Ca_SR_release)
MaxSR = 15.0;   % dimensionless (in Ca_SR_release)
MinSR = 1.0;   % dimensionless (in Ca_SR_release)
kiCa = 500.0;   % per_millimolar_second (in Ca_SR_release)
kim = 5.0;   % per_second (in Ca_SR_release)
koCa = 10000.0;   % per_millimolar2_second (in Ca_SR_release)
kom = 60.0;   % per_second (in Ca_SR_release)
ks = 250000000.0;   % per_second (in Ca_SR_release)
CM_tot = 0.045;   % millimolar (in Ca_buffering)
CQ_tot = 10.0;   % millimolar (in Ca_buffering)
Mgi = 2.5;   % millimolar (in Ca_buffering)
TC_tot = 0.031;   % millimolar (in Ca_buffering)
TMC_tot = 0.062;   % millimolar (in Ca_buffering)
kb_CM = 542.0;   % per_second (in Ca_buffering)
kb_CQ = 445.0;   % per_second (in Ca_buffering)
kb_TC = 446.0;   % per_second (in Ca_buffering)
kb_TMC = 7.51;   % per_second (in Ca_buffering)
kb_TMM = 751.0;   % per_second (in Ca_buffering)
kf_CM = 227700.0;   % per_millimolar_second (in Ca_buffering)
kf_CQ = 534.0;   % per_millimolar_second (in Ca_buffering)
kf_TC = 88800.0;   % per_millimolar_second (in Ca_buffering)
kf_TMC = 227700.0;   % per_millimolar_second (in Ca_buffering)
kf_TMM = 2277.0;   % per_millimolar_second (in Ca_buffering)
T1 = 6.928;   % second (T in Ca_dynamics)
kbBAPTA = 119.38;   % per_second (in Ca_dynamics)
kfBAPTA = 940000.0;   % per_millimolar_second (in Ca_dynamics)
K_up = 0.0006;   % millimolar (in Ca_intracellular_fluxes)
P_up_basal = 12.0;   % millimolar_per_second (in Ca_intracellular_fluxes)
tau_dif_Ca = 4.0e-5;   % second (in Ca_intracellular_fluxes)
tau_tr = 0.04;   % second (in Ca_intracellular_fluxes)
L_cell = 70.0;   % micrometre (in Cell_parameters)
L_sub = 0.02;   % micrometre (in Cell_parameters)
R_cell = 4.0;   % micrometre (in Cell_parameters)
V_i_part = 0.46;   % dimensionless (in Cell_parameters)
V_jsr_part = 0.0012;   % dimensionless (in Cell_parameters)
V_nsr_part = 0.0116;   % dimensionless (in Cell_parameters)
Cao = 1.8;   % millimolar (in Ionic_values)
Ki = 140.0;   % millimolar (in Ionic_values)
Ko = 5.4;   % millimolar (in Ionic_values)
Nao = 140.0;   % millimolar (in Ionic_values)
C = 3.2e-5;   % microF (in Membrane)
F = 96485.3415;   % coulomb_per_mole (in Membrane)
R2 = 8314.472;   % joule_per_kilomole_kelvin (R in Membrane)
T2 = 310.0;   % kelvin (T in Membrane)
clamp_mode = 0.0;   % dimensionless (in Membrane)
ACh = 0.0;   % millimolar (in Rate_modulation_experiments)
BAPTA_10_mM = 0.0;   % dimensionless (in Rate_modulation_experiments)
Cs_5_mM = 0.0;   % dimensionless (in Rate_modulation_experiments)
Iso_1_uM = 0.0;   % dimensionless (in Rate_modulation_experiments)
Iva_3_uM = 0.0;   % dimensionless (in Rate_modulation_experiments)
V_holding = -45.0;   % millivolt (in Voltage_clamp)
V_test = -35.0;   % millivolt (in Voltage_clamp)
t_holding = 0.5;   % second (in Voltage_clamp)
t_test = 0.5;   % second (in Voltage_clamp)
Km_fCa = 0.00035;   % millimolar (in i_CaL_fCa_gate)
alpha_fCa = 0.01;   % per_second (in i_CaL_fCa_gate)
P_CaL = 0.2;   % nanoA_per_millimolar (in i_CaL)
P_CaT = 0.02;   % nanoA_per_millimolar (in i_CaT)
g_KACh = 0.00864;   % microS (in i_KACh)
g_Kr = 0.0021637;   % microS (in i_Kr)
shift = 0.0;   % millivolt (in i_Ks_n_gate)
K1ni = 395.3;   % millimolar (in i_NaCa)
K1no = 1628.0;   % millimolar (in i_NaCa)
K2ni = 2.289;   % millimolar (in i_NaCa)
K2no = 561.4;   % millimolar (in i_NaCa)
K3ni = 26.44;   % millimolar (in i_NaCa)
K3no = 4.663;   % millimolar (in i_NaCa)
K_NaCa = 4.0;   % nanoA (in i_NaCa)
Kci = 0.0207;   % millimolar (in i_NaCa)
Kcni = 26.44;   % millimolar (in i_NaCa)
Kco = 3.663;   % millimolar (in i_NaCa)
Qci = 0.1369;   % dimensionless (in i_NaCa)
Qco = 0.0;   % dimensionless (in i_NaCa)
Qn = 0.4315;   % dimensionless (in i_NaCa)
Km_Kp = 1.4;   % millimolar (in i_NaK)
Km_Nap = 14.0;   % millimolar (in i_NaK)
i_NaK_max = 0.063;   % nanoA (in i_NaK)
delta_m = 1.0e-5;   % millivolt (in i_Na_m_gate)
g_Na = 0.0125;   % microS (in i_Na)
Km_f = 45.0;   % millimolar (in i_f)
g_to = 0.002;   % microS (in i_to)

%-------------------------------------------------------------------------------
% Computed variables
%-------------------------------------------------------------------------------

% j_SRCarel (millimolar_per_second) (in Ca_SR_release)
% kCaSR (dimensionless) (in Ca_SR_release)
% kiSRCa (per_millimolar_second) (in Ca_SR_release)
% koSRCa (per_millimolar2_second) (in Ca_SR_release)
% delta_fCMi (per_second) (in Ca_buffering)
% delta_fCMs (per_second) (in Ca_buffering)
% delta_fCQ (per_second) (in Ca_buffering)
% delta_fTC (per_second) (in Ca_buffering)
% delta_fTMC (per_second) (in Ca_buffering)
% delta_fTMM (per_second) (in Ca_buffering)
% BAPTA (millimolar) (in Ca_dynamics)
% P_up (millimolar_per_second) (in Ca_intracellular_fluxes)
% b_up (dimensionless) (in Ca_intracellular_fluxes)
% j_Ca_dif (millimolar_per_second) (in Ca_intracellular_fluxes)
% j_tr (millimolar_per_second) (in Ca_intracellular_fluxes)
% j_up (millimolar_per_second) (in Ca_intracellular_fluxes)
% V_cell (millimetre3) (in Cell_parameters)
% V_i (millimetre3) (in Cell_parameters)
% V_jsr (millimetre3) (in Cell_parameters)
% V_nsr (millimetre3) (in Cell_parameters)
% V_sub (millimetre3) (in Cell_parameters)
% E_Ca (millivolt) (in Ionic_values)
% E_K (millivolt) (in Ionic_values)
% E_Na (millivolt) (in Ionic_values)
% RTONF (millivolt) (in Membrane)
% V (millivolt) (in Membrane)
% i_tot (nanoA) (in Membrane)
% Nai (millimolar) (in Nai_concentration)
% V_clamp (millivolt) (in Voltage_clamp)
% Iso_shift_1 (millivolt) (Iso_shift in i_CaL_dL_gate)
% Iso_slope (dimensionless) (in i_CaL_dL_gate)
% adVm (millivolt) (in i_CaL_dL_gate)
% alpha_dL (per_second) (in i_CaL_dL_gate)
% bdVm (millivolt) (in i_CaL_dL_gate)
% beta_dL (per_second) (in i_CaL_dL_gate)
% dL_infinity (dimensionless) (in i_CaL_dL_gate)
% tau_dL (second) (in i_CaL_dL_gate)
% fCa_infinity (dimensionless) (in i_CaL_fCa_gate)
% tau_fCa (second) (in i_CaL_fCa_gate)
% fL_infinity (dimensionless) (in i_CaL_fL_gate)
% tau_fL (second) (in i_CaL_fL_gate)
% ACh_block (dimensionless) (in i_CaL)
% Iso_increase_1 (dimensionless) (Iso_increase in i_CaL)
% i_CaL (nanoA) (in i_CaL)
% i_siCa (nanoA) (in i_CaL)
% i_siK (nanoA) (in i_CaL)
% i_siNa (nanoA) (in i_CaL)
% dT_infinity (dimensionless) (in i_CaT_dT_gate)
% tau_dT (second) (in i_CaT_dT_gate)
% fT_infinity (dimensionless) (in i_CaT_fT_gate)
% tau_fT (second) (in i_CaT_fT_gate)
% i_CaT (nanoA) (in i_CaT)
% a_infinity (dimensionless) (in i_KACh_a_gate)
% alpha_a (per_second) (in i_KACh_a_gate)
% beta_a (per_second) (in i_KACh_a_gate)
% tau_a (second) (in i_KACh_a_gate)
% i_KACh (nanoA) (in i_KACh)
% alfapaF (per_second) (in i_Kr_pa_gate)
% betapaF (per_second) (in i_Kr_pa_gate)
% pa_infinity (dimensionless) (in i_Kr_pa_gate)
% tau_paF (second) (in i_Kr_pa_gate)
% tau_paS (second) (in i_Kr_pa_gate)
% pi_infinity (dimensionless) (in i_Kr_pi_gate)
% tau_pi (second) (in i_Kr_pi_gate)
% i_Kr (nanoA) (in i_Kr)
% Iso_shift_2 (millivolt) (Iso_shift in i_Ks_n_gate)
% alpha_n (per_second) (in i_Ks_n_gate)
% beta_n (per_second) (in i_Ks_n_gate)
% n_infinity (dimensionless) (in i_Ks_n_gate)
% tau_n (second) (in i_Ks_n_gate)
% E_Ks (millivolt) (in i_Ks)
% g_Ks (microS) (in i_Ks)
% i_Ks (nanoA) (in i_Ks)
% di (dimensionless) (in i_NaCa)
% do (dimensionless) (in i_NaCa)
% i_NaCa (nanoA) (in i_NaCa)
% k12 (dimensionless) (in i_NaCa)
% k14 (dimensionless) (in i_NaCa)
% k21 (dimensionless) (in i_NaCa)
% k23 (dimensionless) (in i_NaCa)
% k32 (dimensionless) (in i_NaCa)
% k34 (dimensionless) (in i_NaCa)
% k41 (dimensionless) (in i_NaCa)
% k43 (dimensionless) (in i_NaCa)
% x1 (dimensionless) (in i_NaCa)
% x2 (dimensionless) (in i_NaCa)
% x3 (dimensionless) (in i_NaCa)
% x4 (dimensionless) (in i_NaCa)
% Iso_increase_2 (dimensionless) (Iso_increase in i_NaK)
% i_NaK (nanoA) (in i_NaK)
% alpha_h (per_second) (in i_Na_h_gate)
% beta_h (per_second) (in i_Na_h_gate)
% E0_m (millivolt) (in i_Na_m_gate)
% alpha_m (per_second) (in i_Na_m_gate)
% beta_m (per_second) (in i_Na_m_gate)
% E_mh (millivolt) (in i_Na)
% i_Na (nanoA) (in i_Na)
% ACh_shift (millivolt) (in i_f_y_gate)
% Iso_shift_3 (millivolt) (Iso_shift in i_f_y_gate)
% tau_y (second) (in i_f_y_gate)
% y_infinity (dimensionless) (in i_f_y_gate)
% ICs_on_Icontrol (dimensionless) (in i_f)
% g_f_K (microS) (in i_f)
% g_f_Na (microS) (in i_f)
% i_f (nanoA) (in i_f)
% i_fK (nanoA) (in i_f)
% i_fNa (nanoA) (in i_f)
% q_infinity (dimensionless) (in i_to_q_gate)
% tau_q (second) (in i_to_q_gate)
% r_infinity (dimensionless) (in i_to_r_gate)
% tau_r (second) (in i_to_r_gate)
% i_to (nanoA) (in i_to)

%-------------------------------------------------------------------------------
% Computation
%-------------------------------------------------------------------------------

% time (second)

j_SRCarel = ks*Y(2)*(Y(11)-Y(13));
kCaSR = MaxSR-(MaxSR-MinSR)/(1.0+(EC50_SR/Y(11))^HSR);
koSRCa = koCa/kCaSR;
kiSRCa = kiCa*kCaSR;
dY(3, 1) = kim*Y(4)-kiSRCa*Y(13)*Y(3)-(koSRCa*Y(13)^2.0*Y(3)-kom*Y(2));
dY(2, 1) = koSRCa*Y(13)^2.0*Y(3)-kom*Y(2)-(kiSRCa*Y(13)*Y(2)-kim*Y(1));
dY(1, 1) = kiSRCa*Y(13)*Y(2)-kim*Y(1)-(kom*Y(1)-koSRCa*Y(13)^2.0*Y(4));
dY(4, 1) = kom*Y(1)-koSRCa*Y(13)^2.0*Y(4)-(kim*Y(4)-kiSRCa*Y(13)*Y(3));
delta_fTC = kf_TC*Y(14)*(1.0-Y(8))-kb_TC*Y(8);
dY(8, 1) = delta_fTC;
delta_fTMC = kf_TMC*Y(14)*(1.0-(Y(9)+Y(10)))-kb_TMC*Y(9);
dY(9, 1) = delta_fTMC;
delta_fTMM = kf_TMM*Mgi*(1.0-(Y(9)+Y(10)))-kb_TMM*Y(10);
dY(10, 1) = delta_fTMM;
delta_fCMi = kf_CM*Y(14)*(1.0-Y(5))-kb_CM*Y(5);
dY(5, 1) = delta_fCMi;
delta_fCMs = kf_CM*Y(13)*(1.0-Y(6))-kb_CM*Y(6);
dY(6, 1) = delta_fCMs;
delta_fCQ = kf_CQ*Y(11)*(1.0-Y(7))-kb_CQ*Y(7);
dY(7, 1) = delta_fCQ;

if ((BAPTA_10_mM > 0.0) && (time > T1))
   BAPTA = 10.0;
else
   BAPTA = 0.0;
end;

j_Ca_dif = (Y(13)-Y(14))/tau_dif_Ca;
V_sub = 0.000000001*2.0*pi*L_sub*(R_cell-L_sub/2.0)*L_cell;

if (Iso_1_uM > 0.0)
   b_up = -0.25;
elseif (ACh > 0.0)
   b_up = 0.7*ACh/(0.00009+ACh);
else
   b_up = 0.0;
end;

P_up = P_up_basal*(1.0-b_up);
j_up = P_up/(1.0+K_up/Y(14));
V_cell = 0.000000001*pi*R_cell^2.0*L_cell;
V_nsr = V_nsr_part*V_cell;
V_i = V_i_part*V_cell-V_sub;
dY(14, 1) = 1.0*(j_Ca_dif*V_sub-j_up*V_nsr)/V_i-(CM_tot*delta_fCMi+TC_tot*delta_fTC+TMC_tot*delta_fTMC)-(kfBAPTA*Y(14)*(BAPTA-Y(15))-kbBAPTA*Y(15));
dY(15, 1) = kfBAPTA*Y(14)*(BAPTA-Y(15))-kbBAPTA*Y(15);
V_jsr = V_jsr_part*V_cell;

if ((time > t_holding) && (time < t_holding+t_test))
   V_clamp = V_test;
else
   V_clamp = V_holding;
end;

if (clamp_mode >= 1.0)
   V = V_clamp;
else
   V = Y(17);
end;

RTONF = R2*T2/F;
i_siCa = 2.0*P_CaL*(V-0.0)/(RTONF*(1.0-exp(-1.0*(V-0.0)*2.0/RTONF)))*(Y(13)-Cao*exp(-2.0*(V-0.0)/RTONF))*Y(19)*Y(21)*Y(20);
i_CaT = 2.0*P_CaT*V/(RTONF*(1.0-exp(-1.0*V*2.0/RTONF)))*(Y(13)-Cao*exp(-2.0*V/RTONF))*Y(22)*Y(23);
k32 = exp(Qn*V/(2.0*RTONF));

if (BAPTA_10_mM > 0.0)
   Nai = 7.5;
else
   Nai = Y(18);
end;

k43 = Nai/(K3ni+Nai);
di = 1.0+Y(13)/Kci*(1.0+exp(-Qci*V/RTONF)+Nai/Kcni)+Nai/K1ni*(1.0+Nai/K2ni*(1.0+Nai/K3ni));
k14 = Nai/K1ni*Nai/K2ni*(1.0+Nai/K3ni)*exp(Qn*V/(2.0*RTONF))/di;
k12 = Y(13)/Kci*exp(-Qci*V/RTONF)/di;
k41 = exp(-Qn*V/(2.0*RTONF));
k34 = Nao/(K3no+Nao);
x2 = k32*k43*(k14+k12)+k41*k12*(k34+k32);
do = 1.0+Cao/Kco*(1.0+exp(Qco*V/RTONF))+Nao/K1no*(1.0+Nao/K2no*(1.0+Nao/K3no));
k21 = Cao/Kco*exp(Qco*V/RTONF)/do;
k23 = Nao/K1no*Nao/K2no*(1.0+Nao/K3no)*exp(-Qn*V/(2.0*RTONF))/do;
x1 = k41*k34*(k23+k21)+k21*k32*(k43+k41);
x3 = k14*k43*(k23+k21)+k12*k23*(k43+k41);
x4 = k23*k34*(k14+k12)+k14*k21*(k34+k32);
i_NaCa = K_NaCa*(x2*k21-x1*k12)/(x1+x2+x3+x4);
dY(13, 1) = j_SRCarel*V_jsr/V_sub-((i_siCa+i_CaT-2.0*i_NaCa)/(2.0*F*V_sub)+j_Ca_dif+CM_tot*delta_fCMs)-(kfBAPTA*Y(13)*(BAPTA-Y(16))-kbBAPTA*Y(16));
dY(16, 1) = kfBAPTA*Y(13)*(BAPTA-Y(16))-kbBAPTA*Y(16);
j_tr = (Y(12)-Y(11))/tau_tr;
dY(12, 1) = j_up-j_tr*V_jsr/V_nsr;
dY(11, 1) = j_tr-(j_SRCarel+CQ_tot*delta_fCQ);
E_Na = RTONF*log(Nao/Nai);
E_K = RTONF*log(Ko/Ki);
E_Ca = 0.5*RTONF*log(Cao/Y(13));

if (Iva_3_uM >= 1.0)
   g_f_Na = 0.03*(1.0-0.66);
else
   g_f_Na = 0.03;
end;

if (Cs_5_mM >= 1.0)
   ICs_on_Icontrol = 10.6015/5.0/(10.6015/5.0+exp(-0.71*V/25.0));
else
   ICs_on_Icontrol = 1.0;
end;

i_fNa = Y(31)^2.0*Ko/(Ko+Km_f)*g_f_Na*(V-E_Na)*ICs_on_Icontrol;

if (Iva_3_uM >= 1.0)
   g_f_K = 0.03*(1.0-0.66);
else
   g_f_K = 0.03;
end;

i_fK = Y(31)^2.0*Ko/(Ko+Km_f)*g_f_K*(V-E_K)*ICs_on_Icontrol;
i_f = i_fNa+i_fK;
i_Kr = g_Kr*(V-E_K)*(0.9*Y(25)+0.1*Y(26))*Y(27);

if (Iso_1_uM > 0.0)
   g_Ks = 1.2*0.0016576;
else
   g_Ks = 0.0016576;
end;

E_Ks = RTONF*log((Ko+0.0*Nao)/(Ki+0.0*Nai));
i_Ks = g_Ks*(V-E_Ks)*Y(28)^2.0;
i_to = g_to*(V-E_K)*Y(32)*Y(33);

if (Iso_1_uM > 0.0)
   Iso_increase_2 = 1.2;
else
   Iso_increase_2 = 1.0;
end;

i_NaK = Iso_increase_2*i_NaK_max*(1.0+(Km_Kp/Ko)^1.2)^-1.0*(1.0+(Km_Nap/Nai)^1.3)^-1.0*(1.0+exp(-(V-E_Na+110.0)/20.0))^-1.0;
E_mh = RTONF*log((Nao+0.12*Ko)/(Nai+0.12*Ki));
i_Na = g_Na*Y(30)^3.0*Y(29)*(V-E_mh);
i_siK = 0.000365*P_CaL*(V-0.0)/(RTONF*(1.0-exp(-1.0*(V-0.0)/RTONF)))*(Ki-Ko*exp(-1.0*(V-0.0)/RTONF))*Y(19)*Y(21)*Y(20);
i_siNa = 0.0000185*P_CaL*(V-0.0)/(RTONF*(1.0-exp(-1.0*(V-0.0)/RTONF)))*(Nai-Nao*exp(-1.0*(V-0.0)/RTONF))*Y(19)*Y(21)*Y(20);
ACh_block = 0.31*ACh/(ACh+0.00009);

if (Iso_1_uM > 0.0)
   Iso_increase_1 = 1.23;
else
   Iso_increase_1 = 1.0;
end;

i_CaL = (i_siCa+i_siK+i_siNa)*(1.0-ACh_block)*1.0*Iso_increase_1;

if (ACh > 0.0)
   i_KACh = g_KACh*(V-E_K)*(1.0+exp((V+20.0)/20.0))*Y(24);
else
   i_KACh = 0.0;
end;

i_tot = i_f+i_Kr+i_Ks+i_to+i_NaK+i_NaCa+i_Na+i_CaL+i_CaT+i_KACh;
dY(17, 1) = -i_tot/C;
dY(18, 1) = -1.0*(i_Na+i_fNa+i_siNa+3.0*i_NaK+3.0*i_NaCa)/(1.0*(V_i+V_sub)*F);

if (Iso_1_uM > 0.0)
   Iso_shift_1 = -8.0;
else
   Iso_shift_1 = 0.0;
end;

if (Iso_1_uM > 0.0)
   Iso_slope = 0.69;
else
   Iso_slope = 1.0;
end;

dL_infinity = 1.0/(1.0+exp(-(V+20.3-Iso_shift_1)/(Iso_slope*4.2)));

if (V == -41.8)
   adVm = -41.80001;
elseif (V == 0.0)
   adVm = 0.0;
elseif (V == -6.8)
   adVm = -6.80001;
else
   adVm = V;
end;

alpha_dL = -0.02839*(adVm+41.8-Iso_shift_1)/(exp(-(adVm+41.8-Iso_shift_1)/2.5)-1.0)-0.0849*(adVm+6.8-Iso_shift_1)/(exp(-(adVm+6.8-Iso_shift_1)/4.8)-1.0);

if (V == -1.8)
   bdVm = -1.80001;
else
   bdVm = V;
end;

beta_dL = 0.01143*(bdVm+1.8-Iso_shift_1)/(exp((bdVm+1.8-Iso_shift_1)/2.5)-1.0);
tau_dL = 0.001/(alpha_dL+beta_dL);
dY(19, 1) = (dL_infinity-Y(19))/tau_dL;
fCa_infinity = Km_fCa/(Km_fCa+Y(13));
tau_fCa = 0.001*fCa_infinity/alpha_fCa;
dY(20, 1) = (fCa_infinity-Y(20))/tau_fCa;
fL_infinity = 1.0/(1.0+exp((V+37.4)/5.3));
tau_fL = 0.001*(44.3+230.0*exp(-((V+36.0)/10.0)^2.0));
dY(21, 1) = (fL_infinity-Y(21))/tau_fL;
dT_infinity = 1.0/(1.0+exp(-(V+38.3)/5.5));
tau_dT = 0.001/(1.068*exp((V+38.3)/30.0)+1.068*exp(-(V+38.3)/30.0));
dY(22, 1) = (dT_infinity-Y(22))/tau_dT;
fT_infinity = 1.0/(1.0+exp((V+58.7)/3.8));
tau_fT = 1.0/(16.67*exp(-(V+75.0)/83.3)+16.67*exp((V+75.0)/15.38));
dY(23, 1) = (fT_infinity-Y(23))/tau_fT;
alpha_a = (3.5988-0.025641)/(1.0+0.0000012155/(1.0*ACh)^1.6951)+0.025641;
beta_a = 10.0*exp(0.0133*(V+40.0));
a_infinity = alpha_a/(alpha_a+beta_a);
tau_a = 1.0/(alpha_a+beta_a);
dY(24, 1) = (a_infinity-Y(24))/tau_a;
alfapaF = 1.0/(1.0+exp(-(V+23.2)/6.6))/(0.84655354/(37.2*exp(V/11.9)+0.96*exp(-V/18.5)));
betapaF = 4.0*((37.2*exp(V/15.9)+0.96*exp(-V/22.5))/0.84655354-1.0/(1.0+exp(-(V+23.2)/10.6))/(0.84655354/(37.2*exp(V/15.9)+0.96*exp(-V/22.5))));
pa_infinity = 1.0/(1.0+exp(-(V+14.8)/8.5));
tau_paS = 0.84655354/(4.2*exp(V/17.0)+0.15*exp(-V/21.6));
tau_paF = 1.0/(30.0*exp(V/10.0)+exp(-V/12.0));
dY(26, 1) = (pa_infinity-Y(26))/tau_paS;
dY(25, 1) = (pa_infinity-Y(25))/tau_paF;
tau_pi = 1.0/(100.0*exp(-V/54.645)+656.0*exp(V/106.157));
pi_infinity = 1.0/(1.0+exp((V+28.6)/17.1));
dY(27, 1) = (pi_infinity-Y(27))/tau_pi;

if (Iso_1_uM > 0.0)
   Iso_shift_2 = -14.0;
else
   Iso_shift_2 = 0.0;
end;

n_infinity = 14.0/(1.0+exp(-(V-40.0-Iso_shift_2)/12.0))/(14.0/(1.0+exp(-(V-40.0-Iso_shift_2)/12.0))+1.0*exp(-(V-Iso_shift_2)/45.0));
alpha_n = 28.0/(1.0+exp(-(V-40.0-Iso_shift_2)/3.0));
beta_n = 1.0*exp(-(V-Iso_shift_2-shift-5.0)/25.0);
tau_n = 1.0/(alpha_n+beta_n);
dY(28, 1) = (n_infinity-Y(28))/tau_n;
alpha_h = 20.0*exp(-0.125*(V+75.0));
beta_h = 2000.0/(320.0*exp(-0.1*(V+75.0))+1.0);
dY(29, 1) = alpha_h*(1.0-Y(29))-beta_h*Y(29);
E0_m = V+41.0;

if (abs(E0_m) < delta_m)
   alpha_m = 2000.0;
else
   alpha_m = 200.0*E0_m/(1.0-exp(-0.1*E0_m));
end;

beta_m = 8000.0*exp(-0.056*(V+66.0));
dY(30, 1) = alpha_m*(1.0-Y(30))-beta_m*Y(30);

if (ACh > 0.0)
   ACh_shift = -1.0-9.898*(1.0*ACh)^0.618/((1.0*ACh)^0.618+0.00122423);
else
   ACh_shift = 0.0;
end;

if (Iso_1_uM > 0.0)
   Iso_shift_3 = 7.5;
else
   Iso_shift_3 = 0.0;
end;

tau_y = 0.7166529/(0.0708*exp(-(V+5.0-ACh_shift-Iso_shift_3)/20.2791)+10.6*exp((V-ACh_shift-Iso_shift_3)/18.0));
y_infinity = 1.0/(1.0+exp((V+52.5-ACh_shift-Iso_shift_3)/9.0));
dY(31, 1) = (y_infinity-Y(31))/tau_y;
q_infinity = 1.0/(1.0+exp((V+49.0)/13.0));
tau_q = 0.001*0.6*(65.17/(0.57*exp(-0.08*(V+44.0))+0.065*exp(0.1*(V+45.93)))+10.1);
dY(32, 1) = (q_infinity-Y(32))/tau_q;
r_infinity = 1.0/(1.0+exp(-(V-19.3)/15.0));
tau_r = 0.001*0.66*1.4*(15.59/(1.037*exp(0.09*(V+30.61))+0.369*exp(-0.12*(V+23.84)))+2.98);
dY(33, 1) = (r_infinity-Y(33))/tau_r;

%===============================================================================
% End of file
%===============================================================================
