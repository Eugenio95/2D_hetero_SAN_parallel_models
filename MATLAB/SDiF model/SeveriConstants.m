
%-------------------------------------------------------------------------------
% Constants Severi Model 2012
%-------------------------------------------------------------------------------

constStr.EC50_SR = 0.45;   % millimolar (in Ca_SR_release)
constStr.HSR = 2.5;   % dimensionless (in Ca_SR_release)
constStr.MaxSR = 15.0;   % dimensionless (in Ca_SR_release)
constStr.MinSR = 1.0;   % dimensionless (in Ca_SR_release)
constStr.kiCa = 500.0;   % per_millimolar_second (in Ca_SR_release)
constStr.kim = 5.0;   % per_second (in Ca_SR_release)
constStr.koCa = 10000.0;   % per_millimolar2_second (in Ca_SR_release)
constStr.kom = 60.0;   % per_second (in Ca_SR_release)
constStr.ks = 250000000.0;   % per_second (in Ca_SR_release)
constStr.CM_tot = 0.045;   % millimolar (in Ca_buffering)
constStr.CQ_tot = 10.0;   % millimolar (in Ca_buffering)
constStr.Mgi = 2.5;   % millimolar (in Ca_buffering)
constStr.TC_tot = 0.031;   % millimolar (in Ca_buffering)
constStr.TMC_tot = 0.062;   % millimolar (in Ca_buffering)
constStr.kb_CM = 542.0;   % per_second (in Ca_buffering)
constStr.kb_CQ = 445.0;   % per_second (in Ca_buffering)
constStr.kb_TC = 446.0;   % per_second (in Ca_buffering)
constStr.kb_TMC = 7.51;   % per_second (in Ca_buffering)
constStr.kb_TMM = 751.0;   % per_second (in Ca_buffering)
constStr.kf_CM = 227700.0;   % per_millimolar_second (in Ca_buffering)
constStr.kf_CQ = 534.0;   % per_millimolar_second (in Ca_buffering)
constStr.kf_TC = 88800.0;   % per_millimolar_second (in Ca_buffering)
constStr.kf_TMC = 227700.0;   % per_millimolar_second (in Ca_buffering)
constStr.kf_TMM = 2277.0;   % per_millimolar_second (in Ca_buffering)
constStr.T1 = 6.928;   % second (T in Ca_dynamics)
constStr.kbBAPTA = 119.38;   % per_second (in Ca_dynamics)
constStr.kfBAPTA = 940000.0;   % per_millimolar_second (in Ca_dynamics)
constStr.K_up = 0.0006;   % millimolar (in Ca_intracellular_fluxes)
constStr.P_up_basal = 12.0;   % millimolar_per_second (in Ca_intracellular_fluxes)
constStr.tau_dif_Ca = 4.0e-5;   % second (in Ca_intracellular_fluxes)
constStr.tau_tr = 0.04;   % second (in Ca_intracellular_fluxes)
constStr.L_cell = 70.0;   % micrometre (in Cell_parameters)
constStr.L_sub = 0.02;   % micrometre (in Cell_parameters)
constStr.R_cell = 4.0;   % micrometre (in Cell_parameters)
constStr.V_i_part = 0.46;   % dimensionless (in Cell_parameters)
constStr.V_jsr_part = 0.0012;   % dimensionless (in Cell_parameters)
constStr.V_nsr_part = 0.0116;   % dimensionless (in Cell_parameters)
constStr.Cao = 1.8;   % millimolar (in Ionic_values)
constStr.Ki = 140.0;   % millimolar (in Ionic_values)
constStr.Ko = 5.4;   % millimolar (in Ionic_values)
constStr.Nao = 140.0;   % millimolar (in Ionic_values)
constStr.C = 3.2e-5;   % microF (in Membrane)
constStr.F = 96485.3415;   % coulomb_per_mole (in Membrane)
constStr.R2 = 8314.472;   % joule_per_kilomole_kelvin (R in Membrane)
constStr.T2 = 310.0;   % kelvin (T in Membrane)
constStr.clamp_mode = 0.0;   % dimensionless (in Membrane)
constStr.ACh = 0.0;   % millimolar (in Rate_modulation_experiments)
constStr.BAPTA_10_mM = 0.0;   % dimensionless (in Rate_modulation_experiments)
constStr.Cs_5_mM = 0.0;   % dimensionless (in Rate_modulation_experiments)
constStr.Iso_1_uM = 0.0;   % dimensionless (in Rate_modulation_experiments)
constStr.Iva_3_uM = 0.0;   % dimensionless (in Rate_modulation_experiments)
constStr.V_holding = -45.0;   % millivolt (in Voltage_clamp)
constStr.V_test = -35.0;   % millivolt (in Voltage_clamp)
constStr.t_holding = 0.5;   % second (in Voltage_clamp)
constStr.t_test = 0.5;   % second (in Voltage_clamp)
constStr.Km_fCa = 0.00035;   % millimolar (in i_CaL_fCa_gate)
constStr.alpha_fCa = 0.01;   % per_second (in i_CaL_fCa_gate)


constStr.shift = 0.0;   % millivolt (in i_Ks_n_gate)
constStr.K1ni = 395.3;   % millimolar (in i_NaCa)
constStr.K1no = 1628.0;   % millimolar (in i_NaCa)
constStr.K2ni = 2.289;   % millimolar (in i_NaCa)
constStr.K2no = 561.4;   % millimolar (in i_NaCa)
constStr.K3ni = 26.44;   % millimolar (in i_NaCa)
constStr.K3no = 4.663;   % millimolar (in i_NaCa)

constStr.Kci = 0.0207;   % millimolar (in i_NaCa)
constStr.Kcni = 26.44;   % millimolar (in i_NaCa)
constStr.Kco = 3.663;   % millimolar (in i_NaCa)
constStr.Qci = 0.1369;   % dimensionless (in i_NaCa)
constStr.Qco = 0.0;   % dimensionless (in i_NaCa)
constStr.Qn = 0.4315;   % dimensionless (in i_NaCa)
constStr.Km_Kp = 1.4;   % millimolar (in i_NaK)
constStr.Km_Nap = 14.0;   % millimolar (in i_NaK)

constStr.delta_m = 1.0e-5;   % millivolt (in i_Na_m_gate)

constStr.Km_f = 45.0;   % millimolar (in i_f)


constStr.P_CaL = 0.2;   % nanoA_per_millimolar (in i_CaL)
constStr.P_CaT = 0.02;   % nanoA_per_millimolar (in i_CaT)
constStr.g_KACh = 0.00864;   % microS (in i_KACh)
constStr.g_Kr = 0.0021637;   % microS (in i_Kr)
constStr.K_NaCa = 4.0;   % nanoA (in i_NaCa)
constStr.i_NaK_max = 0.063;   % nanoA (in i_NaK)
constStr.g_Na = 0.0125;   % microS (in i_Na)
constStr.g_to = 0.002;   % microS (in i_to)
constStr.g_Ks = 0.0016576; % microS (in i_Ks)
constStr.g_f = 0.03; % microS

