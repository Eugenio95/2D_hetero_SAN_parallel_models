function dY = GPU_MaltsevModel2009(time, Y, S)  %#codegen

% Add kernelfun pragma to trigger kernel creation
coder.gpu.kernelfun;

%-------------------------------------------------------------------------------
% Initial conditions
%-------------------------------------------------------------------------------

% Y = [1.0, 0.0, -65.0, 0.042, 0.089, 0.032, 0.02, 0.22, 0.69, 0.029, 1.35, 0.000223, 0.0001, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.1e-6, 3.4e-6, 0.7499955, 0.25];

% YNames = {'q', 'r', 'Vm', 'fCMi', 'fCMs', 'fCQ', 'fTC', 'fTMC', 'fTMM', 'Ca_jsr', 'Ca_nsr', 'Ca_sub', 'Cai', 'dL', 'fCa', 'fL', 'dT', 'fT', 'paF', 'paS', 'pi_', 'n', 'y', 'qa', 'qi', 'I', 'O', 'R1', 'RI'};
% YUnits = {'dimensionless', 'dimensionless', 'millivolt', 'dimensionless', 'dimensionless', 'dimensionless', 'dimensionless', 'dimensionless', 'dimensionless', 'millimolar', 'millimolar', 'millimolar', 'millimolar', 'dimensionless', 'dimensionless', 'dimensionless', 'dimensionless', 'dimensionless', 'dimensionless', 'dimensionless', 'dimensionless', 'dimensionless', 'dimensionless', 'dimensionless', 'dimensionless', 'dimensionless', 'dimensionless', 'dimensionless', 'dimensionless'};
% YComponents = {'AP_sensitive_currents_q_gate', 'AP_sensitive_currents_r_gate', 'Vm', 'calcium_buffering', 'calcium_buffering', 'calcium_buffering', 'calcium_buffering', 'calcium_buffering', 'calcium_buffering', 'calcium_dynamics', 'calcium_dynamics', 'calcium_dynamics', 'calcium_dynamics', 'i_CaL_dL_gate', 'i_CaL_fCa_gate', 'i_CaL_fL_gate', 'i_CaT_dT_gate', 'i_CaT_fT_gate', 'i_Kr_pa_gate', 'i_Kr_pa_gate', 'i_Kr_pi_gate', 'i_Ks_n_gate', 'i_f_y_gate', 'i_st_qa_gate', 'i_st_qi_gate', 'j_SRCarel', 'j_SRCarel', 'j_SRCarel', 'j_SRCarel'};

%-------------------------------------------------------------------------------
% State variables
%-------------------------------------------------------------------------------

% 1: q (dimensionless) (in AP_sensitive_currents_q_gate)
% 2: r (dimensionless) (in AP_sensitive_currents_r_gate)
% 3: Vm (millivolt) (in Vm)
% 4: fCMi (dimensionless) (in calcium_buffering)
% 5: fCMs (dimensionless) (in calcium_buffering)
% 6: fCQ (dimensionless) (in calcium_buffering)
% 7: fTC (dimensionless) (in calcium_buffering)
% 8: fTMC (dimensionless) (in calcium_buffering)
% 9: fTMM (dimensionless) (in calcium_buffering)
% 10: Ca_jsr (millimolar) (in calcium_dynamics)
% 11: Ca_nsr (millimolar) (in calcium_dynamics)
% 12: Ca_sub (millimolar) (in calcium_dynamics)
% 13: Cai (millimolar) (in calcium_dynamics)
% 14: dL (dimensionless) (in i_CaL_dL_gate)
% 15: fCa (dimensionless) (in i_CaL_fCa_gate)
% 16: fL (dimensionless) (in i_CaL_fL_gate)
% 17: dT (dimensionless) (in i_CaT_dT_gate)
% 18: fT (dimensionless) (in i_CaT_fT_gate)
% 19: paF (dimensionless) (in i_Kr_pa_gate)
% 20: paS (dimensionless) (in i_Kr_pa_gate)
% 21: pi_ (dimensionless) (in i_Kr_pi_gate)
% 22: n (dimensionless) (in i_Ks_n_gate)
% 23: y (dimensionless) (in i_f_y_gate)
% 24: qa (dimensionless) (in i_st_qa_gate)
% 25: qi (dimensionless) (in i_st_qi_gate)
% 26: I (dimensionless) (in j_SRCarel)
% 27: O (dimensionless) (in j_SRCarel)
% 28: R1 (dimensionless) (R in j_SRCarel)
% 29: RI (dimensionless) (in j_SRCarel)

%-------------------------------------------------------------------------------
% Computed variables
%-------------------------------------------------------------------------------

% q_infinity (dimensionless) (in AP_sensitive_currents_q_gate)
% tau_q (millisecond) (in AP_sensitive_currents_q_gate)
% r_infinity (dimensionless) (in AP_sensitive_currents_r_gate)
% tau_r (millisecond) (in AP_sensitive_currents_r_gate)
% i_sus (picoA) (in AP_sensitive_currents)
% i_to (picoA) (in AP_sensitive_currents)
% delta_fCMi (per_millisecond) (in calcium_buffering)
% delta_fCMs (per_millisecond) (in calcium_buffering)
% delta_fCQ (per_millisecond) (in calcium_buffering)
% delta_fTC (per_millisecond) (in calcium_buffering)
% delta_fTMC (per_millisecond) (in calcium_buffering)
% delta_fTMM (per_millisecond) (in calcium_buffering)
% E_K (millivolt) (in electric_potentials)
% E_Ks (millivolt) (in electric_potentials)
% E_Na (millivolt) (in electric_potentials)
% adVm (millivolt) (in i_CaL_dL_gate)
% alpha_dL (per_millisecond) (in i_CaL_dL_gate)
% bdVm (millivolt) (in i_CaL_dL_gate)
% beta_dL (per_millisecond) (in i_CaL_dL_gate)
% dL_infinity (dimensionless) (in i_CaL_dL_gate)
% tau_dL (millisecond) (in i_CaL_dL_gate)
% fCa_infinity (dimensionless) (in i_CaL_fCa_gate)
% tau_fCa (millisecond) (in i_CaL_fCa_gate)
% fL_infinity (dimensionless) (in i_CaL_fL_gate)
% tau_fL (millisecond) (in i_CaL_fL_gate)
% i_CaL (picoA) (in i_CaL)
% dT_infinity (dimensionless) (in i_CaT_dT_gate)
% tau_dT (millisecond) (in i_CaT_dT_gate)
% fT_infinity (dimensionless) (in i_CaT_fT_gate)
% tau_fT (millisecond) (in i_CaT_fT_gate)
% i_CaT (picoA) (in i_CaT)
% pa_infinity (dimensionless) (in i_Kr_pa_gate)
% tau_paF (millisecond) (in i_Kr_pa_gate)
% tau_paS (millisecond) (in i_Kr_pa_gate)
% pi_infinity (dimensionless) (in i_Kr_pi_gate)
% tau_pi (millisecond) (in i_Kr_pi_gate)
% i_Kr (picoA) (in i_Kr)
% alpha_n (per_millisecond) (in i_Ks_n_gate)
% beta_n (per_millisecond) (in i_Ks_n_gate)
% n_infinity (dimensionless) (in i_Ks_n_gate)
% tau_n (millisecond) (in i_Ks_n_gate)
% i_Ks (picoA) (in i_Ks)
% RTOnF (millivolt) (in i_NaCa)
% di (dimensionless) (in i_NaCa)
% do (dimensionless) (in i_NaCa)
% i_NaCa (picoA) (in i_NaCa)
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
% i_NaK (picoA) (in i_NaK)
% i_b_Ca (picoA) (in i_b_Ca)
% i_b_Na (picoA) (in i_b_Na)
% tau_y (millisecond) (in i_f_y_gate)
% y_infinity (dimensionless) (in i_f_y_gate)
% i_f (picoA) (in i_f)
% i_f_K (picoA) (in i_f)
% i_f_Na (picoA) (in i_f)
% alpha_qa (per_millisecond) (in i_st_qa_gate)
% beta_qa (per_millisecond) (in i_st_qa_gate)
% qa_infinity (dimensionless) (in i_st_qa_gate)
% tau_qa (millisecond) (in i_st_qa_gate)
% alpha_qi (per_millisecond) (in i_st_qi_gate)
% beta_qi (per_millisecond) (in i_st_qi_gate)
% qi_infinity (dimensionless) (in i_st_qi_gate)
% tau_qi (millisecond) (in i_st_qi_gate)
% i_st (picoA) (in i_st)
% j_Ca_dif (millimolar_per_millisecond) (in intracellular_calcium_fluxes)
% j_tr (millimolar_per_millisecond) (in intracellular_calcium_fluxes)
% j_up (millimolar_per_millisecond) (in intracellular_calcium_fluxes)
% j_SRCarel (millimolar_per_millisecond) (in j_SRCarel)
% kCaSR (dimensionless) (in j_SRCarel)
% kiSRCa (per_millimolar_millisecond) (in j_SRCarel)
% koSRCa (per_millimolar2_millisecond) (in j_SRCarel)
% V_cell (picolitre) (in model_parameters)
% V_i (picolitre) (in model_parameters)
% V_jsr (picolitre) (in model_parameters)
% V_nsr (picolitre) (in model_parameters)
% V_sub (picolitre) (in model_parameters)

%-------------------------------------------------------------------------------
% Computation
%-------------------------------------------------------------------------------

dY = zeros(29,1);

% time (millisecond)

E_K = S.R2*S.T/S.F*log(S.Ko/S.Ki);
i_to = S.Cm*S.g_to*(Y(3)-E_K)*Y(1)*Y(2);
i_sus = S.Cm*S.g_sus*(Y(3)-E_K)*Y(2);
q_infinity = 1.0/(1.0+exp((Y(3)+49.0)/13.0));
tau_q = 6.06+39.102/(0.57*exp(-0.08*(Y(3)+44.0))+0.065*exp(0.1*(Y(3)+45.93)));
dY(1, 1) = (q_infinity-Y(1))/tau_q;
r_infinity = 1.0/(1.0+exp(-(Y(3)-19.3)/15.0));
tau_r = 2.75352+14.40516/(1.037*exp(0.09*(Y(3)+30.61))+0.369*exp(-0.12*(Y(3)+23.84)));
dY(2, 1) = (r_infinity-Y(2))/tau_r;
i_CaL = S.Cm*S.g_CaL*(Y(3)-S.E_CaL)*Y(14)*Y(16)*Y(15);
i_CaT = S.Cm*S.g_CaT*(Y(3)-S.E_CaT)*Y(17)*Y(18);
E_Na = S.R2*S.T/S.F*log(S.Nao/S.Nai);
i_f_Na = S.Cm*0.3833*S.g_if*(Y(3)-E_Na)*Y(23)^2.0;
i_f_K = S.Cm*0.6167*S.g_if*(Y(3)-E_K)*Y(23)^2.0;
i_f = i_f_Na+i_f_K;
i_st = S.Cm*S.g_st*(Y(3)-S.E_st)*Y(24)*Y(25);
i_Kr = S.Cm*S.g_Kr*(Y(3)-E_K)*(0.6*Y(19)+0.4*Y(20))*Y(21);
E_Ks = S.R2*S.T/S.F*log((S.Ko+0.12*S.Nao)/(S.Ki+0.12*S.Nai));
i_Ks = S.Cm*S.g_Ks*(Y(3)-E_Ks)*Y(22)^2.0;
i_NaK = S.Cm*S.i_NaK_max/((1.0+(S.Km_Kp/S.Ko)^1.2)*(1.0+(S.Km_Nap/S.Nai)^1.3)*(1.0+exp(-(Y(3)-E_Na+120.0)/30.0)));
RTOnF = S.R2*S.T/S.F;
k32 = exp(S.Qn*Y(3)/(2.0*RTOnF));
k43 = S.Nai/(S.K3ni+S.Nai);
di = 1.0+Y(12)/S.Kci*(1.0+exp(-S.Qci*Y(3)/RTOnF)+S.Nai/S.Kcni)+S.Nai/S.K1ni*(1.0+S.Nai/S.K2ni*(1.0+S.Nai/S.K3ni));
k14 = S.Nai/S.K1ni*S.Nai/S.K2ni*(1.0+S.Nai/S.K3ni)*exp(S.Qn*Y(3)/(2.0*RTOnF))/di;
k12 = Y(12)/S.Kci*exp(-S.Qci*Y(3)/RTOnF)/di;
k41 = exp(-S.Qn*Y(3)/(2.0*RTOnF));
k34 = S.Nao/(S.K3no+S.Nao);
x2 = k32*k43*(k14+k12)+k41*k12*(k34+k32);
do = 1.0+S.Cao/S.Kco*(1.0+exp(S.Qco*Y(3)/RTOnF))+S.Nao/S.K1no*(1.0+S.Nao/S.K2no*(1.0+S.Nao/S.K3no));
k21 = S.Cao/S.Kco*exp(S.Qco*Y(3)/RTOnF)/do;
k23 = S.Nao/S.K1no*S.Nao/S.K2no*(1.0+S.Nao/S.K3no)*exp(-S.Qn*Y(3)/(2.0*RTOnF))/do;
x1 = k41*k34*(k23+k21)+k21*k32*(k43+k41);
x3 = k14*k43*(k23+k21)+k12*k23*(k43+k41);
x4 = k23*k34*(k14+k12)+k14*k21*(k34+k32);
i_NaCa = S.Cm*S.kNaCa*(x2*k21-x1*k12)/(x1+x2+x3+x4);
i_b_Ca = S.Cm*S.g_b_Ca*(Y(3)-S.E_CaL);
i_b_Na = S.Cm*S.g_b_Na*(Y(3)-E_Na);
dY(3, 1) = -(i_CaL+i_CaT+i_f+i_st+i_Kr+i_Ks+i_to+i_sus+i_NaK+i_NaCa+i_b_Ca+i_b_Na)/S.Cm;
delta_fTC = S.kf_TC*Y(13)*(1.0-Y(7))-S.kb_TC*Y(7);
dY(7, 1) = delta_fTC;
delta_fTMC = S.kf_TMC*Y(13)*(1.0-(Y(8)+Y(9)))-S.kb_TMC*Y(8);
dY(8, 1) = delta_fTMC;
delta_fTMM = S.kf_TMM*S.Mgi*(1.0-(Y(8)+Y(9)))-S.kb_TMM*Y(9);
dY(9, 1) = delta_fTMM;
delta_fCMi = S.kf_CM*Y(13)*(1.0-Y(4))-S.kb_CM*Y(4);
dY(4, 1) = delta_fCMi;
delta_fCMs = S.kf_CM*Y(12)*(1.0-Y(5))-S.kb_CM*Y(5);
dY(5, 1) = delta_fCMs;
delta_fCQ = S.kf_CQ*Y(10)*(1.0-Y(6))-S.kb_CQ*Y(6);
dY(6, 1) = delta_fCQ;
j_Ca_dif = (Y(12)-Y(13))/S.tau_dif_Ca;
V_sub = 0.001*2.0*pi*S.L_sub*(S.R_cell-S.L_sub/2.0)*S.L_cell;
j_up = S.P_up/(1.0+S.K_up/Y(13));
V_cell = 0.001*pi*S.R_cell^2.0*S.L_cell;
V_nsr = S.V_nsr_part*V_cell;
V_i = S.V_i_part*V_cell-V_sub;
dY(13, 1) = (j_Ca_dif*V_sub-j_up*V_nsr)/V_i-(S.CM_tot*delta_fCMi+S.TC_tot*delta_fTC+S.TMC_tot*delta_fTMC);
j_SRCarel = S.ks*Y(27)*(Y(10)-Y(12));
V_jsr = S.V_jsr_part*V_cell;
dY(12, 1) = j_SRCarel*V_jsr/V_sub-((i_CaL+i_CaT+i_b_Ca-2.0*i_NaCa)/(2.0*S.F*V_sub)+j_Ca_dif+S.CM_tot*delta_fCMs);
j_tr = (Y(11)-Y(10))/S.tau_tr;
dY(11, 1) = j_up-j_tr*V_jsr/V_nsr;
dY(10, 1) = j_tr-(j_SRCarel+S.CQ_tot*delta_fCQ);
dL_infinity = 1.0/(1.0+exp(-(Y(3)+13.5)/6.0));

if (Y(3) == -35.0)
   adVm = -35.00001;
elseif (Y(3) == 0.0)
   adVm = 0.00001;
else
   adVm = Y(3);
end;

alpha_dL = -0.02839*(adVm+35.0)/(exp(-(adVm+35.0)/2.5)-1.0)-0.0849*adVm/(exp(-adVm/4.8)-1.0);

if (Y(3) == 5.0)
   bdVm = 5.00001;
else
   bdVm = Y(3);
end;

beta_dL = 0.01143*(bdVm-5.0)/(exp((bdVm-5.0)/2.5)-1.0);
tau_dL = 1.0/(alpha_dL+beta_dL);
dY(14, 1) = (dL_infinity-Y(14))/tau_dL;
fCa_infinity = S.Km_fCa/(S.Km_fCa+Y(12));
tau_fCa = fCa_infinity/S.alpha_fCa;
dY(15, 1) = (fCa_infinity-Y(15))/tau_fCa;
fL_infinity = 1.0/(1.0+exp((Y(3)+35.0)/7.3));
tau_fL = 44.3+257.1*exp(-((Y(3)+32.5)/13.9)^2.0);
dY(16, 1) = (fL_infinity-Y(16))/tau_fL;
dT_infinity = 1.0/(1.0+exp(-(Y(3)+26.3)/6.0));
tau_dT = 1.0/(1.068*exp((Y(3)+26.3)/30.0)+1.068*exp(-(Y(3)+26.3)/30.0));
dY(17, 1) = (dT_infinity-Y(17))/tau_dT;
fT_infinity = 1.0/(1.0+exp((Y(3)+61.7)/5.6));
tau_fT = 1.0/(0.0153*exp(-(Y(3)+61.7)/83.3)+0.015*exp((Y(3)+61.7)/15.38));
dY(18, 1) = (fT_infinity-Y(18))/tau_fT;
pa_infinity = 1.0/(1.0+exp(-(Y(3)+23.2)/10.6));
tau_paS = 0.84655354/(0.0042*exp(Y(3)/17.0)+0.00015*exp(-Y(3)/21.6));
tau_paF = 0.84655354/(0.0372*exp(Y(3)/15.9)+0.00096*exp(-Y(3)/22.5));
dY(20, 1) = (pa_infinity-Y(20))/tau_paS;
dY(19, 1) = (pa_infinity-Y(19))/tau_paF;
pi_infinity = 1.0/(1.0+exp((Y(3)+28.6)/17.1));
tau_pi = 1.0/(0.1*exp(-Y(3)/54.645)+0.656*exp(Y(3)/106.157));
dY(21, 1) = (pi_infinity-Y(21))/tau_pi;
alpha_n = 0.014/(1.0+exp(-(Y(3)-40.0)/9.0));
beta_n = 0.001*exp(-Y(3)/45.0);
n_infinity = alpha_n/(alpha_n+beta_n);
tau_n = 1.0/(alpha_n+beta_n);
dY(22, 1) = (n_infinity-Y(22))/tau_n;
y_infinity = 1.0/(1.0+exp((Y(3)-S.VIf_half)/13.5));
tau_y = 0.7166529/(exp(-(Y(3)+386.9)/45.302)+exp((Y(3)-73.08)/19.231));
dY(23, 1) = (y_infinity-Y(23))/tau_y;
qa_infinity = 1.0/(1.0+exp(-(Y(3)+57.0)/5.0));
alpha_qa = 1.0/(0.15*exp(-Y(3)/11.0)+0.2*exp(-Y(3)/700.0));
beta_qa = 1.0/(16.0*exp(Y(3)/8.0)+15.0*exp(Y(3)/50.0));
tau_qa = 1.0/(alpha_qa+beta_qa);
dY(24, 1) = (qa_infinity-Y(24))/tau_qa;
alpha_qi = 1.0/(3100.0*exp(Y(3)/13.0)+700.0*exp(Y(3)/70.0));
beta_qi = 1.0/(95.0*exp(-Y(3)/10.0)+50.0*exp(-Y(3)/700.0))+0.000229/(1.0+exp(-Y(3)/5.0));
qi_infinity = alpha_qi/(alpha_qi+beta_qi);
tau_qi = 6.65/(alpha_qi+beta_qi);
dY(25, 1) = (qi_infinity-Y(25))/tau_qi;
kCaSR = S.MaxSR-(S.MaxSR-S.MinSR)/(1.0+(S.EC50_SR/Y(10))^S.HSR);
koSRCa = S.koCa/kCaSR;
kiSRCa = S.kiCa*kCaSR;
dY(28, 1) = S.kim*Y(29)-kiSRCa*Y(12)*Y(28)-(koSRCa*Y(12)^2.0*Y(28)-S.kom*Y(27));
dY(27, 1) = koSRCa*Y(12)^2.0*Y(28)-S.kom*Y(27)-(kiSRCa*Y(12)*Y(27)-S.kim*Y(26));
dY(26, 1) = kiSRCa*Y(12)*Y(27)-S.kim*Y(26)-(S.kom*Y(26)-koSRCa*Y(12)^2.0*Y(29));
dY(29, 1) = S.kom*Y(26)-koSRCa*Y(12)^2.0*Y(29)-(S.kim*Y(29)-kiSRCa*Y(12)*Y(28));

%===============================================================================
% End of file
%===============================================================================
