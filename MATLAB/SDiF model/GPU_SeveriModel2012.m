function dY = GPU_SeveriModel2012(time, Y, S) %#codegen

% Add kernelfun pragma to trigger kernel creation
coder.gpu.kernelfun;

%-------------------------------------------------------------------------------
% Computation
%-------------------------------------------------------------------------------

dY = zeros(33,1);

% time (second)

j_SRCarel = S.ks*Y(2)*(Y(11)-Y(13));
kCaSR = S.MaxSR-(S.MaxSR-S.MinSR)/(1.0+(S.EC50_SR/Y(11))^S.HSR);
koSRCa = S.koCa/kCaSR;
kiSRCa = S.kiCa*kCaSR;
dY(3, 1) = S.kim*Y(4)-kiSRCa*Y(13)*Y(3)-(koSRCa*Y(13)^2.0*Y(3)-S.kom*Y(2));
dY(2, 1) = koSRCa*Y(13)^2.0*Y(3)-S.kom*Y(2)-(kiSRCa*Y(13)*Y(2)-S.kim*Y(1));
dY(1, 1) = kiSRCa*Y(13)*Y(2)-S.kim*Y(1)-(S.kom*Y(1)-koSRCa*Y(13)^2.0*Y(4));
dY(4, 1) = S.kom*Y(1)-koSRCa*Y(13)^2.0*Y(4)-(S.kim*Y(4)-kiSRCa*Y(13)*Y(3));
delta_fTC = S.kf_TC*Y(14)*(1.0-Y(8))-S.kb_TC*Y(8);
dY(8, 1) = delta_fTC;
delta_fTMC = S.kf_TMC*Y(14)*(1.0-(Y(9)+Y(10)))-S.kb_TMC*Y(9);
dY(9, 1) = delta_fTMC;
delta_fTMM = S.kf_TMM*S.Mgi*(1.0-(Y(9)+Y(10)))-S.kb_TMM*Y(10);
dY(10, 1) = delta_fTMM;
delta_fCMi = S.kf_CM*Y(14)*(1.0-Y(5))-S.kb_CM*Y(5);
dY(5, 1) = delta_fCMi;
delta_fCMs = S.kf_CM*Y(13)*(1.0-Y(6))-S.kb_CM*Y(6);
dY(6, 1) = delta_fCMs;
delta_fCQ = S.kf_CQ*Y(11)*(1.0-Y(7))-S.kb_CQ*Y(7);
dY(7, 1) = delta_fCQ;

if ((S.BAPTA_10_mM > 0.0) && (time > S.T1))
   BAPTA = 10.0;
else
   BAPTA = 0.0;
end;

j_Ca_dif = (Y(13)-Y(14))/S.tau_dif_Ca;
V_sub = 0.000000001*2.0*pi*S.L_sub*(S.R_cell-S.L_sub/2.0)*S.L_cell;

if (S.Iso_1_uM > 0.0)
   b_up = -0.25;
elseif (S.ACh > 0.0)
   b_up = 0.7*S.ACh/(0.00009+S.ACh);
else
   b_up = 0.0;
end;

P_up = S.P_up_basal*(1.0-b_up);
j_up = P_up/(1.0+S.K_up/Y(14));
V_cell = 0.000000001*pi*S.R_cell^2.0*S.L_cell;
V_nsr = S.V_nsr_part*V_cell;
V_i = S.V_i_part*V_cell-V_sub;
dY(14, 1) = 1.0*(j_Ca_dif*V_sub-j_up*V_nsr)/V_i-(S.CM_tot*delta_fCMi+S.TC_tot*delta_fTC+S.TMC_tot*delta_fTMC)-(S.kfBAPTA*Y(14)*(BAPTA-Y(15))-S.kbBAPTA*Y(15));
dY(15, 1) = S.kfBAPTA*Y(14)*(BAPTA-Y(15))-S.kbBAPTA*Y(15);
V_jsr = S.V_jsr_part*V_cell;

if ((time > S.t_holding) && (time < S.t_holding+S.t_test))
   V_clamp = S.V_test;
else
   V_clamp = S.V_holding;
end;

if (S.clamp_mode >= 1.0)
   V = V_clamp;
else
   V = Y(17);
end;

RTONF = S.R2*S.T2/S.F;
i_siCa = 2.0*S.P_CaL*(V-0.0)/(RTONF*(1.0-exp(-1.0*(V-0.0)*2.0/RTONF)))*(Y(13)-S.Cao*exp(-2.0*(V-0.0)/RTONF))*Y(19)*Y(21)*Y(20);
i_CaT = 2.0*S.P_CaT*V/(RTONF*(1.0-exp(-1.0*V*2.0/RTONF)))*(Y(13)-S.Cao*exp(-2.0*V/RTONF))*Y(22)*Y(23);
k32 = exp(S.Qn*V/(2.0*RTONF));

if (S.BAPTA_10_mM > 0.0)
   Nai = 7.5;
else
   Nai = Y(18);
end;

k43 = Nai/(S.K3ni+Nai);
di = 1.0+Y(13)/S.Kci*(1.0+exp(-S.Qci*V/RTONF)+Nai/S.Kcni)+Nai/S.K1ni*(1.0+Nai/S.K2ni*(1.0+Nai/S.K3ni));
k14 = Nai/S.K1ni*Nai/S.K2ni*(1.0+Nai/S.K3ni)*exp(S.Qn*V/(2.0*RTONF))/di;
k12 = Y(13)/S.Kci*exp(-S.Qci*V/RTONF)/di;
k41 = exp(-S.Qn*V/(2.0*RTONF));
k34 = S.Nao/(S.K3no+S.Nao);
x2 = k32*k43*(k14+k12)+k41*k12*(k34+k32);
do = 1.0+S.Cao/S.Kco*(1.0+exp(S.Qco*V/RTONF))+S.Nao/S.K1no*(1.0+S.Nao/S.K2no*(1.0+S.Nao/S.K3no));
k21 = S.Cao/S.Kco*exp(S.Qco*V/RTONF)/do;
k23 = S.Nao/S.K1no*S.Nao/S.K2no*(1.0+S.Nao/S.K3no)*exp(-S.Qn*V/(2.0*RTONF))/do;
x1 = k41*k34*(k23+k21)+k21*k32*(k43+k41);
x3 = k14*k43*(k23+k21)+k12*k23*(k43+k41);
x4 = k23*k34*(k14+k12)+k14*k21*(k34+k32);
i_NaCa = S.K_NaCa*(x2*k21-x1*k12)/(x1+x2+x3+x4);
dY(13, 1) = j_SRCarel*V_jsr/V_sub-((i_siCa+i_CaT-2.0*i_NaCa)/(2.0*S.F*V_sub)+j_Ca_dif+S.CM_tot*delta_fCMs)-(S.kfBAPTA*Y(13)*(BAPTA-Y(16))-S.kbBAPTA*Y(16));
dY(16, 1) = S.kfBAPTA*Y(13)*(BAPTA-Y(16))-S.kbBAPTA*Y(16);
j_tr = (Y(12)-Y(11))/S.tau_tr;
dY(12, 1) = j_up-j_tr*V_jsr/V_nsr;
dY(11, 1) = j_tr-(j_SRCarel+S.CQ_tot*delta_fCQ);
E_Na = RTONF*log(S.Nao/Nai);
E_K = RTONF*log(S.Ko/S.Ki);
E_Ca = 0.5*RTONF*log(S.Cao/Y(13));

if (S.Iva_3_uM >= 1.0)
   g_f_Na = S.g_f*(1.0-0.66);
else
   g_f_Na = S.g_f;
end;

if (S.Cs_5_mM >= 1.0)
   ICs_on_Icontrol = 10.6015/5.0/(10.6015/5.0+exp(-0.71*V/25.0));
else
   ICs_on_Icontrol = 1.0;
end;

i_fNa = Y(31)^2.0*S.Ko/(S.Ko+S.Km_f)*g_f_Na*(V-E_Na)*ICs_on_Icontrol;

if (S.Iva_3_uM >= 1.0)
   g_f_K = S.g_f*(1.0-0.66);
else
   g_f_K = S.g_f;
end;
i_fK = Y(31)^2.0*S.Ko/(S.Ko+S.Km_f)*g_f_K*(V-E_K)*ICs_on_Icontrol;
i_f = i_fNa+i_fK;
i_Kr = S.g_Kr*(V-E_K)*(0.9*Y(25)+0.1*Y(26))*Y(27);

if (S.Iso_1_uM > 0.0)
   S.g_Ks = 1.2*S.g_Ks; % 0.0016576
else
   S.g_Ks = S.g_Ks; % 0.0016576;
end;

E_Ks = RTONF*log((S.Ko+0.0*S.Nao)/(S.Ki+0.0*Nai));
i_Ks = S.g_Ks*(V-E_Ks)*Y(28)^2.0;
i_to = S.g_to*(V-E_K)*Y(32)*Y(33);

if (S.Iso_1_uM > 0.0)
   Iso_increase_2 = 1.2;
else
   Iso_increase_2 = 1.0;
end;

i_NaK = Iso_increase_2*S.i_NaK_max*(1.0+(S.Km_Kp/S.Ko)^1.2)^-1.0*(1.0+(S.Km_Nap/Nai)^1.3)^-1.0*(1.0+exp(-(V-E_Na+110.0)/20.0))^-1.0;
E_mh = RTONF*log((S.Nao+0.12*S.Ko)/(Nai+0.12*S.Ki));
i_Na = S.g_Na*Y(30)^3.0*Y(29)*(V-E_mh);
i_siK = 0.000365*S.P_CaL*(V-0.0)/(RTONF*(1.0-exp(-1.0*(V-0.0)/RTONF)))*(S.Ki-S.Ko*exp(-1.0*(V-0.0)/RTONF))*Y(19)*Y(21)*Y(20);
i_siNa = 0.0000185*S.P_CaL*(V-0.0)/(RTONF*(1.0-exp(-1.0*(V-0.0)/RTONF)))*(Nai-S.Nao*exp(-1.0*(V-0.0)/RTONF))*Y(19)*Y(21)*Y(20);
ACh_block = 0.31*S.ACh/(S.ACh+0.00009);

if (S.Iso_1_uM > 0.0)
   Iso_increase_1 = 1.23;
else
   Iso_increase_1 = 1.0;
end;

i_CaL = (i_siCa+i_siK+i_siNa)*(1.0-ACh_block)*1.0*Iso_increase_1;

if (S.ACh > 0.0)
   i_KACh = S.g_KACh*(V-E_K)*(1.0+exp((V+20.0)/20.0))*Y(24);
else
   i_KACh = 0.0;
end;

i_tot = i_f+i_Kr+i_Ks+i_to+i_NaK+i_NaCa+i_Na+i_CaL+i_CaT+i_KACh;
dY(17, 1) = -i_tot/S.C;
dY(18, 1) = -1.0*(i_Na+i_fNa+i_siNa+3.0*i_NaK+3.0*i_NaCa)/(1.0*(V_i+V_sub)*S.F);

if (S.Iso_1_uM > 0.0)
   Iso_shift_1 = -8.0;
else
   Iso_shift_1 = 0.0;
end;

if (S.Iso_1_uM > 0.0)
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
fCa_infinity = S.Km_fCa/(S.Km_fCa+Y(13));
tau_fCa = 0.001*fCa_infinity/S.alpha_fCa;
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
alpha_a = (3.5988-0.025641)/(1.0+0.0000012155/(1.0*S.ACh)^1.6951)+0.025641;
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

if (S.Iso_1_uM > 0.0)
   Iso_shift_2 = -14.0;
else
   Iso_shift_2 = 0.0;
end;

n_infinity = 14.0/(1.0+exp(-(V-40.0-Iso_shift_2)/12.0))/(14.0/(1.0+exp(-(V-40.0-Iso_shift_2)/12.0))+1.0*exp(-(V-Iso_shift_2)/45.0));
alpha_n = 28.0/(1.0+exp(-(V-40.0-Iso_shift_2)/3.0));
beta_n = 1.0*exp(-(V-Iso_shift_2-S.shift-5.0)/25.0);
tau_n = 1.0/(alpha_n+beta_n);
dY(28, 1) = (n_infinity-Y(28))/tau_n;
alpha_h = 20.0*exp(-0.125*(V+75.0));
beta_h = 2000.0/(320.0*exp(-0.1*(V+75.0))+1.0);
dY(29, 1) = alpha_h*(1.0-Y(29))-beta_h*Y(29);
E0_m = V+41.0;

if (abs(E0_m) < S.delta_m)
   alpha_m = 2000.0;
else
   alpha_m = 200.0*E0_m/(1.0-exp(-0.1*E0_m));
end;

beta_m = 8000.0*exp(-0.056*(V+66.0));
dY(30, 1) = alpha_m*(1.0-Y(30))-beta_m*Y(30);

if (S.ACh > 0.0)
   ACh_shift = -1.0-9.898*(1.0*S.ACh)^0.618/((1.0*S.ACh)^0.618+0.00122423);
else
   ACh_shift = 0.0;
end;

if (S.Iso_1_uM > 0.0)
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
