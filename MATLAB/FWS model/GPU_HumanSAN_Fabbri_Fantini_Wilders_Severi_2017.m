function [dY, i_tot] = GPU_HumanSAN_Fabbri_Fantini_Wilders_Severi_2017(time, Y,S) %#codegen

% Add kernelfun pragma to trigger kernel creation
coder.gpu.kernelfun;

%-------------------------------------------------------------------------------
% Computation
%-------------------------------------------------------------------------------

dY = zeros(33,1); % Preallocate state vector for GPU coder

% time (second)

j_SRCarel = S.ks*Y(2)*(Y(11)-Y(13));
% diff = Y(11)-Y(13);
kCaSR = S.MaxSR-(S.MaxSR-S.MinSR)/(1.0+(S.EC50_SR/Y(11))^S.HSR);
koSRCa = S.koCa/kCaSR;
kiSRCa = S.kiCa*kCaSR;
dY(3, 1) = S.kim*Y(4)-kiSRCa*Y(13)*Y(3)-(koSRCa*Y(13)^2.0*Y(3)-S.kom*Y(2));
dY(2, 1) = koSRCa*Y(13)^2.0*Y(3)-S.kom*Y(2)-(kiSRCa*Y(13)*Y(2)-S.kim*Y(1));
dY(1, 1) = kiSRCa*Y(13)*Y(2)-S.kim*Y(1)-(S.kom*Y(1)-koSRCa*Y(13)^2.0*Y(4));
dY(4, 1) = S.kom*Y(1)-koSRCa*Y(13)^2.0*Y(4)-(S.kim*Y(4)-kiSRCa*Y(13)*Y(3));
% P_tot = Y(3)+Y(2)+Y(1)+Y(4);
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
j_Ca_dif = (Y(13)-Y(14))/S.tau_dif_Ca;
V_sub = 0.000000001*2.0*pi*S.L_sub*(S.R_cell-S.L_sub/2.0)*S.L_cell;

if (S.Iso_1_uM > 0.0)
   b_up = -0.25;
elseif (S.ACh > 0.0)
   b_up = 0.7*S.ACh/(0.00009+S.ACh);
else
   b_up = 0.0;
%    b_up = zeros(1,1,'gpuArray');
end

P_up = S.P_up_basal*(1.0-b_up);
j_up = P_up/(1.0+exp((-Y(14)+S.K_up)/S.slope_up));
V_cell = 0.000000001*pi*S.R_cell^2.0*S.L_cell;
V_nsr = S.V_nsr_part*V_cell;
V_i = S.V_i_part*V_cell-V_sub;
dY(14, 1) = 1.0*(j_Ca_dif*V_sub-j_up*V_nsr)/V_i-(S.CM_tot*delta_fCMi+S.TC_tot*delta_fTC+S.TMC_tot*delta_fTMC);
V_jsr = S.V_jsr_part*V_cell;

if ((time > S.t_holding) && (time < S.t_holding+S.t_test))
   V_clamp = S.V_test;
else
   V_clamp = S.V_holding;
end

if (S.clamp_mode >= 1.0)
   V = V_clamp;
else
   V = Y(15);
end

RTONF = S.R2*S.T/S.F;
i_siCa = 2.0*S.P_CaL*(V-0.0)/(RTONF*(1.0-exp(-1.0*(V-0.0)*2.0/RTONF)))*(Y(13)-S.Cao*exp(-2.0*(V-0.0)/RTONF))*Y(17)*Y(19)*Y(18);
i_CaT = 2.0*S.P_CaT*V/(RTONF*(1.0-exp(-1.0*V*2.0/RTONF)))*(Y(13)-S.Cao*exp(-2.0*V/RTONF))*Y(20)*Y(21);
k32 = exp(S.Qn*V/(2.0*RTONF));
Nai = Y(16);
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
i_NaCa = (1.0-S.blockade_NaCa)*S.K_NaCa*(x2*k21-x1*k12)/(x1+x2+x3+x4);
dY(13, 1) = j_SRCarel*V_jsr/V_sub-((i_siCa+i_CaT-2.0*i_NaCa)/(2.0*S.F*V_sub)+j_Ca_dif+S.CM_tot*delta_fCMs);
j_tr = (Y(12)-Y(11))/S.tau_tr;
dY(12, 1) = j_up-j_tr*V_jsr/V_nsr;
dY(11, 1) = j_tr-(j_SRCarel+S.CQ_tot*delta_fCQ);
E_Na = RTONF*log(S.Nao/Nai);
E_K = RTONF*log(S.Ko/S.Ki);
% E_Ca = 0.5*RTONF*log(S.Cao/Y(13));
G_f = S.g_f/(S.Ko/(S.Ko+S.Km_f));
G_f_K = G_f/(S.alpha+1.0);
G_f_Na = S.alpha*G_f_K;
g_f_Na = G_f_Na*S.Ko/(S.Ko+S.Km_f);
i_fNa = Y(31)*g_f_Na*(V-E_Na)*(1.0-S.blockade);
g_f_K = G_f_K*S.Ko/(S.Ko+S.Km_f);
i_fK = Y(31)*g_f_K*(V-E_K)*(1.0-S.blockade);
i_f = i_fNa+i_fK;
i_Kr = S.g_Kr*(V-E_K)*(0.9*Y(23)+0.1*Y(24))*Y(25);

if (S.Iso_1_uM > 0.0)
   g_Ks = 1.2*S.g_Ks_;
else
   g_Ks = S.g_Ks_;
end

E_Ks = RTONF*log((S.Ko+0.12*S.Nao)/(S.Ki+0.12*Nai));
i_Ks = g_Ks*(V-E_Ks)*Y(26)^2.0;
i_to = S.g_to*(V-E_K)*Y(32)*Y(33);

if (S.Iso_1_uM > 0.0)
   Iso_increase_2 = 1.2;
%    Iso_increase_2 = 1.2*ones(1,1,'gpuArray');
else
   Iso_increase_2 = 1.0;
%    Iso_increase_2 = 1.0*ones(1,1,'gpuArray');
end

i_NaK = Iso_increase_2*S.i_NaK_max*(1.0+(S.Km_Kp/S.Ko)^1.2)^-1.0*(1.0+(S.Km_Nap/Nai)^1.3)^-1.0*(1.0+exp(-(V-E_Na+110.0)/20.0))^-1.0;
E_mh = RTONF*log((S.Nao+0.12*S.Ko)/(Nai+0.12*S.Ki));
i_Na_ = S.g_Na*Y(30)^3.0*Y(29)*(V-E_mh);
i_Na_L = S.g_Na_L*Y(30)^3.0*(V-E_mh);
i_Na = i_Na_+i_Na_L;
i_siK = 0.000365*S.P_CaL*(V-0.0)/(RTONF*(1.0-exp(-1.0*(V-0.0)/RTONF)))*(S.Ki-S.Ko*exp(-1.0*(V-0.0)/RTONF))*Y(17)*Y(19)*Y(18);
i_siNa = 0.0000185*S.P_CaL*(V-0.0)/(RTONF*(1.0-exp(-1.0*(V-0.0)/RTONF)))*(Nai-S.Nao*exp(-1.0*(V-0.0)/RTONF))*Y(17)*Y(19)*Y(18);
ACh_block = 0.31*S.ACh/(S.ACh+0.00009);

if (S.Iso_1_uM > 0.0)
   Iso_increase_1 = 1.23;
%    Iso_increase_1 = 1.23*ones(1,1,'gpuArray');
else
   Iso_increase_1 = 1.0;
%    Iso_increase_1 = 1.0*ones(1,1,'gpuArray');
end

i_CaL = (i_siCa+i_siK+i_siNa)*(1.0-ACh_block)*1.0*Iso_increase_1;

if (S.ACh > 0.0)
   i_KACh = S.ACh_on*S.g_KACh*(V-E_K)*(1.0+exp((V+20.0)/20.0))*Y(22);
else
   i_KACh = 0.0;
%    i_KACh = zeros(1,1,'gpuArray');
end

i_Kur = S.g_Kur*Y(27)*Y(28)*(V-E_K);
i_tot = i_f+i_Kr+i_Ks+i_to+i_NaK+i_NaCa+i_Na+i_CaL+i_CaT+i_KACh+i_Kur;
dY(15, 1) = -i_tot/S.C;
dY(16, 1) = (1.0-S.Nai_clamp)*-1.0*(i_Na+i_fNa+i_siNa+3.0*i_NaK+3.0*i_NaCa)/(1.0*(V_i+V_sub)*S.F);

if (S.Iso_1_uM > 0.0)
   Iso_shift_dL = -8.0;
%    Iso_shift_dL = -8.0*ones(1,1,'gpuArray');
else
   Iso_shift_dL = 0.0;
%    Iso_shift_dL = zeros(1,1,'gpuArray');
end

if (S.Iso_1_uM > 0.0)
   Iso_slope_dL = -27.0;
%    Iso_slope_dL = -27.0*ones(1,1,'gpuArray');
else
   Iso_slope_dL = 0.0;
%     Iso_slope_dL = zeros(1,1,'gpuArray');
end

dL_infinity = 1.0/(1.0+exp(-(V-S.V_dL-Iso_shift_dL)/(S.k_dL*(1.0+Iso_slope_dL/100.0))));

if (V == -41.8)
   adVm = -41.80001;
elseif (V == 0.0)
   adVm = 0.0;
elseif (V == -6.8)
   adVm = -6.80001;
%    adVm = -6.80001*ones(1,1,'gpuArray');
else
   adVm = V;
end

alpha_dL = -0.02839*(adVm+41.8)/(exp(-(adVm+41.8)/2.5)-1.0)-0.0849*(adVm+6.8)/(exp(-(adVm+6.8)/4.8)-1.0);

if (V == -1.8)
   bdVm = -1.80001;
%    bdVm = -1.80001*ones(1,1,'gpuArray');
else
   bdVm = V;
end

beta_dL = 0.01143*(bdVm+1.8)/(exp((bdVm+1.8)/2.5)-1.0);
tau_dL = 0.001/(alpha_dL+beta_dL);
dY(17, 1) = (dL_infinity-Y(17))/tau_dL;
fCa_infinity = S.Km_fCa/(S.Km_fCa+Y(13));
tau_fCa = 0.001*fCa_infinity/S.alpha_fCa;
dY(18, 1) = (fCa_infinity-Y(18))/tau_fCa;
fL_infinity = 1.0/(1.0+exp((V+37.4+S.shift_fL)/(5.3+S.k_fL)));
tau_fL = 0.001*(44.3+230.0*exp(-((V+36.0)/10.0)^2.0));
dY(19, 1) = (fL_infinity-Y(19))/tau_fL;
dT_infinity = 1.0/(1.0+exp(-(V+38.3)/5.5));
tau_dT = 0.001/(1.068*exp((V+38.3)/30.0)+1.068*exp(-(V+38.3)/30.0));
dY(20, 1) = (dT_infinity-Y(20))/tau_dT;
fT_infinity = 1.0/(1.0+exp((V+58.7)/3.8));
tau_fT = 1.0/(16.67*exp(-(V+75.0)/83.3)+16.67*exp((V+75.0)/15.38))+S.offset_fT;
dY(21, 1) = (fT_infinity-Y(21))/tau_fT;
alpha_a = (3.5988-0.025641)/(1.0+0.0000012155/(1.0*S.ACh)^1.6951)+0.025641;
beta_a = 10.0*exp(0.0133*(V+40.0));
a_infinity = alpha_a/(alpha_a+beta_a);
tau_a = 1.0/(alpha_a+beta_a);
dY(22, 1) = (a_infinity-Y(22))/tau_a;
% alfapaF = 1.0/(1.0+exp(-(V+23.2)/6.6))/(0.84655354/(37.2*exp(V/11.9)+0.96*exp(-V/18.5)));
% betapaF = 4.0*((37.2*exp(V/15.9)+0.96*exp(-V/22.5))/0.84655354-1.0/(1.0+exp(-(V+23.2)/10.6))/(0.84655354/(37.2*exp(V/15.9)+0.96*exp(-V/22.5))));
pa_infinity = 1.0/(1.0+exp(-(V+10.0144)/7.6607));
tau_paS = 0.84655354/(4.2*exp(V/17.0)+0.15*exp(-V/21.6));
tau_paF = 1.0/(30.0*exp(V/10.0)+exp(-V/12.0));
dY(24, 1) = (pa_infinity-Y(24))/tau_paS;
dY(23, 1) = (pa_infinity-Y(23))/tau_paF;
tau_pi = 1.0/(100.0*exp(-V/54.645)+656.0*exp(V/106.157));
pi_infinity = 1.0/(1.0+exp((V+28.6)/17.1));
dY(25, 1) = (pi_infinity-Y(25))/tau_pi;

if (S.Iso_1_uM > 0.0)
   Iso_shift_1 = -14.0;
%    Iso_shift_1 = -14.0*ones(1,1,'gpuArray');
else
   Iso_shift_1 = 0.0;
%    Iso_shift_1 = zeros(1,1,'gpuArray');
end

n_infinity = sqrt(1.0/(1.0+exp(-(V+0.6383-Iso_shift_1)/10.7071)));
alpha_n = 28.0/(1.0+exp(-(V-40.0-Iso_shift_1)/3.0));
beta_n = 1.0*exp(-(V-Iso_shift_1-5.0)/25.0);
tau_n = 1.0/(alpha_n+beta_n);
dY(26, 1) = (n_infinity-Y(26))/tau_n;
r_Kur_infinity = 1.0/(1.0+exp((V+6.0)/-8.6));
tau_r_Kur = 0.009/(1.0+exp((V+5.0)/12.0))+0.0005;
dY(27, 1) = (r_Kur_infinity-Y(27))/tau_r_Kur;
s_Kur_infinity = 1.0/(1.0+exp((V+7.5)/10.0));
tau_s_Kur = 0.59/(1.0+exp((V+60.0)/10.0))+3.05;
dY(28, 1) = (s_Kur_infinity-Y(28))/tau_s_Kur;
h_infinity = 1.0/(1.0+exp((V+69.804)/4.4565));
alpha_h = 20.0*exp(-0.125*(V+75.0));
beta_h = 2000.0/(320.0*exp(-0.1*(V+75.0))+1.0);
tau_h = 1.0/(alpha_h+beta_h);
dY(29, 1) = (h_infinity-Y(29))/tau_h;
m_infinity = 1.0/(1.0+exp(-(V+42.0504)/8.3106));
E0_m = V+41.0;

if (abs(E0_m) < S.delta_m)
   alpha_m = 2000.0;
%    alpha_m = 2000.0*ones(1,1,'gpuArray');
else
   alpha_m = 200.0*E0_m/(1.0-exp(-0.1*E0_m));
end

beta_m = 8000.0*exp(-0.056*(V+66.0));
tau_m = 1.0/(alpha_m+beta_m);
dY(30, 1) = (m_infinity-Y(30))/tau_m;

if (S.ACh > 0.0)
   ACh_shift = -1.0-9.898*(1.0*S.ACh)^0.618/((1.0*S.ACh)^0.618+0.00122423);
else
   ACh_shift = 0.0;
%    ACh_shift = zeros(1,1,'gpuArray');
end

if (S.Iso_1_uM > 0.0)
   Iso_shift_2 = 7.5;
%    Iso_shift_2 = 7.5*ones(1,1,'gpuArray');
else
   Iso_shift_2 = 0.0;
%    Iso_shift_2 = zeros(1,1,'gpuArray');
end

tau_y = 1.0/(0.36*(V+148.8-ACh_shift-Iso_shift_2)/(exp(0.066*(V+148.8-ACh_shift-Iso_shift_2))-1.0)+0.1*(V+87.3-ACh_shift-Iso_shift_2)/(1.0-exp(-0.2*(V+87.3-ACh_shift-Iso_shift_2))))-0.054;

if (V < -(80.0-ACh_shift-Iso_shift_2-S.y_shift))
   y_infinity = 0.01329+0.99921/(1.0+exp((V+97.134-ACh_shift-Iso_shift_2-S.y_shift)/8.1752));
else
   y_infinity = 0.0002501*exp(-(V-ACh_shift-Iso_shift_2-S.y_shift)/12.861);
end

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
