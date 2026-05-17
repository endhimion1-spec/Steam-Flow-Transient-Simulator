Import numpy as np
from scipy.optimize import fsolve
from iapws import IAPWS97

# ==========================================
# 1. 定数・配管仕様・環境設定
# ==========================================
L = 100.0           # 配管長 [m]
N_elem = 5          # 要素数
N_nodes = N_elem + 1
dx = L / N_elem
dt = 0.5            # 時間ステップ [s]
total_time = 30.0   # シミュレーション時間 [s]

# 配管寸法 (100A 鋼管相当)
D_in = 0.1023       # 管内径 [m]
D_out = 0.1143      # 管外径 [m]
A = np.pi * (D_in**2) / 4
roughness = 0.000045 # 管内粗度 [m]

# 鋼材物性
rho_steel = 7850.0  # [kg/m3]
Cp_steel = 460.0    # [J/kg·K]
lambda_steel = 50.0 # [W/m·K]
m_pipe_per_m = rho_steel * np.pi * (D_out**2 - D_in**2) / 4 # 単位長さ質量

# 断熱材・外気条件
ins_thickness = 0.050 # 50mm
D_ins = D_out + 2 * ins_thickness
lambda_ins = 0.04     # グラスウール相当 [W/m·K]
T_amb = 273.15 + 10.0 # 外気温 10℃
alpha_out = 15.0      # 外表面熱伝達率 [W/m2·K]

# 境界条件 (境界の外側の状態)
P_up = 1.0e6          # 上流供給圧力 [Pa] (1.0 MPa)
T_up = 473.15         # 上流供給温度 [K] (200 °C)
P_atm = 0.1013e6      # 下流大気圧 [Pa]
zeta_L = 5.0          # 左端弁 抵抗係数
zeta_R = 10.0         # 右端弁 抵抗係数

# ==========================================
# 2. 物理計算用ヘルパー関数
# ==========================================
def get_props(P_pa, h_jkg):
    """IAPWS-IF97による蒸気物性計算"""
    s = IAPWS97(P=P_pa*1e-6, h=h_jkg*1e-3)
    # rho[kg/m3], T[K], mu[Pa·s], k[W/m·K], Pr[-], internal_energy[J/kg]
    return s.rho, s.T, s.mu, s.k, s.Pr, s.u * 1e3

def friction_factor(Re, D, eps):
    """Haalandの式による摩擦係数計算"""
    if Re < 2300: return 64 / max(Re, 1.0)
    return (1.8 * np.log10((eps/D/3.7)**1.11 + 6.9/Re))**-2

# ==========================================
# 3. 非定常（慣性・熱容量込）残差方程式
# ==========================================
def residuals_full(vars_new, vars_old, T_pipe_old):
    res = np.zeros(len(vars_new))
    P = vars_new[0::3]; h = vars_new[1::3]; u = vars_new[2::3]
    P_o = vars_old[0::3]; h_o = vars_old[1::3]; u_o = vars_old[2::3]
    
    # 全ノードの物性を計算
    p_n = [get_props(P[i], h[i]) for i in range(N_nodes)]
    p_o = [get_props(P_o[i], h_o[i]) for i in range(N_nodes)]

    # --- 境界条件: 左端弁 (準定常圧損としてモデル化) ---
    res[0] = P_up - P[0] - zeta_L * (p_n[0][0] * u[0]**2 / 2)

    # --- 要素ごとの保存則計算 ---
    for i in range(N_elem):
        idx = 1 + i*3
        # 平均物性 (New/Old)
        rho_n = (p_n[i][0] + p_n[i+1][0]) / 2
        rho_o = (p_o[i][0] + p_o[i+1][0]) / 2
        u_avg_n = (u[i] + u[i+1]) / 2
        
        # (a) 連続の式: d(rho)/dt + d(rho*u)/dx = 0
        term_t_mass = (rho_n - rho_o) / dt
        term_x_mass = (p_n[i+1][0]*u[i+1] - p_n[i][0]*u[i]) / dx
        res[idx] = term_t_mass + term_x_mass
        
        # (b) 運動量方程式: d(rho*u)/dt + d(rho*u^2 + P)/dx + friction = 0
        # 慣性項 (_mom)
        mom_n = (p_n[i][0]*u[i] + p_n[i+1][0]*u[i+1]) / 2
        mom_o = (p_o[i][0]*u_o[i] + p_o[i+1][0]*u_o[i+1]) / 2
        term_t_mom = (mom_n - mom_o) / dt
        # 対流・圧力勾配項
        term_x_mom = ((p_n[i+1][0]*u[i+1]**2 + P[i+1]) - (p_n[i][0]*u[i]**2 + P[i])) / dx
        # 摩擦項 (Pa/m)
        mu_avg = (p_n[i][2] + p_n[i+1][2]) / 2
        Re = rho_n * abs(u_avg_n) * D_in / mu_avg
        f = friction_factor(Re, D_in, roughness)
        f_loss = f * (1/D_in) * (rho_n * u_avg_n**2 / 2)
        
        res[idx+1] = term_t_mom + term_x_mom + f_loss

        # (c) エネルギー方程式: d(rho*E)/dt + d(rho*u*H)/dx = Q/A
        # E: 全内部エネルギー, H: 全エンタルピー
        E_n = ( (p_n[i][5]+0.5*u[i]**2) + (p_n[i+1][5]+0.5*u[i+1]**2) ) / 2
        E_o = ( (p_o[i][5]+0.5*u_o[i]**2) + (p_o[i+1][5]+0.5*u_o[i+1]**2) ) / 2
        H_i = h[i] + 0.5*u[i]**2
        H_i1 = h[i+1] + 0.5*u[i+1]**2
        
        # 4層熱抵抗モデル (蒸気 -> 管壁温度 T_pipe_old)
        k_avg = (p_n[i][3] + p_n[i+1][3]) / 2
        Pr_avg = (p_n[i][4] + p_n[i+1][4]) / 2
        Nu = 0.023 * (Re**0.8) * (Pr_avg**0.4)
        alpha_in = Nu * k_avg / D_in
        R_in = 1 / (np.pi * D_in * alpha_in)
        
        T_steam_avg = (p_n[i][1] + p_n[i+1][1]) / 2
        q_to_pipe = (T_steam_avg - T_pipe_old[i]) / R_in  # [W/m]
        
        res[idx+2] = (rho_n*E_n - rho_o*E_o)/dt + (p_n[i+1][0]*u[i+1]*H_i1 - p_n[i][0]*u[i]*H_i)/dx + q_to_pipe/A

    # --- 境界条件: 右端弁 ---
    res[-1] = P[-1] - P_atm - zeta_R * (p_n[-1][0] * u[-1]**2 / 2)
    return res

# ==========================================
# 4. シミュレーション実行ループ
# ==========================================
# 初期化 (配管は外気温、内圧は大気圧)
inlet_h = IAPWS97(P=P_up*1e-6, T=T_up).h * 1e3
current_vars = []
for i in range(N_nodes):
    current_vars.extend([P_up if i==0 else P_atm, inlet_h, 0.1])
current_vars = np.array(current_vars)
T_pipe = np.ones(N_elem) * T_amb # 配管壁温度初期値

print(f"{'Time [s]':<8} | {'P_mid [MPa]':<12} | {'u_end [m/s]':<10} | {'T_pipe_avg [°C]':<15}")
print("-" * 60)

for t_step in range(int(total_time/dt)):
    # 陰解法ソルバーで次のステップの (P, h, u) を算出
    try:
        new_vars = fsolve(residuals_full, current_vars, args=(current_vars, T_pipe), xtol=1e-5)
    except:
        print("収束エラーが発生しました。時間ステップを小さくしてください。")
        break
    
    # 配管温度 T_pipe の更新 (熱容量に基づく時間発展)
    P_n = new_vars[0::3]; h_n = new_vars[1::3]; u_n = new_vars[2::3]
    for i in range(N_elem):
        rho, T_s, mu, k, pr, _ = get_props((P_n[i]+P_n[i+1])/2, (h_n[i]+h_n[i+1])/2)
        Re = (rho * abs(u_n[i]+u_n[i+1])/2 * D_in) / mu
        alpha_in = (0.023 * Re**0.8 * pr**0.4) * k / D_in
        
        Q_in = (T_s - T_pipe[i]) / (1 / (np.pi * D_in * alpha_in))
        R_out = (np.log(D_out/D_in)/(2*np.pi*lambda_steel) + 
                 np.log(D_ins/D_out)/(2*np.pi*lambda_ins) + 
                 1/(np.pi*D_ins*alpha_out))
        Q_out = (T_pipe[i] - T_amb) / R_out
        
        # 配管の熱収支
        T_pipe[i] += (Q_in - Q_out) / (m_pipe_per_m * Cp_steel) * dt
    
    current_vars = new_vars
    
    # 2秒おきに出力
    if t_step % 4 == 0:
        p_mid = new_vars[len(new_vars)//2] * 1e-6
        print(f"{t_step*dt:<8.1f} | {p_mid:<12.4f} | {u_n[-1]:<10.2f} | {np.mean(T_pipe)-273.15:<15.2f}")

print("\nシミュレーション完了。")
