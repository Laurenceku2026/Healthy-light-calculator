import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import StringIO
import re
import json
import os
from datetime import datetime
import plotly.io as pio
import requests

# ==================== 页面配置 ====================
st.set_page_config(page_title="健康光计算器 (EML / m-EDI)", layout="wide")

# ==================== 常量定义 ====================
KM = 683.002  # 明视觉最大光谱光视效能 (lm/W)

# ==================== 标准网格定义 ====================
STANDARD_WAVELENGTHS = list(range(380, 781, 5))  # 380-780nm，步长5nm
STANDARD_DELTA = 5.0  # 标准网格步长 (nm)

# ==================== Supabase 配置 ====================
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_SERVICE_ROLE_KEY"]
except KeyError as e:
    st.error(f"""
    ❌ **配置错误：缺少 Supabase 密钥**
    
    请确保在 Streamlit Cloud 的 Secrets 中配置了：
    - `SUPABASE_URL`
    - `SUPABASE_SERVICE_ROLE_KEY`
    
    当前缺失的密钥: `{e.args[0]}`
    """)
    st.stop()

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json"
}

def supabase_get(table: str, user_id: str = None, id_field: str = "id"):
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    if user_id:
        url += f"?{id_field}=eq.{user_id}"
    response = requests.get(url, headers=HEADERS)
    return response

def supabase_patch(table: str, user_id: str, data: dict):
    url = f"{SUPABASE_URL}/rest/v1/{table}?id=eq.{user_id}"
    response = requests.patch(url, headers=HEADERS, json=data)
    return response

def supabase_post(table: str, data: dict):
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    response = requests.post(url, headers=HEADERS, json=data)
    return response

def get_user_profile(user_id: str):
    if not user_id or user_id == "admin":
        return {"subscription_tier": "free", "free_trials_remaining": 30, "subscription_expires_at": None}
    try:
        response = supabase_get("profiles", user_id)
        if response.status_code == 200 and response.json():
            data = response.json()[0]
            return {
                "subscription_tier": data.get("subscription_tier", "free"),
                "free_trials_remaining": data.get("free_trials_remaining", 30),
                "subscription_expires_at": data.get("subscription_expires_at")
            }
    except Exception:
        pass
    return {"subscription_tier": "free", "free_trials_remaining": 30, "subscription_expires_at": None}

def get_user_remaining_trials(user_id: str) -> int:
    try:
        response = supabase_get("profiles", user_id)
        if response.status_code == 200 and response.json():
            remaining = response.json()[0].get("free_trials_remaining", 30)
            tier = response.json()[0].get("subscription_tier", "free")
            if tier == "pro":
                return -1
            return remaining
    except Exception:
        pass
    return st.session_state.get("trials_left", 30)

def consume_trial(user_id: str, app_name: str) -> tuple:
    try:
        resp = supabase_get("profiles", user_id)
        if resp.status_code != 200 or not resp.json():
            return False, 0, "用户不存在"
        
        current = resp.json()[0].get("free_trials_remaining", 30)
        tier = resp.json()[0].get("subscription_tier", "free")
        
        if tier == "pro":
            return True, -1, ""
        
        if current <= 0:
            return False, 0, "免费次数已用完（共30次），请联系管理员升级"
        
        patch_resp = supabase_patch("profiles", user_id, {"free_trials_remaining": current - 1})
        
        if patch_resp.status_code not in [200, 204]:
            return False, 0, f"更新失败: {patch_resp.text}"
        
        supabase_post("usage_logs", {
            "user_id": user_id,
            "app_name": app_name,
            "analysis_count": 1,
            "used_at": datetime.now().isoformat()
        })
        
        return True, current - 1, ""
        
    except Exception as e:
        return False, 0, f"计数失败: {str(e)}"

# ==================== 接收门户参数 ====================
query_params = st.query_params

if "user_id" in query_params:
    user_id_val = query_params["user_id"]
    if isinstance(user_id_val, list):
        st.session_state.user_id = user_id_val[0]
    else:
        st.session_state.user_id = user_id_val
    
    email_val = query_params.get("email", "")
    if isinstance(email_val, list):
        st.session_state.user_email = email_val[0] if email_val else ""
    else:
        st.session_state.user_email = email_val
    
    if st.session_state.user_email and "@" in st.session_state.user_email:
        st.session_state.username = st.session_state.user_email.split('@')[0]
    else:
        st.session_state.username = "User"
    
    if "lang" in query_params:
        lang_val = query_params["lang"]
        if isinstance(lang_val, list):
            lang_val = lang_val[0]
        st.session_state.lang = lang_val if lang_val in ["zh", "en"] else "zh"
    else:
        st.session_state.lang = "zh"
else:
    st.warning("请从 TechLife Suite 门户登录后访问")
    st.stop()

# ==================== 预设数据（CIE S 026:2018 标准）====================
# V(λ) 明视觉数据（峰值 555nm = 1.0）
DEFAULT_V_LAMBDA = [
    0.000039, 0.000064, 0.000120, 0.000217, 0.000396, 0.000640, 0.001210, 0.002180, 0.004000, 0.007300,
    0.011600, 0.016840, 0.023000, 0.029800, 0.038000, 0.048000, 0.060000, 0.073900, 0.090980, 0.112600,
    0.139020, 0.169300, 0.208020, 0.258600, 0.323000, 0.407300, 0.503000, 0.608200, 0.710000, 0.793200,
    0.862000, 0.914850, 0.954000, 0.980300, 0.994950, 1.000000, 0.995000, 0.978600, 0.952000, 0.915400,
    0.870000, 0.816300, 0.757000, 0.694900, 0.631000, 0.566800, 0.503000, 0.441200, 0.381000, 0.321000,
    0.265000, 0.217000, 0.175000, 0.138200, 0.107000, 0.081600, 0.061000, 0.044580, 0.032000, 0.023200,
    0.017000, 0.011920, 0.008210, 0.005723, 0.004102, 0.002929, 0.002091, 0.001484, 0.001047, 0.000740,
    0.000520, 0.000361, 0.000249, 0.000172, 0.000120, 0.000085, 0.000060, 0.000042, 0.000030, 0.000021,
    0.000015
]

# Nz(λ) 黑视素数据（CIE S 026:2018 标准，峰值 490nm = 1.0）
DEFAULT_NZ_LAMBDA = [
    0.000000, 0.000000, 0.000000, 0.000000, 0.000100, 0.000200, 0.000400, 0.000700, 0.001300, 0.002200,
    0.003500, 0.005500, 0.008000, 0.012000, 0.018000, 0.026000, 0.038000, 0.054000, 0.077000, 0.110000,
    0.158000, 0.223000, 0.304000, 0.402000, 0.520000, 0.650000, 0.780000, 0.895000, 0.978000, 0.990000,
    0.940000, 0.840000, 0.710000, 0.580000, 0.470000, 0.380000, 0.310000, 0.250000, 0.200000, 0.160000,
    0.130000, 0.100000, 0.080000, 0.062000, 0.047000, 0.036000, 0.027000, 0.021000, 0.016000, 0.012000,
    0.009000, 0.007000, 0.005000, 0.004000, 0.003200, 0.002500, 0.002000, 0.001600, 0.001300, 0.001000,
    0.000800, 0.000600, 0.000500, 0.000400, 0.000300, 0.000250, 0.000200, 0.000150, 0.000120, 0.000100,
    0.000080, 0.000060, 0.000050, 0.000040, 0.000030, 0.000025, 0.000020, 0.000015, 0.000010, 0.000008,
    0.000005
]

# ==================== 调试模式 ====================
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False

# ==================== 积分函数 ====================
def trapezoid(y, x):
    """梯形积分，自动适配步长"""
    y = np.asarray(y)
    x = np.asarray(x)
    
    if len(y) != len(x):
        raise ValueError("y and x must have the same length")
    if len(y) < 2:
        return 0.0
    
    try:
        return np.trapezoid(y, x)
    except AttributeError:
        try:
            return np.trapz(y, x)
        except AttributeError:
            dx = np.diff(x)
            return np.sum((y[:-1] + y[1:]) * dx / 2)

# ==================== 光谱数据管理（全局配置表 app_config）====================
def load_spectral_data(debug=False):
    """从全局配置表 app_config 加载光谱数据"""
    try:
        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/app_config?id=eq.1",
            headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
        )
        if response.status_code == 200 and response.json():
            data = response.json()[0]
            spectral_data = data.get("spectral_data")
            if spectral_data:
                saved = json.loads(spectral_data)
                v_lambda = saved.get('v_lambda', DEFAULT_V_LAMBDA)
                nz_lambda = saved.get('nz_lambda', DEFAULT_NZ_LAMBDA)
                if debug:
                    st.info("📂 从全局配置表加载光谱数据")
                return v_lambda, nz_lambda
    except Exception as e:
        if debug:
            st.warning(f"加载失败，使用预设数据: {e}")
    
    return DEFAULT_V_LAMBDA, DEFAULT_NZ_LAMBDA

def save_spectral_data(v_lambda, nz_lambda):
    """保存光谱数据到全局配置表 app_config"""
    try:
        data = {
            'v_lambda': [float(x) for x in v_lambda],
            'nz_lambda': [float(x) for x in nz_lambda],
            'last_updated': datetime.now().isoformat()
        }
        response = requests.patch(
            f"{SUPABASE_URL}/rest/v1/app_config?id=eq.1",
            headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}", "Content-Type": "application/json"},
            json={"spectral_data": json.dumps(data), "updated_at": datetime.now().isoformat()}
        )
        return response.status_code in [200, 204]
    except Exception as e:
        print(f"保存失败: {e}")
        return False

def reset_to_default():
    """重置为预设数据"""
    return save_spectral_data(DEFAULT_V_LAMBDA, DEFAULT_NZ_LAMBDA)

# ==================== 插值函数 ====================
def interpolate_energy_conserving(x_input, y_input, x_target):
    """能量守恒插值 - 用于用户光谱数据"""
    x_input = np.asarray(x_input)
    y_input = np.asarray(y_input)
    x_target = np.asarray(x_target)
    
    if len(x_input) < 2 or len(y_input) < 2:
        return np.zeros_like(x_target)
    
    dx_input = np.zeros_like(x_input)
    dx_input[0] = x_input[1] - x_input[0]
    dx_input[1:-1] = (x_input[2:] - x_input[:-2]) / 2
    dx_input[-1] = x_input[-1] - x_input[-2]
    
    energy_input = y_input * dx_input
    energy_target = np.interp(x_target, x_input, energy_input, left=0, right=0)
    
    dx_target = np.zeros_like(x_target)
    dx_target[0] = x_target[1] - x_target[0]
    dx_target[1:-1] = (x_target[2:] - x_target[:-2]) / 2
    dx_target[-1] = x_target[-1] - x_target[-2]
    
    dx_target_safe = np.where(dx_target > 0, dx_target, 1.0)
    y_target = energy_target / dx_target_safe
    
    return y_target

def interpolate_linear(x_input, y_input, x_target):
    """线性插值 - 用于 V(λ) 和 Nz(λ)"""
    x_input = np.asarray(x_input)
    y_input = np.asarray(y_input)
    x_target = np.asarray(x_target)
    
    if len(x_input) < 2 or len(y_input) < 2:
        return np.zeros_like(x_target)
    
    return np.interp(x_target, x_input, y_input, left=0, right=0)

# ==================== 加载并自动归一化 Nz(λ) ====================
def get_normalized_spectral_data(debug=False):
    """加载光谱数据并自动积分归一化 Nz(λ)"""
    v_lambda, nz_lambda = load_spectral_data(debug)
    wavelengths = np.asarray(STANDARD_WAVELENGTHS)
    
    # V(λ)：峰值归一化到 1.0
    v_array = np.asarray(v_lambda)
    v_max = np.max(v_array)
    if abs(v_max - 1.0) > 0.01:
        if debug:
            st.warning(f"⚠️ V(λ) 峰值 {v_max:.4f}，自动归一化到 1.0")
        v_lambda = (v_array / v_max).tolist()
        v_array = np.asarray(v_lambda)
    
    # Nz(λ)：积分归一化到 1.0（CIE S026 标准）
    nz_array = np.asarray(nz_lambda)
    current_integral_nz = trapezoid(nz_array, wavelengths)
    
    if abs(current_integral_nz - 1.0) > 0.05:
        scale_factor = 1.0 / current_integral_nz
        nz_lambda = (nz_array * scale_factor).tolist()
        if debug:
            st.info(f"🔧 Nz(λ) 自动积分归一化: ∫Nz {current_integral_nz:.4f} → 1.0 (因子 {scale_factor:.4f})")
    
    # 检查 Nz(λ) 峰值位置
    nz_array = np.asarray(nz_lambda)
    nz_peak_idx = np.argmax(nz_array)
    nz_peak_wl = wavelengths[nz_peak_idx]
    if debug and (nz_peak_wl < 480 or nz_peak_wl > 500):
        st.warning(f"⚠️ Nz(λ) 峰值在 {nz_peak_wl} nm，标准应在 480-500 nm")
    
    if debug:
        v_array = np.asarray(v_lambda)
        nz_array = np.asarray(nz_lambda)
        integral_v = trapezoid(v_array, wavelengths)
        integral_nz = trapezoid(nz_array, wavelengths)
        st.info(f"📊 V(λ) 峰值: {np.max(v_array):.4f} @ {wavelengths[np.argmax(v_array)]} nm")
        st.info(f"📊 V(λ) 积分值: {integral_v:.4f}")
        st.info(f"📊 Nz(λ) 峰值: {np.max(nz_array):.4f} @ {wavelengths[nz_peak_idx]} nm")
        st.info(f"📊 Nz(λ) 积分值: {integral_nz:.4f}")
    
    return v_lambda, nz_lambda

# ==================== EML 计算 ====================
def calculate_eml_and_medi(spectrum_w_m2_nm, v_lambda, nz_lambda, debug=False, lang="zh"):
    """计算 EML 和 m-EDI"""
    spectrum = np.asarray(spectrum_w_m2_nm)
    wavelengths = np.asarray(STANDARD_WAVELENGTHS)
    
    weighted_melanopic = spectrum * nz_lambda
    weighted_photopic = spectrum * v_lambda
    
    integral_nz = trapezoid(weighted_melanopic, wavelengths)
    integral_v = trapezoid(weighted_photopic, wavelengths)
    
    integral_v_spectrum = trapezoid(v_lambda, wavelengths)
    eml_constant = KM * integral_v_spectrum
    
    eml = eml_constant * integral_nz
    illuminance = KM * integral_v
    medi = eml * 0.9063
    
    if debug:
        if lang == "zh":
            st.write("### 🔍 调试信息")
            st.write(f"标准网格: {wavelengths[0]}-{wavelengths[-1]} nm, 步长 {STANDARD_DELTA} nm")
            st.write("---")
            st.write("**V(λ) 和 Nz(λ) 数据检查（理论值：∫V=106.86，∫Nz=1.0）**")
            st.write(f"∫ V(λ) dλ = {integral_v_spectrum:.4f}")
            st.write(f"∫ Nz(λ) dλ = {trapezoid(nz_lambda, wavelengths):.4f}")
            st.write("---")
            st.write(f"光谱最大值: {np.max(spectrum):.6f} W/m²/nm")
            st.write(f"V(λ) 最大值: {np.max(v_lambda):.4f} @ {wavelengths[np.argmax(v_lambda)]} nm")
            st.write(f"Nz(λ) 最大值: {np.max(nz_lambda):.4f} @ {wavelengths[np.argmax(nz_lambda)]} nm")
            st.write("---")
            st.write(f"∫ E(λ) × V(λ) dλ = {integral_v:.6e}")
            st.write(f"∫ E(λ) × Nz(λ) dλ = {integral_nz:.6e}")
            st.write("---")
            st.write(f"动态 EML_CONSTANT = {eml_constant:.2f}")
            st.write(f"KM = {KM}")
            st.write(f"视觉照度 = {KM} × {integral_v:.6e} = {illuminance:.2f} lx")
            st.write(f"EML = {eml_constant:.2f} × {integral_nz:.6e} = {eml:.2f} lx")
            st.write(f"m-EDI = EML × 0.9063 = {medi:.2f} lx")
            st.write("===================")
        else:
            st.write("### 🔍 Debug Info")
            st.write(f"Standard grid: {wavelengths[0]}-{wavelengths[-1]} nm, step {STANDARD_DELTA} nm")
            st.write("---")
            st.write("**V(λ) and Nz(λ) data check (theoretical: ∫V=106.86, ∫Nz=1.0)**")
            st.write(f"∫ V(λ) dλ = {integral_v_spectrum:.4f}")
            st.write(f"∫ Nz(λ) dλ = {trapezoid(nz_lambda, wavelengths):.4f}")
            st.write("---")
            st.write(f"Max spectrum: {np.max(spectrum):.6f} W/m²/nm")
            st.write(f"V(λ) max: {np.max(v_lambda):.4f} @ {wavelengths[np.argmax(v_lambda)]} nm")
            st.write(f"Nz(λ) max: {np.max(nz_lambda):.4f} @ {wavelengths[np.argmax(nz_lambda)]} nm")
            st.write("---")
            st.write(f"∫ E(λ) × V(λ) dλ = {integral_v:.6e}")
            st.write(f"∫ E(λ) × Nz(λ) dλ = {integral_nz:.6e}")
            st.write("---")
            st.write(f"Dynamic EML_CONSTANT = {eml_constant:.2f}")
            st.write(f"KM = {KM}")
            st.write(f"Illuminance = {KM} × {integral_v:.6e} = {illuminance:.2f} lx")
            st.write(f"EML = {eml_constant:.2f} × {integral_nz:.6e} = {eml:.2f} lx")
            st.write(f"m-EDI = EML × 0.9063 = {medi:.2f} lx")
            st.write("===================")
    
    return eml, medi, illuminance

# ==================== 解析用户光谱 ====================
def parse_user_spectrum(text):
    """解析用户输入的光谱数据"""
    try:
        lines = text.strip().split('\n')
        wavelengths = []
        powers = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('//'):
                continue
            
            parts = re.split(r'[,\s\t]+', line)
            parts = [p for p in parts if p]
            
            if len(parts) >= 2:
                try:
                    wl = float(parts[0])
                    power = float(parts[1])
                    wavelengths.append(wl)
                    powers.append(power)
                except ValueError:
                    continue
        
        if len(wavelengths) < 2:
            return None, None
        
        wavelengths = np.array(wavelengths)
        powers = np.array(powers)
        
        sort_idx = np.argsort(wavelengths)
        wavelengths = wavelengths[sort_idx]
        powers = powers[sort_idx]
        
        return wavelengths, powers
    except Exception:
        return None, None

# ==================== WELL 标准对比 ====================
def get_well_comparison(eml):
    standards = [
        {'level': 'well_excellent', 'name_zh': '高品质推荐', 'name_en': 'High Quality', 'eml_min': 250},
        {'level': 'well_basis_a', 'name_zh': '基础达标 (方案A)', 'name_en': 'Basic (Option A)', 'eml_min': 200},
        {'level': 'well_basis_b', 'name_zh': '基础达标 (方案B)', 'name_en': 'Basic (Option B)', 'eml_min': 150},
    ]
    results = []
    for std in standards:
        results.append({
            'level': std['level'],
            'name_zh': std['name_zh'],
            'name_en': std['name_en'],
            'eml_min': std['eml_min'],
            'eml_met': eml >= std['eml_min']
        })
    return results

# ==================== 管理员认证 ====================
ADMIN_USERNAME = "Laurence_ku"
ADMIN_PASSWORD = "Ku_product$2026"

def check_admin_auth(username, password):
    return username == ADMIN_USERNAME and password == ADMIN_PASSWORD

# ==================== 管理员页面 ====================
@st.dialog("⚙️ 管理员设置", width="large")
def admin_dialog():
    st.markdown("### 🔐 管理员验证")
    
    if "admin_authenticated" not in st.session_state:
        st.session_state.admin_authenticated = False
    
    if not st.session_state.admin_authenticated:
        col1, col2 = st.columns(2)
        with col1:
            username = st.text_input("用户名", key="admin_username")
        with col2:
            password = st.text_input("密码", type="password", key="admin_password")
        
        if st.button("登录", type="primary", use_container_width=True):
            if check_admin_auth(username, password):
                st.session_state.admin_authenticated = True
                st.rerun()
            else:
                st.error("用户名或密码错误")
        return
    
    st.success("✅ 管理员已登录")
    
    # 加载当前数据
    current_v, current_nz = load_spectral_data()
    
    # 创建初始 DataFrame
    df_initial = pd.DataFrame({
        '波长 (nm)': STANDARD_WAVELENGTHS,
        'V(λ) 明视觉': current_v,
        'Nz(λ) 黑视素': current_nz
    })
    
    # 使用 session_state 持久化编辑数据
    if "admin_df" not in st.session_state:
        st.session_state.admin_df = df_initial.copy()
    
    # ========== 实时图表预览 ==========
    st.subheader("📈 光谱曲线预览（实时更新）")
    chart_placeholder = st.empty()
    
    def update_chart(dataframe):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dataframe['波长 (nm)'], y=dataframe['V(λ) 明视觉'],
            mode='lines', name='V(λ) 明视觉',
            line=dict(color='red', width=2, dash='dot')
        ))
        fig.add_trace(go.Scatter(
            x=dataframe['波长 (nm)'], y=dataframe['Nz(λ) 黑视素'],
            mode='lines', name='Nz(λ) 黑视素',
            line=dict(color='blue', width=2)
        ))
        fig.update_layout(
            title="明视觉光谱 (Vλ) vs 黑视素光谱 (Nz)",
            xaxis_title="波长 (nm)",
            yaxis_title="相对灵敏度",
            template="plotly_white",
            height=400
        )
        return fig
    
    # ========== 粘贴数据区域 ==========
    st.subheader("📋 快速粘贴数据")
    st.caption("💡 从 Excel 复制整列数据，粘贴到下方文本框，点击按钮即可替换")
    
    col_paste1, col_paste2 = st.columns(2)
    
    with col_paste1:
        st.markdown("**V(λ) 明视觉数据**")
        v_paste = st.text_area("", height=150, key="paste_v", 
                               placeholder="从 Excel 复制一列 81 个数字，粘贴到这里")
        if st.button("📋 替换 V(λ) 列", key="apply_v", use_container_width=True):
            try:
                text = v_paste.strip()
                if ',' in text:
                    values = [float(x.strip()) for x in text.split(',') if x.strip()]
                elif ' ' in text and '\n' not in text:
                    values = [float(x.strip()) for x in text.split() if x.strip()]
                else:
                    values = [float(x.strip()) for x in text.split('\n') if x.strip()]
                
                if len(values) == len(STANDARD_WAVELENGTHS):
                    st.session_state.admin_df['V(λ) 明视觉'] = values
                    st.success(f"✅ 已更新 {len(values)} 个数据")
                    st.rerun()
                else:
                    st.error(f"数据点数错误：需要 {len(STANDARD_WAVELENGTHS)} 个，实际 {len(values)} 个")
            except Exception as e:
                st.error(f"解析失败: {e}")
    
    with col_paste2:
        st.markdown("**Nz(λ) 黑视素数据**")
        nz_paste = st.text_area("", height=150, key="paste_nz",
                               placeholder="从 Excel 复制一列 81 个数字，粘贴到这里")
        if st.button("📋 替换 Nz(λ) 列", key="apply_nz", use_container_width=True):
            try:
                text = nz_paste.strip()
                if ',' in text:
                    values = [float(x.strip()) for x in text.split(',') if x.strip()]
                elif ' ' in text and '\n' not in text:
                    values = [float(x.strip()) for x in text.split() if x.strip()]
                else:
                    values = [float(x.strip()) for x in text.split('\n') if x.strip()]
                
                if len(values) == len(STANDARD_WAVELENGTHS):
                    st.session_state.admin_df['Nz(λ) 黑视素'] = values
                    st.success(f"✅ 已更新 {len(values)} 个数据")
                    st.rerun()
                else:
                    st.error(f"数据点数错误：需要 {len(STANDARD_WAVELENGTHS)} 个，实际 {len(values)} 个")
            except Exception as e:
                st.error(f"解析失败: {e}")
    
    st.markdown("---")
    st.caption("💡 提示：也可以直接在下方表格中编辑")
    
    # ========== 数据编辑表格 ==========
    st.subheader("📊 光谱数据编辑")
    
    column_config = {
        "波长 (nm)": st.column_config.NumberColumn(format="%d"),
        "V(λ) 明视觉": st.column_config.NumberColumn(format="%.8f"),
        "Nz(λ) 黑视素": st.column_config.NumberColumn(format="%.8f"),
    }
    
    edited_df = st.data_editor(
        st.session_state.admin_df,
        column_config=column_config,
        use_container_width=True,
        height=500,
        num_rows="fixed",
        key="admin_data_editor"
    )
    
    # 更新 session_state
    st.session_state.admin_df = edited_df
    
    # 更新图表
    chart_placeholder.plotly_chart(update_chart(edited_df), use_container_width=True)
    
    # ========== 批量操作 ==========
    st.subheader("📁 批量操作")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_data = edited_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 下载当前数据 (CSV)",
            data=csv_data,
            file_name="spectral_data.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        uploaded_file = st.file_uploader(
            "📤 上传 CSV/Excel",
            type=["csv", "xlsx"],
            key="admin_upload",
            help="文件需包含列：波长(nm), V(λ) 明视觉, Nz(λ) 黑视素"
        )
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    upload_df = pd.read_csv(uploaded_file)
                else:
                    upload_df = pd.read_excel(uploaded_file)
                
                wavelength_col = None
                v_col = None
                nz_col = None
                for col in upload_df.columns:
                    col_lower = col.lower()
                    if '波长' in col_lower or 'nm' in col_lower or 'wavelength' in col_lower:
                        wavelength_col = col
                    elif 'v' in col_lower and ('λ' in col_lower or 'lambda' in col_lower):
                        v_col = col
                    elif 'nz' in col_lower or '黑视' in col_lower or 'melanopic' in col_lower:
                        nz_col = col
                
                if wavelength_col and v_col and nz_col:
                    wl_upload = upload_df[wavelength_col].values
                    v_upload = upload_df[v_col].values
                    nz_upload = upload_df[nz_col].values
                    
                    v_interp = interpolate_linear(wl_upload, v_upload, STANDARD_WAVELENGTHS)
                    nz_interp = interpolate_linear(wl_upload, nz_upload, STANDARD_WAVELENGTHS)
                    
                    st.session_state.admin_df['V(λ) 明视觉'] = v_interp
                    st.session_state.admin_df['Nz(λ) 黑视素'] = nz_interp
                    st.success(f"✅ 已加载并插值到 5nm 网格")
                    st.rerun()
                else:
                    st.error("无法识别列名，请确保包含：波长(nm), V(λ) 明视觉, Nz(λ) 黑视素")
            except Exception as e:
                st.error(f"解析失败: {e}")
    
    with col3:
        if st.button("🔄 重置为预设数据", use_container_width=True):
            if reset_to_default():
                st.success("已重置为 CIE S026 标准数据")
                st.session_state.admin_df = pd.DataFrame({
                    '波长 (nm)': STANDARD_WAVELENGTHS,
                    'V(λ) 明视觉': DEFAULT_V_LAMBDA,
                    'Nz(λ) 黑视素': DEFAULT_NZ_LAMBDA
                })
                st.rerun()
            else:
                st.error("重置失败")
    
    # ========== 保存按钮 ==========
    st.markdown("---")
    st.caption("💡 点击保存后数据写入全局配置表，然后手动关闭对话框")
    
    col_save, col_close = st.columns(2)
    
    with col_save:
        if st.button("💾 保存到系统", type="primary", use_container_width=True):
            v_lambda_save = st.session_state.admin_df['V(λ) 明视觉'].tolist()
            nz_lambda_save = st.session_state.admin_df['Nz(λ) 黑视素'].tolist()
            
            if save_spectral_data(v_lambda_save, nz_lambda_save):
                st.success("✅ 数据已保存到全局配置表！")
                st.balloons()
            else:
                st.error("❌ 保存失败，请检查日志")
    
    with col_close:
        if st.button("❌ 关闭", use_container_width=True):
            if "admin_df" in st.session_state:
                del st.session_state.admin_df
            st.session_state.admin_authenticated = False
            st.rerun()

# ==================== 多语言文本 ====================
TEXTS = {
    'zh': {
        'page_title': '健康光计算器 (EML / m-EDI)',
        'page_subtitle': '基于 CIE S026 / WELL 标准 — 计算等值黑视素照度 (EML) 和黑视素等效日光照度 (m-EDI)',
        'debug_mode': '🐛 调试模式',
        'about_system': 'ℹ️ 关于系统',
        'about_text': '''
### 什么是 EML 和 m-EDI？

- **EML (Equivalent Melanopic Lux)**：衡量光源对非视觉感光细胞 (ipRGC) 的刺激强度。
- **m-EDI (melanopic Equivalent Daylight Illuminance)**：衡量当前光源在节律效应上相当于多少勒克斯的日光 (D65)。

### 计算公式

$$EML = K_m \\int E(\\lambda) \\cdot N_z(\\lambda) \\, d\\lambda \\cdot \\frac{\\int V(\\lambda) \\, d\\lambda}{\\int N_z(\\lambda) \\, d\\lambda}$$

$$m\\text{-}EDI \\approx EML \\times 0.9063$$

其中：
- $K_m = 683.002$ lm/W（明视觉最大光谱光视效能）
- $V(\\lambda)$：明视觉光谱光视效能函数（峰值 555nm）
- $N_z(\\lambda)$：黑视素光谱光视效能函数（峰值 490nm）

### 健康基准

- **日间 (办公/学校)**：EML ≥ 250
- **夜间 (家居/睡眠)**：EML ≤ 50
        ''',
        'contact': '📞 联系：✉️ 电邮: Techlife2027@gmail.com',
        'loaded': '✅ 系统已加载标准光谱响应函数 (CIE S026:2018)',
        'input_title': '1️⃣ 输入光源光谱功率分布 (SPD)',
        'upload_label': '选项 A: 上传 CSV/TXT 文件',
        'upload_help': '文件应包含两列: 波长(nm), 功率(W/m²/nm)。支持任意步长',
        'textarea_label': '选项 B: 粘贴或输入光谱数据',
        'textarea_placeholder': '波长(nm),功率(W/m²/nm)\\n380 0.0012\\n385 0.0021',
        'unit_note': '💡 单位说明：功率单位为 W/m²/nm（瓦每平方米每纳米）',
        'enable_scaling': '🎯 启用自动缩放',
        'target_illuminance': '目标照度 (lx)',
        'target_illuminance_help': '输入期望的照度值，系统将自动缩放光谱',
        'calc_btn': '🚀 计算 EML / m-EDI',
        'result_title': '📊 计算结果',
        'eml_label': '等值黑视素照度 (EML)',
        'medi_label': '黑视素等效日光照度 (m-EDI)',
        'lux_label': '视觉照度 (Illuminance)',
        'medi_delta': '≈ EML x 0.9063',
        'well_comparison_title': '📋 与 WELL 标准对比',
        'well_table_header': 'WELL 等级',
        'well_eml_requirement': 'EML 要求',
        'well_status': '当前状态',
        'well_excellent': '高品质推荐',
        'well_basis_a': '基础达标 (方案A)',
        'well_basis_b': '基础达标 (方案B)',
        'well_meet': '✅ 达标',
        'well_not_meet': '❌ 未达标',
        'rating_excellent': '⭐ 日间健康评级：优秀 (EML ≥ 250)',
        'rating_good': '🌤️ 日间健康评级：基础达标 (EML ≥ 150)',
        'rating_night': '🌙 夜间模式识别 (EML ≤ 50)',
        'rating_moderate': '⚠️ 节律刺激中等',
        'vis_title': '📈 光谱可视化',
        'vis_original': '原始数据 (步长{:.1f}nm)',
        'vis_interp': '插值后光谱 (5nm步长)',
        'vis_vlambda': '明视觉光谱 V(λ)',
        'vis_weighted': '有效节律光谱 (SPD × Nz)',
        'data_note_title': '🔧 数据处理说明',
        'data_note_content': '原始数据: {} 个数据点，波长范围 {:.0f} - {:.0f} nm，平均步长 {:.2f} nm',
        'export_btn': '📥 导出 Word 报告 (.doc)',
        'warning_input': '请先上传文件或输入数据。',
        'error_parse': '光谱数据解析失败，请检查格式。',
        'error_no_overlap': '错误：输入的光谱波长范围与标准范围 (380-780nm) 没有重叠。',
        'footer': '⚠️ 免责声明: 本工具基于 CIE S026:2018 标准计算。',
        'detected': '📊 检测到输入数据: 波长范围 {:.0f} - {:.0f} nm，平均步长 {:.2f} nm，数据点数量: {}',
        'name_placeholder': '请输入姓名',
        'title_placeholder': '请输入头衔（可选）',
        'report_title': '健康照明 EML/m-EDI 分析报告',
        'report_date': '报告日期',
        'report_analyst': '分析人',
        'analyst_name': '分析人姓名',
        'analyst_title': '分析人头衔（可选）',
        'chart_title': '光谱功率分布 (SPD)',
        'chart_xlabel': '波长 (nm)',
        'chart_ylabel': '功率 (W/m²/nm)'
    },
    'en': {
        'page_title': 'Healthy Lighting Calculator (EML / m-EDI)',
        'page_subtitle': 'Based on CIE S026 / WELL Standard — Calculate Equivalent Melanopic Lux (EML) and melanopic Equivalent Daylight Illuminance (m-EDI)',
        'debug_mode': '🐛 Debug Mode',
        'about_system': 'ℹ️ About System',
        'about_text': '''
### What are EML and m-EDI?

- **EML (Equivalent Melanopic Lux)**: Measures the stimulation intensity of light on ipRGC cells.
- **m-EDI (melanopic Equivalent Daylight Illuminance)**: Compares circadian effect to standard D65 daylight.

### Calculation Formulas

$$EML = K_m \\int E(\\lambda) \\cdot N_z(\\lambda) \\, d\\lambda \\cdot \\frac{\\int V(\\lambda) \\, d\\lambda}{\\int N_z(\\lambda) \\, d\\lambda}$$

$$m\\text{-}EDI \\approx EML \\times 0.9063$$

Where:
- $K_m = 683.002$ lm/W (maximum photopic luminous efficacy)
- $V(\\lambda)$: photopic spectral efficiency function (peak at 555nm)
- $N_z(\\lambda)$: melanopic spectral efficiency function (peak at 490nm)

### Health Benchmarks

- **Daytime (Office/School)**: EML ≥ 250
- **Nighttime (Home/Sleep)**: EML ≤ 50
        ''',
        'contact': '📞 Contact: ✉️ Email: Techlife2027@gmail.com',
        'loaded': '✅ Standard spectral response functions loaded (CIE S026:2018)',
        'input_title': '1️⃣ Input Light Source Spectral Power Distribution (SPD)',
        'upload_label': 'Option A: Upload CSV/TXT File',
        'upload_help': 'File should contain two columns: Wavelength(nm), Power(W/m²/nm)',
        'textarea_label': 'Option B: Paste or Enter Spectral Data',
        'textarea_placeholder': 'Wavelength(nm),Power(W/m²/nm)\\n380 0.0012\\n385 0.0021',
        'unit_note': '💡 Unit Note: Power unit is W/m²/nm',
        'enable_scaling': '🎯 Enable Auto Scaling',
        'target_illuminance': 'Target Illuminance (lx)',
        'target_illuminance_help': 'Enter desired illuminance, spectrum will be scaled automatically',
        'calc_btn': '🚀 Calculate EML / m-EDI',
        'result_title': '📊 Results',
        'eml_label': 'Equivalent Melanopic Lux (EML)',
        'medi_label': 'melanopic Equivalent Daylight Illuminance (m-EDI)',
        'lux_label': 'Illuminance',
        'medi_delta': '≈ EML x 0.9063',
        'well_comparison_title': '📋 WELL Standard Comparison',
        'well_table_header': 'WELL Level',
        'well_eml_requirement': 'EML Requirement',
        'well_status': 'Status',
        'well_excellent': 'High Quality',
        'well_basis_a': 'Basic (Option A)',
        'well_basis_b': 'Basic (Option B)',
        'well_meet': '✅ Meet',
        'well_not_meet': '❌ Not Meet',
        'rating_excellent': '⭐ Daytime Rating: Excellent (EML ≥ 250)',
        'rating_good': '🌤️ Daytime Rating: Basic Compliance (EML ≥ 150)',
        'rating_night': '🌙 Nighttime Mode Detected (EML ≤ 50)',
        'rating_moderate': '⚠️ Moderate Circadian Stimulation',
        'vis_title': '📈 Spectral Visualization',
        'vis_original': 'Original Data ({:.1f}nm step)',
        'vis_interp': 'Interpolated Spectrum (5nm step)',
        'vis_vlambda': 'Photopic Spectrum V(λ)',
        'vis_weighted': 'Effective Circadian Spectrum (SPD × Nz)',
        'data_note_title': '🔧 Data Processing Notes',
        'data_note_content': 'Original Data: {} points, wavelength range {:.0f} - {:.0f} nm, average step {:.2f} nm',
        'export_btn': '📥 Export Word Report (.doc)',
        'warning_input': 'Please upload a file or enter data first.',
        'error_parse': 'Failed to parse spectral data. Please check format.',
        'error_no_overlap': 'Error: Input wavelength range has no overlap with standard range (380-780nm).',
        'footer': '⚠️ Disclaimer: This tool is based on CIE S026:2018 standard.',
        'detected': '📊 Detected input: wavelength range {:.0f} - {:.0f} nm, average step {:.2f} nm, data points: {}',
        'name_placeholder': 'Enter your name',
        'title_placeholder': 'Enter your title (optional)',
        'report_title': 'Healthy Lighting EML/m-EDI Analysis Report',
        'report_date': 'Report Date',
        'report_analyst': 'Analyst',
        'analyst_name': 'Analyst Name',
        'analyst_title': 'Analyst Title (Optional)',
        'chart_title': 'Spectral Power Distribution (SPD)',
        'chart_xlabel': 'Wavelength (nm)',
        'chart_ylabel': 'Power (W/m²/nm)'
    }
}

# ==================== Word 报告生成 ====================
def generate_word_report(t, analyst_name, analyst_title, eml, medi, lux, 
                         well_results, fig_html, input_min, input_max, step, num_points,
                         scale_factor=None, original_illuminance=None, target_illuminance=None):
    
    if eml >= 250:
        rating_text = "⭐ 日间使用推荐" if t['page_title'].startswith('健康') else "⭐ Recommended for daytime"
    elif eml >= 150:
        rating_text = "🌤️ 满足 WELL 基础要求" if t['page_title'].startswith('健康') else "🌤️ Meets WELL requirements"
    elif eml <= 50:
        rating_text = "🌙 适合睡前照明" if t['page_title'].startswith('健康') else "🌙 Suitable for pre-sleep"
    else:
        rating_text = "⚠️ 需根据使用时间评估" if t['page_title'].startswith('健康') else "⚠️ Evaluate based on usage"
    
    well_rows = ""
    for r in well_results:
        level_text = r['name_zh'] if t['page_title'].startswith('健康') else r['name_en']
        status_icon = "✅" if r['eml_met'] else "❌"
        status_text = t['well_meet'] if r['eml_met'] else t['well_not_meet']
        well_rows += f"""
        <tr>
            <td style="padding: 8px 12px; border: 1px solid #aaa;">{level_text}</td>
            <td style="padding: 8px 12px; border: 1px solid #aaa; text-align: center;">≥ {r['eml_min']} lx</td>
            <td style="padding: 8px 12px; border: 1px solid #aaa; text-align: center;">{status_icon} {status_text}</td>
        </tr>
        """
    
    analyst_info = analyst_name if analyst_name else ("未填写" if t['page_title'].startswith('健康') else "Not filled")
    if analyst_title:
        analyst_info += f" ({analyst_title})"
    
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    scale_note = ""
    if scale_factor is not None and abs(scale_factor - 1.0) > 0.01:
        scale_note = f'<p style="font-size: 10pt; color: #666;">💡 光谱已缩放: {original_illuminance:.1f} lx → {target_illuminance:.1f} lx (因子 {scale_factor:.6f})</p>'
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{t['report_title']}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Arial', '宋体', sans-serif;
            margin: 1.5cm;
            padding: 0;
            font-size: 11pt;
            line-height: 1.4;
            color: #000000;
        }}
        h1 {{ font-size: 18pt; font-weight: bold; margin: 20pt 0 10pt 0; color: #1e3a5f; border-bottom: 2px solid #4f46e5; padding-bottom: 8px; }}
        h2 {{ font-size: 14pt; font-weight: bold; margin: 15pt 0 8pt 0; color: #334155; border-left: 3px solid #4f46e5; padding-left: 10px; }}
        .header-info {{ background-color: #f0f0f0; padding: 10px 15px; margin: 15px 0; border: 1px solid #ccc; }}
        .metrics {{ margin: 15px 0; display: flex; flex-wrap: wrap; gap: 15px; }}
        .metric-item {{ flex: 1; min-width: 140px; padding: 10px 12px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 8px; }}
        .metric-item.illuminance {{ background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); }}
        .metric-value {{ font-size: 16pt; font-weight: bold; color: #000000 !important; }}
        .metric-label {{ font-size: 10pt; color: #333333; margin-top: 4px; }}
        .rating-badge {{ display: inline-block; padding: 5px 12px; margin: 10px 0; background-color: #22c55e; color: #000000; border-radius: 16px; font-weight: bold; font-size: 10pt; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th {{ background-color: #4f46e5; color: white; padding: 8px 10px; border: 1px solid #aaa; font-weight: bold; }}
        td {{ padding: 6px 10px; border: 1px solid #aaa; color: #000000; }}
        .footer {{ margin-top: 30px; padding-top: 12px; border-top: 1px solid #ccc; font-size: 9pt; color: #666; text-align: center; }}
        .spectrum-container {{ margin: 20px 0; text-align: center; }}
        .data-note {{ background-color: #f8fafc; padding: 10px 15px; margin: 15px 0; border: 1px solid #ccc; font-size: 10pt; }}
    </style>
</head>
<body>

<h1>{t['report_title']}</h1>

<div class="header-info">
    <p style="margin: 3px 0;"><strong>{t['report_date']}:</strong> {current_date}</p>
    <p style="margin: 3px 0;"><strong>{t['report_analyst']}:</strong> {analyst_info}</p>
    {scale_note}
</div>

<h2>{t['result_title']}</h2>
<div class="metrics">
    <div class="metric-item"><div class="metric-value">{eml:.1f} lx</div><div class="metric-label">{t['eml_label']}</div></div>
    <div class="metric-item"><div class="metric-value">{medi:.1f} lx</div><div class="metric-label">{t['medi_label']}</div></div>
    <div class="metric-item illuminance"><div class="metric-value">{lux:.1f} lx</div><div class="metric-label">{t['lux_label']}</div></div>
</div>

<div class="rating-badge">{rating_text}</div>

<h2>{t['well_comparison_title']}</h2>
<table>
    <thead><tr><th>{t['well_table_header']}</th><th>{t['well_eml_requirement']}</th><th>{t['well_status']}</th></tr></thead>
    <tbody>{well_rows}</tbody>
</table>

<h2>{t['vis_title']}</h2>
<div class="spectrum-container">{fig_html}</div>

<h2>{t['data_note_title']}</h2>
<div class="data-note">
    {t['data_note_content'].format(num_points, input_min, input_max, step)}<br>
    标准网格: 380-780 nm，固定步长 5 nm（共 81 个点）<br>
    插值方法: 能量守恒插值（用户光谱），线性插值（灵敏度函数）
</div>

<div class="footer">{t['footer']}</div>

</body>
</html>
"""
    return html_content.encode('utf-8')

# ==================== 主应用 ====================
def main():
    lang = st.session_state.lang
    t = TEXTS[lang]
    
    # 顶部按钮
    col_title, col_lang1, col_lang2, col_admin, col_debug = st.columns([5, 0.8, 0.8, 0.8, 0.8])
    
    with col_title:
        st.title("💡 " + t['page_title'])
    
    with col_lang1:
        if st.button("中文", key="lang_zh", use_container_width=True):
            st.session_state.lang = "zh"
            st.rerun()
    
    with col_lang2:
        if st.button("English", key="lang_en", use_container_width=True):
            st.session_state.lang = "en"
            st.rerun()
    
    with col_admin:
        if st.button("⚙️", key="admin_gear", help="管理员设置", use_container_width=True):
            admin_dialog()
    
    with col_debug:
        if st.button("🐛", key="debug_btn", help=t['debug_mode'], use_container_width=True):
            st.session_state.debug_mode = not st.session_state.debug_mode
            st.rerun()
    
    st.markdown(t['page_subtitle'])
    
    # 侧边栏
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.username}")
        
        remaining = get_user_remaining_trials(st.session_state.user_id)
        if remaining == -1:
            st.info(f"🎫 剩余免费次数: ∞ (专业版)" if lang == "zh" else f"🎫 Remaining Trials: ∞ (Pro)")
        else:
            st.info(f"🎫 剩余免费次数: {remaining}" if lang == "zh" else f"🎫 Remaining Trials: {remaining}")
        
        st.markdown("---")
        st.header(t['about_system'])
        st.markdown(t['about_text'])
        st.markdown("---")
        st.markdown(t['contact'])
        st.markdown("---")
        
        analyst_name = st.text_input(t['analyst_name'], placeholder=t['name_placeholder'], key="analyst_name")
        analyst_title = st.text_input(t['analyst_title'], placeholder=t['title_placeholder'], key="analyst_title")
        
        st.markdown("---")
        st.caption("📊 当前使用的光谱数据")
        v_lambda, nz_lambda = get_normalized_spectral_data(debug=False)
        v_peak_idx = np.argmax(v_lambda)
        nz_peak_idx = np.argmax(nz_lambda)
        st.caption(f"标准网格: {STANDARD_WAVELENGTHS[0]}-{STANDARD_WAVELENGTHS[-1]} nm, 步长 {STANDARD_DELTA} nm")
        st.caption(f"V(λ) 峰值: {max(v_lambda):.4f} @ {STANDARD_WAVELENGTHS[v_peak_idx]} nm")
        st.caption(f"Nz(λ) 峰值: {max(nz_lambda):.4f} @ {STANDARD_WAVELENGTHS[nz_peak_idx]} nm")
    
    # 主区域
    st.subheader(t['input_title'])
    
    uploaded_file = st.file_uploader(t['upload_label'], type=["csv", "txt"], help=t['upload_help'])
    spectrum_text = st.text_area(t['textarea_label'], height=150, placeholder=t['textarea_placeholder'])
    st.caption(t['unit_note'])
    
    # 自动缩放选项
    col_scale1, col_scale2 = st.columns([1, 2])
    with col_scale1:
        enable_scaling = st.checkbox(t['enable_scaling'], value=False, key="enable_scaling")
    
    target_illuminance = 100.0
    if enable_scaling:
        with col_scale2:
            target_illuminance = st.number_input(
                t['target_illuminance'],
                min_value=1.0,
                max_value=100000.0,
                value=100.0,
                step=10.0,
                key="target_illuminance_input",
                help=t['target_illuminance_help']
            )
    
    if st.button(t['calc_btn'], type="primary", use_container_width=True):
        allowed, new_remaining, error_msg = consume_trial(st.session_state.user_id, "eml_calculator")
        if not allowed:
            st.error(error_msg)
        else:
            wl_input, power_input = None, None
            
            if uploaded_file is not None:
                text_data = uploaded_file.getvalue().decode("utf-8")
                wl_input, power_input = parse_user_spectrum(text_data)
            elif spectrum_text:
                wl_input, power_input = parse_user_spectrum(spectrum_text)
            else:
                st.warning(t['warning_input'])
            
            if wl_input is not None and power_input is not None and len(wl_input) >= 2:
                step_in = np.mean(np.diff(wl_input))
                input_min, input_max = wl_input[0], wl_input[-1]
                st.info(t['detected'].format(input_min, input_max, step_in, len(wl_input)))
                
                v_lambda, nz_lambda = get_normalized_spectral_data(debug=False)
                
                # 插值用户光谱到标准网格
                interp_spectrum = interpolate_energy_conserving(wl_input, power_input, STANDARD_WAVELENGTHS)
                
                scaled_spectrum = interp_spectrum
                scale_factor = 1.0
                current_illuminance = None
                
                if enable_scaling:
                    # 计算原始照度
                    current_illuminance = KM * trapezoid(interp_spectrum * np.asarray(v_lambda), STANDARD_WAVELENGTHS)
                    
                    # 缩放光谱到目标照度
                    if current_illuminance > 0:
                        scale_factor = target_illuminance / current_illuminance
                        scaled_spectrum = interp_spectrum * scale_factor
                    
                    # 显示缩放信息
                    if abs(scale_factor - 1.0) > 0.01:
                        st.info(f"🔧 光谱已缩放: {current_illuminance:.1f} lx → {target_illuminance:.1f} lx (因子 {scale_factor:.6f})")
                
                debug_mode = st.session_state.get("debug_mode", False)
                eml, medi, lux = calculate_eml_and_medi(
                    scaled_spectrum, v_lambda, nz_lambda, debug=debug_mode, lang=lang
                )
                well_results = get_well_comparison(eml)
                
                st.subheader(t['result_title'])
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(t['eml_label'], f"{eml:.1f} lx")
                with col2:
                    st.metric(t['medi_label'], f"{medi:.1f} lx", delta=t['medi_delta'])
                with col3:
                    st.metric(t['lux_label'], f"{lux:.1f} lx")
                
                if eml >= 250:
                    st.success(t['rating_excellent'])
                elif eml >= 150:
                    st.info(t['rating_good'])
                elif eml <= 50:
                    st.info(t['rating_night'])
                else:
                    st.warning(t['rating_moderate'])
                
                st.subheader(t['well_comparison_title'])
                well_df = pd.DataFrame([
                    {t['well_table_header']: (r['name_zh'] if lang == 'zh' else r['name_en']),
                     t['well_eml_requirement']: f"≥ {r['eml_min']} lx",
                     t['well_status']: "✅ " + t['well_meet'] if r['eml_met'] else "❌ " + t['well_not_meet']}
                    for r in well_results
                ])
                st.table(well_df)
                
                if eml < 150:
                    st.warning("⚠️ 当前 EML 值低于 WELL 基础达标要求 (≥150 lx)")
                elif eml >= 250:
                    st.success("🎉 恭喜！当前光源已达到 WELL 高品质推荐标准！")
                
                # ========== 光谱可视化（双 Y 轴版本 - 只改这里）==========
                st.subheader(t['vis_title'])
                
                from plotly.subplots import make_subplots
                
                # 创建双 Y 轴子图
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # 原始数据（左轴）
                fig.add_trace(
                    go.Scatter(
                        x=wl_input, y=power_input, 
                        mode='markers', 
                        name=t['vis_original'].format(step_in),
                        marker=dict(color='orange', size=6, symbol='circle')
                    ),
                    secondary_y=False
                )
                
                # 插值后光谱（左轴）
                fig.add_trace(
                    go.Scatter(
                        x=STANDARD_WAVELENGTHS, y=scaled_spectrum, 
                        mode='lines', 
                        name=t['vis_interp'],
                        line=dict(color='darkblue', width=2)
                    ),
                    secondary_y=False
                )
                
                # 有效节律光谱 SPD × Nz(λ)（左轴）
                nz_weighted = scaled_spectrum * nz_lambda
                if max(nz_weighted) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=STANDARD_WAVELENGTHS, y=nz_weighted, 
                            mode='lines', 
                            name=t['vis_weighted'],
                            line=dict(color='green', dash='dash', width=2)
                        ),
                        secondary_y=False
                    )
                
                # 明视觉光谱 V(λ) - 使用右 Y 轴（归一化 0-1）
                v_max_val = max(v_lambda)
                if v_max_val > 0:
                    v_normalized = np.array(v_lambda) / v_max_val
                    fig.add_trace(
                        go.Scatter(
                            x=STANDARD_WAVELENGTHS, y=v_normalized, 
                            mode='lines', 
                            name=t['vis_vlambda'],
                            line=dict(color='red', dash='dot', width=2)
                        ),
                        secondary_y=True
                    )
                
                # 设置坐标轴
                fig.update_xaxes(title_text=t['chart_xlabel'])
                fig.update_yaxes(title_text="功率 / 节律响应 (W/m²/nm)", secondary_y=False)
                fig.update_yaxes(title_text="明视觉灵敏度 V(λ) (归一化)", secondary_y=True, range=[0, 1.1])
                
                # 设置布局
                fig.update_layout(
                    title=t['chart_title'],
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                    template="plotly_white",
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                # ========== 光谱可视化结束 ==========
                
                with st.expander(t['data_note_title']):
                    st.markdown(t['data_note_content'].format(len(wl_input), input_min, input_max, step_in))
                    st.markdown(f"标准网格: {STANDARD_WAVELENGTHS[0]}-{STANDARD_WAVELENGTHS[-1]} nm，固定步长 {STANDARD_DELTA} nm（共 {len(STANDARD_WAVELENGTHS)} 个点）")
                    st.markdown("插值方法: 能量守恒插值（用户光谱），线性插值（灵敏度函数）")
                    if enable_scaling and abs(scale_factor - 1.0) > 0.01:
                        st.markdown(f"光谱缩放: {current_illuminance:.1f} lx → {target_illuminance:.1f} lx (因子 {scale_factor:.6f})")
                
                fig_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn', config={'displayModeBar': False})
                report_data = generate_word_report(
                    t, analyst_name, analyst_title, eml, medi, lux, 
                    well_results, fig_html, input_min, input_max, step_in, len(wl_input),
                    scale_factor if enable_scaling and abs(scale_factor - 1.0) > 0.01 else None,
                    current_illuminance if enable_scaling and abs(scale_factor - 1.0) > 0.01 else None,
                    target_illuminance if enable_scaling and abs(scale_factor - 1.0) > 0.01 else None
                )
                st.download_button(
                    label=t['export_btn'],
                    data=report_data,
                    file_name=f"EML_Report_{datetime.now().strftime('%Y%m%d')}.doc",
                    mime="application/msword",
                    use_container_width=True
                )
            else:
                st.error(t['error_parse'])
    
    st.divider()
    st.caption(t['footer'])

if __name__ == "__main__":
    main()
