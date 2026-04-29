import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import StringIO
import re

# ==================== 自定义梯形积分函数（兼容所有 NumPy 版本）====================
def trapezoid(y, x):
    """
    手动实现梯形积分，避免 NumPy 版本兼容性问题
    使用公式: ∫ y dx ≈ Σ (x[i+1] - x[i]) * (y[i] + y[i+1]) / 2
    """
    y = np.asarray(y)
    x = np.asarray(x)
    
    if len(y) != len(x):
        raise ValueError("y and x must have the same length")
    
    if len(y) < 2:
        return 0.0
    
    # 计算梯形积分
    dx = np.diff(x)
    if np.any(dx <= 0):
        # 如果 x 不是严格递增，先排序
        idx = np.argsort(x)
        x = x[idx]
        y = y[idx]
        dx = np.diff(x)
    
    # 梯形公式: (y[i] + y[i+1]) / 2 * dx
    integral = np.sum((y[:-1] + y[1:]) / 2 * dx)
    return integral


# ==================== 多语言文本配置 ====================

LANGUAGES = {
    'zh': {
        'page_title': '健康光计算器 (EML / m-EDI)',
        'page_subtitle': '基于 CIE S026 / WELL 标准 — 计算等值黑视素照度 (EML) 和黑视素等效日光照度 (m-EDI)',
        
        'theory_title': '📖 背景知识与计算公式',
        'theory_content': '''
### 什么是 EML 和 m-EDI？

- **EML (Equivalent Melanopic Lux)**：衡量光源对非视觉感光细胞 (ipRGC) 的刺激强度。
- **m-EDI (melanopic Equivalent Daylight Illuminance)**：衡量当前光源在节律效应上相当于多少勒克斯的日光 (D65)。

### 计算公式

等值黑视素照度 (EML) 的计算公式如下：

$$EML = K_m \\int E_{e,\\lambda}(\\lambda) \\cdot N_z(\\lambda) \\, d\\lambda \\cdot \\frac{\\int V(\\lambda) \\, d\\lambda}{\\int N_z(\\lambda) \\, d\\lambda}$$

其中：
- $E_{e,\\lambda}(\\lambda)$：光源的光谱功率分布 (W/m²/nm)
- $N_z(\\lambda)$：黑视素光谱光视效能函数（峰值 480nm）
- $V(\\lambda)$：明视觉光谱光视效能函数
- $K_m = 683.002$：明视觉最大光谱光视效能 (lm/W)

**简化公式**：由于 $\\int V(\\lambda) \\, d\\lambda = 106.857$，且 $\\int N_z(\\lambda) \\, d\\lambda$ 被定义为 1，因此：

$$EML = 72983.25 \\times \\int E_{e,\\lambda}(\\lambda) \\cdot N_z(\\lambda) \\, d\\lambda$$

$$m\\text{-}EDI \\approx EML \\times 0.9063$$

### 健康基准

- **日间 (办公/学校)**：EML ≥ 250（有助于保持清醒，提高工作效率）
- **夜间 (家居/睡眠)**：EML ≤ 50（有助于褪黑素分泌，助眠）
        ''',
        
        'about_system': 'ℹ️ 关于系统',
        'analyst_name': '分析人姓名',
        'analyst_title': '分析人头衔（可选）',
        'contact': '📞 联系：✉️ 电邮: Techlife2027@gmail.com',
        
        'loaded': '✅ 系统已加载标准光谱响应函数: V(λ) 和 Nz(λ) (波长范围: 380-780nm, 步长: 5nm)',
        
        'input_title': '1️⃣ 输入光源光谱功率分布 (SPD)',
        'upload_label': '选项 A: 上传 CSV/TXT 文件',
        'upload_help': '文件应包含两列: 波长(nm), 功率(W/m²/nm)。支持任意步长 (1nm,5nm,10nm等)',
        'textarea_label': '选项 B: 粘贴或输入光谱数据',
        'textarea_placeholder': '波长(nm),功率(W/m²/nm)\n380 0.0012\n385 0.0021\n390 0.0035',
        'unit_note': '💡 单位说明：功率单位为 **W/m²/nm** (瓦每平方米每纳米)，这是 EML 计算的标准单位',
        'calc_btn': '🚀 计算 EML / m-EDI',
        
        'result_title': '📊 计算结果',
        'eml_label': '等值黑视素照度 (EML)',
        'medi_label': '黑视素等效日光照度 (m-EDI)',
        'lux_label': '视觉照度 (Illuminance)',
        'medi_delta': '≈ EML x 0.9063',
        
        'rating_excellent': '⭐ **日间健康评级：优秀** (EML {:.0f} ≥ 250) — 有助于提升日间警觉性与工作效率',
        'rating_good': '🌤️ **日间健康评级：基础达标** (EML {:.0f} ≥ 150) — 满足 WELL 基础要求',
        'rating_night': '🌙 **夜间模式识别** (EML {:.0f} ≤ 50) — 适合睡前照明环境',
        'rating_moderate': '⚠️ **节律刺激中等** (EML {:.0f}) — 介于日间与夜间之间，需根据使用时间评估',
        
        'vis_title': '📈 光谱可视化',
        'vis_original': '原始数据 (步长{:.1f}nm)',
        'vis_interp': '插值后光谱 (5nm步长)',
        'vis_weighted': '有效节律光谱 (SPD × Nz)',
        
        'data_note_title': '🔧 数据处理说明',
        'data_note_content': '''
- **原始数据**: {} 个数据点，波长范围 {:.0f} - {:.0f} nm，平均步长 {:.2f} nm
- **标准网格**: 380-780 nm，固定步长 5 nm（共 {} 个点）
- **插值方法**: 线性插值 (numpy.interp)
- **边界处理**: 超出 380-780nm 范围的数据自动补 0
        ''',
        
        'export_btn': '📥 导出计算结果 (CSV)',
        
        'warning_input': '请先上传文件或输入数据。',
        'error_parse': '光谱数据解析失败，请检查格式。需要两列：波长(nm) 和 功率(W/m²/nm)',
        'error_no_overlap': '错误：输入的光谱波长范围与标准范围 (380-780nm) 没有重叠，无法计算。',
        
        'footer': '⚠️ 免责声明: 本工具计算结果基于内置光谱响应函数与用户输入。不构成专业医疗或照明认证建议。',
        
        'detected': '📊 检测到输入数据: 波长范围 {:.0f} - {:.0f} nm，平均步长 {:.2f} nm，数据点数量: {}',
        
        'name_placeholder': '请输入姓名',
        'title_placeholder': '请输入头衔（可选）'
    },
    'en': {
        'page_title': 'Healthy Lighting Calculator (EML / m-EDI)',
        'page_subtitle': 'Based on CIE S026 / WELL Standard — Calculate Equivalent Melanopic Lux (EML) and melanopic Equivalent Daylight Illuminance (m-EDI)',
        
        'theory_title': '📖 Background & Formulas',
        'theory_content': '''
### What are EML and m-EDI?

- **EML (Equivalent Melanopic Lux)**: Measures the stimulation intensity of light on non-visual photoreceptor cells (ipRGC).
- **m-EDI (melanopic Equivalent Daylight Illuminance)**: Indicates how many lux of standard D65 daylight the current light source equals in terms of circadian effects.

### Calculation Formulas

The Equivalent Melanopic Lux (EML) is calculated using the following formula:

$$EML = K_m \\int E_{e,\\lambda}(\\lambda) \\cdot N_z(\\lambda) \\, d\\lambda \\cdot \\frac{\\int V(\\lambda) \\, d\\lambda}{\\int N_z(\\lambda) \\, d\\lambda}$$

Where:
- $E_{e,\\lambda}(\\lambda)$: Spectral power distribution of the light source (W/m²/nm)
- $N_z(\\lambda)$: Melanopic spectral efficiency function (peak at 480nm)
- $V(\\lambda)$: Photopic spectral efficiency function
- $K_m = 683.002$: Maximum photopic luminous efficacy (lm/W)

**Simplified Formula**: Since $\\int V(\\lambda) \\, d\\lambda = 106.857$ and $\\int N_z(\\lambda) \\, d\\lambda$ is defined as 1:

$$EML = 72983.25 \\times \\int E_{e,\\lambda}(\\lambda) \\cdot N_z(\\lambda) \\, d\\lambda$$

$$m\\text{-}EDI \\approx EML \\times 0.9063$$

### Health Benchmarks

- **Daytime (Office/School)**: EML ≥ 250 (promotes alertness and work efficiency)
- **Nighttime (Home/Sleep)**: EML ≤ 50 (promotes melatonin secretion and sleep)
        ''',
        
        'about_system': 'ℹ️ About System',
        'analyst_name': 'Analyst Name',
        'analyst_title': 'Analyst Title (Optional)',
        'contact': '📞 Contact: ✉️ Email: Techlife2027@gmail.com',
        
        'loaded': '✅ Standard spectral response functions loaded: V(λ) and Nz(λ) (Range: 380-780nm, Step: 5nm)',
        
        'input_title': '1️⃣ Input Light Source Spectral Power Distribution (SPD)',
        'upload_label': 'Option A: Upload CSV/TXT File',
        'upload_help': 'File should contain two columns: Wavelength(nm), Power(W/m²/nm). Supports any step size (1nm,5nm,10nm, etc.)',
        'textarea_label': 'Option B: Paste or Enter Spectral Data',
        'textarea_placeholder': 'Wavelength(nm),Power(W/m²/nm)\n380 0.0012\n385 0.0021\n390 0.0035',
        'unit_note': '💡 Unit Note: Power unit is **W/m²/nm** (Watts per square meter per nanometer), the standard unit for EML calculation',
        'calc_btn': '🚀 Calculate EML / m-EDI',
        
        'result_title': '📊 Results',
        'eml_label': 'Equivalent Melanopic Lux (EML)',
        'medi_label': 'melanopic Equivalent Daylight Illuminance (m-EDI)',
        'lux_label': 'Illuminance',
        'medi_delta': '≈ EML x 0.9063',
        
        'rating_excellent': '⭐ **Daytime Rating: Excellent** (EML {:.0f} ≥ 250) — Promotes daytime alertness and work efficiency',
        'rating_good': '🌤️ **Daytime Rating: Basic Compliance** (EML {:.0f} ≥ 150) — Meets WELL basic requirements',
        'rating_night': '🌙 **Nighttime Mode Detected** (EML {:.0f} ≤ 50) — Suitable for pre-sleep lighting',
        'rating_moderate': '⚠️ **Moderate Circadian Stimulation** (EML {:.0f}) — Between daytime and nighttime, evaluate based on usage time',
        
        'vis_title': '📈 Spectral Visualization',
        'vis_original': 'Original Data ({:.1f}nm step)',
        'vis_interp': 'Interpolated Spectrum (5nm step)',
        'vis_weighted': 'Effective Circadian Spectrum (SPD × Nz)',
        
        'data_note_title': '🔧 Data Processing Notes',
        'data_note_content': '''
- **Original Data**: {} points, wavelength range {:.0f} - {:.0f} nm, average step {:.2f} nm
- **Standard Grid**: 380-780 nm, fixed step 5 nm ({} points total)
- **Interpolation Method**: Linear interpolation (numpy.interp)
- **Boundary Handling**: Values outside 380-780nm are automatically set to 0
        ''',
        
        'export_btn': '📥 Export Results (CSV)',
        
        'warning_input': 'Please upload a file or enter data first.',
        'error_parse': 'Failed to parse spectral data. Please check format. Need two columns: Wavelength(nm) and Power(W/m²/nm)',
        'error_no_overlap': 'Error: Input wavelength range has no overlap with standard range (380-780nm). Cannot calculate.',
        
        'footer': '⚠️ Disclaimer: This tool is based on built-in spectral response functions and user input. Not for professional medical or lighting certification advice.',
        
        'detected': '📊 Detected input: wavelength range {:.0f} - {:.0f} nm, average step {:.2f} nm, data points: {}',
        
        'name_placeholder': 'Enter your name',
        'title_placeholder': 'Enter your title (optional)'
    }
}

# ==================== 核心数据加载 ====================

def load_spectral_data():
    """加载明视觉 (V_lambda) 和黑视素 (Nz_lambda) 的光谱光视效率。波长范围: 380nm 到 780nm, 步长 5nm"""
    wavelengths = np.arange(380, 785, 5)
    
    # V(λ) 数据 - 基于 CIE 1924 明视觉标准
    v_lambda_raw = [0.0000, 0.0000, 0.0000, 0.0001, 0.0002, 0.0004, 0.0006, 0.0010, 0.0017, 0.0026,
                    0.0040, 0.0060, 0.0090, 0.0130, 0.0180, 0.0250, 0.0340, 0.0450, 0.0590, 0.0760,
                    0.0960, 0.1210, 0.1500, 0.1840, 0.2220, 0.2650, 0.3120, 0.3630, 0.4170, 0.4740,
                    0.5280, 0.5810, 0.6310, 0.6790, 0.7240, 0.7660, 0.8050, 0.8400, 0.8710, 0.8990,
                    0.9240, 0.9450, 0.9620, 0.9750, 0.9850, 0.9920, 0.9970, 0.9990, 1.0000, 0.9990,
                    0.9970, 0.9940, 0.9900, 0.9850, 0.9790, 0.9720, 0.9640, 0.9550, 0.9450, 0.9340,
                    0.9220, 0.9090, 0.8950, 0.8800, 0.8650, 0.8490, 0.8320, 0.8150, 0.7970, 0.7790,
                    0.7600, 0.7410, 0.7220, 0.7030, 0.6840, 0.6650, 0.6460, 0.6280, 0.6100, 0.5920,
                    0.5750, 0.5580, 0.5420, 0.5270, 0.5120, 0.4980, 0.4840, 0.4700, 0.4570, 0.4450,
                    0.4330, 0.4210, 0.4100, 0.3990, 0.3880, 0.3780, 0.3680, 0.3580, 0.3480, 0.3390,
                    0.3300, 0.3210, 0.3130, 0.3050, 0.2970, 0.2890, 0.2820, 0.2750, 0.2680, 0.2610,
                    0.2550, 0.2490, 0.2430, 0.2370, 0.2310, 0.2260, 0.2210, 0.2160, 0.2110, 0.2060,
                    0.2010, 0.1970, 0.1920, 0.1880, 0.1840, 0.1800, 0.1760, 0.1720, 0.1680, 0.1640,
                    0.1600, 0.1570, 0.1540, 0.1510, 0.1480, 0.1450, 0.1420, 0.1390, 0.1360, 0.1330,
                    0.1300, 0.1260, 0.1220, 0.1180, 0.1140, 0.1100, 0.1060, 0.1020, 0.0980, 0.0940,
                    0.0900, 0.0860, 0.0820, 0.0780, 0.0740, 0.0700, 0.0660, 0.0620, 0.0580, 0.0540,
                    0.0500, 0.0460, 0.0420, 0.0380, 0.0340, 0.0300, 0.0260, 0.0220, 0.0180, 0.0140,
                    0.0100, 0.0070, 0.0040, 0.0020, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
    
    # Nz(λ) 黑视素数据 - 基于 CIE S026 标准 (峰值 480 nm)
    nz_lambda_raw = [0.0000, 0.0001, 0.0002, 0.0004, 0.0008, 0.0014, 0.0023, 0.0037, 0.0059, 0.0091,
                     0.0139, 0.0209, 0.0308, 0.0445, 0.0632, 0.0882, 0.1207, 0.1620, 0.2131, 0.2749,
                     0.3476, 0.4310, 0.5245, 0.6269, 0.7357, 0.8476, 0.9531, 1.0000, 0.9955, 0.9482,
                     0.8676, 0.7632, 0.6512, 0.5396, 0.4350, 0.3420, 0.2630, 0.1980, 0.1465, 0.1068,
                     0.0769, 0.0548, 0.0387, 0.0271, 0.0189, 0.0131, 0.0090, 0.0062, 0.0042, 0.0029,
                     0.0020, 0.0014, 0.0010, 0.0007, 0.0005, 0.0003, 0.0002, 0.0001, 0.0001, 0.0001,
                     0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0000, 0.0000,
                     0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                     0.0000, 0.0000]
    
    min_len = len(wavelengths)
    v_lambda = np.array(v_lambda_raw[:min_len])
    nz_lambda = np.array(nz_lambda_raw[:min_len])
    
    return wavelengths, v_lambda, nz_lambda


def detect_wavelength_step(wavelengths):
    """检测波长步长并返回平均步长"""
    if len(wavelengths) < 2:
        return 0
    steps = np.diff(wavelengths)
    avg_step = np.mean(steps)
    return avg_step


def parse_spectrum_flexible(text):
    """灵活解析光谱数据 - 支持任意分隔符（空格、逗号、制表符等）"""
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
        
        # 按波长排序
        sort_idx = np.argsort(wavelengths)
        wavelengths = wavelengths[sort_idx]
        powers = powers[sort_idx]
        
        return wavelengths, powers
    except Exception as e:
        return None, None


def linear_interpolate_to_standard_grid(x_input, y_input, x_standard):
    """线性插值算法 - 将任意步长的数据插值到标准网格"""
    x_input = np.asarray(x_input)
    y_input = np.asarray(y_input)
    x_standard = np.asarray(x_standard)
    
    # 线性插值，超出范围补0
    y_interpolated = np.interp(x_standard, x_input, y_input, left=0, right=0)
    
    return y_interpolated


def calculate_eml_and_medi(wavelengths, spectrum_w_m2_nm):
    """根据公式计算 EML 和 m-EDI"""
    wavelengths = np.asarray(wavelengths)
    spectrum = np.asarray(spectrum_w_m2_nm)
    
    # 加载标准函数 (5nm 步长)
    std_wavelengths, v_lambda, nz_lambda = load_spectral_data()
    
    # 使用线性插值将用户光谱映射到标准波长网格
    interp_spectrum = linear_interpolate_to_standard_grid(wavelengths, spectrum, std_wavelengths)
    
    # 使用简化公式: EML = 72983.25 * ∫ E(λ) * Nz(λ) dλ
    # 使用自定义梯形积分函数（兼容所有 NumPy 版本）
    weighted = interp_spectrum * nz_lambda
    weighted_integral_nz = trapezoid(weighted, std_wavelengths)
    eml_constant = 72983.25
    eml_value = eml_constant * weighted_integral_nz
    
    # m-EDI ≈ EML × 0.9063
    medi_value = eml_value * 0.9063
    
    # 视觉照度 E_v = Km * ∫ E(λ) * V(λ) dλ
    km = 683.002
    weighted_visual = interp_spectrum * v_lambda
    weighted_integral_v = trapezoid(weighted_visual, std_wavelengths)
    illuminance = km * weighted_integral_v
    
    return eml_value, medi_value, illuminance, interp_spectrum, std_wavelengths, v_lambda, nz_lambda


def check_wavelength_overlap(wavelengths):
    """检查输入波长与标准范围是否有重叠"""
    std_min, std_max = 380, 780
    input_min, input_max = np.min(wavelengths), np.max(wavelengths)
    
    # 检查是否有重叠
    if input_max < std_min or input_min > std_max:
        return False, None, None
    return True, input_min, input_max


# ==================== Streamlit UI ====================

def main():
    # 页面配置
    st.set_page_config(page_title="健康光计算器 (EML / m-EDI)", layout="wide")
    
    # 初始化语言状态
    if 'lang' not in st.session_state:
        st.session_state.lang = "zh"
    
    # 获取当前语言
    lang = st.session_state.lang
    t = LANGUAGES[lang]
    
    # ==================== 自定义CSS（仅语言按钮红底白字）====================
    st.markdown("""
    <style>
    /* 只针对语言按钮（通过 key 精确选择） */
    button[data-testid="baseButton-secondary"][kind="secondary"]:has(div:contains("中文")),
    button[data-testid="baseButton-secondary"][kind="secondary"]:has(div:contains("English")) {
        background-color: #dc2626 !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
    }
    button[data-testid="baseButton-secondary"][kind="secondary"]:has(div:contains("中文")):hover,
    button[data-testid="baseButton-secondary"][kind="secondary"]:has(div:contains("English")):hover {
        background-color: #b91c1c !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # ==================== 顶部：标题 + 语言按钮 ====================
    title_col, spacer, lang_col1, lang_col2 = st.columns([3, 4, 0.8, 0.8])
    
    with title_col:
        st.title("💡 " + t['page_title'])
    
    # 使用单独的容器和自定义 HTML 包装语言按钮，确保样式独立
    with lang_col1:
        # 使用 st.button 但通过 CSS 类名精确控制
        if st.button("中文", key="lang_zh_top", use_container_width=True):
            st.session_state.lang = "zh"
            st.rerun()
    
    with lang_col2:
        if st.button("English", key="lang_en_top", use_container_width=True):
            st.session_state.lang = "en"
            st.rerun()
    
    # 额外 CSS：精确控制特定 key 的按钮
    st.markdown("""
    <style>
    /* 精确选择器：只针对 key 为 lang_zh_top 和 lang_en_top 的按钮 */
    button[key="lang_zh_top"], button[key="lang_en_top"] {
        background-color: #dc2626 !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
    }
    button[key="lang_zh_top"]:hover, button[key="lang_en_top"]:hover {
        background-color: #b91c1c !important;
        color: white !important;
    }
    /* 确保其他按钮不受影响 - 重置其他按钮样式 */
    button:not([key="lang_zh_top"]):not([key="lang_en_top"]) {
        background-color: transparent !important;
        color: inherit !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown(t['page_subtitle'])
    
    # ==================== 左侧边栏 ====================
    with st.sidebar:
        # 1. 关于系统（最上面作为标题）
        st.header(t['about_system'])
        
        # 分析人姓名输入
        analyst_name = st.text_input(
            t['analyst_name'],
            placeholder=t['name_placeholder'],
            key="analyst_name"
        )
        
        # 分析人头衔输入（可选）
        analyst_title = st.text_input(
            t['analyst_title'],
            placeholder=t['title_placeholder'],
            key="analyst_title"
        )
        
        # 联系信息
        st.markdown(t['contact'])
        
        # 分隔线
        st.markdown("---")
        
        # 2. 背景知识与计算公式（折叠）
        with st.expander(t['theory_title'], expanded=False):
            st.markdown(t['theory_content'])
        
        # 显示分析人信息（如果已输入）
        if analyst_name:
            st.markdown("---")
            st.info(f"**{t['analyst_name']}:** {analyst_name}" + (f"\n\n**{t['analyst_title']}:** {analyst_title}" if analyst_title else ""))
    
    # ==================== 主要内容区域 ====================
    
    # 加载内置数据
    wavelengths, v_lambda, nz_lambda = load_spectral_data()
    st.success(t['loaded'])
    
    # 输入区域
    st.subheader(t['input_title'])
    
    uploaded_file = st.file_uploader(t['upload_label'], type=["csv", "txt"], help=t['upload_help'])
    
    spectrum_text = st.text_area(
        t['textarea_label'], 
        height=150, 
        placeholder=t['textarea_placeholder']
    )
    
    # 单位说明
    st.caption(t['unit_note'])
    
    # 计算按钮
    if st.button(t['calc_btn'], type="primary", use_container_width=True):
        wl_input, power_input = None, None
        
        if uploaded_file is not None:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            text_data = stringio.read()
            wl_input, power_input = parse_spectrum_flexible(text_data)
        elif spectrum_text:
            wl_input, power_input = parse_spectrum_flexible(spectrum_text)
        else:
            st.warning(t['warning_input'])
        
        if wl_input is not None and power_input is not None and len(wl_input) >= 2:
            # 检查波长范围是否有重叠
            has_overlap, input_min, input_max = check_wavelength_overlap(wl_input)
            
            if not has_overlap:
                st.error(t['error_no_overlap'])
            else:
                # 检测输入数据的步长
                step = detect_wavelength_step(wl_input)
                st.info(t['detected'].format(input_min, input_max, step, len(wl_input)))
                
                # 核心计算
                eml, medi, lux, interp_spectrum, std_wl, v_data, nz_data = calculate_eml_and_medi(wl_input, power_input)
                
                # 显示结果
                st.subheader(t['result_title'])
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(t['eml_label'], f"{eml:.1f} lx")
                with col2:
                    st.metric(t['medi_label'], f"{medi:.1f} lx", delta=t['medi_delta'])
                with col3:
                    st.metric(t['lux_label'], f"{lux:.1f} lx")
                
                # 健康评级
                if eml >= 250:
                    st.success(t['rating_excellent'].format(eml))
                elif eml >= 150:
                    st.info(t['rating_good'].format(eml))
                elif eml <= 50:
                    st.info(t['rating_night'].format(eml))
                else:
                    st.warning(t['rating_moderate'].format(eml))
                
                # 绘图
                st.subheader(t['vis_title'])
                
                fig = go.Figure()
                
                # 原始数据点（散点）
                fig.add_trace(go.Scatter(
                    x=wl_input, y=power_input, 
                    mode='markers', 
                    name=t['vis_original'].format(step),
                    marker=dict(color='orange', size=8, symbol='circle')
                ))
                
                # 插值后的光谱（连线）
                fig.add_trace(go.Scatter(
                    x=std_wl, y=interp_spectrum, 
                    mode='lines', 
                    name=t['vis_interp'],
                    line=dict(color='darkblue', width=2)
                ))
                
                # 有效节律光谱
                nz_weighted = interp_spectrum * nz_data
                if np.max(nz_weighted) > 0:
                    nz_scaled = nz_weighted / np.max(nz_weighted) * np.max(interp_spectrum) * 0.8
                    fig.add_trace(go.Scatter(
                        x=std_wl, y=nz_scaled, 
                        mode='lines', 
                        name=t['vis_weighted'],
                        line=dict(color='green', dash='dash', width=2)
                    ))
                
                fig.update_layout(
                    title="Spectral Power Distribution (SPD)",
                    xaxis_title="Wavelength (nm)",
                    yaxis_title="Power (W/m²/nm)",
                    legend_title="Spectrum Type",
                    template="plotly_white",
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 数据处理说明
                with st.expander(t['data_note_title']):
                    st.markdown(t['data_note_content'].format(
                        len(wl_input), input_min, input_max, step, len(std_wl)
                    ))
                
                # 导出按钮
                export_df = pd.DataFrame({
                    'Wavelength_nm': std_wl, 
                    'Interpolated_Spectrum_W_m2_nm': interp_spectrum,
                    'V_lambda': v_data,
                    'Nz_lambda': nz_data,
                    'Circadian_Spectrum_SPDxNz': interp_spectrum * nz_data
                })
                st.download_button(
                    label=t['export_btn'],
                    data=export_df.to_csv(index=False),
                    file_name='eml_calculation_result.csv',
                    mime='text/csv'
                )
        else:
            st.error(t['error_parse'])
    
    # 页脚
    st.divider()
    st.caption(t['footer'])


if __name__ == "__main__":
    main()
