import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import StringIO, BytesIO
import re
import base64
from datetime import datetime
import plotly.io as pio

# 设置 kaleido 作为图片导出引擎
pio.kaleido.scope.default_format = "png"

# ==================== 自定义梯形积分函数 ====================
def trapezoid(y, x):
    """手动实现梯形积分，避免 NumPy 版本兼容性问题"""
    y = np.asarray(y)
    x = np.asarray(x)
    
    if len(y) != len(x):
        raise ValueError("y and x must have the same length")
    
    if len(y) < 2:
        return 0.0
    
    dx = np.diff(x)
    if np.any(dx <= 0):
        idx = np.argsort(x)
        x = x[idx]
        y = y[idx]
        dx = np.diff(x)
    
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
        
        'well_comparison_title': '📋 与 WELL 标准对比',
        'well_table_header': 'WELL 等级',
        'well_eml_requirement': 'EML 要求',
        'well_medi_requirement': 'm-EDI 要求',
        'well_status': '当前状态',
        'well_excellent': '高品质推荐',
        'well_basis_a': '基础达标 (方案A)',
        'well_basis_b': '基础达标 (方案B)',
        'well_meet': '✅ 达标',
        'well_not_meet': '❌ 未达标',
        
        'rating_excellent': '⭐ **日间健康评级：优秀** (EML {:.0f} ≥ 250)',
        'rating_good': '🌤️ **日间健康评级：基础达标** (EML {:.0f} ≥ 150)',
        'rating_night': '🌙 **夜间模式识别** (EML {:.0f} ≤ 50)',
        'rating_moderate': '⚠️ **节律刺激中等** (EML {:.0f})',
        
        'vis_title': '📈 光谱可视化',
        'vis_original': '原始数据 (步长{:.1f}nm)',
        'vis_interp': '插值后光谱 (5nm步长)',
        'vis_vlambda': '明视觉光谱 V(λ)',
        'vis_weighted': '有效节律光谱 (SPD × Nz)',
        
        'data_note_title': '🔧 数据处理说明',
        'data_note_content': '原始数据: {} 个数据点，波长范围 {:.0f} - {:.0f} nm，平均步长 {:.2f} nm',
        
        'export_btn': '📥 导出 Word 报告 (.doc)',
        
        'warning_input': '请先上传文件或输入数据。',
        'error_parse': '光谱数据解析失败，请检查格式。需要两列：波长(nm) 和 功率(W/m²/nm)',
        'error_no_overlap': '错误：输入的光谱波长范围与标准范围 (380-780nm) 没有重叠，无法计算。',
        
        'footer': '⚠️ 免责声明: 本工具计算结果基于内置光谱响应函数与用户输入。不构成专业医疗或照明认证建议。',
        
        'detected': '📊 检测到输入数据: 波长范围 {:.0f} - {:.0f} nm，平均步长 {:.2f} nm，数据点数量: {}',
        
        'name_placeholder': '请输入姓名',
        'title_placeholder': '请输入头衔（可选）',
        
        'report_title': '健康照明 EML/m-EDI 分析报告',
        'report_date': '报告日期',
        'report_analyst': '分析人',
        
        'clear_data': '清除计算结果',
        
        # 图表标题
        'chart_title': '光谱功率分布 (SPD)',
        'chart_xlabel': '波长 (nm)',
        'chart_ylabel': '功率 (W/m²/nm)'
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

$$EML = K_m \\int E_{e,\\lambda}(\\lambda) \\cdot N_z(\\lambda) \\, d\\lambda \\cdot \\frac{\\int V(\\lambda) \\, d\\lambda}{\\int N_z(\\lambda) \\, d\\lambda}$$

$$EML = 72983.25 \\times \\int E_{e,\\lambda}(\\lambda) \\cdot N_z(\\lambda) \\, d\\lambda$$

$$m\\text{-}EDI \\approx EML \\times 0.9063$$

### Health Benchmarks

- **Daytime (Office/School)**: EML ≥ 250
- **Nighttime (Home/Sleep)**: EML ≤ 50
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
        'unit_note': '💡 Unit Note: Power unit is **W/m²/nm** (Watts per square meter per nanometer)',
        'calc_btn': '🚀 Calculate EML / m-EDI',
        
        'result_title': '📊 Results',
        'eml_label': 'Equivalent Melanopic Lux (EML)',
        'medi_label': 'melanopic Equivalent Daylight Illuminance (m-EDI)',
        'lux_label': 'Illuminance',
        'medi_delta': '≈ EML x 0.9063',
        
        'well_comparison_title': '📋 WELL Standard Comparison',
        'well_table_header': 'WELL Level',
        'well_eml_requirement': 'EML Requirement',
        'well_medi_requirement': 'm-EDI Requirement',
        'well_status': 'Status',
        'well_excellent': 'High Quality',
        'well_basis_a': 'Basic (Option A)',
        'well_basis_b': 'Basic (Option B)',
        'well_meet': '✅ Meet',
        'well_not_meet': '❌ Not Meet',
        
        'rating_excellent': '⭐ **Daytime Rating: Excellent** (EML {:.0f} ≥ 250)',
        'rating_good': '🌤️ **Daytime Rating: Basic Compliance** (EML {:.0f} ≥ 150)',
        'rating_night': '🌙 **Nighttime Mode Detected** (EML {:.0f} ≤ 50)',
        'rating_moderate': '⚠️ **Moderate Circadian Stimulation** (EML {:.0f})',
        
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
        
        'footer': '⚠️ Disclaimer: This tool is based on built-in spectral response functions and user input.',
        
        'detected': '📊 Detected input: wavelength range {:.0f} - {:.0f} nm, average step {:.2f} nm, data points: {}',
        
        'name_placeholder': 'Enter your name',
        'title_placeholder': 'Enter your title (optional)',
        
        'report_title': 'Healthy Lighting EML/m-EDI Analysis Report',
        'report_date': 'Report Date',
        'report_analyst': 'Analyst',
        
        'clear_data': 'Clear Results',
        
        # Chart labels
        'chart_title': 'Spectral Power Distribution (SPD)',
        'chart_xlabel': 'Wavelength (nm)',
        'chart_ylabel': 'Power (W/m²/nm)'
    }
}


# ==================== 核心功能函数 ====================

def load_spectral_data():
    wavelengths = np.arange(380, 785, 5)
    
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
    if len(wavelengths) < 2:
        return 0
    steps = np.diff(wavelengths)
    return np.mean(steps)


def parse_spectrum_flexible(text):
    try:
        lines = text.strip().split('\n')
        wavelengths = []
        powers = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
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


def linear_interpolate_to_standard_grid(x_input, y_input, x_standard):
    x_input = np.asarray(x_input)
    y_input = np.asarray(y_input)
    x_standard = np.asarray(x_standard)
    return np.interp(x_standard, x_input, y_input, left=0, right=0)


def calculate_eml_and_medi(wavelengths, spectrum_w_m2_nm):
    wavelengths = np.asarray(wavelengths)
    spectrum = np.asarray(spectrum_w_m2_nm)
    
    std_wavelengths, v_lambda, nz_lambda = load_spectral_data()
    interp_spectrum = linear_interpolate_to_standard_grid(wavelengths, spectrum, std_wavelengths)
    
    weighted = interp_spectrum * nz_lambda
    weighted_integral_nz = trapezoid(weighted, std_wavelengths)
    eml_value = 72983.25 * weighted_integral_nz
    medi_value = eml_value * 0.9063
    
    km = 683.002
    weighted_visual = interp_spectrum * v_lambda
    weighted_integral_v = trapezoid(weighted_visual, std_wavelengths)
    illuminance = km * weighted_integral_v
    
    return eml_value, medi_value, illuminance, interp_spectrum, std_wavelengths, v_lambda, nz_lambda


def check_wavelength_overlap(wavelengths):
    input_min, input_max = np.min(wavelengths), np.max(wavelengths)
    if input_max < 380 or input_min > 780:
        return False, None, None
    return True, input_min, input_max


def get_well_comparison(eml):
    standards = [
        {'level': 'well_excellent', 'eml_min': 250},
        {'level': 'well_basis_a', 'eml_min': 200},
        {'level': 'well_basis_b', 'eml_min': 150},
    ]
    results = []
    for std in standards:
        results.append({
            'level': std['level'],
            'eml_min': std['eml_min'],
            'eml_met': eml >= std['eml_min']
        })
    return results


def create_spectrum_figure(wl_input, power_input, interp_spectrum, std_wl, v_lambda, nz_lambda, step, t):
    """创建光谱可视化图表，使用传入的语言文本"""
    fig = go.Figure()
    
    # 原始数据点
    fig.add_trace(go.Scatter(
        x=wl_input, y=power_input, 
        mode='markers', 
        name=t['vis_original'].format(step),
        marker=dict(color='orange', size=8, symbol='circle')
    ))
    
    # 插值光谱
    fig.add_trace(go.Scatter(
        x=std_wl, y=interp_spectrum, 
        mode='lines', 
        name=t['vis_interp'],
        line=dict(color='darkblue', width=2)
    ))
    
    # 明视觉光谱
    v_max = np.max(v_lambda)
    if v_max > 0 and np.max(interp_spectrum) > 0:
        v_scaled = v_lambda / v_max * np.max(interp_spectrum) * 0.6
        fig.add_trace(go.Scatter(
            x=std_wl, y=v_scaled, 
            mode='lines', 
            name=t['vis_vlambda'],
            line=dict(color='red', dash='dot', width=2)
        ))
    
    # 节律光谱
    nz_weighted = interp_spectrum * nz_lambda
    if np.max(nz_weighted) > 0:
        nz_scaled = nz_weighted / np.max(nz_weighted) * np.max(interp_spectrum) * 0.8
        fig.add_trace(go.Scatter(
            x=std_wl, y=nz_scaled, 
            mode='lines', 
            name=t['vis_weighted'],
            line=dict(color='green', dash='dash', width=2)
        ))
    
    fig.update_layout(
        title=t['chart_title'],
        xaxis_title=t['chart_xlabel'],
        yaxis_title=t['chart_ylabel'],
        legend_title="Spectrum Type",
        template="plotly_white",
        hovermode='x unified',
        height=500,
        width=900
    )
    
    return fig


def generate_word_report(t, analyst_name, analyst_title, eml, medi, lux, 
                         well_results, fig, input_min, input_max, step, num_points):
    """生成 Word 报告，使用图片格式的光谱图"""
    
    # 获取健康评级
    if eml >= 250:
        rating_text = "优秀 - 日间使用推荐" if t['page_title'].startswith('健康') else "Excellent - Recommended for daytime"
    elif eml >= 150:
        rating_text = "基础达标 - 满足 WELL 基础要求" if t['page_title'].startswith('健康') else "Basic compliance - Meets WELL requirements"
    elif eml <= 50:
        rating_text = "夜间模式 - 适合睡前照明" if t['page_title'].startswith('健康') else "Nighttime mode - Suitable for pre-sleep lighting"
    else:
        rating_text = "中等刺激" if t['page_title'].startswith('健康') else "Moderate stimulation"
    
    # 生成 WELL 对比表格
    well_rows = ""
    for r in well_results:
        level_text = t[r['level']]
        status_icon = "✅" if r['eml_met'] else "❌"
        status_text = t['well_meet'] if r['eml_met'] else t['well_not_meet']
        well_rows += f"""
        <tr>
            <td style="padding: 8px 12px; border: 1px solid #ddd;">{level_text}</td>
            <td style="padding: 8px 12px; border: 1px solid #ddd; text-align: center;">≥ {r['eml_min']} lx</td>
            <td style="padding: 8px 12px; border: 1px solid #ddd; text-align: center;">{status_icon} {status_text}</td>
        </tr>
        """
    
    # 分析人信息
    analyst_info = analyst_name if analyst_name else ("未填写" if t['page_title'].startswith('健康') else "Not filled")
    if analyst_title:
        analyst_info += f" ({analyst_title})"
    
    # 只显示日期，不显示时间
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    # 将图表转换为 base64 图片
    try:
        img_bytes = fig.to_image(format="png", width=900, height=500, scale=1.5)
        img_base64 = base64.b64encode(img_bytes).decode()
        img_tag = f'<img src="data:image/png;base64,{img_base64}" style="width: 100%; max-width: 900px; margin: 20px 0; border: 1px solid #ddd; border-radius: 8px;">'
    except Exception as e:
        # 如果导出失败，显示占位符
        img_tag = '<p style="color: red; padding: 20px; background: #fee; text-align: center;">⚠️ 图表导出失败，请确保已安装 kaleido 包</p>'
    
    # Word HTML - 使用更宽的页面设置
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{t['report_title']}</title>
    <style>
        body {{
            font-family: '宋体', 'SimSun', 'Arial', sans-serif;
            margin: 1.5cm 1.5cm;
            padding: 0;
            font-size: 11pt;
            line-height: 1.4;
            color: #000000;
        }}
        .report-container {{
            max-width: 1000px;
            margin: 0 auto;
            background: white;
        }}
        h1 {{
            font-size: 20pt;
            font-weight: bold;
            margin: 20pt 0 10pt 0;
            padding: 0;
            color: #1e3a5f;
            border-bottom: 2px solid #4f46e5;
            padding-bottom: 8px;
        }}
        h2 {{
            font-size: 16pt;
            font-weight: bold;
            margin: 15pt 0 8pt 0;
            padding: 0;
            color: #334155;
            border-left: 4px solid #4f46e5;
            padding-left: 12px;
        }}
        .header-info {{
            background-color: #f5f5f5;
            padding: 12px 16px;
            margin: 15px 0;
            border-radius: 8px;
            border: 1px solid #ddd;
        }}
        .metrics {{
            margin: 15px 0;
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
        }}
        .metric-item {{
            flex: 1;
            min-width: 150px;
            padding: 12px 16px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            color: white;
        }}
        .metric-item.illuminance {{
            background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        }}
        .metric-value {{
            font-size: 22pt;
            font-weight: bold;
        }}
        .metric-label {{
            font-size: 10pt;
            opacity: 0.9;
            margin-top: 5px;
        }}
        .rating-badge {{
            display: inline-block;
            padding: 6px 14px;
            margin: 10px 0;
            background-color: #22c55e;
            color: white;
            border-radius: 20px;
            font-weight: bold;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th {{
            background-color: #4f46e5;
            color: white;
            padding: 10px 12px;
            border: 1px solid #ddd;
            font-weight: bold;
        }}
        td {{
            padding: 8px 12px;
            border: 1px solid #ddd;
        }}
        .footer {{
            margin-top: 30px;
            padding-top: 15px;
            border-top: 1px solid #ddd;
            font-size: 9pt;
            color: #666;
            text-align: center;
        }}
        .spectrum-container {{
            margin: 20px 0;
            text-align: center;
        }}
        .data-note {{
            background-color: #f8fafc;
            padding: 12px 16px;
            margin: 15px 0;
            border-radius: 8px;
            border: 1px solid #ddd;
            font-size: 10pt;
        }}
    </style>
</head>
<body>
<div class="report-container">
    <h1>{t['report_title']}</h1>
    
    <div class="header-info">
        <p style="margin: 4px 0;"><strong>{t['report_date']}:</strong> {current_date}</p>
        <p style="margin: 4px 0;"><strong>{t['report_analyst']}:</strong> {analyst_info}</p>
    </div>
    
    <h2>{t['result_title']}</h2>
    <div class="metrics">
        <div class="metric-item">
            <div class="metric-value">{eml:.1f} lx</div>
            <div class="metric-label">{t['eml_label']}</div>
        </div>
        <div class="metric-item">
            <div class="metric-value">{medi:.1f} lx</div>
            <div class="metric-label">{t['medi_label']}</div>
        </div>
        <div class="metric-item illuminance">
            <div class="metric-value">{lux:.1f} lx</div>
            <div class="metric-label">{t['lux_label']}</div>
        </div>
    </div>
    
    <div class="rating-badge">{rating_text}</div>
    
    <h2>{t['well_comparison_title']}</h2>
    <table>
        <thead>
            <tr>
                <th>{t['well_table_header']}</th>
                <th>{t['well_eml_requirement']}</th>
                <th>{t['well_status']}</th>
            </tr>
        </thead>
        <tbody>
            {well_rows}
        </tbody>
    </table>
    
    <h2>{t['vis_title']}</h2>
    <div class="spectrum-container">
        {img_tag}
    </div>
    
    <h2>{t['data_note_title']}</h2>
    <div class="data-note">
        {t['data_note_content'].format(num_points, input_min, input_max, step)}
        <br>标准网格: 380-780 nm，固定步长 5 nm（共 81 个点）
        <br>插值方法: 线性插值，超出范围自动补 0
    </div>
    
    <div class="footer">
        {t['footer']}
    </div>
</div>
</body>
</html>
"""
    
    return html_content.encode('utf-8')


# ==================== Streamlit UI ====================

def main():
    st.set_page_config(page_title="健康光计算器 (EML / m-EDI)", layout="wide")
    
    # 初始化状态
    if 'lang' not in st.session_state:
        st.session_state.lang = "zh"
    if 'calc_data' not in st.session_state:
        st.session_state.calc_data = None
    
    lang = st.session_state.lang
    t = LANGUAGES[lang]
    
    # CSS - 语言按钮红底白字，英文标题不分行
    st.markdown("""
    <style>
    button[key="lang_zh_top"], button[key="lang_en_top"] {
        background-color: #dc2626 !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
    }
    button[key="lang_zh_top"]:hover, button[key="lang_en_top"]:hover {
        background-color: #b91c1c !important;
    }
    button:not([key="lang_zh_top"]):not([key="lang_en_top"]) {
        background-color: transparent !important;
        color: inherit !important;
    }
    /* 确保英文标题不分行 */
    .stTitle {
        white-space: nowrap !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 顶部标题和语言按钮
    title_col, spacer, lang_col1, lang_col2 = st.columns([3, 4, 0.8, 0.8])
    
    with title_col:
        st.title("💡 " + t['page_title'])
    
    with lang_col1:
        if st.button("中文", key="lang_zh_top", use_container_width=True):
            st.session_state.lang = "zh"
            st.rerun()
    
    with lang_col2:
        if st.button("English", key="lang_en_top", use_container_width=True):
            st.session_state.lang = "en"
            st.rerun()
    
    st.markdown(t['page_subtitle'])
    
    # 侧边栏
    with st.sidebar:
        st.header(t['about_system'])
        
        analyst_name = st.text_input(t['analyst_name'], placeholder=t['name_placeholder'], key="analyst_name")
        analyst_title = st.text_input(t['analyst_title'], placeholder=t['title_placeholder'], key="analyst_title")
        
        st.markdown(t['contact'])
        st.markdown("---")
        st.markdown(f"### {t['theory_title']}")
        st.markdown(t['theory_content'])
        
        if analyst_name:
            st.markdown("---")
            st.info(f"**{t['analyst_name']}:** {analyst_name}" + (f"\n\n**{t['analyst_title']}:** {analyst_title}" if analyst_title else ""))
        
        if st.session_state.calc_data is not None:
            st.markdown("---")
            if st.button(t['clear_data'], use_container_width=True):
                st.session_state.calc_data = None
                st.rerun()
    
    # 主区域
    std_wavelengths, v_lambda, nz_lambda = load_spectral_data()
    st.success(t['loaded'])
    
    st.subheader(t['input_title'])
    
    uploaded_file = st.file_uploader(t['upload_label'], type=["csv", "txt"], help=t['upload_help'])
    spectrum_text = st.text_area(t['textarea_label'], height=150, placeholder=t['textarea_placeholder'])
    st.caption(t['unit_note'])
    
    if st.button(t['calc_btn'], type="primary", use_container_width=True):
        wl_input, power_input = None, None
        
        if uploaded_file is not None:
            text_data = uploaded_file.getvalue().decode("utf-8")
            wl_input, power_input = parse_spectrum_flexible(text_data)
        elif spectrum_text:
            wl_input, power_input = parse_spectrum_flexible(spectrum_text)
        else:
            st.warning(t['warning_input'])
        
        if wl_input is not None and power_input is not None and len(wl_input) >= 2:
            has_overlap, input_min, input_max = check_wavelength_overlap(wl_input)
            
            if not has_overlap:
                st.error(t['error_no_overlap'])
            else:
                step = detect_wavelength_step(wl_input)
                st.info(t['detected'].format(input_min, input_max, step, len(wl_input)))
                
                eml, medi, lux, interp_spectrum, std_wl, v_data, nz_data = calculate_eml_and_medi(wl_input, power_input)
                well_results = get_well_comparison(eml)
                fig = create_spectrum_figure(wl_input, power_input, interp_spectrum, std_wl, v_data, nz_data, step, t)
                
                # 保存到 session_state
                st.session_state.calc_data = {
                    'eml': eml, 'medi': medi, 'lux': lux,
                    'well_results': well_results,
                    'fig': fig,
                    'input_min': input_min, 'input_max': input_max,
                    'step': step, 'num_points': len(wl_input),
                    'analyst_name': analyst_name, 'analyst_title': analyst_title
                }
                
                # 显示结果
                st.subheader(t['result_title'])
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(t['eml_label'], f"{eml:.1f} lx")
                with col2:
                    st.metric(t['medi_label'], f"{medi:.1f} lx", delta=t['medi_delta'])
                with col3:
                    st.metric(t['lux_label'], f"{lux:.1f} lx")
                
                if eml >= 250:
                    st.success(t['rating_excellent'].format(eml))
                elif eml >= 150:
                    st.info(t['rating_good'].format(eml))
                elif eml <= 50:
                    st.info(t['rating_night'].format(eml))
                else:
                    st.warning(t['rating_moderate'].format(eml))
                
                # WELL 对比
                st.subheader(t['well_comparison_title'])
                well_df = pd.DataFrame([
                    {t['well_table_header']: t[r['level']], t['well_eml_requirement']: f"≥ {r['eml_min']} lx", t['well_status']: "✅ " + t['well_meet'] if r['eml_met'] else "❌ " + t['well_not_meet']}
                    for r in well_results
                ])
                st.table(well_df)
                
                if eml < 150:
                    st.warning("⚠️ 当前 EML 值低于 WELL 基础达标要求 (≥150 lx)")
                elif eml >= 250:
                    st.success("🎉 恭喜！当前光源已达到 WELL 高品质推荐标准！")
                
                # 光谱可视化
                st.subheader(t['vis_title'])
                st.plotly_chart(fig, use_container_width=True)
                
                # 数据处理说明
                with st.expander(t['data_note_title']):
                    st.markdown(t['data_note_content'].format(len(wl_input), input_min, input_max, step))
                    st.markdown(f"标准网格: 380-780 nm，固定步长 5 nm（共 {len(std_wl)} 个点）")
                    st.markdown("插值方法: 线性插值，超出范围自动补 0")
                
                # 导出报告
                report_data = generate_word_report(
                    t, analyst_name, analyst_title, eml, medi, lux, 
                    well_results, fig, input_min, input_max, step, len(wl_input)
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
    
    # 显示保存的结果
    elif st.session_state.calc_data is not None:
        data = st.session_state.calc_data
        
        st.subheader(t['result_title'])
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(t['eml_label'], f"{data['eml']:.1f} lx")
        with col2:
            st.metric(t['medi_label'], f"{data['medi']:.1f} lx", delta=t['medi_delta'])
        with col3:
            st.metric(t['lux_label'], f"{data['lux']:.1f} lx")
        
        if data['eml'] >= 250:
            st.success(t['rating_excellent'].format(data['eml']))
        elif data['eml'] >= 150:
            st.info(t['rating_good'].format(data['eml']))
        elif data['eml'] <= 50:
            st.info(t['rating_night'].format(data['eml']))
        else:
            st.warning(t['rating_moderate'].format(data['eml']))
        
        st.subheader(t['well_comparison_title'])
        well_df = pd.DataFrame([
            {t['well_table_header']: t[r['level']], t['well_eml_requirement']: f"≥ {r['eml_min']} lx", t['well_status']: "✅ " + t['well_meet'] if r['eml_met'] else "❌ " + t['well_not_meet']}
            for r in data['well_results']
        ])
        st.table(well_df)
        
        st.subheader(t['vis_title'])
        st.plotly_chart(data['fig'], use_container_width=True)
        
        with st.expander(t['data_note_title']):
            st.markdown(t['data_note_content'].format(data['num_points'], data['input_min'], data['input_max'], data['step']))
            st.markdown("标准网格: 380-780 nm，固定步长 5 nm（共 81 个点）")
        
        report_data = generate_word_report(
            t, analyst_name, analyst_title, data['eml'], data['medi'], data['lux'], 
            data['well_results'], data['fig'], data['input_min'], data['input_max'], 
            data['step'], data['num_points']
        )
        st.download_button(
            label=t['export_btn'],
            data=report_data,
            file_name=f"EML_Report_{datetime.now().strftime('%Y%m%d')}.doc",
            mime="application/msword",
            use_container_width=True
        )
    
    st.divider()
    st.caption(t['footer'])


if __name__ == "__main__":
    main()
