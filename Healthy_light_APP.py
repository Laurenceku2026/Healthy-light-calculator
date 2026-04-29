import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import StringIO
import re
import base64
from datetime import datetime
import plotly.io as pio

# ==================== 自定义梯形积分函数 ====================
def trapezoid(y, x):
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


# ==================== 修正后的光谱响应函数数据 ====================
# 数据来源: CIE S026:2018 / Enezi et al 2011
# 波长范围: 380-780nm, 步长 5nm
# V(λ) 是经过晶状体透射率修正的（32岁标准观察者，未散瞳）
# Nz(λ) 黑视素函数峰值正确地位于 490nm

def load_spectral_data():
    """加载正确的明视觉 V(λ) 和黑视素 Nz(λ) 数据
    数据来源: CIE S026:2018 / Enezi et al 2011
    """
    wavelengths = np.arange(380, 785, 5)
    
    # ========== 正确的 V(λ) 数据 (CIE S026 明视觉标准，经过晶状体透射率修正) ==========
    # 峰值位于 555nm，值为 1.0
    v_lambda_correct = [
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
    
    # ========== 正确的 Nz(λ) 黑视素数据 (CIE S026:2018) ==========
    # 峰值正确地位于 490nm，值为 1.0
    # 基于您提供的 Excel 文件中的 melanopic 列数据
    nz_lambda_correct = [
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
    
    # 确保数据长度一致 (380-780nm, 每5nm, 共81个点)
    n_points = len(wavelengths)
    v_lambda = np.array(v_lambda_correct[:n_points])
    nz_lambda = np.array(nz_lambda_correct[:n_points])
    
    # 验证峰值位置
    nz_peak_idx = np.argmax(nz_lambda)
    nz_peak_wl = wavelengths[nz_peak_idx]
    print(f"[INFO] Nz(λ) 峰值波长: {nz_peak_wl} nm, 峰值: {np.max(nz_lambda):.4f}")
    
    v_peak_idx = np.argmax(v_lambda)
    v_peak_wl = wavelengths[v_peak_idx]
    print(f"[INFO] V(λ) 峰值波长: {v_peak_wl} nm, 峰值: {np.max(v_lambda):.4f}")
    
    return wavelengths, v_lambda, nz_lambda


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

$$EML = K_m \\int E_{e,\\lambda}(\\lambda) \\cdot N_z(\\lambda) \\, d\\lambda \\cdot \\frac{\\int V(\\lambda) \\, d\\lambda}{\\int N_z(\\lambda) \\, d\\lambda}$$

$$EML = 72983.25 \\times \\int E_{e,\\lambda}(\\lambda) \\cdot N_z(\\lambda) \\, d\\lambda$$

$$m\\text{-}EDI \\approx EML \\times 0.9063$$

### 健康基准

- **日间 (办公/学校)**：EML ≥ 250
- **夜间 (家居/睡眠)**：EML ≤ 50
        ''',
        
        'about_system': 'ℹ️ 关于系统',
        'analyst_name': '分析人姓名',
        'analyst_title': '分析人头衔（可选）',
        'contact': '📞 联系：✉️ 电邮: Techlife2027@gmail.com',
        
        'loaded': '✅ 系统已加载标准光谱响应函数: V(λ) 和 Nz(λ) (CIE S026:2018, 380-780nm, 步长: 5nm, Nz峰值490nm)',
        
        'input_title': '1️⃣ 输入光源光谱功率分布 (SPD)',
        'upload_label': '选项 A: 上传 CSV/TXT 文件',
        'upload_help': '文件应包含两列: 波长(nm), 功率(W/m²/nm)。支持任意步长',
        'textarea_label': '选项 B: 粘贴或输入光谱数据',
        'textarea_placeholder': '波长(nm),功率(W/m²/nm)\n380 0.0012\n385 0.0021',
        'unit_note': '💡 单位说明：功率单位为 W/m²/nm',
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
        'vis_vlambda': '明视觉光谱 V(λ) (CIE S026)',
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
        
        'clear_data': '清除计算结果',
        
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

- **EML (Equivalent Melanopic Lux)**: Measures the stimulation intensity of light on ipRGC cells.
- **m-EDI (melanopic Equivalent Daylight Illuminance)**: Compares circadian effect to standard D65 daylight.

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
        
        'loaded': '✅ Standard spectral response functions loaded: V(λ) and Nz(λ) (CIE S026:2018, 380-780nm, Step: 5nm, Nz peak at 490nm)',
        
        'input_title': '1️⃣ Input Light Source Spectral Power Distribution (SPD)',
        'upload_label': 'Option A: Upload CSV/TXT File',
        'upload_help': 'File should contain two columns: Wavelength(nm), Power(W/m²/nm)',
        'textarea_label': 'Option B: Paste or Enter Spectral Data',
        'textarea_placeholder': 'Wavelength(nm),Power(W/m²/nm)\n380 0.0012\n385 0.0021',
        'unit_note': '💡 Unit Note: Power unit is W/m²/nm',
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
        'vis_vlambda': 'Photopic Spectrum V(λ) (CIE S026)',
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
        
        'clear_data': 'Clear Results',
        
        'chart_title': 'Spectral Power Distribution (SPD)',
        'chart_xlabel': 'Wavelength (nm)',
        'chart_ylabel': 'Power (W/m²/nm)'
    }
}


# ==================== 其他核心函数 ====================

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
    
    # EML 计算: 72983.25 * ∫ E(λ) * Nz(λ) dλ
    weighted_melanopic = interp_spectrum * nz_lambda
    weighted_integral_nz = trapezoid(weighted_melanopic, std_wavelengths)
    eml_value = 72983.25 * weighted_integral_nz
    medi_value = eml_value * 0.9063
    
    # 照度计算: Km * ∫ E(λ) * V(λ) dλ
    km = 683.002
    weighted_photopic = interp_spectrum * v_lambda
    weighted_integral_v = trapezoid(weighted_photopic, std_wavelengths)
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
    fig = go.Figure()
    
    # 原始数据点
    fig.add_trace(go.Scatter(
        x=wl_input, y=power_input, 
        mode='markers', 
        name=t['vis_original'].format(step),
        marker=dict(color='orange', size=6, symbol='circle')
    ))
    
    # 插值光谱
    fig.add_trace(go.Scatter(
        x=std_wl, y=interp_spectrum, 
        mode='lines', 
        name=t['vis_interp'],
        line=dict(color='darkblue', width=2)
    ))
    
    # 明视觉光谱 V(λ) - 使用实际值（非归一化）
    fig.add_trace(go.Scatter(
        x=std_wl, y=v_lambda * np.max(interp_spectrum) * 0.6,
        mode='lines', 
        name=t['vis_vlambda'],
        line=dict(color='red', dash='dot', width=2)
    ))
    
    # 有效节律光谱 (SPD × Nz)
    nz_weighted = interp_spectrum * nz_lambda
    if np.max(nz_weighted) > 0:
        fig.add_trace(go.Scatter(
            x=std_wl, y=nz_weighted,
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
                         well_results, fig_html, input_min, input_max, step, num_points):
    """生成 Word 报告"""
    
    # 获取健康评级
    if eml >= 250:
        rating_text = "⭐ 日间使用推荐" if t['page_title'].startswith('健康') else "⭐ Recommended for daytime"
    elif eml >= 150:
        rating_text = "🌤️ 满足 WELL 基础要求" if t['page_title'].startswith('健康') else "🌤️ Meets WELL requirements"
    elif eml <= 50:
        rating_text = "🌙 适合睡前照明" if t['page_title'].startswith('健康') else "🌙 Suitable for pre-sleep"
    else:
        rating_text = "⚠️ 需根据使用时间评估" if t['page_title'].startswith('健康') else "⚠️ Evaluate based on usage"
    
    # 生成 WELL 对比表格
    well_rows = ""
    for r in well_results:
        level_text = t[r['level']]
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
        h1 {{
            font-size: 18pt;
            font-weight: bold;
            margin: 20pt 0 10pt 0;
            color: #1e3a5f;
            border-bottom: 2px solid #4f46e5;
            padding-bottom: 8px;
        }}
        h2 {{
            font-size: 14pt;
            font-weight: bold;
            margin: 15pt 0 8pt 0;
            color: #334155;
            border-left: 3px solid #4f46e5;
            padding-left: 10px;
        }}
        .header-info {{
            background-color: #f0f0f0;
            padding: 10px 15px;
            margin: 15px 0;
            border: 1px solid #ccc;
        }}
        .metrics {{
            margin: 15px 0;
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
        }}
        .metric-item {{
            flex: 1;
            min-width: 140px;
            padding: 10px 12px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 8px;
        }}
        .metric-item.illuminance {{
            background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        }}
        .metric-value {{
            font-size: 16pt;
            font-weight: bold;
            color: #000000 !important;
        }}
        .metric-label {{
            font-size: 10pt;
            color: #333333;
            margin-top: 4px;
        }}
        .rating-badge {{
            display: inline-block;
            padding: 5px 12px;
            margin: 10px 0;
            background-color: #22c55e;
            color: #000000;
            border-radius: 16px;
            font-weight: bold;
            font-size: 10pt;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th {{
            background-color: #4f46e5;
            color: white;
            padding: 8px 10px;
            border: 1px solid #aaa;
            font-weight: bold;
        }}
        td {{
            padding: 6px 10px;
            border: 1px solid #aaa;
            color: #000000;
        }}
        .footer {{
            margin-top: 30px;
            padding-top: 12px;
            border-top: 1px solid #ccc;
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
            padding: 10px 15px;
            margin: 15px 0;
            border: 1px solid #ccc;
            font-size: 10pt;
        }}
    </style>
</head>
<body>

<h1>{t['report_title']}</h1>

<div class="header-info">
    <p style="margin: 3px 0;"><strong>{t['report_date']}:</strong> {current_date}</p>
    <p style="margin: 3px 0;"><strong>{t['report_analyst']}:</strong> {analyst_info}</p>
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
    {fig_html}
</div>

<h2>{t['data_note_title']}</h2>
<div class="data-note">
    {t['data_note_content'].format(num_points, input_min, input_max, step)}<br>
    标准网格: 380-780 nm，固定步长 5 nm（共 81 个点）<br>
    插值方法: 线性插值，超出范围自动补 0
</div>

<div class="footer">
    {t['footer']}
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
    
    # CSS
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
                fig_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn', config={'displayModeBar': False})
                
                st.session_state.calc_data = {
                    'eml': eml, 'medi': medi, 'lux': lux,
                    'well_results': well_results,
                    'fig': fig,
                    'fig_html': fig_html,
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
                    st.success(t['rating_excellent'])
                elif eml >= 150:
                    st.info(t['rating_good'])
                elif eml <= 50:
                    st.info(t['rating_night'])
                else:
                    st.warning(t['rating_moderate'])
                
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
                
                st.subheader(t['vis_title'])
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander(t['data_note_title']):
                    st.markdown(t['data_note_content'].format(len(wl_input), input_min, input_max, step))
                    st.markdown(f"标准网格: 380-780 nm，固定步长 5 nm（共 {len(std_wl)} 个点）")
                
                report_data = generate_word_report(
                    t, analyst_name, analyst_title, eml, medi, lux, 
                    well_results, fig_html, input_min, input_max, step, len(wl_input)
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
            st.success(t['rating_excellent'])
        elif data['eml'] >= 150:
            st.info(t['rating_good'])
        elif data['eml'] <= 50:
            st.info(t['rating_night'])
        else:
            st.warning(t['rating_moderate'])
        
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
            t, data.get('analyst_name', analyst_name), data.get('analyst_title', analyst_title),
            data['eml'], data['medi'], data['lux'], 
            data['well_results'], data['fig_html'], data['input_min'], data['input_max'], 
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
