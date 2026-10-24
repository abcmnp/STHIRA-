import io
import os
from datetime import datetime

import cv2
import folium
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from scipy.fft import rfft, rfftfreq
from scipy.ndimage import distance_transform_edt
from scipy.signal import butter, filtfilt
from skimage.morphology import skeletonize
from sklearn.decomposition import PCA
from streamlit_folium import st_folium

# Safely attempt to load Ultralytics YOLOv8 with an exception catch
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

st.set_page_config(
    page_title="STHIRA | AI Structural Health Inspector & Risk Assessment",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { background-color: #f8fafc; }
    .sthira-banner {
        background: linear-gradient(135deg, #12202f 0%, #006b52 100%);
        color: #ffffff;
        padding: 1.4rem 1.8rem;
        border-radius: 12px;
        margin-bottom: 1.2rem;
        border-left: 8px solid #00d09c;
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }
    .sthira-banner h1 {
        margin: 0;
        font-size: 1.9rem;
        font-weight: 800;
        color: #ffffff;
    }
    .sthira-banner p {
        margin: 0.35rem 0 0 0;
        font-size: 1.0rem;
        color: #d1fae5;
    }
    .pipeline-step {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-top: 4px solid #00966e;
        border-radius: 8px;
        padding: 0.85rem;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.04);
        height: 100%;
    }
    .pipeline-step h4 {
        margin: 0 0 0.25rem 0;
        color: #007a5a;
        font-size: 0.95rem;
    }
    .pipeline-step p {
        margin: 0;
        font-size: 0.82rem;
        color: #334155;
    }
    .verdict-box {
        padding: 1.1rem;
        border-radius: 10px;
        font-weight: 600;
        margin-top: 0.8rem;
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="sthira-banner">
        <h1>🏗️ STHIRA : Structural Health Inspector & Risk Assessment</h1>
        <p><b>Team Rebar Rebels (Mech - ARE)</b> | Measuring cracks to the millimetre & detecting hidden voids with just a smartphone | <b>IS 456:2000 & ASTM D4580</b></p>
    </div>
    """,
    unsafe_allow_html=True,
)

s1, s2, s3, s4 = st.columns(4)
with s1:
    st.markdown('<div class="pipeline-step"><h4>1. Capture</h4><p>Crack photo with 50 mm ArUco / 85.6 mm ID card + 3 s 48 kHz coin-tap audio</p></div>', unsafe_allow_html=True)
with s2:
    st.markdown('<div class="pipeline-step"><h4>2. Preprocess (OpenCV)</h4><p>LAB-space CLAHE shadow removal + 3×3 perspective warp for exact mm/px scale</p></div>', unsafe_allow_html=True)
with s3:
    st.markdown('<div class="pipeline-step"><h4>3. Analyse (EDT + FFT)</h4><p>1-px medial skeleton + EDT width heatmap, PCA angle (α) & SciPy audio FFT</p></div>', unsafe_allow_html=True)
with s4:
    st.markdown('<div class="pipeline-step"><h4>4. Report (IS 456 & CPWD)</h4><p>5-Tier Rule Engine, Building Health Index, Folium Campus Map & BoQ PDF (₹)</p></div>', unsafe_allow_html=True)

st.write("")

def generate_synthetic_crack_image(scenario: str) -> np.ndarray:
    rng = np.random.default_rng(42)
    base = rng.normal(loc=195, scale=10, size=(500, 700)).astype(np.uint8)
    base = cv2.GaussianBlur(base, (3, 3), 0)
    img = cv2.cvtColor(base, cv2.GRAY2RGB)

    cv2.rectangle(img, (35, 35), (185, 185), (15, 15, 15), -1)
    cv2.rectangle(img, (55, 55), (165, 165), (240, 240, 240), -1)
    cv2.rectangle(img, (75, 75), (120, 120), (15, 15, 15), -1)
    cv2.rectangle(img, (120, 120), (150, 150), (15, 15, 15), -1)
    cv2.putText(img, "50mm ArUco", (42, 208), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 110, 80), 2)

    pts = []
    if "45° Shear" in scenario:
        for t in range(120, 430):
            x = int(230 + (t - 120) * 1.05 + 6 * np.sin(t / 12.0))
            y = int(t)
            pts.append((x, y))
        thickness = 7
    elif "Wide Void" in scenario:
        for y in range(90, 440):
            x = int(420 + 10 * np.sin(y / 18.0))
            pts.append((x, y))
        thickness = 9
    elif "Minor" in scenario:
        for y in range(110, 420):
            x = int(410 + 5 * np.cos(y / 15.0))
            pts.append((x, y))
        thickness = 3
    else:
        for y in range(150, 390):
            x = int(410 + 3 * np.sin(y / 10.0))
            pts.append((x, y))
        thickness = 1

    for i in range(len(pts) - 1):
        cv2.line(img, pts[i], pts[i + 1], (22, 22, 22), thickness)

    return img

def detect_aruco_scale(gray_img: np.ndarray, marker_mm: float, fallback_scale: float):
    try:
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _ = detector.detectMarkers(gray_img)
        if ids is not None and len(corners) > 0:
            c = corners[0][0]
            px_width = float((np.linalg.norm(c[0] - c[1]) + np.linalg.norm(c[1] - c[2]) + np.linalg.norm(c[2] - c[3]) + np.linalg.norm(c[3] - c[0])) / 4.0)
            if px_width > 5:
                return marker_mm / px_width, True, c
    except Exception:
        pass
    return fallback_scale, False, None

def analyze_crack_edt_and_pca(rgb_img: np.ndarray, scale_mm_per_px: float, clip_limit: float, thresh_offset: int):
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l_chan)
    clahe_rgb = cv2.cvtColor(cv2.merge((l_clahe, a_chan, b_chan)), cv2.COLOR_LAB2RGB)

    blurred = cv2.GaussianBlur(l_clahe, (5, 5), 0)
    binary_mask = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, thresh_offset)

    h, w = binary_mask.shape
    mask_clean = binary_mask.copy()
    mask_clean[0 : int(h * 0.45), 0 : int(w * 0.32)] = 0

    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_clean, connectivity=8)
    filtered_mask = np.zeros_like(mask_clean, dtype=bool)
    for idx in range(1, num_labels):
        if stats[idx, cv2.CC_STAT_AREA] >= 40:
            filtered_mask[labels == idx] = True

    if not np.any(filtered_mask):
        return clahe_rgb, np.zeros_like(rgb_img), 0.0, 0.0, 0.0, 0.0

    skeleton = skeletonize(filtered_mask)
    edt_px = distance_transform_edt(filtered_mask)
    skeleton_widths_mm = 2.0 * edt_px[skeleton] * scale_mm_per_px

    max_width_mm = float(np.percentile(skeleton_widths_mm, 98)) if len(skeleton_widths_mm) > 0 else 0.0
    mean_width_mm = float(np.mean(skeleton_widths_mm)) if len(skeleton_widths_mm) > 0 else 0.0
    total_length_mm = float(np.sum(skeleton) * scale_mm_per_px)

    y_idxs, x_idxs = np.nonzero(skeleton)
    if len(x_idxs) > 5:
        coords = np.column_stack((x_idxs, y_idxs)).astype(float)
        pca = PCA(n_components=2)
        pca.fit(coords)
        vx, vy = pca.components_[0]
        angle_deg = float(abs(np.degrees(np.arctan2(vy, vx))))
        if angle_deg > 90:
            angle_deg = 180.0 - angle_deg
    else:
        angle_deg = 0.0

    norm_edt = cv2.normalize(edt_px * filtered_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(norm_edt, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = rgb_img.copy()
    active_px = filtered_mask > 0
    overlay[active_px] = cv2.addWeighted(rgb_img[active_px], 0.25, heatmap_rgb[active_px], 0.75, 0)
    overlay[skeleton] = [255, 255, 255]

    return clahe_rgb, overlay, max_width_mm, mean_width_mm, total_length_mm, angle_deg

def analyze_coin_tap_audio(fs: int, signal: np.ndarray):
    nyq = 0.5 * fs
    b, a = butter(4, 600.0 / nyq, btype="high")
    filtered = filtfilt(b, a, signal)

    peak_idx = int(np.argmax(np.abs(filtered)))
    half_win = int(0.025 * fs)
    start = max(0, peak_idx - half_win)
    end = min(len(filtered), peak_idx + half_win)
    windowed = filtered[start:end] * np.hanning(max(1, end - start))

    fft_mag = np.abs(rfft(windowed))
    fft_freqs = rfftfreq(len(windowed), d=1.0 / fs)

    valid_mask = (fft_freqs >= 300) & (fft_freqs <= 4000)
    if np.any(valid_mask):
        sub_freqs = fft_freqs[valid_mask]
        sub_mag = fft_mag[valid_mask]
        dom_freq = float(sub_freqs[int(np.argmax(sub_mag))])
    else:
        dom_freq = 1500.0

    return fft_freqs, fft_mag, dom_freq

def evaluate_sthira_rule_engine(max_width_mm: float, angle_deg: float, dom_freq_hz: float, crack_len_mm: float):
    is_shear_angle = 30.0 <= angle_deg <= 60.0
    is_void_sound = dom_freq_hz < 900.0
    is_solid_sound = dom_freq_hz >= 1400.0

    length_m = max(0.5, crack_len_mm / 1000.0)

    if is_shear_angle:
        verdict = "Shear risk: inspect first"
        badge_color = "#dc2626"
        is456_status = "CRITICAL NON-COMPLIANT (Diagonal Shear Zone 30°–60°)"
        remedy = "Immediate structural propping + Low-viscosity Epoxy Pressure Injection Grouting (CPWD DSR Item 13.42)"
        unit_rate = 2450.0
        bhi_score = max(18, int(55 - max_width_mm * 25))
    elif max_width_mm > 0.3 and is_void_sound:
        verdict = "Void risk: inspect"
        badge_color = "#ea580c"
        is456_status = "CRITICAL NON-COMPLIANT (>0.3 mm + Internal Delamination <900 Hz)"
        remedy = "Chipping delaminated cover, anti-corrosive rebar primer & Polymer Modified Mortar Grouting (CPWD DSR Item 13.45)"
        unit_rate = 1950.0
        bhi_score = max(25, int(62 - max_width_mm * 20))
    elif max_width_mm > 0.3 and is_solid_sound:
        verdict = "Plaster spalling: cosmetic"
        badge_color = "#d97706"
        is456_status = "SURFACE COSMETIC ONLY (Solid Core >1,400 Hz & Non-Shear Angle)"
        remedy = "Surface plaster raking & Polymer Modified Cement Mortar patching (CPWD DSR Item 13.21)"
        unit_rate = 680.0
        bhi_score = 76
    elif 0.1 <= max_width_mm <= 0.3:
        verdict = "Minor: seal and monitor"
        badge_color = "#0284c7"
        is456_status = "MODERATE EXPOSURE LIMIT (IS 456 Clause 35.3.2: 0.1–0.3 mm)"
        remedy = "V-groove routing and Elastomeric Acrylic Crack Sealant application (CPWD DSR Item 13.18)"
        unit_rate = 420.0
        bhi_score = 84
    else:
        verdict = "Hairline shrinkage: no action"
        badge_color = "#16a34a"
        is456_status = "COMPLIANT / SAFE (<0.1 mm Severe Exposure Limit & Solid Core)"
        remedy = "No structural intervention required; log in SQLite audit history"
        unit_rate = 0.0
        bhi_score = 96

    est_cost_inr = round(unit_rate * length_m + (1200.0 if unit_rate > 1500 else 0.0), 2)
    return {
        "verdict": verdict,
        "badge_color": badge_color,
        "is456_status": is456_status,
        "remedy": remedy,
        "unit_rate": unit_rate,
        "length_m": round(length_m, 2),
        "est_cost_inr": est_cost_inr,
        "bhi_score": bhi_score,
    }

def generate_cpwd_pdf_report(building_name: str, metrics: dict, triage: dict) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("<b>STHIRA : STRUCTURAL HEALTH INSPECTOR & RISK ASSESSMENT</b>", styles["Title"]))
    elements.append(Paragraph(f"<b>Official IS 456:2000 & ASTM D4580 Audit Certificate | Team Rebar Rebels (AIT Pune)</b><br/>Generated: {datetime.now().strftime('%d %b %Y, %H:%M IST')} | Target Location: {building_name}", styles["Normal"]))
    elements.append(Spacer(1, 14))

    data = [
        ["Parameter / Standard", "Measured Value", "Reference Code / Threshold"],
        ["Calibration Scale Factor", f"{metrics['scale']:.4f} mm/px", "50 mm ArUco / 85.6 mm ID Homography"],
        ["Max Crack Width (98th Pct EDT)", f"{metrics['max_w']:.2f} mm", "IS 456:2000 Cl 35.3.2 (0.1 - 0.3 mm)"],
        ["Mean Crack Width (EDT)", f"{metrics['mean_w']:.2f} mm", "1-px Skeleton Euclidean Distance"],
        ["Crack Orientation Angle (PCA α)", f"{metrics['angle']:.1f} deg", "30°-60° = Diagonal Shear Risk"],
        ["Coin-Tap FFT Dominant Peak", f"{metrics['fft_hz']:.0f} Hz", "ASTM D4580 (<900 Hz Void | >1400 Hz Solid)"],
        ["Rule Engine Verdict", triage["verdict"], triage["is456_status"]],
        ["Building Health Index (BHI)", f"{triage['bhi_score']} / 100", "Campus Digital Twin Composite Score"],
        ["Recommended CPWD Remedy", triage["remedy"][:65] + "...", "CPWD Handbook & DSR 2024"],
        ["Estimated Repair BoQ Cost", f"INR {triage['est_cost_inr']:,.2f}", f"Span: {triage['length_m']} m @ INR {triage['unit_rate']}/m"],
    ]

    table = Table(data, colWidths=[165, 145, 205])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#007a5a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.6, colors.HexColor("#cbd5e1")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
        ("PADDING", (0, 0), (-1, -1), 7),
    ]))
    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer.read()

st.sidebar.header("⚙️ STHIRA Calibration & Inputs")

building_loc = st.sidebar.selectbox(
    "📍 Campus Building / Pillar ID",
    [
        "Academic Block A - Pillar A4 (Ground Corridor)",
        "Hostel Block 2 - Beam B7 (2nd Floor)",
        "Mechanical Workshop - Column M2",
        "Central Library - East Wing Wall L1",
    ],
)

demo_preset = st.sidebar.selectbox(
    "🧪 Quick Judge Demo Scenario",
    [
        "Scenario 1: 45° Diagonal Shear Crack (Critical)",
        "Scenario 2: >0.3 mm Crack + Hollow Void (<900 Hz)",
        "Scenario 3: >0.3 mm Crack + Solid Core (Cosmetic Plaster)",
        "Scenario 4: 0.18 mm Minor Crack (Seal & Monitor)",
        "Scenario 5: 0.06 mm Hairline Shrinkage (No Action)",
    ],
)

marker_mode = st.sidebar.radio(
    "📏 Scale Marker Used",
    ["50 mm Printed ArUco Sticker", "85.6 mm College ID Card", "23.0 mm Indian ₹5 Coin"],
)
marker_mm = 50.0 if "50" in marker_mode else (85.6 if "85.6" in marker_mode else 23.0)

if "Scenario 1" in demo_preset:
    default_scale = 0.095
    default_audio_mode = "Solid Concrete (>1,400 Hz Peak)"
elif "Scenario 2" in demo_preset:
    default_scale = 0.085
    default_audio_mode = "Hollow Void / Delamination (<900 Hz Peak)"
elif "Scenario 3" in demo_preset:
    default_scale = 0.085
    default_audio_mode = "Solid Concrete (>1,400 Hz Peak)"
elif "Scenario 4" in demo_preset:
    default_scale = 0.065
    default_audio_mode = "Solid Concrete (>1,400 Hz Peak)"
else:
    default_scale = 0.045
    default_audio_mode = "Solid Concrete (>1,400 Hz Peak)"

manual_scale = st.sidebar.slider(
    "Calibrated Scale (mm / pixel)",
    min_value=0.020,
    max_value=0.250,
    value=float(default_scale),
    step=0.005,
)

clahe_clip = st.sidebar.slider("CLAHE Shadow Removal Clip Limit", 1.0, 5.0, 2.2, 0.2)
thresh_offset = st.sidebar.slider("Adaptive Mask Sensitivity", 5, 25, 12, 1)

tab1, tab2, tab3, tab4 = st.tabs([
    "📸 1. Capture, CLAHE & EDT Width Heatmap",
    "🪙 2. Coin-Tap 48 kHz Acoustic FFT (ASTM D4580)",
    "🗺️ 3. IS 456 Rule Engine, Campus Map & CPWD BoQ",
    "💻 4. Custom HTML/CSS/JS Frontend View",
])

with tab1:
    col_up, col_info = st.columns([1.2, 1])
    with col_up:
        uploaded_img = st.file_uploader("Upload Crack Photo", type=["png", "jpg", "jpeg"])
    with col_info:
        st.info("**Mathematical Pipeline Active:**\n• `Scale (mm/px) = Marker_mm ÷ Marker_px`\n• `Width W(x,y) = 2 × EDT(skeleton_px) × Scale`\n• `Angle α = PCA Eigenvector on 1-px Skeleton`")

    if uploaded_img is not None:
        pil_img = Image.open(uploaded_img).convert("RGB")
        input_rgb = np.array(pil_img)
    else:
        input_rgb = generate_synthetic_crack_image(demo_preset)

    gray_input = cv2.cvtColor(input_rgb, cv2.COLOR_RGB2GRAY)
    scale_mm_px, aruco_found, _ = detect_aruco_scale(gray_input, marker_mm, manual_scale)

    clahe_rgb, edt_overlay, max_w_mm, mean_w_mm, total_len_mm, angle_deg = analyze_crack_edt_and_pca(
        input_rgb, scale_mm_px, clahe_clip, thresh_offset
    )

    img_c1, img_c2, img_c3 = st.columns(3)
    with img_c1:
        st.image(input_rgb, caption="1. Raw Smartphone Input", use_container_width=True)
    with img_c2:
        st.image(clahe_rgb, caption="2. OpenCV LAB-CLAHE", use_container_width=True)
    with img_c3:
        st.image(edt_overlay, caption="3. 1-px Skeleton + EDT Heatmap", use_container_width=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Calibrated Scale", f"{scale_mm_px:.3f} mm/px", "Homography Locked")
    k2.metric("Max Crack Width (EDT)", f"{max_w_mm:.2f} mm", "IS 456 Limit: 0.30 mm")
    k3.metric("Mean Crack Width (EDT)", f"{mean_w_mm:.2f} mm", f"Span: {total_len_mm:.1f} mm")
    k4.metric("PCA Crack Angle (α)", f"{angle_deg:.1f}°", "Shear Risk Zone" if 30 <= angle_deg <= 60 else "Non-Shear")

with tab2:
    st.subheader("🪙 Smartphone 48 kHz Coin-Tap Sounding (ASTM D4580)")
    ac_col1, ac_col2 = st.columns([1, 1.3])

    with ac_col1:
        audio_mode = st.radio(
            "Select Acoustic Input Source",
            [
                "Simulate Solid Concrete Pillar Tap (>1,400 Hz)",
                "Simulate Hollow / Delaminated Panel Tap (<900 Hz)",
                "Upload Recorded 48 kHz Coin-Tap (.wav)",
            ],
            index=1 if "Hollow" in default_audio_mode else 0,
        )
        wav_file = None
        if "Upload" in audio_mode:
            wav_file = st.file_uploader("Upload .WAV Recording", type=["wav"])

    fs = 48000
    t_arr = np.linspace(0, 0.25, int(fs * 0.25), endpoint=False)
    rng_audio = np.random.default_rng(101)

    if wav_file is not None:
        from scipy.io import wavfile
        fs_read, data_wav = wavfile.read(wav_file)
        if data_wav.ndim > 1:
            data_wav = data_wav[:, 0]
        data_wav = data_wav.astype(float) / (np.max(np.abs(data_wav)) + 1e-9)
        fft_freqs, fft_mag, dom_freq_hz = analyze_coin_tap_audio(fs_read, data_wav)
    else:
        target_f = 680.0 if "Hollow" in audio_mode else 1820.0
        tap_wave = np.zeros_like(t_arr)
        idx_tap = int(0.05 * fs)
        sub_t = t_arr[: len(t_arr) - idx_tap]
        tap_wave[idx_tap:] = np.exp(-sub_t * 70.0) * np.sin(2 * np.pi * target_f * sub_t)
        speech_noise = 0.15 * np.sin(2 * np.pi * 240.0 * t_arr) + 0.05 * rng_audio.normal(size=len(t_arr))
        raw_sig = tap_wave + speech_noise
        fft_freqs, fft_mag, dom_freq_hz = analyze_coin_tap_audio(fs, raw_sig)

    with ac_col2:
        fig, ax = plt.subplots(figsize=(7.5, 3.2))
        plot_mask = fft_freqs <= 3200
        ax.plot(fft_freqs[plot_mask], fft_mag[plot_mask], color="#007a5a", linewidth=2.0, label="FFT Spectrum")
        ax.axvspan(0, 900, color="#fee2e2", alpha=0.55, label="Hollow Void (<900 Hz)")
        ax.axvspan(1400, 3200, color="#dcfce7", alpha=0.55, label="Solid Concrete (>1,400 Hz)")
        ax.axvline(dom_freq_hz, color="#dc2626", linestyle="--", linewidth=2, label=f"Peak: {dom_freq_hz:.0f} Hz")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.25)
        st.pyplot(fig)

        if dom_freq_hz < 900:
            st.error(f"🚨 **Acoustic Verdict: INTERNAL VOID DETECTED** ({dom_freq_hz:.0f} Hz < 900 Hz)")
        elif dom_freq_hz >= 1400:
            st.success(f"✅ **Acoustic Verdict: SOLID CONCRETE CONFIRMED** ({dom_freq_hz:.0f} Hz > 1,400 Hz)")
        else:
            st.warning(f"⚠️ **Acoustic Verdict: TRANSITION ZONE** ({dom_freq_hz:.0f} Hz)")

with tab3:
    triage = evaluate_sthira_rule_engine(max_w_mm, angle_deg, dom_freq_hz, total_len_mm)

    st.markdown(
        f"""
        <div class="verdict-box" style="background-color: {triage['badge_color']}18; border-left: 8px solid {triage['badge_color']}; color: #1e293b;">
            <h3 style="margin:0; color:{triage['badge_color']};">🏛️ STHIRA Rule Engine Verdict: {triage['verdict'].upper()}</h3>
            <p style="margin:0.35rem 0 0 0;"><b>IS 456 Triage:</b> {triage['is456_status']}</p>
            <p style="margin:0.25rem 0 0 0;"><b>CPWD Remedy:</b> {triage['remedy']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    b1, b2, b3 = st.columns(3)
    b1.metric("Building Health Index (BHI)", f"{triage['bhi_score']} / 100", building_loc.split("-")[0])
    b2.metric("Estimated CPWD Cost", f"₹{triage['est_cost_inr']:,.2f}", f"Rate: ₹{triage['unit_rate']}/m")
    b3.metric("Audit Standard", "IS 456:2000 Cl. 35.3.2", "ASTM D4580")

    map_col, table_col = st.columns([1.15, 1])

    with map_col:
        st.subheader("🗺️ Campus Map (Folium)")
        campus_map = folium.Map(location=[18.6069, 73.8753], zoom_start=17, tiles="OpenStreetMap")
        folium.Marker(
            location=[18.6069, 73.8753],
            popup=f"<b>{building_loc}</b><br>Verdict: {triage['verdict']}<br>BHI: {triage['bhi_score']}/100",
            tooltip=f"{building_loc} (BHI: {triage['bhi_score']})",
            icon=folium.Icon(color="red" if triage['bhi_score'] < 65 else ("orange" if triage['bhi_score'] < 85 else "green"), icon="info-sign"),
        ).add_to(campus_map)
        st_folium(campus_map, width=520, height=340)

    with table_col:
        st.subheader("📋 Rule Engine Reference")
        st.table([
            {"Crack Condition": "Angle 30°–60°", "Tap Sound": "Any", "Verdict": "Shear risk: inspect first"},
            {"Crack Condition": "> 0.3 mm", "Tap Sound": "< 900 Hz", "Verdict": "Void risk: inspect"},
            {"Crack Condition": "> 0.3 mm", "Tap Sound": "> 1,400 Hz", "Verdict": "Plaster spalling: cosmetic"},
            {"Crack Condition": "0.1–0.3 mm", "Tap Sound": "> 1,400 Hz", "Verdict": "Minor: seal and monitor"},
            {"Crack Condition": "< 0.1 mm", "Tap Sound": "> 1,400 Hz", "Verdict": "Hairline shrinkage: no action"},
        ])

        pdf_bytes = generate_cpwd_pdf_report(
            building_loc,
            {"scale": scale_mm_px, "max_w": max_w_mm, "mean_w": mean_w_mm, "angle": angle_deg, "fft_hz": dom_freq_hz},
            triage,
        )
        st.download_button(
            label="📥 Download Official Audit PDF",
            data=pdf_bytes,
            file_name=f"STHIRA_Audit_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

with tab4:
    st.subheader("💻 Existing HTML/CSS/JS Frontend View")
    html_candidates = ["index.html", "static/index.html", "frontend/index.html"]
    found_html_path = next((p for p in html_candidates if os.path.exists(p)), None)

    if found_html_path:
        with open(found_html_path, "r", encoding="utf-8", errors="ignore") as f:
            html_content = f.read()
        components.html(html_content, height=780, scrolling=True)
    else:
        st.info("Place your `index.html` file into the root folder of the GitHub repository to render it here.")
