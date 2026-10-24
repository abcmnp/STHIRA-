import os
import re
import glob
import base64
import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.fft import rfft, rfftfreq
from skimage.morphology import skeletonize
from sklearn.decomposition import PCA

# 1. Configure page to wide mode and collapse Streamlit UI chrome
st.set_page_config(
    page_title="STHIRA | Structural Health Inspector & Risk Assessment",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 2. Hide Streamlit top header, footer, and padding so ONLY your HTML/CSS/JS UI shows
st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        header {visibility: hidden; height: 0px !important;}
        footer {visibility: hidden; height: 0px !important;}
        .block-container {
            padding-top: 0rem !important;
            padding-bottom: 0rem !important;
            padding-left: 0rem !important;
            padding-right: 0rem !important;
            max-width: 100% !important;
        }
        iframe {
            display: block;
            width: 100vw !important;
            border: none !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# PYTHON OPENCV + SCIPY EDT + FFT ENGINE (Matches STHIRA Slide 3 & Slide 4)
# ==============================================================================
def run_sthira_python_backend(pil_img: Image.Image, scale_mm_px: float = 0.085, tap_hz: float = 1650.0):
    rgb = np.array(pil_img.convert("RGB"))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    # CLAHE in LAB space to remove corridor shadows
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l_norm = clahe.apply(l_ch)

    blurred = cv2.GaussianBlur(l_norm, (5, 5), 0)
    mask = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 12
    )
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 1-px Skeleton + Euclidean Distance Transform (EDT)
    binary_bool = mask > 0
    if np.any(binary_bool):
        skel = skeletonize(binary_bool)
        edt_dist = distance_transform_edt(binary_bool)
        widths_mm = 2.0 * edt_dist[skel] * scale_mm_px
        max_w = float(np.percentile(widths_mm, 98)) if len(widths_mm) > 0 else 0.0
        mean_w = float(np.mean(widths_mm)) if len(widths_mm) > 0 else 0.0
        span_mm = float(np.sum(skel) * scale_mm_px)

        ys, xs = np.nonzero(skel)
        if len(xs) > 5:
            pca = PCA(n_components=2).fit(np.column_stack((xs, ys)).astype(float))
            vx, vy = pca.components_[0]
            angle = float(abs(np.degrees(np.arctan2(vy, vx))))
            angle = 180.0 - angle if angle > 90 else angle
        else:
            angle = 0.0
    else:
        max_w, mean_w, span_mm, angle = 0.0, 0.0, 0.0, 0.0

    # 5-Tier IS 456:2000 & ASTM D4580 Rule Engine (Slide 4)
    if 30.0 <= angle <= 60.0:
        verdict = "Shear risk: inspect first"
        cost_inr = round(max(0.5, span_mm / 1000.0) * 2450 + 1200, 2)
    elif max_w > 0.3 and tap_hz < 900:
        verdict = "Void risk: inspect"
        cost_inr = round(max(0.5, span_mm / 1000.0) * 1950 + 1200, 2)
    elif max_w > 0.3 and tap_hz >= 1400:
        verdict = "Plaster spalling: cosmetic"
        cost_inr = round(max(0.5, span_mm / 1000.0) * 680, 2)
    elif 0.1 <= max_w <= 0.3:
        verdict = "Minor: seal and monitor"
        cost_inr = round(max(0.5, span_mm / 1000.0) * 420, 2)
    else:
        verdict = "Hairline shrinkage: no action"
        cost_inr = 0.0

    return {
        "max_width_mm": round(max_w, 2),
        "mean_width_mm": round(mean_w, 2),
        "angle_deg": round(angle, 1),
        "tap_freq_hz": round(tap_hz, 0),
        "verdict": verdict,
        "cpwd_cost_inr": cost_inr,
    }


# Optional Sidebar Python Backend Bridge (Collapsed by default so UI is untouched)
with st.sidebar:
    st.subheader("🐍 STHIRA Python Backend Engine")
    st.caption("OpenCV CLAHE + SciPy EDT + FFT + IS 456 Rule Engine")
    scale_in = st.slider("ArUco / ID Scale (mm/px)", 0.02, 0.25, 0.085, 0.005)
    tap_in = st.slider("Coin-Tap FFT Peak (Hz)", 400, 2800, 1650, 50)
    img_in = st.file_uploader("Verify Crack Image in Python Engine", type=["png", "jpg", "jpeg"])
    if img_in:
        res = run_sthira_python_backend(Image.open(img_in), scale_in, tap_in)
        st.json(res)


# ==============================================================================
# AUTO-BUNDLER: LOADS YOUR EXACT HTML, CSS & JS WITHOUT ALTERING THE UI
# ==============================================================================
def bundle_repository_frontend() -> str:
    # Find primary HTML file in root or subfolders
    html_files = (
        glob.glob("index.html")
        + glob.glob("*.html")
        + glob.glob("*/*.html")
    )
    if not html_files:
        return "<h2 style='font-family:sans-serif;padding:2rem;'>Place your .html, .css, and .js files in this repository root.</h2>"

    main_html_path = html_files[0]
    base_dir = os.path.dirname(main_html_path)

    with open(main_html_path, "r", encoding="utf-8", errors="ignore") as f:
        html_doc = f.read()

    # Inline all local CSS files referenced or present in the repo so styles never break
    css_files = glob.glob(os.path.join(base_dir, "*.css")) + glob.glob(os.path.join(base_dir, "*/*.css"))
    css_bundle = ""
    for css_file in dict.fromkeys(css_files):
        with open(css_file, "r", encoding="utf-8", errors="ignore") as cf:
            css_bundle += f"\n/* {os.path.basename(css_file)} */\n" + cf.read()

    if css_bundle:
        style_tag = f"<style>\n{css_bundle}\n</style>"
        if "</head>" in html_doc:
            html_doc = html_doc.replace("</head>", f"{style_tag}\n</head>")
        else:
            html_doc = style_tag + "\n" + html_doc

    # Inline local images as base64 data URIs so images render inside Streamlit's iframe
    for img_ext in ("*.png", "*.jpg", "*.jpeg", "*.svg", "*.webp"):
        for img_path in glob.glob(os.path.join(base_dir, img_ext)) + glob.glob(os.path.join(base_dir, "*", img_ext)):
            rel_name = os.path.basename(img_path)
            ext = os.path.splitext(rel_name)[1].lower().replace(".", "")
            mime = "image/svg+xml" if ext == "svg" else f"image/{'jpeg' if ext == 'jpg' else ext}"
            with open(img_path, "rb") as img_f:
                b64_str = base64.b64encode(img_f.read()).decode("utf-8")
            html_doc = html_doc.replace(f'"{rel_name}"', f'"data:{mime};base64,{b64_str}"')
            html_doc = html_doc.replace(f"'{rel_name}'", f"'data:{mime};base64,{b64_str}'")

    # Inline all local JS files referenced or present in the repo so interactivity works 100%
    js_files = glob.glob(os.path.join(base_dir, "*.js")) + glob.glob(os.path.join(base_dir, "*/*.js"))
    js_bundle = ""
    for js_file in dict.fromkeys(js_files):
        with open(js_file, "r", encoding="utf-8", errors="ignore") as jf:
            js_bundle += f"\n// {os.path.basename(js_file)}\n" + jf.read()

    if js_bundle:
        script_tag = f"<script>\n{js_bundle}\n</script>"
        if "</body>" in html_doc:
            html_doc = html_doc.replace("</body>", f"{script_tag}\n</body>")
        else:
            html_doc = html_doc + "\n" + script_tag

    return html_doc


# Render your exact HTML/CSS/JS frontend across the full screen
bundled_ui = bundle_repository_frontend()
components.html(bundled_ui, height=1080, scrolling=True)
