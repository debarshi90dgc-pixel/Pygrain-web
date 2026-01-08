import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage import morphology
from skimage.segmentation import clear_border
from streamlit_image_coordinates import streamlit_image_coordinates

# ----------------------------
# Helpers
# ----------------------------
def odd_clip(x, lo=3, hi=51):
    x = int(x)
    x = max(lo, min(hi, x))
    if x % 2 == 0:
        x += 1 if x < hi else -1
    return x

def ensure_session():
    if "clicks" not in st.session_state:
        st.session_state.clicks = []
    if "px_per_mm" not in st.session_state:
        st.session_state.px_per_mm = None

# ----------------------------
# Page
# ----------------------------
st.set_page_config(page_title="PyGrain Field", page_icon="🪨", layout="wide")
st.title("🪨 PyGrain Field")
st.caption("Field-ready pebble sizing (tap-to-calibrate scale + automatic minimum area).")

ensure_session()

# ----------------------------
# Sidebar: Upload + Settings
# ----------------------------
st.sidebar.header("1) Upload")
uploaded_file = st.sidebar.file_uploader("Upload field photo", type=["jpg", "jpeg", "png"])

st.sidebar.header("2) Detection settings")
invert = st.sidebar.checkbox(
    "Invert threshold (use if pebbles are darker than background)",
    value=True,
    help="If detection looks wrong (background detected as pebble), toggle this."
)

min_diam_mm = st.sidebar.number_input(
    "Minimum pebble diameter (mm)",
    value=4.0, min_value=0.1, step=0.5,
    help="Pebbles smaller than this will be ignored. This replaces 'Minimum Pebble Area' and is field-friendly."
)

auto_kernel = st.sidebar.checkbox(
    "Auto kernel size (recommended)",
    value=True,
    help="Auto chooses kernel based on your scale + minimum diameter. Turn off only if you want manual tuning."
)

manual_kernel = st.sidebar.slider(
    "Morph kernel size (odd)",
    min_value=3, max_value=51, value=25, step=2,
    help=("Controls separation of touching pebbles. "
          "Higher = stronger separation but may slightly shrink edges. "
          "Lower = less separation, may merge pebbles.")
)

st.sidebar.markdown("---")
st.sidebar.header("3) Controls")
if st.sidebar.button("Clear calibration clicks"):
    st.session_state.clicks = []
    st.session_state.px_per_mm = None

# ----------------------------
# If no upload
# ----------------------------
if uploaded_file is None:
    st.info("👈 Upload an image to begin. Include a reference object (ruler/coin/card) in the same plane as pebbles.")
    st.stop()

# ----------------------------
# Read image
# ----------------------------
img_bytes = uploaded_file.read()
img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
if img_bgr is None:
    st.error("Could not read image. Please upload a valid JPG/PNG.")
    st.stop()

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(img_rgb)
H, W = img_bgr.shape[:2]

# ----------------------------
# Step 1: Calibration by tapping 2 points
# ----------------------------
st.subheader("Step 1 — Set the scale (tap two points on the reference)")
st.write("Tap **start point** and **end point** of a known-length reference (e.g., 0 to 100 mm on a ruler).")

display_width = min(900, W)  # phone-friendly
scale_factor = W / display_width

coords = streamlit_image_coordinates(pil_img, width=display_width, key="img_coords")

if coords is not None and "x" in coords and "y" in coords:
    # Save click
    st.session_state.clicks.append((coords["x"], coords["y"]))
    # keep only last 2 clicks
    st.session_state.clicks = st.session_state.clicks[-2:]

c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    st.write("**Clicks:**", len(st.session_state.clicks))
with c2:
    if len(st.session_state.clicks) == 2:
        (x1d, y1d), (x2d, y2d) = st.session_state.clicks
        dx = (x2d - x1d) * scale_factor
        dy = (y2d - y1d) * scale_factor
        ref_len_px = float(np.sqrt(dx*dx + dy*dy))
        st.metric("Reference length (px)", f"{ref_len_px:.1f}")
    else:
        ref_len_px = 0.0
        st.metric("Reference length (px)", "—")
with c3:
    ref_len_mm = st.number_input(
        "Enter known length (mm)",
        value=100.0, min_value=0.1,
        help="Example: if you tapped 0 to 10 cm on a ruler, enter 100 mm."
    )

if ref_len_px > 0:
    st.session_state.px_per_mm = ref_len_px / ref_len_mm
    st.success(f"✅ Scale set: **1 mm = {st.session_state.px_per_mm:.3f} px**")
else:
    st.warning("Tap **two points** on the image to set the scale.")

# ----------------------------
# Step 2: Automatic Minimum Area + Kernel suggestion
# ----------------------------
st.subheader("Step 2 — Automatic filters & guidance")

if st.session_state.px_per_mm is None:
    st.info("Set the scale first (two taps) — then the app will auto-compute minimum area and kernel suggestion.")
    st.stop()

px_per_mm = st.session_state.px_per_mm

# Minimum area derived from min diameter (circle approximation)
min_area_mm2 = np.pi * (min_diam_mm / 2.0) ** 2
min_area_px = min_area_mm2 * (px_per_mm ** 2)

# Kernel suggestion based on min diameter in pixels (simple practical heuristic)
min_diam_px = min_diam_mm * px_per_mm
kernel_suggest = odd_clip(0.6 * min_diam_px, lo=3, hi=51)

kcol1, kcol2, kcol3 = st.columns(3)
kcol1.metric("Min diameter (px)", f"{min_diam_px:.1f}")
kcol2.metric("Min area (px²)", f"{int(min_area_px)}")
kcol3.metric("Suggested kernel", f"{kernel_suggest}")

kernel_size = kernel_suggest if auto_kernel else manual_kernel

with st.expander("How to choose Morph Kernel (field guidance)"):
    st.markdown(
        f"""
- **Kernel controls separation of touching pebbles** (morphological closing/cleanup strength).
- If **two pebbles merge into one**, increase kernel a bit.
- If **pebbles look over-smoothed / edges shrink**, decrease kernel.
- Current suggestion is based on your minimum diameter: **{kernel_suggest}** (odd).
"""
    )

# ----------------------------
# Step 3: Run analysis
# ----------------------------
st.subheader("Step 3 — Run analysis")

run = st.button("Run Analysis", type="primary", use_container_width=True)

if run:
    with st.spinner("Processing..."):
        try:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (7, 7), 2)

            # Otsu threshold
            thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
            _, binary = cv2.threshold(blur, 0, 255, thresh_type + cv2.THRESH_OTSU)

            # Morph close
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            # skimage cleaning
            bw = closed > 0
            bw = morphology.remove_small_objects(bw, min_size=int(min_area_px))
            bw = morphology.remove_small_holes(bw, area_threshold=int(min_area_px))
            bw = clear_border(bw)
            binary_clean = (bw.astype(np.uint8) * 255)

            contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            out = img_bgr.copy()
            deq_list = []
            pid = 0

            for cnt in contours:
                if cv2.contourArea(cnt) < min_area_px:
                    continue

                if len(cnt) < 5:
                    continue  # ellipse needs >=5 points

                ellipse = cv2.fitEllipse(cnt)
                (xc, yc), (major, minor), angle = ellipse

                pid += 1
                cv2.ellipse(out, ellipse, (0, 255, 255), 2)
                cv2.putText(out, str(pid), (int(xc), int(yc)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                a_mm = (major / 2.0) / px_per_mm
                b_mm = (minor / 2.0) / px_per_mm
                deq_mm = float(np.sqrt(a_mm * b_mm))
                deq_list.append(deq_mm)

            st.image(out, channels="BGR", caption=f"Detected pebbles: {pid}")

            if len(deq_list) == 0:
                st.warning("No pebbles detected. Try lowering minimum diameter or toggling invert threshold.")
            else:
                df = pd.DataFrame({"Pebble_ID": np.arange(1, len(deq_list) + 1),
                                   "Deq_mm": np.round(deq_list, 3)})
                d50 = float(np.median(deq_list))
                st.success(f"✅ **D50 = {d50:.2f} mm**")

                with st.expander("Raw results table"):
                    st.dataframe(df, use_container_width=True)

                st.download_button(
                    "Download CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="pygrain_field_results.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Processing failed: {e}")
