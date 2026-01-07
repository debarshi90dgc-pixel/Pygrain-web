import streamlit as st
import cv2
import numpy as np
import pandas as pd
from skimage import morphology
from skimage.segmentation import clear_border
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import io

# --- PAGE SETUP ---
st.set_page_config(page_title="PyGrain Field", page_icon="🪨", layout="wide")
st.title("🪨 PyGrain Field")
st.write("A cloud-based tool for pebble size analysis in the field.")

# --- SIDEBAR & INSTRUCTIONS ---
st.sidebar.header("1. Image & Calibration")
uploaded_file = st.sidebar.file_uploader("Upload Field Photo", type=['jpg', 'jpeg', 'png'])

# Initialize session state for scale if not present
if 'px_per_mm' not in st.session_state:
    st.session_state['px_per_mm'] = None

if uploaded_file is None:
    st.info("👈 Please upload an image to begin. Ensure a scale reference (e.g., ruler) is visible.")
    st.stop()

# --- MAIN LOGIC START ---
# Decode image immediately so we know its dimensions for the canvas
file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
img_bgr = cv2.imdecode(file_bytes, 1)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(img_rgb)
img_height, img_width = img_bgr.shape[:2]

# =========================================
# STEP 1: VISUAL CALIBRATION (The biggest change)
# =========================================
st.write("### Step 2: Set the Scale")
st.write("Draw a line along your reference object (e.g., ruler) in the image below.")

# Calculate canvas dimensions to fit screen while maintaining aspect ratio
# We limit max width to 700px for usability on phones
canvas_width = min(700, img_width)
canvas_height = int(canvas_width * (img_height / img_width))

# The drawing canvas widget
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=3,
    stroke_color="#FF0000",
    background_image=pil_img,
    update_streamlit=True,
    height=canvas_height,
    width=canvas_width,
    drawing_mode="line",
    key="canvas",
)

# Calculate pixel distance from drawn line
ref_len_px = 0.0
if canvas_result.json_data is not None:
    objects = pd.json_normalize(canvas_result.json_data["objects"])
    if len(objects) > 0:
        # Get coordinates of the last drawn line
        obj = objects.iloc[-1]
        x1, y1, x2, y2 = obj['left'], obj['top'], obj['left'] + obj['width'] * obj['scaleX'], obj['top'] + obj['height'] * obj['scaleY']
        # Euclidean distance formulation
        drawn_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        # Scale back up to original image dimensions
        scale_factor = img_width / canvas_width
        ref_len_px = drawn_distance * scale_factor

# Inputs for physical length
col1, col2 = st.columns(2)
with col1:
    st.metric("Measured Pixels (px)", f"{ref_len_px:.1f}")
with col2:
    ref_len_mm = st.number_input("Enter Known Length (mm)", value=50.0, min_value=0.1, 
                                 help="Enter the physical length in millimeters of the line you just drew.")

if ref_len_px > 0:
    st.session_state['px_per_mm'] = ref_len_px / ref_len_mm
    st.success(f"Calibration set: 1 mm = {st.session_state['px_per_mm']:.2f} pixels")
else:
    st.warning("Please draw a line on the scale object above to calibrate.")

# =========================================
# STEP 2: USER-FRIENDLY TUNING
# =========================================
st.write("---")
st.write("### Step 3: Analysis Settings")
st.sidebar.header("2. Tuning Parameters")

# 1. User-friendly "Min Diameter mm" instead of abstract "Min Area px"
min_diam_mm = st.sidebar.number_input(
    "Minimum Pebble Diameter (mm)", 
    value=4.0, min_value=0.1, step=0.5,
    help="The smallest grain size you want to detect. Pebbles smaller than this will be ignored."
)

# Calculate min area in pixels based on scale
if st.session_state['px_per_mm']:
    # Area of circle = pi * r^2 = pi * (diam/2)^2
    min_area_mm2 = np.pi * (min_diam_mm / 2)**2
    # Convert mm^2 to px^2: area_px = area_mm2 * (px_per_mm)^2
    min_area_px = min_area_mm2 * (st.session_state['px_per_mm']**2)
    st.sidebar.caption(f"(Calculated Min Area: {int(min_area_px)} px²)")
else:
    min_area_px = 700 # Fallback default if scale isn't set yet

# 2. Morph Kernel with helpful tooltip
morph_size = st.sidebar.slider(
    "Segmentation Strength (Morph Kernel)", 3, 51, 25, step=2,
    help="Controls how aggressively the app separates touching pebbles. Higher values separate better but may slightly shrink the detected size. Try increasing this if pebbles are merged together."
)

# =========================================
# STEP 3: RUN ANALYSIS (Essentially original logic)
# =========================================
st.write("---")
btn_col1, _ = st.columns([1, 2])
with btn_col1:
    run_btn = st.button("Run Analysis", type="primary", use_container_width=True)

if run_btn:
    if st.session_state['px_per_mm'] is None or ref_len_px == 0:
        st.error("Please calibrate the scale by drawing a line before running analysis.")
        st.stop()

    with st.spinner('Processing image...'):
        try:
            pixels_per_mm = st.session_state['px_per_mm']
            
            # Preprocess
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (7, 7), 2)
            _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Morph Closing
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            # Cleaning using calculated min_area_px
            bin_clean = morphology.remove_small_objects(closed > 0, min_size=min_area_px)
            bin_clean = morphology.remove_small_holes(bin_clean, area_threshold=min_area_px)
            bin_clean = clear_border(bin_clean)
            binary_clean = (bin_clean * 255).astype(np.uint8)

            # Contours
            contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            results = []
            img_disp = img_bgr.copy()
            pebble_id = 0
            
            for cnt in contours:
                if cv2.contourArea(cnt) < min_area_px: continue
                
                pebble_id += 1
                
                # Fit Ellipse
                if len(cnt) >= 5:
                    ellipse = cv2.fitEllipse(cnt)
                    (xc, yc), (major, minor), angle = ellipse
                    
                    # Draw
                    cv2.ellipse(img_disp, ellipse, (0, 255, 255), 2)
                    cv2.putText(img_disp, str(pebble_id), (int(xc), int(yc)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    
                    # Metrics
                    a_mm = (major / 2) / pixels_per_mm
                    b_mm = (minor / 2) / pixels_per_mm
                    Deq_mm = np.sqrt(a_mm * b_mm)
                    results.append(Deq_mm)

            # --- SHOW RESULTS ---
            st.image(img_disp, caption=f"Detected {pebble_id} Pebbles", channels="BGR")
            
            if results:
                d50 = np.median(results)
                st.success(f"✅ **D50 Size:** {d50:.2f} mm")
                
                # Data Table
                df = pd.DataFrame(results, columns=["Deq (mm)"])
                df.index += 1
                with st.expander("View Raw Data Table"):
                    st.dataframe(df)
                    
                # Download
                csv = df.to_csv().encode('utf-8')
                st.download_button("Download Results (CSV)", csv, "field_results.csv", "text/csv")
            else:
                st.warning("No pebbles detected. Try decreasing the 'Minimum Pebble Diameter'.")
                
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
