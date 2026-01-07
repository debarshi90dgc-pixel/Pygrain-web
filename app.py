import streamlit as st
import cv2
import numpy as np
import pandas as pd
from skimage import morphology
from skimage.segmentation import clear_border

# 1. SETUP THE PAGE
st.set_page_config(page_title="PyGrain Mobile", page_icon="🪨")
st.title("🪨 PyGrain Mobile")
st.write("Upload a pebble photo. This app runs entirely in the cloud.")

# 2. SIDEBAR CONTROLS
st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

# Scale inputs
ref_len_mm = st.sidebar.number_input("Reference Length (mm)", value=50.0)
ref_len_px = st.sidebar.number_input("Reference Length (px)", value=200.0)

# Tuning inputs
st.sidebar.subheader("Tuning")
min_area = st.sidebar.number_input("Min Pebble Area (px)", value=700)
morph_size = st.sidebar.slider("Morph Kernel Size", 3, 51, 25, step=2)

# 3. MAIN LOGIC
if uploaded_file is not None:
    # Convert the uploaded file into an image OpenCV can read
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Show original
    st.image(img, caption="Original Image", channels="BGR", use_column_width=True)

    if st.button("Analyze Now"):
        with st.spinner('Measuring pebbles...'):
            try:
                # A. Math & Processing
                pixels_per_mm = ref_len_px / ref_len_mm

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (7, 7), 2)
                _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))
                closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

                # B. Cleaning
                # Remove small noise and clear borders
                bin_clean = morphology.remove_small_objects(closed > 0, min_size=min_area)
                bin_clean = morphology.remove_small_holes(bin_clean, area_threshold=min_area)
                bin_clean = clear_border(bin_clean)
                binary_clean = (bin_clean * 255).astype(np.uint8)

                # C. Measure Contours
                contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                results = []
                img_disp = img.copy()
                pebble_id = 0

                for cnt in contours:
                    if cv2.contourArea(cnt) < min_area: continue
                    pebble_id += 1

                    # Ellipse Fit
                    if len(cnt) >= 5:
                        ellipse = cv2.fitEllipse(cnt)
                        (xc, yc), (major, minor), angle = ellipse

                        # Draw on image
                        cv2.ellipse(img_disp, ellipse, (0, 255, 255), 2)
                        cv2.putText(img_disp, str(pebble_id), (int(xc), int(yc)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                        # Save data
                        a_mm = (major / 2) / pixels_per_mm
                        b_mm = (minor / 2) / pixels_per_mm
                        Deq_mm = np.sqrt(a_mm * b_mm)
                        results.append(Deq_mm)

                # D. Show Results
                st.image(img_disp, caption=f"Found {pebble_id} pebbles", channels="BGR", use_column_width=True)

                if results:
                    d50 = np.median(results)
                    st.success(f"✅ **D50 Size:** {d50:.2f} mm")

                    # Create Table
                    df = pd.DataFrame(results, columns=["Deq (mm)"])
                    st.dataframe(df)

                    # Download Button
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Data (CSV)", csv, "results.csv", "text/csv")
                else:
                    st.warning("No pebbles found. Try lowering the 'Min Pebble Area'.")

            except Exception as e:
                st.error(f"Something went wrong: {e}")
