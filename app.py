import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import cv2
from PIL import Image, ImageOps
from skimage import morphology
from skimage.segmentation import clear_border
from streamlit_image_coordinates import streamlit_image_coordinates
import matplotlib.pyplot as plt


# ============================================================
# Helpers
# ============================================================
def odd_clip(x, lo=3, hi=51):
    x = int(round(float(x)))
    x = max(lo, min(hi, x))
    if x % 2 == 0:
        x = x + 1 if x < hi else x - 1
    return x

def decode_uploaded_image(uploaded_file):
    raw = uploaded_file.read()
    pil = Image.open(io.BytesIO(raw))
    pil = ImageOps.exif_transpose(pil).convert("RGB")  # fixes phone rotation
    rgb = np.array(pil)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr, pil

def threshold_otsu(gray, invert=True):
    blur = cv2.GaussianBlur(gray, (7, 7), 2)
    ttype = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(blur, 0, 255, ttype + cv2.THRESH_OTSU)
    return binary

def as_bytes_png(img_bgr):
    ok, buf = cv2.imencode(".png", img_bgr)
    return buf.tobytes() if ok else None

def dataframe_to_excel_bytes(df_pergrain, df_summary):
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as xlw:
        df_pergrain.to_excel(xlw, sheet_name="per_grain", index=False)
        df_summary.to_excel(xlw, sheet_name="summary", index=False)
    out.seek(0)
    return out.getvalue()

def safe_numeric_series(s):
    return pd.to_numeric(s, errors="coerce")

def fit_calibration(x_img, y_cal):
    """
    Calibrate: y = a + b*x
    - If <5 pairs: ratio (a=0, b=median(y/x))
    - Else: linear regression using polyfit
    Returns (a, b, method, r2, rmse)
    """
    x = np.asarray(x_img, dtype=float)
    y = np.asarray(y_cal, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[mask]; y = y[mask]
    if x.size < 2:
        return np.nan, np.nan, "insufficient", np.nan, np.nan

    if x.size < 5:
        b = float(np.median(y / x))
        a = 0.0
        yhat = a + b * x
        rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
        r2 = 1.0 - float(np.sum((y - yhat) ** 2) / np.sum((y - np.mean(y)) ** 2)) if np.sum((y - np.mean(y)) ** 2) > 0 else np.nan
        return a, b, "ratio", r2, rmse

    b, a = np.polyfit(x, y, 1)  # slope, intercept
    a = float(a); b = float(b)
    yhat = a + b * x
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum((y - yhat) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    return a, b, "linear", r2, rmse

def plot_hist_overlay(before, after, bins=30):
    before = safe_numeric_series(before).dropna().to_numpy()
    after = safe_numeric_series(after).dropna().to_numpy()

    fig = plt.figure(figsize=(7, 4.5))
    plt.hist(before, bins=bins, alpha=0.5, label="Raw (image)")
    plt.hist(after, bins=bins, alpha=0.5, label="Calibrated")
    plt.xlabel("Deq (mm)")
    plt.ylabel("Count")
    plt.title("Grain-size distribution (Histogram)")
    plt.legend()
    plt.tight_layout()
    return fig

def plot_cdf_overlay(before, after):
    b = safe_numeric_series(before).dropna().to_numpy()
    a = safe_numeric_series(after).dropna().to_numpy()

    b.sort(); a.sort()
    yb = np.linspace(0, 1, b.size, endpoint=True) if b.size else np.array([])
    ya = np.linspace(0, 1, a.size, endpoint=True) if a.size else np.array([])

    fig = plt.figure(figsize=(7, 4.5))
    if b.size:
        plt.plot(b, yb, label="Raw (image)")
    if a.size:
        plt.plot(a, ya, label="Calibrated")
    plt.xlabel("Deq (mm)")
    plt.ylabel("CDF")
    plt.title("Grain-size distribution (CDF)")
    plt.legend()
    plt.tight_layout()
    return fig

def compute_d_stats(values):
    v = safe_numeric_series(values).dropna().to_numpy()
    if v.size == 0:
        return {"D10": np.nan, "D50": np.nan, "D90": np.nan, "N": 0}
    return {
        "D10": float(np.quantile(v, 0.10)),
        "D50": float(np.quantile(v, 0.50)),
        "D90": float(np.quantile(v, 0.90)),
        "N": int(v.size),
    }

def read_any_table(uploaded):
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded)
    return pd.read_csv(uploaded)

def compute_deq_from_table(df):
    """
    Try to find Deq column or major/minor diameters.
    If a_mm/b_mm exist (semi-axes), convert to diameter by 2*sqrt(a*b).
    """
    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]
    cols_l = [c.lower() for c in d.columns]

    def col_match(cands):
        for cand in cands:
            cl = cand.lower()
            if cl in cols_l:
                return d.columns[cols_l.index(cl)]
            for i, cc in enumerate(cols_l):
                if cl in cc:
                    return d.columns[i]
        return None

    deq = col_match(["Deq_mm", "deq", "equiv", "equivalent"])
    if deq is not None:
        out = d.copy()
        out["Deq_mm"] = safe_numeric_series(out[deq])
        return out

    maj = col_match(["major_diam_mm", "major", "a-axis", "long", "l_mm"])
    minr = col_match(["minor_diam_mm", "minor", "b-axis", "short", "i_mm"])
    if maj is not None and minr is not None:
        out = d.copy()
        A = safe_numeric_series(out[maj])
        B = safe_numeric_series(out[minr])
        out["Deq_mm"] = np.sqrt(A * B)
        return out

    a = col_match(["a_mm", "a (mm)"])
    b = col_match(["b_mm", "b (mm)"])
    if a is not None and b is not None:
        out = d.copy()
        A = safe_numeric_series(out[a])
        B = safe_numeric_series(out[b])
        out["Deq_mm"] = 2.0 * np.sqrt(A * B)
        return out

    out = d.copy()
    out["Deq_mm"] = np.nan
    return out


# ============================================================
# App state
# ============================================================
st.set_page_config(page_title="PyGrain Field", page_icon="🪨", layout="wide")
st.title("🪨 PyGrain Field App")
st.caption("Tap-to-calibrate scale • automatic min area • in-field caliper calibration • distribution graphs • exports")

if "clicks" not in st.session_state:
    st.session_state.clicks = []
if "last_click" not in st.session_state:
    st.session_state.last_click = None
if "px_per_mm" not in st.session_state:
    st.session_state.px_per_mm = None
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None
if "img_annot_bgr" not in st.session_state:
    st.session_state.img_annot_bgr = None
if "summary_raw" not in st.session_state:
    st.session_state.summary_raw = None


tab1, tab2 = st.tabs(["🧪 Field Analysis + Calibration", "📏 Validation (Files)"])


# ============================================================
# TAB 1: FIELD ANALYSIS + IN-FIELD CALIBRATION
# ============================================================
with tab1:
    st.sidebar.header("1) Upload")
    uploaded = st.sidebar.file_uploader("Upload field photo (JPG/PNG)", type=["jpg", "jpeg", "png"], key="img_up")

    st.sidebar.header("2) Segmentation")
    invert = st.sidebar.checkbox(
        "Invert threshold (pebbles darker than background)",
        value=True,
        help="If background is detected as pebbles, turn this OFF."
    )
    remove_border_opt = st.sidebar.checkbox(
        "Remove border-touching objects",
        value=True,
        help="Turn OFF if you want to keep pebbles touching the image edges."
    )

    st.sidebar.header("3) Size filters (field-friendly)")
    min_diam_mm = st.sidebar.number_input(
        "Minimum pebble diameter (mm)",
        value=4.0, min_value=0.1, step=0.5,
        help="Smallest grain you want to detect (auto converts to pixel area threshold)."
    )
    max_area_mm2 = st.sidebar.number_input(
        "Maximum pebble area (mm²) (optional)",
        value=0.0, min_value=0.0, step=100.0,
        help="Set 0 to disable. Useful to reject large shadows/objects."
    )

    st.sidebar.header("4) Morphology / Separation")
    auto_kernel = st.sidebar.checkbox(
        "Auto kernel (recommended)",
        value=True,
        help="Auto chooses kernel based on scale + minimum diameter."
    )
    kernel_manual = st.sidebar.slider(
        "Manual kernel size (odd)",
        3, 51, 25, step=2,
        help="Increase if pebbles merge; decrease if edges over-smooth."
    )

    st.sidebar.header("5) CSI (optional thickness)")
    use_thickness = st.sidebar.checkbox(
        "Use thickness ratio for CSI",
        value=False,
        help="If you assume thickness c ≈ k×minor_diameter, enable and set k."
    )
    thickness_ratio = None
    if use_thickness:
        thickness_ratio = st.sidebar.slider("k (c ≈ k×minor)", 0.1, 1.0, 0.6, step=0.05)

    st.sidebar.markdown("---")
    cA, cB = st.sidebar.columns(2)
    with cA:
        if st.button("Undo tap"):
            if len(st.session_state.clicks) > 0:
                st.session_state.clicks.pop()
            st.session_state.px_per_mm = None
    with cB:
        if st.button("Clear taps"):
            st.session_state.clicks = []
            st.session_state.px_per_mm = None
            st.session_state.last_click = None

    if uploaded is None:
        st.info("👈 Upload an image to begin. Include a ruler/coin/card in the same plane as pebbles.")
        st.stop()

    img_bgr, pil_rgb = decode_uploaded_image(uploaded)
    H, W = img_bgr.shape[:2]

    # ---------------- Scale: tap two points ----------------
    st.subheader("Step 1 — Tap-to-calibrate scale")
    st.write("Tap **two points** on your reference object (e.g., 0–100 mm on a ruler). Then enter the known length (mm).")

    display_w = min(900, W)
    scale_factor = W / display_w

    coords = streamlit_image_coordinates(pil_rgb, width=display_w, key="tap_img")

    if coords and "x" in coords and "y" in coords:
        pt = (coords["x"], coords["y"])
        if st.session_state.last_click != pt:
            st.session_state.clicks.append(pt)
            st.session_state.clicks = st.session_state.clicks[-2:]
            st.session_state.last_click = pt

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.write("**Taps stored:**", len(st.session_state.clicks))

    ref_len_px = 0.0
    if len(st.session_state.clicks) == 2:
        (x1d, y1d), (x2d, y2d) = st.session_state.clicks
        dx = (x2d - x1d) * scale_factor
        dy = (y2d - y1d) * scale_factor
        ref_len_px = float(np.sqrt(dx*dx + dy*dy))

    with col2:
        st.metric("Reference length (px)", f"{ref_len_px:.1f}" if ref_len_px > 0 else "—")

    with col3:
        ref_len_mm = st.number_input(
            "Known reference length (mm)",
            value=100.0, min_value=0.1,
            help="Example: if you tapped 0 to 10 cm on ruler, enter 100."
        )

    if ref_len_px > 0:
        st.session_state.px_per_mm = ref_len_px / ref_len_mm
        st.success(f"✅ Scale set: 1 mm = **{st.session_state.px_per_mm:.3f} px**")
    else:
        st.warning("Tap two points to set the scale.")

    if st.session_state.px_per_mm is None:
        st.stop()

    px_per_mm = st.session_state.px_per_mm

    # ---------------- Auto thresholds ----------------
    min_area_mm2 = math.pi * (min_diam_mm / 2.0) ** 2
    min_area_px = int(max(1, round(min_area_mm2 * (px_per_mm ** 2))))

    if max_area_mm2 > 0:
        max_area_px = float(max_area_mm2 * (px_per_mm ** 2))
    else:
        max_area_px = 1e18

    min_diam_px = min_diam_mm * px_per_mm
    kernel_suggest = odd_clip(0.6 * min_diam_px, lo=3, hi=51)
    kernel_size = kernel_suggest if auto_kernel else kernel_manual

    st.subheader("Step 2 — Auto settings preview")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Min diam (px)", f"{min_diam_px:.1f}")
    m2.metric("Min area (px²)", f"{min_area_px}")
    m3.metric("Kernel used", f"{kernel_size}")
    m4.metric("px per mm", f"{px_per_mm:.3f}")

    with st.expander("Kernel quick guidance"):
        st.markdown(
            "- **Merged pebbles** → increase kernel.\n"
            "- **Over-smoothed edges / sizes shrink** → decrease kernel.\n"
            "- Auto kernel is recommended in field conditions."
        )

    # ---------------- Run analysis ----------------
    st.subheader("Step 3 — Run analysis")
    run = st.button("Run Analysis", type="primary", use_container_width=True)

    if run:
        with st.spinner("Processing..."):
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            binary = threshold_otsu(gray, invert=invert)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            bw = closed > 0
            bw = morphology.remove_small_objects(bw, min_size=min_area_px)
            bw = morphology.remove_small_holes(bw, area_threshold=min_area_px)
            if remove_border_opt:
                bw = clear_border(bw)

            binary_clean = (bw.astype(np.uint8) * 255)

            contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            img_disp = img_bgr.copy()
            results = []
            pebble_id = 0

            for cnt in contours:
                area_px = cv2.contourArea(cnt)
                if area_px < min_area_px or area_px > max_area_px:
                    continue
                if len(cnt) < 5:
                    continue

                ellipse = cv2.fitEllipse(cnt)
                (xc, yc), (major_px, minor_px), angle = ellipse  # OpenCV returns DIAMETERS in pixels

                pebble_id += 1

                major_mm = major_px / px_per_mm
                minor_mm = minor_px / px_per_mm

                perimeter_px = cv2.arcLength(cnt, True)
                perimeter_mm = perimeter_px / px_per_mm if perimeter_px else np.nan
                area_mm2 = area_px / (px_per_mm ** 2)

                roundness = (minor_mm / major_mm) if (major_mm > 0) else np.nan

                # Equivalent diameter (correct for diameters):
                # area ellipse = π*(major/2)*(minor/2) => Deq = sqrt(major*minor)
                deq_mm = np.sqrt(major_mm * minor_mm) if (major_mm > 0 and minor_mm > 0) else np.nan

                # CSI proxy
                csi = np.sqrt(minor_mm / major_mm) if (major_mm > 0 and minor_mm > 0) else np.nan
                if thickness_ratio is not None and np.isfinite(csi):
                    csi = thickness_ratio * csi

                # Draw annotation
                cv2.drawContours(img_disp, [cnt], -1, (0, 255, 0), 2)
                cv2.ellipse(img_disp, ellipse, (0, 255, 255), 1)
                cv2.putText(
                    img_disp, str(pebble_id), (int(xc) - 12, int(yc) + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA
                )

                results.append({
                    "Pebble_ID": pebble_id,
                    "Major_diam_mm": major_mm,
                    "Minor_diam_mm": minor_mm,
                    "Area_mm2": area_mm2,
                    "Perimeter_mm": perimeter_mm,
                    "Roundness_(minor/major)": roundness,
                    "Corey_Shape_Index": csi,
                    "Deq_mm": deq_mm
                })

            df = pd.DataFrame(results)

            st.session_state.df_raw = df
            st.session_state.img_annot_bgr = img_disp

            if df.empty or df["Deq_mm"].dropna().empty:
                st.warning("No pebbles detected. Try lowering min diameter, toggling invert, or adjusting kernel.")
                with st.expander("Debug images"):
                    st.image(binary, channels="GRAY", caption="Otsu binary")
                    st.image(binary_clean, channels="GRAY", caption="Cleaned binary")
                st.stop()

            stats_raw = compute_d_stats(df["Deq_mm"])
            st.session_state.summary_raw = stats_raw

            st.image(img_disp, channels="BGR", caption=f"Detected pebbles: {stats_raw['N']}")

            st.success(
                f"✅ Raw distribution: D10={stats_raw['D10']:.2f} mm | "
                f"D50={stats_raw['D50']:.2f} mm | D90={stats_raw['D90']:.2f} mm | N={stats_raw['N']}"
            )

            with st.expander("Raw results table"):
                st.dataframe(df, use_container_width=True)

            # Downloads: raw
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            summary = pd.DataFrame([{
                "N": stats_raw["N"],
                "D10_mm": stats_raw["D10"],
                "D50_mm": stats_raw["D50"],
                "D90_mm": stats_raw["D90"],
                "px_per_mm": float(px_per_mm),
                "min_diameter_mm": float(min_diam_mm),
                "min_area_px2": int(min_area_px),
                "kernel_size": int(kernel_size),
                "invert": bool(invert),
                "remove_border": bool(remove_border_opt),
                "thickness_ratio_used": float(thickness_ratio) if thickness_ratio is not None else np.nan,
            }])

            excel_bytes = dataframe_to_excel_bytes(df, summary)
            img_bytes = as_bytes_png(img_disp)

            d1, d2, d3 = st.columns(3)
            with d1:
                st.download_button("Download RAW CSV", csv_bytes, "pygrain_raw.csv", "text/csv", use_container_width=True)
            with d2:
                st.download_button(
                    "Download RAW Excel", excel_bytes, "pygrain_raw.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            with d3:
                if img_bytes:
                    st.download_button("Download Annotated Image", img_bytes, "pygrain_annotated.png", "image/png", use_container_width=True)

            with st.expander("Debug images"):
                st.image(binary, channels="GRAY", caption="Otsu binary")
                st.image(binary_clean, channels="GRAY", caption="Cleaned binary")

    # ---------------- In-field calibration (caliper) ----------------
    if st.session_state.df_raw is not None and st.session_state.df_raw.shape[0] > 0:
        df = st.session_state.df_raw.copy()

        st.markdown("---")
        st.subheader("Step 4 — In-field calibration using caliper (optional)")
        st.write(
            "Measure **some** pebbles in the field using caliper, using the **Pebble_ID** labels on the annotated image. "
            "Enter the caliper measurements, then the app will calibrate the entire distribution and generate graphs + calibrated CSV/Excel."
        )

        with st.expander("Enter caliper measurements (Pebble_ID matched)"):
            st.write("You can either upload a CSV or enter manually.")
            st.write("Recommended: **10–30 pebbles**, covering small–medium–large sizes.")

            method = st.radio(
                "Calibration model",
                ["Auto (ratio if <5 pairs, linear otherwise)", "Force Ratio (0 intercept)", "Force Linear (a + b×x)"],
                horizontal=True
            )

            cal_csv = st.file_uploader(
                "Upload Caliper CSV (optional)",
                type=["csv"],
                help="CSV columns supported: Pebble_ID + (Deq_caliper_mm) OR Pebble_ID + (L_mm, I_mm).",
                key="cal_csv_up"
            )

            # Manual entry
            st.write("Manual entry (you can add rows):")
            template = pd.DataFrame({"Pebble_ID": [1, 2, 3], "L_mm": [20.0, 35.0, 15.0], "I_mm": [14.0, 25.0, 10.0]})
            entered = st.data_editor(template, num_rows="dynamic", use_container_width=True, key="cal_editor")

            # Build pairs dataframe
            pairs = None
            if cal_csv is not None:
                tmp = pd.read_csv(cal_csv)
                tmp.columns = [c.strip() for c in tmp.columns]
                pairs = tmp.copy()
            else:
                pairs = entered.copy()

            # Normalize/compute Deq_caliper_mm
            for col in ["Pebble_ID", "L_mm", "I_mm", "Deq_caliper_mm"]:
                if col not in pairs.columns:
                    pairs[col] = np.nan

            pairs["Pebble_ID"] = safe_numeric_series(pairs["Pebble_ID"])
            pairs["L_mm"] = safe_numeric_series(pairs["L_mm"])
            pairs["I_mm"] = safe_numeric_series(pairs["I_mm"])
            pairs["Deq_caliper_mm"] = safe_numeric_series(pairs["Deq_caliper_mm"])

            # If Deq_caliper_mm missing but L/I available, compute Deq = sqrt(L*I)
            need = pairs["Deq_caliper_mm"].isna() & pairs["L_mm"].notna() & pairs["I_mm"].notna()
            pairs.loc[need, "Deq_caliper_mm"] = np.sqrt(pairs.loc[need, "L_mm"] * pairs.loc[need, "I_mm"])

            pairs = pairs[["Pebble_ID", "Deq_caliper_mm"]].dropna()
            pairs = pairs[(pairs["Pebble_ID"] >= 1) & (pairs["Deq_caliper_mm"] > 0)]

            if pairs.shape[0] < 2:
                st.info("Enter at least **2 matched** pebbles to calibrate (Pebble_ID + caliper Deq).")
            else:
                df["Pebble_ID"] = safe_numeric_series(df["Pebble_ID"])
                merged = df.merge(pairs, on="Pebble_ID", how="inner")

                if merged.shape[0] < 2:
                    st.warning("No matching Pebble_ID found between image detections and caliper entries.")
                else:
                    x = merged["Deq_mm"].to_numpy()
                    y = merged["Deq_caliper_mm"].to_numpy()

                    if method == "Force Ratio (0 intercept)":
                        b = float(np.median(y / x))
                        a = 0.0
                        used = "ratio"
                        yhat = a + b * x
                        rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
                        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                        ss_res = float(np.sum((y - yhat) ** 2))
                        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
                    elif method == "Force Linear (a + b×x)":
                        b_fit, a_fit = np.polyfit(x, y, 1)
                        a = float(a_fit); b = float(b_fit)
                        used = "linear"
                        yhat = a + b * x
                        rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
                        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                        ss_res = float(np.sum((y - yhat) ** 2))
                        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
                    else:
                        a, b, used, r2, rmse = fit_calibration(x, y)

                    if not np.isfinite(a) or not np.isfinite(b):
                        st.error("Calibration failed. Check your caliper inputs.")
                    else:
                        st.success(f"✅ Calibration: **Deq_calibrated = {a:.3f} + {b:.3f} × Deq_raw**  (method: {used})")

                        st.write(f"Pairs used: **{merged.shape[0]}** | RMSE: **{rmse:.3f} mm** | R²: **{r2:.3f}**" if np.isfinite(r2) else f"Pairs used: **{merged.shape[0]}** | RMSE: **{rmse:.3f} mm**")

                        df["Deq_calibrated_mm"] = (a + b * df["Deq_mm"]).clip(lower=0)

                        raw_stats = compute_d_stats(df["Deq_mm"])
                        cal_stats = compute_d_stats(df["Deq_calibrated_mm"])

                        s1, s2 = st.columns(2)
                        with s1:
                            st.info(f"Raw: D10={raw_stats['D10']:.2f}, D50={raw_stats['D50']:.2f}, D90={raw_stats['D90']:.2f} (N={raw_stats['N']})")
                        with s2:
                            st.success(f"Calibrated: D10={cal_stats['D10']:.2f}, D50={cal_stats['D50']:.2f}, D90={cal_stats['D90']:.2f} (N={cal_stats['N']})")

                        # Graphs
                        g1, g2 = st.columns(2)
                        with g1:
                            fig = plot_hist_overlay(df["Deq_mm"], df["Deq_calibrated_mm"], bins=30)
                            st.pyplot(fig)
                        with g2:
                            fig = plot_cdf_overlay(df["Deq_mm"], df["Deq_calibrated_mm"])
                            st.pyplot(fig)

                        # Downloads
                        cal_summary = pd.DataFrame([{
                            "pairs_used": int(merged.shape[0]),
                            "calibration_method": used,
                            "a_intercept": a,
                            "b_slope": b,
                            "rmse_mm": rmse,
                            "r2": r2,
                            "D50_raw_mm": raw_stats["D50"],
                            "D50_cal_mm": cal_stats["D50"],
                            "D10_raw_mm": raw_stats["D10"],
                            "D10_cal_mm": cal_stats["D10"],
                            "D90_raw_mm": raw_stats["D90"],
                            "D90_cal_mm": cal_stats["D90"],
                        }])

                        cal_csv_bytes = df.to_csv(index=False).encode("utf-8")
                        cal_excel_bytes = dataframe_to_excel_bytes(df, cal_summary)

                        d1, d2 = st.columns(2)
                        with d1:
                            st.download_button("Download CALIBRATED CSV", cal_csv_bytes, "pygrain_calibrated.csv", "text/csv", use_container_width=True)
                        with d2:
                            st.download_button(
                                "Download CALIBRATED Excel", cal_excel_bytes, "pygrain_calibrated.xlsx",
                                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )


# ============================================================
# TAB 2: VALIDATION USING FILES (HAND vs PY)
# ============================================================
with tab2:
    st.subheader("Validation using files (HAND vs PY)")
    st.write("Upload HAND and PY tables (CSV/XLSX). The app will standardize Deq and compare distributions.")

    c1, c2 = st.columns(2)
    with c1:
        hand_up = st.file_uploader("Upload HAND table (CSV/XLSX)", type=["csv", "xlsx", "xls"], key="hand_up")
    with c2:
        py_up = st.file_uploader("Upload PY table (CSV/XLSX)", type=["csv", "xlsx", "xls"], key="py_up")

    if hand_up and py_up:
        hand_raw = read_any_table(hand_up)
        py_raw = read_any_table(py_up)

        hand = compute_deq_from_table(hand_raw)
        py = compute_deq_from_table(py_raw)

        hs = compute_d_stats(hand["Deq_mm"])
        ps = compute_d_stats(py["Deq_mm"])

        bias = ps["D50"] - hs["D50"] if np.isfinite(ps["D50"]) and np.isfinite(hs["D50"]) else np.nan

        m1, m2, m3 = st.columns(3)
        m1.metric("HAND D50 (mm)", f"{hs['D50']:.2f}" if np.isfinite(hs["D50"]) else "—")
        m2.metric("PY D50 (mm)", f"{ps['D50']:.2f}" if np.isfinite(ps["D50"]) else "—")
        m3.metric("Bias (PY - HAND)", f"{bias:.2f} mm" if np.isfinite(bias) else "—")

        g1, g2 = st.columns(2)
        with g1:
            fig = plot_hist_overlay(hand["Deq_mm"], py["Deq_mm"], bins=30)
            plt.title("Histogram: HAND vs PY")
            st.pyplot(fig)
        with g2:
            fig = plot_cdf_overlay(hand["Deq_mm"], py["Deq_mm"])
            plt.title("CDF: HAND vs PY")
            st.pyplot(fig)

        with st.expander("Preview standardized tables (first 20 rows)"):
            st.write("HAND (with Deq_mm):")
            st.dataframe(hand.head(20), use_container_width=True)
            st.write("PY (with Deq_mm):")
            st.dataframe(py.head(20), use_container_width=True)
    else:
        st.info("Upload both HAND and PY files to validate.")
