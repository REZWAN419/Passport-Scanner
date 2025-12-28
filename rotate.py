import cv2
import numpy as np
import math
import pytesseract
import easyocr
from concurrent.futures import ThreadPoolExecutor

# ============================================================
# ▶️ DEBUG UTILITY
# ============================================================

def debug_show(win, img, w=350, h=250):
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, w, h)
    cv2.imshow(win, img)

# =============================================
# ROTATE IMAGE BY TEXT GEOMETRY
# =============================================

# PROJECTION VARIANCE FOR A SINGLE ANGLE
def angle_score(angle, bw):
    h, w = bw.shape
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    rotated_bw = cv2.warpAffine(bw, M, (w, h),
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    hist = np.sum(rotated_bw, axis=1)
    return np.var(hist)


def rotate_image_by_text_geometry(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Failed to load image")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Downscale image for speed
    scale = 0.25
    small_gray = cv2.resize(gray, (0, 0), fx=scale, fy=scale)
    bw = cv2.adaptiveThreshold(small_gray, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 25, 15)

    
    debug_show("Original", img)
    debug_show("Binarized (downscaled)", bw)

    # ----------------------------
    # Coarse search (0-180°) in parallel
    # ----------------------------
    coarse_angles = np.arange(0, 180, 2.0)
    with ThreadPoolExecutor() as executor:
        coarse_scores = list(executor.map(lambda a: angle_score(a, bw), coarse_angles))
    coarse_best_angle = coarse_angles[np.argmax(coarse_scores)]

    # ----------------------------
    # Fine search ±2° around coarse angle in parallel
    # ----------------------------
    fine_angles = np.arange(coarse_best_angle-2, coarse_best_angle+2, 0.1)
    with ThreadPoolExecutor() as executor:
        fine_scores = list(executor.map(lambda a: angle_score(a, bw), fine_angles))
    best_angle = fine_angles[np.argmax(fine_scores)]
    print(f"📊 Detected skew angle: {best_angle:.2f}°")

    # Rotate full-resolution image
    h_full, w_full = img.shape[:2]
    M_final = cv2.getRotationMatrix2D((w_full//2, h_full//2), best_angle, 1.0)
    rotated = cv2.warpAffine(img, M_final, (w_full, h_full),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    
    debug_show("6 - Rotated Result", rotated)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return rotated

# ============================================================
# ▶️ DOCUMENT DETECTION
# ============================================================

def detect_document_candidate_debug(image):
    # Single page detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    debug_show("DOC 1 - Grayscale", gray)

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 7, 7, ksize=15)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 7, 7, ksize=15)
    gradient = cv2.magnitude(grad_x, grad_y)
    gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    debug_show("DOC 2 - Gradient Magnitude", gradient)

    blur = cv2.GaussianBlur(gradient, (15, 15), 0)
    debug_show("DOC 3 - Gradient Blurred", blur)

    edges = cv2.Canny(blur, 5, 5)
    debug_show("DOC 4 - Edges", edges)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    debug_show("DOC 5 - Morph Close", closed)

    filled = closed.copy()
    h, w = filled.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(filled, mask, (7, 7), 255)
    filled = cv2.bitwise_not(filled) | closed
    debug_show("DOC 6 - Filled Regions", filled)

    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    debug_contours = image.copy()
    min_area = 0.12 * h * w
    page = image.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if 4 <= len(approx) <= 8:
            x, y, wc, hc = cv2.boundingRect(approx)
            page = image[y:y + hc, x:x + wc]
            cv2.rectangle(debug_contours, (x, y), (x + wc, y + hc), (0, 255, 0), 3)
            break  # only one page

    debug_show("DOC 7 - Document Candidate", debug_contours)
    debug_show("DOC 8 - Extracted Page", page)
    return page

# ============================================================
# ▶️ FINAL ORIENTATION AND CROPPING BASED ON COLOR VARIANCE (MRZ)
# ============================================================
def detect_mrz_orientation_and_crop(page):
    h, w = page.shape[:2]
    page_type = "SINGLE" if w > h else "DOUBLE"

    # --- Saturation boost (for analysis only) ---
    hsv = cv2.cvtColor(page, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Adaptive saturation boost
    sat = hsv[:, :, 1]
    sat_mean = sat.mean()

    # Boost more if image is dull, less if already colorful
    boost = 3.2 if sat_mean < 100 else 1.05  # analysis-only boost
    hsv[:, :, 1] = np.clip(sat * boost, 0, 255)

    page = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    debug_show("Saturated Image", page)
    # --- Convert image to float for calculations ---
    img = page.astype(np.float32)

    # 1️⃣ Compute row-wise color intensity variance
    row_variances = []
    for y in range(h):
        row = img[y, :, :]
        variance = np.var(row, axis=0).mean()
        row_variances.append(variance)

    row_variances = np.array(row_variances)

    # Smooth variances (reduce noise & spikes)
    window = max(15, h // 100)
    smoothed_variances = np.zeros_like(row_variances)
    for y in range(h):
        start = max(0, y - window)
        end = min(h, y + window + 1)
        smoothed_variances[y] = np.mean(row_variances[start:end])

    # MRZ row = lowest smoothed variance (most sober region)
    mrz_row = np.argmin(smoothed_variances)

    # Orientation & cropping logic (UNCHANGED)
    if page_type == "SINGLE":
        if mrz_row < h // 2:
            print("➡️ SINGLE PAGE: MRZ on top → rotate 180°")
            final = cv2.rotate(page, cv2.ROTATE_180)
        else:
            print("➡️ SINGLE PAGE: MRZ on bottom → correct")
            final = page.copy()
    else:  # DOUBLE PAGE
        mid = h // 2
        if mrz_row < mid:
            print("➡️ DOUBLE PAGE: MRZ on top → crop upper half and rotate")
            final = cv2.rotate(page[:mid, :, :], cv2.ROTATE_180)
        else:
            print("➡️ DOUBLE PAGE: MRZ on bottom → crop lower half")
            final = page[mid:, :, :]

    
    # REMOVE SATURATION BEFORE RETURN
    # Convert final image back to original saturation by dividing S channel
    hsv_final = cv2.cvtColor(final, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_final[:, :, 1] = np.clip(hsv_final[:, :, 1] / boost, 0, 255)
    final = cv2.cvtColor(hsv_final.astype(np.uint8), cv2.COLOR_HSV2BGR)

    debug_show("FINAL PAGE", final, w=350, h=250)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return final

# ============================================================
# ▶️ RUN ENTIRE PIPELINE FOR SINGLE PAGE
# ============================================================

image_path = r"C:\Users\Lenovo\OneDrive\Desktop\Passport Scanner\PassportIMG\Screenshot1.png"  #tilted.jpg Screenshot1.png

rotated = rotate_image_by_text_geometry(image_path)
page = detect_document_candidate_debug(rotated)

h, w = page.shape[:2]
page_type = "SINGLE PAGE" if w > h else "DOUBLE PAGE"
debug = page.copy()
cv2.putText(debug, page_type, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
debug_show("PAGE TYPE", debug)
print(f"📄 Detected: {page_type} (w={w}, h={h})")

final_page = detect_mrz_orientation_and_crop(page)


# ============================================================
# ▶️ NON-MRZ OCR PIPELINE (TOP 75%)
# ============================================================

# ▶️ NON-MRZ STEP 1 — CROP TOP 75%

# DEBUG TOGGLE
DEBUG_NON_MRZ_SPLIT = True

# TUNING KNOB
NON_MRZ_RATIO = 0.75   # top 75% = non-MRZ zone

# CROP LOGIC
h, w = final_page.shape[:2]
non_mrz = final_page[:int(h * NON_MRZ_RATIO), :].copy()

if DEBUG_NON_MRZ_SPLIT:
    debug_show("NON-MRZ 0 - Top 75% Crop", non_mrz)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# ▶️ NON-MRZ STEP 2 — REMOVE MARGINS (GEOMETRIC CROP)

# DEBUG TOGGLE
DEBUG_NON_MRZ_MARGIN_CROP = True


# TUNING KNOBS (PERCENTAGES)
CROP_LEFT_RATIO   = 0.28   # remove 30% from left (portrait side)
CROP_TOP_RATIO    = 0.42   # remove 10% from top
CROP_RIGHT_RATIO  = 0.08   # remove 5% from right
CROP_BOTTOM_RATIO = 0.15   # remove 5% from bottom


# CROP LOGIC
h_nm, w_nm = non_mrz.shape[:2]

x1 = int(w_nm * CROP_LEFT_RATIO)
y1 = int(h_nm * CROP_TOP_RATIO)

x2 = int(w_nm * (1.0 - CROP_RIGHT_RATIO))
y2 = int(h_nm * (1.0 - CROP_BOTTOM_RATIO))

non_mrz_cropped = non_mrz[y1:y2, x1:x2].copy()


# DEBUG VIEW
if DEBUG_NON_MRZ_MARGIN_CROP:
    debug_show("NON-MRZ 1 - Margin Cropped", non_mrz_cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# ▶️ NON-MRZ STEP 3 — WHITE FILL UNWANTED AREAS

# DEBUG TOGGLE
DEBUG_NON_MRZ_WHITE_MASK = True

# ------------------------------------------------------------
# TUNING KNOBS — WHITE MASK BOXES (RATIOS)
# Each box defined as: (x_start, y_start, width, height)
# All values are ratios (0.0 – 1.0) relative to image size
# ------------------------------------------------------------

WHITE_MASK_BOXES = [
    # Box 1 — example: top-left noise area
    (0.00, 0.00, 0.35, 0.70),

    # Box 2 — example: middle decorative area
    (0.00, 0.53, 1.00, 0.21),

    # Box 3 — example: bottom-left emblem/text
    (0.90, 0.65, 0.30, 0.40),
]


# APPLY WHITE MASKS
masked_non_mrz = non_mrz_cropped.copy()
h_nm, w_nm = masked_non_mrz.shape[:2]

for (rx, ry, rw, rh) in WHITE_MASK_BOXES:
    x1 = int(rx * w_nm)
    y1 = int(ry * h_nm)
    x2 = int((rx + rw) * w_nm)
    y2 = int((ry + rh) * h_nm)

    cv2.rectangle(
        masked_non_mrz,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        thickness=-1
    )

# DEBUG VIEW
if DEBUG_NON_MRZ_WHITE_MASK:
    debug_show("NON-MRZ 2 - White Masked Areas", masked_non_mrz)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================
# ▶️ NON-MRZ STEP 4 — PREPROCESS FOR EASYOCR
# ============================================================

# -------------------------------
# DEBUG TOGGLE
# -------------------------------
DEBUG_NON_MRZ_PREOCR = True

# -------------------------------
# TUNING KNOBS
# -------------------------------
CLAHE_CLIP_LIMIT = 2.0      # higher = more contrast
CLAHE_TILE_SIZE  = 8        # local region
DENOISE_STRENGTH = 15        # 0=off, 5-12 typical
SHARPEN_AMOUNT  = 0.7       # 0=off, 0.5-1.2 typical

# -------------------------------
# GRAYSCALE
# -------------------------------
nm_gray = cv2.cvtColor(masked_non_mrz, cv2.COLOR_BGR2GRAY)

# -------------------------------
# CLAHE (LOCAL CONTRAST ENHANCEMENT)
# -------------------------------
clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(CLAHE_TILE_SIZE, CLAHE_TILE_SIZE))
nm_clahe = clahe.apply(nm_gray)

# -------------------------------
# DENOISE
# -------------------------------
if DENOISE_STRENGTH > 0:
    nm_denoised = cv2.fastNlMeansDenoising(nm_clahe, h=DENOISE_STRENGTH, templateWindowSize=7, searchWindowSize=21)
else:
    nm_denoised = nm_clahe.copy()

# -------------------------------
# LIGHT SHARPEN
# -------------------------------
if SHARPEN_AMOUNT > 0:
    blur = cv2.GaussianBlur(nm_denoised, (0,0), 2.0)
    nm_preocr = cv2.addWeighted(nm_denoised, 1+SHARPEN_AMOUNT, blur, -SHARPEN_AMOUNT, 0)
else:
    nm_preocr = nm_denoised.copy()

# -------------------------------
# DEBUG VIEW
# -------------------------------
if DEBUG_NON_MRZ_PREOCR:
    debug_show("NON-MRZ Preprocessed for OCR", nm_preocr, w=900, h=600)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ============================================================
# ▶️ EASY OCR ON PREPROCESSED IMAGE
# ============================================================

DEBUG_NON_MRZ_OCR = True
EASYOCR_LANG     = ['en']
EASYOCR_GPU      = False
EASYOCR_MIN_CONF = 0.25
EASYOCR_PARAGRAPH = False

reader = easyocr.Reader(EASYOCR_LANG, gpu=EASYOCR_GPU, verbose=False)

ocr_results = reader.readtext(
    nm_preocr,
    detail=1,
    paragraph=EASYOCR_PARAGRAPH
)

# -------------------------------
# DRAW DETECTIONS (DEBUG)
# -------------------------------
ocr_debug = cv2.cvtColor(nm_preocr, cv2.COLOR_GRAY2BGR)
for box, text, conf in ocr_results:
    if conf < EASYOCR_MIN_CONF:
        continue
    pts = np.array(box).astype(int)
    cv2.polylines(ocr_debug, [pts], isClosed=True, color=(0,255,0), thickness=2)
    x, y = pts[0]
    label = f"{text} ({conf:.2f})"
    cv2.putText(ocr_debug, label, (x, max(20,y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

if DEBUG_NON_MRZ_OCR:
    debug_show("NON-MRZ OCR Detection", ocr_debug, w=900, h=600)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -------------------------------
# FINAL OCR OUTPUT
# -------------------------------
non_mrz_ocr_results = [(text, conf, box) for box, text, conf in ocr_results if conf >= EASYOCR_MIN_CONF]

# -------------------------------
# PRINT TO TERMINAL
# -------------------------------
print("📄 NON-MRZ OCR RESULTS:")
for idx, (text, conf, box) in enumerate(non_mrz_ocr_results, start=1):
    print(f"{idx}. {text} ({conf:.2f})")
