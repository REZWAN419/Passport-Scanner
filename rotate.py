import cv2
import numpy as np
import math
import pytesseract
import easyocr

# ============================================================
# ▶️ DEBUG UTILITY
# ============================================================

def debug_show(win, img, w=350, h=250):
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, w, h)
    cv2.imshow(win, img)

# ================================
# FAST ROTATE IMAGE BY TEXT GEOMETRY
# ================================
def rotate_image_by_text_geometry(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Failed to load image")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Downscale image for speed
    scale = 0.25  # 25% size
    small_gray = cv2.resize(gray, (0, 0), fx=scale, fy=scale)
    bw = cv2.adaptiveThreshold(small_gray, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 25, 15)

    
    debug_show("Original", img)
    debug_show("Binarized (downscaled)", bw)

    # 1️⃣ Coarse search (every 2°)
    coarse_angles = np.arange(0, 180, 2.0)
    coarse_scores = []
    h, w = bw.shape
    for angle in coarse_angles:
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        rotated_bw = cv2.warpAffine(bw, M, (w, h),
                                    flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        hist = np.sum(rotated_bw, axis=1)
        coarse_scores.append(np.var(hist))
    coarse_best_angle = coarse_angles[np.argmax(coarse_scores)]

    # 2️⃣ Fine search ±2° around coarse angle with 0.1° steps
    fine_angles = np.arange(coarse_best_angle-2, coarse_best_angle+2, 0.1)
    fine_scores = []
    for angle in fine_angles:
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        rotated_bw = cv2.warpAffine(bw, M, (w, h),
                                    flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        hist = np.sum(rotated_bw, axis=1)
        fine_scores.append(np.var(hist))
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

image_path = r"C:\Users\Lenovo\OneDrive\Desktop\Passport Scanner\PassportIMG\Screenshot5.png"  #tilted.jpg Screenshot1.png

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
CROP_TOP_RATIO    = 0.40   # remove 10% from top
CROP_RIGHT_RATIO  = 0.10   # remove 5% from right
CROP_BOTTOM_RATIO = 0.07   # remove 5% from bottom


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




# NON-MRZ STEP 3 — KEEP ONLY BLACK PIXELS

# DEBUG TOGGLE
DEBUG_NON_MRZ_BLACK_ONLY = True


# TUNING KNOB
BLACK_PIXEL_MAX_INTENSITY = 80   # any pixel darker than this is black, else white


# CREATE BLACK-ONLY IMAGE
# start with white image
black_only_nm = np.ones_like(non_mrz_cropped) * 255

# mask pixels where all channels are below threshold
mask_black = np.all(non_mrz_cropped <= BLACK_PIXEL_MAX_INTENSITY, axis=2)
black_only_nm[mask_black] = 0


# DEBUG VIEW
if DEBUG_NON_MRZ_BLACK_ONLY:
    debug_show("NON-MRZ 2 - Pure Black Pixels", black_only_nm)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
