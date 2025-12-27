import cv2
import numpy as np
import math
import pytesseract

# ============================================================
# ▶️ DEBUG UTILITY
# ============================================================

def debug_show(win, img, w=350, h=250):
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, w, h)
    cv2.imshow(win, img)

# ============================================================
# ▶️ ROTATION BASED ON TEXT GEOMETRY
# ============================================================

def show_image_rotated_by_text_geometry_debug(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Failed to load image")
        return

    debug_show("1 - Original", image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    debug_show("2 - Grayscale", gray)

    blur = cv2.GaussianBlur(gray, (5, 5), 10)  #5, 5), 10
    debug_show("3 - Blurred", blur)

    edges = cv2.Canny(blur, 75, 130, apertureSize=3) # 75, 150
    debug_show("4 - Edges", edges)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    line_img = image.copy()
    angles = []

    if lines is not None:
        for line in lines[:50]:
            rho, theta = line[0]
            angle = (theta - np.pi / 2) * 180 / np.pi
            angles.append(angle)

            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 1)

        print(f"✅ Detected median angle: {np.median(angles):.2f}°")
    else:
        print("⚠️ No lines detected")

    debug_show("5 - Text Lines", line_img)

    angle = np.median(angles) if angles else 0
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

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

    # Convert image to float for calculations
    img = page.astype(np.float32)

    # 1️⃣ Compute row-wise color intensity variance
    row_variances = []
    for y in range(h):
        row = img[y, :, :]  # entire row
        variance = np.var(row, axis=0).mean()  # mean variance over R,G,B
        row_variances.append(variance)

    row_variances = np.array(row_variances)

    # Smooth variances (reduce noise & spikes)
    window = max(15, h // 100)  # adaptive smoothing window
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

    debug_show("FINAL PAGE", final, w=350, h=250)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return final

# ============================================================
# ▶️ RUN ENTIRE PIPELINE FOR SINGLE PAGE
# ============================================================

image_path = r"C:\Users\Lenovo\OneDrive\Desktop\Passport Scanner\PassportIMG\tilt1.jpg"  #tilted.jpg Screenshot1.png

rotated = show_image_rotated_by_text_geometry_debug(image_path)
page = detect_document_candidate_debug(rotated)

h, w = page.shape[:2]
page_type = "SINGLE PAGE" if w > h else "DOUBLE PAGE"
debug = page.copy()
cv2.putText(debug, page_type, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
debug_show("PAGE TYPE", debug)
print(f"📄 Detected: {page_type} (w={w}, h={h})")

final_page = detect_mrz_orientation_and_crop(page)
