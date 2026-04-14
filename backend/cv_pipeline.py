import cv2
import numpy as np
import re
import os

# ── YOLO (generic detector – used for region guidance only) ────────────────────
try:
    from ultralytics import YOLO
    _yolo_model = YOLO('yolov8n.pt')
    _has_yolo = True
    print("✓ YOLOv8 loaded")
except Exception as e:
    _yolo_model = None
    _has_yolo = False
    print(f"⚠️  YOLO not available: {e}")

# ── EasyOCR ────────────────────────────────────────────────────────────────────
try:
    import easyocr
    _ocr_reader = easyocr.Reader(['en'], gpu=False)
    _has_easyocr = True
    print("✓ EasyOCR loaded")
except Exception as e:
    _ocr_reader = None
    _has_easyocr = False
    print(f"⚠️  EasyOCR not available: {e}")

# ── Tesseract fallback ─────────────────────────────────────────────────────────
try:
    import pytesseract
    pytesseract.get_tesseract_version()
    _has_tesseract = True
    print("✓ Tesseract OCR loaded")
except Exception:
    _has_tesseract = False


# ── BAD OCR STRINGS to discard ─────────────────────────────────────────────────
_BAD_STRINGS = {
    "book cover", "book with text regions detected", "book cover detected",
    "could not detect text from image", "text regions detected", "cover"
}


def _ocr_on_image(img_bgr) -> str:
    """
    Run OCR on a BGR image. Returns raw text or empty string.
    Never returns dummy strings like 'Book cover'.
    """
    if img_bgr is None or img_bgr.size == 0:
        return ""

    text = ""

    # 1. EasyOCR — best quality
    if _has_easyocr:
        try:
            results = _ocr_reader.readtext(img_bgr, detail=1, paragraph=False)
            # Keep results with confidence >= 0.3
            lines = [r[1] for r in results if r[2] >= 0.3]
            text = "\n".join(lines).strip()
            if text:
                return text
        except Exception as e:
            print(f"EasyOCR error: {e}")

    # 2. Tesseract fallback
    if _has_tesseract:
        try:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray, config='--psm 3').strip()
            if text:
                return text
        except Exception as e:
            print(f"Tesseract error: {e}")

    return ""


def _preprocess_variants(img_bgr):
    """
    Generate multiple preprocessed variants of the image for OCR.
    Returns list of BGR images to try OCR on.
    OPTIMIZED for speed: only runs original, grayscale, and contrast enhanced.
    """
    variants = []
    
    # 1. Original
    variants.append(img_bgr)

    # 2. Grayscale (often much faster for OCR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    variants.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

    # 3. CLAHE contrast enhancement — good for dim/low-contrast covers
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)
    variants.append(enhanced)

    return variants


def _clean_text(raw: str) -> str:
    """Remove noise chars, keep alphanumeric + spaces + apostrophes."""
    # Keep words that are at least 2 alphanumeric chars
    words = re.findall(r"[A-Za-z]{2,}(?:['\-]?[A-Za-z]+)*", raw)
    return " ".join(words).strip()


def _extract_candidate_lines(raw: str):
    """
    Split raw OCR text into individual lines or distinct blocks.
    Separates by newlines or large spatial gaps (represented as 2+ spaces).
    """
    if not raw:
        return []

    # Split by newlines OR 2+ spaces (common in EasyOCR output for distinct blocks)
    parts = re.split(r'\n|\s{2,}', raw)
    
    unique_lines = []
    seen = set()

    for p in parts:
        cleaned = _clean_text(p.strip())
        if len(cleaned) < 3:
            continue
            
        # Ignore obvious noise / metadata
        lower = cleaned.lower()
        if any(bad in lower for bad in ["over million", "copies sold", "best seller", "new york times"]):
            continue

        if lower not in seen:
            seen.add(lower)
            unique_lines.append(cleaned)

    # Sort: 2-5 word phrases first (perfect title size)
    def sort_key(s):
        wc = len(s.split())
        if 2 <= wc <= 5: return (100, len(s))
        return (wc, len(s))

    return sorted(unique_lines, key=sort_key, reverse=True)


def _get_title_region(img_bgr):
    """
    Heuristic: book titles are usually in the top 55% of the cover.
    Returns top crop of the image.
    """
    h, w = img_bgr.shape[:2]
    return img_bgr[0:int(h * 0.55), :]


def _get_bottom_region(img_bgr):
    """
    Author names are usually in the bottom 30% of the cover.
    """
    h, w = img_bgr.shape[:2]
    return img_bgr[int(h * 0.70):, :]


def process_image(image_bytes: bytes):
    """
    Process uploaded image bytes:
      1. Decode & resize
      2. Use YOLO to find rectangular objects (books on shelf) — each becomes a region
      3. Also use heuristic top-crop (title region) and full image as regions
      4. Apply multi-variant OCR on each region
      5. Return filtered, ranked candidate title strings (no dummy strings)
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Failed to decode image.")

    # Resize if too large
    h, w = img.shape[:2]
    max_dim = 1200
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    regions = []

    # ── A. YOLO regions (best effort, only for wide shots) ────────────────────
    # Only run YOLO if image is roughly square or landscape (likely a shelf)
    # If it's a tall portrait image, it's probably already a single book cover.
    is_portrait = h > (w * 1.2)
    
    if _has_yolo and _yolo_model is not None and not is_portrait:
        try:
            # Resize specifically for YOLO to speed up CPU inference
            yolo_img = img
            if max(h, w) > 640:
                y_scale = 640 / max(h, w)
                yolo_img = cv2.resize(img, (int(w * y_scale), int(h * y_scale)))
                
            predictions = _yolo_model(yolo_img, verbose=False)
            for r in predictions:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf > 0.3:
                        # Scale coordinates back up to original image size
                        orig_scale = 1.0 if max(h, w) <= 640 else max(h, w) / 640
                        x1 = int(float(box.xyxy[0][0]) * orig_scale)
                        y1 = int(float(box.xyxy[0][1]) * orig_scale)
                        x2 = int(float(box.xyxy[0][2]) * orig_scale)
                        y2 = int(float(box.xyxy[0][3]) * orig_scale)
                        
                        # Only add if region is reasonably sized
                        rw, rh = x2 - x1, y2 - y1
                        if rw > 40 and rh > 40:
                            crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                            if crop.size > 0:
                                # Add title region of the YOLO crop
                                regions.append(crop)
                                regions.append(_get_title_region(crop))
        except Exception as e:
            print(f"YOLO inference error: {e}")

    # ── B. Heuristic regions ───────────────────────────────────────────────────
    regions.append(_get_title_region(img))   # Top 55% — title zone
    regions.append(img)                       # Full image as fallback

    # ── C. Run OCR across all regions and variants ─────────────────────────────
    all_candidates = []
    seen = set()

    for region in regions:
        if region is None or region.size == 0:
            continue

        # Try multiple preprocessed variants
        for variant in _preprocess_variants(region):
            raw = _ocr_on_image(variant)
            if not raw:
                continue

            # Filter out dummy/bad strings
            raw_lower = raw.lower().strip()
            if raw_lower in _BAD_STRINGS:
                continue

            # Extract individual candidate lines
            candidates = _extract_candidate_lines(raw)
            for cand in candidates:
                cand_lower = cand.lower()
                # Skip known-bad phrases
                if cand_lower in _BAD_STRINGS:
                    continue
                # Skip pure numbers or very short strings
                if not re.search(r'[A-Za-z]{3,}', cand):
                    continue
                if cand_lower not in seen:
                    seen.add(cand_lower)
                    all_candidates.append(cand)

    # ── D. Rank candidates — prefer 2-6 word strings (typical title length) ───
    def rank_key(s):
        words = re.findall(r'[A-Za-z]{3,}', s)
        wc = len(words)
        # Give highest priority to 2-6 word phrases.
        # Massive strings (>6 words) are likely body text/taglines and should be lower priority.
        if 2 <= wc <= 6:
            score = 100 + wc # High priority group
        elif wc == 1:
            score = 50 + len(s) # Single words, prioritize longer ones
        else:
            score = 10 - wc  # Extreme long strings get lowest priority
        return score

    ranked = sorted(all_candidates, key=rank_key, reverse=True)

    # Return top 6 candidates (main.py will try each against Google Books)
    result = ranked[:6]

    if not result:
        print("[cv_pipeline] No text detected from image.")

    return result


def find_best_match(ocr_texts, candidates):
    """
    Flexible matching: substring + Levenshtein + partial token overlap.
    Returns the best matching candidate title or None.
    """
    if not ocr_texts or not candidates:
        return None

    from Levenshtein import ratio as lev_ratio

    best_match = None
    best_score = 0.0

    combined_ocr = " ".join(ocr_texts).lower()
    ocr_tokens = set(re.findall(r'[a-z]{3,}', combined_ocr))

    for candidate in candidates:
        cand_lower = candidate.lower().strip()
        if not cand_lower:
            continue

        # 1. Direct substring
        if cand_lower in combined_ocr or combined_ocr in cand_lower:
            score = 0.95
        else:
            score = lev_ratio(combined_ocr, cand_lower)
            for chunk in ocr_texts:
                s = lev_ratio(chunk.lower(), cand_lower)
                score = max(score, s)
            cand_tokens = set(re.findall(r'[a-z]{3,}', cand_lower))
            if cand_tokens:
                overlap = len(ocr_tokens.intersection(cand_tokens)) / len(cand_tokens)
                score = max(score, overlap * 0.9)

        if score > best_score:
            best_score = score
            best_match = candidate

    return best_match if best_score >= 0.35 else None
