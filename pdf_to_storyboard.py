#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pdf_to_storyboard.py
Storyboard: PDF -> crop -> Perspectiva -> N copies with frame label and time.
All text in English. Perspectiva.py runs in the same console (no capture).
"""

import os
import sys
import json
import shutil
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import argparse
import subprocess
import fitz  # PyMuPDF

if sys.platform == "win32":
    try:
        os.system("chcp 65001 > nul")
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

CONFIG_PATH = "storyboard_config.json"
LOG_PATH = "storyboard_log.txt"
PERSPECTIVA_SCRIPT = "Perspectiva.py"
PERSPECTIVA_OUTPUT = "result_with_grid_PROJ_full.jpg"
TEMP_INPUT = "temp_perspectiva_input.jpg"

# ---------------------------------------------------------------------------
# Standard display resolutions
# ---------------------------------------------------------------------------
STANDARD_RESOLUTIONS = {
    1: ("HD",       1280,  720),
    2: ("FHD",      1920, 1080),
    3: ("2K",       2560, 1440),
    4: ("4K",       3840, 2160),
    5: ("HD+",      1366,  768),
    6: ("HD ready", 1600,  900),
    7: ("2K+",      2048, 1152),
    8: ("4K+",      4096, 2304),
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_log_file = None

def log_init():
    global _log_file
    _log_file = open(LOG_PATH, "w", encoding="utf-8")

def log(msg):
    print(msg)
    if _log_file:
        _log_file.write(msg + "\n")
        _log_file.flush()

def log_close():
    global _log_file
    if _log_file:
        _log_file.close()
        _log_file = None

def load_config():
    if os.path.isfile(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}

def save_config(config):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

def detect_pdf_dpi(pdf_path):
    """Auto-detect DPI from raster images inside the PDF."""
    try:
        doc = fitz.open(pdf_path)
        best_dpi = 0
        for page in doc:
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                pix = fitz.Pixmap(doc, xref)
                img_w = pix.width
                img_h = pix.height
                page_w_pt = page.rect.width
                page_h_pt = page.rect.height
                if page_w_pt > 0 and page_h_pt > 0:
                    dpi_x = img_w / page_w_pt * 72
                    dpi_y = img_h / page_h_pt * 72
                    dpi = max(dpi_x, dpi_y)
                    log(f"  [DPI] Image: {img_w}x{img_h} px -> DPI ~{int(dpi)}")
                    if dpi > best_dpi:
                        best_dpi = dpi
                pix = None
        doc.close()
        if best_dpi > 0:
            return int(round(best_dpi))
        log("  [DPI] No raster images found in PDF, using fallback 300 DPI.")
        return 300
    except Exception as e:
        log(f"  [DPI] Auto-detection failed: {e}. Using fallback 300 DPI.")
        return 300

def pdf_to_image(pdf_path, dpi=150):
    """Converts all PDF pages into one long image via PyMuPDF."""
    log(f"[INFO] Rendering PDF at {dpi} DPI...")
    doc = fitz.open(pdf_path)
    pages_pil = []
    for page in doc:
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img_pil = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        pages_pil.append(img_pil)
        log(f"  Page rendered: {pix.width}x{pix.height} px")
    doc.close()

    if not pages_pil:
        raise RuntimeError("PDF has no pages")

    total_w = max(p.width for p in pages_pil)
    total_h = sum(p.height for p in pages_pil)
    combined = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))
    y_off = 0
    for p in pages_pil:
        combined.paste(p, (0, y_off))
        y_off += p.height

    log(f"[INFO] Full image size: {combined.width}x{combined.height} px")
    return combined

def get_font(size):
    """Finds a font that supports the needed glyphs."""
    candidates = []
    if sys.platform == "win32":
        candidates.extend([
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\segoeui.ttf",
            r"C:\Windows\Fonts\times.ttf",
            r"C:\Windows\Fonts\tahoma.ttf"
        ])
    if sys.platform.startswith("linux"):
        candidates.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf"
        ])
    candidates.append("arial.ttf")
    candidates.append("DejaVuSans.ttf")
    for path in candidates:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    log("[!] No suitable font found. Using default (may show boxes).")
    return ImageFont.load_default()

# ---------------------------------------------------------------------------
# Interactive crop window
# ---------------------------------------------------------------------------

def select_crop_rect(img_cv, saved_rect=None, config=None):
    """
    Opens a GUI window to select crop area.
    - Frame starts at FHD (1920x1080) proportions
    - 1-8: switch to standard resolution (frame changes SIZE to exact pixels)
    - Drag inside frame: move
    - Drag edge/corner: resize (switches to manual mode)
    - ?: toggle help
    - Enter: confirm, ESC: cancel

    Key fix: when a standard resolution is selected, the crop is EXACTLY
    rw x rh pixels (no rounding errors). Manual resize uses preview coords.
    """
    win_name = "Crop: drag=move/resize  ?=help  1-8=res  Enter=OK  ESC=cancel"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    h_img, w_img = img_cv.shape[:2]

    # --- Preview scaling (same as original) ---
    max_pw, max_ph = 1600, 900
    scale = min(1.0, max_pw / w_img, max_ph / h_img)
    if scale < 1.0:
        preview = cv2.resize(img_cv, (int(w_img * scale), int(h_img * scale)))
        log(f"[INFO] Preview scaled to {int(w_img * scale)}x{int(h_img * scale)}. Crop at full resolution.")
    else:
        preview = img_cv.copy()
        scale = 1.0

    pv_h, pv_w = preview.shape[:2]

    # --- Frame state ---
    # rw_full / rh_full: exact full-resolution crop size (used for final crop)
    # fx/fy/fw/fh: preview coords (for display and mouse interaction)
    # manual: False = standard resolution, True = user resized manually
    state = {
        "fx": 0, "fy": 0, "fw": 0, "fh": 0,
        "rw_full": 1920,
        "rh_full": 1080,
        "label": "FHD",
        "manual": False,
    }

    def set_resolution(rw, rh, label):
        """Set frame to exact resolution. Crop will be exactly rw x rh px."""
        # Clamp to image size if resolution > image
        rw_c = min(rw, w_img)
        rh_c = min(rh, h_img)
        if rw_c != rw or rh_c != rh:
            log(f"  [INFO] Resolution {rw}x{rh} larger than image {w_img}x{h_img}, clamped to {rw_c}x{rh_c}")

        # Preview frame size (for display only)
        fw = int(rw_c * scale)
        fh = int(rh_c * scale)

        # Clamp to 95% of preview if still too big
        if fw > pv_w * 0.95 or fh > pv_h * 0.95:
            s = min(pv_w * 0.95 / fw, pv_h * 0.95 / fh)
            fw = int(fw * s)
            fh = int(fh * s)

        fw = max(20, fw)
        fh = max(20, fh)

        # Center
        fx = (pv_w - fw) // 2
        fy = (pv_h - fh) // 2

        state["fx"] = fx
        state["fy"] = fy
        state["fw"] = fw
        state["fh"] = fh
        state["rw_full"] = rw_c
        state["rh_full"] = rh_c
        state["label"] = label
        state["manual"] = False

    def set_manual_from_preview():
        """Convert preview coords to full-res and mark as manual."""
        state["rw_full"] = int(state["fw"] / scale)
        state["rh_full"] = int(state["fh"] / scale)
        state["rw_full"] = max(1, min(state["rw_full"], w_img))
        state["rh_full"] = max(1, min(state["rh_full"], h_img))
        state["manual"] = True

    # --- Initial frame ---
    if saved_rect is not None and len(saved_rect) == 4:
        state["fx"] = max(0, int(saved_rect[0] * scale))
        state["fy"] = max(0, int(saved_rect[1] * scale))
        state["fw"] = min(int(saved_rect[2] * scale), pv_w)
        state["fh"] = min(int(saved_rect[3] * scale), pv_h)
        state["rw_full"] = min(saved_rect[2], w_img)
        state["rh_full"] = min(saved_rect[3], h_img)
        state["label"] = config.get("resolution_label", "FHD") if config else "FHD"
        state["manual"] = True
    else:
        set_resolution(1920, 1080, "FHD")

    # --- Mouse state ---
    dragging = False
    drag_mode = "move"
    drag_ox = 0
    drag_oy = 0
    EDGE_TH = 15
    help_visible = True

    def draw_frame(base):
        out = base.copy()
        fx, fy, fw, fh = state["fx"], state["fy"], state["fw"], state["fh"]
        label = state["label"]
        rw = state["rw_full"]
        rh = state["rh_full"]

        # Dark overlay outside frame
        overlay = np.full_like(out, (0, 0, 0))
        mask = np.zeros((pv_h, pv_w), dtype=np.uint8)
        cv2.rectangle(mask, (fx, fy), (fx + fw, fy + fh), 255, -1)
        inv = cv2.bitwise_not(mask)
        out = cv2.add(
            cv2.bitwise_and(overlay, overlay, mask=inv),
            cv2.bitwise_and(out, out, mask=mask)
        )

        # Green frame border
        cv2.rectangle(out, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)

        # Corner markers
        for cx, cy in [(fx, fy), (fx + fw, fy), (fx, fy + fh), (fx + fw, fy + fh)]:
            cv2.circle(out, (cx, cy), 5, (0, 255, 0), -1)

        # Edge midpoints
        for pt in [(fx, fy + fh // 2), (fx + fw, fy + fh // 2),
                   (fx + fw // 2, fy), (fx + fw // 2, fy + fh)]:
            cv2.circle(out, pt, 4, (0, 255, 0), -1)

        # Size label — shows EXACT full-resolution crop size
        mode_tag = " [manual]" if state["manual"] else ""
        txt = f"{label} ({rw}x{rh}) -> crop: {rw}x{rh}{mode_tag}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(txt, font, 0.6, 2)
        ty = fy - 10 if fy > th + 10 else fy + fh + th + 10
        cv2.rectangle(out, (fx - 3, ty - th - 3), (fx + tw + 3, ty + 3), (0, 0, 0), -1)
        cv2.putText(out, txt, (fx, ty), font, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        # Help panel
        if help_visible:
            lines = [
                "HELP - press ? to hide",
                "=========================",
                "Drag inside  = move frame",
                "Drag edge    = resize edge",
                "Drag corner  = resize 2 edges",
                "1 = HD       2 = FHD",
                "3 = 2K       4 = 4K",
                "5 = HD+      6 = HD ready",
                "7 = 2K+      8 = 4K+",
                "Enter = confirm  ESC = cancel",
                "? = toggle help",
                "=========================",
            ]
            pw_box, lh = 310, 22
            ph = len(lines) * lh + 12
            px = pv_w - pw_box - 8
            py = 8
            sub = out[py:py + ph, px:px + pw_box]
            dark_bg = np.full_like(sub, (15, 15, 15))
            out[py:py + ph, px:px + pw_box] = cv2.addWeighted(sub, 0.6, dark_bg, 0.4, 0)
            cv2.rectangle(out, (px, py), (px + pw_box, py + ph), (0, 255, 0), 1)
            for i, ln in enumerate(lines):
                cv2.putText(out, ln, (px + 8, py + lh + i * lh),
                            font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        return out

    def hit_test(x, y):
        fx, fy, fw, fh = state["fx"], state["fy"], state["fw"], state["fh"]
        if not (fx - EDGE_TH <= x <= fx + fw + EDGE_TH and
                fy - EDGE_TH <= y <= fy + fh + EDGE_TH):
            return None
        left = abs(x - fx) <= EDGE_TH
        right = abs(x - (fx + fw)) <= EDGE_TH
        top = abs(y - fy) <= EDGE_TH
        bot = abs(y - (fy + fh)) <= EDGE_TH
        if left and top: return "corner_tl"
        if right and top: return "corner_tr"
        if left and bot: return "corner_bl"
        if right and bot: return "corner_br"
        if left: return "edge_l"
        if right: return "edge_r"
        if top: return "edge_t"
        if bot: return "edge_b"
        if fx < x < fx + fw and fy < y < fy + fh: return "move"
        return None

    def on_mouse(event, x, y, flags, param):
        nonlocal dragging, drag_mode, drag_ox, drag_oy
        if event == cv2.EVENT_LBUTTONDOWN:
            m = hit_test(x, y)
            if m is not None:
                dragging = True
                drag_mode = m
                drag_ox = x - state["fx"]
                drag_oy = y - state["fy"]
        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            fx, fy, fw, fh = state["fx"], state["fy"], state["fw"], state["fh"]
            if drag_mode == "move":
                fx = max(0, min(x - drag_ox, pv_w - fw))
                fy = max(0, min(y - drag_oy, pv_h - fh))
            elif drag_mode == "edge_l":
                nx = max(0, min(x, fx + fw - 20))
                fw += fx - nx; fx = nx
            elif drag_mode == "edge_r":
                fw = max(20, min(x - fx, pv_w - fx))
            elif drag_mode == "edge_t":
                ny = max(0, min(y, fy + fh - 20))
                fh += fy - ny; fy = ny
            elif drag_mode == "edge_b":
                fh = max(20, min(y - fy, pv_h - fy))
            elif drag_mode == "corner_tl":
                nx = max(0, min(x, fx + fw - 20))
                ny = max(0, min(y, fy + fh - 20))
                fw += fx - nx; fh += fy - ny; fx = nx; fy = ny
            elif drag_mode == "corner_tr":
                ny = max(0, min(y, fy + fh - 20))
                fh += fy - ny; fy = ny
                fw = max(20, min(x - fx, pv_w - fx))
            elif drag_mode == "corner_bl":
                nx = max(0, min(x, fx + fw - 20))
                fw += fx - nx; fx = nx
                fh = max(20, min(y - fy, pv_h - fy))
            elif drag_mode == "corner_br":
                fw = max(20, min(x - fx, pv_w - fx))
                fh = max(20, min(y - fy, pv_h - fy))

            # Any resize (not move) switches to manual mode
            if drag_mode != "move":
                state["manual"] = True

            state["fx"] = fx
            state["fy"] = fy
            state["fw"] = fw
            state["fh"] = fh

            # Update rw_full/rh_full from preview coords when manual
            if state["manual"]:
                state["rw_full"] = max(1, min(int(fw / scale), w_img))
                state["rh_full"] = max(1, min(int(fh / scale), h_img))

        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False

    cv2.setMouseCallback(win_name, on_mouse)

    while True:
        disp = draw_frame(preview)
        cv2.imshow(win_name, disp)
        key = cv2.waitKey(30) & 0xFF

        if key == 13:  # Enter
            break
        elif key == 27:  # ESC
            cv2.destroyAllWindows()
            if saved_rect is not None:
                log("[INFO] Cancelled. Using saved crop area.")
                cur_lbl = config.get("resolution_label", "Full") if config else "Full"
                return tuple(saved_rect), cur_lbl
            else:
                log("[INFO] Cancelled. Using full image.")
                return (0, 0, w_img, h_img), "Full"
        elif key == ord("?") or key == ord("/"):
            help_visible = not help_visible
        elif key in [ord(str(i)) for i in range(1, 9)]:
            idx = int(chr(key))
            if idx in STANDARD_RESOLUTIONS:
                lbl, rw, rh = STANDARD_RESOLUTIONS[idx]
                set_resolution(rw, rh, lbl)
                log(f"[INFO] Switched to: {lbl} ({rw}x{rh}) -> crop exactly {state['rw_full']}x{state['rh_full']}")

    cv2.destroyAllWindows()

    # Final crop: use exact rw_full x rh_full (no rounding)
    x = int(state["fx"] / scale)
    y = int(state["fy"] / scale)
    w_r = state["rw_full"]
    h_r = state["rh_full"]

    x = max(0, min(x, w_img - 1))
    y = max(0, min(y, h_img - 1))
    w_r = max(1, min(w_r, w_img - x))
    h_r = max(1, min(h_r, h_img - y))

    mode_str = "manual" if state["manual"] else "standard"
    log(f"[INFO] Crop selected ({mode_str}): {state['label']} -> {w_r}x{h_r} at ({x},{y})")
    return (x, y, w_r, h_r), state["label"]

# ---------------------------------------------------------------------------
# Apply crop
# ---------------------------------------------------------------------------

def apply_crop(img_cv, crop_rect):
    x, y, w, h = crop_rect
    h_img, w_img = img_cv.shape[:2]
    x = max(0, min(x, w_img - 1))
    y = max(0, min(y, h_img - 1))
    w = max(1, min(w, w_img - x))
    h = max(1, min(h, h_img - y))
    return img_cv[y:y + h, x:x + w], (x, y, w, h)

def run_perspectiva(input_jpg):
    """Запускает Perspectiva в текущей консоли.

    В frozen-режиме (PyInstaller .exe) sys.executable указывает на сам .exe,
    поэтому мы перезапускаем себя же с флагом --run-perspectiva.
    """
    frozen = getattr(sys, "frozen", False)

    if frozen:
        script_path = sys.executable
        temp_dir = os.path.dirname(sys.executable)
        search_dirs = list(dict.fromkeys([
            os.getcwd(),
            temp_dir,
            os.path.dirname(os.path.abspath(input_jpg)),
        ]))
    else:
        search_paths = [
            PERSPECTIVA_SCRIPT,
            os.path.join(os.getcwd(), PERSPECTIVA_SCRIPT),
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         PERSPECTIVA_SCRIPT),
        ]
        script_path = None
        for p in search_paths:
            if os.path.isfile(p):
                script_path = os.path.abspath(p)
                break
        if script_path is None:
            log(f"[!] {PERSPECTIVA_SCRIPT} not found.")
            log("    Searched in:")
            for p in search_paths:
                log(f"      {p}")
            log("    Using cropped image without Perspectiva processing.")
            return None

        temp_dir = os.path.dirname(script_path)
        search_dirs = list(dict.fromkeys([
            os.getcwd(),
            temp_dir,
            os.path.dirname(os.path.abspath(input_jpg)),
        ]))

    temp_input_path = os.path.join(temp_dir, TEMP_INPUT)

    try:
        shutil.copy2(input_jpg, temp_input_path)
    except Exception as e:
        log(f"[!] Failed to create temp copy: {e}")
        return None

    if os.path.dirname(temp_input_path) == os.getcwd():
        pass_cmd = TEMP_INPUT
    else:
        pass_cmd = temp_input_path

    files_before = set()
    for d in search_dirs:
        if os.path.isdir(d):
            for f in os.listdir(d):
                files_before.add(os.path.join(d, f))

    if frozen:
        cmd = [sys.executable, "--run-perspectiva", pass_cmd]
        run_cwd = temp_dir
    else:
        cmd = [sys.executable, script_path, pass_cmd]
        run_cwd = temp_dir

    log("")
    log("=" * 62)
    log("  MANUAL STEPS in the Perspectiva window:")
    log("    1) Left-click the 4 anchor points (corners of the road area)")
    log("    2) Press D to auto-detect tracking dots")
    log("    3) Press P to enable road projection")
    log("    4) Press W to SAVE (creates *_PROJ_full.jpg  << REQUIRED)")
    log("    5) Optionally press S to save settings JSON")
    log("    6) Press ESC to close the window and return here")
    log("=" * 62)
    log("")

    log(f"[INFO] Launching: {cmd}")
    log("[INFO] Perspectiva runs in this console. Follow its prompts.")
    log("-" * 62)

    launch_time = time.time()

    try:
        result = subprocess.run(cmd, cwd=run_cwd)
    except Exception as e:
        log(f"[!] Failed to launch Perspectiva: {e}")
        return None

    log("-" * 62)

    if result.returncode != 0:
        log(f"[!] Perspectiva exited with code {result.returncode}")
        return None

    log("[INFO] Perspectiva finished successfully.")

    found_result = None
    for d in search_dirs:
        exact = os.path.join(d, PERSPECTIVA_OUTPUT)
        if os.path.isfile(exact):
            if os.path.getmtime(exact) >= launch_time - 1.0:
                found_result = exact
                break
            else:
                log(f"[INFO] Ignoring stale file (older than launch): {exact}")

    if found_result is None:
        log("[INFO] Exact output name not found (or stale). Searching new files...")
        new_files = []
        for d in search_dirs:
            if os.path.isdir(d):
                for f in os.listdir(d):
                    full = os.path.join(d, f)
                    if full in files_before:
                        continue
                    if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue
                    try:
                        if os.path.getmtime(full) < launch_time - 1.0:
                            continue
                    except OSError:
                        continue
                    new_files.append(full)
        if new_files:
            log(f"[INFO] Found {len(new_files)} new file(s):")
            for nf in new_files:
                log(f"      {nf}")
            for nf in new_files:
                lower = os.path.basename(nf).lower()
                if "proj_full" in lower:
                    found_result = nf
                    break
            if found_result is None:
                for nf in new_files:
                    lower = os.path.basename(nf).lower()
                    if "proj" in lower or "result" in lower:
                        found_result = nf
                        break
            if found_result is None and new_files:
                found_result = max(new_files, key=os.path.getmtime)
        else:
            log("[INFO] No new files found.")

    if found_result:
        log(f"[INFO] Result found: {found_result}")
    else:
        log("[!] Result not found by exact name or among new files.")
        for d in search_dirs:
            log(f"      {d}")

    try:
        os.remove(temp_input_path)
        log(f"[CLEANUP] Removed: {temp_input_path}")
    except OSError:
        pass

    return found_result

def add_info_to_frame(frame_pil, frame_index, fps):
    frame = frame_pil.copy()
    draw = ImageDraw.Draw(frame)
    t_sec = frame_index / fps
    minutes = int(t_sec // 60)
    seconds = int(t_sec % 60)
    millis = int((t_sec - int(t_sec)) * 1000)
    time_str = f"{minutes:02d}:{seconds:02d}.{millis:03d}"
    text = f"Frame {frame_index + 1} | {time_str}"
    font_size = max(24, frame.width // 30)
    font = get_font(font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    margin = max(15, frame.width // 80)
    x = margin
    y = frame.height - margin - th
    if y < 0:
        y = margin
    draw.rectangle([(x - 8, y - 8), (x + tw + 8, y + th + 8)], fill=(0, 0, 0))
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    return frame

def derive_basename(input_path):
    return os.path.splitext(os.path.basename(input_path))[0]

def interactive_input(prompt_text, default_val, val_type):
    while True:
        raw = input(f"{prompt_text}").strip()
        if raw == "":
            return default_val
        try:
            return val_type(raw)
        except ValueError:
            log("[!] Invalid input, try again.")

def main():
    log_init()
    log("=== Script started ===")

    parser = argparse.ArgumentParser(description="PDF Storyboard with Perspectiva processing")
    parser.add_argument("pdf_file", help="Path to PDF file")
    parser.add_argument("--fps", type=float, default=None, help="Frames per second")
    parser.add_argument("--frames", type=int, default=None, help="Number of frames")
    parser.add_argument("--dpi", type=int, default=None, help="DPI for PDF rendering")
    args = parser.parse_args()

    pdf_path = args.pdf_file
    if not os.path.isfile(pdf_path):
        log(f"[!] File not found: {pdf_path}")
        log_close()
        sys.exit(1)

    config = load_config()

    # --- DPI ---
    if args.dpi is not None:
        dpi = args.dpi
        log(f"[INFO] DPI from CLI: {dpi}")
    elif "dpi" in config:
        dpi = config["dpi"]
        log(f"[INFO] DPI from JSON: {dpi}")
    else:
        log("[INFO] Auto-detecting DPI from PDF...")
        dpi = detect_pdf_dpi(pdf_path)
        log(f"[INFO] Auto-detected DPI: {dpi}")
    config["dpi"] = dpi

    # --- FPS ---
    if args.fps is not None:
        fps = args.fps
        log(f"[INFO] FPS from CLI: {fps}")
    elif "fps" in config:
        fps = config["fps"]
        log(f"[INFO] FPS from JSON: {fps}")
    else:
        fps = interactive_input(f"Enter FPS (press Enter for 20.0): ", 20.0, float)
        log(f"[INFO] FPS: {fps}")
    config["fps"] = fps

    # --- FRAMES ---
    if args.frames is not None:
        num_frames = args.frames
        log(f"[INFO] Frames from CLI: {num_frames}")
    elif "frames" in config:
        num_frames = config["frames"]
        log(f"[INFO] Frames from JSON: {num_frames}")
    else:
        num_frames = interactive_input(f"Enter number of frames (press Enter for 10): ", 10, int)
        log(f"[INFO] Frames: {num_frames}")
    config["frames"] = num_frames

    save_config(config)

    # --- PDF to image ---
    log("[INFO] Converting PDF to image...")
    combined_pil = pdf_to_image(pdf_path, dpi=dpi)
    img_cv = cv2.cvtColor(np.array(combined_pil), cv2.COLOR_RGB2BGR)

    # --- Crop (graphical window, no console questions) ---
    saved_rect = config.get("crop_rect")
    crop_rect, cur_label = select_crop_rect(img_cv, saved_rect, config)
    config["crop_rect"] = list(crop_rect)
    config["resolution_label"] = cur_label
    save_config(config)

    cropped_cv, _ = apply_crop(img_cv, crop_rect)
    cropped_pil = Image.fromarray(cv2.cvtColor(cropped_cv, cv2.COLOR_BGR2RGB))
    log(f"[INFO] Cropped size: {cropped_pil.width}x{cropped_pil.height} px")

    temp_crop_path = os.path.join(os.getcwd(), "temp_crop.jpg")
    cropped_pil.save(temp_crop_path, "JPEG", quality=100)
    log(f"[INFO] Cropped image saved to: {temp_crop_path} (quality=100)")

    result_path = run_perspectiva(temp_crop_path)

    if result_path is not None:
        final_pil = Image.open(result_path).convert("RGB")
        log(f"[INFO] Using perspective-corrected image: {result_path}")
        log(f"[INFO] Result image size: {final_pil.width}x{final_pil.height} px")
    else:
        final_pil = cropped_pil
        log("[INFO] Using original cropped image (Perspectiva skipped or failed).")

    log(f"[INFO] Output resolution: {final_pil.width}x{final_pil.height} px (quality=100)")

    base_name = derive_basename(pdf_path)
    out_dir = os.path.join(os.getcwd(), "storyboard")
    os.makedirs(out_dir, exist_ok=True)

    log(f"[INFO] Generating {num_frames} frames at {fps} FPS...")

    for i in range(num_frames):
        frame = add_info_to_frame(final_pil, i, fps)
        out_path = os.path.join(out_dir, f"{base_name}_frame_{i + 1:03d}.jpg")
        frame.save(out_path, "JPEG", quality=100)
        log(f"  [{i + 1}/{num_frames}] Saved: {out_path} ({frame.width}x{frame.height})")

    for tmp in [temp_crop_path]:
        try:
            if os.path.isfile(tmp):
                os.remove(tmp)
                log(f"[CLEANUP] Removed: {tmp}")
        except OSError:
            pass

    # Удаляем только устаревшие кадры ЭТОГО pdf-файла.
    try:
        prefix = f"{base_name}_frame_"
        keep = {f"{base_name}_frame_{i + 1:03d}.jpg"
                for i in range(num_frames)}
        for f in os.listdir(out_dir):
            if f.startswith(prefix) and f not in keep:
                full_path = os.path.join(out_dir, f)
                os.remove(full_path)
                log(f"[CLEANUP] Removed stale frame: {f}")
    except OSError:
        pass

    log(f"[INFO] {PERSPECTIVA_OUTPUT} is kept (not deleted).")
    log(f"\n[DONE] {num_frames} frames saved to: {out_dir}")
    log("=== Script finished ===")
    log_close()

def _run_perspectiva_cli(image_path):
    """Точка входа для self-перезапуска из frozen-режима."""
    from gui import interactive_grid
    interactive_grid(image_path, rows=0, cols=0,
                     output_path="result_with_grid.jpg")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--run-perspectiva":
        img_arg = sys.argv[2] if len(sys.argv) > 2 else None
        if not img_arg:
            print("[!] --run-perspectiva requires an image path")
            sys.exit(1)
        try:
            _run_perspectiva_cli(img_arg)
        except Exception:
            import traceback
            traceback.print_exc()
            sys.exit(1)
        sys.exit(0)

    try:
        main()
    except Exception as e:
        import traceback
        print("\n" + "=" * 50)
        print("!!! CRITICAL ERROR !!!")
        print("=" * 50)
        traceback.print_exc()
        print("=" * 50)
        try:
            log_close()
        except Exception:
            pass
        sys.exit(1)
