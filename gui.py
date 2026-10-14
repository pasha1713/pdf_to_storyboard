# -*- coding: utf-8 -*-
"""
gui.py — графический интерфейс для разметки сетки перспективы
и поиска трекинговых точек (салатовых или голубых).
Нумерация точек и X-пересечений: снизу вверх, слева направо.
CSV выровнен пробелами — запятые строго друг под другом.
Проекция трекинговых точек на дорогу (z=0) при известной высоте ГРЗ.
Калибровка камеры — аналитическая декомпозиция гомографии.
"""

import cv2
import numpy as np
import json
import os
from utils import set_unicode_title
from transformations import draw_perspective_grid

# ── Глобальное состояние ────────────────────────────────────────
click_points = []
drag_index = -1
display_image = None
preview_image = None
show_grid = False
cur_rows = 12
cur_cols = 14
grid_color = (0, 255, 0)
grid_thickness = 2
green_dots = []

detect_sensitivity = 50
dot_color_mode = "green"

green_dot_radius = 8
blue_dot_radius = 16

debug_mode = False
debug_candidates = []

selected_line_mode = None
selected_line_idx = 0

pad_left = 0
pad_right = 0
pad_top = 0
pad_bottom = 0
img_w = 0
img_h = 0
img_resized = None
canvas_bg = (40, 40, 40)
expand_margin = 10
max_expansion_ratio = 0.5

_expand_left_ok = True
_expand_right_ok = True
_expand_top_ok = True
_expand_bottom_ok = True

_win_key = "PerspectiveGrid"
_win_title = "Perspective Grid — 4 points"
grab_radius = 12
_cur_image_path = None
_scale = 1.0
_window_created = False

_last_mouse_x = 0
_last_mouse_y = 0
add_dot_mode = False
drag_dot_index = -1

# ── Параметры проекции на дорогу ────────────────────────────────
cell_width_m = 0.5
cell_length_m = 3.0
camera_height_m = 2.0
track_height_m = 1.1
show_projection = False
projected_dots = []
projected_world = []


def _sort_dots_bottom_up(dots):
    return sorted(dots, key=lambda pt: (-pt[1], pt[0]))


def _draw_cross(img, pt, size=7, color=(0, 0, 255), thickness=2):
    x, y = pt
    cv2.line(img, (x - size, y - size), (x + size, y + size), color, thickness)
    cv2.line(img, (x + size, y - size), (x - size, y + size), color, thickness)


# ── Гомография и геометрия сетки ────────────────────────────────
def _compute_homography(points, total_rows, total_cols):
    src = np.array([
        [0, 0],
        [total_cols, 0],
        [total_cols, total_rows],
        [0, total_rows],
    ], dtype=np.float32)
    dst = np.array(points, dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    return H


def _persp_pt(H, x, y):
    pt = np.array([[[x, y]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, H)
    return int(round(out[0, 0, 0])), int(round(out[0, 0, 1]))


def _grid_line_row(points, row_idx, total_rows, total_cols):
    H = _compute_homography(points, total_rows, total_cols)
    return _persp_pt(H, 0, row_idx), _persp_pt(H, total_cols, row_idx)


def _grid_line_col(points, col_idx, total_rows, total_cols):
    H = _compute_homography(points, total_rows, total_cols)
    return _persp_pt(H, col_idx, 0), _persp_pt(H, col_idx, total_rows)


def _grid_intersection(points, row_idx, col_idx, total_rows, total_cols):
    H = _compute_homography(points, total_rows, total_cols)
    return _persp_pt(H, col_idx, row_idx)


def _row_intersections_sorted(points, row_idx, total_rows, total_cols):
    pts = []
    for col_idx in range(total_cols + 1):
        pt = _grid_intersection(points, row_idx, col_idx, total_rows, total_cols)
        pts.append(pt)
    pts.sort(key=lambda p: -p[1])
    return pts


def _col_intersections_sorted(points, col_idx, total_rows, total_cols):
    pts = []
    for row_idx in range(total_rows + 1):
        pt = _grid_intersection(points, row_idx, col_idx, total_rows, total_cols)
        pts.append(pt)
    pts.sort(key=lambda p: -p[1])
    return pts


def _print_intersections():
    if selected_line_mode is None or len(click_points) != 4:
        return
    if selected_line_mode == "row":
        pts = _row_intersections_sorted(click_points, selected_line_idx,
                                        cur_rows, cur_cols)
        print("  Row " + str(selected_line_idx) + "/" + str(cur_rows)
              + " — intersections (bottom to top):")
        for i, pt in enumerate(pts):
            print("    X" + str(i + 1) + ": ("
                  + str(pt[0]) + ", " + str(pt[1]) + ")")
    elif selected_line_mode == "col":
        pts = _col_intersections_sorted(click_points, selected_line_idx,
                                        cur_rows, cur_cols)
        print("  Col " + str(selected_line_idx) + "/" + str(cur_cols)
              + " — intersections (bottom to top):")
        for i, pt in enumerate(pts):
            print("    X" + str(i + 1) + ": ("
                  + str(pt[0]) + ", " + str(pt[1]) + ")")


# ── Проекция точек на дорогу ───────────────────────────────────
def _compute_projection():
    """Калибровка камеры перебором фокусного расстояния через solvePnP.

    Мировые координаты нормализуются по высоте камеры, чтобы solvePnP
    работал в устойчивом диапазоне чисел независимо от реальных
    размеров сетки.
    """
    global _proj_norm_scale

    if len(click_points) != 4:
        return None, None, None, None

    W = cur_cols * cell_width_m
    L = cur_rows * cell_length_m

    # Нормализация: делим все мировые координаты на высоту камеры
    norm_scale = max(camera_height_m, 0.1)
    _proj_norm_scale = norm_scale

    Wn = W / norm_scale
    Ln = L / norm_scale
    cam_h_n = camera_height_m / norm_scale  # = 1.0

    world_3d = np.array([
        [0,   Ln, 0],
        [Wn,  Ln, 0],
        [Wn,  0,  0],
        [0,   0,  0],
    ], dtype=np.float64)

    image_pts = np.array([
        [p[0] - pad_left, p[1] - pad_top] for p in click_points
    ], dtype=np.float64)

    cx = img_w / 2.0
    cy = img_h / 2.0

    def _try_f(f_val):
        K = np.array([[f_val, 0, cx], [0, f_val, cy], [0, 0, 1]],
                      dtype=np.float64)
        try:
            ok, rvec, tvec = cv2.solvePnP(
                world_3d, image_pts, K, None,
                flags=cv2.SOLVEPNP_IPPE)
        except Exception:
            ok, rvec, tvec = cv2.solvePnP(
                world_3d, image_pts, K, None,
                flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        R, _ = cv2.Rodrigues(rvec)
        C = -R.T @ tvec
        cz = float(C[2])
        if cz <= 0:
            return None
        err = abs(cz - cam_h_n)
        return (K.copy(), R.copy(), tvec.copy(), C.copy(), err)

    # Грубый поиск — расширенный диапазон
    best = None
    f_candidates_all = []

    # Первый проход: широкий шаг до 50000
    for f_int in range(200, 50001, 100):
        res = _try_f(float(f_int))
        if res is not None:
            f_candidates_all.append(f_int)
            if best is None or res[4] < best[4]:
                best = res

    if best is None:
        print("  Projection: no solution found.")
        return None, None, None, None

    # Точный поиск вокруг лучшего f — шаг 1
    K, R, tvec, C, err = best
    f_best = K[0, 0]
    f_lo = max(10, int(f_best) - 100)
    f_hi = int(f_best) + 101
    for f_int in range(f_lo, f_hi):
        res = _try_f(float(f_int))
        if res is not None and res[4] < err:
            best = res
            err = res[4]

    # Если ошибка всё ещё велика — проверим, не упёрлись ли в границу
    K, R, tvec, C, err = best
    if err > 0.1:
        print("  Warning: cam_h mismatch. Trying higher f range...")
        for f_int in range(50001, 200001, 200):
            res = _try_f(float(f_int))
            if res is not None and res[4] < err:
                best = res
                err = res[4]
                K, R, tvec, C, _ = best
        # Точный поиск вокруг нового лучшего
        f_best = K[0, 0]
        for f_int in range(max(10, int(f_best) - 50),
                           int(f_best) + 51):
            res = _try_f(float(f_int))
            if res is not None and res[4] < err:
                best = res
                err = res[4]

    K, R, tvec, C, _ = best

    # Возврат в реальные метры
    tvec = tvec * norm_scale
    C = -R.T @ tvec

    print("  Projection: f=" + str(round(K[0, 0], 1))
          + "  cam_h=" + str(round(float(C[2]), 3))
          + "m (target=" + str(camera_height_m) + "m)")

    return K, R, tvec, C


def _project_dots_to_road():
    """Проекция трекинговых точек (на высоте track_height_m)
    на дорогу (z=0)."""
    global projected_dots, projected_world

    if len(click_points) != 4 or not green_dots:
        projected_dots = []
        projected_world = []
        return

    K, R, tvec, C = _compute_projection()
    if K is None:
        print("  Could not compute projection.")
        projected_dots = []
        projected_world = []
        return

    K_inv = np.linalg.inv(K)

    W = cur_cols * cell_width_m
    L = cur_rows * cell_length_m
    world_2d = np.array([
        [0, L], [W, L], [W, 0], [0, 0]
    ], dtype=np.float32)
    canvas_pts = np.array(click_points, dtype=np.float32)
    H_world, _ = cv2.findHomography(world_2d, canvas_pts)

    if H_world is None:
        print("  Could not compute world-to-canvas homography.")
        projected_dots = []
        projected_world = []
        return

    projected = []
    world_coords = []

    for (cx_dot, cy_dot) in green_dots:
        u = cx_dot - pad_left
        v = cy_dot - pad_top

        # Луч из камеры через пиксель (u, v) в мировых координатах
        ray_cam = K_inv @ np.array([u, v, 1.0])
        D = R.T @ ray_cam

        if abs(D[2]) < 1e-10:
            continue

        # Пересечение луча с плоскостью z = track_height_m
        lam = (track_height_m - float(C[2])) / float(D[2])
        if lam < 0:
            continue

        X = float(C[0]) + lam * float(D[0])
        Y = float(C[1]) + lam * float(D[1])

        world_coords.append((X, Y))

        # Проекция (X, Y, 0) на холст через гомографию дороги
        pt = np.array([X, Y, 1.0])
        proj = H_world @ pt
        if abs(proj[2]) < 1e-10:
            continue

        px = int(round(proj[0] / proj[2]))
        py = int(round(proj[1] / proj[2]))
        projected.append((px, py))

    projected_dots = projected
    projected_world = world_coords

    print("  Projected " + str(len(projected)) + " points onto road.")


def _save_projection_csv(image_path):
    if not projected_world:
        return

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    csv_path = base_name + "_ROAD_PROJECTION.csv"
    img_label = os.path.basename(image_path)

    rows = [[" ", "Label", "X_road_m", "Y_road_m", "Slice"]]
    for i, (wx, wy) in enumerate(projected_world, start=1):
        rows.append([str(i), img_label,
                      str(round(wx, 4)), str(round(wy, 4)), "1"])

    num_cols = len(rows[0])
    col_widths = [0] * num_cols
    for row in rows:
        for c in range(num_cols):
            col_widths[c] = max(col_widths[c], len(row[c]))

    lines = []
    for row in rows:
        padded = [row[c].ljust(col_widths[c]) for c in range(num_cols)]
        lines.append(",".join(padded))

    try:
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print("  Projection CSV saved: " + csv_path
              + " (" + str(len(projected_world)) + " points)")
    except Exception as e:
        print("  Projection CSV error: " + str(e))

def _save_projection_only(image_path, output_path):
    """Сохранить .jpg без перспективной сетки, но с точками ГРЗ
    и их проекциями на дорогу.

    Всегда создаёт ДВА файла:
        <output>_PROJ.jpg       - превью
        <output>_PROJ_full.jpg  - полное разрешение (обязательно
                                   для pdf_to_storyboard.py)
    """
    if preview_image is None:
        print("  No image to save.")
        return

    if not projected_dots:
        print("  No projections to save. Enable projection (P) first.")
        return

    img = preview_image.copy()
    cur_r = green_dot_radius if dot_color_mode == "green" \
        else blue_dot_radius
    proj_color = (0, 255, 255)

    for i, (gx, gy) in enumerate(green_dots):
        ring_c = (0, 255, 255) if dot_color_mode == "green" \
            else (255, 128, 0)
        cv2.circle(img, (gx, gy), cur_r, ring_c, 2)
        cv2.line(img, (gx - 3, gy), (gx + 3, gy), ring_c, 1)
        cv2.line(img, (gx, gy - 3), (gx, gy + 3), ring_c, 1)
        label = str(i + 1)
        tx = gx + cur_r + 4
        ty = gy + 5
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                cv2.putText(img, label, (tx + dx, ty + dy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(img, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, ring_c, 1)

    for i, (px, py) in enumerate(projected_dots):
        if i < len(green_dots):
            gx, gy = green_dots[i]
            cv2.line(img, (gx, gy), (px, py), proj_color, 1)
        cv2.circle(img, (px, py), 6, proj_color, 1)
        cv2.line(img, (px - 3, py), (px + 3, py), proj_color, 1)
        cv2.line(img, (px, py - 3), (px, py + 3), proj_color, 1)

    proj_path = output_path.replace(".jpg", "_PROJ.jpg")
    cv2.imwrite(proj_path, img)
    print("  Saved (projection only): " + proj_path)

    if _scale == 1.0 or not _cur_image_path:
        full_orig = img.copy()
    else:
        full_orig = cv2.imread(_cur_image_path)
        if full_orig is None:
            print("  [!] Cannot load full-res original, using preview.")
            full_orig = img.copy()
        else:
            sc = 1.0 / _scale
            for i, (gx, gy) in enumerate(green_dots):
                fx = int((gx - pad_left) * sc)
                fy = int((gy - pad_top) * sc)
                fr = max(2, int(cur_r * sc))
                ring_c = (0, 255, 255) if dot_color_mode == "green" \
                    else (255, 128, 0)
                cv2.circle(full_orig, (fx, fy), fr, ring_c,
                           max(1, int(2 * sc)))
                cv2.line(full_orig,
                         (int(fx - 3 * sc), fy),
                         (int(fx + 3 * sc), fy),
                         ring_c, 1)
                cv2.line(full_orig,
                         (fx, int(fy - 3 * sc)),
                         (fx, int(fy + 3 * sc)),
                         ring_c, 1)
                cv2.putText(full_orig, str(i + 1),
                            (int(fx + fr + 4 * sc),
                             int(fy + 5 * sc)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5 * sc, ring_c, 1)

            for i, (px, py) in enumerate(projected_dots):
                fpx = int((px - pad_left) * sc)
                fpy = int((py - pad_top) * sc)
                if i < len(green_dots):
                    gx, gy = green_dots[i]
                    fx = int((gx - pad_left) * sc)
                    fy = int((gy - pad_top) * sc)
                    cv2.line(full_orig, (fx, fy), (fpx, fpy),
                             proj_color, 1)
                fr6 = max(3, int(6 * sc))
                cv2.circle(full_orig, (fpx, fpy), fr6, proj_color, 1)
                cv2.line(full_orig,
                         (int(fpx - 3 * sc), fpy),
                         (int(fpx + 3 * sc), fpy),
                         proj_color, 1)
                cv2.line(full_orig,
                         (fpx, int(fpy - 3 * sc)),
                         (fpx, int(fpy + 3 * sc)),
                         proj_color, 1)

    full_path = proj_path.replace(".jpg", "_full.jpg")
    cv2.imwrite(full_path, full_orig)
    print("  Saved full-res: " + full_path)

def _canvas_to_image_coord(cx, cy):
    ox = int(round((cx - pad_left) / _scale)) if _scale > 0 else cx
    oy = int(round((cy - pad_top) / _scale)) if _scale > 0 else cy
    return ox, oy


def _save_csv(image_path, points, suffix):
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    csv_path = base_name + "_" + suffix + ".csv"
    img_label = os.path.basename(image_path)
    
    # Группируем точки по строкам
    rows = [[" ", "Label", "X", "Y", "Slice"]]
    slice_num = 1
    points_per_slice = cur_cols + 1  # Количество точек в строке
    
    for i, (cx, cy) in enumerate(points, start=1):
        ox, oy = _canvas_to_image_coord(cx, cy)
        slice_num = (i - 1) // points_per_slice + 1
        rows.append([str(i), img_label, str(ox), str(oy), str(slice_num)])
    
    # Форматирование CSV
    num_cols = len(rows[0])
    col_widths = [0] * num_cols
    for row in rows:
        for c in range(num_cols):
            col_widths[c] = max(col_widths[c], len(row[c]))
    
    lines = []
    for row in rows:
        padded = [row[c].ljust(col_widths[c]) for c in range(num_cols)]
        lines.append(",".join(padded))
    
    try:
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print("  CSV saved: " + csv_path + " (" + str(len(points)) + " points)")
    except Exception as e:
        print("  CSV save error: " + str(e))


def _collect_line_points():
    if selected_line_mode is None or len(click_points) != 4:
        return []
    if selected_line_mode == "row":
        return _row_intersections_sorted(click_points, selected_line_idx,
                                         cur_rows, cur_cols)
    elif selected_line_mode == "col":
        return _col_intersections_sorted(click_points, selected_line_idx,
                                          cur_rows, cur_cols)
    return []


# ── Настройки ──────────────────────────────────────────────────
def _get_settings_path(image_path):
    base, ext = os.path.splitext(image_path)
    return base + "_grid_settings.json"


def _save_settings(image_path, points, rows, cols, dots, sensitivity):
    settings_path = _get_settings_path(image_path)
    img_points = [(p[0] - pad_left, p[1] - pad_top) for p in points]
    img_dots = [(d[0] - pad_left, d[1] - pad_top) for d in dots]
    settings = {
        "image": os.path.basename(image_path),
        "points": img_points,
        "rows": rows,
        "cols": cols,
        "green_dots": img_dots,
        "sensitivity": sensitivity,
        "dot_color_mode": dot_color_mode,
        "green_dot_radius": green_dot_radius,
        "blue_dot_radius": blue_dot_radius,
        "selected_line_mode": selected_line_mode,
        "selected_line_idx": selected_line_idx,
        "cell_width_m": cell_width_m,
        "cell_length_m": cell_length_m,
        "camera_height_m": camera_height_m,
        "track_height_m": track_height_m,
        "show_projection": show_projection,
    }
    try:
        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
        print("Settings saved: " + settings_path)
    except Exception as e:
        print("Save error: " + str(e))


def _load_settings(image_path):
    settings_path = _get_settings_path(image_path)
    if not os.path.exists(settings_path):
        return None
    try:
        with open(settings_path, "r", encoding="utf-8") as f:
            settings = json.load(f)
        if os.path.basename(image_path) != settings.get("image", ""):
            print("Settings for different file, skip.")
            return None
        return settings
    except Exception as e:
        print("Load error: " + str(e))
        return None


# ── Холст ──────────────────────────────────────────────────────
def _rebuild_canvas():
    global preview_image
    if img_resized is None:
        return
    max_pad = int(img_w * max_expansion_ratio)
    pl = min(pad_left, max_pad)
    pr = min(pad_right, max_pad)
    pt = min(pad_top, max_pad)
    pb = min(pad_bottom, max_pad)
    ph = img_h + pt + pb
    pw = img_w + pl + pr
    canvas = np.full((ph, pw, 3), canvas_bg, dtype=np.uint8)
    canvas[pt:pt + img_h, pl:pl + img_w] = img_resized
    preview_image = canvas


def _safe_resize_window():
    if _window_created and preview_image is not None:
        try:
            cv2.resizeWindow(_win_key,
                             preview_image.shape[1], preview_image.shape[0])
        except Exception:
            pass


def _fit_canvas_to_points():
    global pad_left, pad_right, pad_top, pad_bottom
    if not click_points:
        return
    max_pad = int(img_w * max_expansion_ratio)
    min_x = min(p[0] for p in click_points)
    max_x = max(p[0] for p in click_points)
    min_y = min(p[1] for p in click_points)
    max_y = max(p[1] for p in click_points)
    expanded = False

    if min_x < 0:
        shift = -min_x
        new_pad_left = min(pad_left + shift, max_pad)
        actual_shift = new_pad_left - pad_left
        if actual_shift > 0:
            pad_left = new_pad_left
            for i in range(len(click_points)):
                click_points[i] = (click_points[i][0] + actual_shift,
                                   click_points[i][1])
            for i in range(len(green_dots)):
                green_dots[i] = (green_dots[i][0] + actual_shift,
                                 green_dots[i][1])
            expanded = True

    if min_y < 0:
        shift = -min_y
        new_pad_top = min(pad_top + shift, max_pad)
        actual_shift = new_pad_top - pad_top
        if actual_shift > 0:
            pad_top = new_pad_top
            for i in range(len(click_points)):
                click_points[i] = (click_points[i][0],
                                   click_points[i][1] + actual_shift)
            for i in range(len(green_dots)):
                green_dots[i] = (green_dots[i][0],
                                green_dots[i][1] + actual_shift)
            expanded = True

    if preview_image is None:
        return

    h, w = preview_image.shape[:2]
    max_x_now = max(p[0] for p in click_points)
    max_y_now = max(p[1] for p in click_points)

    if max_x_now >= w:
        need = max_x_now - w + 10
        new_pad_right = min(pad_right + need, max_pad)
        if new_pad_right > pad_right:
            pad_right = new_pad_right
            expanded = True

    if max_y_now >= h:
        need = max_y_now - h + 10
        new_pad_bottom = min(pad_bottom + need, max_pad)
        if new_pad_bottom > pad_bottom:
            pad_bottom = new_pad_bottom
            expanded = True

    if expanded:
        _rebuild_canvas()
        _safe_resize_window()


def _expand_if_needed(x, y):
    global pad_left, pad_right, pad_top, pad_bottom
    global _expand_left_ok, _expand_right_ok
    global _expand_top_ok, _expand_bottom_ok

    if preview_image is None:
        return

    h, w = preview_image.shape[:2]
    max_pad = int(img_w * max_expansion_ratio)
    dx = 0
    dy = 0
    expanded = False

    if w > 0 and x > w * 0.3:
        _expand_left_ok = True
    if w > 0 and x < w * 0.7:
        _expand_right_ok = True
    if h > 0 and y > h * 0.3:
        _expand_top_ok = True
    if h > 0 and y < h * 0.7:
        _expand_bottom_ok = True

    if x < expand_margin and _expand_left_ok:
        remaining = max_pad - pad_left
        if remaining > 0:
            chunk = min(max(50, img_w // 4), remaining)
            pad_left += chunk
            dx = chunk
            _expand_left_ok = False
            expanded = True

    if y < expand_margin and _expand_top_ok:
        remaining = max_pad - pad_top
        if remaining > 0:
            chunk = min(max(50, img_h // 4), remaining)
            pad_top += chunk
            dy = chunk
            _expand_top_ok = False
            expanded = True

    if x > w - expand_margin and _expand_right_ok:
        remaining = max_pad - pad_right
        if remaining > 0:
            chunk = min(max(50, img_w // 4), remaining)
            pad_right += chunk
            _expand_right_ok = False
            expanded = True

    if y > h - expand_margin and _expand_bottom_ok:
        remaining = max_pad - pad_bottom
        if remaining > 0:
            chunk = min(max(50, img_h // 4), remaining)
            pad_bottom += chunk
            _expand_bottom_ok = False
            expanded = True

    if expanded:
        for i in range(len(click_points)):
            if i != drag_index:
                click_points[i] = (click_points[i][0] + dx,
                                   click_points[i][1] + dy)
        for i in range(len(green_dots)):
            green_dots[i] = (green_dots[i][0] + dx,
                             green_dots[i][1] + dy)
        _rebuild_canvas()
        _safe_resize_window()


# ── Параметры поиска ───────────────────────────────────────────
def _sensitivity_to_params(s, color_mode="green", dot_radius=8):
    s = max(1, min(100, s))
    r = max(2, dot_radius)

    if color_mode == "blue":
        hue_low = 80
        hue_high = 140
    else:
        hue_low = 20
        hue_high = 95

    sat_low = max(5, int(60 - s * 0.55))
    val_low = max(5, int(65 - s * 0.55))

    ref_area = np.pi * r * r
    min_area = max(1, int(ref_area * 0.10))
    max_area = int(ref_area * 5.0)

    min_r_eff = r * 0.5
    max_r_eff = r * 2.5

    min_circ = max(0.05, 0.3 - s * 0.002)
    merge_radius = int(r * 1.8)

    blob_min_area = float(min_area)
    blob_max_area = float(max_area)
    blob_min_circ = max(0.2, 0.6 - s * 0.003)
    blob_min_convexity = max(0.4, 0.8 - s * 0.003)
    blob_min_inertia = max(0.1, 0.5 - s * 0.003)

    return {
        "hue_low": hue_low,
        "hue_high": hue_high,
        "sat_low": sat_low,
        "val_low": val_low,
        "min_area": min_area,
        "max_area": max_area,
        "min_circ": min_circ,
        "merge_radius": merge_radius,
        "blob_min_area": blob_min_area,
        "blob_max_area": blob_max_area,
        "blob_min_circ": blob_min_circ,
        "blob_min_convexity": blob_min_convexity,
        "blob_min_inertia": blob_min_inertia,
        "ref_area": int(ref_area),
        "min_r_eff": min_r_eff,
        "max_r_eff": max_r_eff,
    }


# ── Поиск на ч/б через SimpleBlobDetector ───────────────────────
def _detect_grayscale_blobs(sub, p, sub_mask):
    gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    all_dots = []

    for invert in [False, True]:
        g = gray.copy()
        if invert:
            g = 255 - g
        g = cv2.GaussianBlur(g, (5, 5), 0)
        if sub_mask is not None:
            g_masked = cv2.bitwise_and(g, g, mask=sub_mask)
        else:
            g_masked = g

        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 10
        params.maxThreshold = 220
        params.thresholdStep = 10
        params.filterByColor = True
        params.blobColor = 255
        params.filterByArea = True
        params.minArea = float(p["blob_min_area"])
        params.maxArea = float(p["blob_max_area"])
        params.filterByCircularity = True
        params.minCircularity = float(p["blob_min_circ"])
        params.filterByConvexity = True
        params.minConvexity = float(p["blob_min_convexity"])
        params.filterByInertia = True
        params.minInertiaRatio = float(p["blob_min_inertia"])

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(g_masked)

        for kp in keypoints:
            cx = int(kp.pt[0])
            cy = int(kp.pt[1])
            r_kp = kp.size / 2.0
            area = max(1, int(np.pi * r_kp * r_kp))
            all_dots.append((cx, cy, area))

        mode_label = "dark-on-light" if invert else "light-on-dark"
        print("    Blob " + mode_label + ": " + str(len(keypoints)) + " found")

    return all_dots


# ── Разделение слитого контура через distance transform ───────
def _split_contour_dt(cnt_mask, dot_radius, min_area):
    if np.max(cnt_mask) == 0:
        return []

    dist = cv2.distanceTransform(cnt_mask, cv2.DIST_L2, 5)
    dist = dist.astype(np.float32)
    dist_max = float(np.max(dist))

    if dist_max < 1.0:
        return []

    threshold = max(1.5, dot_radius * 0.7)

    core_mask = (dist >= threshold).astype(np.uint8) * 255

    if np.sum(core_mask) == 0:
        threshold = max(1.0, dot_radius * 0.4)
        core_mask = (dist >= threshold).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    core_mask = cv2.morphologyEx(core_mask, cv2.MORPH_CLOSE,
                                 kernel, iterations=1)

    peak_contours, _ = cv2.findContours(
        core_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    raw_dots = []
    for pc in peak_contours:
        area_pc = cv2.contourArea(pc)
        if area_pc < 1:
            continue
        M = cv2.moments(pc)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        mask_single = np.zeros_like(core_mask)
        cv2.drawContours(mask_single, [pc], -1, 255, -1)
        M_dist = cv2.moments(mask_single.astype(float) * dist)
        if M_dist["m00"] > 0:
            cx = int(M_dist["m10"] / M_dist["m00"])
            cy = int(M_dist["m01"] / M_dist["m00"])

        if 0 <= cy < dist.shape[0] and 0 <= cx < dist.shape[1]:
            if cv2.pointPolygonTest(pc, (cx, cy), False) >= 0:
                r_est = float(dist[cy, cx])
                if r_est < dot_radius * 0.3:
                    continue
                est_area = int(np.pi * r_est * r_est)
                raw_dots.append((cx, cy,
                                 max(est_area, int(area_pc), 1), r_est))

    min_dist = dot_radius * 0.7
    raw_dots.sort(key=lambda d: -d[3])
    kept = []
    for (cx, cy, area, r_est) in raw_dots:
        too_close = False
        for (kx, ky, _, _) in kept:
            if (cx - kx) ** 2 + (cy - ky) ** 2 < min_dist * min_dist:
                too_close = True
                break
        if not too_close:
            kept.append((cx, cy, area, r_est))

    return [(d[0], d[1], d[2]) for d in kept]


# ── Поиск на цветном через HSV ─────────────────────────────────
def _detect_color_dots(hsv, p, sub_mask, save_debug=False, dot_radius=8):
    lower = np.array([p["hue_low"], p["sat_low"], p["val_low"]])
    upper = np.array([p["hue_high"], 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    if sub_mask is not None:
        mask = cv2.bitwise_and(mask, mask, mask=sub_mask)

    if save_debug:
        mask_to_save = mask.copy()
        cv2.imwrite("debug_hsv_mask.png", mask_to_save)
        print("    HSV mask saved: debug_hsv_mask.png")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    if save_debug:
        cv2.imwrite("debug_hsv_mask_clean.png", mask)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []
    total_contours = len(contours)
    small_dots = 0
    big_split = 0
    too_small = 0

    if save_debug:
        dbg_contours = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    else:
        dbg_contours = None

    for ci, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        if dbg_contours is not None:
            color = (0, 255, 0) if area <= p["max_area"] else (0, 0, 255)
            cv2.drawContours(dbg_contours, [cnt], -1, color, 1)
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(dbg_contours, "a=" + str(int(area)),
                            (cx, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                            (255, 255, 0), 1)

        if area < p["min_area"]:
            if area >= 3:
                cnt_mask = np.zeros_like(mask)
                cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)
                sub_dots = _split_contour_dt(cnt_mask, dot_radius,
                                             p["min_area"])
                if sub_dots:
                    candidates.extend(sub_dots)
                    big_split += 1
                    continue
            too_small += 1
            continue

        is_small = area <= p["max_area"]

        if is_small:
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity >= p["min_circ"]:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    candidates.append((cx, cy, int(area)))
                    small_dots += 1
                    continue

            cnt_mask = np.zeros_like(mask)
            cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)
            sub_dots = _split_contour_dt(cnt_mask, dot_radius, p["min_area"])
            if sub_dots:
                candidates.extend(sub_dots)
                big_split += 1
                continue

        cnt_mask = np.zeros_like(mask)
        cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)
        sub_dots = _split_contour_dt(cnt_mask, dot_radius, p["min_area"])

        if sub_dots:
            candidates.extend(sub_dots)
            big_split += 1

    if dbg_contours is not None:
        cv2.imwrite("debug_contours.png", dbg_contours)
        print("    Debug contours saved: debug_contours.png")

    print("    HSV contours: " + str(total_contours)
          + "  small_dots: " + str(small_dots)
          + "  big(split): " + str(big_split)
          + "  too_small: " + str(too_small))

    return candidates


# ── Универсальный поиск ────────────────────────────────────────
def _detect_dots(img, roi=None, sensitivity=50, polygon_mask=None,
                 color_mode="green", dot_radius=8):
    global debug_candidates

    if img is None or img.size == 0:
        return []

    p = _sensitivity_to_params(sensitivity, color_mode, dot_radius)

    if roi is not None:
        x1, y1, x2, y2 = roi
        h_img, w_img = img.shape[:2]
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w_img, int(x2))
        y2 = min(h_img, int(y2))
        if x2 <= x1 or y2 <= y1:
            return []
        sub = img[y1:y2, x1:x2]
        hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
        offset_x, offset_y = x1, y1
        sub_mask = polygon_mask[y1:y2, x1:x2] \
            if polygon_mask is not None else None
    else:
        sub = img
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        offset_x, offset_y = 0, 0
        sub_mask = polygon_mask

    mean_sat = float(np.mean(hsv[:, :, 1]))
    color_name = "GREEN" if color_mode == "green" else "BLUE"

    print("  Mean sat: " + str(round(mean_sat, 1))
          + " — trying HSV " + color_name + " first...")
    print("  HSV range: H[" + str(p["hue_low"]) + "-" + str(p["hue_high"])
          + "] S>=" + str(p["sat_low"]) + " V>=" + str(p["val_low"]))

    hsv_candidates = _detect_color_dots(
        hsv, p, sub_mask,
        save_debug=debug_mode,
        dot_radius=dot_radius,
    )

    print("  HSV found " + str(len(hsv_candidates)) + " candidates.")

    gray_candidates = _detect_grayscale_blobs(sub, p, sub_mask)
    print("  Grayscale found " + str(len(gray_candidates)) + " candidates.")

    all_candidates = hsv_candidates + gray_candidates
    if not all_candidates:
        debug_candidates = []
        return []

    accepted = []
    rejected = []
    min_a = p["min_area"]
    max_a = p["max_area"]
    merge_sq = p["merge_radius"] * p["merge_radius"]

    for cx, cy, area in all_candidates:
        ax = cx + offset_x
        ay = cy + offset_y
        if min_a <= area <= max_a:
            dup = False
            for (ex, ey, _) in accepted:
                if (ax - ex) ** 2 + (ay - ey) ** 2 < merge_sq:
                    dup = True
                    break
            if not dup:
                accepted.append((ax, ay, area))
                debug_candidates.append((ax, ay, area, True))
            else:
                debug_candidates.append((ax, ay, area, False))
        else:
            rejected.append((ax, ay, area))
            debug_candidates.append((ax, ay, area, False))

    print("    Accepted: " + str(len(accepted))
          + "  Rejected: " + str(len(rejected)))

    if not accepted:
        return []

    cluster_r = max(dot_radius * 1.5, p["merge_radius"])
    cluster_sq = cluster_r * cluster_r
    clusters = []

    for cx, cy, area in accepted:
        merged = False
        for i, (ccx, ccy, ccount, carea) in enumerate(clusters):
            dx = cx - ccx
            dy = cy - ccy
            if dx * dx + dy * dy < cluster_sq:
                total_area = carea + area
                new_x = int((ccx * carea + cx * area) / total_area)
                new_y = int((ccy * carea + cy * area) / total_area)
                clusters[i] = (new_x, new_y, ccount + 1, total_area)
                merged = True
                break
        if not merged:
            clusters.append((cx, cy, 1, area))

    dots = [(c[0], c[1]) for c in clusters]
    dots = _sort_dots_bottom_up(dots)
    return dots


# ── Предсказание точек через геометрию ─────────────────────────
def _predict_grid_dots(points, rows, cols, detected_dots, dot_radius):
    if not detected_dots:
        return detected_dots

    final_dots = list(detected_dots)
    added = []
    group_tol = 40
    max_added = 2

    sorted_by_y = sorted(detected_dots, key=lambda p: p[1])
    row_groups = []
    if sorted_by_y:
        cur = [sorted_by_y[0]]
        for i in range(1, len(sorted_by_y)):
            if abs(sorted_by_y[i][1] - cur[-1][1]) > group_tol:
                row_groups.append(cur)
                cur = [sorted_by_y[i]]
            else:
                cur.append(sorted_by_y[i])
        row_groups.append(cur)

    sorted_by_x = sorted(detected_dots, key=lambda p: p[0])
    col_groups = []
    if sorted_by_x:
        cur = [sorted_by_x[0]]
        for i in range(1, len(sorted_by_x)):
            if abs(sorted_by_x[i][0] - cur[-1][0]) > group_tol:
                col_groups.append(cur)
                cur = [sorted_by_x[i]]
            else:
                cur.append(sorted_by_x[i])
        col_groups.append(cur)

    best_row = None
    best_row_run = 0
    for row in row_groups:
        if len(row) < 3:
            continue
        row.sort(key=lambda p: p[0])
        max_run = 1
        cur_run = 1
        for k in range(1, len(row)):
            if row[k][0] - row[k - 1][0] < 40:
                cur_run += 1
                max_run = max(max_run, cur_run)
            else:
                cur_run = 1
        if max_run > best_row_run:
            best_row_run = max_run
            best_row = row

    best_col = None
    best_col_run = 0
    for col in col_groups:
        if len(col) < 3:
            continue
        col.sort(key=lambda p: p[1])
        max_run = 1
        cur_run = 1
        for k in range(1, len(col)):
            if abs(col[k][1] - col[k - 1][1]) < 40:
                cur_run += 1
                max_run = max(max_run, cur_run)
            else:
                cur_run = 1
        if max_run > best_col_run:
            best_col_run = max_run
            best_col = col

    if best_row and len(added) < max_added:
        row = sorted(best_row, key=lambda p: p[0])
        steps = []
        for k in range(len(row) - 1):
            dx = row[k + 1][0] - row[k][0]
            if 5 < dx < 80:
                steps.append(dx)
        if steps and len(steps) >= 2:
            step = float(np.median(steps))
            first = row[0]
            added.append((int(first[0] - step), int(first[1])))

    if best_col and len(added) < max_added:
        col = sorted(best_col, key=lambda p: p[1])
        steps = []
        for k in range(len(col) - 1):
            dy = abs(col[k + 1][1] - col[k][1])
            if 5 < dy < 80:
                steps.append(dy)
        if steps and len(steps) >= 2:
            step = float(np.median(steps))
            ny_dir = 1 if col[-1][1] > col[0][1] else -1
            first = col[0]
            added.append((int(first[0]), int(first[1] - step * ny_dir)))

    merge_r = max(dot_radius * 1.5, 15)
    merge_sq = merge_r * merge_r
    for (nx, ny) in added:
        too_close = False
        for (ex, ey) in final_dots:
            if (nx - ex) ** 2 + (ny - ey) ** 2 < merge_sq:
                too_close = True
                break
        if not too_close:
            final_dots.append((nx, ny))

    final_dots = _sort_dots_bottom_up(final_dots)

    print("  Extrapolation: " + str(len(detected_dots)) + " detected, "
          + str(len(final_dots) - len(detected_dots)) + " added, "
          + str(len(final_dots)) + " total")

    return final_dots


def _detect_and_mark_dots():
    global green_dots, debug_candidates

    if preview_image is None:
        return

    if len(click_points) != 4:
        print("  Need 4 anchor points.")
        green_dots = []
        return

    debug_candidates = []

    h_img, w_img = preview_image.shape[:2]
    polygon_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    pts = np.array(click_points, dtype=np.int32)
    cv2.fillPoly(polygon_mask, [pts], 255)

    min_x = min(pt[0] for pt in click_points)
    max_x = max(pt[0] for pt in click_points)
    min_y = min(pt[1] for pt in click_points)
    max_y = max(pt[1] for pt in click_points)
    roi = (min_x, min_y, max_x, max_y)

    color_name = "GREEN" if dot_color_mode == "green" else "BLUE"
    cur_radius = green_dot_radius if dot_color_mode == "green" \
        else blue_dot_radius

    print("  Search: " + color_name + " r=" + str(cur_radius)
          + "px  sens=" + str(detect_sensitivity))

    p = _sensitivity_to_params(detect_sensitivity, dot_color_mode,
                                cur_radius)
    print("  ref_area=" + str(p["ref_area"])
          + "  accept_area=" + str(p["min_area"]) + "-" + str(p["max_area"]))

    raw_dots = _detect_dots(preview_image, roi=roi,
                            sensitivity=detect_sensitivity,
                            polygon_mask=polygon_mask,
                            color_mode=dot_color_mode,
                            dot_radius=cur_radius)

    print("  Raw dots found: " + str(len(raw_dots)))

    green_dots = _predict_grid_dots(click_points, cur_rows, cur_cols,
                                    raw_dots, cur_radius)

    print("  After grid prediction: " + str(len(green_dots)) + " points")

    if _cur_image_path:
        _save_csv(_cur_image_path, green_dots, "GRID_POINTS")

    if show_projection:
        _project_dots_to_road()


# ── Поиск ближайшей точки для ручного редактирования ───────────
def _find_nearest_dot(x, y, max_r=None):
    if not green_dots:
        return -1
    cur_r = green_dot_radius if dot_color_mode == "green" \
        else blue_dot_radius
    if max_r is None:
        max_r = max(cur_r + 4, grab_radius)
    best_idx = -1
    best_sq = max_r * max_r
    for i, (dx, dy) in enumerate(green_dots):
        d_sq = (dx - x) ** 2 + (dy - y) ** 2
        if d_sq < best_sq:
            best_sq = d_sq
            best_idx = i
    return best_idx


def _delete_nearest_dot(x, y):
    global green_dots
    idx = _find_nearest_dot(x, y)
    if idx >= 0:
        removed = green_dots.pop(idx)
        print("  Deleted dot #" + str(idx + 1) + " at ("
              + str(removed[0]) + ", " + str(removed[1]) + ")")
        if _cur_image_path:
            _save_csv(_cur_image_path, green_dots, "GRID_POINTS")
        if show_projection:
            _project_dots_to_road()
    else:
        print("  No dot near cursor.")


# ── Отрисовка ─────────────────────────────────────────────────
def _render():
    global display_image

    if preview_image is None:
        return

    img = preview_image.copy()
    cur_r = green_dot_radius if dot_color_mode == "green" \
        else blue_dot_radius

    if debug_mode and debug_candidates:
        for (cx, cy, area, accepted) in debug_candidates:
            r_eff = int(round(np.sqrt(area / np.pi)))
            if r_eff < 2:
                r_eff = 2
            if accepted:
                cv2.circle(img, (cx, cy), r_eff, (0, 255, 0), 1)
                label = "OK a=" + str(area) + " r=" + str(r_eff)
            else:
                cv2.circle(img, (cx, cy), r_eff, (0, 0, 255), 1)
                label = "NO a=" + str(area) + " r=" + str(r_eff)
            cv2.putText(img, label,
                        (cx + r_eff + 3, cy - r_eff - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (0, 0, 0), 2)
            cv2.putText(img, label,
                        (cx + r_eff + 3, cy - r_eff - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (0, 255, 255) if accepted else (0, 0, 255), 1)

    if dot_color_mode == "blue":
        dot_ring_color = (255, 128, 0)
        dot_text_color = (255, 255, 255)
    else:
        dot_ring_color = (0, 255, 255)
        dot_text_color = (0, 255, 255)

    for i, (gx, gy) in enumerate(green_dots):
        is_dragged = (i == drag_dot_index)
        ring_c = (0, 0, 255) if is_dragged else dot_ring_color
        ring_thick = 3 if is_dragged else 2
        cv2.circle(img, (gx, gy), cur_r, ring_c, ring_thick)
        cv2.line(img, (gx - 3, gy), (gx + 3, gy), ring_c, 1)
        cv2.line(img, (gx, gy - 3), (gx, gy + 3), ring_c, 1)
        label = str(i + 1)
        tx = gx + cur_r + 4
        ty = gy + 5
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                cv2.putText(img, label, (tx + dx, ty + dy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 0), 1)
        cv2.putText(img, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    dot_text_color, 1)

    # ── Проекции точек на дорогу ──                                
    if show_projection and projected_dots:                          
        proj_color = (0, 255, 255)  # Жёлтый цвет                   
        for i, (px, py) in enumerate(projected_dots):               
            if i < len(green_dots):                                 
                gx, gy = green_dots[i]                              
                cv2.line(img, (gx, gy), (px, py), proj_color, 1)    
            cv2.circle(img, (px, py), 6, proj_color, 1)             
            cv2.line(img, (px - 3, py), (px + 3, py), proj_color, 1)
            cv2.line(img, (px, py - 3), (px, py + 3), proj_color, 1)

    if show_grid and len(click_points) == 4:
        draw_perspective_grid(
            img, click_points,
            rows=cur_rows, cols=cur_cols,
            color=grid_color, thickness=grid_thickness,
        )

    if selected_line_mode is not None and len(click_points) == 4:
        line_color = (0, 165, 255)
        if selected_line_mode == "row":
            idx = min(selected_line_idx, cur_rows)
            start, end = _grid_line_row(click_points, idx,
                                        cur_rows, cur_cols)
            cv2.line(img, start, end, line_color, 3)
            pts = _row_intersections_sorted(click_points, idx,
                                            cur_rows, cur_cols)
            for i, pt in enumerate(pts):
                cv2.circle(img, pt, 5, line_color, 2)
                label = "X" + str(i + 1)
                tx, ty = pt[0] + 9, pt[1] + 5
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        if dx == 0 and dy == 0:
                            continue
                        cv2.putText(img, label, (tx + dx, ty + dy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 0, 0), 1)
                cv2.putText(img, label, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            line_color, 1)
        elif selected_line_mode == "col":
            idx = min(selected_line_idx, cur_cols)
            start, end = _grid_line_col(click_points, idx,
                                        cur_rows, cur_cols)
            cv2.line(img, start, end, line_color, 3)
            pts = _col_intersections_sorted(click_points, idx,
                                             cur_rows, cur_cols)
            for i, pt in enumerate(pts):
                cv2.circle(img, pt, 5, line_color, 2)
                label = "X" + str(i + 1)
                tx, ty = pt[0] + 9, pt[1] + 5
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        if dx == 0 and dy == 0:
                            continue
                        cv2.putText(img, label, (tx + dx, ty + dy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 0, 0), 1)
                cv2.putText(img, label, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            line_color, 1)

    cv2.rectangle(img, (pad_left, pad_top),
                  (pad_left + img_w - 1, pad_top + img_h - 1),
                  (80, 80, 80), 1)

    panel_x = pad_left + img_w - 120
    panel_y = pad_top + img_h - 70
    cv2.rectangle(img, (panel_x - 10, panel_y - 25),
                  (pad_left + img_w - 5, pad_top + img_h - 5),
                  (0, 0, 0), -1)
    cv2.rectangle(img, (panel_x - 10, panel_y - 25),
                  (pad_left + img_w - 5, pad_top + img_h - 5),
                  (200, 200, 200), 1)

    gx, gy = panel_x, panel_y
    cv2.circle(img, (gx, gy), green_dot_radius, (0, 255, 0), 1)
    cv2.putText(img, "G:" + str(green_dot_radius),
                (gx + green_dot_radius + 5, gy + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    bx, by = panel_x, panel_y + 25
    cv2.circle(img, (bx, by), blue_dot_radius, (255, 128, 0), 1)
    cv2.putText(img, "B:" + str(blue_dot_radius),
                (bx + blue_dot_radius + 5, by + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 128, 0), 1)

    cv2.rectangle(img, (pad_left + img_w - 350, pad_top + 5),
                  (pad_left + img_w - 5, pad_top + 35),
                  (0, 0, 0), -1)
    cv2.rectangle(img, (pad_left + img_w - 350, pad_top + 5),
                  (pad_left + img_w - 5, pad_top + 35),
                  (200, 200, 200), 1)
    color_label = "GREEN" if dot_color_mode == "green" else "BLUE"
    dbg = " DBG" if debug_mode else ""
    add_lbl = " | ADD DOT" if add_dot_mode else ""
    proj_lbl = " | PROJ" if show_projection else ""
    sens_text = (str(color_label) + " r=" + str(cur_r)
                 + " Sens:" + str(detect_sensitivity) + dbg + add_lbl
                 + proj_lbl + " (D A X 0=dbg)")
    cv2.putText(img, sens_text,
                (pad_left + img_w - 345, pad_top + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                (200, 200, 200), 1)

    if add_dot_mode:
        cv2.rectangle(img, (pad_left + 155, pad_top + 5),
                      (pad_left + 285, pad_top + 35),
                      (0, 0, 80), -1)
        cv2.rectangle(img, (pad_left + 155, pad_top + 5),
                      (pad_left + 285, pad_top + 35),
                      (255, 255, 0), 1)
        cv2.putText(img, "** ADD DOT **",
                    (pad_left + 165, pad_top + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 0), 1)
        if _last_mouse_x > 0 or _last_mouse_y > 0:
            mx, my = _last_mouse_x, _last_mouse_y
            cv2.line(img, (mx - 15, my), (mx + 15, my),
                     (255, 255, 0), 1)
            cv2.line(img, (mx, my - 15), (mx, my + 15),
                     (255, 255, 0), 1)
            cv2.circle(img, (mx, my), cur_r, (255, 255, 0), 1)

    if selected_line_mode is not None:
        if selected_line_mode == "row":
            line_text = "Row " + str(selected_line_idx) + "/" + str(cur_rows)
        else:
            line_text = "Col " + str(selected_line_idx) + "/" + str(cur_cols)
        cv2.rectangle(img, (pad_left + 5, pad_top + 5),
                      (pad_left + 145, pad_top + 35),
                      (0, 0, 0), -1)
        cv2.rectangle(img, (pad_left + 5, pad_top + 5),
                      (pad_left + 145, pad_top + 35),
                      (200, 200, 200), 1)
        cv2.putText(img, line_text,
                    (pad_left + 12, pad_top + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 165, 255), 1)

    if len(click_points) >= 2:
        n = len(click_points)
        closed = (n == 4)
        for i in range(n):
            p1 = click_points[i]
            p2 = click_points[(i + 1) % n]
            if not closed and i == n - 1:
                break
            cv2.line(img, p1, p2, (0, 0, 255), 2)

    for i, pt in enumerate(click_points):
        color = (0, 255, 255) if i == drag_index else (0, 0, 255)
        _draw_cross(img, pt, size=7, color=color, thickness=2)
        cv2.putText(img, str(i + 1),
                    (pt[0] + 10, pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    display_image = img
    cv2.imshow(_win_key, display_image)


# ── Мышь ───────────────────────────────────────────────────────
def mouse_callback(event, x, y, flags, param):
    global click_points, drag_index, drag_dot_index
    global _last_mouse_x, _last_mouse_y, add_dot_mode, green_dots

    if event == cv2.EVENT_MOUSEMOVE:
        _last_mouse_x = x
        _last_mouse_y = y

        if drag_dot_index != -1 and drag_dot_index < len(green_dots):
            green_dots[drag_dot_index] = (x, y)
            _expand_if_needed(x, y)
            _fit_canvas_to_points()
        elif drag_index != -1 and drag_index < len(click_points):
            click_points[drag_index] = (x, y)
            _expand_if_needed(x, y)
            _fit_canvas_to_points()

        if add_dot_mode:
            _render()

    elif event == cv2.EVENT_LBUTTONDOWN:
        _last_mouse_x = x
        _last_mouse_y = y

        if add_dot_mode:
            green_dots.append((x, y))
            green_dots = _sort_dots_bottom_up(green_dots)
            add_dot_mode = False
            print("  Dot added at (" + str(x) + ", " + str(y)
                  + ") — total " + str(len(green_dots)))
            if _cur_image_path:
                _save_csv(_cur_image_path, green_dots, "GRID_POINTS")
            if show_projection:
                _project_dots_to_road()
            _render()
            return

        dot_idx = _find_nearest_dot(x, y)
        if dot_idx >= 0:
            drag_dot_index = dot_idx
            print("  Dragging dot #" + str(dot_idx + 1))
            _render()
            return

        nearest = -1
        best_dist_sq = grab_radius * grab_radius
        for i, pt in enumerate(click_points):
            dx = pt[0] - x
            dy = pt[1] - y
            d_sq = dx * dx + dy * dy
            if d_sq < best_dist_sq:
                best_dist_sq = d_sq
                nearest = i

        if nearest != -1:
            drag_index = nearest
        elif len(click_points) < 4:
            click_points.append((x, y))
            drag_index = len(click_points) - 1
            _fit_canvas_to_points()
            if len(click_points) == 4:
                print("4 points set. Auto-search...")
                _detect_and_mark_dots()

        _render()

    elif event == cv2.EVENT_RBUTTONDOWN:
        _last_mouse_x = x
        _last_mouse_y = y
        _delete_nearest_dot(x, y)
        _render()

    elif event == cv2.EVENT_LBUTTONUP:
        if drag_dot_index != -1:
            print("  Dot moved to ("
                  + str(green_dots[drag_dot_index][0]) + ", "
                  + str(green_dots[drag_dot_index][1]) + ")")
            green_dots = _sort_dots_bottom_up(green_dots)
            if _cur_image_path:
                _save_csv(_cur_image_path, green_dots, "GRID_POINTS")
            if show_projection:
                _project_dots_to_road()
        if drag_index != -1:
            if show_projection:
                _project_dots_to_road()
        drag_index = -1
        drag_dot_index = -1
        _render()


# ── Главная функция ────────────────────────────────────────────
def interactive_grid(
    image_path,
    rows=12, cols=14,
    color=(0, 255, 0), thickness=2,
    output_path="result_with_grid.jpg"
):
    global click_points, display_image, preview_image, show_grid
    global cur_rows, cur_cols, grid_color, grid_thickness, green_dots
    global img_resized, img_w, img_h
    global pad_left, pad_right, pad_top, pad_bottom
    global _cur_image_path, _scale, _window_created
    global detect_sensitivity
    global selected_line_mode, selected_line_idx
    global dot_color_mode, debug_mode
    global green_dot_radius, blue_dot_radius
    global add_dot_mode, drag_dot_index
    global cell_width_m, cell_length_m, camera_height_m, track_height_m
    global show_projection, projected_dots, projected_world

    grid_color = color
    grid_thickness = thickness
    green_dots = []
    _cur_image_path = image_path
    _window_created = False

    selected_line_mode = None
    selected_line_idx = 0
    dot_color_mode = "green"
    debug_mode = False
    add_dot_mode = False
    drag_dot_index = -1

    green_dot_radius = 8
    blue_dot_radius = 16

    cell_width_m = 0.5
    cell_length_m = 3.0
    camera_height_m = 2.0
    track_height_m = 1.1
    show_projection = False
    projected_dots = []
    projected_world = []

    pad_left = 0
    pad_right = 0
    pad_top = 0
    pad_bottom = 0

    orig = cv2.imread(image_path)
    if orig is None:
        raise FileNotFoundError("Cannot load: " + image_path)

    h, w = orig.shape[:2]
    max_screen_w = 1600
    _scale = 1.0
    if w > max_screen_w:
        _scale = max_screen_w / w
        orig_resized = cv2.resize(
            orig,
            (int(w * _scale), int(h * _scale)),
            interpolation=cv2.INTER_LANCZOS4,
        )
        print("Resized to "
              + str(orig_resized.shape[1]) + "x"
              + str(orig_resized.shape[0]))
    else:
        orig_resized = orig.copy()
        _scale = 1.0

    img_resized = orig_resized
    img_h, img_w = orig_resized.shape[:2]

    if rows <= 0 or cols <= 0:
        print("=" * 60)
        print("  Grid setup")
        print("=" * 60)
        try:
            user_rows = input("  Rows (default 12): ").strip()
            user_cols = input("  Cols (default 14): ").strip()
            cur_rows = int(user_rows) if user_rows else 12
            cur_cols = int(user_cols) if user_cols else 14
            if cur_rows < 1:
                cur_rows = 1
            if cur_cols < 1:
                cur_cols = 1
        except ValueError:
            cur_rows = 12
            cur_cols = 14
        print("  Rows: " + str(cur_rows) + ", cols: " + str(cur_cols))
        print("=" * 60)
    else:
        cur_rows = rows
        cur_cols = cols

    click_points = []
    show_grid = False

    _rebuild_canvas()

    cv2.namedWindow(_win_key, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(_win_key, preview_image.shape[1],
                     preview_image.shape[0])
    _window_created = True
    cv2.setMouseCallback(_win_key, mouse_callback)

    settings = _load_settings(image_path)
    if settings:
        click_points[:] = [
            (pt[0] + pad_left, pt[1] + pad_top)
            for pt in settings["points"]
        ]
        cur_rows = settings.get("rows", cur_rows)
        cur_cols = settings.get("cols", cur_cols)
        saved_dots = settings.get("green_dots", [])
        green_dots = [
            (d[0] + pad_left, d[1] + pad_top) for d in saved_dots
        ]
        green_dots = _sort_dots_bottom_up(green_dots)
        detect_sensitivity = settings.get("sensitivity", 50)
        dot_color_mode = settings.get("dot_color_mode", "green")
        green_dot_radius = settings.get("green_dot_radius", 8)
        blue_dot_radius = settings.get("blue_dot_radius", 16)
        selected_line_mode = settings.get("selected_line_mode", None)
        selected_line_idx = settings.get("selected_line_idx", 0)
        cell_width_m = settings.get("cell_width_m", 0.5)
        cell_length_m = settings.get("cell_length_m", 3.0)
        camera_height_m = settings.get("camera_height_m", 2.0)
        track_height_m = settings.get("track_height_m", 1.1)
        show_projection = settings.get("show_projection", False)
        print("Settings loaded.")
        _fit_canvas_to_points()

    display_image = preview_image.copy()
    cv2.imshow(_win_key, display_image)

    set_unicode_title(_win_key, _win_title)

    print("=" * 60)
    print("  Interactive perspective grid")
    print("  Numbering: bottom to top, left to right")
    print("=" * 60)
    print()
    print("  LMB     - place/drag anchor OR drag dot")
    print("  RMB     - delete nearest dot")
    print("  A       - add dot mode (next click adds a dot)")
    print("  X       - delete dot nearest to cursor")
    print("  Enter   - toggle grid")
    print("  ] / [   - +1 / -1 col")
    print("  + / -   - +1 / -1 row")
    print("  Q / E   - sensitivity -5 / +5")
    print("  C       - color: GREEN / BLUE")
    print("  1 / 2   - GREEN radius -1 / +1  ("
          + str(green_dot_radius) + ")")
    print("  3 / 4   - BLUE radius  -1 / +1  ("
          + str(blue_dot_radius) + ")")
    print("  0       - DEBUG (show candidates + save masks)")
    print("  D       - search dots")
    print("  F / V   - next / prev row line")
    print("  G / B   - next / prev col line")
    print("  H       - clear line selection")
    print("  P       - toggle road projection")
    print("  N / M   - cell width  -0.05 / +0.05  ("
          + str(cell_width_m) + "m)")
    print("  J / K   - cell length -0.05 / +0.05  ("
          + str(cell_length_m) + "m)")
    print("  T / Y   - camera height -0.05 / +0.05 ("
          + str(camera_height_m) + "m)")
    print("  U / I   - track height -0.05 / +0.05 ("
          + str(track_height_m) + "m)")
    print("  S       - save")
    print("  W       - save .jpg without grid, with projections only")
    print("  R       - reset")
    print("  Esc     - exit")
    print("=" * 60)

    if len(click_points) == 4 and not green_dots:
        _detect_and_mark_dots()

    if show_projection:
        _project_dots_to_road()

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r") or key == ord("R"):
            click_points = []
            show_grid = False
            green_dots = []
            selected_line_mode = None
            selected_line_idx = 0
            add_dot_mode = False
            drag_dot_index = -1
            projected_dots = []
            projected_world = []
            pad_left = 0
            pad_right = 0
            pad_top = 0
            pad_bottom = 0
            _rebuild_canvas()
            _render()
            _safe_resize_window()
            set_unicode_title(_win_key, _win_title)
            print("Reset.")

        elif key == ord("a") or key == ord("A"):
            add_dot_mode = not add_dot_mode
            if add_dot_mode:
                print("  ADD DOT mode ON — left-click to place a dot")
            else:
                print("  ADD DOT mode OFF")
            _render()

        elif key == ord("x") or key == ord("X"):
            if _last_mouse_x > 0 or _last_mouse_y > 0:
                _delete_nearest_dot(_last_mouse_x, _last_mouse_y)
            else:
                print("  Move cursor over the image first.")
            _render()

        elif key == 13 and len(click_points) == 4:
            show_grid = not show_grid
            if show_grid:
                set_unicode_title(_win_key,
                                  "Grid " + str(cur_rows) + "x"
                                  + str(cur_cols))
                print("Grid ON: " + str(cur_rows) + "x" + str(cur_cols))
            else:
                set_unicode_title(_win_key, _win_title)
                print("Grid OFF.")
            _render()

        elif key == ord("]"):
            if len(click_points) == 4:
                cur_cols += 1
                print("Cols: " + str(cur_cols))
                if show_grid:
                    set_unicode_title(_win_key,
                                      "Grid " + str(cur_rows) + "x"
                                      + str(cur_cols))
                if show_projection:
                    _project_dots_to_road()
                _render()

        elif key == ord("["):
            if len(click_points) == 4 and cur_cols > 1:
                cur_cols -= 1
                print("Cols: " + str(cur_cols))
                if show_grid:
                    set_unicode_title(_win_key,
                                      "Grid " + str(cur_rows) + "x"
                                      + str(cur_cols))
                if show_projection:
                    _project_dots_to_road()
                _render()

        elif key == ord("+") or key == ord("="):
            if len(click_points) == 4:
                cur_rows += 1
                print("Rows: " + str(cur_rows))
                if show_grid:
                    set_unicode_title(_win_key,
                                      "Grid " + str(cur_rows) + "x"
                                      + str(cur_cols))
                if show_projection:
                    _project_dots_to_road()
                _render()

        elif key == ord("-") or key == ord("_"):
            if len(click_points) == 4 and cur_rows > 1:
                cur_rows -= 1
                print("Rows: " + str(cur_rows))
                if show_grid:
                    set_unicode_title(_win_key,
                                      "Grid " + str(cur_rows) + "x"
                                      + str(cur_cols))
                if show_projection:
                    _project_dots_to_road()
                _render()

        elif key == ord("q") or key == ord("Q"):
            detect_sensitivity = max(1, detect_sensitivity - 5)
            print("Sensitivity: " + str(detect_sensitivity))
            _render()

        elif key == ord("e") or key == ord("E"):
            detect_sensitivity = min(100, detect_sensitivity + 5)
            print("Sensitivity: " + str(detect_sensitivity))
            _render()

        elif key == ord("c") or key == ord("C"):
            dot_color_mode = "blue" if dot_color_mode == "green" \
                else "green"
            print("Color: " + dot_color_mode.upper())
            _render()

        elif key == ord("1"):
            green_dot_radius = max(2, green_dot_radius - 1)
            print("Green radius: " + str(green_dot_radius))
            _render()

        elif key == ord("2"):
            green_dot_radius = min(100, green_dot_radius + 1)
            print("Green radius: " + str(green_dot_radius))
            _render()

        elif key == ord("3"):
            blue_dot_radius = max(2, blue_dot_radius - 1)
            print("Blue radius: " + str(blue_dot_radius))
            _render()

        elif key == ord("4"):
            blue_dot_radius = min(100, blue_dot_radius + 1)
            print("Blue radius: " + str(blue_dot_radius))
            _render()

        elif key == ord("0"):
            debug_mode = not debug_mode
            print("Debug mode: " + ("ON" if debug_mode else "OFF"))
            _render()

        elif key == ord("d") or key == ord("D"):
            _detect_and_mark_dots()
            _render()

        elif key == ord("f") or key == ord("F"):
            if len(click_points) == 4:
                if selected_line_mode == "row":
                    selected_line_idx = (selected_line_idx + 1) \
                        % (cur_rows + 1)
                else:
                    selected_line_mode = "row"
                    selected_line_idx = 0
                print("Row: " + str(selected_line_idx) + "/"
                      + str(cur_rows))
                _print_intersections()
                _render()

        elif key == ord("v") or key == ord("V"):
            if len(click_points) == 4:
                if selected_line_mode == "row":
                    selected_line_idx = selected_line_idx - 1
                    if selected_line_idx < 0:
                        selected_line_idx = cur_rows
                else:
                    selected_line_mode = "row"
                    selected_line_idx = cur_rows
                print("Row: " + str(selected_line_idx) + "/"
                      + str(cur_rows))
                _print_intersections()
                _render()

        elif key == ord("g") or key == ord("G"):
            if len(click_points) == 4:
                if selected_line_mode == "col":
                    selected_line_idx = (selected_line_idx + 1) \
                        % (cur_cols + 1)
                else:
                    selected_line_mode = "col"
                    selected_line_idx = 0
                print("Col: " + str(selected_line_idx) + "/"
                      + str(cur_cols))
                _print_intersections()
                _render()

        elif key == ord("b") or key == ord("B"):
            if len(click_points) == 4:
                if selected_line_mode == "col":
                    selected_line_idx = selected_line_idx - 1
                    if selected_line_idx < 0:
                        selected_line_idx = cur_cols
                else:
                    selected_line_mode = "col"
                    selected_line_idx = cur_cols
                print("Col: " + str(selected_line_idx) + "/"
                      + str(cur_cols))
                _print_intersections()
                _render()

        elif key == ord("h") or key == ord("H"):
            selected_line_mode = None
            selected_line_idx = 0
            print("Line cleared.")
            _render()

        elif key == ord("p") or key == ord("P"):
            show_projection = not show_projection
            if show_projection:
                print("  Road projection ON")
                _project_dots_to_road()
            else:
                print("  Road projection OFF")
                projected_dots = []
                projected_world = []
            _render()

        elif key == ord("n") or key == ord("N"):
            cell_width_m = max(0.05, round(cell_width_m - 0.05, 2))
            print("  Cell width: " + str(cell_width_m) + "m")
            if show_projection:
                _project_dots_to_road()
            _render()

        elif key == ord("m") or key == ord("M"):
            cell_width_m = round(cell_width_m + 0.05, 2)
            print("  Cell width: " + str(cell_width_m) + "m")
            if show_projection:
                _project_dots_to_road()
            _render()

        elif key == ord("j") or key == ord("J"):
            cell_length_m = max(0.05, round(cell_length_m - 0.05, 2))
            print("  Cell length: " + str(cell_length_m) + "m")
            if show_projection:
                _project_dots_to_road()
            _render()

        elif key == ord("k") or key == ord("K"):
            cell_length_m = round(cell_length_m + 0.05, 2)
            print("  Cell length: " + str(cell_length_m) + "m")
            if show_projection:
                _project_dots_to_road()
            _render()

        elif key == ord("t") or key == ord("T"):
            camera_height_m = max(0.1, round(camera_height_m - 0.05, 2))
            print("  Camera height: " + str(camera_height_m) + "m")
            if show_projection:
                _project_dots_to_road()
            _render()

        elif key == ord("y") or key == ord("Y"):
            camera_height_m = round(camera_height_m + 0.05, 2)
            print("  Camera height: " + str(camera_height_m) + "m")
            if show_projection:
                _project_dots_to_road()
            _render()

        elif key == ord("u") or key == ord("U"):
            track_height_m = max(0.05, round(track_height_m - 0.05, 2))
            print("  Track height (GRZ center): "
                  + str(track_height_m) + "m")
            if show_projection:
                _project_dots_to_road()
            _render()

        elif key == ord("i") or key == ord("I"):
            track_height_m = round(track_height_m + 0.05, 2)
            print("  Track height (GRZ center): "
                  + str(track_height_m) + "m")
            if show_projection:
                _project_dots_to_road()
            _render()

        elif key == ord("s") or key == ord("S"):
            if len(click_points) != 4:
                print("Need 4 points.")
                continue

            _save_settings(image_path, click_points, cur_rows,
                           cur_cols, green_dots, detect_sensitivity)
            if green_dots:
                _save_csv(image_path, green_dots, "GRID_POINTS")

            # Добавляем сохранение полной матрицы                  
            full_grid_points = _generate_full_grid()               
            _save_csv(_cur_image_path, full_grid_points, "0_full")

            if selected_line_mode is not None:
                line_pts = _collect_line_points()
                if line_pts:
                    _save_csv(image_path, line_pts, "0")
            if show_projection and projected_world:
                _save_projection_csv(image_path)

            if not show_grid:
                print("Grid OFF. Only JSON + CSV saved.")
                continue

            cv2.imwrite(output_path, display_image)
            print("Saved: " + output_path)

            if _scale != 1.0:
                scaled_points = [
                    (int((pt[0] - pad_left) / _scale),
                     int((pt[1] - pad_top) / _scale))
                    for pt in click_points
                ]
                full_orig = cv2.imread(image_path)
                thickness_full = max(1, int(thickness / _scale))
                full_result = draw_perspective_grid(
                    full_orig, scaled_points,
                    rows=cur_rows, cols=cur_cols,
                    color=color, thickness=thickness_full,
                )
                for i, pt in enumerate(scaled_points):
                    _draw_cross(full_result, pt, size=7,
                                color=(0, 0, 255), thickness=3)
                    cv2.putText(full_result, str(i + 1),
                                (pt[0] + 10, pt[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 0, 255), 2)
                full_path = output_path.replace(".jpg", "_full.jpg")
                cv2.imwrite(full_path, full_result)
                print("Saved full: " + full_path)

            print("Done!")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            _window_created = False
            return display_image

        elif key == ord("w") or key == ord("W"):
            if len(click_points) != 4:
                print("Need 4 points.")
                continue
            if not show_projection:
                print("Projection is OFF. Press P to enable.")
                continue
            _save_projection_only(image_path, output_path)

        elif key == 27:
            if len(click_points) == 4:
                _save_settings(image_path, click_points,
                               cur_rows, cur_cols,
                               green_dots, detect_sensitivity)
            cv2.destroyAllWindows()
            _window_created = False
            print("Exit. Settings saved.")
            return None

    return None

def _generate_full_grid():
    if len(click_points) != 4:
        return []
    
    H = _compute_homography(click_points, cur_rows, cur_cols)
    grid_points = []
    
    # Генерируем точки построчно
    for row in range(cur_rows + 1):
        for col in range(cur_cols + 1):
            x = col
            y = row
            pt = _persp_pt(H, x, y)
            grid_points.append(pt)
    
    return grid_points
