# -*- coding: utf-8 -*-
"""
transformations.py — отрисовка сетки перспективы.
"""
import cv2
import numpy as np


def draw_perspective_grid(img, points, rows=12, cols=14,
                          color=(0, 255, 0), thickness=2):
    """
    Рисует сетку перспективы по 4 точкам привязки.
    points — [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] в порядке:
      левый-верхний, правый-верхний, правый-нижний, левый-нижний.
    """
    if len(points) != 4:
        return img

    src = np.float32(points)
    w_grid, h_grid = cols * 50, rows * 50
    dst = np.float32([
        [0, 0],
        [w_grid, 0],
        [w_grid, h_grid],
        [0, h_grid],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)

    # Горизонтальные линии
    for r in range(rows + 1):
        y = int(h_grid * r / rows)
        pts_src = []
        for c in range(cols + 1):
            x = int(w_grid * c / cols)
            pt = np.array([x, y, 1.0])
            pt_img = M_inv @ pt
            pt_img = pt_img / pt_img[2]
            pts_src.append((int(pt_img[0]), int(pt_img[1])))
        for i in range(len(pts_src) - 1):
            cv2.line(img, pts_src[i], pts_src[i + 1], color, thickness)

    # Вертикальные линии
    for c in range(cols + 1):
        x = int(w_grid * c / cols)
        pts_src = []
        for r in range(rows + 1):
            y = int(h_grid * r / rows)
            pt = np.array([x, y, 1.0])
            pt_img = M_inv @ pt
            pt_img = pt_img / pt_img[2]
            pts_src.append((int(pt_img[0]), int(pt_img[1])))
        for i in range(len(pts_src) - 1):
            cv2.line(img, pts_src[i], pts_src[i + 1], color, thickness)

    return img
