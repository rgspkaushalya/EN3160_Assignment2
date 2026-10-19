from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
IMAGES = ROOT / "images"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def check_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}\nCheck that the supplied images are in ./images.")
    return image


def save_rgb_figure(path: Path, dpi: int = 180) -> None:
    plt.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()


# ---------------------------------------------------------------------------
# Q1 - LoG scale-space extrema for sunflower heads
# ---------------------------------------------------------------------------
def detect_log_blobs(image_path: Path) -> tuple[np.ndarray, list[dict], list[dict]]:
    # Assignment explicitly specifies reduced-color reading by a factor of four.
    bgr = check_image(image_path, cv2.IMREAD_REDUCED_COLOR_4)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # sigma is measured in pixels in the reduced (360 x 360) image.
    sigmas = np.arange(1.0, 10.0, 0.3, dtype=np.float32)
    responses = []
    for sigma in sigmas:
        smoothed = cv2.GaussianBlur(
            gray, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma),
            borderType=cv2.BORDER_REPLICATE
        )
        # Scale-normalized Laplacian: sigma^2 * |LoG response|.
        lap = cv2.Laplacian(smoothed, cv2.CV_32F, ksize=3)
        responses.append((sigma * sigma) * np.abs(lap))
    scale_space = np.stack(responses, axis=0)

    # Extrema across adjacent scales and a local 7x7 pixel neighbourhood.
    local_max = maximum_filter(scale_space, size=(3, 7, 7), mode="nearest")
    extremum_mask = (scale_space == local_max)
    extrema = np.argwhere(extremum_mask)
    values = scale_space[extremum_mask]

    candidates: list[dict] = []
    height, width = gray.shape
    for (scale_i, y, x), score in zip(extrema, values):
        sigma = float(sigmas[int(scale_i)])
        # The subject is the sunflower field, not the sky. Ignore its horizon,
        # a narrow border, weak responses, and detections at the scale limit.
        if y < int(0.50 * height) or x < 8 or y < 8 or x >= width - 8 or y >= height - 8:
            continue
        if sigma >= float(sigmas[-1]) - 0.05 or float(score) < 0.25:
            continue
        radius = float(np.sqrt(2.0) * sigma)
        candidates.append({
            "x": int(x), "y": int(y), "sigma": sigma,
            "radius": radius, "response": float(score)
        })
    candidates.sort(key=lambda z: z["response"], reverse=True)

    # Radius-aware non-maximum suppression reduces duplicate detections across
    # neighbouring scales without suppressing adjacent sunflower heads.
    kept: list[dict] = []
    for candidate in candidates:
        separated = True
        for prev in kept:
            d = np.hypot(candidate["x"] - prev["x"], candidate["y"] - prev["y"])
            if d <= max(4.0, 0.55 * (candidate["radius"] + prev["radius"])):
                separated = False
                break
        if separated:
            kept.append(candidate)

    # Draw the strongest detections in the field while ensuring the largest
    # accepted scales are visible in the figure too.
    draw_items = kept[:65]
    # Ensure at least the largest 8 accepted blobs are shown.
    for item in sorted(kept, key=lambda z: z["radius"], reverse=True)[:8]:
        if item not in draw_items:
            draw_items.append(item)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    ax.imshow(rgb)
    for item in draw_items:
        c = plt.Circle((item["x"], item["y"]), item["radius"], fill=False,
                       edgecolor="#ff3b30", linewidth=0.8, alpha=0.85)
        ax.add_patch(c)
        ax.plot(item["x"], item["y"], "+", color="#00e5ff", markersize=3.0, mew=0.6)
    ax.set_title("Sunflower blob detection using scale-normalized LoG extrema")
    ax.set_xlim(0, width - 1); ax.set_ylim(height - 1, 0)
    ax.set_xlabel("x (pixels in reduced image)"); ax.set_ylabel("y (pixels in reduced image)")
    ax.axis("on")
    save_rgb_figure(OUT / "q1_log_blobs.png")

    largest = sorted(kept, key=lambda z: z["radius"], reverse=True)[:8]
    with (OUT / "q1_largest_circles.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "x", "y", "sigma", "radius", "response"])
        writer.writeheader()
        for i, item in enumerate(largest, 1):
            writer.writerow({"rank": i, **{k: round(item[k], 3) if isinstance(item[k], float) else item[k]
                                             for k in ["x", "y", "sigma", "radius", "response"]}})
    return bgr, kept, largest


# ---------------------------------------------------------------------------
# Q2 - Custom RANSAC for line, followed by RANSAC circle and refinement
# ---------------------------------------------------------------------------
def generate_noisy_line_circle(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Follows Listing 1 with np.random.seed(0) added to make the run repeatable.
    np.random.seed(seed)
    N = 100
    half_n = N // 2
    r = 10.0
    x0_gt, y0_gt = 2.0, 3.0
    s = r / 16.0
    t = np.random.uniform(0, 2 * np.pi, half_n)
    n = s * np.random.randn(half_n)
    x = x0_gt + (r + n) * np.cos(t)
    y = y0_gt + (r + n) * np.sin(t)
    X_circ = np.column_stack((x, y))
    s = 1.0
    m, b = -1.0, 2.0
    x = np.linspace(-12, 12, half_n)
    y = m * x + b + s * np.random.randn(half_n)
    X_line = np.column_stack((x, y))
    X = np.vstack((X_circ, X_line))
    return X, X_circ, X_line


def line_from_two_points(p1: np.ndarray, p2: np.ndarray) -> Optional[np.ndarray]:
    v = p2 - p1
    normal = np.array([-v[1], v[0]], dtype=float)
    norm = np.linalg.norm(normal)
    if norm < 1e-10:
        return None
    normal /= norm  # Enforce a^2 + b^2 = 1.
    d = float(normal @ p1)
    # Canonical sign makes plots/parameters easier to compare between runs.
    if d < 0:
        normal, d = -normal, -d
    return np.array([normal[0], normal[1], d], dtype=float)


def line_distances(points: np.ndarray, model: np.ndarray) -> np.ndarray:
    a, b, d = model
    # Valid as a signed normal distance because (a,b) is unit length.
    return np.abs(a * points[:, 0] + b * points[:, 1] - d)


def tls_line(points: np.ndarray) -> np.ndarray:
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vt[-1]
    normal = normal / np.linalg.norm(normal)
    d = float(normal @ centroid)
    if d < 0:
        normal, d = -normal, -d
    return np.array([normal[0], normal[1], d], dtype=float)


def circle_from_three(points: np.ndarray) -> Optional[np.ndarray]:
    (x1, y1), (x2, y2), (x3, y3) = points
    det = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(det) < 1e-4:
        return None  # Three nearly collinear points do not define a stable circle.
    q1, q2, q3 = x1*x1 + y1*y1, x2*x2 + y2*y2, x3*x3 + y3*y3
    xc = (q1*(y2-y3) + q2*(y3-y1) + q3*(y1-y2)) / det
    yc = (q1*(x3-x2) + q2*(x1-x3) + q3*(x2-x1)) / det
    radius = float(np.hypot(xc - x1, yc - y1))
    if not np.isfinite(radius) or radius <= 0 or radius > 100:
        return None
    return np.array([xc, yc, radius], dtype=float)


def circle_errors(points: np.ndarray, circle: np.ndarray) -> np.ndarray:
    return np.abs(np.linalg.norm(points - circle[:2], axis=1) - circle[2])


def ransac_line(points: np.ndarray, threshold: float = 1.4, iterations: int = 3000,
                seed: int = 14) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    best_model = None; best_mask = None; best_sample = None; best_count = -1
    for _ in range(iterations):
        sample = rng.choice(len(points), 2, replace=False)
        model = line_from_two_points(points[sample[0]], points[sample[1]])
        if model is None:
            continue
        mask = line_distances(points, model) < threshold
        count = int(mask.sum())
        if count > best_count:
            best_model, best_mask, best_sample, best_count = model, mask, sample, count
    if best_model is None:
        raise RuntimeError("Line RANSAC failed to produce a model.")
    # Refit the line with TLS from the best consensus, then reclassify.
    refined = tls_line(points[best_mask])
    final_mask = line_distances(points, refined) < threshold
    return best_model, refined, best_sample, final_mask


def ransac_circle(points: np.ndarray, threshold: float = 1.2, iterations: int = 6000,
                  seed: int = 27) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    best_model = None; best_mask = None; best_sample = None; best_count = -1
    for _ in range(iterations):
        sample = rng.choice(len(points), 3, replace=False)
        model = circle_from_three(points[sample])
        if model is None:
            continue
        mask = circle_errors(points, model) < threshold
        count = int(mask.sum())
        if count > best_count:
            best_model, best_mask, best_sample, best_count = model, mask, sample, count
    if best_model is None:
        raise RuntimeError("Circle RANSAC failed to produce a model.")
    # Minimize sum of squared radial residuals on RANSAC inliers.
    inlier_points = points[best_mask]
    def objective(params: np.ndarray) -> float:
        xc, yc, radius = params
        err = np.linalg.norm(inlier_points - np.array([xc, yc]), axis=1) - radius
        return float(np.sum(err * err))
    opt = minimize(objective, best_model, method="L-BFGS-B",
                   bounds=[(None, None), (None, None), (1e-6, None)])
    refined = np.asarray(opt.x, dtype=float)
    final_mask = circle_errors(points, refined) < threshold
    return best_model, refined, best_sample, final_mask


def draw_line(ax, model: np.ndarray, x_range: tuple[float, float], **kwargs) -> None:
    a, b, d = model
    xs = np.array(x_range)
    if abs(b) > 1e-8:
        ys = (d - a * xs) / b
        ax.plot(xs, ys, **kwargs)
    else:
        x_const = d / a
        ax.axvline(x_const, **kwargs)


def draw_circle(ax, model: np.ndarray, **kwargs) -> None:
    ax.add_patch(plt.Circle(tuple(model[:2]), model[2], fill=False, **kwargs))


def run_q2() -> dict:
    X, X_circ, X_line = generate_noisy_line_circle(seed=0)
    candidate_line, fitted_line, line_sample, line_inliers = ransac_line(X)
    remaining_mask = ~line_inliers
    remaining = X[remaining_mask]
    remaining_global_indices = np.flatnonzero(remaining_mask)

    candidate_circle, fitted_circle, circle_sample_local, circle_inliers_local = ransac_circle(remaining)
    circle_inliers = np.zeros(len(X), dtype=bool)
    circle_inliers[remaining_global_indices[circle_inliers_local]] = True
    circle_sample_global = remaining_global_indices[circle_sample_local]

    # Single, legible plot with both sample models and refined models.
    fig, ax = plt.subplots(figsize=(7.1, 6.5))
    noninlier_mask = ~(line_inliers | circle_inliers)
    ax.scatter(X[noninlier_mask, 0], X[noninlier_mask, 1], s=18, c="0.70", label="Other / rejected points")
    ax.scatter(X[line_inliers, 0], X[line_inliers, 1], s=22, c="#1677ff", label=f"Line inliers ({line_inliers.sum()})")
    ax.scatter(X[circle_inliers, 0], X[circle_inliers, 1], s=22, c="#ff8c00", label=f"Circle inliers ({circle_inliers.sum()})")
    ax.scatter(X[line_sample, 0], X[line_sample, 1], s=100, marker="s", facecolors="none",
               edgecolors="#003a8c", linewidths=1.5, label="Best line sample (2 points)")
    ax.scatter(X[circle_sample_global, 0], X[circle_sample_global, 1], s=150, marker="*",
               c="#b300b3", edgecolors="black", linewidths=0.4, label="Best circle sample (3 points)")
    draw_line(ax, candidate_line, (-14, 14), color="#43a2ff", linewidth=1.6, linestyle="--",
              label="Line from best RANSAC sample")
    draw_line(ax, fitted_line, (-14, 14), color="#003a8c", linewidth=2.0,
              label="Best-fit line (TLS refit)")
    draw_circle(ax, candidate_circle, color="#ffb366", linewidth=1.6, linestyle="--",
                label="Circle from best RANSAC sample")
    draw_circle(ax, fitted_circle, color="#c45a00", linewidth=2.0,
                label="Best-fit circle (optimized)")
    # Ground truth is included as a useful visual reference, not as an input to RANSAC.
    draw_line(ax, np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 2 / np.sqrt(2)]),
              (-14, 14), color="0.25", linewidth=1.0, linestyle=":", label="Ground-truth line")
    draw_circle(ax, np.array([2.0, 3.0, 10.0]), color="0.25", linewidth=1.0,
                linestyle=":", label="Ground-truth circle")
    ax.set_xlim(-14, 14); ax.set_ylim(-13, 17); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Line then circle estimation using custom RANSAC")
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.legend(fontsize=6.8, loc="upper right", framealpha=0.95, ncol=1)
    save_rgb_figure(OUT / "q2_ransac_line_circle.png")

    line_a, line_b, line_d = fitted_line
    result = {
        "line": {
            "sample_indices_zero_based": [int(v) for v in line_sample],
            "sample_model_a_b_d": [float(v) for v in candidate_line],
            "best_fit_a": float(line_a), "best_fit_b": float(line_b), "best_fit_d": float(line_d),
            "constraint_a2_plus_b2": float(line_a*line_a + line_b*line_b),
            "inliers": int(line_inliers.sum()), "threshold": 1.4
        },
        "circle": {
            "sample_indices_zero_based_in_remaining_set": [int(v) for v in circle_sample_local],
            "sample_indices_zero_based_in_full_set": [int(v) for v in circle_sample_global],
            "sample_model_x_y_r": [float(v) for v in candidate_circle],
            "best_fit_x": float(fitted_circle[0]), "best_fit_y": float(fitted_circle[1]),
            "best_fit_r": float(fitted_circle[2]), "inliers": int(circle_inliers.sum()),
            "threshold": 1.2
        },
        "ground_truth": {"line_slope": -1.0, "line_intercept": 2.0, "circle_x": 2.0, "circle_y": 3.0, "circle_r": 10.0}
    }
    (OUT / "q2_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# Q3 - Planar homography and alpha blending
# ---------------------------------------------------------------------------
def make_poster(path: Path) -> None:
    w, h = 1200, 600
    poster = np.full((h, w, 3), (244, 239, 215), dtype=np.uint8)
    cv2.rectangle(poster, (18, 18), (w-19, h-19), (35, 62, 90), 20)
    cv2.rectangle(poster, (60, 65), (w-60, h-65), (228, 104, 47), 8)
    cv2.putText(poster, "COMPUTER VISION", (95, 240), cv2.FONT_HERSHEY_DUPLEX,
                2.0, (20, 54, 89), 5, cv2.LINE_AA)
    cv2.putText(poster, "PLANAR HOMOGRAPHY DEMO", (97, 330), cv2.FONT_HERSHEY_SIMPLEX,
                1.45, (20, 54, 89), 4, cv2.LINE_AA)
    cv2.circle(poster, (1015, 180), 62, (35, 62, 90), -1, cv2.LINE_AA)
    cv2.circle(poster, (1015, 180), 35, (244, 239, 215), -1, cv2.LINE_AA)
    cv2.line(poster, (980, 390), (1110, 480), (35, 62, 90), 15, cv2.LINE_AA)
    cv2.line(poster, (1110, 390), (980, 480), (35, 62, 90), 15, cv2.LINE_AA)
    cv2.imwrite(str(path), poster)


def warp_and_blend(scene: np.ndarray, texture: np.ndarray, dest_points: np.ndarray,
                   alpha: float = 0.88) -> tuple[np.ndarray, np.ndarray]:
    h_tex, w_tex = texture.shape[:2]
    src_points = np.array([[0, 0], [w_tex-1, 0], [w_tex-1, h_tex-1], [0, h_tex-1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src_points, dest_points.astype(np.float32))
    h_scene, w_scene = scene.shape[:2]
    warped = cv2.warpPerspective(texture, H, (w_scene, h_scene))
    src_mask = np.full((h_tex, w_tex), 255, dtype=np.uint8)
    mask = cv2.warpPerspective(src_mask, H, (w_scene, h_scene))
    # Slightly smooth the mask edges to avoid a hard cut at the quad boundary.
    alpha_mask = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (7, 7), 1.2)
    alpha_mask = np.clip(alpha_mask * alpha, 0.0, 1.0)[:, :, None]
    composite = (scene.astype(np.float32) * (1.0 - alpha_mask) + warped.astype(np.float32) * alpha_mask)
    return np.clip(composite, 0, 255).astype(np.uint8), H


def run_q3() -> None:
    scene = check_image(IMAGES / "wadham.jpg")
    flag = check_image(IMAGES / "flag.png", cv2.IMREAD_UNCHANGED)
    if flag.ndim == 3 and flag.shape[2] == 4:
        flag = cv2.cvtColor(flag, cv2.COLOR_BGRA2BGR)
    poster_path = OUT / "sample_poster.png"
    make_poster(poster_path)
    poster = check_image(poster_path)

    # Four corners in TL, TR, BR, BL order; edit these points to fit your own surfaces.
    right_facade = np.array([[526, 291], [1010, 205], [1018, 462], [526, 516]], dtype=np.float32)
    left_facade = np.array([[0, 179], [520, 287], [520, 482], [0, 426]], dtype=np.float32)
    flag_composite, H1 = warp_and_blend(scene, flag, right_facade, alpha=0.92)
    poster_composite, H2 = warp_and_blend(scene, poster, left_facade, alpha=0.88)
    cv2.imwrite(str(OUT / "q3_wadham_flag.png"), flag_composite)
    cv2.imwrite(str(OUT / "q3_wadham_poster.png"), poster_composite)
    cv2.imwrite(str(OUT / "q3_original_scene.jpg"), scene)
    (OUT / "q3_homographies.json").write_text(
        json.dumps({"flag_to_wadham_right_facade": H1.tolist(), "poster_to_wadham_left_facade": H2.tolist()}, indent=2),
        encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0))
    axes[0].imshow(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB)); axes[0].set_title("Original scene")
    axes[1].imshow(cv2.cvtColor(flag_composite, cv2.COLOR_BGR2RGB)); axes[1].set_title("Flag -> right facade")
    axes[2].imshow(cv2.cvtColor(poster_composite, cv2.COLOR_BGR2RGB)); axes[2].set_title("Poster -> left facade")
    for ax in axes: ax.axis("off")
    fig.tight_layout()
    save_rgb_figure(OUT / "q3_homography_composites.png")


# ---------------------------------------------------------------------------
# Q4 - SIFT matching, RANSAC homography, perspective warp and differences
# ---------------------------------------------------------------------------
def draw_match_figure(template: np.ndarray, source: np.ndarray, k1, k2, matches, inlier_mask: np.ndarray,
                      out_path: Path, title: str) -> None:
    inlier_indices = np.flatnonzero(inlier_mask.ravel().astype(bool))
    # Limit the figure to the 60 strongest inlier matches so individual lines remain readable.
    chosen = sorted(inlier_indices.tolist(), key=lambda i: matches[i].distance)[:60]
    chosen_matches = [matches[i] for i in chosen]
    canvas = cv2.drawMatches(source, k2, template, k1, chosen_matches, None,
                             matchColor=(40, 220, 40), singlePointColor=(200, 200, 200),
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    ax.set_title(title + f" (displaying {len(chosen_matches)} inlier matches)")
    ax.axis("off")
    save_rgb_figure(out_path, dpi=150)


def template_board_mask(template: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
    # Blue solder mask of the template PCB in HSV; use its largest connected region.
    blue = cv2.inRange(hsv, np.array([88, 35, 18], dtype=np.uint8), np.array([135, 255, 255], dtype=np.uint8))
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(blue.shape, dtype=np.uint8)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest)
        cv2.fillConvexPoly(mask, hull, 255)
    # If saturation segmentation fails for a future template, fall back to a broad centered crop.
    if cv2.countNonZero(mask) < 1000:
        h, w = template.shape[:2]
        cv2.rectangle(mask, (int(.15*w), int(.06*h)), (int(.88*w), int(.96*h)), 255, -1)
    return cv2.dilate(mask, np.ones((19, 19), np.uint8), iterations=1)


def make_montage(paths: list[Path], labels: list[str], out_path: Path,
                 cols: int = 2, cell_size: tuple[int, int] = (620, 570),
                 grayscale: bool = False) -> None:
    rows = int(np.ceil(len(paths) / cols))
    cw, ch = cell_size
    canvas = np.full((rows * ch, cols * cw, 3), 255, dtype=np.uint8)
    for i, (path, label) in enumerate(zip(paths, labels)):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if grayscale:
            g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        x0, y0 = (i % cols) * cw, (i // cols) * ch
        cv2.putText(canvas, label, (x0 + 12, y0 + 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (20, 20, 20), 2, cv2.LINE_AA)
        max_w, max_h = cw - 20, ch - 48
        scale = min(max_w / image.shape[1], max_h / image.shape[0])
        new_w, new_h = max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        px = x0 + (cw - new_w) // 2
        py = y0 + 35 + (max_h - new_h) // 2
        canvas[py:py+new_h, px:px+new_w] = resized
    cv2.imwrite(str(out_path), canvas)


def run_q4() -> dict:
    pcb_dir = IMAGES / "pcbs"
    template_path = pcb_dir / "AECC1.jpg"
    template = check_image(template_path)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=2500)
    k1, d1 = sift.detectAndCompute(template_gray, None)
    if d1 is None or len(k1) < 4:
        raise RuntimeError("SIFT could not find enough features in AECC1.jpg.")
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    board_mask = template_board_mask(template)
    th, tw = template.shape[:2]
    match_paths: list[Path] = []
    diff_paths: list[Path] = []
    metrics: dict[str, dict] = {}

    source_names = ["AAAA1.jpg", "ADAD1.jpg", "AECC2.jpg", "AECC3.jpg", "AECC4.jpg"]
    for name in source_names:
        source_path = pcb_dir / name
        source = check_image(source_path)
        source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        k2, d2 = sift.detectAndCompute(source_gray, None)
        if d2 is None or len(k2) < 4:
            raise RuntimeError(f"SIFT could not find enough features in {name}.")

        selected = None
        active_k1, active_k2 = k1, k2
        # Run BFMatcher once per image pair; reuse the nearest-neighbour pairs
        # while widening Lowe's ratio threshold only if consensus is too small.
        pairs = matcher.knnMatch(d2, d1, k=2)
        for ratio in (0.75, 0.80, 0.85):
            good = [m for pair in pairs if len(pair) == 2 for m, n in [pair] if m.distance < ratio * n.distance]
            if len(good) < 4:
                continue
            src_pts = np.float32([k2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0, maxIters=3000, confidence=0.995)
            inliers = int(mask.sum()) if mask is not None else 0
            selected = (ratio, good, H, mask, src_pts, dst_pts, inliers)
            if inliers >= 50:
                break
        # AAAA1 is blurrier and strongly rotated relative to AECC1. A one-way
        # ratio test can map many source descriptors to the same template feature,
        # producing a degenerate homography. Retry this pair with more SIFT points
        # and reciprocal (mutual) Lowe-ratio matches to enforce correspondence quality.
        if selected is None or selected[2] is None or selected[3] is None or (name == "AAAA1.jpg" and selected[6] < 50):
            if name == "AAAA1.jpg":
                sift_full = cv2.SIFT_create(nfeatures=8000)
                active_k1, active_d1 = sift_full.detectAndCompute(template_gray, None)
                active_k2, active_d2 = sift_full.detectAndCompute(source_gray, None)
                forward_pairs = matcher.knnMatch(active_d2, active_d1, k=2)
                backward_pairs = matcher.knnMatch(active_d1, active_d2, k=2)
                ratio_fallback = 0.90
                forward_good = [m for pair in forward_pairs if len(pair) == 2 for m, n in [pair]
                                if m.distance < ratio_fallback * n.distance]
                backward_good = [m for pair in backward_pairs if len(pair) == 2 for m, n in [pair]
                                 if m.distance < ratio_fallback * n.distance]
                mutual_pairs = {(m.queryIdx, m.trainIdx) for m in backward_good}
                good_fallback = [m for m in forward_good if (m.trainIdx, m.queryIdx) in mutual_pairs]
                src_f = np.float32([active_k2[m.queryIdx].pt for m in good_fallback]).reshape(-1, 1, 2)
                dst_f = np.float32([active_k1[m.trainIdx].pt for m in good_fallback]).reshape(-1, 1, 2)
                H_f, mask_f = cv2.findHomography(src_f, dst_f, cv2.RANSAC, 5.0,
                                                maxIters=10000, confidence=0.995)
                inliers_f = int(mask_f.sum()) if mask_f is not None else 0
                selected = (ratio_fallback, good_fallback, H_f, mask_f, src_f, dst_f, inliers_f)
            if selected is None or selected[2] is None or selected[3] is None or selected[6] < 4:
                metrics[name] = {"status": "homography failed", "source_keypoints": len(active_k2), "template_keypoints": len(active_k1)}
                continue
        ratio, good, H, mask, src_pts, dst_pts, inlier_count = selected
        aligned = cv2.warpPerspective(source, H, (tw, th), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        diff = cv2.absdiff(template, aligned)
        # Show grayscale absolute difference only where the reference board lies, so
        # background/empty pixels do not dominate the visual comparison.
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        diff_board = np.zeros_like(diff_gray)
        diff_board[board_mask > 0] = diff_gray[board_mask > 0]
        # Percentile scaling suppresses a few extreme pixels so subtle differences
        # in near-identical AECC2/3/4 photos remain visible. The raw masked grayscale
        # absolute difference is also saved separately for numeric fidelity.
        board_values = diff_board[board_mask > 0]
        scale_98 = max(float(np.percentile(board_values, 98)), 1.0) if board_values.size else 1.0
        diff_display = np.clip(diff_board.astype(np.float32) * (255.0 / scale_98), 0, 255).astype(np.uint8)

        stem = Path(name).stem
        cv2.imwrite(str(OUT / f"q4_{stem}_aligned.jpg"), aligned)
        cv2.imwrite(str(OUT / f"q4_{stem}_difference.png"), diff_display)
        cv2.imwrite(str(OUT / f"q4_{stem}_difference_raw.png"), diff_board)
        draw_match_figure(template, source, active_k1, active_k2, good, mask, OUT / f"q4_{stem}_matches.png",
                          f"SIFT + BFMatcher + RANSAC: {name} -> AECC1")
        match_paths.append(OUT / f"q4_{stem}_matches.png")
        diff_paths.append(OUT / f"q4_{stem}_difference.png")
        inlier_ratio = inlier_count / max(1, len(good))
        metrics[name] = {
            "source_keypoints": len(active_k2), "template_keypoints": len(active_k1),
            "ratio_test_threshold": float(ratio), "good_matches": len(good),
            "ransac_inliers": inlier_count, "inlier_ratio": float(inlier_ratio),
            "homography": H.tolist(),
            "board_difference_mean_gray": float(cv2.mean(diff_gray, mask=board_mask)[0]),
        }

    (OUT / "q4_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    labels = [p.name.replace("q4_", "").replace("_matches.png", " -> AECC1") for p in match_paths]
    if match_paths:
        make_montage(match_paths, labels, OUT / "q4_matches_montage.jpg", cols=3, cell_size=(640, 460))
    diff_labels = [p.name.replace("q4_", "").replace("_difference.png", " difference") for p in diff_paths]
    if diff_paths:
        make_montage(diff_paths, diff_labels, OUT / "q4_differences_montage.jpg", cols=3,
                     cell_size=(480, 500), grayscale=True)
    aligned_paths = [OUT / f"q4_{Path(n).stem}_aligned.jpg" for n in source_names if (OUT / f"q4_{Path(n).stem}_aligned.jpg").exists()]
    aligned_labels = [p.name.replace("q4_", "").replace("_aligned.jpg", " aligned") for p in aligned_paths]
    if aligned_paths:
        make_montage(aligned_paths, aligned_labels, OUT / "q4_aligned_montage.jpg", cols=3, cell_size=(480, 500))
    return metrics


def write_run_summary(q1_largest: list[dict], q2: dict, q4: dict) -> None:
    summary = {
        "q1": {"scale_sigma_min": 1.0, "scale_sigma_max": 9.7, "scale_step": 0.3,
               "scale_normalization": "sigma^2 * abs(Laplacian(Gaussian(image, sigma)))",
               "largest_circles": q1_largest},
        "q2": q2,
        "q4": q4,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["EN3160 Assignment 2 outputs", "==========================", "",
             "Question 1: LoG scale extrema", "sigma range: 1.0 to 9.7 pixels (step 0.3 pixels)",
             "Largest detected blobs (coordinates are in the reduced image):"]
    for idx, item in enumerate(sorted(q1_largest, key=lambda z: z["radius"], reverse=True), 1):
        lines.append(f"{idx}. (x={item['x']}, y={item['y']}), sigma={item['sigma']:.2f}, radius={item['radius']:.2f}px, response={item['response']:.3f}")
    lines.extend(["", "Question 2: RANSAC", f"Line inliers: {q2['line']['inliers']}; fitted (a,b,d)=({q2['line']['best_fit_a']:.4f}, {q2['line']['best_fit_b']:.4f}, {q2['line']['best_fit_d']:.4f})",
                  f"Circle inliers: {q2['circle']['inliers']}; fitted centre=({q2['circle']['best_fit_x']:.4f}, {q2['circle']['best_fit_y']:.4f}), radius={q2['circle']['best_fit_r']:.4f}", "",
                  "Question 4: SIFT/RANSAC metrics"])
    for name, m in q4.items():
        if m.get("status"):
            lines.append(f"{name}: {m['status']}")
        else:
            lines.append(f"{name}: good matches={m['good_matches']}, inliers={m['ransac_inliers']}, inlier ratio={m['inlier_ratio']:.3f}, mean board abs difference={m['board_difference_mean_gray']:.2f}")
    (OUT / "run_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    required = [IMAGES / "the_berry_farms_sunflower_field.jpeg", IMAGES / "wadham.jpg", IMAGES / "flag.png",
                IMAGES / "pcbs" / "AECC1.jpg"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required input(s):\n" + "\n".join(missing))
    print("[1/4] LoG scale-space detection ...")
    _, q1_kept, q1_largest = detect_log_blobs(IMAGES / "the_berry_farms_sunflower_field.jpeg")
    print(f"    accepted detections: {len(q1_kept)}; wrote outputs/q1_log_blobs.png and q1_largest_circles.csv")
    print("[2/4] Custom line and circle RANSAC ...")
    q2 = run_q2()
    print(f"    line inliers: {q2['line']['inliers']}; circle inliers: {q2['circle']['inliers']}")
    print("[3/4] Homography and image compositing ...")
    run_q3()
    print("    wrote two homography composites")
    print("[4/4] SIFT matching and PCB alignment ...")
    q4 = run_q4()
    for name, item in q4.items():
        if "good_matches" in item:
            print(f"    {name}: {item['good_matches']} good matches, {item['ransac_inliers']} inliers")
        else:
            print(f"    {name}: {item.get('status', 'unknown result')}")
    write_run_summary(q1_largest, q2, q4)
    print(f"Done. Results are in: {OUT}")
    print("Review images and summary values before submitting the report.")


if __name__ == "__main__":
    main()
