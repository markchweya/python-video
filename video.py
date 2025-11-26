import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

# MoviePy import compatible with v1.x and v2.x
try:
    from moviepy.editor import VideoClip
except ModuleNotFoundError:
    from moviepy.video.VideoClip import VideoClip

# -----------------------------
# SETTINGS
# -----------------------------
W, H = 1280, 720
FPS = 30
DURATION = 12.0
OUT = "better_cinematic.mp4"

SEED = 42
rng = np.random.default_rng(SEED)

# Optional end-card text (set to "" to remove)
END_TEXT = "Stronger. Every day."

# -----------------------------
# FONT (no assets required)
# -----------------------------
def load_font(size: int):
    candidates = [
        "C:\\Windows\\Fonts\\seguibl.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "arial.ttf",
        "Arial.ttf",
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()

FONT = load_font(52)

# -----------------------------
# HELPERS
# -----------------------------
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))

def smoothstep(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)

def ease(x):
    return smoothstep(x)

def lerp(a, b, t):
    return a + (b - a) * t

def add_vignette(arr, strength=0.55):
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = W / 2.0, H / 2.0
    dx = (xx - cx) / (W / 2.0)
    dy = (yy - cy) / (H / 2.0)
    r = np.sqrt(dx * dx + dy * dy)
    v = 1.0 - strength * (r ** 1.6)
    v = np.clip(v, 0.0, 1.0).astype(np.float32)
    arr *= v[:, :, None]
    return arr

def add_grain(arr, t, amount=10.0):
    local_rng = np.random.default_rng(int(SEED * 1000 + t * 1000))
    noise = local_rng.normal(0, 1.0, (H, W, 1)).astype(np.float32)
    arr += noise * amount
    return np.clip(arr, 0, 255)

def glow_spot(arr, cx, cy, radius, color, intensity=1.0):
    # arr shape: (H,W,3)
    yy, xx = np.mgrid[0:H, 0:W]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    g = np.exp(-d2 / (2 * radius * radius)).astype(np.float32) * 255.0 * intensity
    for c in range(3):
        arr[:, :, c] += g * (color[c] / 255.0)
    return arr

def draw_soft_line(d, p1, p2, rgba, width):
    d.line([p1, p2], fill=rgba, width=width)
    d.line([p1, p2], fill=(rgba[0], rgba[1], rgba[2], max(0, rgba[3] - 80)), width=max(1, width + 2))

# -----------------------------
# WORLD DATA (precomputed)
# -----------------------------
WORLD_W = W * 2  # allow camera pan

# Buildings
buildings = []
x = 0
while x < WORLD_W:
    bw = int(rng.integers(60, 150))
    bh = int(rng.integers(160, 520))
    base = (int(rng.integers(8, 22)), int(rng.integers(10, 26)), int(rng.integers(18, 40)))
    buildings.append((x, bw, bh, base))
    x += bw + int(rng.integers(12, 35))

win_seed = rng.integers(0, 10_000, size=len(buildings))

# Cars
N_CARS = 10
cars = []
for i in range(N_CARS):
    cars.append({
        "lane": int(rng.integers(0, 2)),
        "x": float(rng.uniform(0, WORLD_W)),
        "v": float(rng.uniform(140, 260)),
        "col": (255, 220, 160) if i % 2 == 0 else (180, 210, 255),
    })

# Rain particles
N_RAIN = 500
rain_x = rng.uniform(0, W, N_RAIN)
rain_y = rng.uniform(0, H, N_RAIN)
rain_s = rng.uniform(600, 1200, N_RAIN)
rain_len = rng.uniform(10, 26, N_RAIN)

# Confetti (scene 3)
N_CONF = 220
conf_x = rng.uniform(0, W, N_CONF)
conf_y = rng.uniform(-H, 0, N_CONF)
conf_vy = rng.uniform(120, 380, N_CONF)
conf_vx = rng.uniform(-60, 60, N_CONF)
conf_sz = rng.uniform(2, 6, N_CONF)
conf_col = rng.integers(0, 255, size=(N_CONF, 3))

# -----------------------------
# SCENE DRAWING
# -----------------------------
def sky_gradient(scene, t):
    # returns float32 (H,W,3)  ✅ FIXED SHAPE
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]  # (H,1)

    if scene == 1:
        top = np.array([6, 10, 25], np.float32)
        bot = np.array([35, 18, 70], np.float32)
    elif scene == 2:
        top = np.array([10, 10, 14], np.float32)
        bot = np.array([18, 18, 24], np.float32)
    else:
        top = np.array([8, 16, 40], np.float32)
        bot = np.array([120, 70, 140], np.float32)

    # (H,3)
    col = top * (1 - yy) + bot * yy
    # (H,W,3)
    base = np.repeat(col[:, None, :], W, axis=1)
    return base

def draw_city(img, t, camx):
    d = ImageDraw.Draw(img, "RGBA")

    # Moon glow
    mx = int(W * 0.23)
    my = int(H * 0.22)
    d.ellipse((mx - 22, my - 22, mx + 22, my + 22), fill=(230, 235, 255, 255))
    d.ellipse((mx - 55, my - 55, mx + 55, my + 55), fill=(180, 210, 255, 40))

    # Buildings
    horizon = int(H * 0.78)
    for i, (bx, bw, bh, base) in enumerate(buildings):
        x0 = int(bx - camx)
        if x0 > W or x0 + bw < 0:
            continue
        y0 = horizon - bh
        d.rectangle((x0, y0, x0 + bw, horizon), fill=(base[0], base[1], base[2], 255))

        # Windows flicker
        local_rng = np.random.default_rng(int(win_seed[i] + t * 7))
        wx = x0 + 10
        while wx < x0 + bw - 12:
            wy = y0 + 18
            while wy < horizon - 18:
                if local_rng.random() > 0.35:
                    d.rectangle((wx, wy, wx + 6, wy + 8), fill=(255, 220, 120, 180))
                wy += 18
            wx += 16

    # Road
    road_y = int(H * 0.86)
    d.rectangle((0, road_y, W, H), fill=(10, 10, 12, 255))
    d.line((0, road_y, W, road_y), fill=(255, 255, 255, 30), width=2)

    # Cars + headlight trails
    # Frame-step update keeps it stable for MoviePy’s sampling style too.
    dt = 1.0 / FPS
    for c in cars:
        lane_y = road_y + 30 + c["lane"] * 26
        c["x"] = (c["x"] + c["v"] * dt) % WORLD_W
        x = c["x"] - camx
        if x < -100 or x > W + 100:
            continue
        d.line([(x - 60, lane_y), (x, lane_y)], fill=(c["col"][0], c["col"][1], c["col"][2], 110), width=4)
        d.rounded_rectangle((x - 18, lane_y - 10, x + 18, lane_y + 6), radius=5, fill=(25, 25, 30, 255))
        d.ellipse((x + 12, lane_y - 6, x + 18, lane_y), fill=(255, 245, 220, 220))

def draw_rain_and_lightning(img, t):
    d = ImageDraw.Draw(img, "RGBA")

    # Rain
    for i in range(N_RAIN):
        x = float(rain_x[i])
        y = (float(rain_y[i]) + float(rain_s[i]) * t) % (H + 40) - 20
        ln = float(rain_len[i])
        d.line([(x, y), (x + 6, y + ln)], fill=(190, 210, 255, 60), width=1)

    # Lightning flashes
    flash = 0.0
    for center in (5.2, 6.4):
        flash = max(flash, math.exp(-((t - center) ** 2) / (2 * 0.03 ** 2)))
    if flash > 0.02:
        a = int(190 * clamp(flash * 1.8))
        d.rectangle((0, 0, W, H), fill=(255, 255, 255, a))

        # bolt
        bolt_rng = np.random.default_rng(int(SEED * 100 + t * 1000))
        bx = int(W * 0.7)
        pts = [(bx, 0)]
        y = 0
        for _ in range(9):
            y += int(bolt_rng.integers(45, 80))
            bx += int(bolt_rng.integers(-70, 70))
            pts.append((bx, y))
        for i in range(len(pts) - 1):
            draw_soft_line(d, pts[i], pts[i + 1], (255, 255, 255, a), 3)

def character_run_pose(t, base_x, base_y, phase=0.0, jump=0.0):
    w = 2 * math.pi * 2.2 * t + phase
    bob = 10 * math.sin(w) * (1.0 - jump) - 45 * jump
    x = base_x
    y = base_y + bob
    leg1 = math.sin(w) * 0.85
    leg2 = math.sin(w + math.pi) * 0.85
    arm1 = math.sin(w + math.pi) * 0.75
    arm2 = math.sin(w) * 0.75
    return x, y, leg1, leg2, arm1, arm2

def draw_character(img, t, scene, camx):
    d = ImageDraw.Draw(img, "RGBA")

    if scene in (1, 2):
        ground = int(H * 0.72)
        d.rectangle((0, ground, W, ground + 14), fill=(12, 12, 16, 255))
        d.rectangle((0, ground + 14, W, H), fill=(8, 8, 10, 255))
    else:
        ground = int(H * 0.82)

    if scene == 1:
        prog = ease(clamp(t / 4.0))
        px = int(lerp(W * 0.25, W * 0.65, prog))
        jump = 0.0
    elif scene == 2:
        prog = ease(clamp((t - 4.0) / 4.0))
        px = int(lerp(W * 0.20, W * 0.75, prog))
        jump = clamp(1.0 - abs(t - 5.8) / 0.55)
        jump = clamp(jump) ** 1.4

        ground = int(H * 0.72)
        gap_x = int(W * 0.52)
        d.rectangle((gap_x - 60, ground, gap_x + 60, H), fill=(0, 0, 0, 255))
    else:
        prog = ease(clamp((t - 8.0) / 3.2))
        px = int(lerp(W * 0.35, W * 0.52, prog))
        jump = 0.0

    py = ground - 6
    x, y, leg1, leg2, arm1, arm2 = character_run_pose(t, px, py, phase=0.4, jump=jump)

    col = (12, 12, 16, 255)
    head_r = 16
    torso = 56 if scene != 3 else 52
    legL = 52
    armL = 40

    d.ellipse((x - head_r, y - torso - head_r * 2, x + head_r, y - torso), fill=col)
    d.rounded_rectangle((x - 18, y - torso + 6, x + 18, y - 6), radius=10, fill=col)

    ax, ay = x, y - torso + 14
    d.line((ax, ay, ax + int(armL * arm1), ay + int(armL * 0.35)), fill=col, width=8)
    d.line((ax, ay, ax + int(armL * arm2), ay + int(armL * 0.35)), fill=col, width=8)

    lx, ly = x, y - 8
    d.line((lx, ly, lx + int(legL * leg1), ly + legL), fill=col, width=10)
    d.line((lx, ly, lx + int(legL * leg2), ly + legL), fill=col, width=10)

    if scene == 3 and t > 10.8:
        d.line((x, y - torso + 10, x - 30, y - torso - 20), fill=col, width=10)
        d.line((x, y - torso + 10, x + 30, y - torso - 20), fill=col, width=10)

    d.ellipse((x - 32, py + 28, x + 32, py + 40), fill=(0, 0, 0, 90))

def draw_mountains_sunrise(img, t):
    d = ImageDraw.Draw(img, "RGBA")

    def ridge(ampl, basey, speed, seedoff, col):
        xs = np.arange(W, dtype=np.float32)
        y = np.zeros(W, dtype=np.float32)
        for k in range(1, 6):
            y += (1 / k) * np.sin((xs / W) * 2 * math.pi * (k * (2 + seedoff)) + t * speed * (0.3 + 0.1 * k))
        y = (y - y.min()) / (y.max() - y.min() + 1e-6)
        y = basey - ampl * (0.2 + 0.8 * y)
        pts = [(0, H)] + [(int(x), int(y[int(x)])) for x in range(W)] + [(W, H)]
        d.polygon(pts, fill=col)

    ridge(90, int(H * 0.60), 0.6, 1, (42, 32, 78, 255))
    ridge(120, int(H * 0.72), 0.9, 2, (30, 24, 52, 255))
    ridge(170, int(H * 0.86), 1.2, 3, (16, 14, 22, 255))

    fog = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    fd = ImageDraw.Draw(fog, "RGBA")
    for i in range(12):
        y = int(H * 0.55 + i * 10)
        fd.rectangle((0, y, W, y + 10), fill=(255, 255, 255, int(10 - i * 0.6)))
    img.alpha_composite(fog)

def draw_confetti(img, t):
    d = ImageDraw.Draw(img, "RGBA")
    tt = (t - 10.2)
    for i in range(N_CONF):
        x = conf_x[i] + conf_vx[i] * tt
        y = conf_y[i] + conf_vy[i] * tt
        if 0 <= x < W and 0 <= y < H:
            s = conf_sz[i]
            c = conf_col[i]
            d.rectangle((x, y, x + s, y + s), fill=(int(c[0]), int(c[1]), int(c[2]), 190))

def end_card_text(img, t):
    if not END_TEXT or t < 10.2:
        return img

    a = int(255 * ease(clamp((t - 10.2) / 1.2)))
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    bbox = d.textbbox((0, 0), END_TEXT, font=FONT)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (W - tw) // 2
    y = int(H * 0.16)

    d.text((x + 3, y + 3), END_TEXT, font=FONT, fill=(0, 0, 0, int(a * 0.6)))
    d.text((x, y), END_TEXT, font=FONT, fill=(245, 248, 255, a))

    return Image.alpha_composite(img, layer)

# -----------------------------
# FRAME FUNCTION
# -----------------------------
def make_frame(t):
    if t < 4.0:
        scene = 1
        camx = lerp(0, W * 0.65, ease(t / 4.0))
    elif t < 8.0:
        scene = 2
        camx = lerp(W * 0.35, W * 0.95, ease((t - 4.0) / 4.0))
    else:
        scene = 3
        camx = 0

    base = sky_gradient(scene, t)  # (H,W,3)

    if scene == 1:
        base = glow_spot(base, int(W * 0.23), int(H * 0.22), 120, (180, 210, 255), 0.55)
    elif scene == 3:
        sx = int(lerp(int(W * 0.30), int(W * 0.62), ease((t - 8.0) / 4.0)))
        base = glow_spot(base, sx, int(H * 0.33), 240, (255, 200, 120), 0.85)

    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB").convert("RGBA")

    if scene in (1, 2):
        draw_city(img, t, camx)
        draw_character(img, t, scene, camx)
        if scene == 2:
            draw_rain_and_lightning(img, t)
            img = Image.alpha_composite(img, img.filter(ImageFilter.GaussianBlur(0.6)).convert("RGBA"))
    else:
        draw_mountains_sunrise(img, t)
        draw_character(img, t, scene, 0)
        if t > 10.2:
            draw_confetti(img, t)
        img = end_card_text(img, t)

    arr = np.array(img.convert("RGB")).astype(np.float32)
    arr = add_vignette(arr, 0.55)
    arr = add_grain(arr, t, amount=9.0)

    out_img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    out_img = ImageEnhance.Contrast(out_img).enhance(1.10)
    out_img = ImageEnhance.Color(out_img).enhance(1.08)

    return np.array(out_img)

# -----------------------------
# RENDER
# -----------------------------
if __name__ == "__main__":
    print("Rendering better cinematic video (no external assets)...")
    clip = VideoClip(make_frame, duration=DURATION)
    clip.write_videofile(
        OUT,
        fps=FPS,
        codec="libx264",
        audio=False,
        preset="medium",
        threads=4,
    )
    clip.close()
    print(f"Done: {OUT}")
