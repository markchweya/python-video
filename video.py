import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

# MoviePy import compatible with v1.x and v2.x
try:
    from moviepy.editor import VideoClip
except ModuleNotFoundError:
    from moviepy.video.VideoClip import VideoClip

# =============================
# CONFIG (EDIT THESE)
# =============================
W, H = 1280, 720
FPS = 30
DURATION = 10.0
OUT = "moving_ad.mp4"

COMPANY = "MoveNest Movers"
TAGLINE = "We pack. We move. You settle in."
CTA = "Call/WhatsApp: +254 700 000 000  •  movenest.co.ke"

# Visual copy on end card
END_HEADLINE = "Your home, safely moved."
END_SUB = "Door-to-door • Packing • Unpacking • Careful handling"

SEED = 7
rng = np.random.default_rng(SEED)

# =============================
# FONT (no assets required)
# =============================
def load_font(size: int, bold=False):
    candidates = []
    if bold:
        candidates += [
            "C:\\Windows\\Fonts\\seguibl.ttf",
            "C:\\Windows\\Fonts\\arialbd.ttf",
        ]
    candidates += [
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\segoeui.ttf",
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

FONT_H1 = load_font(58, bold=True)
FONT_H2 = load_font(30, bold=False)
FONT_CTA = load_font(26, bold=False)

# =============================
# EASING / HELPERS
# =============================
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))

def smoothstep(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)

def ease(x):
    return smoothstep(x)

def lerp(a, b, t):
    return a + (b - a) * t

def add_vignette(arr, strength=0.50):
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = W / 2.0, H / 2.0
    dx = (xx - cx) / (W / 2.0)
    dy = (yy - cy) / (H / 2.0)
    r = np.sqrt(dx * dx + dy * dy)
    v = 1.0 - strength * (r ** 1.5)
    v = np.clip(v, 0.0, 1.0).astype(np.float32)
    arr *= v[:, :, None]
    return arr

def add_grain(arr, t, amount=7.0):
    local_rng = np.random.default_rng(int(SEED * 1000 + t * 1000))
    noise = local_rng.normal(0, 1.0, (H, W, 1)).astype(np.float32)
    arr += noise * amount
    return np.clip(arr, 0, 255)

def glow(arr, cx, cy, radius, color, intensity=1.0):
    yy, xx = np.mgrid[0:H, 0:W]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    g = np.exp(-d2 / (2 * radius * radius)).astype(np.float32) * 255.0 * intensity
    for c in range(3):
        arr[:, :, c] += g * (color[c] / 255.0)
    return arr

def text_center(draw, xy, text, font, fill, shadow=True, shadow_offset=(3, 3), shadow_alpha=140):
    x, y = xy
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    px = int(x - tw / 2)
    py = int(y - th / 2)
    if shadow:
        r, g, b, a = fill
        draw.text((px + shadow_offset[0], py + shadow_offset[1]), text, font=font,
                  fill=(0, 0, 0, int(min(255, a * shadow_alpha / 255))))
    draw.text((px, py), text, font=font, fill=fill)

# =============================
# BACKGROUND
# =============================
# subtle dust particles
N_DUST = 120
dust = np.column_stack([
    rng.uniform(0, W, N_DUST),
    rng.uniform(0, H, N_DUST),
    rng.uniform(10, 45, N_DUST),   # speed
    rng.uniform(1.2, 2.8, N_DUST), # size
    rng.uniform(50, 140, N_DUST),  # alpha
])

def make_bg(t):
    # cinematic dark-to-warm gradient
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]  # (H,1)
    top = np.array([10, 12, 22], np.float32)
    bot = np.array([70, 40, 95], np.float32)
    col = top * (1 - yy) + bot * yy              # (H,3) via broadcasting
    base = np.repeat(col[:, None, :], W, axis=1) # (H,W,3)

    # warm spotlight behind the box
    base = glow(base, int(W * 0.5), int(H * 0.48), 280, (255, 200, 140), 0.45)
    base = glow(base, int(W * 0.5), int(H * 0.52), 520, (150, 110, 255), 0.20)

    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB").convert("RGBA")
    d = ImageDraw.Draw(img, "RGBA")

    # floating dust motes
    for i in range(N_DUST):
        x, y, sp, sz, a = dust[i]
        yy = (y + sp * t) % (H + 50) - 25
        xx = (x + 18 * math.sin(t * 0.4 + i)) % W
        r = sz
        d.ellipse((xx - r, yy - r, xx + r, yy + r), fill=(255, 245, 230, int(a)))

    img = Image.alpha_composite(img, img.filter(ImageFilter.GaussianBlur(0.6)).convert("RGBA"))
    return img

# =============================
# BOX + HOUSE DRAW
# =============================
def poly(draw, pts, fill):
    draw.polygon([(int(x), int(y)) for x, y in pts], fill=fill)

def draw_box_and_house(img, t):
    """
    Timeline:
      0.0 - 2.0 : box idle (closed)
      2.0 - 5.0 : lid opens
      4.0 - 6.5 : house rises out (reveal)
      6.5+      : hold open + end copy
    """
    d = ImageDraw.Draw(img, "RGBA")

    # open progress
    open_p = ease(clamp((t - 2.0) / 3.0))  # 0..1
    # house reveal (starts before fully open)
    reveal_p = ease(clamp((t - 4.0) / 2.5))  # 0..1

    # Box geometry (isometric-ish)
    cx, cy = W * 0.5, H * 0.56
    bw, bh = 420, 220
    depth = 120  # "3D" depth

    # Key points
    p1 = (cx - bw/2, cy - bh/2)              # top-front-left
    p2 = (cx + bw/2, cy - bh/2)              # top-front-right
    p3 = (cx + bw/2 + depth, cy - bh/2 - depth)  # top-back-right
    p4 = (cx - bw/2 + depth, cy - bh/2 - depth)  # top-back-left

    f1 = (cx - bw/2, cy + bh/2)              # bottom-front-left
    f2 = (cx + bw/2, cy + bh/2)              # bottom-front-right
    s2 = (cx + bw/2 + depth, cy + bh/2 - depth)  # bottom-back-right

    # colors
    front = (210, 165, 115, 255)
    side  = (185, 140, 95, 255)
    topc  = (230, 185, 130, 255)
    edge  = (120, 85, 55, 180)

    # Shadow on ground
    sh_w = bw * 0.95
    sh_h = 65
    sh_y = cy + bh/2 + 64
    d.ellipse((cx - sh_w/2, sh_y - sh_h/2, cx + sh_w/2, sh_y + sh_h/2), fill=(0, 0, 0, 110))

    # ----- Interior mask (opening polygon) -----
    opening_poly = [p1, p2, p3, p4]

    # Draw the interior "house" on a separate layer, then clip into opening
    interior = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    idraw = ImageDraw.Draw(interior, "RGBA")

    # House rises out
    rise = lerp(70, -120, reveal_p)
    scale = lerp(0.85, 1.06, reveal_p)

    # House base position aligned to opening center
    open_cx = (p1[0] + p2[0] + p3[0] + p4[0]) / 4
    open_cy = (p1[1] + p2[1] + p3[1] + p4[1]) / 4
    hx = open_cx
    hy = open_cy + rise

    # Draw a neat “mini house” (floorplan + furniture icons)
    house_w = 360 * scale
    house_h = 220 * scale

    # floor panel
    floor = (245, 240, 230, int(220 * reveal_p))
    floor_edge = (25, 25, 28, int(140 * reveal_p))
    fx0, fy0 = hx - house_w/2, hy - house_h/2
    fx1, fy1 = hx + house_w/2, hy + house_h/2

    idraw.rounded_rectangle((fx0, fy0, fx1, fy1), radius=int(26*scale), fill=floor, outline=floor_edge, width=max(1, int(3*scale)))

    # room dividers
    wall = (40, 40, 45, int(120 * reveal_p))
    idraw.line((hx, fy0 + 12*scale, hx, fy1 - 12*scale), fill=wall, width=max(1, int(3*scale)))
    idraw.line((fx0 + 12*scale, hy, fx1 - 12*scale, hy), fill=wall, width=max(1, int(3*scale)))

    # furniture icons (tidy)
    # bed (top-left)
    idraw.rounded_rectangle((fx0+24*scale, fy0+24*scale, hx-18*scale, hy-20*scale),
                            radius=int(16*scale), fill=(220, 230, 255, int(230*reveal_p)))
    idraw.rounded_rectangle((fx0+30*scale, fy0+30*scale, fx0+120*scale, fy0+70*scale),
                            radius=int(12*scale), fill=(255, 255, 255, int(230*reveal_p)))

    # sofa (top-right)
    idraw.rounded_rectangle((hx+18*scale, fy0+34*scale, fx1-26*scale, fy0+98*scale),
                            radius=int(18*scale), fill=(210, 255, 230, int(220*reveal_p)))
    idraw.rounded_rectangle((hx+30*scale, fy0+46*scale, hx+95*scale, fy0+88*scale),
                            radius=int(14*scale), fill=(235, 255, 245, int(220*reveal_p)))

    # dining table (bottom-left)
    idraw.ellipse((fx0+56*scale, hy+34*scale, fx0+140*scale, hy+118*scale),
                  fill=(255, 235, 215, int(230*reveal_p)), outline=(50, 50, 55, int(120*reveal_p)))
    for ang in [0, math.pi/2, math.pi, 3*math.pi/2]:
        cx2 = (fx0+98*scale) + math.cos(ang)*56*scale
        cy2 = (hy+76*scale) + math.sin(ang)*56*scale
        idraw.ellipse((cx2-10*scale, cy2-10*scale, cx2+10*scale, cy2+10*scale),
                      fill=(255, 255, 255, int(220*reveal_p)))

    # plant + boxes (bottom-right)
    idraw.rounded_rectangle((hx+40*scale, hy+50*scale, hx+90*scale, hy+120*scale),
                            radius=int(10*scale), fill=(170, 220, 170, int(220*reveal_p)))
    idraw.polygon([(hx+65*scale, hy+20*scale), (hx+50*scale, hy+55*scale), (hx+80*scale, hy+55*scale)],
                  fill=(120, 200, 120, int(220*reveal_p)))

    # moving boxes (tidy stack)
    bx0 = fx1 - 130*scale
    by0 = fy1 - 120*scale
    for k in range(3):
        ox = (k % 2) * 40 * scale
        oy = k * 26 * scale
        idraw.rounded_rectangle((bx0 + ox, by0 + oy, bx0 + ox + 80*scale, by0 + oy + 50*scale),
                                radius=int(10*scale),
                                fill=(240, 210, 150, int(210*reveal_p)),
                                outline=(120, 90, 60, int(160*reveal_p)),
                                width=max(1, int(2*scale)))

    # Mask interior within the opening area (so it feels "inside the box")
    mask = Image.new("L", (W, H), 0)
    mdraw = ImageDraw.Draw(mask)
    mdraw.polygon([(int(x), int(y)) for x, y in opening_poly], fill=int(255 * clamp(0.25 + 0.75 * open_p)))
    interior_clipped = Image.composite(interior, Image.new("RGBA", (W, H), (0, 0, 0, 0)), mask)

    # Add faint “inside glow” as box opens
    if open_p > 0:
        glow_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow_layer, "RGBA")
        # soft highlight around opening
        gd.polygon([(int(x), int(y)) for x, y in opening_poly], fill=(255, 220, 170, int(38 * open_p)))
        glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(8))
        img.alpha_composite(glow_layer)

    # Composite interior (before box faces so faces sit on top)
    img.alpha_composite(interior_clipped)

    # ----- Box faces -----
    # side face
    poly(d, [p2, p3, s2, f2], side)
    # front face
    poly(d, [p1, p2, f2, f1], front)
    # top face (opening rim) - darker when open
    rim_alpha = int(255 * (1.0 - 0.55 * open_p))
    poly(d, opening_poly, (topc[0], topc[1], topc[2], rim_alpha))

    # edges for crispness
    d.line([p1, p2, f2, f1, p1], fill=edge, width=3)
    d.line([p2, p3, s2, f2, p2], fill=edge, width=3)
    d.line([p1, p4, (p4[0], p4[1] + bh), f1, p1], fill=(0, 0, 0, 45), width=2)

    # ----- Lid flap (hinged on back edge p4-p3) -----
    # flap opens from 0deg -> ~110deg
    theta = lerp(0.0, math.radians(110), open_p)
    lift = math.sin(theta) * (depth * 1.25)
    push = (1.0 - math.cos(theta)) * (depth * 0.95)

    p1o = (p1[0] - push, p1[1] - lift)
    p2o = (p2[0] - push, p2[1] - lift)

    flap = [p4, p3, p2o, p1o]
    # flap shading changes as it opens
    flap_col = (235, 198, 145, int(255 * (0.95)))
    poly(d, flap, flap_col)
    d.line([p4, p3, p2o, p1o, p4], fill=(110, 80, 55, 170), width=3)

    # small “brand sticker” on box front
    sticker_a = int(230 * (1.0 - 0.2 * open_p))
    sx0, sy0 = cx - 150, cy + 20
    d.rounded_rectangle((sx0, sy0, sx0 + 300, sy0 + 64), radius=16,
                        fill=(255, 255, 255, sticker_a), outline=(0, 0, 0, int(70*sticker_a/255)), width=2)
    text_center(d, (cx, sy0 + 32), COMPANY, FONT_H2, (22, 22, 24, sticker_a), shadow=False)

    return open_p, reveal_p

# =============================
# COPY / END CARD
# =============================
def draw_copy(img, t, open_p, reveal_p):
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer, "RGBA")

    # Show big copy after reveal
    show = ease(clamp((t - 6.0) / 1.2))

    if show > 0:
        a = int(255 * show)
        # Headline
        text_center(d, (W * 0.5, H * 0.16), END_HEADLINE, FONT_H1, (245, 248, 255, a), shadow=True)
        text_center(d, (W * 0.5, H * 0.24), END_SUB, FONT_H2, (255, 255, 255, int(210 * show)), shadow=True)

        # CTA bar
        bar_a = int(210 * show)
        d.rounded_rectangle((W*0.18, H*0.86, W*0.82, H*0.93), radius=18,
                            fill=(15, 15, 19, bar_a), outline=(255, 255, 255, int(40*show)), width=2)
        text_center(d, (W * 0.5, H * 0.895), CTA, FONT_CTA, (255, 255, 255, int(240*show)), shadow=False)

    # Small “reveal” line while opening
    mid_show = ease(clamp((t - 3.2) / 1.0)) * (1.0 - ease(clamp((t - 6.0) / 0.8)))
    if mid_show > 0.01:
        a2 = int(220 * mid_show)
        text_center(d, (W * 0.5, H * 0.78), TAGLINE, FONT_H2, (255, 255, 255, a2), shadow=True)

    return Image.alpha_composite(img, layer)

# =============================
# FRAME FUNCTION
# =============================
def make_frame(t):
    bg = make_bg(t)

    open_p, reveal_p = draw_box_and_house(bg, t)

    # subtle “camera push-in” after the house rises
    push = ease(clamp((t - 4.8) / 2.2)) * 0.045  # scale factor
    if push > 0:
        scale = 1.0 + push
        nw, nh = int(W * scale), int(H * scale)
        bg2 = bg.resize((nw, nh), resample=Image.LANCZOS)
        x0 = (nw - W) // 2
        y0 = (nh - H) // 2
        bg = bg2.crop((x0, y0, x0 + W, y0 + H))

    bg = draw_copy(bg, t, open_p, reveal_p)

    # Post-grade
    arr = np.array(bg.convert("RGB")).astype(np.float32)
    arr = add_vignette(arr, strength=0.52)
    arr = add_grain(arr, t, amount=6.5)

    out = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    out = ImageEnhance.Contrast(out).enhance(1.10)
    out = ImageEnhance.Color(out).enhance(1.08)

    return np.array(out)

# =============================
# RENDER
# =============================
if __name__ == "__main__":
    print("Rendering moving advert...")
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
