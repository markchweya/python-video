import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# MoviePy import that works on both v1.x and v2.x
try:
    from moviepy.editor import ImageSequenceClip  # MoviePy 1.x
except ModuleNotFoundError:
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip  # MoviePy 2.x

# -----------------------------
# VIDEO SETTINGS
# -----------------------------
WIDTH = 1280
HEIGHT = 720
FPS = 30
DURATION = 10
FRAMES = FPS * DURATION

TEXT = "You Are Becoming Stronger Every Day"
FONT_SIZE = 60

FRAME_DIR = "generated_frames"
OUTPUT_FILE = "output.mp4"

os.makedirs(FRAME_DIR, exist_ok=True)

def load_font(size: int):
    candidates = [
        "arial.ttf",
        "Arial.ttf",
        "DejaVuSans.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()

FONT = load_font(FONT_SIZE)

def generate_gradient(step: int) -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT))
    draw = ImageDraw.Draw(img)

    shift = int(8 * np.sin(2 * np.pi * (step / FRAMES)))
    for y in range(HEIGHT):
        yy = (y + shift) / HEIGHT
        r = int(10 + 80 * yy)
        g = int(20 + 40 * yy)
        b = int(100 + 120 * yy)
        draw.line([(0, y), (WIDTH, y)], fill=(r, g, b))
    return img

def add_particles(img: Image.Image, step: int) -> Image.Image:
    base = img.convert("RGBA")
    layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    rng = np.random.default_rng(seed=step)
    for _ in range(45):
        x = int(rng.integers(0, WIDTH))
        y0 = int(rng.integers(0, HEIGHT))
        y = (y0 + step * 3) % HEIGHT
        size = int(rng.integers(2, 7))
        alpha = int(rng.integers(80, 200))
        draw.ellipse((x, y, x + size, y + size), fill=(255, 248, 235, alpha))

    layer = layer.filter(ImageFilter.GaussianBlur(radius=0.8))
    return Image.alpha_composite(base, layer)

def add_text(img: Image.Image, step: int) -> Image.Image:
    base = img.convert("RGBA")
    layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    bbox = draw.textbbox((0, 0), TEXT, font=FONT)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x = (WIDTH - text_w) // 2
    y = (HEIGHT - text_h) // 2 - 20

    fade_frames = FPS * 3
    alpha = min(255, int((step / fade_frames) * 255))
    shadow_alpha = int(alpha * 0.65)

    draw.text((x + 4, y + 4), TEXT, font=FONT, fill=(0, 0, 0, shadow_alpha))
    draw.text((x, y), TEXT, font=FONT, fill=(255, 255, 255, alpha))

    return Image.alpha_composite(base, layer)

print("Generating frames...")
frame_paths = []

for step in range(FRAMES):
    img = generate_gradient(step)
    img = add_particles(img, step)
    img = add_text(img, step)

    frame_path = os.path.join(FRAME_DIR, f"frame_{step:04d}.png")
    img.convert("RGB").save(frame_path, optimize=True)
    frame_paths.append(frame_path)

print("Rendering video...")
clip = ImageSequenceClip(frame_paths, fps=FPS)
clip.write_videofile(OUTPUT_FILE, codec="libx264", audio=False, fps=FPS, preset="medium", threads=4)
clip.close()

print(f"Done! Video saved as {OUTPUT_FILE}")
