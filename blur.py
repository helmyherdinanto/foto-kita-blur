import cv2
import mediapipe as mp
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont

# detect tangan
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,          # dinaikkan jadi 2 supaya bisa deteksi 2 tangan sekaligus
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def finger_up(tip, pip, landmarks):
    return landmarks[tip].y < landmarks[pip].y


def is_peace(landmarks):

    index_up = finger_up(8, 6, landmarks)
    middle_up = finger_up(12, 10, landmarks)

    ring_up = finger_up(16, 14, landmarks)
    pinky_up = finger_up(20, 18, landmarks)

    return (
        index_up
        and middle_up
        and not ring_up
        and not pinky_up
    )


def distance(p1, p2):
    return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5


def is_two_hand_heart(landmarks_list):
    """
    Gesture hati dari 2 tangan: ujung jempol kiri & kanan saling
    berdekatan, dan ujung telunjuk kiri & kanan juga saling berdekatan
    (dua titik sentuh yang membentuk lekukan atas & pucuk bawah hati).
    Butuh 2 tangan terdeteksi.
    """
    if len(landmarks_list) < 2:
        return False, None

    lm1, lm2 = landmarks_list[0], landmarks_list[1]

    thumb1, index1 = lm1[4], lm1[8]
    thumb2, index2 = lm2[4], lm2[8]

    thumbs_close = distance(thumb1, thumb2) < 0.08
    index_close = distance(index1, index2) < 0.08

    if thumbs_close and index_close:
        # titik tengah keempat ujung jari sebagai pusat hati
        cx = (thumb1.x + thumb2.x + index1.x + index2.x) / 4
        cy = (thumb1.y + thumb2.y + index1.y + index2.y) / 4
        return True, (cx, cy)

    return False, None


# --- sistem render emoji (via Pillow, karena OpenCV tidak bisa gambar emoji) ---

# font emoji bawaan macOS. Kalau di Windows ganti ke "seguiemj.ttf",
# kalau di Linux biasanya "NotoColorEmoji.ttf"
EMOJI_FONT_PATH = "/System/Library/Fonts/Apple Color Emoji.ttc"

# font emoji berwarna cuma punya beberapa ukuran bitmap "bawaan" (bukan
# vektor yang bisa di-scale bebas), dan ukurannya bisa beda-beda tergantung
# versi macOS. Jadi coba beberapa kandidat sampai ketemu yang valid.
EMOJI_SIZE_CANDIDATES = [160, 137, 128, 108, 96, 80, 64, 48, 40, 32, 24, 20]

LOVE_EMOJIS = ["💖", "💕", "💗", "💓", "💘", "🩷", "❤️"]


def get_working_emoji_font():
    for candidate_size in EMOJI_SIZE_CANDIDATES:
        try:
            font = ImageFont.truetype(EMOJI_FONT_PATH, candidate_size)
            print(f"Font emoji berhasil dimuat pakai ukuran {candidate_size}px")
            return font
        except OSError:
            continue
    return None


EMOJI_FONT = get_working_emoji_font()


def render_emoji_sprite(emoji_char):
    """Render satu emoji jadi array RGBA numpy, di-crop pas ke bounding box-nya."""
    if EMOJI_FONT is None:
        return None

    canvas_size = 300
    img = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text(
        (canvas_size // 4, canvas_size // 4),
        emoji_char,
        font=EMOJI_FONT,
        embedded_color=True
    )

    arr = np.array(img)
    alpha = arr[:, :, 3]
    coords = cv2.findNonZero(alpha)

    if coords is not None:
        x, y, cw, ch = cv2.boundingRect(coords)
        arr = arr[y:y + ch, x:x + cw]

    return arr


# render semua emoji sekali di awal, simpan di cache (biar tidak render ulang tiap frame)
EMOJI_SPRITES = {}

if EMOJI_FONT is None:
    print("=" * 60)
    print("PERINGATAN: Font emoji gagal dimuat di semua ukuran yang dicoba.")
    print("Efek hati emoji TIDAK akan muncul (hati lain tetap jalan normal).")
    print("Coba jalankan: pip3 install --upgrade pillow")
    print("=" * 60)
else:
    for emoji_char in LOVE_EMOJIS:
        try:
            EMOJI_SPRITES[emoji_char] = render_emoji_sprite(emoji_char)
        except Exception as e:
            print(f"Gagal render emoji {emoji_char}: {e}")


def overlay_emoji(frame_bgr, sprite_rgba, center, size, alpha=1.0):
    """Tempel sprite emoji (RGBA) ke frame BGR di posisi tertentu, dengan alpha blending."""
    if sprite_rgba is None or size <= 0:
        return

    sprite = cv2.resize(sprite_rgba, (size, size), interpolation=cv2.INTER_AREA)
    sh, sw = sprite.shape[:2]

    fh, fw = frame_bgr.shape[:2]
    x, y = center
    x1, y1 = x - sw // 2, y - sh // 2
    x2, y2 = x1 + sw, y1 + sh

    # lewati kalau posisinya di luar frame sama sekali
    if x2 <= 0 or y2 <= 0 or x1 >= fw or y1 >= fh:
        return

    # potong bagian sprite yang keluar dari batas frame
    src_x1, src_y1 = max(0, -x1), max(0, -y1)
    x1c, y1c = max(x1, 0), max(y1, 0)
    x2c, y2c = min(x2, fw), min(y2, fh)
    src_x2 = src_x1 + (x2c - x1c)
    src_y2 = src_y1 + (y2c - y1c)

    sprite_crop = sprite[src_y1:src_y2, src_x1:src_x2]
    if sprite_crop.size == 0:
        return

    roi = frame_bgr[y1c:y2c, x1c:x2c]

    sprite_rgb = sprite_crop[:, :, :3][:, :, ::-1]  # RGB (PIL) -> BGR (OpenCV)
    sprite_alpha = (sprite_crop[:, :, 3:4].astype(np.float32) / 255.0) * alpha

    blended = (
        sprite_rgb.astype(np.float32) * sprite_alpha
        + roi.astype(np.float32) * (1 - sprite_alpha)
    ).astype(np.uint8)

    frame_bgr[y1c:y2c, x1c:x2c] = blended


# open camera
cap = cv2.VideoCapture(0)

# daftar partikel hati yang sedang aktif dianimasikan
# tiap entri: {"pos": [x, y], "vx": float, "age": int, "max_age": int,
#              "size": int, "emoji": str}
heart_particles = []

PARTICLES_PER_FRAME = 2          # hati yang muncul persis di posisi tangan
AMBIENT_PARTICLES_PER_FRAME = 3  # hati acak yang muncul di seluruh layar

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    hand_result = hands.process(rgb)

    peace_detected = False
    all_landmarks = []

    if hand_result.multi_hand_landmarks:

        for hand_landmarks in hand_result.multi_hand_landmarks:

            landmarks = hand_landmarks.landmark
            all_landmarks.append(landmarks)

            if is_peace(landmarks):
                peace_detected = True

    heart_detected, heart_center_norm = is_two_hand_heart(all_landmarks)

    # blur efek saat peace sign
    if peace_detected:
        frame = cv2.GaussianBlur(
            frame,
            (61, 61),
            0
        )

    # (A) hati yang muncul tepat di posisi tangan
    if heart_detected:
        cx = int(heart_center_norm[0] * w)
        cy = int(heart_center_norm[1] * h)

        for _ in range(PARTICLES_PER_FRAME):
            heart_particles.append({
                "pos": [
                    cx + random.randint(-40, 40),
                    cy + random.randint(-20, 20)
                ],
                "vx": random.uniform(-0.8, 0.8),
                "age": 0,
                "max_age": random.randint(10, 18),
                "size": random.randint(28, 46),
                "emoji": random.choice(LOVE_EMOJIS),
            })

    # (B) hati acak di seluruh layar (atas kepala, samping, dll)
    if heart_detected:
        for _ in range(AMBIENT_PARTICLES_PER_FRAME):
            heart_particles.append({
                "pos": [
                    random.randint(0, w),
                    random.randint(0, int(h * 0.9))
                ],
                "vx": random.uniform(-0.5, 0.5),
                "age": 0,
                "max_age": random.randint(10, 18),
                "size": random.randint(22, 38),
                "emoji": random.choice(LOVE_EMOJIS),
            })

    # gambar & update semua partikel hati yang sedang aktif
    remaining_particles = []
    for p in heart_particles:
        progress = p["age"] / p["max_age"]  # 0.0 -> 1.0

        # bergerak ke atas (melayang) sambil sedikit bergoyang ke samping
        p["pos"][1] -= 2.5
        p["pos"][0] += p["vx"]

        # alpha: memudar menjelang akhir animasi
        alpha = max(1.0 - progress, 0.0)

        overlay_emoji(
            frame,
            EMOJI_SPRITES.get(p["emoji"]),
            (int(p["pos"][0]), int(p["pos"][1])),
            p["size"],
            alpha=alpha
        )

        p["age"] += 1
        if p["age"] < p["max_age"]:
            remaining_particles.append(p)

    heart_particles = remaining_particles

    cv2.imshow(
        "helleye",
        frame
    )

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
