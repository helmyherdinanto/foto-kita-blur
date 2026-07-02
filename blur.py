import cv2
import mediapipe as mp
import numpy as np
import random

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


def draw_heart(img, center, size, color=(0, 0, 255), alpha=1.0):
    """Menggambar bentuk hati sederhana pakai 2 lingkaran + 1 segitiga."""
    overlay = img.copy()
    x, y = center
    r = max(size // 2, 1)

    cv2.circle(overlay, (x - r // 2, y - r // 2), r // 2, color, -1)
    cv2.circle(overlay, (x + r // 2, y - r // 2), r // 2, color, -1)

    pts = np.array([
        [x - r, y - r // 4],
        [x + r, y - r // 4],
        [x, y + r]
    ], np.int32)
    cv2.fillPoly(overlay, [pts], color)

    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


# palet warna hati (BGR), campuran merah - pink - magenta seperti Photo Booth
HEART_COLORS = [
    (60, 20, 255),    # merah terang
    (130, 60, 230),   # pink tua
    (180, 105, 255),  # pink muda
    (150, 40, 200),   # magenta
]

# open camera
cap = cv2.VideoCapture(0)

# daftar partikel hati yang sedang aktif dianimasikan
# tiap entri: {"pos": [x, y], "vx": float, "age": int, "max_age": int,
#              "size": int, "color": tuple}
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
                "size": random.randint(18, 34),
                "color": random.choice(HEART_COLORS),
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
                "size": random.randint(14, 30),
                "color": random.choice(HEART_COLORS),
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

        draw_heart(
            frame,
            (int(p["pos"][0]), int(p["pos"][1])),
            p["size"],
            color=p["color"],
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
