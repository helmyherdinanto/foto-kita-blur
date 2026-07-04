# FOTO KITA BLUR 📸

---
🌐 **Coba langsung di Web tanpa install!**  
Kamu bisa langsung mengakses dan mencoba aplikasi ini langsung melalui web browser HP/PC (tanpa perlu install apa pun) lewat link berikut:  
👉 **[https://helmyherdinanto.github.io/foto-kita-blur/](https://helmyherdinanto.github.io/foto-kita-blur/)**
---

## Fitur

- ✌️ **Peace sign → blur** — angkat dua jari (telunjuk + tengah), layar otomatis nge-blur
- 🫶 **Gesture hati (dua tangan) → hujan hati** ["💖", "💕", "💗", "💓", "💘", "🩷", "❤️"] — dekatkan jempol dan telunjuk dari kedua tangan sampai bentuk hati, nanti muncul partikel hati beterbangan di layar (ada yang persis di posisi tangan, ada juga yang random di sekitar layar)
- Semua deteksi jalan real-time pakai webcam, tanpa perlu training model sendiri (full pakai MediaPipe Hands)

## Cara jalanin

**1. Clone repo ini**
```bash
git clone https://github.com/GagukPurnotow/foto-kita-blur.git
cd foto-kita-blur
```

**2. Install dependency**
```bash
pip3 install -r requirements.txt
```

**3. Jalankan**
```bash
python3 blur.py
```

**4. Kasih izin kamera**
Kalau pakai Mac dan muncul error kamera nggak keluar, cek dulu di `System Settings → Privacy & Security → Camera`, pastikan terminal/editor yang kamu pakai udah diizinkan akses kamera.

Tekan `ESC` buat keluar dari aplikasinya.

## Requirements

- Python 3.9+
- opencv-python
- mediapipe
- numpy

(semua sudah kecatat di `requirements.txt`)

## Struktur project

```
foto-kita-blur/
├── blur.py           # script utama
├── requirements.txt
└── README.md
```

## Ide pengembangan selanjutnya

- [ ] Tambah gesture lain (misal jempol ke atas → efek confetti)
- [ ] Simpan hasil rekaman jadi video/gif otomatis
- [ ] Bikin threshold gesture bisa diatur dari command line, biar nggak perlu edit kode tiap kali sensitivitasnya kurang pas
- [ ] Ganti bentuk hati custom (gambar/PNG) biar lebih halus dibanding versi gambar manual pakai OpenCV

## Catatan

Deteksi gesture di sini masih pakai aturan geometris sederhana (jarak antar landmark tangan dari MediaPipe), bukan model machine learning khusus kayak yang dipakai Apple. Jadi kadang perlu sedikit penyesuaian angka threshold tergantung jarak tangan ke kamera atau pencahayaan ruangan.

---
📷 Instagram:
@helmyherdinanto
