"""
Extrae frames de los videos del dataset para subirlos a Roboflow.

Uso (en Terminal, con un entorno que tenga opencv-python):
    python3 extraer_frames.py

Lee:    ~/dev/boxeo_proyecto/videos/<clase>/<video>.mp4|.mov
Escribe: ~/dev/boxeo_proyecto/frames/<clase>/<video>_f00000.jpg

El nombre de cada frame conserva el video (toma) y el número de frame, para poder
rearmar las secuencias más adelante y separar train/test POR TOMA.
"""
from pathlib import Path
import cv2

# ------------------------- PARÁMETROS (ajustables) -------------------------
BASE = Path.home() / "dev" / "boxeo_proyecto"
VIDEOS = BASE / "videos"
SALIDA = BASE / "frames"

SEGUNDOS_ENTRE_FRAMES = 0.5   # 0.5 = un frame cada medio segundo (~450 imágenes en total)
MARGEN_SEG = 0.0              # segundos a descartar al inicio y al final de cada video
LADO_MAX = 1280               # reduce el lado mayor a este tamaño (1920 -> 1280)
CALIDAD_JPG = 90
EXTENSIONES = {".mp4", ".mov", ".m4v", ".avi"}
# ---------------------------------------------------------------------------

if not VIDEOS.exists():
    raise SystemExit(f"No encuentro {VIDEOS}. Revisa que exista y que los videos estén en subcarpetas por clase.")

resumen = {}
for carpeta in sorted(p for p in VIDEOS.iterdir() if p.is_dir()):
    clase = carpeta.name
    salida_clase = SALIDA / clase
    salida_clase.mkdir(parents=True, exist_ok=True)
    total_clase = 0

    for video in sorted(carpeta.iterdir()):
        if video.suffix.lower() not in EXTENSIONES or video.name.startswith("."):
            continue

        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        paso = max(1, round(fps * SEGUNDOS_ENTRE_FRAMES))
        inicio = int(MARGEN_SEG * fps)
        fin = n_total - int(MARGEN_SEG * fps) if n_total > 0 else float("inf")

        guardados = 0
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if inicio <= i < fin and (i - inicio) % paso == 0:
                h, w = frame.shape[:2]
                mayor = max(h, w)
                if mayor > LADO_MAX:
                    esc = LADO_MAX / mayor
                    frame = cv2.resize(frame, (int(w * esc), int(h * esc)), interpolation=cv2.INTER_AREA)
                nombre = f"{video.stem}_f{i:05d}.jpg"
                cv2.imwrite(str(salida_clase / nombre), frame, [cv2.IMWRITE_JPEG_QUALITY, CALIDAD_JPG])
                guardados += 1
            i += 1
        cap.release()

        print(f"  {clase}/{video.name}: {guardados} frames")
        total_clase += guardados

    resumen[clase] = total_clase

print("\n=== Resumen ===")
for clase, n in resumen.items():
    print(f"{clase:10s} {n:4d} imágenes")
print(f"{'TOTAL':10s} {sum(resumen.values()):4d} imágenes")
print(f"\nListo. Las imágenes están en: {SALIDA}")
