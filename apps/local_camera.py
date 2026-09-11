from __future__ import annotations

import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import argparse
import queue
import threading
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
import requests

CLASS_NAMES = [
    "Unknown",
    "glass",
    "metal",
    "organic_waste",
    "paper_cardboard",
    "plastic",
    "textiles",
    "trash",
]

TARGET_MAP = {
    "glass": "glass",
    "metal": "metal",
    "organic_waste": "organic",
    "paper_cardboard": "paper",
    "plastic": "plastic",
}

TARGETS = {"plastic", "metal", "organic", "paper", "glass"}


def center_square(frame, scale):
    h, w = frame.shape[:2]
    side = max(32, int(min(h, w) * scale))
    x1 = max(0, (w - side) // 2)
    y1 = max(0, (h - side) // 2)
    x2 = min(w, x1 + side)
    y2 = min(h, y1 + side)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


class CameraReader:
    def __init__(self, index=0, width=640, height=480):
        self.lock = threading.Lock()
        self.frame = None
        self.seq = 0
        self.running = True

        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = cv2.VideoCapture(index)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {index}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue

            with self.lock:
                self.frame = frame
                self.seq += 1

    def latest(self):
        with self.lock:
            if self.frame is None:
                return None, self.seq
            return self.frame.copy(), self.seq

    def close(self):
        self.running = False
        self.thread.join(timeout=1)
        self.cap.release()


class Publisher:
    def __init__(self, url, source, enabled):
        self.url = url.rstrip("/")
        self.source = source
        self.enabled = enabled
        self.status = "OFF" if not enabled else "READY"
        self.q = queue.Queue(maxsize=1)
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def submit(self, label, confidence):
        if not self.enabled:
            return

        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass

        try:
            self.q.put_nowait((label, confidence))
        except queue.Full:
            pass

    def _loop(self):
        while self.running:
            try:
                item = self.q.get(timeout=0.25)
            except queue.Empty:
                continue

            if item is None:
                break

            label, confidence = item

            try:
                r = requests.post(
                    f"{self.url}/api/detections",
                    json={
                        "label": label,
                        "confidence": float(confidence),
                        "source": self.source,
                    },
                    timeout=4,
                )
                r.raise_for_status()
                self.status = "OK"
                print(f"[sent] {label:8s} {confidence:5.1%}")

            except Exception as exc:
                self.status = "ERROR"
                print(f"[api warning] {exc}")

    def close(self):
        self.running = False
        try:
            self.q.put_nowait(None)
        except queue.Full:
            pass
        self.thread.join(timeout=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(Path(__file__).resolve().parents[1] / "best_model.keras"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--roi", type=float, default=0.62)
    parser.add_argument("--threshold", type=float, default=0.62)
    parser.add_argument("--infer-every", type=int, default=4)
    parser.add_argument(
        "--url",
        default="https://ecosort-live.pretty-chord-2514.chatgpt.site",
    )
    parser.add_argument("--source", default="station-01")
    parser.add_argument("--no-publish", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    import keras

    print(f"Loading model: {model_path}")
    model = keras.models.load_model(str(model_path), compile=False)

    if int(model.output_shape[-1]) != len(CLASS_NAMES):
        raise RuntimeError(
            f"Expected {len(CLASS_NAMES)} outputs, got {model.output_shape[-1]}"
        )

    input_h = int(model.input_shape[1])
    input_w = int(model.input_shape[2])

    print(
        f"Model ready: input={input_w}x{input_h}, "
        f"outputs={model.output_shape[-1]}"
    )
    print(f"Class order: {CLASS_NAMES}")
    print("Press Q to quit.")

    camera = CameraReader(args.camera)
    publisher = Publisher(
        args.url,
        args.source,
        enabled=not args.no_publish,
    )

    ema = None
    history = deque(maxlen=5)
    last_sent_label = None
    last_sent_at = 0.0
    frame_counter = 0
    last_seq = -1

    raw_class = "waiting"
    raw_conf = 0.0
    final_label = "waiting"
    final_conf = 0.0
    infer_ms = 0.0
    top3 = []

    try:
        while True:
            frame, seq = camera.latest()

            if frame is None:
                time.sleep(0.005)
                continue

            if seq != last_seq:
                last_seq = seq
                frame_counter += 1

                if frame_counter % max(1, args.infer_every) == 0:
                    crop, _ = center_square(frame, args.roi)
                    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    rgb = cv2.resize(
                        rgb,
                        (input_w, input_h),
                        interpolation=cv2.INTER_AREA,
                    )
                    batch = rgb.astype(np.float32)[None, ...]

                    t0 = time.perf_counter()
                    out = model([batch], training=False)
                    infer_ms = (time.perf_counter() - t0) * 1000.0

                    probs = np.asarray(
                        keras.ops.convert_to_numpy(out)[0],
                        dtype=np.float32,
                    )

                    if ema is None:
                        ema = probs
                    else:
                        ema = 0.42 * probs + 0.58 * ema

                    order = np.argsort(ema)[::-1]
                    idx = int(order[0])
                    raw_class = CLASS_NAMES[idx]
                    raw_conf = float(ema[idx])

                    mapped = TARGET_MAP.get(raw_class, "other")
                    current = (
                        mapped
                        if mapped in TARGETS
                        and raw_conf >= args.threshold
                        else "other"
                    )

                    history.append((current, raw_conf))

                    winner, votes = Counter(
                        label for label, _ in history
                    ).most_common(1)[0]

                    winner_conf = max(
                        (conf for label, conf in history if label == winner),
                        default=0.0,
                    )

                    if (
                        len(history) >= 3
                        and winner in TARGETS
                        and votes >= 3
                        and winner_conf >= args.threshold
                    ):
                        final_label = winner
                        final_conf = winner_conf
                    else:
                        final_label = "other"
                        final_conf = raw_conf

                    top3 = [
                        (CLASS_NAMES[int(i)], float(ema[int(i)]))
                        for i in order[:3]
                    ]

                    now = time.monotonic()

                    if final_label in TARGETS:
                        if (
                            final_label != last_sent_label
                            or now - last_sent_at >= 2.0
                        ):
                            publisher.submit(final_label, final_conf)
                            last_sent_label = final_label
                            last_sent_at = now

            display = frame.copy()
            h, w = display.shape[:2]

            _, (x1, y1, x2, y2) = center_square(display, args.roi)
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 255), 2)

            cv2.rectangle(display, (0, 0), (w, 92), (0, 0, 0), -1)

            cv2.putText(
                display,
                f"{final_label.upper()} {final_conf:.0%}",
                (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.78,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                f"raw: {raw_class} {raw_conf:.0%} | AI {infer_ms:.0f}ms | API {publisher.status}",
                (14, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            top_text = "  ".join(
                f"{name}:{conf:.0%}"
                for name, conf in top3
            )

            cv2.putText(
                display,
                top_text,
                (14, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Compact local window
            display = cv2.resize(
                display,
                (560, 420),
                interpolation=cv2.INTER_AREA,
            )

            cv2.imshow("EcoSort Local Camera", display)

            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break

    finally:
        publisher.close()
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
