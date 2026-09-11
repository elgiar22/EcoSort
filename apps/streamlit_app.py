from __future__ import annotations

import os
os.environ.setdefault("KERAS_BACKEND", "torch")

from pathlib import Path
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image

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

st.set_page_config(
    page_title="EcoSort Tester",
    page_icon="♻️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 820px;
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading EcoSort AI...")
def load_model(path):
    import keras

    model = keras.models.load_model(path, compile=False)

    if int(model.output_shape[-1]) != len(CLASS_NAMES):
        raise RuntimeError(
            f"Expected {len(CLASS_NAMES)} outputs, got {model.output_shape[-1]}"
        )

    return model, keras


def crop_and_predict(model, keras_module, image_rgb, crop_ratio):
    h, w = image_rgb.shape[:2]
    side = max(32, int(min(h, w) * crop_ratio))

    x1 = max(0, (w - side) // 2)
    y1 = max(0, (h - side) // 2)

    crop = image_rgb[
        y1:y1 + side,
        x1:x1 + side,
    ]

    input_h = int(model.input_shape[1])
    input_w = int(model.input_shape[2])

    resized = cv2.resize(
        crop,
        (input_w, input_h),
        interpolation=cv2.INTER_AREA,
    )

    batch = resized.astype(np.float32)[None, ...]

    t0 = time.perf_counter()
    output = model([batch], training=False)
    infer_ms = (time.perf_counter() - t0) * 1000

    probs = np.asarray(
        keras_module.ops.convert_to_numpy(output)[0],
        dtype=np.float32,
    )

    order = np.argsort(probs)[::-1]
    idx = int(order[0])

    raw_class = CLASS_NAMES[idx]
    confidence = float(probs[idx])
    mapped = TARGET_MAP.get(raw_class, "other")

    top3 = [
        (CLASS_NAMES[int(i)], float(probs[int(i)]))
        for i in order[:3]
    ]

    return {
        "raw_class": raw_class,
        "confidence": confidence,
        "mapped": mapped,
        "top3": top3,
        "infer_ms": infer_ms,
        "model_input": resized,
    }


def show_result(result, threshold):
    accepted = (
        result["mapped"] in TARGETS
        and result["confidence"] >= threshold
    )

    final_label = result["mapped"] if accepted else "other"

    c1, c2 = st.columns(2)
    c1.metric("Class", final_label.upper())
    c2.metric("Confidence", f"{result['confidence']:.1%}")

    st.caption(
        f"Raw: {result['raw_class']} | "
        f"Inference: {result['infer_ms']:.0f} ms"
    )

    if accepted:
        st.success(
            f"{final_label.upper()} — {result['confidence']:.1%}"
        )
    else:
        st.warning(
            f"OTHER / REJECT — "
            f"{result['raw_class']} {result['confidence']:.1%}"
        )

    st.markdown("#### Top 3")

    for name, score in result["top3"]:
        st.progress(
            min(max(score, 0.0), 1.0),
            text=f"{name} — {score:.1%}",
        )

    with st.expander("Exact model input"):
        st.image(
            result["model_input"],
            width=300,
        )


st.title("♻️ EcoSort Tester")
st.caption("Camera photo + image upload")

model_path = Path(__file__).resolve().parents[1] / "best_model.keras"

if not model_path.exists():
    st.error(
        "best_model.keras not found in the EcoSort project root."
    )
    st.stop()

model, keras_module = load_model(str(model_path.resolve()))

with st.sidebar:
    threshold = st.slider(
        "Confidence",
        0.30,
        0.95,
        0.62,
        0.01,
    )

    crop_ratio = st.slider(
        "Center crop",
        0.45,
        1.00,
        0.62,
        0.01,
    )

camera_tab, upload_tab = st.tabs(
    ["📷 Camera", "🖼️ Upload"]
)

with camera_tab:
    st.caption(
        "Camera mode is intentionally NON-LIVE for stability."
    )

    captured = st.camera_input(
        "Camera",
        label_visibility="collapsed",
    )

    if captured is not None:
        image = Image.open(captured).convert("RGB")

        result = crop_and_predict(
            model,
            keras_module,
            np.asarray(image),
            crop_ratio,
        )

        show_result(result, threshold)


with upload_tab:
    uploaded = st.file_uploader(
        "Upload JPG / PNG / WEBP",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")

        st.image(
            image,
            caption="Uploaded image",
            use_container_width=True,
        )

        result = crop_and_predict(
            model,
            keras_module,
            np.asarray(image),
            crop_ratio,
        )

        show_result(result, threshold)
