# EcoSort Model Card

## Overview

EcoSort uses a Keras image-classification model built around **EfficientNetB3**. The saved model accepts `300 × 300 × 3` RGB images and produces probabilities for eight classes.

## Model artifact

- File: `best_model.keras`
- Saved with: Keras 3.10.0
- Input shape: `(300, 300, 3)`
- Output classes: 8
- Backbone: EfficientNetB3

## Class order

1. `Unknown`
2. `glass`
3. `metal`
4. `organic_waste`
5. `paper_cardboard`
6. `plastic`
7. `textiles`
8. `trash`

## Routing classes used by the prototype

The current prototype exposes five recycling routes:

- `glass` → glass
- `metal` → metal
- `organic_waste` → organic
- `paper_cardboard` → paper
- `plastic` → plastic

`Unknown`, `textiles`, `trash`, and low-confidence predictions are routed to `other` by the application layer.

## Stabilization in local-camera mode

The local camera application applies:

- center-region cropping;
- exponential moving average smoothing over model probabilities;
- a confidence threshold (default `0.62`);
- voting over the latest five inference results;
- acceptance only when at least three votes agree on a supported target class.

Only accepted target classes are published to the dashboard API.

## Important limitations

This repository does not include a formal benchmark report or dataset documentation, so no accuracy, precision, recall, or F1 claims are made here. The confidence threshold is an application-level rejection rule; it should not be treated as a calibrated probability guarantee.
