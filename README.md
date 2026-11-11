# NeuroLens

Real-time facial emotion recognition on a standalone Raspberry Pi Zero 2W. The Pi and its camera do all the processing on the device: they capture video, detect faces, and classify emotions, then stream the results over WebSockets to a web app that displays them. It's designed as an assistive tool to help neurodivergent people recognize emotions.

```
┌──────────── Raspberry Pi Zero 2W (standalone) ────────────┐
│ Arducam v3 → Haar face detection → CNN emotion classifier │ ──WebSocket──▶ web app (any browser)
└───────────────────────────────────────────────────────────┘
```

## How it works

1. **Capture:** an Arducam v3 (12 MP IMX708) feeds 640×480 frames to the Pi through `picamera2`.
2. **Face detection:** OpenCV Haar cascades run on a frame downscaled to 320×240 for speed. The face boxes are then scaled back up to 640×480.
3. **Preprocessing:** each face is cropped with 20% padding, contrast-equalized with CLAHE to help in poor lighting, and resized to 48×48 grayscale.
4. **Classification:** a compact CNN trained on FER-2013 outputs probabilities for 7 emotions: angry, disgust, fear, happy, neutral, sad, and surprise.
5. **Streaming:** the Pi runs a WebSocket server on port `8765` and pushes results to every connected browser, about 4 times a second.

## Making it fit on a Pi Zero

**The model** ([`training/train_emotion.py`](training/train_emotion.py)) is small: three convolution blocks (16, 32, and 64 filters, each with batch norm and max pooling), then Dense(64) with dropout and a 7-way softmax. That's about 172,000 parameters, trained for 25 epochs. Exported to TensorFlow Lite with default optimization, it's about 178 KB.

**No ML framework runs on the Pi.** The trained weights are exported to `emotion_weights.npy`. The forward pass is written by hand in NumPy inside [`pi/server.py`](pi/server.py): convolution, batch norm, ReLU, max pooling, the dense layers, and softmax. The Pi only needs NumPy and OpenCV.

**Detection runs in a background thread**, so the WebSocket server stays responsive while each frame is processed.

## Repository layout

| Path | What it is |
|---|---|
| `pi/server.py` | Camera capture, face detection, NumPy inference, and the WebSocket server. Runs on the Pi. |
| `pi/emotion_weights.npy` | The trained CNN weights. |
| `pi/haarcascade_frontalface_default.xml` | The OpenCV face detector. |
| `training/train_emotion.py` | Trains the CNN on FER-2013 and exports a TFLite model. |
| `web-app/` | The browser app (HTML, CSS, and JavaScript with no build step). It shows the top emotion, its confidence, the runner-up emotions, and a high-contrast mode. |

## Running it

**On the Raspberry Pi**, with the camera attached:
```bash
cd pi
pip install -r requirements.txt
python server.py   # WebSocket server on ws://0.0.0.0:8765
```

**Web app:** open `web-app/pages/home.html` in a browser on the same network. It connects to `ws://cvpi.local:8765`, so if your Pi has a different hostname, change it in `web-app/emotion_client.js`. To try the interface without a Pi, set `DEMO_MODE = true` at the top of that file.

**Training:** download FER-2013 in the `train/<emotion>/` and `test/<emotion>/` image-folder layout and set `DATA_DIR` in `training/train_emotion.py`. Then:
```bash
pip install -r training/requirements.txt
python training/train_emotion.py
```

## Next steps

- Retrain with augmentation (flips, rotations, and brightness shifts) and more varied lighting and angles.
- Add body language through pose estimation.
- Package the Pi and camera as a module that attaches to any pair of glasses.
