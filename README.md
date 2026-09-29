# RealSense mixed-package volume demo

Live overlay on a Jetson Orin Nano Super with a RealSense D436: segments **boxes**, **padded bags**, and **jiffy envelopes**, shows L×W×H and volume for each, warns when items are stacked, and keeps three unique session counters.

## Setup (once)

```bash
cd /home/nvidia/Projects/boxes
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -U pip wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
pip install -r requirements.txt
```

This board has 8 GB unified memory and no swap by default. An 8 GB file was created at `/home/nvidia/swapfile`. Enable it before heavy installs or the first model download:

```bash
sudo swapon /home/nvidia/swapfile
```

To enable swap on boot, add this line to `/etc/fstab`:

```
/home/nvidia/swapfile none swap sw 0 0
```

## Run

Camera on a USB 3 port, display on `:0`:

```bash
cd /home/nvidia/Projects/boxes
./run.sh
```

Windowed:

```bash
./run.sh --windowed
```

Smaller/faster model if FPS is low:

```bash
./run.sh --model yoloe-26n-seg.pt --imgsz 480
```

Keys: `q` quit, `r` reset all three counts, `p` pause.

## What you should see

- Color-coded masks: blue boxes, green padded bags, orange jiffy envelopes
- Per-item `L×W×H cm` and volume in liters
- HUD: `Boxes / Padded bags / Jiffy envelopes / Total` (counts only go up)
- Red **STACKED** tag plus a warning banner when items overlap in 3D

First run downloads YOLOE weights plus a MobileCLIP text encoder (~600 MB) used only to bake in the package prompts. Later launches reuse `models/package_text_pe.pt` and skip CLIP.

Hold items about **0.5–1.5 m** from the camera so they fill a good portion of the frame. Default `--conf 0.08` is tuned for open-vocab YOLOE; raise it if you see false positives (e.g. instrument cases).
