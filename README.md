# audio-cnn

Small project: a PyTorch audio classifier served with FastAPI (deployed optionally via Modal) and a React frontend that uploads audio and shows predictions.

## Repo layout (important files)
- train.py — training, Modal functions, and a Modal-hosted FastAPI app (`fastapi_app`)
- model.py — model definition
- /models — expected place for `best_model.pth`
- frontend/ — React app (Vite + Tailwind in this repo)
  - frontend/src/Audioclassifier.jsx — main UI component

## Prerequisites

- macOS with:
  - Python 3.8+ and pip
  - Node.js 16+ and npm
- (Optional) Modal CLI if calling Modal remote functions from your machine

## Backend — local development (recommended)
1. Ensure you have a trained model at `./models/best_model.pth`. If you trained on Modal, copy it to `./models` or update paths in the server.
2. Install Python deps: