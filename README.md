# Face Detection — Streamlit App

Uses OpenCV's pretrained **Haar Cascade** classifier to find and box every
human face in a photo. This is Project 3 from the notebook, wrapped in an
interactive UI:

- Upload a photo (or use the built-in sample image)
- Adjust `scaleFactor` and `minNeighbors` live with sliders and see the
  effect instantly
- Side-by-side original vs. annotated image
- Bounding box coordinates for every detected face
- A plain-English explainer of how Haar Cascades work

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

Note: this uses `opencv-python-headless` (not `opencv-python`) — the
headless build has no GUI dependencies, which is what you want on a server
like Streamlit Cloud. Don't install both in the same environment.

## Deploy for free on Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, and `runtime.txt` to a GitHub repo
   (all three in the same folder, no duplicate/renamed files).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **New app**, pick your repo/branch, set the main file path to
   `app.py` exactly.
4. Click **Deploy**.

## Deploy elsewhere (Render, Railway, Hugging Face Spaces, etc.)

Start command:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```
