# Spam Message Classifier — Netlify Frontend + Python AI API

This is a deployable split architecture: a responsive static frontend hosted on Netlify and a Python FastAPI machine-learning API hosted separately (Render is configured as the example host). The design follows the supplied dashboard reference and intentionally does **not** include sample/demo messages in the user interface.

## Pages
- Home: live dataset statistics, accuracy, distribution chart, quick message classification
- Predict Message: message input, Spam/Not Spam result, confidence
- Batch Prediction: CSV upload, results table, CSV download
- Model Performance: accuracy, precision, recall, F1, confusion matrix, classification report
- Dataset Information: dataset counts and preparation details
- About: technology stack and workflow

## Important deployment note
Netlify hosts the frontend files. The Python scikit-learn model runs in a separate backend service; Netlify should not be treated as the host for this full Python ML API. This repository is structured for **Netlify frontend + Render FastAPI backend**.

## 1. Deploy backend to Render
1. Push this project to a GitHub repository.
2. In Render, create a new **Blueprint** from the repository and select `backend/render.yaml`, or create a Python web service manually with:
   - Root Directory: `backend`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
3. First startup downloads the UCI SMS Spam Collection and trains the model. This may take a little while.
4. When deployed, test `https://YOUR-API.onrender.com/health` and `https://YOUR-API.onrender.com/metrics`.
5. Copy the backend public URL.

## 2. Connect frontend to backend
1. Open `frontend/config.js`.
2. Replace the empty API base URL with your Render URL, with no trailing slash:
   `window.API_BASE_URL = "https://YOUR-API.onrender.com";`
3. Save the file and commit/push the change to GitHub.

For tighter CORS settings, set the backend environment variable `FRONTEND_ORIGIN` to your Netlify URL after the frontend is deployed. The default `*` is convenient for initial setup, but setting the exact origin is better for production.

## 3. Deploy frontend to Netlify
1. In Netlify, choose **Add new site → Import an existing project** and connect the GitHub repository.
2. Set **Base directory** to `frontend`.
3. Set **Build command** to blank / no build command.
4. Set **Publish directory** to `.`.
5. Deploy. Netlify serves `index.html`, `styles.css`, `app.js`, and `config.js`.
6. Open the Netlify site URL and check that the sidebar says **Classifier online**.

You can also drag-and-drop the `frontend` folder into Netlify Drop after editing `config.js`, but Git-based deploys are easier to maintain.

## Local development
### Backend
```bash
cd backend
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
### Frontend
Edit `frontend/config.js` to:
```js
window.API_BASE_URL = "http://127.0.0.1:8000";
```
Then open `frontend/index.html` in a browser, or serve the folder:
```bash
python -m http.server 5500 --directory frontend
```
Open `http://127.0.0.1:5500`.

## CSV input format
CSV must contain `message` or `text` column. The backend limits uploads to 5 MB and 10,000 messages.

## ML pipeline
UCI SMS Spam Collection → text cleaning → TF-IDF unigrams/bigrams → Logistic Regression → held-out evaluation. Metrics are calculated from the actual dataset and test split; the UI does not hard-code accuracy numbers.

## Before production use
- Set `FRONTEND_ORIGIN` to your exact Netlify domain.
- Use a persistent or prebuilt model artifact if your backend host's cold-start time is too high.
- Free backend instances may sleep when idle, so the first request after inactivity can be slower.
- Confidence is a model estimate, not a guarantee. Do not treat this as a security filter without human review.
- The initial backend startup needs outbound internet access to download the dataset.
