# HR Attrition Prediction

I built this to predict employee attrition and to keep the data side in SQL — load into SQLite, do analytics and feature prep with SQL, then train a Random Forest. I added a FastAPI endpoint and a Streamlit app so I could share a working demo without people touching the code.

---

## How I run it

```bash
pip install -r requirements.txt
python run_pipeline.py
```

If there’s nothing in `data/`, it pulls down a benchmark HR CSV so the pipeline runs straight away. After that I have `output/attrition_model.pkl` and charts in `output/figures/` (attrition by department, confusion matrix, feature importance).

I prefer using real survey data when I can. I keep the [Saudi Employee Attrition](https://data.mendeley.com/datasets/6z2hty8php/1) CSV in `data/` and run the same command; it picks it up as long as there’s an Attrition-style column.

---

## Data I use

- **Saudi Employee Attrition** (Mendeley) — 1,191 employees, 34 attributes, real survey. I use this when I want real data. [Link](https://data.mendeley.com/datasets/6z2hty8php/1)
- **Benchmark CSV** — auto-downloaded when `data/` is empty. IBM-style, ~1.5k rows. I use it for quick runs or demos.

---

## How the pipeline works (on my side)

1. CSV goes into SQLite as `hr_raw`.
2. I run the analytics I care about in SQL (`sql/queries_analytics.sql` — attrition by department, role, overtime, etc.).
3. For modeling, either the SQL feature view (`queries_ml_features.sql`) or a generic `SELECT *` plus target encoding in Python, depending on the dataset.
4. Training is a sklearn pipeline: scaler, one-hot encoding, Random Forest. I get a classification report and 5-fold CV F1 in the console, and the figures are written to `output/figures/`.

So everything before the model lives in the DB and in SQL; the Python side is load → train → save.

---

## Flags I use

| Flag | What I use it for |
|------|-------------------|
| `--data path/to/file.csv` | Point at a specific CSV. |
| `--db path/to/hr.db` | Custom DB path (default is `hr_analytics.db`). |
| `--model path/to/model.pkl` | Where to save the model. |
| `--no-train` | Only load and run SQL; no training. |

---

## API and Streamlit (how I serve it)

**API** — I run the FastAPI app from the project root:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Docs and schema at `http://localhost:8000/docs`; predictions via `POST /predict`.

**Streamlit** — for a click-through demo:

```bash
streamlit run app_streamlit.py
```

When I want a public link (e.g. for a resume or portfolio), I deploy the Streamlit app on [Streamlit Community Cloud](https://share.streamlit.io) and use that URL. I wrote down the steps in [DEPLOY_LIVE_LINK.md](DEPLOY_LIVE_LINK.md).

**Docker** — when I need the API in a container:

```bash
docker build -t hr-attrition-api .
docker run -p 8000:8000 -v "%cd%\output:/app/output" hr-attrition-api
```

On Linux/Mac I use `-v "$(pwd)/output:/app/output"`.

---

## What’s in the repo

```
├── api/                 # FastAPI app (main.py, predict.py)
├── data/                # Where I put CSVs
├── sql/
│   ├── schema.sql
│   ├── queries_analytics.sql
│   └── queries_ml_features.sql
├── src/
│   ├── data_loader.py
│   ├── db.py
│   ├── train.py
│   └── visualize.py
├── output/              # model + figures after a run
├── run_pipeline.py
├── app_streamlit.py
├── Dockerfile
└── requirements.txt
```

---

## Dataset citation (Saudi)

Mendeley Data: [10.17632/6z2hty8php.1](https://data.mendeley.com/datasets/6z2hty8php/1)  
Alqahtani H et al., *Dataset for predictive modelling and analysis of employee attrition and retention*, Data in Brief (2025), DOI: 10.1016/j.dib.2025.112242
