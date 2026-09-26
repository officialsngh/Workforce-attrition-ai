# Get a clickable link recruiters can open anytime

**localhost** only works on your PC. To share a link that works for anyone, deploy the app online (free).

---

## Option 1: Streamlit Community Cloud (recommended — ~5 minutes)

You get a permanent URL like: **https://your-app-name.streamlit.app**

### Step 1: Put your project on GitHub

1. Create a new repository on [GitHub](https://github.com/new) (e.g. `hr-attrition-predictor`).
2. Push your project:
   ```bash
   cd C:\Users\msk24\hr_analytics_ml_pipeline
   git init
   git add .
   git commit -m "HR attrition ML pipeline + Streamlit app"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/hr-attrition-predictor.git
   git push -u origin main
   ```
   (Replace `YOUR_USERNAME` and repo name with yours.)

### Step 2: Deploy on Streamlit Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.
2. Click **“New app”**.
3. Choose:
   - **Repository:** `YOUR_USERNAME/hr-attrition-predictor`
   - **Branch:** `main`
   - **Main file path:** `app_streamlit.py`
4. Click **“Deploy!”**.

After a few minutes you’ll see a URL like:

**https://hr-attrition-predictor-xxxx.streamlit.app**

That is your **clickable link** — paste it in your resume, LinkedIn, or email. It works on any device, anytime.

### Important: Streamlit Cloud must see your model file

The app needs `output/attrition_model.pkl` to run. Two ways:

**A) Commit the model to GitHub (simplest)**  
- Temporarily allow the model in git:
  - Open `.gitignore` and comment out or remove the line: `output/*.pkl`
  - Run the pipeline once: `python run_pipeline.py`
  - Then:
    ```bash
    git add output/attrition_model.pkl
    git commit -m "Add trained model for Streamlit deploy"
    git push
    ```
  - After deploy works, you can add `output/*.pkl` back to `.gitignore` if you prefer.

**B) Build the model during deploy (no model in repo)**  
- Add a script that runs the pipeline once when the app starts (slower first load, more setup). Option A is easier for a portfolio.

---

## Option 2: Hugging Face Spaces (alternative free host)

1. Create an account at [huggingface.co](https://huggingface.co).
2. Create a new **Space**, choose **Streamlit** as SDK.
3. Upload your project files (or connect GitHub) and set the app file to `app_streamlit.py`.
4. Put `attrition_model.pkl` in the Space (e.g. upload to the repo or use “Files and versions”).
5. Your link will be: **https://huggingface.co/spaces/YOUR_USERNAME/hr-attrition**

---

## What to put on your resume / LinkedIn

- **Portfolio / Projects:**  
  *“HR Attrition Predictor — Live demo: [https://your-app.streamlit.app](https://your-app.streamlit.app)”*
- Or: *“Interactive Streamlit app deployed on Streamlit Cloud — [Link](your-url)”*

Use the **Streamlit Cloud URL** (or Hugging Face Space URL), not localhost. That way the link is clickable and works on the recruiter’s PC anytime.
