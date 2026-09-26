# Push this project to GitHub

Run these commands **in order** in PowerShell or Command Prompt. Open the terminal in the project folder:  
`cd C:\Users\msk24\hr_analytics_ml_pipeline`

---

## Step 1: Initialize Git and make the first commit

```powershell
git init
git add .
git status
git commit -m "HR analytics ML pipeline: SQL, Random Forest, API, Streamlit, Docker"
```

---

## Step 2: Create the repository on GitHub

1. Go to **https://github.com/new**
2. **Repository name:** e.g. `hr-attrition-ml-pipeline` (or `hr-analytics-pipeline`)
3. **Description (optional):** `End-to-end HR attrition prediction — SQL, ML, FastAPI, Streamlit`
4. Choose **Public**
5. Do **not** check "Add a README" (you already have one)
6. Click **Create repository**

---

## Step 3: Connect and push

GitHub will show you commands. Use these (replace `YOUR_USERNAME` and `YOUR_REPO` with your actual GitHub username and repo name):

```powershell
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

**Example:** If your username is `johndoe` and repo is `hr-attrition-ml-pipeline`:

```powershell
git remote add origin https://github.com/johndoe/hr-attrition-ml-pipeline.git
git push -u origin main
```

If GitHub asks for login, use your GitHub username and a **Personal Access Token** (not your password):  
**Settings → Developer settings → Personal access tokens → Generate new token.**

---

## Step 4 (optional): Add the model so Streamlit Cloud works

Right now `.gitignore` excludes `output/*.pkl`. To deploy the Streamlit app on Streamlit Cloud, the model file must be in the repo.

1. Run the pipeline once so the model exists:
   ```powershell
   python run_pipeline.py
   ```

2. Temporarily allow the model to be committed. In `.gitignore`, change:
   - From: `output/*.pkl`
   - To: `# output/*.pkl`   (comment it out)

3. Add and push the model:
   ```powershell
   git add output/attrition_model.pkl output/figures/
   git commit -m "Add trained model and figures for Streamlit deploy"
   git push
   ```

4. (Optional) Restore `.gitignore` later by uncommenting `output/*.pkl` if you don’t want future model changes committed.

---

Done. Your project is on GitHub. Use the repo URL for Streamlit Cloud (see **DEPLOY_LIVE_LINK.md**).
