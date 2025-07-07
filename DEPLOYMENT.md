# 🚀 Deployment Guide

## Deploy to Render (Free)

### Prerequisites
- GitHub account with your code pushed
- Render account (free): https://render.com

### Step 1: Prepare Your Repository
1. Ensure all files are committed and pushed to GitHub
2. Your repository should include:
   - `Procfile`
   - `requirements.txt`
   - `runtime.txt`
   - `build.sh`
   - `web_app.py`
   - `data/sonar.csv`

### Step 2: Deploy on Render

1. **Sign up/Login to Render**: https://render.com
2. **Connect GitHub**: Link your GitHub account
3. **Create New Web Service**:
   - Click "New +" → "Web Service"
   - Connect your `rock_vs_mine_ml` repository
   - Choose "Free" plan

4. **Configure Settings**:
   ```
   Name: rock-vs-mine-prediction
   Environment: Python 3
   Build Command: bash build.sh
   Start Command: gunicorn web_app:app
   ```

5. **Environment Variables** (if needed):
   ```
   PORT=10000 (automatically set by Render)
   ```

6. **Deploy**: Click "Create Web Service"

### Step 3: Access Your App
- Your app will be available at: `https://rock-vs-mine-prediction.onrender.com`
- First deployment takes 5-10 minutes
- The app will train the model during the first build

### Important Notes
- **Cold Starts**: Free tier sleeps after 15 minutes of inactivity
- **Build Time**: ~3-5 minutes for first deployment
- **Memory**: 512MB limit (sufficient for your models)
- **Monthly Hours**: 750 hours free per month

## Alternative: Deploy to Railway

1. **Sign up**: https://railway.app
2. **Deploy from GitHub**: 
   - Click "Deploy from GitHub"
   - Select your repository
   - Railway auto-detects Python and Flask

3. **Environment Variables**:
   ```
   PORT=${{PORT}} (auto-set)
   ```

## Alternative: Deploy to Streamlit Cloud

If you want to convert to Streamlit (simpler but requires code changes):
1. Create `streamlit_app.py`
2. Deploy at: https://streamlit.io/cloud

## Testing Your Deployment

1. **Check Health**: Visit your app URL
2. **Test Prediction**: Use the sample data provided
3. **Monitor Logs**: Check Render dashboard for any issues

## Troubleshooting

- **Model Loading Error**: Ensure `data/sonar.csv` is in repository
- **Build Failure**: Check build logs in Render dashboard  
- **Memory Issues**: Optimize model size if needed
- **Cold Start**: First request after sleep takes 30-60 seconds
