# 🌐 Deployment Guide - PROJECT NETRA Web Dashboard

This guide will help you deploy your NETRA traffic management dashboard online.

## 📋 Table of Contents
- [Deployment Options](#deployment-options)
- [Option 1: Streamlit Cloud (Recommended)](#option-1-streamlit-cloud-recommended)
- [Option 2: Render](#option-2-render)
- [Option 3: Railway](#option-3-railway)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Deployment Options

### Comparison

| Platform | Free Tier | Ease | Best For |
|----------|-----------|------|----------|
| **Streamlit Cloud** | ✅ Yes | ⭐⭐⭐⭐⭐ | Streamlit apps (RECOMMENDED) |
| **Render** | ✅ Yes | ⭐⭐⭐⭐ | General web apps |
| **Railway** | ✅ Limited | ⭐⭐⭐⭐ | Quick deployment |
| **Heroku** | ❌ Paid only | ⭐⭐⭐ | Production apps |

### Vercel Note (Why You See 404)

If you deploy this repository directly to Vercel without an entry route,
Vercel returns:

```text
Failed to load resource: the server responded with a status of 404
```

This project's dashboard is built with Streamlit, which is not a native
long-running runtime on Vercel. To avoid 404 on your Vercel URL, this repo now
includes:

- `index.html` as a landing page
- `vercel.json` rewrite rules so all routes resolve to `index.html`

Result: your Vercel deployment always serves a valid page and links to the live
Streamlit dashboard.

### Quick Vercel Redeploy Steps

```bash
git add vercel.json index.html docs/DEPLOYMENT_GUIDE.md
git commit -m "Fix Vercel 404 with static entry route"
git push
```

Then trigger a new deploy in Vercel (or wait for auto-deploy).

---

## ✅ Option 1: Streamlit Cloud (Recommended)

**Best for:** Quick, free, and officially supported Streamlit deployment.

### Step 1: Prepare Your Project

✅ Already done! Your project has:
- ✅ `requirements.txt` - Python dependencies
- ✅ `packages.txt` - System dependencies for OpenCV
- ✅ `.streamlit/config.toml` - Streamlit configuration
- ✅ `.gitignore` - Excludes sensitive files

### Step 2: Push to GitHub

```bash
# Initialize git (if not already done)
cd /Users/faizahmadkhan/Desktop/PROJECT-NETRA
git init

# Add all files
git add .

# Commit
git commit -m "Prepare for deployment - Add web dashboard"

# Create a new repository on GitHub
# Then connect and push:
git remote add origin https://github.com/YOUR_USERNAME/PROJECT-NETRA.git
git branch -M main
git push -u origin main
```

### Step 3: Deploy on Streamlit Cloud

1. **Go to:** https://share.streamlit.io/

2. **Sign in** with GitHub

3. **Click "New app"**

4. **Configure:**
   - Repository: `YOUR_USERNAME/PROJECT-NETRA`
   - Branch: `main`
   - Main file path: `src/web_dashboard.py`

5. **Advanced settings (Optional):**
   - Python version: 3.11 or 3.12
   - Add secrets if needed

6. **Click "Deploy"**

7. **Wait 5-10 minutes** for deployment

8. **Your app will be live at:**
   ```
   https://YOUR_USERNAME-project-netra.streamlit.app
   ```

### Step 4: Add Sample Data (Important!)

Since your CSV files are gitignored, you need sample data:

**Option A: Include Sample Data**
```bash
# Create a sample data file
mkdir -p data/traffic_logs
echo "Timestamp,Lane1_Count,Lane2_Count,Ambulance_Detected,Green_Time_L1,Green_Time_L2
10:15:30,5,3,0,15,11
10:15:35,7,4,0,19,13
10:15:40,6,8,1,17,21" > data/traffic_logs/Sample_Traffic_Data.csv

# Commit and push
git add data/traffic_logs/Sample_Traffic_Data.csv
git commit -m "Add sample traffic data"
git push
```

**Option B: Modify .gitignore to track one file**
```bash
# Edit .gitignore to allow sample data
echo "!data/traffic_logs/Sample_Traffic_Data.csv" >> .gitignore
```

---

## 🔧 Option 2: Render

**Free tier includes:** 512 MB RAM, auto-sleep after 15 min inactivity

### Step 1: Create `render.yaml`

```yaml
services:
  - type: web
    name: netra-dashboard
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run src/web_dashboard.py --server.port $PORT --server.address 0.0.0.0
    envVars:
      - key: PYTHON_VERSION
        value: 3.11
```

### Step 2: Deploy

1. Go to https://render.com
2. Sign up/Sign in with GitHub
3. Click "New +" → "Web Service"
4. Connect your GitHub repository
5. Render auto-detects settings
6. Click "Create Web Service"
7. Deployment takes 5-10 minutes

---

## 🚂 Option 3: Railway

**Free tier:** $5 credit/month

### Step 1: Create `railway.toml`

```toml
[build]
builder = "NIXPACKS"

[deploy]
startCommand = "streamlit run src/web_dashboard.py --server.port $PORT --server.address 0.0.0.0"
```

### Step 2: Deploy

1. Go to https://railway.app
2. Sign in with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select PROJECT-NETRA repository
5. Add environment variables if needed
6. Railway auto-deploys

---

## ⚠️ Important Notes

### Model Files (YOLO Weights)

Your model files (`yolov8m.pt`, `best.pt`) are **gitignored** because they're large (50MB+). 

**For demonstration deployment:**
- The web dashboard works **without** running `main.py` live
- It only displays **CSV data** you've already collected
- Models are **NOT needed** for the dashboard

**If you need live detection online:**
- Use Git LFS (Large File Storage)
- Or host models separately (AWS S3, Google Drive)
- Or use smaller YOLOv8n model

### Sample Data Strategy

**Recommended:** Create a demo dataset
```bash
# Copy your best data as sample
cp data/traffic_logs/Traffic_Data_LATEST.csv data/traffic_logs/Sample_Traffic_Data.csv

# Force add to git
git add -f data/traffic_logs/Sample_Traffic_Data.csv
git commit -m "Add sample data for demo"
git push
```

---

## 🐛 Troubleshooting

### Error: "No traffic data found"

**Solution:** Add sample CSV file to repository
```bash
git add -f data/traffic_logs/*.csv
git commit -m "Add traffic data"
git push
```

### Error: "OpenCV error - libGL.so.1"

**Solution (Streamlit Cloud dashboard):** keep `packages.txt` empty.

The dashboard (`src/web_dashboard.py`) does not import OpenCV directly, so no
apt system packages are required for cloud deployment.

If you later deploy OpenCV-dependent scripts in cloud runtime, add only the
minimum required package and test the install on your target Debian version.

### Error: "Memory limit exceeded"

**Solutions:**
1. Use `opencv-python-headless` instead of `opencv-python`
2. Remove unused imports
3. Optimize caching with `@st.cache_data`

### App is slow

**Solutions:**
1. Reduce cache TTL: `@st.cache_data(ttl=300)`
2. Limit data loading: Only load recent files
3. Upgrade to paid tier for more resources

---

## 🎉 Quick Start (Fastest Method)

```bash
# 1. Make sure you're in project directory
cd /Users/faizahmadkhan/Desktop/PROJECT-NETRA

# 2. Add sample data
mkdir -p data/traffic_logs
echo "Timestamp,Lane1_Count,Lane2_Count,Ambulance_Detected,Green_Time_L1,Green_Time_L2
10:15:30,5,3,0,15,11
10:15:35,7,4,0,19,13
10:15:40,6,8,1,17,21
10:15:45,10,5,0,25,15
10:15:50,8,9,0,21,23
10:15:55,12,6,1,29,17" > data/traffic_logs/Sample_Traffic_Data.csv

# 3. Commit everything
git add .
git commit -m "Ready for deployment"

# 4. Push to GitHub
git remote add origin https://github.com/YOUR_USERNAME/PROJECT-NETRA.git
git push -u origin main

# 5. Deploy on Streamlit Cloud
# Visit: https://share.streamlit.io
# Repository: YOUR_USERNAME/PROJECT-NETRA
# Main file: src/web_dashboard.py
# Click Deploy!
```

---

## 📊 What Gets Deployed

✅ Web dashboard (`src/web_dashboard.py`)  
✅ Sample traffic data  
✅ Analytics module  
✅ Documentation  
✅ Configuration files  

❌ Video files (too large)  
❌ Model weights (can be added with Git LFS)  
❌ Main detection script (dashboard only)  

---

## 🔗 Additional Resources

- [Streamlit Cloud Docs](https://docs.streamlit.io/streamlit-community-cloud)
- [Render Python Guide](https://render.com/docs/deploy-streamlit)
- [Railway Deployment Guide](https://docs.railway.app/deploy/deployments)

---

## 💡 Pro Tips

1. **Use environment variables** for sensitive data
2. **Monitor usage** - Free tiers have limits
3. **Add analytics** to track visitors (Google Analytics)
4. **Custom domain** available on paid tiers
5. **Auto-deploy** enabled - push to GitHub = auto-update

---

**Ready to deploy?** Follow **Option 1** for the easiest path! 🚀
