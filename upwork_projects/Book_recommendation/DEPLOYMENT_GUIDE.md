# 🚀 Deployment Guide - GitHub + Streamlit Cloud

## Step 1: Prepare Your Repository

### Files to Upload to GitHub:
✅ `app.py` - Main application
✅ `requirements.txt` - Dependencies
✅ `README.md` - Project documentation
✅ `.gitignore` - Files to ignore
✅ `.streamlit/config.toml` - Streamlit configuration
✅ `Books.csv` - Book metadata (271K books)
✅ `Ratings.csv` - User ratings (1.1M ratings)
✅ `Users.csv` - User data (optional, not used in app)

### Files NOT to Upload (Optional):
❌ `book_rec_env/` - Virtual environment
❌ `advanced_models.py` - For local analysis only
❌ `book_recommendation_system.py` - Old version
❌ `book-recommender-system-project.ipynb` - Jupyter notebook
❌ `.playwright-mcp/` - Browser screenshots
❌ `*.pkl` - Trained model files

---

## Step 2: Upload to GitHub

### Option A: Using Git Commands (Recommended)

```bash
cd "C:\Users\safix\Desktop\upwork_projects\Book_recommendation"

# Initialize git repository
git init

# Add all files
git add .

# Create first commit
git commit -m "Initial commit: Book Recommendation System"

# Set main branch
git branch -M main

# Add remote repository
git remote add origin https://github.com/elandil2/Book-Recommend.git

# Push to GitHub
git push -u origin main
```

### Option B: Using GitHub Desktop
1. Open GitHub Desktop
2. Click "Add" → "Add existing repository"
3. Select the Book_recommendation folder
4. Click "Publish repository"
5. Choose repository name: `Book-Recommend`
6. Uncheck "Keep this code private" (or keep it private)
7. Click "Publish repository"

### Option C: Upload via GitHub Website
1. Go to https://github.com/elandil2/Book-Recommend
2. Click "uploading an existing file"
3. Drag and drop all files (except virtual environment)
4. Commit changes

---

## Step 3: Deploy to Streamlit Cloud

### 3.1 Sign Up for Streamlit Cloud
1. Go to https://streamlit.io/cloud
2. Click "Sign up" or "Sign in with GitHub"
3. Authorize Streamlit to access your GitHub repos

### 3.2 Deploy Your App
1. Click "New app" button
2. Fill in the deployment details:
   - **Repository:** `elandil2/Book-Recommend`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** Choose a custom URL like `book-recommender-ml`

3. Click "Deploy!" button

### 3.3 Wait for Deployment
- Initial deployment takes 2-5 minutes
- Streamlit Cloud will:
  - Clone your repository
  - Install dependencies from `requirements.txt`
  - Launch your app
  - Provide a public URL

### 3.4 Your App Will Be Live At:
```
https://book-recommender-ml.streamlit.app
```
*(Replace with your actual URL)*

---

## Step 4: Test Your Deployed App

### Things to Check:
✅ App loads without errors
✅ CSV files are loading (Books.csv, Ratings.csv)
✅ Search functionality works
✅ Both algorithms work (Collaborative & Content-Based)
✅ Book images display
✅ Recommendations appear correctly

### Common Issues & Solutions:

#### Issue 1: "FileNotFoundError: Books.csv"
**Solution:** Make sure CSV files are in the root directory and uploaded to GitHub

#### Issue 2: "Module not found"
**Solution:** Check that all dependencies are in `requirements.txt`

#### Issue 3: App loads slow
**Solution:** Normal! First load takes 30-60 seconds as Streamlit caches data

#### Issue 4: Images not showing
**Solution:** Some image URLs might be broken/expired - this is normal

---

## Step 5: Update Your Upwork Portfolio

### Add These Links:

**GitHub Repository:**
```
https://github.com/elandil2/Book-Recommend
```

**Live Demo:**
```
https://your-app.streamlit.app
```

**Update README.md:**
After deployment, update the README.md with your actual Streamlit Cloud URL:
```bash
git pull origin main
# Edit README.md - replace 'your-app.streamlit.app' with actual URL
git add README.md
git commit -m "Update README with live demo URL"
git push origin main
```

---

## Step 6: Monitor Your App

### Streamlit Cloud Dashboard:
- **Analytics:** View app usage and visitors
- **Logs:** Check for errors or issues
- **Settings:** Update Python version, secrets, etc.
- **Reboot:** Restart app if needed

### App URL:
```
https://share.streamlit.io/elandil2/book-recommend/main/app.py
```

---

## Troubleshooting

### If CSV Files Are Too Large for GitHub:

**Option 1: Use Git LFS (Large File Storage)**
```bash
# Install Git LFS
git lfs install

# Track large CSV files
git lfs track "*.csv"
git add .gitattributes
git add Books.csv Ratings.csv Users.csv
git commit -m "Add CSV files with LFS"
git push origin main
```

**Option 2: Use External Storage**
- Upload CSVs to Google Drive
- Make them publicly accessible
- Modify `app.py` to download from URL:
```python
@st.cache_data
def load_data():
    books_url = "https://drive.google.com/..."
    ratings_url = "https://drive.google.com/..."
    books = pd.read_csv(books_url)
    ratings = pd.read_csv(ratings_url)
    return books, ratings
```

**Option 3: Use Streamlit Secrets**
- Use Streamlit's secrets management for sensitive data
- Configure in Streamlit Cloud dashboard

---

## Maintenance

### To Update Your App:
```bash
# Make changes to code
# Commit and push
git add .
git commit -m "Update: description of changes"
git push origin main

# Streamlit Cloud will auto-redeploy!
```

### To Stop/Delete App:
1. Go to Streamlit Cloud dashboard
2. Click on your app
3. Click "Settings" → "Delete app"

---

## Optional: Add Custom Domain

If you have a custom domain:
1. Go to Streamlit Cloud dashboard
2. Click on your app → "Settings"
3. Add custom domain (e.g., `books.yourdomain.com`)
4. Update DNS records as instructed

---

## Quick Command Reference

```bash
# Initialize and push to GitHub
cd "C:\Users\safix\Desktop\upwork_projects\Book_recommendation"
git init
git add .
git commit -m "Initial commit: Book Recommendation System"
git branch -M main
git remote add origin https://github.com/elandil2/Book-Recommend.git
git push -u origin main

# Update after changes
git add .
git commit -m "Update: your message"
git push origin main

# Check status
git status

# View commit history
git log --oneline
```

---

## Success Checklist

- [ ] GitHub repository created
- [ ] All files uploaded (app.py, requirements.txt, CSVs)
- [ ] Streamlit Cloud account created
- [ ] App deployed successfully
- [ ] App is accessible via public URL
- [ ] Search and recommendations work
- [ ] README updated with live demo link
- [ ] Upwork portfolio updated with links
- [ ] Project description added to Upwork

---

**🎉 Your app is now live and ready to showcase!**

**Live Demo:** https://your-app.streamlit.app
**GitHub:** https://github.com/elandil2/Book-Recommend
