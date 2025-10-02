# 🚀 Deployment Guide

## Deploy to GitHub Pages

### Step 1: Prepare for Deployment

1. **Install gh-pages**:
```bash
npm install --save-dev gh-pages
```

2. **Update `package.json`** - Add these lines:
```json
{
  "homepage": "https://elandil2.github.io/portfolio",
  "scripts": {
    "predeploy": "npm run build",
    "deploy": "gh-pages -d dist"
  }
}
```

3. **Update `vite.config.ts`** - Add base URL:
```typescript
export default defineConfig({
  plugins: [react()],
  base: '/portfolio/', // Change 'portfolio' to your repo name
})
```

### Step 2: Push to GitHub

```bash
# Add all files
git add .

# Commit
git commit -m "Initial commit - Portfolio site"

# Create repository on GitHub first, then:
git branch -M main
git remote add origin https://github.com/elandil2/portfolio.git
git push -u origin main
```

### Step 3: Deploy

```bash
npm run deploy
```

This will:
- Build your site
- Create a `gh-pages` branch
- Push the build to GitHub Pages

### Step 4: Enable GitHub Pages

1. Go to your repo: `https://github.com/elandil2/portfolio`
2. Click **Settings** → **Pages**
3. Under **Source**, select: `gh-pages` branch
4. Click **Save**

Your site will be live at: `https://elandil2.github.io/portfolio/`

---

## 🔐 Admin Access

**Secret Admin URL**: `https://elandil2.github.io/portfolio/safi-admin-2024`

⚠️ **Don't share this URL publicly!**

---

## Alternative: Deploy to Netlify (Easier!)

1. Create account at [netlify.com](https://netlify.com)
2. Click **"Add new site"** → **"Import an existing project"**
3. Connect your GitHub repo
4. Build settings:
   - Build command: `npm run build`
   - Publish directory: `dist`
5. Click **Deploy**

Done! You'll get a custom URL like: `your-portfolio.netlify.app`

**Admin URL**: `your-portfolio.netlify.app/safi-admin-2024`

---

## Alternative: Deploy to Vercel (Best for React!)

1. Create account at [vercel.com](https://vercel.com)
2. Click **"Add New Project"**
3. Import your GitHub repo
4. Click **Deploy**

Done! You'll get a URL like: `your-portfolio.vercel.app`

**Admin URL**: `your-portfolio.vercel.app/safi-admin-2024`

---

## 📝 Adding Your Photo

Place your photo as `public/avatar.jpg` before deploying!

---

## 🛠️ Local Development

```bash
npm run dev       # Start dev server
npm run build     # Build for production
npm run preview   # Preview production build
```

---

## 🎯 Next Steps After Deployment

1. ✅ Visit your admin panel: `/safi-admin-2024`
2. ✅ Add your first project
3. ✅ Publish it
4. ✅ Visit public portfolio to see it live!
5. ✅ Share your portfolio URL with recruiters!

---

## ⚠️ Important Notes

### Images:
- Images are stored as **Base64** in IndexedDB
- Keep images under **5MB** each
- For best performance, use optimized images (compressed JPG/PNG)

### Data Storage:
- All data stored **locally** in browser (IndexedDB)
- No backend needed!
- Data persists even after refresh
- Export data regularly from Settings page

### Browser Support:
- Works in all modern browsers (Chrome, Firefox, Safari, Edge)
- IndexedDB supported everywhere

---

## 🔒 Security Reminder

The secret admin URL (`/safi-admin-2024`) provides basic security through obscurity:
- Don't share the URL
- Don't link to it from public pages
- For stronger security, consider adding password authentication

---

## 📊 What You Built

- ✅ Modern portfolio with animations
- ✅ Admin panel with drag-and-drop
- ✅ Light/Dark mode
- ✅ Fully responsive
- ✅ No backend needed
- ✅ Free hosting on GitHub/Netlify/Vercel

**You're ready to deploy! 🎉**
