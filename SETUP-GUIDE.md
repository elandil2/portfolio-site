# Portfolio Setup Guide

## 🎯 Quick Start

Your portfolio is now set up with **two separate sections**:

### 1. **Public Portfolio** (What visitors see)
- **URL**: `http://localhost:5173/`
- This is your main portfolio page
- Shows your photo, name, title, and published projects
- Has light/dark mode toggle (moon/sun icon)
- **NO admin UI visible** - clean and professional

### 2. **Admin Panel** (Your private dashboard)
- **URL**: `http://localhost:5173/admin`
- Manage all your projects
- Dashboard with stats
- Settings page
- Only you access this

---

## 📸 Adding Your Photo

1. **Save your photo** as `avatar.jpg` (or `avatar.png`)
2. **Place it in**: `public/avatar.jpg`
3. **Refresh the page** - it will appear automatically!

### Example:
```
portfolio-site/
├── public/
│   ├── avatar.jpg  ← PUT YOUR PHOTO HERE
│   └── ...
```

**Photo Requirements:**
- Square image (500x500px or larger recommended)
- Professional headshot
- Formats: JPG, PNG, or WebP

---

## 🌐 How It Works

### For Visitors (Public)
1. They visit your site: `yoursite.com`
2. They see a beautiful portfolio with:
   - Your photo with green gradient border
   - "Hello, I'm Safi Cengiz"
   - "Data Scientist" title
   - Your bio
   - Published projects
   - Light/Dark mode toggle

### For You (Admin)
1. Click tiny "⚙️ Admin" link in bottom-right corner
2. Or visit: `yoursite.com/admin`
3. You see the admin panel with sidebar:
   - 📊 Dashboard
   - 📁 Projects (add/edit/delete)
   - ⚙️ Settings

---

## ✅ What's Fixed

### 1. ✅ **Separate Public/Admin Views**
- `/` = Public portfolio (NO sidebar, NO admin UI)
- `/admin` = Admin panel (WITH sidebar and management tools)

### 2. ✅ **Light & Dark Mode**
- Click moon 🌙 icon for dark mode
- Click sun ☀️ icon for light mode
- Preference saved automatically

### 3. ✅ **Avatar System**
- Just drop `avatar.jpg` in `public` folder
- Automatically displays with green gradient border
- Responsive on all devices

### 4. ✅ **Correct Navigation**
- Admin sidebar only shows in `/admin` routes
- "Back to Portfolio" button in admin
- No confusion between public/admin

---

## 🚀 Using the Admin Panel

### Adding a Project:
1. Go to `http://localhost:5173/admin/projects`
2. Click **"+ Add Project"** button
3. Fill in:
   - Title
   - Description (rich text editor)
   - Upload images (drag & drop)
   - Video URL (optional)
   - GitHub link
   - Live demo link
   - Tech stack tags (press Enter to add)
   - Published/Draft toggle
4. Click **"Create Project"**

### Managing Projects:
- **Drag** cards to reorder
- Click **"Edit"** to modify
- Click **"Delete"** to remove
- **Select multiple** for bulk actions
- **Search** to filter projects
- **Published** projects appear on public portfolio

---

## 🎨 Theme Toggle

The theme toggle appears in the **top-right corner** of the public portfolio:
- 🌙 = Switch to dark mode
- ☀️ = Switch to light mode

Colors change for:
- Background
- Text
- Cards
- All UI elements

---

## 📂 Project Structure

```
portfolio-site/
├── public/
│   └── avatar.jpg          ← YOUR PHOTO
├── src/
│   ├── components/         ← Reusable UI
│   ├── context/           ← Theme context
│   ├── layouts/           ← Admin layout
│   ├── pages/
│   │   ├── Portfolio.tsx  ← Public portfolio
│   │   ├── Dashboard.tsx  ← Admin dashboard
│   │   ├── ProjectsManager.tsx
│   │   └── Settings.tsx
│   ├── styles/
│   │   ├── theme.ts       ← Light/Dark themes
│   │   └── GlobalStyles.ts
│   └── utils/
│       └── avatar.ts      ← Avatar helper
```

---

## 🔑 Key Routes

| URL | What You See |
|-----|--------------|
| `/` | Public portfolio (visitor view) |
| `/admin` | Admin dashboard |
| `/admin/projects` | Project manager |
| `/admin/settings` | Settings & export |

---

## 💡 Pro Tips

1. **Start with Projects**: Add 2-3 projects in admin panel
2. **Publish Them**: Toggle status to "Published"
3. **Check Public View**: Go to `/` to see them live
4. **Test Dark Mode**: Try both themes
5. **Mobile Test**: Resize browser to see responsive design

---

## 🎯 Next Steps

1. ✅ Add your photo to `public/avatar.jpg`
2. ✅ Visit `/admin/projects` and add your first project
3. ✅ Publish it and view on main page (`/`)
4. ✅ Test dark mode toggle
5. ✅ Customize your bio in `src/pages/Portfolio.tsx` (line 379-380)

---

## 🆘 Need Help?

### Photo Not Showing?
- Check file name is exactly `avatar.jpg`
- Check it's in `public` folder
- Refresh browser (Ctrl+F5)

### Admin Panel Not Working?
- Make sure you're at `/admin` not `/`
- Check sidebar shows "Admin" at top

### Projects Not Showing?
- Check project status is "Published"
- Go to `/admin/projects` to verify

---

## 🎨 Customization

### Change Your Name/Title:
Edit `src/pages/Portfolio.tsx`:
- Line 363: "Safi Cengiz" ← Your name
- Line 371: "Data Scientist" ← Your title
- Line 379-380: Your bio

### Change Email:
Edit `src/pages/Portfolio.tsx`:
- Line 397: `mailto:safi@example.com`

---

Enjoy your beautiful DataCamp-style portfolio! 🚀✨
