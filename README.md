# Portfolio Admin Panel

A cutting-edge, ultra-modern admin panel for managing your portfolio with stunning animations and an intuitive interface.

## ✨ Features

### 🎨 Design
- **Animated Background**: Particles, geometric shapes, and gradient mesh animations
- **Glassmorphic UI**: Beautiful frosted glass cards with smooth animations
- **Dark Theme**: Vibrant accents with purple, cyan, and pink gradients
- **Fully Responsive**: Optimized for mobile and desktop

### 🚀 Core Functionality
- **Drag & Drop Reordering**: Reorder projects with smooth animations
- **Rich Project Management**:
  - Title and rich text description editor
  - Multiple image uploads with drag & drop
  - Video URL support (YouTube/Vimeo/direct)
  - GitHub and live demo links
  - Tech stack tags (searchable/creatable)
  - Published/Draft status toggle
- **Inline Editing**: Edit projects quickly
- **Bulk Actions**: Select multiple projects to delete, publish, or hide
- **Real-time Preview**: See how your portfolio looks live
- **Auto-save**: Data automatically saved to IndexedDB
- **Search & Filter**: Find projects by name, description, or tech stack
- **Export/Import**: Backup and restore data (JSON/CSV)

### 📊 Admin Sections
1. **Dashboard**: Overview with stats (total projects, published, drafts, views)
2. **Projects Manager**: Full CRUD interface with drag-and-drop
3. **Preview Mode**: Live portfolio view
4. **Settings**: Theme customization, data export/import, danger zone

## 🛠️ Tech Stack

- **React 18** + **TypeScript**
- **Framer Motion** - Animations & drag-and-drop
- **Styled Components** - Styling
- **React Hook Form** - Form handling
- **Dexie.js** - IndexedDB wrapper for data persistence
- **React Dropzone** - File uploads
- **React Quill** - Rich text editor
- **React Player** - Video previews
- **Vite** - Build tool

## 🚀 Getting Started

### Installation

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

### Development

The app will be available at `http://localhost:5173`

## 📁 Project Structure

```
portfolio-site/
├── src/
│   ├── components/          # Reusable UI components
│   │   ├── AnimatedBackground.tsx
│   │   ├── GlassCard.tsx
│   │   ├── Button.tsx
│   │   ├── ProjectCard.tsx
│   │   ├── ProjectModal.tsx
│   │   └── Sidebar.tsx
│   ├── pages/              # Page components
│   │   ├── Dashboard.tsx
│   │   ├── ProjectsManager.tsx
│   │   ├── Preview.tsx
│   │   └── Settings.tsx
│   ├── db/                 # Database configuration
│   │   └── database.ts
│   ├── styles/             # Global styles and theme
│   │   ├── GlobalStyles.ts
│   │   └── theme.ts
│   ├── types/              # TypeScript types
│   │   └── index.ts
│   ├── App.tsx             # Main app component
│   └── main.tsx            # App entry point
├── index.html
├── package.json
├── tsconfig.json
└── vite.config.ts
```

## 🎯 Key Features Explained

### Drag & Drop Reordering
Projects can be reordered using Framer Motion's `Reorder` component. The order is automatically saved to IndexedDB.

### Data Persistence
All data is stored locally in IndexedDB using Dexie.js. This means:
- No backend required
- Works offline
- Fast and reliable
- Can be exported/imported

### Animations
- **Background**: Animated particles and gradient meshes
- **Cards**: Smooth hover effects and transitions
- **Drag**: Visual feedback during reordering
- **Page transitions**: Animated route changes

### Export/Import
- **JSON**: Full data backup with all project details
- **CSV**: Simplified project list for spreadsheet tools

## 🎨 Customization

### Theme Colors
Edit `src/styles/theme.ts` to customize colors:

```typescript
colors: {
  primary: '#8a2be2',      // Purple
  secondary: '#00d9ff',    // Cyan
  accent: '#ff006e',       // Pink
  // ... more colors
}
```

### Glassmorphic Effect
Adjust the glassmorphism settings in `theme.ts`:

```typescript
glassmorphism: {
  background: 'rgba(255, 255, 255, 0.03)',
  backdropFilter: 'blur(20px) saturate(180%)',
  border: '1px solid rgba(255, 255, 255, 0.1)',
  shadow: '0 8px 32px 0 rgba(138, 43, 226, 0.15)',
}
```

## 📱 Responsive Design

The admin panel is fully responsive:
- **Desktop**: Full sidebar with labels
- **Tablet**: Optimized layouts
- **Mobile**: Icon-only sidebar, stacked cards

## 🔒 Data Management

### Export Data
Go to Settings → Export Data → Choose format (JSON/CSV)

### Import Data
Go to Settings → Import Data → Upload JSON file

### Clear Data
Settings → Danger Zone → Clear All Data (⚠️ irreversible)

## 🚀 Performance

- **Framer Motion**: Hardware-accelerated animations
- **IndexedDB**: Fast local storage
- **Code splitting**: Optimized bundle sizes
- **Lazy loading**: Images and components loaded on demand

## 🐛 Troubleshooting

### Node Version Warning
The project requires Node.js 20+ for optimal performance. If you see warnings, consider upgrading Node.js.

### Port Already in Use
If port 5173 is busy, Vite will automatically use the next available port.

## 📄 License

MIT

## 🤝 Contributing

Feel free to open issues and pull requests!

---

Built with ❤️ using React, TypeScript, and Framer Motion
