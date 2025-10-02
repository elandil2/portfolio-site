import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider as StyledThemeProvider } from 'styled-components';
import { useTheme } from './context/ThemeContext';
import { GlobalStyles } from './styles/GlobalStyles';
import { lightTheme, darkTheme } from './styles/theme';
import { Portfolio } from './pages/Portfolio';
import { AdminLayout } from './layouts/AdminLayout';
import { Dashboard } from './pages/Dashboard';
import { ProjectsManager } from './pages/ProjectsManager';
import { Settings } from './pages/Settings';

function App() {
  const { mode } = useTheme();
  const theme = mode === 'light' ? lightTheme : darkTheme;

  return (
    <StyledThemeProvider theme={theme}>
      <GlobalStyles />
      <Router>
        <Routes>
          {/* Public Portfolio Route - Main site visitors see this */}
          <Route path="/" element={<Portfolio />} />

          {/* Admin Routes - Secret admin panel URL */}
          <Route path="/safi-admin-2024" element={<AdminLayout />}>
            <Route index element={<Dashboard />} />
            <Route path="projects" element={<ProjectsManager />} />
            <Route path="settings" element={<Settings />} />
          </Route>
        </Routes>
      </Router>
    </StyledThemeProvider>
  );
}

export default App;
