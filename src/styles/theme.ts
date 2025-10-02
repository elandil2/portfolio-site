export const lightTheme = {
  colors: {
    primary: '#05192D',
    secondary: '#03EF62',
    accent: '#00D4FF',
    background: '#FFFFFF',
    surface: '#F5F8FA',
    surfaceHover: '#E8F0F7',
    text: '#05192D',
    textSecondary: '#5A6C7D',
    success: '#03EF62',
    error: '#FF3B3B',
    warning: '#FFB800',
    border: '#E0E7EE',
  },

  glassmorphism: {
    background: 'rgba(255, 255, 255, 0.95)',
    backdropFilter: 'blur(20px) saturate(180%)',
    border: '1px solid rgba(5, 25, 45, 0.08)',
    shadow: '0 4px 24px 0 rgba(5, 25, 45, 0.08)',
  },

  spacing: {
    xs: '0.5rem',
    sm: '1rem',
    md: '1.5rem',
    lg: '2rem',
    xl: '3rem',
    xxl: '4rem',
  },

  borderRadius: {
    sm: '8px',
    md: '12px',
    lg: '16px',
    xl: '24px',
  },

  transitions: {
    fast: '150ms ease-in-out',
    normal: '250ms ease-in-out',
    slow: '400ms ease-in-out',
  },
};

export const darkTheme = {
  colors: {
    primary: '#FFFFFF',
    secondary: '#03EF62',
    accent: '#00D4FF',
    background: '#05192D',
    surface: '#0A2540',
    surfaceHover: '#0F2F4F',
    text: '#FFFFFF',
    textSecondary: '#8BA3B8',
    success: '#03EF62',
    error: '#FF3B3B',
    warning: '#FFB800',
    border: '#1A3A5A',
  },

  glassmorphism: {
    background: 'rgba(10, 37, 64, 0.95)',
    backdropFilter: 'blur(20px) saturate(180%)',
    border: '1px solid rgba(255, 255, 255, 0.08)',
    shadow: '0 4px 24px 0 rgba(0, 0, 0, 0.3)',
  },

  spacing: {
    xs: '0.5rem',
    sm: '1rem',
    md: '1.5rem',
    lg: '2rem',
    xl: '3rem',
    xxl: '4rem',
  },

  borderRadius: {
    sm: '8px',
    md: '12px',
    lg: '16px',
    xl: '24px',
  },

  transitions: {
    fast: '150ms ease-in-out',
    normal: '250ms ease-in-out',
    slow: '400ms ease-in-out',
  },
};

export type Theme = typeof lightTheme;
