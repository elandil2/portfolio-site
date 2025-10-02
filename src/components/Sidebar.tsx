import { NavLink } from 'react-router-dom';
import { motion } from 'framer-motion';
import styled from 'styled-components';
import { lightTheme as theme } from '../styles/theme';

const SidebarContainer = styled(motion.nav)`
  width: 250px;
  height: 100vh;
  position: fixed;
  left: 0;
  top: 0;
  background: ${theme.colors.background};
  border-right: 1px solid ${theme.colors.border};
  padding: ${theme.spacing.xl} ${theme.spacing.md};
  display: flex;
  flex-direction: column;
  z-index: 100;
  box-shadow: 4px 0 24px rgba(5, 25, 45, 0.04);

  @media (max-width: 768px) {
    width: 70px;
    padding: ${theme.spacing.lg} ${theme.spacing.sm};
  }
`;

const Logo = styled.div`
  font-size: 1.5rem;
  font-weight: 800;
  background: linear-gradient(135deg, #03EF62, #00D4FF);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: ${theme.spacing.xxl};
  text-align: center;

  @media (max-width: 768px) {
    font-size: 1.2rem;
  }
`;

const NavList = styled.ul`
  list-style: none;
  padding: 0;
  margin: 0;
  flex: 1;
`;

const NavItem = styled.li`
  margin-bottom: ${theme.spacing.xs};
`;

const StyledNavLink = styled(NavLink)`
  display: flex;
  align-items: center;
  gap: ${theme.spacing.sm};
  padding: ${theme.spacing.sm} ${theme.spacing.md};
  color: ${theme.colors.textSecondary};
  text-decoration: none;
  border-radius: ${theme.borderRadius.sm};
  transition: all ${theme.transitions.fast};
  font-size: 0.95rem;
  font-weight: 600;

  &:hover {
    background: ${theme.colors.surface};
    color: ${theme.colors.text};
  }

  &.active {
    background: linear-gradient(135deg, #03EF62, #00D4FF);
    color: white;
    box-shadow: 0 4px 12px rgba(3, 239, 98, 0.25);
  }

  .icon {
    font-size: 1.3rem;
  }

  .label {
    @media (max-width: 768px) {
      display: none;
    }
  }
`;

const BackButton = styled.a`
  display: flex;
  align-items: center;
  gap: ${theme.spacing.sm};
  padding: ${theme.spacing.sm} ${theme.spacing.md};
  color: ${theme.colors.textSecondary};
  text-decoration: none;
  border-radius: ${theme.borderRadius.sm};
  transition: all ${theme.transitions.fast};
  font-size: 0.95rem;
  font-weight: 600;
  border: 1px solid ${theme.colors.border};
  margin-top: auto;

  &:hover {
    background: ${theme.colors.surface};
    color: ${theme.colors.text};
  }

  .label {
    @media (max-width: 768px) {
      display: none;
    }
  }
`;

const Footer = styled.div`
  margin-top: ${theme.spacing.md};
  padding-top: ${theme.spacing.lg};
  border-top: 1px solid ${theme.colors.border};
  font-size: 0.8rem;
  color: ${theme.colors.textSecondary};
  text-align: center;

  @media (max-width: 768px) {
    display: none;
  }
`;

export const Sidebar = () => {
  const navItems = [
    { path: '/safi-admin-2024', label: 'Dashboard', icon: '📊' },
    { path: '/safi-admin-2024/projects', label: 'Projects', icon: '📁' },
    { path: '/safi-admin-2024/settings', label: 'Settings', icon: '⚙️' },
  ];

  return (
    <SidebarContainer
      initial={{ x: -250 }}
      animate={{ x: 0 }}
      transition={{ duration: 0.3 }}
    >
      <Logo>Admin</Logo>

      <NavList>
        {navItems.map((item) => (
          <NavItem key={item.path}>
            <StyledNavLink to={item.path} end={item.path === '/safi-admin-2024'}>
              <span className="icon">{item.icon}</span>
              <span className="label">{item.label}</span>
            </StyledNavLink>
          </NavItem>
        ))}
      </NavList>

      <BackButton href="/">
        <span>←</span>
        <span className="label">Back to Portfolio</span>
      </BackButton>

      <Footer>
        <p>Portfolio Admin v1.0</p>
      </Footer>
    </SidebarContainer>
  );
};
