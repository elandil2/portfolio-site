import { Outlet } from 'react-router-dom';
import styled from 'styled-components';
import { AnimatedBackground } from '../components/AnimatedBackground';
import { Sidebar } from '../components/Sidebar';

const AppContainer = styled.div`
  display: flex;
  min-height: 100vh;
`;

const MainContent = styled.main`
  flex: 1;
  margin-left: 250px;
  min-height: 100vh;

  @media (max-width: 768px) {
    margin-left: 70px;
  }
`;

export const AdminLayout = () => {
  return (
    <AppContainer>
      <AnimatedBackground />
      <Sidebar />
      <MainContent>
        <Outlet />
      </MainContent>
    </AppContainer>
  );
};
