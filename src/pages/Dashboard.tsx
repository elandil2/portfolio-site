import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import styled from 'styled-components';
import { db } from '../db/database';
import { Stats } from '../types';
import { GlassCard } from '../components/GlassCard';
import { lightTheme as theme } from '../styles/theme';

const Container = styled.div`
  padding: ${theme.spacing.xl};
  max-width: 1400px;
  margin: 0 auto;

  @media (max-width: 768px) {
    padding: ${theme.spacing.md};
  }
`;

const Title = styled.h1`
  font-size: 2.5rem;
  font-weight: 700;
  background: linear-gradient(135deg, ${theme.colors.primary}, ${theme.colors.secondary});
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: ${theme.spacing.xl};

  @media (max-width: 768px) {
    font-size: 2rem;
  }
`;

const StatsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: ${theme.spacing.lg};
  margin-bottom: ${theme.spacing.xl};

  @media (max-width: 768px) {
    grid-template-columns: 1fr;
  }
`;

const StatCard = styled(GlassCard)`
  text-align: center;
  padding: ${theme.spacing.xl};
`;

const StatValue = styled(motion.div)`
  font-size: 3rem;
  font-weight: 700;
  background: linear-gradient(135deg, ${theme.colors.primary}, ${theme.colors.secondary});
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: ${theme.spacing.sm};
`;

const StatLabel = styled.div`
  font-size: 1rem;
  color: ${theme.colors.textSecondary};
  text-transform: uppercase;
  letter-spacing: 1px;
`;

const ChartSection = styled.div`
  margin-top: ${theme.spacing.xl};
`;

const ChartTitle = styled.h2`
  font-size: 1.5rem;
  font-weight: 600;
  color: ${theme.colors.text};
  margin-bottom: ${theme.spacing.lg};
`;

const ActivityList = styled(GlassCard)`
  padding: ${theme.spacing.lg};
`;

const ActivityItem = styled(motion.div)`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: ${theme.spacing.md};
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);

  &:last-child {
    border-bottom: none;
  }
`;

const ActivityInfo = styled.div`
  flex: 1;

  h4 {
    font-size: 1rem;
    font-weight: 600;
    color: ${theme.colors.text};
    margin: 0 0 ${theme.spacing.xs} 0;
  }

  p {
    font-size: 0.85rem;
    color: ${theme.colors.textSecondary};
    margin: 0;
  }
`;

const ActivityTime = styled.span`
  font-size: 0.85rem;
  color: ${theme.colors.textSecondary};
`;

const QuickActions = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: ${theme.spacing.md};
  margin-top: ${theme.spacing.xl};
`;

const ActionCard = styled(motion.div)`
  background: ${theme.glassmorphism.background};
  backdrop-filter: ${theme.glassmorphism.backdropFilter};
  border: ${theme.glassmorphism.border};
  border-radius: ${theme.borderRadius.md};
  padding: ${theme.spacing.lg};
  cursor: pointer;
  transition: all ${theme.transitions.normal};
  text-align: center;

  &:hover {
    border-color: ${theme.colors.primary};
    transform: translateY(-4px);
    box-shadow: 0 12px 48px rgba(138, 43, 226, 0.25);
  }

  h3 {
    font-size: 1.1rem;
    font-weight: 600;
    color: ${theme.colors.text};
    margin-bottom: ${theme.spacing.xs};
  }

  p {
    font-size: 0.85rem;
    color: ${theme.colors.textSecondary};
    margin: 0;
  }
`;

export const Dashboard = () => {
  const [stats, setStats] = useState<Stats>({
    totalProjects: 0,
    publishedProjects: 0,
    draftProjects: 0,
    totalViews: 0,
  });

  useEffect(() => {
    loadStats();
  }, []);

  const loadStats = async () => {
    const projects = await db.projects.toArray();
    const published = projects.filter((p) => p.status === 'published');
    const drafts = projects.filter((p) => p.status === 'draft');

    setStats({
      totalProjects: projects.length,
      publishedProjects: published.length,
      draftProjects: drafts.length,
      totalViews: Math.floor(Math.random() * 10000), // Mock data
    });
  };

  const recentActivity = [
    { id: 1, action: 'Created new project', project: 'E-commerce Platform', time: '2 hours ago' },
    { id: 2, action: 'Updated project', project: 'Portfolio Website', time: '5 hours ago' },
    { id: 3, action: 'Published project', project: 'Mobile App', time: '1 day ago' },
  ];

  return (
    <Container>
      <Title>Dashboard</Title>

      <StatsGrid>
        <StatCard>
          <StatValue
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: 'spring', stiffness: 200, delay: 0.1 }}
          >
            {stats.totalProjects}
          </StatValue>
          <StatLabel>Total Projects</StatLabel>
        </StatCard>

        <StatCard>
          <StatValue
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: 'spring', stiffness: 200, delay: 0.2 }}
          >
            {stats.publishedProjects}
          </StatValue>
          <StatLabel>Published</StatLabel>
        </StatCard>

        <StatCard>
          <StatValue
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: 'spring', stiffness: 200, delay: 0.3 }}
          >
            {stats.draftProjects}
          </StatValue>
          <StatLabel>Drafts</StatLabel>
        </StatCard>

        <StatCard>
          <StatValue
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: 'spring', stiffness: 200, delay: 0.4 }}
          >
            {stats.totalViews.toLocaleString()}
          </StatValue>
          <StatLabel>Total Views</StatLabel>
        </StatCard>
      </StatsGrid>

      <ChartSection>
        <ChartTitle>Recent Activity</ChartTitle>
        <ActivityList>
          {recentActivity.map((activity, index) => (
            <ActivityItem
              key={activity.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <ActivityInfo>
                <h4>{activity.project}</h4>
                <p>{activity.action}</p>
              </ActivityInfo>
              <ActivityTime>{activity.time}</ActivityTime>
            </ActivityItem>
          ))}
        </ActivityList>
      </ChartSection>

      <QuickActions>
        <ActionCard
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <h3>📊 Analytics</h3>
          <p>View detailed project analytics</p>
        </ActionCard>

        <ActionCard
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <h3>🎨 Themes</h3>
          <p>Customize your portfolio theme</p>
        </ActionCard>

        <ActionCard
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <h3>📤 Export</h3>
          <p>Export your portfolio data</p>
        </ActionCard>

        <ActionCard
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <h3>⚙️ Settings</h3>
          <p>Configure admin settings</p>
        </ActionCard>
      </QuickActions>
    </Container>
  );
};
