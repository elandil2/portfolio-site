import { motion } from 'framer-motion';
import styled from 'styled-components';
import { useTheme } from '../context/ThemeContext';
import { getAvatarUrl } from '../utils/avatar';
import { lightTheme as theme } from '../styles/theme';

const Container = styled.div`
  min-height: 100vh;
  background: ${theme.colors.background};
`;

const Nav = styled.nav`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  background: ${theme.colors.background}ee;
  backdrop-filter: blur(10px);
  border-bottom: 1px solid ${theme.colors.border};
  z-index: 100;
  padding: ${theme.spacing.md} ${theme.spacing.xl};

  @media (max-width: 768px) {
    padding: ${theme.spacing.sm} ${theme.spacing.md};
  }
`;

const NavContent = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const Logo = styled.a`
  font-size: 1.5rem;
  font-weight: 800;
  background: linear-gradient(135deg, #03EF62, #00D4FF);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  text-decoration: none;
  cursor: pointer;
`;

const NavLinks = styled.div`
  display: flex;
  gap: ${theme.spacing.lg};
  align-items: center;

  @media (max-width: 768px) {
    gap: ${theme.spacing.md};
  }
`;

const NavLink = styled.a`
  color: ${theme.colors.text};
  text-decoration: none;
  font-weight: 500;
  transition: color ${theme.transitions.fast};

  &:hover {
    color: #03EF62;
  }

  @media (max-width: 768px) {
    font-size: 0.9rem;
  }
`;

const SocialIcon = styled(motion.a)`
  display: flex;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;
  background: ${theme.colors.surface};
  border: 1px solid ${theme.colors.border};
  border-radius: 50%;
  color: ${theme.colors.text};
  text-decoration: none;
  font-size: 1.1rem;
  transition: all ${theme.transitions.fast};

  &:hover {
    background: linear-gradient(135deg, #03EF62, #00D4FF);
    color: white;
    border-color: transparent;
    transform: translateY(-2px);
  }
`;

const ThemeToggle = styled(motion.button)`
  background: ${theme.colors.surface};
  border: 1px solid ${theme.colors.border};
  border-radius: 50%;
  width: 36px;
  height: 36px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  font-size: 1.1rem;
  transition: all ${theme.transitions.fast};

  &:hover {
    background: ${theme.colors.surfaceHover};
  }
`;

const Main = styled.main`
  padding-top: 80px;
`;

const Hero = styled.section`
  min-height: 85vh;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: ${theme.spacing.xxl} ${theme.spacing.xl};
  text-align: center;

  @media (max-width: 768px) {
    padding: ${theme.spacing.xl} ${theme.spacing.md};
  }
`;

const HeroContent = styled(motion.div)`
  max-width: 800px;
`;

const Avatar = styled(motion.img)`
  width: 180px;
  height: 180px;
  border-radius: 50%;
  object-fit: cover;
  margin-bottom: ${theme.spacing.lg};
  border: 4px solid ${theme.colors.background};
  box-shadow: 0 0 0 4px #03EF62;

  @media (max-width: 768px) {
    width: 150px;
    height: 150px;
  }
`;

const Greeting = styled(motion.p)`
  font-size: 1.1rem;
  color: ${theme.colors.textSecondary};
  margin-bottom: ${theme.spacing.md};
`;

const Name = styled(motion.h1)`
  font-size: 4rem;
  font-weight: 800;
  color: ${theme.colors.text};
  margin-bottom: ${theme.spacing.sm};
  line-height: 1.1;

  @media (max-width: 768px) {
    font-size: 2.5rem;
  }
`;

const Title = styled(motion.h2)`
  font-size: 2rem;
  font-weight: 600;
  background: linear-gradient(135deg, #03EF62, #00D4FF);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: ${theme.spacing.lg};

  @media (max-width: 768px) {
    font-size: 1.5rem;
  }
`;

const Bio = styled(motion.p)`
  font-size: 1.15rem;
  color: ${theme.colors.textSecondary};
  line-height: 1.8;
  margin-bottom: ${theme.spacing.xl};
`;

const Section = styled.section`
  padding: ${theme.spacing.xxl} ${theme.spacing.xl};
  max-width: 1200px;
  margin: 0 auto;

  @media (max-width: 768px) {
    padding: ${theme.spacing.xl} ${theme.spacing.md};
  }
`;

const SectionTitle = styled(motion.h2)`
  font-size: 2.5rem;
  font-weight: 700;
  color: ${theme.colors.text};
  margin-bottom: ${theme.spacing.lg};
  text-align: center;

  @media (max-width: 768px) {
    font-size: 2rem;
  }
`;

const SectionSubtitle = styled.p`
  font-size: 1.1rem;
  color: ${theme.colors.textSecondary};
  text-align: center;
  max-width: 600px;
  margin: 0 auto ${theme.spacing.xxl};
  line-height: 1.6;
`;

const ProjectsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: ${theme.spacing.xl};

  @media (max-width: 768px) {
    grid-template-columns: 1fr;
  }
`;

const ProjectCard = styled(motion.div)`
  background: ${theme.colors.surface};
  border: 1px solid ${theme.colors.border};
  border-radius: ${theme.borderRadius.lg};
  overflow: hidden;
  transition: all ${theme.transitions.normal};
  cursor: pointer;

  &:hover {
    transform: translateY(-8px);
    box-shadow: 0 12px 40px rgba(3, 239, 98, 0.15);
    border-color: #03EF62;
  }
`;

const ProjectImage = styled.div`
  width: 100%;
  height: 200px;
  background: linear-gradient(135deg, #03EF62, #00D4FF);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 4rem;
  color: white;
`;

const ProjectContent = styled.div`
  padding: ${theme.spacing.lg};
`;

const ProjectTitle = styled.h3`
  font-size: 1.5rem;
  font-weight: 700;
  color: ${theme.colors.text};
  margin-bottom: ${theme.spacing.sm};
`;

const ProjectDescription = styled.p`
  color: ${theme.colors.textSecondary};
  line-height: 1.6;
  margin-bottom: ${theme.spacing.md};
`;

const TechStack = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: ${theme.spacing.xs};
`;

const TechTag = styled.span`
  padding: 4px 12px;
  background: linear-gradient(135deg, rgba(3, 239, 98, 0.1), rgba(0, 212, 255, 0.1));
  border: 1px solid rgba(3, 239, 98, 0.3);
  border-radius: 12px;
  font-size: 0.85rem;
  color: ${theme.colors.text};
  font-weight: 600;
`;

const Footer = styled.footer`
  text-align: center;
  padding: ${theme.spacing.xl};
  color: ${theme.colors.textSecondary};
  border-top: 1px solid ${theme.colors.border};
`;

export const PortfolioClean = () => {
  const { mode, toggleTheme } = useTheme();

  const placeholderProjects = [
    {
      id: 1,
      emoji: '📊',
      title: 'Sales Forecasting Dashboard',
      description: 'Machine learning model to predict sales trends using time-series analysis. Built interactive dashboard for stakeholders.',
      tech: ['Python', 'TensorFlow', 'Pandas', 'Plotly']
    },
    {
      id: 2,
      emoji: '🤖',
      title: 'Customer Churn Prediction',
      description: 'Predictive model to identify customers likely to churn. Improved retention rate by 23% through targeted interventions.',
      tech: ['Python', 'Scikit-learn', 'XGBoost', 'SQL']
    }
  ];

  const scrollTo = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <Container>
      <Nav>
        <NavContent>
          <Logo href="#" onClick={() => scrollTo('hero')}>SC</Logo>
          <NavLinks>
            <NavLink href="#projects" onClick={() => scrollTo('projects')}>Projects</NavLink>
            <SocialIcon
              href="https://www.linkedin.com/in/safi-cengiz/"
              target="_blank"
              rel="noopener noreferrer"
              whileHover={{ scale: 1.1 }}
              title="LinkedIn"
            >
              💼
            </SocialIcon>
            <SocialIcon
              href="https://github.com/elandil2"
              target="_blank"
              rel="noopener noreferrer"
              whileHover={{ scale: 1.1 }}
              title="GitHub"
            >
              🔗
            </SocialIcon>
            <ThemeToggle
              onClick={toggleTheme}
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.95 }}
            >
              {mode === 'light' ? '🌙' : '☀️'}
            </ThemeToggle>
          </NavLinks>
        </NavContent>
      </Nav>

      <Main>
        <Hero id="hero">
          <HeroContent
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <Avatar
              src={getAvatarUrl()}
              alt="Safi Cengiz"
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ type: 'spring', stiffness: 200, delay: 0.2 }}
            />
            <Greeting
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.4 }}
            >
              Hey, I'm
            </Greeting>
            <Name
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
            >
              Safi Cengiz
            </Name>
            <Title
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
            >
              Data Scientist
            </Title>
            <Bio
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.7 }}
            >
              Passionate about transforming data into actionable insights.
              Specializing in machine learning, statistical analysis, and
              creating impactful data-driven solutions.
            </Bio>
          </HeroContent>
        </Hero>

        <Section id="projects">
          <SectionTitle
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            Featured Projects
          </SectionTitle>
          <SectionSubtitle>
            A collection of my recent work in data science and machine learning.
          </SectionSubtitle>
          <ProjectsGrid>
            {placeholderProjects.map((project, index) => (
              <ProjectCard
                key={project.id}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
              >
                <ProjectImage>{project.emoji}</ProjectImage>
                <ProjectContent>
                  <ProjectTitle>{project.title}</ProjectTitle>
                  <ProjectDescription>{project.description}</ProjectDescription>
                  <TechStack>
                    {project.tech.map((tech, idx) => (
                      <TechTag key={idx}>{tech}</TechTag>
                    ))}
                  </TechStack>
                </ProjectContent>
              </ProjectCard>
            ))}
          </ProjectsGrid>
        </Section>
      </Main>

      <Footer>
        <p>© 2025 Safi Cengiz. Built with React & TypeScript.</p>
      </Footer>
    </Container>
  );
};
