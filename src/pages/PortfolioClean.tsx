import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import styled from 'styled-components';
import { useTheme } from '../context/ThemeContext';
import { getAvatarUrl } from '../utils/avatar';
import { lightTheme as theme } from '../styles/theme';

const Container = styled.div`
  min-height: 100vh;
  background: rgb(249, 250, 251);
`;

const Nav = styled.nav`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  background: rgba(249, 250, 251, 0.95);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid rgba(0, 0, 0, 0.1);
  border-top: 3px solid #00C853;
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
  color: rgb(0, 0, 0);
  text-decoration: none;
  cursor: pointer;
  transition: color 0.3s ease;

  &:hover {
    color: #00C853;
  }
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
  color: rgb(0, 0, 0);
  text-decoration: none;
  font-weight: 500;
  transition: color 0.3s ease;

  &:hover {
    color: #00C853;
  }

  @media (max-width: 768px) {
    font-size: 0.9rem;
  }
`;

const SocialIcon = styled(motion.a)`
  display: flex;
  align-items: center;
  justify-content: center;
  width: 40px;
  height: 40px;
  background: white;
  border: 1px solid rgba(0, 0, 0, 0.1);
  border-radius: 50%;
  color: rgb(0, 0, 0);
  text-decoration: none;
  font-size: 1.2rem;
  transition: all 0.3s ease;

  &:hover {
    background: #00C853;
    color: white;
    border-color: #00C853;
    transform: translateY(-4px) scale(1.1);
    box-shadow: 0 8px 16px rgba(0, 200, 83, 0.3);
  }
`;

const ThemeToggle = styled(motion.button)`
  background: white;
  border: 1px solid rgba(0, 0, 0, 0.1);
  border-radius: 50%;
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  font-size: 1.2rem;
  transition: all 0.3s ease;

  &:hover {
    background: #00C853;
    border-color: #00C853;
    transform: translateY(-4px) scale(1.1);
    box-shadow: 0 8px 16px rgba(0, 200, 83, 0.3);
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
  border: 4px solid white;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
  transition: all 0.3s ease;
  cursor: pointer;

  &:hover {
    transform: scale(1.1) translateY(-8px);
    box-shadow: 0 12px 40px rgba(0, 200, 83, 0.3);
    border-color: #00C853;
  }

  @media (max-width: 768px) {
    width: 150px;
    height: 150px;
  }
`;

const Greeting = styled(motion.p)`
  font-size: 1.1rem;
  color: rgb(100, 100, 100);
  margin-bottom: ${theme.spacing.md};
`;

const Name = styled(motion.h1)`
  font-size: 4rem;
  font-weight: 800;
  color: rgb(0, 0, 0);
  margin-bottom: ${theme.spacing.sm};
  line-height: 1.1;

  @media (max-width: 768px) {
    font-size: 2.5rem;
  }
`;

const Title = styled(motion.h2)`
  font-size: 2rem;
  font-weight: 600;
  color: #00C853;
  margin-bottom: ${theme.spacing.lg};

  @media (max-width: 768px) {
    font-size: 1.5rem;
  }
`;

const Bio = styled(motion.p)`
  font-size: 1.15rem;
  color: rgb(100, 100, 100);
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
  color: rgb(0, 0, 0);
  margin-bottom: ${theme.spacing.lg};
  text-align: center;

  @media (max-width: 768px) {
    font-size: 2rem;
  }
`;

const SectionSubtitle = styled.p`
  font-size: 1.1rem;
  color: rgb(100, 100, 100);
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
  background: white;
  border: 1px solid rgba(0, 0, 0, 0.1);
  border-radius: 12px;
  overflow: hidden;
  transition: all 0.3s ease;
  cursor: pointer;

  &:hover {
    transform: translateY(-8px);
    box-shadow: 0 12px 40px rgba(0, 200, 83, 0.2);
    border-color: #00C853;
  }
`;

const ProjectImage = styled.div`
  width: 100%;
  height: 200px;
  background: linear-gradient(135deg, #00C853, #00E676);
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
  color: rgb(0, 0, 0);
  margin-bottom: ${theme.spacing.sm};
`;

const ProjectDescription = styled.p`
  color: rgb(100, 100, 100);
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
  background: rgba(0, 200, 83, 0.1);
  border: 1px solid rgba(0, 200, 83, 0.3);
  border-radius: 12px;
  font-size: 0.85rem;
  color: rgb(0, 0, 0);
  font-weight: 600;
`;

const Footer = styled.footer`
  text-align: center;
  padding: ${theme.spacing.xl};
  color: rgb(100, 100, 100);
  border-top: 1px solid rgba(0, 0, 0, 0.1);
`;

const ModalOverlay = styled(motion.div)`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.7);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  padding: ${theme.spacing.md};
`;

const ModalContent = styled(motion.div)`
  background: white;
  border-radius: 12px;
  max-width: 700px;
  width: 100%;
  max-height: 90vh;
  overflow-y: auto;
  position: relative;
`;

const ModalHeader = styled.div`
  position: sticky;
  top: 0;
  background: white;
  padding: ${theme.spacing.lg};
  border-bottom: 1px solid rgba(0, 0, 0, 0.1);
  display: flex;
  justify-content: space-between;
  align-items: center;
  z-index: 1;
`;

const ModalTitle = styled.h3`
  font-size: 1.8rem;
  font-weight: 700;
  color: rgb(0, 0, 0);
  margin: 0;
`;

const CloseButton = styled.button`
  background: none;
  border: none;
  font-size: 1.5rem;
  cursor: pointer;
  color: rgb(100, 100, 100);
  width: 36px;
  height: 36px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  transition: all 0.2s ease;

  &:hover {
    background: rgba(0, 0, 0, 0.05);
    color: rgb(0, 0, 0);
  }
`;

const ModalBody = styled.div`
  padding: ${theme.spacing.lg};
`;

const ModalImage = styled.div`
  width: 100%;
  height: 250px;
  background: linear-gradient(135deg, #00C853, #00E676);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 5rem;
  color: white;
  border-radius: 8px;
  margin-bottom: ${theme.spacing.lg};
`;

const ModalDescription = styled.p`
  font-size: 1.1rem;
  color: rgb(100, 100, 100);
  line-height: 1.8;
  margin-bottom: ${theme.spacing.lg};
`;

const ModalTechStack = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: ${theme.spacing.sm};
  margin-bottom: ${theme.spacing.lg};
`;

const ModalSection = styled.div`
  margin-bottom: ${theme.spacing.lg};
`;

const ModalSectionTitle = styled.h4`
  font-size: 1.2rem;
  font-weight: 600;
  color: rgb(0, 0, 0);
  margin-bottom: ${theme.spacing.sm};
`;

export const PortfolioClean = () => {
  const { mode, toggleTheme } = useTheme();
  const [selectedProject, setSelectedProject] = useState<any>(null);

  const placeholderProjects = [
    {
      id: 1,
      emoji: '📊',
      title: 'Sales Forecasting Dashboard',
      description: 'Machine learning model to predict sales trends using time-series analysis. Built interactive dashboard for stakeholders.',
      tech: ['Python', 'TensorFlow', 'Pandas', 'Plotly'],
      fullDescription: 'Developed a comprehensive sales forecasting system using advanced machine learning techniques. The model analyzes historical sales data, seasonal patterns, and market trends to provide accurate predictions for the next quarter. The interactive dashboard allows stakeholders to visualize trends, explore different scenarios, and make data-driven decisions.'
    },
    {
      id: 2,
      emoji: '🤖',
      title: 'Customer Churn Prediction',
      description: 'Predictive model to identify customers likely to churn. Improved retention rate by 23% through targeted interventions.',
      tech: ['Python', 'Scikit-learn', 'XGBoost', 'SQL'],
      fullDescription: 'Built a machine learning model to predict customer churn with 89% accuracy. The system analyzes customer behavior patterns, transaction history, and engagement metrics to identify at-risk customers. This enabled the company to implement targeted retention strategies, resulting in a 23% improvement in retention rate and significant cost savings.'
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
                onClick={() => setSelectedProject(project)}
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

      <AnimatePresence>
        {selectedProject && (
          <ModalOverlay
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setSelectedProject(null)}
          >
            <ModalContent
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
            >
              <ModalHeader>
                <ModalTitle>{selectedProject.title}</ModalTitle>
                <CloseButton onClick={() => setSelectedProject(null)}>×</CloseButton>
              </ModalHeader>
              <ModalBody>
                <ModalImage>{selectedProject.emoji}</ModalImage>
                <ModalSection>
                  <ModalSectionTitle>Overview</ModalSectionTitle>
                  <ModalDescription>{selectedProject.fullDescription}</ModalDescription>
                </ModalSection>
                <ModalSection>
                  <ModalSectionTitle>Technologies Used</ModalSectionTitle>
                  <ModalTechStack>
                    {selectedProject.tech.map((tech: string, idx: number) => (
                      <TechTag key={idx}>{tech}</TechTag>
                    ))}
                  </ModalTechStack>
                </ModalSection>
              </ModalBody>
            </ModalContent>
          </ModalOverlay>
        )}
      </AnimatePresence>
    </Container>
  );
};
