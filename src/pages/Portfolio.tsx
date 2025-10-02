import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import styled from 'styled-components';
import { db } from '../db/database';
import { Project } from '../types';
import { AnimatedBackground } from '../components/AnimatedBackground';
import { useTheme } from '../context/ThemeContext';
import { getAvatarUrl } from '../utils/avatar';

const Container = styled.div`
  min-height: 100vh;
  position: relative;
`;

const Header = styled.header`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  padding: ${props => props.theme.spacing.md} ${props => props.theme.spacing.xl};
  background: ${props => props.theme.colors.background}95;
  backdrop-filter: blur(10px);
  border-bottom: 1px solid ${props => props.theme.colors.border};
  z-index: 100;
  display: flex;
  justify-content: space-between;
  align-items: center;

  @media (max-width: 768px) {
    padding: ${props => props.theme.spacing.sm} ${props => props.theme.spacing.md};
  }
`;

const Logo = styled.div`
  font-size: 1.3rem;
  font-weight: 700;
  color: ${props => props.theme.colors.text};
`;

const ThemeToggle = styled(motion.button)`
  background: ${props => props.theme.colors.surface};
  border: 1px solid ${props => props.theme.colors.border};
  border-radius: 50px;
  padding: ${props => props.theme.spacing.xs} ${props => props.theme.spacing.sm};
  cursor: pointer;
  font-size: 1.2rem;
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.xs};
  transition: all ${props => props.theme.transitions.fast};

  &:hover {
    background: ${props => props.theme.colors.surfaceHover};
  }
`;

const Main = styled.main`
  padding: ${props => props.theme.spacing.xxl} ${props => props.theme.spacing.xl};
  padding-top: calc(${props => props.theme.spacing.xxl} + 80px);

  @media (max-width: 768px) {
    padding: ${props => props.theme.spacing.lg} ${props => props.theme.spacing.md};
    padding-top: calc(${props => props.theme.spacing.lg} + 80px);
  }
`;

const Hero = styled.div`
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.xxl};
  max-width: 1200px;
  margin: 0 auto ${props => props.theme.spacing.xxl};
  padding: ${props => props.theme.spacing.xxl} 0;

  @media (max-width: 968px) {
    flex-direction: column;
    text-align: center;
    gap: ${props => props.theme.spacing.xl};
  }
`;

const ProfileSection = styled(motion.div)`
  flex: 0 0 auto;
`;

const ProfileImage = styled(motion.div)`
  width: 280px;
  height: 280px;
  border-radius: 50%;
  overflow: hidden;
  border: 8px solid ${props => props.theme.colors.background};
  box-shadow: 0 20px 60px rgba(5, 25, 45, 0.15);
  position: relative;

  &::before {
    content: '';
    position: absolute;
    top: -4px;
    left: -4px;
    right: -4px;
    bottom: -4px;
    border-radius: 50%;
    background: linear-gradient(135deg, #03EF62, #00D4FF);
    z-index: -1;
  }

  img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }

  @media (max-width: 968px) {
    width: 220px;
    height: 220px;
  }
`;

const ContentSection = styled(motion.div)`
  flex: 1;
`;

const Greeting = styled(motion.div)`
  display: inline-flex;
  align-items: center;
  gap: ${props => props.theme.spacing.xs};
  padding: ${props => props.theme.spacing.xs} ${props => props.theme.spacing.md};
  background: linear-gradient(135deg, rgba(3, 239, 98, 0.1), rgba(0, 212, 255, 0.1));
  border: 2px solid ${props => props.theme.colors.secondary};
  border-radius: 50px;
  font-size: 0.95rem;
  font-weight: 600;
  color: ${props => props.theme.colors.text};
  margin-bottom: ${props => props.theme.spacing.md};

  span {
    animation: wave 2s ease-in-out infinite;
  }

  @keyframes wave {
    0%, 100% { transform: rotate(0deg); }
    25% { transform: rotate(20deg); }
    75% { transform: rotate(-15deg); }
  }
`;

const Name = styled(motion.h1)`
  font-size: 4rem;
  font-weight: 800;
  color: ${props => props.theme.colors.text};
  margin-bottom: ${props => props.theme.spacing.sm};
  line-height: 1.1;

  @media (max-width: 968px) {
    font-size: 3rem;
  }

  @media (max-width: 480px) {
    font-size: 2.5rem;
  }
`;

const JobTitle = styled(motion.h2)`
  font-size: 2rem;
  font-weight: 600;
  background: linear-gradient(135deg, #03EF62, #00D4FF);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: ${props => props.theme.spacing.lg};

  @media (max-width: 968px) {
    font-size: 1.75rem;
  }
`;

const Bio = styled(motion.p)`
  font-size: 1.15rem;
  color: ${props => props.theme.colors.textSecondary};
  line-height: 1.8;
  margin-bottom: ${props => props.theme.spacing.lg};
  max-width: 600px;

  @media (max-width: 968px) {
    margin: 0 auto ${props => props.theme.spacing.lg};
  }
`;

const CTAButtons = styled(motion.div)`
  display: flex;
  gap: ${props => props.theme.spacing.sm};
  flex-wrap: wrap;

  @media (max-width: 968px) {
    justify-content: center;
  }
`;

const CTAButton = styled(motion.a)<{ $primary?: boolean }>`
  display: inline-flex;
  align-items: center;
  gap: ${props => props.theme.spacing.xs};
  padding: ${props => props.theme.spacing.sm} ${props => props.theme.spacing.lg};
  background: ${props => props.$primary
    ? 'linear-gradient(135deg, #03EF62, #00D4FF)'
    : props.theme.colors.background};
  color: ${props => props.$primary ? 'white' : props.theme.colors.text};
  border: ${props => props.$primary ? 'none' : `2px solid ${props.theme.colors.border}`};
  border-radius: 12px;
  font-size: 1rem;
  font-weight: 600;
  text-decoration: none;
  cursor: pointer;
  box-shadow: 0 4px 12px rgba(5, 25, 45, 0.1);
  transition: all ${props => props.theme.transitions.normal};

  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(5, 25, 45, 0.15);
  }
`;

const SectionTitle = styled(motion.h2)`
  font-size: 2.5rem;
  font-weight: 700;
  color: ${props => props.theme.colors.text};
  text-align: center;
  margin-bottom: ${props => props.theme.spacing.xl};
  position: relative;

  &::after {
    content: '';
    position: absolute;
    bottom: -12px;
    left: 50%;
    transform: translateX(-50%);
    width: 80px;
    height: 4px;
    background: linear-gradient(135deg, #03EF62, #00D4FF);
    border-radius: 2px;
  }
`;

const ProjectsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
  gap: ${props => props.theme.spacing.xl};
  max-width: 1400px;
  margin: 0 auto;

  @media (max-width: 768px) {
    grid-template-columns: 1fr;
  }
`;

const ProjectCard = styled(motion.div)`
  background: ${props => props.theme.colors.background};
  border: 1px solid ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.borderRadius.lg};
  overflow: hidden;
  box-shadow: 0 4px 24px rgba(5, 25, 45, 0.08);
  transition: all ${props => props.theme.transitions.normal};

  &:hover {
    transform: translateY(-8px);
    box-shadow: 0 12px 48px rgba(5, 25, 45, 0.12);
  }
`;

const MediaContainer = styled.div`
  width: 100%;
  height: 240px;
  background: ${props => props.theme.colors.surface};
  position: relative;
  overflow: hidden;

  img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    transition: transform ${props => props.theme.transitions.slow};
  }

  &:hover img {
    transform: scale(1.05);
  }
`;

const CardContent = styled.div`
  padding: ${props => props.theme.spacing.lg};
`;

const ProjectTitle = styled.h3`
  font-size: 1.5rem;
  font-weight: 700;
  color: ${props => props.theme.colors.text};
  margin-bottom: ${props => props.theme.spacing.sm};
`;

const Description = styled.div`
  color: ${props => props.theme.colors.textSecondary};
  line-height: 1.7;
  margin-bottom: ${props => props.theme.spacing.md};
  font-size: 0.95rem;

  p {
    margin: ${props => props.theme.spacing.xs} 0;
  }
`;

const TechStack = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: ${props => props.theme.spacing.xs};
  margin-bottom: ${props => props.theme.spacing.md};
`;

const TechTag = styled.span`
  padding: 6px 14px;
  background: linear-gradient(135deg, rgba(3, 239, 98, 0.1), rgba(0, 212, 255, 0.1));
  border: 1px solid rgba(3, 239, 98, 0.3);
  border-radius: 20px;
  font-size: 0.85rem;
  color: ${props => props.theme.colors.text};
  font-weight: 600;
`;

const Links = styled.div`
  display: flex;
  gap: ${props => props.theme.spacing.sm};
  flex-wrap: wrap;
`;

const Link = styled(motion.a)`
  display: inline-flex;
  align-items: center;
  gap: ${props => props.theme.spacing.xs};
  padding: ${props => props.theme.spacing.xs} ${props => props.theme.spacing.md};
  background: linear-gradient(135deg, #03EF62, #00D4FF);
  color: white;
  text-decoration: none;
  border-radius: 8px;
  font-weight: 600;
  font-size: 0.9rem;
  transition: all ${props => props.theme.transitions.fast};

  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 16px rgba(3, 239, 98, 0.3);
  }
`;

const EmptyState = styled.div`
  text-align: center;
  padding: ${props => props.theme.spacing.xxl};
  color: ${props => props.theme.colors.textSecondary};

  h3 {
    font-size: 1.5rem;
    margin-bottom: ${props => props.theme.spacing.sm};
    color: ${props => props.theme.colors.text};
  }
`;

const SocialLinks = styled.div`
  display: flex;
  gap: ${props => props.theme.spacing.sm};
  align-items: center;
`;

const SocialLink = styled(motion.a)`
  display: flex;
  align-items: center;
  justify-content: center;
  width: 40px;
  height: 40px;
  background: ${props => props.theme.colors.surface};
  border: 1px solid ${props => props.theme.colors.border};
  border-radius: 50%;
  color: ${props => props.theme.colors.text};
  text-decoration: none;
  font-size: 1.2rem;
  transition: all ${props => props.theme.transitions.fast};

  &:hover {
    background: linear-gradient(135deg, #03EF62, #00D4FF);
    color: white;
    border-color: transparent;
    transform: translateY(-2px);
  }
`;

export const Portfolio = () => {
  const [projects, setProjects] = useState<Project[]>([]);
  const { mode, toggleTheme } = useTheme();

  useEffect(() => {
    loadProjects();
  }, []);

  const loadProjects = async () => {
    const published = await db.projects
      .where('status')
      .equals('published')
      .sortBy('order');
    setProjects(published);
  };

  return (
    <Container>
      <AnimatedBackground />

      <Header>
        <Logo>Safi Cengiz</Logo>
        <SocialLinks>
          <SocialLink
            href="https://www.linkedin.com/in/safi-cengiz/"
            target="_blank"
            rel="noopener noreferrer"
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.95 }}
            title="LinkedIn"
          >
            💼
          </SocialLink>
          <SocialLink
            href="https://github.com/elandil2"
            target="_blank"
            rel="noopener noreferrer"
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.95 }}
            title="GitHub"
          >
            🔗
          </SocialLink>
          <ThemeToggle
            onClick={toggleTheme}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            {mode === 'light' ? '🌙' : '☀️'}
          </ThemeToggle>
        </SocialLinks>
      </Header>

      <Main>
        <Hero>
          <ProfileSection
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6 }}
          >
            <ProfileImage
              whileHover={{ scale: 1.05 }}
              transition={{ type: 'spring', stiffness: 300 }}
            >
              <img src={getAvatarUrl()} alt="Safi Cengiz" />
            </ProfileImage>
          </ProfileSection>

          <ContentSection
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            <Greeting
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
            >
              <span>👋</span> Hello, I'm
            </Greeting>

            <Name
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
            >
              Safi Cengiz
            </Name>

            <JobTitle
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
            >
              Data Scientist
            </JobTitle>

            <Bio
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.7 }}
            >
              Passionate about transforming data into actionable insights. Specializing in machine learning,
              statistical analysis, and creating impactful data-driven solutions.
            </Bio>

            <CTAButtons
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.8 }}
            >
              <CTAButton
                $primary
                href="#projects"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                View My Work →
              </CTAButton>
              <CTAButton
                href="mailto:safi@example.com"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                📧 Get In Touch
              </CTAButton>
            </CTAButtons>
          </ContentSection>
        </Hero>

        <div id="projects">
          <SectionTitle
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
          >
            Featured Projects
          </SectionTitle>

          {projects.length === 0 ? (
            <EmptyState>
              <h3>No published projects yet</h3>
              <p>Check back soon for amazing data science projects!</p>
            </EmptyState>
          ) : (
            <ProjectsGrid>
              {projects.map((project, index) => (
                <ProjectCard
                  key={project.id}
                  initial={{ opacity: 0, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                >
                  {project.images && project.images.length > 0 && (
                    <MediaContainer>
                      <img src={project.images[0]} alt={project.title} />
                    </MediaContainer>
                  )}

                  <CardContent>
                    <ProjectTitle>{project.title}</ProjectTitle>

                    <Description dangerouslySetInnerHTML={{ __html: project.description }} />

                    {project.techStack && project.techStack.length > 0 && (
                      <TechStack>
                        {project.techStack.map((tech, idx) => (
                          <TechTag key={idx}>{tech}</TechTag>
                        ))}
                      </TechStack>
                    )}

                    <Links>
                      {project.githubUrl && (
                        <Link
                          href={project.githubUrl}
                          target="_blank"
                          rel="noopener noreferrer"
                          whileHover={{ scale: 1.05 }}
                          whileTap={{ scale: 0.95 }}
                        >
                          <span>🔗</span> GitHub
                        </Link>
                      )}
                      {project.liveDemoUrl && (
                        <Link
                          href={project.liveDemoUrl}
                          target="_blank"
                          rel="noopener noreferrer"
                          whileHover={{ scale: 1.05 }}
                          whileTap={{ scale: 0.95 }}
                        >
                          <span>🚀</span> Live Demo
                        </Link>
                      )}
                    </Links>
                  </CardContent>
                </ProjectCard>
              ))}
            </ProjectsGrid>
          )}
        </div>
      </Main>
    </Container>
  );
};
