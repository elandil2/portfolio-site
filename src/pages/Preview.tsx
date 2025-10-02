import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import styled from 'styled-components';
import ReactPlayer from 'react-player';
import { db } from '../db/database';
import { Project } from '../types';
import { lightTheme as theme } from '../styles/theme';

const Container = styled.div`
  min-height: 100vh;
  padding: ${theme.spacing.xl};

  @media (max-width: 768px) {
    padding: ${theme.spacing.md};
  }
`;

const Hero = styled.div`
  display: flex;
  align-items: center;
  gap: ${theme.spacing.xxl};
  max-width: 1200px;
  margin: 0 auto ${theme.spacing.xxl};
  padding: ${theme.spacing.xxl} 0;

  @media (max-width: 968px) {
    flex-direction: column;
    text-align: center;
    gap: ${theme.spacing.xl};
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
  border: 8px solid white;
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
  gap: ${theme.spacing.xs};
  padding: ${theme.spacing.xs} ${theme.spacing.md};
  background: linear-gradient(135deg, rgba(3, 239, 98, 0.1), rgba(0, 212, 255, 0.1));
  border: 2px solid ${theme.colors.secondary};
  border-radius: 50px;
  font-size: 0.95rem;
  font-weight: 600;
  color: ${theme.colors.primary};
  margin-bottom: ${theme.spacing.md};

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
  color: ${theme.colors.primary};
  margin-bottom: ${theme.spacing.sm};
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
  margin-bottom: ${theme.spacing.lg};

  @media (max-width: 968px) {
    font-size: 1.75rem;
  }
`;

const Bio = styled(motion.p)`
  font-size: 1.15rem;
  color: ${theme.colors.textSecondary};
  line-height: 1.8;
  margin-bottom: ${theme.spacing.lg};
  max-width: 600px;

  @media (max-width: 968px) {
    margin: 0 auto ${theme.spacing.lg};
  }
`;

const CTAButtons = styled(motion.div)`
  display: flex;
  gap: ${theme.spacing.sm};
  flex-wrap: wrap;

  @media (max-width: 968px) {
    justify-content: center;
  }
`;

const CTAButton = styled(motion.a)`
  display: inline-flex;
  align-items: center;
  gap: ${theme.spacing.xs};
  padding: ${theme.spacing.sm} ${theme.spacing.lg};
  background: ${props => props.theme === 'primary'
    ? 'linear-gradient(135deg, #03EF62, #00D4FF)'
    : 'white'};
  color: ${props => props.theme === 'primary' ? 'white' : theme.colors.primary};
  border: ${props => props.theme === 'primary' ? 'none' : `2px solid ${theme.colors.border}`};
  border-radius: 12px;
  font-size: 1rem;
  font-weight: 600;
  text-decoration: none;
  cursor: pointer;
  box-shadow: 0 4px 12px rgba(5, 25, 45, 0.1);
  transition: all ${theme.transitions.normal};

  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(5, 25, 45, 0.15);
  }
`;

const SectionTitle = styled(motion.h2)`
  font-size: 2.5rem;
  font-weight: 700;
  color: ${theme.colors.primary};
  text-align: center;
  margin-bottom: ${theme.spacing.xl};
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
  gap: ${theme.spacing.xl};
  max-width: 1400px;
  margin: 0 auto;

  @media (max-width: 768px) {
    grid-template-columns: 1fr;
  }
`;

const ProjectCard = styled(motion.div)`
  background: white;
  border-radius: ${theme.borderRadius.lg};
  overflow: hidden;
  box-shadow: 0 4px 24px rgba(5, 25, 45, 0.08);
  transition: all ${theme.transitions.normal};

  &:hover {
    transform: translateY(-8px);
    box-shadow: 0 12px 48px rgba(5, 25, 45, 0.12);
  }
`;

const MediaContainer = styled.div`
  width: 100%;
  height: 240px;
  background: ${theme.colors.surface};
  position: relative;
  overflow: hidden;

  img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    transition: transform ${theme.transitions.slow};
  }

  &:hover img {
    transform: scale(1.05);
  }
`;

const CardContent = styled.div`
  padding: ${theme.spacing.lg};
`;

const ProjectTitle = styled.h3`
  font-size: 1.5rem;
  font-weight: 700;
  color: ${theme.colors.primary};
  margin-bottom: ${theme.spacing.sm};
`;

const Description = styled.div`
  color: ${theme.colors.textSecondary};
  line-height: 1.7;
  margin-bottom: ${theme.spacing.md};
  font-size: 0.95rem;

  p {
    margin: ${theme.spacing.xs} 0;
  }
`;

const TechStack = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: ${theme.spacing.xs};
  margin-bottom: ${theme.spacing.md};
`;

const TechTag = styled.span`
  padding: 6px 14px;
  background: linear-gradient(135deg, rgba(3, 239, 98, 0.1), rgba(0, 212, 255, 0.1));
  border: 1px solid rgba(3, 239, 98, 0.3);
  border-radius: 20px;
  font-size: 0.85rem;
  color: ${theme.colors.primary};
  font-weight: 600;
`;

const Links = styled.div`
  display: flex;
  gap: ${theme.spacing.sm};
  flex-wrap: wrap;
`;

const Link = styled(motion.a)`
  display: inline-flex;
  align-items: center;
  gap: ${theme.spacing.xs};
  padding: ${theme.spacing.xs} ${theme.spacing.md};
  background: linear-gradient(135deg, #03EF62, #00D4FF);
  color: white;
  text-decoration: none;
  border-radius: 8px;
  font-weight: 600;
  font-size: 0.9rem;
  transition: all ${theme.transitions.fast};

  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 16px rgba(3, 239, 98, 0.3);
  }
`;

const EmptyState = styled.div`
  text-align: center;
  padding: ${theme.spacing.xxl};
  color: ${theme.colors.textSecondary};

  h3 {
    font-size: 1.5rem;
    margin-bottom: ${theme.spacing.sm};
    color: ${theme.colors.primary};
  }
`;

export const Preview = () => {
  const [projects, setProjects] = useState<Project[]>([]);

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
            <img src="/api/placeholder/280/280" alt="Safi Cengiz" />
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
              theme="primary"
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
    </Container>
  );
};
