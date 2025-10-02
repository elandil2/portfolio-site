import { useState, useEffect } from 'react';
import { Reorder } from 'framer-motion';
import styled from 'styled-components';
import { db } from '../db/database';
import { Project } from '../types';
import { ProjectCard } from '../components/ProjectCard';
import { ProjectModal } from '../components/ProjectModal';
import { Button } from '../components/Button';
import { lightTheme as theme } from '../styles/theme';

const Container = styled.div`
  padding: ${theme.spacing.xl};
  max-width: 1400px;
  margin: 0 auto;

  @media (max-width: 768px) {
    padding: ${theme.spacing.md};
  }
`;

const Header = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: ${theme.spacing.xl};
  flex-wrap: wrap;
  gap: ${theme.spacing.md};

  @media (max-width: 768px) {
    flex-direction: column;
    align-items: stretch;
  }
`;

const Title = styled.h1`
  font-size: 2.5rem;
  font-weight: 700;
  background: linear-gradient(135deg, ${theme.colors.primary}, ${theme.colors.secondary});
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin: 0;

  @media (max-width: 768px) {
    font-size: 2rem;
  }
`;

const Actions = styled.div`
  display: flex;
  gap: ${theme.spacing.sm};
  align-items: center;

  @media (max-width: 768px) {
    flex-wrap: wrap;
  }
`;

const SearchBar = styled.input`
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: ${theme.borderRadius.sm};
  padding: ${theme.spacing.sm} ${theme.spacing.md};
  color: ${theme.colors.text};
  font-size: 1rem;
  min-width: 250px;
  transition: all ${theme.transitions.fast};

  &:focus {
    outline: none;
    border-color: ${theme.colors.primary};
    background: rgba(255, 255, 255, 0.08);
  }

  &::placeholder {
    color: ${theme.colors.textSecondary};
  }

  @media (max-width: 768px) {
    min-width: 100%;
  }
`;

const BulkActions = styled.div`
  display: flex;
  gap: ${theme.spacing.sm};
  align-items: center;
  padding: ${theme.spacing.md};
  background: rgba(138, 43, 226, 0.1);
  border: 1px solid ${theme.colors.primary};
  border-radius: ${theme.borderRadius.md};
  margin-bottom: ${theme.spacing.lg};

  span {
    color: ${theme.colors.text};
    margin-right: ${theme.spacing.sm};
  }
`;

const ProjectsGrid = styled(Reorder.Group)`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
  gap: ${theme.spacing.lg};
  list-style: none;
  padding: 0;
  margin: 0;

  @media (max-width: 768px) {
    grid-template-columns: 1fr;
  }
`;

const EmptyState = styled.div`
  text-align: center;
  padding: ${theme.spacing.xxl};
  color: ${theme.colors.textSecondary};

  h3 {
    font-size: 1.5rem;
    margin-bottom: ${theme.spacing.sm};
    color: ${theme.colors.text};
  }

  p {
    margin-bottom: ${theme.spacing.lg};
  }
`;

export const ProjectsManager = () => {
  const [projects, setProjects] = useState<Project[]>([]);
  const [filteredProjects, setFilteredProjects] = useState<Project[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedProjects, setSelectedProjects] = useState<string[]>([]);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [editingProject, setEditingProject] = useState<Project | null>(null);

  useEffect(() => {
    loadProjects();
  }, []);

  useEffect(() => {
    const filtered = projects.filter(
      (project) =>
        project.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
        project.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
        project.techStack.some((tech) =>
          tech.toLowerCase().includes(searchTerm.toLowerCase())
        )
    );
    setFilteredProjects(filtered);
  }, [searchTerm, projects]);

  const loadProjects = async () => {
    const allProjects = await db.projects.orderBy('order').toArray();
    setProjects(allProjects);
  };

  const handleSave = async (data: Partial<Project>) => {
    if (editingProject) {
      await db.projects.update(editingProject.id, {
        ...data,
        updatedAt: new Date(),
      });
    } else {
      const maxOrder = projects.reduce((max, p) => Math.max(max, p.order), 0);
      await db.projects.add({
        id: crypto.randomUUID(),
        ...data,
        order: maxOrder + 1,
        createdAt: new Date(),
        updatedAt: new Date(),
      } as Project);
    }
    loadProjects();
    setEditingProject(null);
  };

  const handleDelete = async (id: string) => {
    if (confirm('Are you sure you want to delete this project?')) {
      await db.projects.delete(id);
      loadProjects();
    }
  };

  const handleReorder = async (newOrder: Project[]) => {
    setProjects(newOrder);
    await Promise.all(
      newOrder.map((project, index) =>
        db.projects.update(project.id, { order: index })
      )
    );
  };

  const handleEdit = (project: Project) => {
    setEditingProject(project);
    setIsModalOpen(true);
  };

  const handleSelect = (id: string) => {
    setSelectedProjects((prev) =>
      prev.includes(id) ? prev.filter((pId) => pId !== id) : [...prev, id]
    );
  };

  const handleBulkDelete = async () => {
    if (confirm(`Delete ${selectedProjects.length} projects?`)) {
      await Promise.all(selectedProjects.map((id) => db.projects.delete(id)));
      setSelectedProjects([]);
      loadProjects();
    }
  };

  const handleBulkPublish = async () => {
    await Promise.all(
      selectedProjects.map((id) =>
        db.projects.update(id, { status: 'published' })
      )
    );
    setSelectedProjects([]);
    loadProjects();
  };

  const handleBulkHide = async () => {
    await Promise.all(
      selectedProjects.map((id) => db.projects.update(id, { status: 'draft' }))
    );
    setSelectedProjects([]);
    loadProjects();
  };

  return (
    <Container>
      <Header>
        <Title>Projects</Title>
        <Actions>
          <SearchBar
            type="text"
            placeholder="Search projects..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
          <Button
            onClick={() => {
              setEditingProject(null);
              setIsModalOpen(true);
            }}
          >
            + Add Project
          </Button>
        </Actions>
      </Header>

      {selectedProjects.length > 0 && (
        <BulkActions>
          <span>{selectedProjects.length} selected</span>
          <Button variant="secondary" onClick={handleBulkPublish}>
            Publish
          </Button>
          <Button variant="secondary" onClick={handleBulkHide}>
            Hide
          </Button>
          <Button variant="danger" onClick={handleBulkDelete}>
            Delete
          </Button>
        </BulkActions>
      )}

      {filteredProjects.length === 0 ? (
        <EmptyState>
          <h3>No projects yet</h3>
          <p>Create your first project to get started</p>
          <Button
            onClick={() => {
              setEditingProject(null);
              setIsModalOpen(true);
            }}
          >
            + Add Your First Project
          </Button>
        </EmptyState>
      ) : (
        <ProjectsGrid axis="y" values={filteredProjects} onReorder={handleReorder as any}>
          {filteredProjects.map((project) => (
            <ProjectCard
              key={project.id}
              project={project}
              onEdit={handleEdit}
              onDelete={handleDelete}
              isSelected={selectedProjects.includes(project.id)}
              onSelect={handleSelect}
            />
          ))}
        </ProjectsGrid>
      )}

      <ProjectModal
        isOpen={isModalOpen}
        onClose={() => {
          setIsModalOpen(false);
          setEditingProject(null);
        }}
        onSave={handleSave}
        project={editingProject}
      />
    </Container>
  );
};
