import { useState } from 'react';
import { motion, Reorder, useDragControls } from 'framer-motion';
import styled from 'styled-components';
import { Project } from '../types';
import { lightTheme as theme } from '../styles/theme';

interface ProjectCardProps {
  project: Project;
  onEdit: (project: Project) => void;
  onDelete: (id: string) => void;
  isSelected: boolean;
  onSelect: (id: string) => void;
}

const Card = styled(motion.div)<{ $isSelected: boolean }>`
  background: ${theme.glassmorphism.background};
  backdrop-filter: ${theme.glassmorphism.backdropFilter};
  border: ${props => props.$isSelected ? `2px solid ${theme.colors.primary}` : theme.glassmorphism.border};
  border-radius: ${theme.borderRadius.md};
  padding: ${theme.spacing.md};
  cursor: grab;
  position: relative;
  transition: all ${theme.transitions.normal};

  &:active {
    cursor: grabbing;
  }

  &:hover {
    box-shadow: 0 8px 32px rgba(138, 43, 226, 0.2);
    border-color: ${theme.colors.primary};
  }
`;

const CardHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: ${theme.spacing.sm};
`;

const Title = styled.h3`
  font-size: 1.25rem;
  font-weight: 600;
  color: ${theme.colors.text};
  margin: 0;
  flex: 1;
`;

const StatusBadge = styled.span<{ $status: string }>`
  padding: 4px 12px;
  border-radius: ${theme.borderRadius.sm};
  font-size: 0.75rem;
  font-weight: 600;
  background: ${props => props.$status === 'published'
    ? `linear-gradient(135deg, ${theme.colors.success}, #00cc77)`
    : `linear-gradient(135deg, ${theme.colors.warning}, #cc9900)`};
  color: white;
  text-transform: uppercase;
`;

const Description = styled.p`
  color: ${theme.colors.textSecondary};
  font-size: 0.9rem;
  margin: ${theme.spacing.sm} 0;
  line-height: 1.5;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
`;

const TechStack = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: ${theme.spacing.xs};
  margin: ${theme.spacing.sm} 0;
`;

const TechTag = styled.span`
  padding: 4px 10px;
  background: rgba(138, 43, 226, 0.2);
  border: 1px solid rgba(138, 43, 226, 0.3);
  border-radius: ${theme.borderRadius.sm};
  font-size: 0.75rem;
  color: ${theme.colors.secondary};
`;

const Actions = styled.div`
  display: flex;
  gap: ${theme.spacing.xs};
  margin-top: ${theme.spacing.md};
`;

const ActionButton = styled(motion.button)`
  background: transparent;
  border: 1px solid rgba(255, 255, 255, 0.2);
  color: ${theme.colors.text};
  padding: 6px 14px;
  border-radius: ${theme.borderRadius.sm};
  font-size: 0.85rem;
  cursor: pointer;
  transition: all ${theme.transitions.fast};

  &:hover {
    border-color: ${theme.colors.primary};
    background: rgba(138, 43, 226, 0.1);
  }
`;

const Checkbox = styled.input`
  position: absolute;
  top: ${theme.spacing.sm};
  left: ${theme.spacing.sm};
  width: 20px;
  height: 20px;
  cursor: pointer;
  accent-color: ${theme.colors.primary};
`;

const DragHandle = styled.div`
  position: absolute;
  top: ${theme.spacing.sm};
  right: ${theme.spacing.sm};
  cursor: grab;
  color: ${theme.colors.textSecondary};
  font-size: 1.2rem;

  &:active {
    cursor: grabbing;
  }
`;

export const ProjectCard = ({ project, onEdit, onDelete, isSelected, onSelect }: ProjectCardProps) => {
  const controls = useDragControls();

  return (
    <Reorder.Item
      value={project}
      dragListener={false}
      dragControls={controls}
      whileHover={{ scale: 1.02 }}
      whileDrag={{ scale: 1.05, zIndex: 10 }}
    >
      <Card $isSelected={isSelected}>
        <Checkbox
          type="checkbox"
          checked={isSelected}
          onChange={() => onSelect(project.id)}
        />

        <DragHandle onPointerDown={(e) => controls.start(e)}>
          ⋮⋮
        </DragHandle>

        <CardHeader>
          <Title>{project.title}</Title>
          <StatusBadge $status={project.status}>{project.status}</StatusBadge>
        </CardHeader>

        <Description>{project.description}</Description>

        {project.techStack.length > 0 && (
          <TechStack>
            {project.techStack.slice(0, 4).map((tech, idx) => (
              <TechTag key={idx}>{tech}</TechTag>
            ))}
            {project.techStack.length > 4 && (
              <TechTag>+{project.techStack.length - 4} more</TechTag>
            )}
          </TechStack>
        )}

        <Actions>
          <ActionButton
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => onEdit(project)}
          >
            Edit
          </ActionButton>
          <ActionButton
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => onDelete(project.id)}
            style={{ borderColor: theme.colors.error, color: theme.colors.error }}
          >
            Delete
          </ActionButton>
        </Actions>
      </Card>
    </Reorder.Item>
  );
};
