import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useForm } from 'react-hook-form';
import styled from 'styled-components';
import { useDropzone } from 'react-dropzone';
import ReactQuill from 'react-quill';
import 'react-quill/dist/quill.snow.css';
import { Project } from '../types';
import { lightTheme as theme } from '../styles/theme';
import { Button } from './Button';

interface ProjectModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (data: Partial<Project>) => void;
  project?: Project | null;
}

const Overlay = styled(motion.div)`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.7);
  backdrop-filter: blur(8px);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  padding: ${theme.spacing.md};
`;

const ModalContainer = styled(motion.div)`
  background: ${theme.glassmorphism.background};
  backdrop-filter: ${theme.glassmorphism.backdropFilter};
  border: ${theme.glassmorphism.border};
  border-radius: ${theme.borderRadius.xl};
  padding: ${theme.spacing.xl};
  max-width: 800px;
  width: 100%;
  max-height: 90vh;
  overflow-y: auto;
  box-shadow: 0 20px 60px rgba(138, 43, 226, 0.3);

  &::-webkit-scrollbar {
    width: 6px;
  }

  &::-webkit-scrollbar-thumb {
    background: ${theme.colors.primary};
    border-radius: 3px;
  }
`;

const ModalHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: ${theme.spacing.lg};
`;

const Title = styled.h2`
  font-size: 1.75rem;
  font-weight: 700;
  background: linear-gradient(135deg, ${theme.colors.primary}, ${theme.colors.secondary});
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin: 0;
`;

const CloseButton = styled.button`
  background: transparent;
  border: none;
  color: ${theme.colors.textSecondary};
  font-size: 1.5rem;
  cursor: pointer;
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: ${theme.borderRadius.sm};
  transition: all ${theme.transitions.fast};

  &:hover {
    background: rgba(255, 255, 255, 0.1);
    color: ${theme.colors.text};
  }
`;

const Form = styled.form`
  display: flex;
  flex-direction: column;
  gap: ${theme.spacing.md};
`;

const FormGroup = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${theme.spacing.xs};
`;

const Label = styled.label`
  font-size: 0.9rem;
  font-weight: 600;
  color: ${theme.colors.text};
`;

const Input = styled.input`
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: ${theme.borderRadius.sm};
  padding: ${theme.spacing.sm};
  color: ${theme.colors.text};
  font-size: 1rem;
  transition: all ${theme.transitions.fast};

  &:focus {
    outline: none;
    border-color: ${theme.colors.primary};
    background: rgba(255, 255, 255, 0.08);
  }

  &::placeholder {
    color: ${theme.colors.textSecondary};
  }
`;

const DropzoneContainer = styled.div<{ $isDragActive: boolean }>`
  border: 2px dashed ${props => props.$isDragActive ? theme.colors.primary : 'rgba(255, 255, 255, 0.2)'};
  border-radius: ${theme.borderRadius.md};
  padding: ${theme.spacing.lg};
  text-align: center;
  cursor: pointer;
  transition: all ${theme.transitions.fast};
  background: ${props => props.$isDragActive ? 'rgba(138, 43, 226, 0.1)' : 'rgba(255, 255, 255, 0.03)'};

  &:hover {
    border-color: ${theme.colors.primary};
    background: rgba(138, 43, 226, 0.05);
  }
`;

const ImagePreviewGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
  gap: ${theme.spacing.sm};
  margin-top: ${theme.spacing.sm};
`;

const ImagePreview = styled.div`
  position: relative;
  aspect-ratio: 1;
  border-radius: ${theme.borderRadius.sm};
  overflow: hidden;
  border: 1px solid rgba(255, 255, 255, 0.1);

  img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }
`;

const RemoveImage = styled.button`
  position: absolute;
  top: 4px;
  right: 4px;
  background: rgba(0, 0, 0, 0.7);
  border: none;
  color: white;
  width: 24px;
  height: 24px;
  border-radius: 50%;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.9rem;

  &:hover {
    background: ${theme.colors.error};
  }
`;

const TagInput = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: ${theme.spacing.xs};
  padding: ${theme.spacing.xs};
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: ${theme.borderRadius.sm};
  min-height: 42px;

  input {
    background: transparent;
    border: none;
    color: ${theme.colors.text};
    flex: 1;
    min-width: 120px;
    padding: 4px;

    &:focus {
      outline: none;
    }
  }
`;

const Tag = styled.span`
  background: rgba(138, 43, 226, 0.3);
  padding: 4px 10px;
  border-radius: ${theme.borderRadius.sm};
  font-size: 0.85rem;
  display: flex;
  align-items: center;
  gap: 6px;

  button {
    background: transparent;
    border: none;
    color: ${theme.colors.text};
    cursor: pointer;
    padding: 0;
    line-height: 1;

    &:hover {
      color: ${theme.colors.error};
    }
  }
`;

const Toggle = styled.label`
  display: flex;
  align-items: center;
  gap: ${theme.spacing.sm};
  cursor: pointer;

  input {
    display: none;
  }

  span {
    position: relative;
    width: 50px;
    height: 26px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 13px;
    transition: all ${theme.transitions.fast};

    &::after {
      content: '';
      position: absolute;
      width: 20px;
      height: 20px;
      border-radius: 50%;
      background: white;
      top: 3px;
      left: 3px;
      transition: all ${theme.transitions.fast};
    }
  }

  input:checked + span {
    background: ${theme.colors.primary};

    &::after {
      left: 27px;
    }
  }
`;

const Actions = styled.div`
  display: flex;
  gap: ${theme.spacing.sm};
  margin-top: ${theme.spacing.lg};
  justify-content: flex-end;
`;

const QuillWrapper = styled.div`
  .ql-container {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 0 0 ${theme.borderRadius.sm} ${theme.borderRadius.sm};
    color: ${theme.colors.text};
    min-height: 150px;
  }

  .ql-toolbar {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: ${theme.borderRadius.sm} ${theme.borderRadius.sm} 0 0;
  }

  .ql-stroke {
    stroke: ${theme.colors.textSecondary};
  }

  .ql-fill {
    fill: ${theme.colors.textSecondary};
  }

  .ql-picker-label {
    color: ${theme.colors.textSecondary};
  }

  .ql-editor.ql-blank::before {
    color: ${theme.colors.textSecondary};
  }
`;

export const ProjectModal = ({ isOpen, onClose, onSave, project }: ProjectModalProps) => {
  const { register, handleSubmit } = useForm<Partial<Project>>({
    defaultValues: project || {},
  });

  const [description, setDescription] = useState(project?.description || '');
  const [images, setImages] = useState<string[]>(project?.images || []);
  const [techStack, setTechStack] = useState<string[]>(project?.techStack || []);
  const [techInput, setTechInput] = useState('');
  const [status, setStatus] = useState<'published' | 'draft'>(project?.status || 'draft');

  const convertToBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = (error) => reject(error);
    });
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: { 'image/*': [] },
    maxSize: 5242880, // 5MB max file size
    onDrop: async (acceptedFiles) => {
      const base64Images = await Promise.all(
        acceptedFiles.map(file => convertToBase64(file))
      );
      setImages([...images, ...base64Images]);
    },
  });

  const onSubmit = handleSubmit((data) => {
    onSave({
      ...data,
      description,
      images,
      techStack,
      status,
    });
    onClose();
  });

  const addTech = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && techInput.trim()) {
      e.preventDefault();
      setTechStack([...techStack, techInput.trim()]);
      setTechInput('');
    }
  };

  const removeTech = (index: number) => {
    setTechStack(techStack.filter((_, i) => i !== index));
  };

  const removeImage = (index: number) => {
    setImages(images.filter((_, i) => i !== index));
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <Overlay
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
        >
          <ModalContainer
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.9, opacity: 0 }}
            onClick={(e) => e.stopPropagation()}
          >
            <ModalHeader>
              <Title>{project ? 'Edit Project' : 'Add New Project'}</Title>
              <CloseButton onClick={onClose}>×</CloseButton>
            </ModalHeader>

            <Form onSubmit={onSubmit}>
              <FormGroup>
                <Label>Project Title</Label>
                <Input
                  {...register('title', { required: true })}
                  placeholder="Enter project title"
                />
              </FormGroup>

              <FormGroup>
                <Label>Description</Label>
                <QuillWrapper>
                  <ReactQuill
                    value={description}
                    onChange={setDescription}
                    theme="snow"
                    placeholder="Write a detailed description..."
                  />
                </QuillWrapper>
              </FormGroup>

              <FormGroup>
                <Label>Upload Images</Label>
                <DropzoneContainer {...getRootProps()} $isDragActive={isDragActive}>
                  <input {...getInputProps()} />
                  {isDragActive ? (
                    <p>Drop images here...</p>
                  ) : (
                    <p>Drag & drop images here, or click to select</p>
                  )}
                </DropzoneContainer>
                {images.length > 0 && (
                  <ImagePreviewGrid>
                    {images.map((img, idx) => (
                      <ImagePreview key={idx}>
                        <img src={img} alt={`Preview ${idx}`} />
                        <RemoveImage onClick={() => removeImage(idx)}>×</RemoveImage>
                      </ImagePreview>
                    ))}
                  </ImagePreviewGrid>
                )}
              </FormGroup>

              <FormGroup>
                <Label>Video URL (Optional)</Label>
                <Input
                  {...register('videoUrl')}
                  placeholder="YouTube, Vimeo, or direct video URL"
                />
              </FormGroup>

              <FormGroup>
                <Label>GitHub URL</Label>
                <Input
                  {...register('githubUrl')}
                  placeholder="https://github.com/..."
                />
              </FormGroup>

              <FormGroup>
                <Label>Live Demo URL</Label>
                <Input
                  {...register('liveDemoUrl')}
                  placeholder="https://..."
                />
              </FormGroup>

              <FormGroup>
                <Label>Tech Stack</Label>
                <TagInput>
                  {techStack.map((tech, idx) => (
                    <Tag key={idx}>
                      {tech}
                      <button type="button" onClick={() => removeTech(idx)}>×</button>
                    </Tag>
                  ))}
                  <input
                    type="text"
                    value={techInput}
                    onChange={(e) => setTechInput(e.target.value)}
                    onKeyDown={addTech}
                    placeholder="Type and press Enter"
                  />
                </TagInput>
              </FormGroup>

              <FormGroup>
                <Toggle>
                  <span>Publish</span>
                  <input
                    type="checkbox"
                    checked={status === 'published'}
                    onChange={(e) => setStatus(e.target.checked ? 'published' : 'draft')}
                  />
                  <span></span>
                </Toggle>
              </FormGroup>

              <Actions>
                <Button type="button" variant="secondary" onClick={onClose}>
                  Cancel
                </Button>
                <Button type="submit" variant="primary">
                  {project ? 'Update' : 'Create'} Project
                </Button>
              </Actions>
            </Form>
          </ModalContainer>
        </Overlay>
      )}
    </AnimatePresence>
  );
};
