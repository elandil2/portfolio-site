import { motion, HTMLMotionProps } from 'framer-motion';
import styled from 'styled-components';
import { lightTheme as theme } from '../styles/theme';

interface GlassCardProps extends HTMLMotionProps<'div'> {
  children: React.ReactNode;
}

const StyledCard = styled(motion.div)`
  background: white;
  border: 1px solid ${theme.colors.border};
  border-radius: ${theme.borderRadius.lg};
  box-shadow: 0 4px 24px rgba(5, 25, 45, 0.08);
  padding: ${theme.spacing.lg};
  transition: all ${theme.transitions.normal};

  &:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 32px rgba(5, 25, 45, 0.12);
  }
`;

export const GlassCard = ({ children, ...props }: GlassCardProps) => {
  return (
    <StyledCard
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
      {...props}
    >
      {children}
    </StyledCard>
  );
};
