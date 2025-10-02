import { motion, HTMLMotionProps } from 'framer-motion';
import styled from 'styled-components';
import { lightTheme as theme } from '../styles/theme';

interface ButtonProps extends Omit<HTMLMotionProps<'button'>, 'children'> {
  variant?: 'primary' | 'secondary' | 'danger';
  children: React.ReactNode;
}

const StyledButton = styled(motion.button)<{ $variant: string }>`
  background: ${props =>
    props.$variant === 'primary'
      ? `linear-gradient(135deg, #03EF62, #00D4FF)`
      : props.$variant === 'danger'
      ? theme.colors.error
      : 'white'};
  color: ${props => (props.$variant === 'secondary' ? theme.colors.primary : 'white')};
  border: ${props => (props.$variant === 'secondary' ? `2px solid ${theme.colors.border}` : 'none')};
  border-radius: ${theme.borderRadius.sm};
  padding: ${theme.spacing.sm} ${theme.spacing.md};
  font-size: 0.95rem;
  font-weight: 700;
  cursor: pointer;
  transition: all ${theme.transitions.normal};
  position: relative;
  overflow: hidden;
  box-shadow: ${props =>
    props.$variant === 'primary' ? '0 4px 12px rgba(3, 239, 98, 0.25)' : '0 2px 8px rgba(5, 25, 45, 0.08)'};

  &::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 0;
    height: 0;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.3);
    transform: translate(-50%, -50%);
    transition: width 0.6s, height 0.6s;
  }

  &:hover::before {
    width: 300px;
    height: 300px;
  }

  &:hover {
    transform: translateY(-2px);
    box-shadow: ${props =>
      props.$variant === 'primary'
        ? '0 8px 24px rgba(3, 239, 98, 0.35)'
        : '0 4px 16px rgba(5, 25, 45, 0.12)'};
  }

  &:active {
    transform: translateY(0);
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  span {
    position: relative;
    z-index: 1;
  }
`;

export const Button = ({ variant = 'primary', children, ...props }: ButtonProps) => {
  return (
    <StyledButton
      $variant={variant}
      whileTap={{ scale: 0.95 }}
      {...props}
    >
      <span>{children}</span>
    </StyledButton>
  );
};
