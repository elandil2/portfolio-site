import { motion } from 'framer-motion';
import styled from 'styled-components';
import { useTheme } from '../context/ThemeContext';

const BackgroundContainer = styled.div<{ $isDark: boolean }>`
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  z-index: -1;
  overflow: hidden;
  background: ${props => props.$isDark
    ? 'linear-gradient(135deg, #05192D 0%, #0A2540 50%, #0F2F4F 100%)'
    : 'linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 50%, #DBEAFE 100%)'
  };
  transition: background 0.3s ease;
`;

const FloatingShape = styled(motion.div)<{ $color: string; $size: number }>`
  position: absolute;
  width: ${props => props.$size}px;
  height: ${props => props.$size}px;
  border-radius: 50%;
  background: ${props => props.$color};
  filter: blur(60px);
  opacity: 0.3;
`;

const GridPattern = styled.div<{ $isDark: boolean }>`
  position: absolute;
  width: 100%;
  height: 100%;
  background-image:
    linear-gradient(${props => props.$isDark ? 'rgba(255, 255, 255, 0.02)' : 'rgba(5, 25, 45, 0.02)'} 1px, transparent 1px),
    linear-gradient(90deg, ${props => props.$isDark ? 'rgba(255, 255, 255, 0.02)' : 'rgba(5, 25, 45, 0.02)'} 1px, transparent 1px);
  background-size: 50px 50px;
`;

const Particle = styled(motion.div)`
  position: absolute;
  width: 4px;
  height: 4px;
  background: linear-gradient(135deg, #03EF62, #00D4FF);
  border-radius: 50%;
`;

export const AnimatedBackground = () => {
  const { mode } = useTheme();
  const isDark = mode === 'dark';

  const shapes = [
    { id: 1, size: 400, color: 'rgba(3, 239, 98, 0.4)', x: '10%', y: '20%' },
    { id: 2, size: 350, color: 'rgba(0, 212, 255, 0.35)', x: '70%', y: '60%' },
    { id: 3, size: 300, color: 'rgba(3, 239, 98, 0.3)', x: '40%', y: '80%' },
  ];

  const particles = Array.from({ length: 30 }, (_, i) => ({
    id: i,
    x: Math.random() * 100,
    y: Math.random() * 100,
  }));

  return (
    <BackgroundContainer $isDark={isDark}>
      <GridPattern $isDark={isDark} />

      {shapes.map((shape) => (
        <FloatingShape
          key={shape.id}
          $color={shape.color}
          $size={shape.size}
          style={{ left: shape.x, top: shape.y }}
          animate={{
            x: [0, 30, 0, -30, 0],
            y: [0, -40, -80, -40, 0],
            scale: [1, 1.1, 1, 0.9, 1],
          }}
          transition={{
            duration: 20 + shape.id * 3,
            repeat: Infinity,
            repeatType: 'loop',
            ease: 'easeInOut',
          }}
        />
      ))}

      {particles.map((particle) => (
        <Particle
          key={particle.id}
          initial={{
            x: `${particle.x}vw`,
            y: `${particle.y}vh`,
            scale: 0,
            opacity: 0,
          }}
          animate={{
            x: [`${particle.x}vw`, `${(particle.x + 15) % 100}vw`],
            y: [`${particle.y}vh`, `${(particle.y + 20) % 100}vh`],
            scale: [0, 1, 1, 0],
            opacity: [0, 0.6, 0.6, 0],
          }}
          transition={{
            duration: 10 + Math.random() * 5,
            repeat: Infinity,
            repeatType: 'loop',
            delay: Math.random() * 3,
            ease: 'easeInOut',
          }}
        />
      ))}
    </BackgroundContainer>
  );
};
