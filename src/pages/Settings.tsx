import { useState } from 'react';
import { motion } from 'framer-motion';
import styled from 'styled-components';
import { db } from '../db/database';
import { GlassCard } from '../components/GlassCard';
import { Button } from '../components/Button';
import { lightTheme as theme } from '../styles/theme';

const Container = styled.div`
  padding: ${theme.spacing.xl};
  max-width: 1000px;
  margin: 0 auto;

  @media (max-width: 768px) {
    padding: ${theme.spacing.md};
  }
`;

const Title = styled.h1`
  font-size: 2.5rem;
  font-weight: 700;
  background: linear-gradient(135deg, ${theme.colors.primary}, ${theme.colors.secondary});
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: ${theme.spacing.xl};

  @media (max-width: 768px) {
    font-size: 2rem;
  }
`;

const Section = styled(GlassCard)`
  margin-bottom: ${theme.spacing.lg};
`;

const SectionTitle = styled.h2`
  font-size: 1.5rem;
  font-weight: 600;
  color: ${theme.colors.text};
  margin-bottom: ${theme.spacing.md};
`;

const SectionDescription = styled.p`
  color: ${theme.colors.textSecondary};
  margin-bottom: ${theme.spacing.lg};
  line-height: 1.6;
`;

const Actions = styled.div`
  display: flex;
  gap: ${theme.spacing.sm};
  flex-wrap: wrap;
`;

const FileInput = styled.input`
  display: none;
`;

const InfoBox = styled(motion.div)<{ $type: 'success' | 'error' | 'info' }>`
  padding: ${theme.spacing.md};
  border-radius: ${theme.borderRadius.sm};
  margin-top: ${theme.spacing.md};
  background: ${props =>
    props.$type === 'success'
      ? 'rgba(0, 255, 136, 0.1)'
      : props.$type === 'error'
      ? 'rgba(255, 59, 59, 0.1)'
      : 'rgba(0, 217, 255, 0.1)'};
  border: 1px solid ${props =>
    props.$type === 'success'
      ? theme.colors.success
      : props.$type === 'error'
      ? theme.colors.error
      : theme.colors.secondary};
  color: ${theme.colors.text};
`;

const ThemeOption = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: ${theme.spacing.md};
  margin-top: ${theme.spacing.md};
`;

const ColorPicker = styled.div`
  display: flex;
  align-items: center;
  gap: ${theme.spacing.sm};
  padding: ${theme.spacing.sm};
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: ${theme.borderRadius.sm};

  label {
    flex: 1;
    color: ${theme.colors.text};
    font-size: 0.9rem;
  }

  input[type='color'] {
    width: 60px;
    height: 40px;
    border: none;
    border-radius: ${theme.borderRadius.sm};
    cursor: pointer;
  }
`;

const DangerZone = styled(GlassCard)`
  border-color: ${theme.colors.error};
  background: rgba(255, 59, 59, 0.05);
`;

export const Settings = () => {
  const [message, setMessage] = useState<{ type: 'success' | 'error' | 'info'; text: string } | null>(null);
  const [primaryColor, setPrimaryColor] = useState(theme.colors.primary);
  const [secondaryColor, setSecondaryColor] = useState(theme.colors.secondary);

  const exportJSON = async () => {
    try {
      const projects = await db.projects.toArray();
      const media = await db.media.toArray();
      const data = { projects, media };

      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `portfolio-backup-${new Date().toISOString().split('T')[0]}.json`;
      a.click();
      URL.revokeObjectURL(url);

      setMessage({ type: 'success', text: 'Data exported successfully!' });
      setTimeout(() => setMessage(null), 3000);
    } catch (error) {
      setMessage({ type: 'error', text: 'Export failed. Please try again.' });
      setTimeout(() => setMessage(null), 3000);
    }
  };

  const exportCSV = async () => {
    try {
      const projects = await db.projects.toArray();
      const headers = ['Title', 'Description', 'Status', 'GitHub URL', 'Live Demo URL', 'Tech Stack'];
      const rows = projects.map(p => [
        p.title,
        p.description.replace(/,/g, ';'),
        p.status,
        p.githubUrl || '',
        p.liveDemoUrl || '',
        p.techStack.join('; ')
      ]);

      const csv = [
        headers.join(','),
        ...rows.map(row => row.map(cell => `"${cell}"`).join(','))
      ].join('\n');

      const blob = new Blob([csv], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `portfolio-projects-${new Date().toISOString().split('T')[0]}.csv`;
      a.click();
      URL.revokeObjectURL(url);

      setMessage({ type: 'success', text: 'CSV exported successfully!' });
      setTimeout(() => setMessage(null), 3000);
    } catch (error) {
      setMessage({ type: 'error', text: 'Export failed. Please try again.' });
      setTimeout(() => setMessage(null), 3000);
    }
  };

  const importJSON = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      const text = await file.text();
      const data = JSON.parse(text);

      if (data.projects) {
        await db.projects.clear();
        await db.projects.bulkAdd(data.projects);
      }

      if (data.media) {
        await db.media.clear();
        await db.media.bulkAdd(data.media);
      }

      setMessage({ type: 'success', text: 'Data imported successfully!' });
      setTimeout(() => setMessage(null), 3000);
      setTimeout(() => window.location.reload(), 1500);
    } catch (error) {
      setMessage({ type: 'error', text: 'Import failed. Invalid file format.' });
      setTimeout(() => setMessage(null), 3000);
    }
  };

  const clearAllData = async () => {
    if (confirm('⚠️ This will delete ALL data permanently. Are you absolutely sure?')) {
      if (confirm('This action cannot be undone. Continue?')) {
        await db.projects.clear();
        await db.media.clear();
        setMessage({ type: 'info', text: 'All data cleared successfully.' });
        setTimeout(() => window.location.reload(), 1500);
      }
    }
  };

  return (
    <Container>
      <Title>Settings</Title>

      <Section>
        <SectionTitle>Theme Customization</SectionTitle>
        <SectionDescription>
          Customize the color scheme of your admin panel
        </SectionDescription>
        <ThemeOption>
          <ColorPicker>
            <label>Primary Color</label>
            <input
              type="color"
              value={primaryColor}
              onChange={(e) => setPrimaryColor(e.target.value)}
            />
          </ColorPicker>
          <ColorPicker>
            <label>Secondary Color</label>
            <input
              type="color"
              value={secondaryColor}
              onChange={(e) => setSecondaryColor(e.target.value)}
            />
          </ColorPicker>
        </ThemeOption>
        <Actions style={{ marginTop: theme.spacing.md }}>
          <Button variant="secondary">Apply Theme</Button>
          <Button variant="secondary">Reset to Default</Button>
        </Actions>
      </Section>

      <Section>
        <SectionTitle>Export Data</SectionTitle>
        <SectionDescription>
          Download your portfolio data for backup or migration
        </SectionDescription>
        <Actions>
          <Button onClick={exportJSON}>Export as JSON</Button>
          <Button variant="secondary" onClick={exportCSV}>
            Export as CSV
          </Button>
        </Actions>
      </Section>

      <Section>
        <SectionTitle>Import Data</SectionTitle>
        <SectionDescription>
          Import previously exported data (this will replace existing data)
        </SectionDescription>
        <Actions>
          <FileInput
            id="import-file"
            type="file"
            accept=".json"
            onChange={importJSON}
          />
          <Button
            as="label"
            htmlFor="import-file"
            style={{ cursor: 'pointer' }}
          >
            Import JSON
          </Button>
        </Actions>
      </Section>

      <DangerZone>
        <SectionTitle style={{ color: theme.colors.error }}>Danger Zone</SectionTitle>
        <SectionDescription>
          Irreversible actions that will permanently delete your data
        </SectionDescription>
        <Actions>
          <Button variant="danger" onClick={clearAllData}>
            Clear All Data
          </Button>
        </Actions>
      </DangerZone>

      {message && (
        <InfoBox
          $type={message.type}
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0 }}
        >
          {message.text}
        </InfoBox>
      )}
    </Container>
  );
};
