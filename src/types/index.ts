export interface Project {
  id: string;
  title: string;
  description: string;
  images: string[];
  videoUrl?: string;
  githubUrl?: string;
  liveDemoUrl?: string;
  techStack: string[];
  status: 'published' | 'draft';
  order: number;
  createdAt: Date;
  updatedAt: Date;
}

export interface MediaFile {
  id: string;
  url: string;
  name: string;
  size: number;
  type: string;
  uploadedAt: Date;
}

export interface Stats {
  totalProjects: number;
  publishedProjects: number;
  draftProjects: number;
  totalViews: number;
}
