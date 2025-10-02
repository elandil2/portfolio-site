import Dexie, { Table } from 'dexie';
import { Project, MediaFile } from '../types';

export class PortfolioDB extends Dexie {
  projects!: Table<Project>;
  media!: Table<MediaFile>;

  constructor() {
    super('PortfolioDB');
    this.version(1).stores({
      projects: 'id, title, status, order, createdAt, updatedAt',
      media: 'id, name, type, uploadedAt'
    });
  }
}

export const db = new PortfolioDB();
