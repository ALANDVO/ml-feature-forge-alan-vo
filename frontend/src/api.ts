/**
 * Typed API client derived from backend FastAPI routes.
 * Tokens are kept in memory to adhere strictly to SPA security standards.
 */
import {
  AuthConfig,
  AuditLog,
  Dataset,
  FeatureCandidate,
  FeatureEvaluation,
  Pipeline,
  UserProfile,
} from './types';

let inMemoryToken: string | null = null;

export function setAuthToken(token: string | null): void {
  inMemoryToken = token;
}

export function getAuthToken(): string | null {
  return inMemoryToken;
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(options.headers as Record<string, string>),
  };

  if (inMemoryToken) {
    headers['Authorization'] = `Bearer ${inMemoryToken}`;
  }

  const response = await fetch(`/api${path}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    let errMessage = `HTTP ${response.status} ${response.statusText}`;
    try {
      const errJson = await response.json();
      if (errJson && errJson.detail) {
        errMessage = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail);
      }
    } catch {
      // ignore
    }
    throw new Error(errMessage);
  }

  return response.json();
}

export async function getAuthConfig(): Promise<AuthConfig> {
  return request<AuthConfig>('/auth/config');
}

export async function demoLogin(role: string = 'analyst', username?: string): Promise<{ access_token: string; user: UserProfile }> {
  const res = await request<{ access_token: string; user: UserProfile }>('/auth/demo-login', {
    method: 'POST',
    body: JSON.stringify({ role, username }),
  });
  setAuthToken(res.access_token);
  return res;
}

export async function getCurrentUser(): Promise<UserProfile> {
  return request<UserProfile>('/auth/me');
}

export async function logout(): Promise<void> {
  if (inMemoryToken) {
    try {
      await request('/auth/logout', { method: 'POST' });
    } catch {
      // ignore
    }
  }
  setAuthToken(null);
}

export async function listDatasets(): Promise<Dataset[]> {
  return request<Dataset[]>('/datasets');
}

export async function getDataset(id: string): Promise<Dataset> {
  return request<Dataset>(`/datasets/${id}`);
}

export async function createDataset(params: {
  name: string;
  target_column: string;
  csv_content?: string;
  data_rows?: Record<string, any>[];
}): Promise<Dataset> {
  return request<Dataset>('/datasets', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

export async function loadSampleDataset(): Promise<Dataset> {
  return request<Dataset>('/datasets/demo-load', {
    method: 'POST',
  });
}

export async function deleteDataset(id: string): Promise<{ status: string }> {
  return request<{ status: string }>(`/datasets/${id}`, {
    method: 'DELETE',
  });
}

export async function listCandidates(datasetId: string): Promise<FeatureCandidate[]> {
  return request<FeatureCandidate[]>(`/features/${datasetId}`);
}

export async function proposeFeatures(params: {
  dataset_id: string;
  enable_llm?: boolean;
  max_candidates?: number;
}): Promise<FeatureCandidate[]> {
  return request<FeatureCandidate[]>('/features/propose', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

export async function listEvaluations(datasetId: string): Promise<FeatureEvaluation[]> {
  return request<FeatureEvaluation[]>(`/evaluations/${datasetId}`);
}

export async function runEvaluation(params: {
  dataset_id: string;
  collinearity_threshold?: number;
  min_importance_threshold?: number;
}): Promise<FeatureEvaluation[]> {
  return request<FeatureEvaluation[]>('/evaluations', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

export async function listPipelines(datasetId?: string): Promise<Pipeline[]> {
  const query = datasetId ? `?dataset_id=${encodeURIComponent(datasetId)}` : '';
  return request<Pipeline[]>(`/pipelines${query}`);
}

export async function getPipeline(id: string): Promise<Pipeline> {
  return request<Pipeline>(`/pipelines/${id}`);
}

export async function compilePipeline(params: {
  dataset_id: string;
  name?: string;
  selected_feature_ids?: string[];
}): Promise<Pipeline> {
  return request<Pipeline>('/pipelines', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

export async function listAuditLogs(limit: number = 50): Promise<AuditLog[]> {
  return request<AuditLog[]>(`/audit?limit=${limit}`);
}
