export interface DatasetProfile {
  row_count: number;
  col_count: number;
  columns: string[];
  dtypes: Record<string, string>;
  numeric_columns: string[];
  categorical_columns: string[];
  datetime_columns: string[];
  missing_counts: Record<string, number>;
  missing_rates: Record<string, number>;
  stats: Record<string, any>;
  target_column: string;
  problem_type: string;
}

export interface Dataset {
  id: string;
  name: string;
  filename: string;
  row_count: number;
  col_count: number;
  target_column: string;
  problem_type: string;
  profile: DatasetProfile;
  sample_rows: Record<string, any>[];
  created_by: string;
  created_at: string;
}

export interface FeatureCandidate {
  id: string;
  dataset_id: string;
  name: string;
  transform_type: string;
  expression: string;
  input_columns: string[];
  hypothesis: string;
  source: 'deterministic' | 'llm_advisory';
  expected_signal: string;
  risk: string;
  created_at: string;
}

export interface FeatureEvaluation {
  id: string;
  dataset_id: string;
  feature_name: string;
  importance_score: number;
  correlation_target: number;
  leakage_risk: string;
  collinearity_with: string[];
  verdict: 'keep' | 'drop';
  reason: string;
  evaluated_at: string;
}

export interface Pipeline {
  id: string;
  dataset_id: string;
  name: string;
  selected_features: string[];
  python_code: string;
  json_spec: Record<string, any>;
  created_by: string;
  created_at: string;
}

export interface AuditLog {
  id: string;
  user_id: string;
  action: string;
  resource_type: string;
  resource_id: string;
  details: Record<string, any>;
  ip_address: string | null;
  timestamp: string;
}

export interface UserProfile {
  user_id: string;
  username: string;
  email: string;
  roles: string[];
  is_demo: boolean;
}

export interface AuthConfig {
  issuer_url: string;
  client_id: string;
  audience: string;
  demo_mode: boolean;
  roles: string[];
}
