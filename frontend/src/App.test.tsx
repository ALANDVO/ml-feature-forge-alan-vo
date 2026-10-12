import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { App } from './App';
import * as api from './api';

describe('ML Feature Forge Frontend Application', () => {
  beforeEach(() => {
    vi.restoreAllMocks();

    vi.spyOn(api, 'demoLogin').mockResolvedValue({
      access_token: 'synthetic-test-token-000000000',
      user: {
        user_id: 'demo-analyst-id',
        username: 'demo-analyst',
        email: 'demo@local.dev',
        roles: ['analyst'],
        is_demo: true,
      },
    });

    vi.spyOn(api, 'listDatasets').mockResolvedValue([
      {
        id: 'ds-test-1',
        name: 'Housing Market Test',
        filename: 'housing.csv',
        row_count: 100,
        col_count: 5,
        target_column: 'price',
        problem_type: 'regression',
        profile: {
          row_count: 100,
          col_count: 5,
          columns: ['sqft', 'bedrooms', 'bathrooms', 'zipcode', 'price'],
          dtypes: {
            sqft: 'numeric',
            bedrooms: 'numeric',
            bathrooms: 'numeric',
            zipcode: 'categorical',
            price: 'numeric',
          },
          numeric_columns: ['sqft', 'bedrooms', 'bathrooms', 'price'],
          categorical_columns: ['zipcode'],
          datetime_columns: [],
          missing_counts: { sqft: 0, bedrooms: 0, bathrooms: 0, zipcode: 0, price: 0 },
          missing_rates: { sqft: 0, bedrooms: 0, bathrooms: 0, zipcode: 0, price: 0 },
          stats: {
            sqft: { min: 500, max: 4000, mean: 1800, std: 600 },
            price: { min: 100000, max: 900000, mean: 400000, std: 150000 },
          },
          target_column: 'price',
          problem_type: 'regression',
        },
        sample_rows: [
          { sqft: 1500, bedrooms: 3, bathrooms: 2, zipcode: '94102', price: 450000 },
        ],
        created_by: 'demo-analyst',
        created_at: '2026-10-12T10:00:00Z',
      },
    ]);

    vi.spyOn(api, 'listCandidates').mockResolvedValue([
      {
        id: 'cand-1',
        dataset_id: 'ds-test-1',
        name: 'ratio_sqft_per_bedrooms',
        transform_type: 'ratio',
        expression: "df['sqft'] / (df['bedrooms'] + 1e-5)",
        input_columns: ['sqft', 'bedrooms'],
        hypothesis: 'Average space per bedroom indicates luxury density.',
        source: 'deterministic',
        expected_signal: 'strong',
        risk: 'none',
        created_at: '2026-10-12T10:05:00Z',
      },
    ]);

    vi.spyOn(api, 'listEvaluations').mockResolvedValue([
      {
        id: 'eval-1',
        dataset_id: 'ds-test-1',
        feature_name: 'ratio_sqft_per_bedrooms',
        importance_score: 0.65,
        correlation_target: 0.65,
        leakage_risk: 'none',
        collinearity_with: [],
        verdict: 'keep',
        reason: 'Strong correlation with price without collinearity.',
        evaluated_at: '2026-10-12T10:10:00Z',
      },
    ]);

    vi.spyOn(api, 'listPipelines').mockResolvedValue([]);
    vi.spyOn(api, 'listAuditLogs').mockResolvedValue([
      {
        id: 'audit-1',
        user_id: 'demo-analyst',
        action: 'create_dataset',
        resource_type: 'dataset',
        resource_id: 'ds-test-1',
        details: { name: 'Housing Market Test' },
        ip_address: '127.0.0.1',
        timestamp: '2026-10-12T10:00:00Z',
      },
    ]);
  });

  it('renders application header and branding correctly', async () => {
    render(<App />);
    expect(screen.getByText(/ML Feature Forge/i)).toBeInTheDocument();
    expect(screen.getByText(/Alan Vo/i)).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText('Housing Market Test')).toBeInTheDocument();
    });
  });

  it('switches navigation tabs and displays candidate features', async () => {
    render(<App />);

    await waitFor(() => {
      expect(screen.getByText(/2. Feature Forge/i)).toBeInTheDocument();
    });

    const forgeTab = screen.getByText(/2. Feature Forge/i);
    fireEvent.click(forgeTab);

    await waitFor(() => {
      expect(screen.getByText('ratio_sqft_per_bedrooms')).toBeInTheDocument();
      expect(screen.getByText(/Average space per bedroom/i)).toBeInTheDocument();
    });
  });

  it('displays evaluation metrics and verdict tags', async () => {
    render(<App />);

    await waitFor(() => {
      expect(screen.getByText(/3. Evaluation & Quality/i)).toBeInTheDocument();
    });

    const evalTab = screen.getByText(/3. Evaluation & Quality/i);
    fireEvent.click(evalTab);

    await waitFor(() => {
      expect(screen.getByText('keep')).toBeInTheDocument();
      expect(screen.getByText(/Strong correlation with price/i)).toBeInTheDocument();
    });
  });
});
