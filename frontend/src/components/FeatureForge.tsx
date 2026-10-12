import React, { useState } from 'react';
import { Dataset, FeatureCandidate } from '../types';

interface FeatureForgeProps {
  dataset: Dataset | null;
  candidates: FeatureCandidate[];
  onPropose: (enableLlm: boolean) => Promise<void>;
  isLoading: boolean;
}

export const FeatureForge: React.FC<FeatureForgeProps> = ({
  dataset,
  candidates,
  onPropose,
  isLoading,
}) => {
  const [enableLlm, setEnableLlm] = useState(false);
  const [filterType, setFilterType] = useState('all');

  if (!dataset) {
    return (
      <div className="card">
        <h3>Feature Hypothesis Forge</h3>
        <p style={{ color: '#94a3b8', marginTop: '8px' }}>
          Please select or ingest a dataset first to generate candidate feature hypotheses.
        </p>
      </div>
    );
  }

  const filteredCandidates = candidates.filter((c) => {
    if (filterType === 'all') return true;
    if (filterType === 'deterministic') return c.source === 'deterministic';
    if (filterType === 'advisory') return c.source === 'llm_advisory';
    return c.transform_type === filterType;
  });

  return (
    <div>
      <div className="card" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h3>Feature Generation: {dataset.name}</h3>
          <p style={{ color: '#94a3b8', fontSize: '0.85rem' }}>
            Target: <strong>{dataset.target_column}</strong> | Problem: <strong>{dataset.problem_type}</strong>
          </p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.85rem', color: '#cbd5e1' }}>
            <input
              type="checkbox"
              checked={enableLlm}
              onChange={(e) => setEnableLlm(e.target.checked)}
            />
            Enable LLM Reasoning (Opt-in Advisory)
          </label>
          <button className="primary-btn" onClick={() => onPropose(enableLlm)} disabled={isLoading}>
            {isLoading ? 'Generating...' : 'Propose Features'}
          </button>
        </div>
      </div>

      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
          <h3>Candidate Features ({filteredCandidates.length})</h3>
          <div>
            <label style={{ fontSize: '0.85rem', color: '#94a3b8', marginRight: '8px' }}>Filter:</label>
            <select value={filterType} onChange={(e) => setFilterType(e.target.value)}>
              <option value="all">All Candidates</option>
              <option value="deterministic">Deterministic Only</option>
              <option value="advisory">LLM Advisory Only</option>
              <option value="ratio">Ratios</option>
              <option value="product">Products</option>
              <option value="log1p">Log transforms</option>
              <option value="standard_scale">Z-scores</option>
              <option value="quantile_bin">Quantile bins</option>
            </select>
          </div>
        </div>

        {filteredCandidates.length === 0 ? (
          <p style={{ color: '#94a3b8' }}>
            No candidate features generated yet. Click &quot;Propose Features&quot; to execute domain transforms.
          </p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Feature Name</th>
                <th>Transform Type</th>
                <th>Inputs</th>
                <th>Expression</th>
                <th>Hypothesis & Mechanism</th>
                <th>Source</th>
              </tr>
            </thead>
            <tbody>
              {filteredCandidates.map((c) => (
                <tr key={c.id}>
                  <td>
                    <code>{c.name}</code>
                  </td>
                  <td>
                    <span className="badge badge-deterministic">{c.transform_type}</span>
                  </td>
                  <td>
                    <small>{c.input_columns.join(', ')}</small>
                  </td>
                  <td>
                    <code style={{ fontSize: '0.8rem', color: '#38bdf8' }}>{c.expression}</code>
                  </td>
                  <td style={{ fontSize: '0.8rem', color: '#cbd5e1', maxWidth: '300px' }}>
                    {c.hypothesis}
                  </td>
                  <td>
                    <span className={`badge ${c.source === 'llm_advisory' ? 'badge-advisory' : 'badge-deterministic'}`}>
                      {c.source === 'llm_advisory' ? 'LLM Advisory' : 'Deterministic'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
};
