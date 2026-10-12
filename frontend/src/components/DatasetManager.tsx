import React, { useState } from 'react';
import { Dataset } from '../types';

interface DatasetManagerProps {
  datasets: Dataset[];
  selectedDataset: Dataset | null;
  onSelectDataset: (dataset: Dataset) => void;
  onUploadCsv: (name: string, targetCol: string, content: string) => Promise<void>;
  onLoadSample: () => Promise<void>;
  isLoading: boolean;
}

export const DatasetManager: React.FC<DatasetManagerProps> = ({
  datasets,
  selectedDataset,
  onSelectDataset,
  onUploadCsv,
  onLoadSample,
  isLoading,
}) => {
  const [name, setName] = useState('');
  const [targetCol, setTargetCol] = useState('');
  const [csvText, setCsvText] = useState('');
  const [showUploadModal, setShowUploadModal] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name || !targetCol || !csvText) return;
    await onUploadCsv(name, targetCol, csvText);
    setName('');
    setTargetCol('');
    setCsvText('');
    setShowUploadModal(false);
  };

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <h2>Dataset Repositories & Ingestion</h2>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button className="primary-btn" onClick={onLoadSample} disabled={isLoading}>
            Load Credit Loan Sample
          </button>
          <button className="secondary-btn" onClick={() => setShowUploadModal(!showUploadModal)}>
            {showUploadModal ? 'Cancel' : 'Upload CSV Dataset'}
          </button>
        </div>
      </div>

      {showUploadModal && (
        <div className="card" style={{ border: '1px solid #3b82f6' }}>
          <h3>Upload Raw CSV Dataset</h3>
          <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginTop: '12px' }}>
            <div>
              <label style={{ display: 'block', fontSize: '0.85rem', marginBottom: '4px', color: '#94a3b8' }}>
                Dataset Name:
              </label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g. Housing Pricing Matrix"
                required
                style={{ width: '100%' }}
              />
            </div>
            <div>
              <label style={{ display: 'block', fontSize: '0.85rem', marginBottom: '4px', color: '#94a3b8' }}>
                Target Column (for ML prediction):
              </label>
              <input
                type="text"
                value={targetCol}
                onChange={(e) => setTargetCol(e.target.value)}
                placeholder="e.g. price or approved"
                required
                style={{ width: '100%' }}
              />
            </div>
            <div>
              <label style={{ display: 'block', fontSize: '0.85rem', marginBottom: '4px', color: '#94a3b8' }}>
                Raw CSV Content (with headers):
              </label>
              <textarea
                value={csvText}
                onChange={(e) => setCsvText(e.target.value)}
                rows={6}
                placeholder="feature_a,feature_b,target&#10;1.2,3.4,1&#10;5.6,7.8,0"
                required
                style={{ width: '100%', fontFamily: 'monospace' }}
              />
            </div>
            <button type="submit" className="primary-btn" disabled={isLoading}>
              Ingest & Profile Dataset
            </button>
          </form>
        </div>
      )}

      <div className="card">
        <h3>Available Datasets ({datasets.length})</h3>
        {datasets.length === 0 ? (
          <p style={{ color: '#94a3b8', marginTop: '8px' }}>
            No datasets ingested yet. Click &quot;Load Credit Loan Sample&quot; to begin.
          </p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Dataset Name</th>
                <th>Rows</th>
                <th>Cols</th>
                <th>Target</th>
                <th>Problem Type</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {datasets.map((d) => {
                const isSelected = selectedDataset?.id === d.id;
                return (
                  <tr key={d.id} style={{ backgroundColor: isSelected ? 'rgba(59, 130, 246, 0.15)' : 'transparent' }}>
                    <td>
                      <strong>{d.name}</strong>
                    </td>
                    <td>{d.row_count}</td>
                    <td>{d.col_count}</td>
                    <td>
                      <code>{d.target_column}</code>
                    </td>
                    <td>
                      <span className="badge badge-deterministic">{d.problem_type}</span>
                    </td>
                    <td>
                      <button
                        className={isSelected ? 'primary-btn' : 'secondary-btn'}
                        onClick={() => onSelectDataset(d)}
                        style={{ padding: '4px 10px', fontSize: '0.8rem' }}
                      >
                        {isSelected ? 'Active' : 'Select'}
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>

      {selectedDataset && (
        <div className="card">
          <h3>Column Profiling: {selectedDataset.name}</h3>
          <p style={{ color: '#94a3b8', fontSize: '0.85rem', marginBottom: '12px' }}>
            Target Column: <strong>{selectedDataset.target_column}</strong> | Total Columns: {selectedDataset.col_count}
          </p>
          <table>
            <thead>
              <tr>
                <th>Column</th>
                <th>Inferred Type</th>
                <th>Null %</th>
                <th>Summary Statistics</th>
              </tr>
            </thead>
            <tbody>
              {selectedDataset.profile.columns.map((col) => {
                const dtype = selectedDataset.profile.dtypes[col] || 'unknown';
                const nullRate = selectedDataset.profile.missing_rates[col] || 0;
                const stats = selectedDataset.profile.stats[col] || {};
                const isTarget = col === selectedDataset.target_column;

                return (
                  <tr key={col}>
                    <td>
                      <code>{col}</code> {isTarget && <span className="badge badge-advisory">Target</span>}
                    </td>
                    <td>
                      <span className="badge badge-deterministic">{dtype}</span>
                    </td>
                    <td>{(nullRate * 100).toFixed(1)}%</td>
                    <td style={{ fontSize: '0.8rem', color: '#cbd5e1' }}>
                      {dtype === 'numeric' && stats.mean !== undefined ? (
                        <span>
                          min: {stats.min}, max: {stats.max}, mean: {stats.mean}, std: {stats.std}
                        </span>
                      ) : stats.top_categories ? (
                        <span>top: {stats.top_categories.join(', ')}</span>
                      ) : (
                        <span>distinct: {stats.distinct_count || 'n/a'}</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};
