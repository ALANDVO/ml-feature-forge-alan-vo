import React from 'react';
import { Dataset, FeatureEvaluation } from '../types';

interface EvaluationDashboardProps {
  dataset: Dataset | null;
  evaluations: FeatureEvaluation[];
  onRunEvaluation: () => Promise<void>;
  isLoading: boolean;
}

export const EvaluationDashboard: React.FC<EvaluationDashboardProps> = ({
  dataset,
  evaluations,
  onRunEvaluation,
  isLoading,
}) => {
  if (!dataset) {
    return (
      <div className="card">
        <h3>Feature Evaluation & Statistical Quality</h3>
        <p style={{ color: '#94a3b8', marginTop: '8px' }}>
          Select an active dataset to evaluate feature candidates against target signal.
        </p>
      </div>
    );
  }

  const keptCount = evaluations.filter((e) => e.verdict === 'keep').length;
  const droppedCount = evaluations.filter((e) => e.verdict === 'drop').length;
  const maxImportance = Math.max(...evaluations.map((e) => e.importance_score), 0.001);

  return (
    <div>
      <div className="card" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h3>Feature Quality Evaluation: {dataset.name}</h3>
          <p style={{ color: '#94a3b8', fontSize: '0.85rem' }}>
            Target: <strong>{dataset.target_column}</strong> | Statistical Importance & Leakage Gate
          </p>
        </div>
        <button className="primary-btn" onClick={onRunEvaluation} disabled={isLoading}>
          {isLoading ? 'Evaluating...' : 'Run Statistical Evaluation'}
        </button>
      </div>

      {evaluations.length > 0 && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px', marginBottom: '20px' }}>
          <div className="card" style={{ margin: 0, textAlign: 'center' }}>
            <div style={{ fontSize: '0.85rem', color: '#94a3b8' }}>Evaluated Features</div>
            <div style={{ fontSize: '1.75rem', fontWeight: 'bold' }}>{evaluations.length}</div>
          </div>
          <div className="card" style={{ margin: 0, textAlign: 'center' }}>
            <div style={{ fontSize: '0.85rem', color: '#10b981' }}>Kept (Production Ready)</div>
            <div style={{ fontSize: '1.75rem', fontWeight: 'bold', color: '#10b981' }}>{keptCount}</div>
          </div>
          <div className="card" style={{ margin: 0, textAlign: 'center' }}>
            <div style={{ fontSize: '0.85rem', color: '#ef4444' }}>Dropped (Pruned)</div>
            <div style={{ fontSize: '1.75rem', fontWeight: 'bold', color: '#ef4444' }}>{droppedCount}</div>
          </div>
          <div className="card" style={{ margin: 0, textAlign: 'center' }}>
            <div style={{ fontSize: '0.85rem', color: '#3b82f6' }}>Peak Signal |r|</div>
            <div style={{ fontSize: '1.75rem', fontWeight: 'bold', color: '#3b82f6' }}>{maxImportance.toFixed(3)}</div>
          </div>
        </div>
      )}

      <div className="card">
        <h3>Feature Ranking & Pruning Rationale</h3>
        {evaluations.length === 0 ? (
          <p style={{ color: '#94a3b8', marginTop: '8px' }}>
            No evaluation executed yet. Click &quot;Run Statistical Evaluation&quot; to test candidate quality.
          </p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Verdict</th>
                <th>Feature Name</th>
                <th>Target Signal |r|</th>
                <th>Importance Relative Bar</th>
                <th>Leakage / Collinearity</th>
                <th>Verdict Rationale</th>
              </tr>
            </thead>
            <tbody>
              {evaluations.map((e) => {
                const barWidth = Math.min(100, (e.importance_score / maxImportance) * 100);
                return (
                  <tr key={e.id}>
                    <td>
                      <span className={`badge ${e.verdict === 'keep' ? 'badge-keep' : 'badge-drop'}`}>
                        {e.verdict}
                      </span>
                    </td>
                    <td>
                      <code>{e.feature_name}</code>
                    </td>
                    <td>
                      <strong>{e.importance_score.toFixed(4)}</strong>
                    </td>
                    <td style={{ width: '150px' }}>
                      <div style={{ width: '100%', height: '8px', backgroundColor: '#334155', borderRadius: '4px' }}>
                        <div
                          style={{
                            width: `${barWidth}%`,
                            height: '100%',
                            backgroundColor: e.verdict === 'keep' ? '#10b981' : '#ef4444',
                            borderRadius: '4px',
                          }}
                        />
                      </div>
                    </td>
                    <td>
                      {e.leakage_risk !== 'none' ? (
                        <span className="badge badge-drop">{e.leakage_risk}</span>
                      ) : e.collinearity_with.length > 0 ? (
                        <span className="badge badge-advisory">Collinear with {e.collinearity_with.join(', ')}</span>
                      ) : (
                        <span style={{ color: '#94a3b8', fontSize: '0.8rem' }}>Clean</span>
                      )}
                    </td>
                    <td style={{ fontSize: '0.8rem', color: '#cbd5e1', maxWidth: '300px' }}>
                      {e.reason}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
};
