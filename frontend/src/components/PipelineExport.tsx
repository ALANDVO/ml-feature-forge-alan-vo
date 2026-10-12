import React, { useState } from 'react';
import { Dataset, Pipeline } from '../types';

interface PipelineExportProps {
  dataset: Dataset | null;
  pipelines: Pipeline[];
  onCompile: () => Promise<void>;
  isLoading: boolean;
}

export const PipelineExport: React.FC<PipelineExportProps> = ({
  dataset,
  pipelines,
  onCompile,
  isLoading,
}) => {
  const [selectedPipeId, setSelectedPipeId] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  if (!dataset) {
    return (
      <div className="card">
        <h3>Pipeline Export & Code Generation</h3>
        <p style={{ color: '#94a3b8', marginTop: '8px' }}>
          Select a dataset and run evaluations to compile an exportable pipeline.
        </p>
      </div>
    );
  }

  const activePipeline = pipelines.find((p) => p.id === selectedPipeId) || pipelines[0] || null;

  const handleCopy = () => {
    if (!activePipeline) return;
    navigator.clipboard.writeText(activePipeline.python_code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    if (!activePipeline) return;
    const blob = new Blob([activePipeline.python_code], { type: 'text/x-python' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${activePipeline.name.toLowerCase().replace(/[^a-z0-9]/g, '_')}.py`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div>
      <div className="card" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h3>Pipeline Compilation: {dataset.name}</h3>
          <p style={{ color: '#94a3b8', fontSize: '0.85rem' }}>
            Compiles kept features into a self-contained, reproducible Python Pandas pipeline module.
          </p>
        </div>
        <button className="primary-btn" onClick={onCompile} disabled={isLoading}>
          {isLoading ? 'Compiling...' : 'Compile Production Pipeline'}
        </button>
      </div>

      {pipelines.length > 0 && (
        <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
          {pipelines.map((p) => (
            <button
              key={p.id}
              className={(activePipeline?.id === p.id) ? 'primary-btn' : 'secondary-btn'}
              onClick={() => setSelectedPipeId(p.id)}
              style={{ fontSize: '0.85rem' }}
            >
              {p.name} ({p.selected_features.length} features)
            </button>
          ))}
        </div>
      )}

      {activePipeline ? (
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
            <div>
              <h4>{activePipeline.name}</h4>
              <p style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
                Features: {activePipeline.selected_features.join(', ')}
              </p>
            </div>
            <div style={{ display: 'flex', gap: '8px' }}>
              <button className="secondary-btn" onClick={handleCopy} style={{ padding: '6px 12px', fontSize: '0.85rem' }}>
                {copied ? 'Copied!' : 'Copy Code'}
              </button>
              <button className="primary-btn" onClick={handleDownload} style={{ padding: '6px 12px', fontSize: '0.85rem' }}>
                Download .py
              </button>
            </div>
          </div>
          <pre className="code-viewer">{activePipeline.python_code}</pre>
        </div>
      ) : (
        <div className="card">
          <p style={{ color: '#94a3b8' }}>
            No compiled pipelines yet. Click &quot;Compile Production Pipeline&quot; to generate production code.
          </p>
        </div>
      )}
    </div>
  );
};
