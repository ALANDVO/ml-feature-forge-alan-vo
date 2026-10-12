import React, { useEffect, useState } from 'react';
import {
  compilePipeline,
  createDataset,
  demoLogin,
  listAuditLogs,
  listCandidates,
  listDatasets,
  listEvaluations,
  listPipelines,
  loadSampleDataset,
  logout,
  proposeFeatures,
  runEvaluation,
} from './api';
import { AuditLogView } from './components/AuditLogView';
import { DatasetManager } from './components/DatasetManager';
import { EvaluationDashboard } from './components/EvaluationDashboard';
import { FeatureForge } from './components/FeatureForge';
import { Navbar } from './components/Navbar';
import { PipelineExport } from './components/PipelineExport';
import {
  AuditLog,
  Dataset,
  FeatureCandidate,
  FeatureEvaluation,
  Pipeline,
  UserProfile,
} from './types';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'datasets' | 'forge' | 'evaluations' | 'pipelines' | 'audit'>('datasets');
  const [user, setUser] = useState<UserProfile | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
  const [candidates, setCandidates] = useState<FeatureCandidate[]>([]);
  const [evaluations, setEvaluations] = useState<FeatureEvaluation[]>([]);
  const [pipelines, setPipelines] = useState<Pipeline[]>([]);
  const [auditLogs, setAuditLogs] = useState<AuditLog[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    // Initial bootstrap: log in as demo analyst to enable exploration by default
    handleDemoLogin('analyst');
  }, []);

  useEffect(() => {
    if (user) {
      loadInitialData();
    }
  }, [user]);

  useEffect(() => {
    if (selectedDataset) {
      loadDatasetDetails(selectedDataset.id);
    }
  }, [selectedDataset]);

  const handleDemoLogin = async (role: string) => {
    try {
      setIsLoading(true);
      const res = await demoLogin(role);
      setUser(res.user);
      setErrorMessage(null);
    } catch (err: any) {
      setErrorMessage(err.message || 'Failed to authenticate');
    } finally {
      setIsLoading(false);
    }
  };

  const handleLogout = async () => {
    await logout();
    setUser(null);
    setDatasets([]);
    setSelectedDataset(null);
  };

  const loadInitialData = async () => {
    try {
      setIsLoading(true);
      const ds = await listDatasets();
      setDatasets(ds);
      if (ds.length > 0 && !selectedDataset) {
        setSelectedDataset(ds[0]);
      }
      const logs = await listAuditLogs();
      setAuditLogs(logs);
    } catch (err: any) {
      setErrorMessage(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const loadDatasetDetails = async (datasetId: string) => {
    try {
      setIsLoading(true);
      const [cands, evals, pipes] = await Promise.all([
        listCandidates(datasetId),
        listEvaluations(datasetId),
        listPipelines(datasetId),
      ]);
      setCandidates(cands);
      setEvaluations(evals);
      setPipelines(pipes);
    } catch (err: any) {
      setErrorMessage(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleUploadCsv = async (name: string, targetCol: string, content: string) => {
    try {
      setIsLoading(true);
      const created = await createDataset({
        name,
        target_column: targetCol,
        csv_content: content,
      });
      setDatasets((prev) => [created, ...prev]);
      setSelectedDataset(created);
      setErrorMessage(null);
    } catch (err: any) {
      setErrorMessage(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleLoadSample = async () => {
    try {
      setIsLoading(true);
      const sample = await loadSampleDataset();
      setDatasets((prev) => [sample, ...prev]);
      setSelectedDataset(sample);
      setErrorMessage(null);
    } catch (err: any) {
      setErrorMessage(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleProposeFeatures = async (enableLlm: boolean) => {
    if (!selectedDataset) return;
    try {
      setIsLoading(true);
      const res = await proposeFeatures({
        dataset_id: selectedDataset.id,
        enable_llm: enableLlm,
      });
      setCandidates(res);
      setErrorMessage(null);
    } catch (err: any) {
      setErrorMessage(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRunEvaluation = async () => {
    if (!selectedDataset) return;
    try {
      setIsLoading(true);
      const res = await runEvaluation({
        dataset_id: selectedDataset.id,
      });
      setEvaluations(res);
      setErrorMessage(null);
    } catch (err: any) {
      setErrorMessage(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCompilePipeline = async () => {
    if (!selectedDataset) return;
    try {
      setIsLoading(true);
      const pipe = await compilePipeline({
        dataset_id: selectedDataset.id,
        name: `${selectedDataset.name} Production Pipeline`,
      });
      setPipelines((prev) => [pipe, ...prev]);
      setErrorMessage(null);
    } catch (err: any) {
      setErrorMessage(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div>
      <Navbar user={user} onLoginDemo={handleDemoLogin} onLogout={handleLogout} />

      <main className="container">
        {errorMessage && (
          <div className="card" style={{ border: '1px solid #ef4444', backgroundColor: 'rgba(239, 68, 68, 0.1)', color: '#f87171' }}>
            <strong>Error:</strong> {errorMessage}
            <button
              onClick={() => setErrorMessage(null)}
              style={{ float: 'right', background: 'transparent', color: '#f87171', border: 'none', cursor: 'pointer' }}
            >
              &times;
            </button>
          </div>
        )}

        <div className="tabs-header">
          <button
            className={`tab-btn ${activeTab === 'datasets' ? 'active' : ''}`}
            onClick={() => setActiveTab('datasets')}
          >
            1. Datasets & Ingestion
          </button>
          <button
            className={`tab-btn ${activeTab === 'forge' ? 'active' : ''}`}
            onClick={() => setActiveTab('forge')}
          >
            2. Feature Forge ({candidates.length})
          </button>
          <button
            className={`tab-btn ${activeTab === 'evaluations' ? 'active' : ''}`}
            onClick={() => setActiveTab('evaluations')}
          >
            3. Evaluation & Quality ({evaluations.length})
          </button>
          <button
            className={`tab-btn ${activeTab === 'pipelines' ? 'active' : ''}`}
            onClick={() => setActiveTab('pipelines')}
          >
            4. Export Pipeline ({pipelines.length})
          </button>
          <button
            className={`tab-btn ${activeTab === 'audit' ? 'active' : ''}`}
            onClick={() => setActiveTab('audit')}
          >
            5. Audit Log ({auditLogs.length})
          </button>
        </div>

        {activeTab === 'datasets' && (
          <DatasetManager
            datasets={datasets}
            selectedDataset={selectedDataset}
            onSelectDataset={(ds) => setSelectedDataset(ds)}
            onUploadCsv={handleUploadCsv}
            onLoadSample={handleLoadSample}
            isLoading={isLoading}
          />
        )}

        {activeTab === 'forge' && (
          <FeatureForge
            dataset={selectedDataset}
            candidates={candidates}
            onPropose={handleProposeFeatures}
            isLoading={isLoading}
          />
        )}

        {activeTab === 'evaluations' && (
          <EvaluationDashboard
            dataset={selectedDataset}
            evaluations={evaluations}
            onRunEvaluation={handleRunEvaluation}
            isLoading={isLoading}
          />
        )}

        {activeTab === 'pipelines' && (
          <PipelineExport
            dataset={selectedDataset}
            pipelines={pipelines}
            onCompile={handleCompilePipeline}
            isLoading={isLoading}
          />
        )}

        {activeTab === 'audit' && (
          <AuditLogView
            logs={auditLogs}
            onRefresh={async () => {
              const logs = await listAuditLogs();
              setAuditLogs(logs);
            }}
            isLoading={isLoading}
          />
        )}
      </main>
    </div>
  );
};
