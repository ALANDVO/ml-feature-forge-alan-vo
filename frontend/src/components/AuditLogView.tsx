import React from 'react';
import { AuditLog } from '../types';

interface AuditLogViewProps {
  logs: AuditLog[];
  onRefresh: () => Promise<void>;
  isLoading: boolean;
}

export const AuditLogView: React.FC<AuditLogViewProps> = ({ logs, onRefresh, isLoading }) => {
  return (
    <div className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <div>
          <h3>System Audit Trail & Security Ledger</h3>
          <p style={{ color: '#94a3b8', fontSize: '0.85rem' }}>
            Immutable persistent record of all dataset ingestions, feature proposals, and pipeline compiles.
          </p>
        </div>
        <button className="secondary-btn" onClick={onRefresh} disabled={isLoading}>
          {isLoading ? 'Refreshing...' : 'Refresh Audit Log'}
        </button>
      </div>

      {logs.length === 0 ? (
        <p style={{ color: '#94a3b8' }}>No audit events logged yet.</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Timestamp (UTC)</th>
              <th>User</th>
              <th>Action</th>
              <th>Resource Type</th>
              <th>Details</th>
            </tr>
          </thead>
          <tbody>
            {logs.map((log) => (
              <tr key={log.id}>
                <td style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
                  {new Date(log.timestamp).toLocaleString()}
                </td>
                <td>
                  <strong>{log.user_id}</strong>
                </td>
                <td>
                  <span className="badge badge-deterministic">{log.action}</span>
                </td>
                <td>
                  <code>{log.resource_type}</code>
                </td>
                <td style={{ fontSize: '0.8rem', color: '#cbd5e1' }}>
                  <code>{JSON.stringify(log.details)}</code>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
};
