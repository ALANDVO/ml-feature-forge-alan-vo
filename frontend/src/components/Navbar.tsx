import React from 'react';
import { UserProfile } from '../types';

interface NavbarProps {
  user: UserProfile | null;
  onLoginDemo: (role: string) => void;
  onLogout: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({ user, onLoginDemo, onLogout }) => {
  return (
    <header className="nav-bar">
      <div>
        <h1 style={{ fontSize: '1.25rem', fontWeight: 'bold' }}>
          ML Feature Forge <span style={{ fontSize: '0.9rem', color: '#94a3b8', fontWeight: 'normal' }}>| Alan Vo &bull; AI & Machine Learning</span>
        </h1>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        {user ? (
          <>
            <span style={{ fontSize: '0.85rem', color: '#cbd5e1' }}>
              User: <strong>{user.username}</strong>
            </span>
            <span className="badge badge-deterministic">
              Role: {user.roles[0] || 'viewer'}
            </span>
            {user.is_demo && (
              <span className="badge badge-advisory">Demo Mode</span>
            )}
            <button className="secondary-btn" onClick={onLogout} style={{ padding: '4px 10px', fontSize: '0.85rem' }}>
              Logout
            </button>
          </>
        ) : (
          <div style={{ display: 'flex', gap: '8px' }}>
            <button className="primary-btn" onClick={() => onLoginDemo('analyst')} style={{ padding: '6px 12px' }}>
              Demo Login (Analyst)
            </button>
            <button className="secondary-btn" onClick={() => onLoginDemo('viewer')} style={{ padding: '6px 12px' }}>
              Demo Login (Viewer)
            </button>
          </div>
        )}
      </div>
    </header>
  );
};
