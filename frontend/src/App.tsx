import { useEffect, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { AlertCircle, ChevronDown, ChevronUp, TrendingUp, History } from 'lucide-react';

interface Game {
  gamePk: string;
  v_team: string;
  h_team: string;
  v_win_prob: number;
  h_win_prob: number;
  vegas_home_ml: number;
  vegas_implied: number;
  edge: number;
  bet: string;
  confidence: string;
  over_under: number;
}

interface Scouting {
  reasons: string[];
  h_p_elo: number;
  v_p_elo: number;
  park_factor: number;
}

export default function App() {
  const [activeTab, setActiveTab] = useState<'today' | 'standings' | 'history'>('today');
  const [games, setGames] = useState<Game[]>([]);
  const [standings, setStandings] = useState<any[]>([]);
  const [history, setHistory] = useState<any[]>([]);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [scouting, setScouting] = useState<Record<string, Scouting>>({});

  useEffect(() => {
    fetch('http://localhost:8000/predictions').then(r => r.json()).then(setGames);
    fetch('http://localhost:8000/elo-standings').then(r => r.json()).then(setStandings);
    fetch('http://localhost:8000/history?year=2026').then(r => r.json()).then(setHistory);
  }, []);

  const toggleScouting = async (gamePk: string) => {
    if (expanded === gamePk) {
      setExpanded(null);
      return;
    }
    setExpanded(gamePk);
    if (!scouting[gamePk]) {
      const res = await fetch(`http://localhost:8000/game/${gamePk}/scouting`);
      const data = await res.json();
      setScouting(s => ({ ...s, [gamePk]: data }));
    }
  };

  return (
    <div className="dashboard-container">
      <header className="header">
        <h1>MLB Prediction Engine</h1>
        <div style={{ color: 'var(--text-secondary)', marginTop: '0.5rem' }}>Autonomous Rebuild V5</div>
      </header>

      <div className="tabs">
        <button className={`tab ${activeTab === 'today' ? 'active' : ''}`} onClick={() => setActiveTab('today')}>
          <TrendingUp size={18} style={{ display: 'inline', marginRight: '8px' }} />
          Today's Games
        </button>
        <button className={`tab ${activeTab === 'standings' ? 'active' : ''}`} onClick={() => setActiveTab('standings')}>
          Elo Standings
        </button>
        <button className={`tab ${activeTab === 'history' ? 'active' : ''}`} onClick={() => setActiveTab('history')}>
          <History size={18} style={{ display: 'inline', marginRight: '8px' }} />
          Historical Audit
        </button>
      </div>

      {activeTab === 'today' && (
        <div className="grid">
          {games.map(g => (
            <div key={g.gamePk} className="card">
              <div className="match-header">
                <div className="team">
                  <img src={`https://a.espncdn.com/i/teamlogos/mlb/500/${g.v_team}.png`} alt={g.v_team} className="team-logo" onError={(e:any) => e.target.style.display='none'} />
                  <span>{g.v_team}</span>
                </div>
                <div style={{ color: 'var(--text-secondary)' }}>@</div>
                <div className="team">
                  <img src={`https://a.espncdn.com/i/teamlogos/mlb/500/${g.h_team}.png`} alt={g.h_team} className="team-logo" onError={(e:any) => e.target.style.display='none'} />
                  <span>{g.h_team}</span>
                </div>
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
                <span>{g.v_team} {(g.v_win_prob*100).toFixed(1)}%</span>
                <span>{g.h_team} {(g.h_win_prob*100).toFixed(1)}%</span>
              </div>
              
              <div className="prob-bar-container">
                <div className="prob-fill prob-away" style={{ width: `${g.v_win_prob * 100}%` }}></div>
                <div className="prob-fill prob-home" style={{ width: `${g.h_win_prob * 100}%` }}></div>
              </div>

              <div style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', display: 'flex', justifyContent: 'space-between' }}>
                <span>Vegas Home ML: {g.vegas_home_ml}</span>
                <span>Implied: {(g.vegas_implied*100).toFixed(1)}%</span>
              </div>

              <div className={`bet-badge ${g.bet.includes('BET') ? 'bet-bet' : g.bet.includes('FADE') ? 'bet-fade' : 'bet-pass'}`}>
                {g.bet} | {g.confidence} | O/U: {g.over_under}
              </div>

              <button onClick={() => toggleScouting(g.gamePk)}>
                {expanded === g.gamePk ? <><ChevronUp size={16} style={{ verticalAlign: 'middle' }}/> Hide Scouting</> : <><ChevronDown size={16} style={{ verticalAlign: 'middle' }}/> View Scouting Report</>}
              </button>

              {expanded === g.gamePk && scouting[g.gamePk] && (
                <div className="scouting-report">
                  <strong>Why the model favors this team:</strong>
                  <ul>
                    {scouting[g.gamePk].reasons.map((r, i) => <li key={i}>{r}</li>)}
                  </ul>
                  <div style={{ marginTop: '1rem' }}>
                    <strong>Pitching Matchup (Elo):</strong><br/>
                    Away P: {scouting[g.gamePk].v_p_elo.toFixed(0)} | Home P: {scouting[g.gamePk].h_p_elo.toFixed(0)}
                  </div>
                </div>
              )}
            </div>
          ))}
          {games.length === 0 && <div className="card">No games loaded. Check backend.</div>}
        </div>
      )}

      {activeTab === 'standings' && (
        <div className="card" style={{ height: '600px' }}>
          <h2>Team Composite Elo Standings</h2>
          <ResponsiveContainer width="100%" height="90%">
            <BarChart data={standings.slice(0, 15)} layout="vertical" margin={{ left: 40 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2d3748" />
              <XAxis type="number" domain={['dataMin - 50', 'dataMax + 50']} stroke="#a0aec0" />
              <YAxis dataKey="team" type="category" stroke="#a0aec0" />
              <Tooltip contentStyle={{ background: '#151b2b', border: '1px solid #2d3748' }} />
              <Bar dataKey="elo" fill="#00d26a" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {activeTab === 'history' && (
        <div className="card">
          <h2>2026 Historical Audit</h2>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', textAlign: 'left', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-color)', color: 'var(--text-secondary)' }}>
                  <th style={{ padding: '0.5rem' }}>Date</th>
                  <th style={{ padding: '0.5rem' }}>Matchup</th>
                  <th style={{ padding: '0.5rem' }}>Model Pick</th>
                  <th style={{ padding: '0.5rem' }}>Actual</th>
                  <th style={{ padding: '0.5rem' }}>Prob</th>
                </tr>
              </thead>
              <tbody>
                {history.slice(-50).map((h, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid var(--border-color)' }}>
                    <td style={{ padding: '0.5rem' }}>{h.date.split(' ')[0]}</td>
                    <td style={{ padding: '0.5rem' }}>{h.matchup}</td>
                    <td style={{ padding: '0.5rem', color: h.correct ? 'var(--accent-green)' : 'var(--accent-red)' }}>{h.pick}</td>
                    <td style={{ padding: '0.5rem' }}>{h.actual}</td>
                    <td style={{ padding: '0.5rem' }}>{(h.prob * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
