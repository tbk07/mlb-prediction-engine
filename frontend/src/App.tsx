import { useEffect, useState } from 'react';
import { ChevronDown, ChevronUp, TrendingUp, History } from 'lucide-react';

interface Game {
  gamePk: string;
  v_team: string;
  h_team: string;
  v_team_info: TeamInfo;
  h_team_info: TeamInfo;
  date: string;
  game_datetime: string;
  status: string;
  venue?: string;
  v_score?: number | null;
  h_score?: number | null;
  v_win_prob: number | null;
  h_win_prob: number | null;
  vegas_home_ml: number | null;
  vegas_away_ml: number | null;
  vegas_implied: number | null;
  odds_available: boolean;
  edge: number | null;
  bet: string;
  confidence: string;
  over_under: number | null;
}

interface TeamInfo {
  abbr: string;
  full_name: string;
  mlb_logo?: string | null;
  espn_logo: string;
}

interface HistoryRow {
  gamePk: string;
  date: string;
  game_datetime: string;
  status: string;
  matchup: string;
  v_team: string;
  h_team: string;
  v_team_info: TeamInfo;
  h_team_info: TeamInfo;
  pick: string | null;
  actual: string | null;
  correct: boolean | null;
  prob: number | null;
  vegas_home_ml: number | null;
  vegas_away_ml: number | null;
}

interface Scouting {
  reasons: string[];
  h_p_elo: number;
  v_p_elo: number;
  h_bp_elo: number;
  v_bp_elo: number;
  h_off_elo: number;
  v_off_elo: number;
  park_factor: number;
}

export default function App() {
  const [activeTab, setActiveTab] = useState<'today' | 'standings' | 'history'>('today');
  const [games, setGames] = useState<Game[]>([]);
  const [standings, setStandings] = useState<any[]>([]);
  const [history, setHistory] = useState<HistoryRow[]>([]);
  const [historySummary, setHistorySummary] = useState<any>(null);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [scouting, setScouting] = useState<Record<string, Scouting>>({});

  useEffect(() => {
    fetch('/predictions').then(r => r.json()).then(setGames);
    fetch('/elo-standings').then(r => r.json()).then(setStandings);
    fetch('/history?year=2026').then(r => r.json()).then(data => {
      setHistory(data.history || []);
      setHistorySummary(data.summary || null);
    });
  }, []);

  const toggleScouting = async (gamePk: string) => {
    if (expanded === gamePk) {
      setExpanded(null);
      return;
    }
    setExpanded(gamePk);
    if (!scouting[gamePk]) {
      const res = await fetch(`/game/${gamePk}/scouting`);
      const data = await res.json();
      setScouting(s => ({ ...s, [gamePk]: data }));
    }
  };

  const formatGameDateTime = (value?: string) => {
    if (!value) return 'Time TBD';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value.split('T')[0] || value;
    return new Intl.DateTimeFormat(undefined, {
      weekday: 'short',
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
    }).format(date);
  };

  const formatDate = (value?: string) => {
    if (!value) return '';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value.split('T')[0] || value;
    return new Intl.DateTimeFormat(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    }).format(date);
  };

  const formatProb = (value: number | null) => value == null ? 'Pending' : `${(value * 100).toFixed(1)}%`;

  const TeamLogo = ({ team }: { team: TeamInfo }) => (
    <img
      src={team.mlb_logo || team.espn_logo}
      alt={team.full_name}
      className="team-logo"
      onError={(event) => {
        const img = event.currentTarget;
        if (img.src !== team.espn_logo) {
          img.src = team.espn_logo;
        } else {
          img.style.display = 'none';
        }
      }}
    />
  );

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
              <div className="game-meta">
                <span>{formatGameDateTime(g.game_datetime)}</span>
                <span>{g.status}</span>
              </div>
              {g.venue && <div className="venue">{g.venue}</div>}

              <div className="match-header">
                <div className="team">
                  <TeamLogo team={g.v_team_info} />
                  <span className="team-name">{g.v_team_info.full_name}</span>
                  <span className="team-abbr">{g.v_team}</span>
                </div>
                <div style={{ color: 'var(--text-secondary)' }}>@</div>
                <div className="team">
                  <TeamLogo team={g.h_team_info} />
                  <span className="team-name">{g.h_team_info.full_name}</span>
                  <span className="team-abbr">{g.h_team}</span>
                </div>
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
                <span>{g.v_team} {formatProb(g.v_win_prob)}</span>
                <span>{g.h_team} {formatProb(g.h_win_prob)}</span>
              </div>
              
              <div className="prob-bar-container">
                <div className="prob-fill prob-away" style={{ width: `${(g.v_win_prob ?? 0.5) * 100}%` }}></div>
                <div className="prob-fill prob-home" style={{ width: `${(g.h_win_prob ?? 0.5) * 100}%` }}></div>
              </div>

              <div style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', display: 'flex', justifyContent: 'space-between' }}>
                {g.odds_available ? (
                  <>
                    <span>Away ML: {g.vegas_away_ml}</span>
                    <span>Home ML: {g.vegas_home_ml} | Implied: {formatProb(g.vegas_implied)}</span>
                  </>
                ) : (
                  <span>Market odds unavailable</span>
                )}
              </div>

              <div className={`bet-badge ${g.bet.includes('BET') ? 'bet-bet' : g.bet.includes('FADE') ? 'bet-fade' : 'bet-pass'}`}>
                {g.bet} | {g.confidence}{g.over_under != null ? ` | O/U: ${g.over_under}` : ''}
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
                  <div style={{ marginTop: '1rem', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', fontSize: '0.85rem' }}>
                    <div>
                      <strong>Away Elo Ratings:</strong><br/>
                      Pitching: {(scouting[g.gamePk].v_p_elo || 0).toFixed(0)}<br/>
                      Bullpen: {(scouting[g.gamePk].v_bp_elo || 0).toFixed(0)}<br/>
                      Offense: {(scouting[g.gamePk].v_off_elo || 0).toFixed(0)}
                    </div>
                    <div>
                      <strong>Home Elo Ratings:</strong><br/>
                      Pitching: {(scouting[g.gamePk].h_p_elo || 0).toFixed(0)}<br/>
                      Bullpen: {(scouting[g.gamePk].h_bp_elo || 0).toFixed(0)}<br/>
                      Offense: {(scouting[g.gamePk].h_off_elo || 0).toFixed(0)}
                    </div>
                  </div>
                </div>
              )}
            </div>
          ))}
          {games.length === 0 && (
            <div className="card" style={{ textAlign: 'center', padding: '2rem' }}>
              <h3>No Games Scheduled Today</h3>
              <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem', lineHeight: '1.5' }}>
                There are no MLB games currently loaded. If it's mid-July, this is due to the <strong>MLB All-Star Break</strong>!
                <br /><br />
                During the break, there are no active regular-season games, so The Odds API does not return any active moneyline or totals markets. Once the second half of the season resumes, your odds and predictions will automatically reappear here.
              </p>
            </div>
          )}
        </div>
      )}

      {activeTab === 'standings' && (
        <div className="card">
          <h2>Detailed Elo Standings</h2>
          <div className="table-summary">Real-time ratings based on rolling performance and platoon adjustments.</div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', textAlign: 'left', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-color)', color: 'var(--text-secondary)' }}>
                  <th style={{ padding: '0.8rem' }}>Team</th>
                  <th style={{ padding: '0.8rem' }}>Composite</th>
                  <th style={{ padding: '0.8rem' }}>Offense</th>
                  <th style={{ padding: '0.8rem' }}>Defense</th>
                  <th style={{ padding: '0.8rem' }}>Starter</th>
                  <th style={{ padding: '0.8rem' }}>Bullpen</th>
                </tr>
              </thead>
              <tbody>
                {standings.map((s, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid var(--border-color)' }}>
                    <td style={{ padding: '0.8rem' }}>
                      <div className="history-matchup">
                        <TeamLogo team={s.team_info} /> {s.team_info.full_name}
                      </div>
                    </td>
                    <td style={{ padding: '0.8rem', fontWeight: 'bold', color: 'var(--accent-green)' }}>{(s.elo || 0).toFixed(0)}</td>
                    <td style={{ padding: '0.8rem' }}>{(s.off_elo || 0).toFixed(0)}</td>
                    <td style={{ padding: '0.8rem' }}>{(s.def_elo || 0).toFixed(0)}</td>
                    <td style={{ padding: '0.8rem' }}>{(s.p_elo || 0).toFixed(0)}</td>
                    <td style={{ padding: '0.8rem' }}>{(s.bp_elo || 0).toFixed(0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {activeTab === 'history' && (
        <div className="card">
          <h2>2026 Historical Audit</h2>
          {historySummary && (
            <div className="accuracy-summary" style={{ marginBottom: '2rem', padding: '1.5rem', background: 'rgba(255,255,255,0.03)', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
              <div style={{ display: 'flex', gap: '2rem', flexWrap: 'wrap' }}>
                <div>
                  <div style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Total Accuracy</div>
                  <div style={{ fontSize: '2rem', fontWeight: 'bold', color: 'var(--accent-green)' }}>{(historySummary.total_accuracy * 100).toFixed(1)}%</div>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>on {historySummary.total_games} games</div>
                </div>
                <div style={{ flex: 1, minWidth: '300px' }}>
                  <div style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>Weekly Breakdown</div>
                  <div style={{ display: 'flex', gap: '0.5rem', overflowX: 'auto', paddingBottom: '0.5rem' }}>
                    {historySummary.weekly.map((w: any, i: number) => (
                      <div key={i} style={{ minWidth: '100px', padding: '0.5rem', background: 'rgba(255,255,255,0.05)', borderRadius: '4px', textAlign: 'center' }}>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>{w.week.split(' ')[1]}</div>
                        <div style={{ fontWeight: 'bold', color: w.accuracy > 0.55 ? 'var(--accent-green)' : 'inherit' }}>{(w.accuracy * 100).toFixed(0)}%</div>
                        <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>{w.games} games</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
          <div className="table-summary">Showing {history.length} MLB games from the 2026 season schedule.</div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', textAlign: 'left', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-color)', color: 'var(--text-secondary)' }}>
                  <th style={{ padding: '0.5rem' }}>Date / Time</th>
                  <th style={{ padding: '0.5rem' }}>Matchup</th>
                  <th style={{ padding: '0.5rem' }}>Status</th>
                  <th style={{ padding: '0.5rem' }}>Model Pick</th>
                  <th style={{ padding: '0.5rem' }}>Actual</th>
                  <th style={{ padding: '0.5rem' }}>Prob</th>
                </tr>
              </thead>
              <tbody>
                {history.map((h, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid var(--border-color)' }}>
                    <td style={{ padding: '0.5rem' }}>{formatDate(h.game_datetime)}<br/><span className="muted">{formatGameDateTime(h.game_datetime).split(',').slice(-1).join(',').trim()}</span></td>
                    <td style={{ padding: '0.5rem' }}>
                      <div className="history-matchup">
                        <TeamLogo team={h.v_team_info} /> {h.v_team_info.full_name} @ <TeamLogo team={h.h_team_info} /> {h.h_team_info.full_name}
                      </div>
                    </td>
                    <td style={{ padding: '0.5rem' }}>{h.status}</td>
                    <td style={{ padding: '0.5rem', color: h.correct == null ? 'var(--text-secondary)' : h.correct ? 'var(--accent-green)' : 'var(--accent-red)' }}>{h.pick ?? 'Pending'}</td>
                    <td style={{ padding: '0.5rem' }}>{h.actual ?? 'Pending'}</td>
                    <td style={{ padding: '0.5rem' }}>{formatProb(h.prob)}</td>
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
