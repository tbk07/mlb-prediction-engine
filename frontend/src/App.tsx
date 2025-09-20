import { useEffect, useState } from 'react'

interface Prediction {
  id: number
  match: string
  h_team: string
  v_team: string
  h_p: string
  v_p: string
  h_win_prob: number
  v_win_prob: number
  reasoning: string[]
}

interface ModelInfo {
  features: { name: string; description: string }[]
  algorithm: string
  training_data: string
}

function App() {
  const [predictions, setPredictions] = useState<Prediction[]>([])
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([
      fetch('http://localhost:8000/predictions').then(res => res.json()),
      fetch('http://localhost:8000/model-info').then(res => res.json())
    ]).then(([preds, info]) => {
      setPredictions(preds)
      setModelInfo(info)
      setLoading(false)
    })
  }, [])

  if (loading) return <div className="container">Loading predictions...</div>

  return (
    <div className="container">
      <header>
        <h1>MLB Prediction Engine</h1>
        <p className="subtitle">Real-time Win Probability & Scouting Reports</p>
      </header>

      <div className="match-grid">
        {predictions.map(pred => (
          <div key={pred.id} className="match-card">
            <div className="match-header">
              <div className="teams">{pred.match}</div>
              <div className="date">{new Date().toLocaleDateString()}</div>
            </div>

            <div className="pitchers">
              <div className="pitcher-info">Away: <strong>{pred.v_p}</strong></div>
              <div className="pitcher-info">Home: <strong>{pred.h_p}</strong></div>
            </div>

            <div className="probability-bar">
              <div 
                className="prob-fill away-fill" 
                style={{ width: `${pred.v_win_prob * 100}%` }}
              >
                {pred.v_team} {(pred.v_win_prob * 100).toFixed(1)}%
              </div>
              <div 
                className="prob-fill home-fill" 
                style={{ width: `${pred.h_win_prob * 100}%` }}
              >
                {pred.h_team} {(pred.h_win_prob * 100).toFixed(1)}%
              </div>
            </div>

            <div className="scouting-report">
              <h4>Scouting Report: Why {pred.h_win_prob > 0.5 ? pred.h_team : pred.v_team} is favored?</h4>
              <ul className="reasoning-list">
                {pred.reasoning.map((r, i) => <li key={i}>{r}</li>)}
              </ul>
            </div>
          </div>
        ))}
      </div>

      {modelInfo && (
        <div className="model-explanation">
          <h2>How does the model predict?</h2>
          <p>The engine uses a <strong>{modelInfo.algorithm}</strong> trained on <strong>{modelInfo.training_data}</strong>. Wins are predicted based on the following key metrics:</p>
          <ul>
            {modelInfo.features.map((f, i) => (
              <li key={i}>
                <strong>{f.name}:</strong> {f.description}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

export default App
