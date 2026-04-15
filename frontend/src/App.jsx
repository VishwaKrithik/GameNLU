import { useState } from 'react'
import { predictCommands } from './api/client'
import ControlPanel from './components/ControlPanel'
import ResultCard from './components/ResultCard'
import TokenInputHelp from './components/TokenInputHelp'

export default function App() {
  const [inputMode, setInputMode] = useState('raw')
  const [embeddingModel, setEmbeddingModel] = useState('sentence_transformers')
  const [slotMode, setSlotMode] = useState('rule_based')
  const [text, setText] = useState('attack the closest enemy then move forward')
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleSubmit = async () => {
    setLoading(true)
    setError('')
    setResults(null)

    try {
      let payload = {
        embedding_model: embeddingModel,
        slot_mode: slotMode,
      }

      if (inputMode === 'raw') {
        payload.raw_text = text
      } else {
        payload.tokenized_sentences = JSON.parse(text)
      }

      const data = await predictCommands(payload)
      setResults(data)
    } catch (err) {
      setError(err?.response?.data?.detail || err.message || 'Something went wrong')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-shell">
      <div className="container">
        <h1>Game Command NLU</h1>
        <p className="subtitle">
          Switch between embedding models and slot-filling strategies from one UI.
        </p>

        <ControlPanel
          embeddingModel={embeddingModel}
          setEmbeddingModel={setEmbeddingModel}
          slotMode={slotMode}
          setSlotMode={setSlotMode}
          inputMode={inputMode}
          setInputMode={setInputMode}
          onSubmit={handleSubmit}
          loading={loading}
        />

        <TokenInputHelp />

        <textarea
          className="input-box"
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={10}
          placeholder={inputMode === 'raw' ? 'Enter raw text here...' : 'Enter tokenized JSON here...'}
        />

        {error && <div className="error-box">{error}</div>}

        {results && (
          <div className="results-section">
            <div className="meta-box">
              <p><strong>Processed sentences:</strong></p>
              <pre>{JSON.stringify(results.processed_sentences, null, 2)}</pre>
              <p><strong>Embedding model:</strong> {results.embedding_model}</p>
              <p><strong>Slot mode:</strong> {results.slot_mode}</p>
            </div>

            {results.results.map((item, index) => (
              <ResultCard key={index} item={item} index={index} />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}