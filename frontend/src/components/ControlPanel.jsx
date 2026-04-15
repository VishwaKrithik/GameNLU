export default function ControlPanel({
  embeddingModel,
  setEmbeddingModel,
  slotMode,
  setSlotMode,
  inputMode,
  setInputMode,
  onSubmit,
  loading,
}) {
  return (
    <div className="panel">
      <div className="row">
        <div>
          <label>Input type</label>
          <select value={inputMode} onChange={(e) => setInputMode(e.target.value)}>
            <option value="raw">Raw text</option>
            <option value="tokens">Tokenized JSON</option>
          </select>
        </div>

        <div>
          <label>Embedding model</label>
          <select value={embeddingModel} onChange={(e) => setEmbeddingModel(e.target.value)}>
            <option value="fasttext">FastText</option>
            <option value="word2vec">Word2Vec</option>
            <option value="bert">BERT</option>
            <option value="sentence_transformers">Sentence Transformers</option>
          </select>
        </div>

        <div>
          <label>Slot filling mode</label>
          <select value={slotMode} onChange={(e) => setSlotMode(e.target.value)}>
            <option value="rule_based">Rule Based</option>
            <option value="neural">Neural</option>
          </select>
        </div>
      </div>

      <button onClick={onSubmit} disabled={loading}>
        {loading ? 'Processing...' : 'Run Pipeline'}
      </button>
    </div>
  )
}