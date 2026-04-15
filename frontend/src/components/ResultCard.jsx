export default function ResultCard({ item, index }) {
  return (
    <div className="result-card">
      <h3>Sentence {index + 1}</h3>
      <p><strong>Intent:</strong> {item.intent}</p>
      <div>
        <strong>Slots:</strong>
        {Object.keys(item.slots).length === 0 ? (
          <p>No slots found</p>
        ) : (
          <pre>{JSON.stringify(item.slots, null, 2)}</pre>
        )}
      </div>
    </div>
  )
}