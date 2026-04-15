export default function TokenInputHelp() {
  return (
    <div className="help-box">
      <p><strong>Example raw text:</strong> attack the closest enemy then move forward</p>
      <p><strong>Example tokenized JSON:</strong></p>
      <pre>{`[
  ["attack", "closest", "enemy"],
  ["move", "forward"]
]`}</pre>
    </div>
  )
}