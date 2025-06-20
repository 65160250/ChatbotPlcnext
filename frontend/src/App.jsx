import { useState } from 'react';
import './App.css';

function App() {
  const [message, setMessage] = useState('');
  const [reply, setReply] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async () => {
    if (!message.trim()) return;
    setLoading(true);
    setError('');
    setReply('');

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message }),
      });
      const data = await res.json();
      if (res.ok) {
        setReply(data.reply);
      } else {
        setError(data.error || 'Something went wrong');
      }
    } catch (err) {
      setError(err.message || 'Network error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>💬 PLCnext Chatbot</h1>
      <textarea
        rows="4"
        placeholder="พิมพ์คำถามเกี่ยวกับ PLCnext..."
        value={message}
        onChange={(e) => setMessage(e.target.value)}
      />
      <button onClick={handleSubmit} disabled={loading}>
        {loading ? 'กำลังตอบ...' : 'ถาม'}
      </button>

      {reply && <div className="reply"><strong>คำตอบ:</strong> {reply}</div>}
      {error && <div className="error">❌ {error}</div>}
    </div>
  );
}

export default App;
