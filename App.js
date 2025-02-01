import React, { useState } from 'react';
import axios from 'axios';
import Results from './components/Results';

function App() {
  const [file, setFile] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    setLoading(true);
    try {
      const response = await axios.post('http://localhost:8000/api/detect-bots/', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setResults(response.data);
    } catch (error) {
      console.error('Error detecting bots:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>Bot Profile Detection</h1>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleSubmit} disabled={loading}>
        {loading ? 'Processing...' : 'Detect Bots'}
      </button>
      {results && <Results data={results} />}
    </div>
  );
}

export default App;
