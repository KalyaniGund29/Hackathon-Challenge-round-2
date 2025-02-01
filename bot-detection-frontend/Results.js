import React, { useState } from 'react';
import { Bar } from 'react-chartjs-2';
import { Chart, registerables } from 'chart.js';
import './results.css';

Chart.register(...registerables);

const Results = ({ data }) => {
  const [filter, setFilter] = useState('all'); // 'all', 'bot', 'genuine'
  const [showMetrics, setShowMetrics] = useState(false); // Show/Hide Metrics
  const [showDetails, setShowDetails] = useState(false); // Show/Hide Detailed Metrics for each user

  const chartData = {
    labels: ['Bots Detected', 'Genuine Users'],
    datasets: [
      {
        label: 'Bot Detection Results',
        data: [data.bot_count, data.genuine_count],
        backgroundColor: ['#FF6384', '#36A2EB'],
      },
    ],
  };

  // Filter the details based on the selected filter
  const filteredDetails = data.details.filter((detail) => {
    if (filter === 'bot') return detail.is_bot; // Show only bots
    if (filter === 'genuine') return !detail.is_bot; // Show only genuine users
    return true; // Show all if no filter
  });

  return (
    <div>
      <h2>Detection Results</h2>

      {/* Display Evaluation Metrics Toggle */}
      <div>
        <button onClick={() => setShowMetrics(!showMetrics)}>
          {showMetrics ? 'Hide' : 'Show'} Evaluation Metrics
        </button>
      </div>
      {showMetrics && (
        <div className="metrics">
          <h3>Evaluation Metrics:</h3>
          <p>Precision: {data.metrics.precision !== null ? data.metrics.precision.toFixed(2) : 'N/A'}</p>
          <p>Recall: {data.metrics.recall !== null ? data.metrics.recall.toFixed(2) : 'N/A'}</p>
          <p>F1 Score: {data.metrics.f1_score !== null ? data.metrics.f1_score.toFixed(2) : 'N/A'}</p>
          <p>AUC-ROC: {data.metrics.auc_roc !== null ? data.metrics.auc_roc.toFixed(2) : 'N/A'}</p>
          <p>False Positive Rate: {data.metrics.false_positive_rate !== null ? data.metrics.false_positive_rate.toFixed(2) : 'N/A'}</p>
          <p>False Negative Rate: {data.metrics.false_negative_rate !== null ? data.metrics.false_negative_rate.toFixed(2) : 'N/A'}</p>
        </div>
      )}

      <Bar data={chartData} />

      {/* Buttons to filter results */}
      <div>
        <button onClick={() => setFilter('all')} className={filter === 'all' ? 'active' : ''}>All Users</button>
        <button onClick={() => setFilter('bot')} className={filter === 'bot' ? 'active' : ''}>Bots</button>
        <button onClick={() => setFilter('genuine')} className={filter === 'genuine' ? 'active' : ''}>Genuine Users</button>
      </div>

      {/* Toggle for showing detailed user-level metrics */}
      <div>
        <button onClick={() => setShowDetails(!showDetails)}>
          {showDetails ? 'Hide' : 'Show'} User-Level Details
        </button>
      </div>
      {showDetails && (
        <div>
          <h3>Details:</h3>
          <ul>
            {filteredDetails.map((detail, index) => (
              <li key={index} className={detail.is_bot ? 'bot' : 'genuine'}>
                User: {detail.username} - Confidence: {detail.confidence}% - {detail.is_bot ? 'Bot' : 'Genuine'}
                {detail.metrics && (
                  <div className="metrics-detail">
                    <p>Precision: {detail.metrics.precision ? detail.metrics.precision.toFixed(2) : 'N/A'}</p>
                    <p>Recall: {detail.metrics.recall ? detail.metrics.recall.toFixed(2) : 'N/A'}</p>
                    <p>F1 Score: {detail.metrics.f1_score ? detail.metrics.f1_score.toFixed(2) : 'N/A'}</p>
                    <p>AUC-ROC: {detail.metrics.auc_roc ? detail.metrics.auc_roc.toFixed(2) : 'N/A'}</p>
                    <p>False Positive Rate: {detail.metrics.false_positive_rate ? detail.metrics.false_positive_rate.toFixed(2) : 'N/A'}</p>
                    <p>False Negative Rate: {detail.metrics.false_negative_rate ? detail.metrics.false_negative_rate.toFixed(2) : 'N/A'}</p>
                  </div>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default Results;
