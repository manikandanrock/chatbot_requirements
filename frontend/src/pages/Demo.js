// demo.js
import React from 'react';
import './Demo.css';

const Demo = () => {
  return (
    <div className="demo-container">
      <header className="demo-header">
        <h1>DEMO VIDEOS</h1>
        <p>Learn more about our product features and capabilities</p>
      </header>

      <div className="demo-content">
        <div className="demo-video-placeholder">
          <span role="img" aria-label="video-icon" className="video-icon">
            🎥
          </span>
          <h2>Demo Video Coming Soon</h2>
          <p>
            This section will contain product demonstration videos to help you understand the features and capabilities.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Demo;