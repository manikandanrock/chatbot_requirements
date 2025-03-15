import React from 'react';
import { Link } from 'react-router-dom';
import { FaFileUpload, FaBrain, FaCheckCircle, FaChartLine } from 'react-icons/fa';
import './Home.css';

function Home() {
  return (
    <div className="home-container">
      <section className="workflow-section">
        <header className="workflow-header">
          <h2 className="section-subtitle">Streamlined Requirement Processing</h2>
          <h1 className="section-title">Transform Documents into Actionable Insights</h1>
          <p className="section-description">
            Leverage AI-powered automation to extract, analyze, and manage requirements efficiently
          </p>
        </header>

        <div className="workflow-steps">
          <div className="workflow-grid">
            <div className="step-card">
              <div className="step-header">
                <div className="step-icon-container upload">
                  <FaFileUpload className="step-icon" />
                </div>
                <div className="step-progress" />
              </div>
              <h3 className="step-title">Smart Document Upload</h3>
              <p className="step-description">Support for PDF, DOCX, and Text formats</p>
              <Link to="/upload" className="step-cta">
                Begin Processing <span className="cta-arrow">→</span>
              </Link>
            </div>

            <div className="step-card">
              <div className="step-header">
                <div className="step-icon-container analyze">
                  <FaBrain className="step-icon" />
                  <span className="ai-badge">AI Engine</span>
                </div>
                <div className="step-progress" />
              </div>
              <h3 className="step-title">Intelligent Analysis</h3>
              <p className="step-description">Automated extraction & categorization</p>
            </div>

            <div className="step-card">
              <div className="step-header">
                <div className="step-icon-container validate">
                  <FaCheckCircle className="step-icon" />
                </div>
                <div className="step-progress" />
              </div>
              <h3 className="step-title">Collaborative Validation</h3>
              <p className="step-description">Interactive quality assurance portal</p>
            </div>

            <div className="step-card">
              <div className="step-header">
                <div className="step-icon-container insights">
                  <FaChartLine className="step-icon" />
                </div>
              </div>
              <h3 className="step-title">Real-time Analytics</h3>
              <p className="step-description">Comprehensive tracking & management</p>
              <Link to="/dashboard" className="step-cta">
                View Analytics <span className="cta-arrow">→</span>
              </Link>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}

export default Home;