import React, { useState, useEffect } from 'react';
import axios from 'axios';
import qs from 'qs';
import './Dashboard.css';

const Dashboard = () => {
  const [requirements, setRequirements] = useState([]);
  const [overallStats, setOverallStats] = useState({ 
    total: 0, 
    approved: 0, 
    inReview: 0, 
    disapproved: 0 
  });
  const [filteredStats, setFilteredStats] = useState({ 
    total: 0, 
    approved: 0, 
    inReview: 0, 
    disapproved: 0 
  });
  const [searchQuery, setSearchQuery] = useState('');
  const [filters, setFilters] = useState({ 
    type: [], 
    status: [], 
    complexity: [], 
    priority: [] 
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [pagination, setPagination] = useState({ 
    page: 1, 
    pages: 1, 
    total: 0 
  });

  // Fetch overall statistics
  useEffect(() => {
    const fetchOverallStats = async () => {
      try {
        const response = await axios.get('http://localhost:5000/api/requirements/stats');
        setOverallStats(response.data);
      } catch (error) {
        console.error('Error fetching overall stats:', error);
        setError('Failed to load overall statistics');
        setOverallStats({ 
          total: 0, 
          approved: 0, 
          inReview: 0, 
          disapproved: 0 
        });
      }
    };
    fetchOverallStats();
  }, []);

  // Fetch filtered requirements and stats
  useEffect(() => {
    const fetchRequirements = async () => {
      setLoading(true);
      setError('');
      try {
        const response = await axios.get('http://localhost:5000/api/requirements', {
          params: {
            search: searchQuery,
            type: filters.type,
            status: filters.status,
            complexity: filters.complexity,
            priority: filters.priority,
            page: pagination.page,
            stats: true
          },
          paramsSerializer: params => qs.stringify(params, { 
            arrayFormat: 'repeat',
            skipNulls: true 
          })
        });

        // Update requirements and filtered stats
        setRequirements(response.data.requirements || []);
        setFilteredStats(response.data.stats || { 
          total: 0, 
          approved: 0, 
          inReview: 0, 
          disapproved: 0 
        });
        setPagination({
          page: response.data.page || 1,
          pages: response.data.pages || 1,
          total: response.data.total || 0
        });

      } catch (error) {
        console.error('Error fetching requirements:', error);
        setError(error.response?.data?.error || 'Failed to fetch requirements');
        setRequirements([]);
        setFilteredStats({ 
          total: 0, 
          approved: 0, 
          inReview: 0, 
          disapproved: 0 
        });
      } finally {
        setLoading(false);
      }
    };

    const debounceTimer = setTimeout(fetchRequirements, 500);
    return () => clearTimeout(debounceTimer);
  }, [searchQuery, filters, pagination.page]);

  // Handle filter changes
  const handleFilterChange = (filterType, value) => {
    setFilters(prev => ({
      ...prev,
      [filterType]: prev[filterType].includes(value)
        ? prev[filterType].filter(f => f !== value)
        : [...prev[filterType], value]
    }));
    // Reset to first page when filters change
    setPagination(prev => ({ ...prev, page: 1 }));
  };

  // Handle pagination
  const handlePageChange = (newPage) => {
    if (newPage >= 1 && newPage <= pagination.pages) {
      setPagination(prev => ({ ...prev, page: newPage }));
    }
  };

  // Export data as CSV
  const exportData = () => {
    const csvContent = [
      ['ID', 'Requirement', 'Status', 'Priority', 'Complexity', 'Author', 'Date', 'Estimated Hours'],
      ...requirements.map(req => [
        req.id,
        `"${req.requirement.replace(/"/g, '""')}"`,
        req.status,
        req.priority,
        req.complexity,
        req.author,
        new Date(req.date).toISOString().split('T')[0],
        req.estimated_time
      ])
    ].map(e => e.join(',')).join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'requirements_export.csv';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="dashboard-container">
      <header className="dashboard-header">
        <h1>REQUIREMENTS DASHBOARD</h1>
        <div className="header-controls">
          <p>Manage and track all project requirements</p>
          <button 
            className="export-button" 
            onClick={exportData}
            disabled={requirements.length === 0}
          >
            Export as CSV
          </button>
        </div>
      </header>

      <div className="stats-container">
        <div className="stats-section">
          <h2>Overall Statistics</h2>
          <div className="dashboard-stats">
            <StatCard label="Total" value={overallStats.total} />
            <StatCard label="Approved" value={overallStats.approved} type="approved" />
            <StatCard label="In Review" value={overallStats.inReview} type="review" />
            <StatCard label="Disapproved" value={overallStats.disapproved} type="disapproved" />
          </div>
        </div>

        <div className="stats-section">
          <h2>Filtered Statistics</h2>
          <div className="dashboard-stats">
            <StatCard label="Matching" value={filteredStats.total} />
            <StatCard label="Approved" value={filteredStats.approved} type="approved" />
            <StatCard label="In Review" value={filteredStats.inReview} type="review" />
            <StatCard label="Disapproved" value={filteredStats.disapproved} type="disapproved" />
          </div>
        </div>
      </div>

      <div className="dashboard-content">
        <div className="filters-sidebar">
          <div className="search-box">
            <input
              type="text"
              placeholder="Search requirements..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              disabled={loading}
            />
            {loading && <div className="spinner"></div>}
          </div>

          <FilterSection 
            title="Type"
            options={['Functional', 'Non-Functional', 'UI', 'Security', 'Performance']}
            selected={filters.type}
            onChange={(value) => handleFilterChange('type', value)}
          />

          <FilterSection
            title="Status"
            options={['Draft', 'Review', 'Approved', 'Disapproved']}
            selected={filters.status}
            onChange={(value) => handleFilterChange('status', value)}
          />

          <FilterSection
            title="Complexity"
            options={['Low', 'Moderate', 'High']}
            selected={filters.complexity}
            onChange={(value) => handleFilterChange('complexity', value)}
          />

          <FilterSection
            title="Priority"
            options={['Low', 'Medium', 'High']}
            selected={filters.priority}
            onChange={(value) => handleFilterChange('priority', value)}
          />
        </div>

        <div className="requirements-list">
          {error && (
            <div className="error-message">
              {error}
              <button 
                className="retry-button" 
                onClick={() => window.location.reload()}
              >
                Retry
              </button>
            </div>
          )}

          {loading ? (
            <div className="loading-indicator">
              <div className="spinner"></div>
              Loading requirements...
            </div>
          ) : requirements.length === 0 ? (
            <div className="empty-state">
              No requirements found matching your criteria
            </div>
          ) : (
            <>
              {requirements.map(req => (
                <RequirementCard 
                  key={req.id} 
                  requirement={req} 
                />
              ))}
              
              <div className="pagination-controls">
                <button 
                  onClick={() => handlePageChange(pagination.page - 1)}
                  disabled={pagination.page === 1 || loading}
                >
                  Previous
                </button>
                <span>
                  Page {pagination.page} of {pagination.pages}
                </span>
                <button 
                  onClick={() => handlePageChange(pagination.page + 1)}
                  disabled={pagination.page === pagination.pages || loading}
                >
                  Next
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

const StatCard = ({ label, value, type }) => (
  <div className={`stat-card ${type || ''}`}>
    <h3>{label}</h3>
    <div className="stat-value">{value}</div>
  </div>
);

const FilterSection = ({ title, options, selected, onChange }) => (
  <div className="filter-section">
    <h4>{title}</h4>
    <div className="filter-options">
      {options.map(option => (
        <label 
          key={option} 
          className="filter-option"
        >
          <input
            type="checkbox"
            checked={selected.includes(option)}
            onChange={() => onChange(option)}
            className="filter-checkbox"
          />
          <span className="filter-label">{option}</span>
        </label>
      ))}
    </div>
  </div>
);

const RequirementCard = ({ requirement }) => (
  <div className="requirement-card">
    <div className="card-header">
      <span className="requirement-id">ID: {requirement.id}</span>
      <span className={`status-badge ${requirement.status.toLowerCase()}`}>
        {requirement.status}
      </span>
      <span className={`priority-tag ${requirement.priority.toLowerCase()}`}>
        {requirement.priority}
      </span>
    </div>
    <h3 className="requirement-text">{requirement.requirement}</h3>
    <div className="categories">
      {requirement.categories?.split(', ').map(cat => (
        <span key={cat} className="category-tag">
          {cat}
        </span>
      ))}
    </div>
    <div className="card-footer">
      <span className="complexity-badge">
        {requirement.complexity} complexity
      </span>
      <span className="estimated-time">
        Est. {requirement.estimated_time}h
      </span>
      <span className="author-date">
        {requirement.author} • {new Date(requirement.date).toLocaleDateString()}
      </span>
    </div>
  </div>
);

export default Dashboard;