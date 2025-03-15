import React, { useState, useEffect } from 'react';
import axios from 'axios';
import DOMPurify from 'dompurify';
import './Upload.css';

function Upload() {
  const [requirements, setRequirements] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('upload');
  const [selectedFile, setSelectedFile] = useState(null);
  const [editingReq, setEditingReq] = useState(null);
  const [showEditModal, setShowEditModal] = useState(false);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newRequirement, setNewRequirement] = useState({
    requirement: '',
    author: '',
    priority: 'Medium',
    complexity: 'Moderate',
    estimated_time: 4,
    date: new Date().toISOString()
  });
  const [pagination, setPagination] = useState({
    page: 1,
    totalPages: 1,
    totalItems: 0
  });

  useEffect(() => {
    fetchRequirements();
  }, []);

  const fetchRequirements = async (page = 1) => {
    setLoading(true);
    try {
      const response = await axios.get('http://localhost:5000/api/requirements', {
        params: { page, per_page: 10 }
      });
      setRequirements(response.data.requirements);
      setPagination({
        page: response.data.page,
        totalPages: response.data.pages,
        totalItems: response.data.total
      });
    } catch (err) {
      setError(`Failed to fetch requirements: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async () => {
    if (!selectedFile) {
      setError('Please select a file to upload');
      return;
    }

    const allowedTypes = ['application/pdf', 'text/plain', 'text/markdown'];
    const MAX_SIZE = 10 * 1024 * 1024;

    if (!allowedTypes.includes(selectedFile.type)) {
      setError('Invalid file type. Only PDF, TXT, MD allowed');
      return;
    }

    if (selectedFile.size > MAX_SIZE) {
      setError('File size exceeds 10MB limit');
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      setLoading(true);
      await axios.post('http://localhost:5000/api/analyze', formData);
      await fetchRequirements();
      setSelectedFile(null);
    } catch (err) {
      setError(`Upload failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleStatusUpdate = async (id, status) => {
    try {
      await axios.patch(`http://localhost:5000/api/requirements/${id}/status`, { status });
      await fetchRequirements(pagination.page);
    } catch (err) {
      setError(`Status update failed: ${err.message}`);
    }
  };

  const handleCreateRequirement = async () => {
    if (!newRequirement.requirement.trim()) {
      setError('Requirement text is required');
      return;
    }

    try {
      await axios.post('http://localhost:5000/api/requirements', {
        ...newRequirement,
        date: new Date(newRequirement.date).toISOString()
      });
      setShowCreateModal(false);
      setNewRequirement({
        requirement: '',
        author: '',
        priority: 'Medium',
        complexity: 'Moderate',
        estimated_time: 4,
        date: new Date().toISOString()
      });
      await fetchRequirements();
    } catch (err) {
      setError(`Creation failed: ${err.message}`);
    }
  };

  const handleUpdateRequirement = async () => {
    try {
      await axios.put(`http://localhost:5000/api/requirements/${editingReq.id}`, {
        ...editingReq,
        date: new Date(editingReq.date).toISOString(),
        categories: editingReq.categories.join(', ')
      });
      setShowEditModal(false);
      await fetchRequirements(pagination.page);
    } catch (err) {
      setError(`Update failed: ${err.message}`);
    }
  };

  const handleDeleteRequirement = async (id) => {
    if (!window.confirm('Delete this requirement permanently?')) return;
    
    try {
      await axios.delete(`http://localhost:5000/api/requirements/${id}`);
      await fetchRequirements(pagination.page);
    } catch (err) {
      setError(`Deletion failed: ${err.message}`);
    }
  };

  const sanitizeHTML = (text) => ({
    __html: DOMPurify.sanitize(text)
  });

  return (
    <div className="container">
      <h1 className="page-title">Upload & Review</h1>

      <div className="tab-container">
        <button 
          className={`tab ${activeTab === 'upload' ? 'active' : ''}`} 
          onClick={() => setActiveTab('upload')}
        >
          Upload Documents
        </button>
        <button 
          className={`tab ${activeTab === 'review' ? 'active' : ''}`} 
          onClick={() => setActiveTab('review')}
        >
          Review Requirements ({pagination.totalItems})
        </button>
        <button 
          className="floating-action-btn" 
          onClick={() => setShowCreateModal(true)}
        >
          +
        </button>
      </div>

      {activeTab === 'upload' && (
        <div className="upload-section">
          <div className="file-upload-box">
            <input
              type="file"
              onChange={(e) => setSelectedFile(e.target.files[0])}
              accept=".pdf,.txt,.md"
            />
            {selectedFile && (
              <div className="file-info">
                {selectedFile.name} - {(selectedFile.size / 1024 / 1024).toFixed(2)}MB
              </div>
            )}
            <button 
              onClick={handleFileUpload}
              disabled={!selectedFile || loading}
            >
              {loading ? 'Processing...' : 'Upload & Analyze'}
            </button>
            {error && <div className="error-message">{error}</div>}
          </div>
        </div>
      )}

      {activeTab === 'review' && (
        <>
          <div className="requirements-grid">
            {requirements.map((req) => (
              <div key={req.id} className="requirement-card">
                <div className="card-header">
                  <span className="requirement-id">ID: {req.id}</span>
                  <span className="status-badge" style={{ 
                    backgroundColor: {
                      Approved: '#4CAF50',
                      Disapproved: '#f44336',
                      Review: '#ff9800',
                      Draft: '#607d8b'
                    }[req.status]
                  }}>
                    {req.status}
                  </span>
                  <span className="priority-tag">Priority: {req.priority}</span>
                  <span className="complexity-tag">Complexity: {req.complexity}</span>
                </div>
                <div className="card-content">
                  <h3 dangerouslySetInnerHTML={sanitizeHTML(req.requirement)} />
                  <div className="meta-info">
                    <span>📅 {new Date(req.date).toLocaleDateString()}</span>
                    <span>👤 {req.author}</span>
                    <span>⏱️ {req.estimated_time}h</span>
                  </div>
                  <div className="categories">
                    {req.categories.split(', ').map((cat) => (
                      <span key={cat} className="category-tag">{cat}</span>
                    ))}
                  </div>
                </div>
                <div className="card-actions">
                  <button onClick={() => handleStatusUpdate(req.id, 'Approved')}>Approve</button>
                  <button onClick={() => handleStatusUpdate(req.id, 'Disapproved')}>Reject</button>
                  <button onClick={() => {
                    setEditingReq({
                      ...req,
                      categories: req.categories.split(', ')
                    });
                    setShowEditModal(true);
                  }}>
                    Edit
                  </button>
                  <button className="delete-btn" onClick={() => handleDeleteRequirement(req.id)}>
                    Delete
                  </button>
                </div>
              </div>
            ))}
          </div>

          <div className="pagination-controls">
            <button
              onClick={() => fetchRequirements(pagination.page - 1)}
              disabled={pagination.page === 1}
            >
              Previous
            </button>
            <span>Page {pagination.page} of {pagination.totalPages}</span>
            <button
              onClick={() => fetchRequirements(pagination.page + 1)}
              disabled={pagination.page === pagination.totalPages}
            >
              Next
            </button>
          </div>
        </>
      )}

      {/* Edit Modal */}
      {showEditModal && (
        <div className="modal-overlay">
          <div className="modal-content">
            <h2>Edit Requirement</h2>
            <div className="form-grid">
              <div className="form-group">
                <label>Requirement Text</label>
                <textarea
                  value={editingReq?.requirement || ''}
                  onChange={(e) => setEditingReq({ ...editingReq, requirement: e.target.value })}
                />
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label>Author</label>
                  <input
                    type="text"
                    value={editingReq?.author || ''}
                    onChange={(e) => setEditingReq({ ...editingReq, author: e.target.value })}
                  />
                </div>
                <div className="form-group">
                  <label>Date & Time</label>
                  <input
                    type="datetime-local"
                    value={new Date(editingReq?.date).toISOString().slice(0, 16)}
                    onChange={(e) => setEditingReq({ ...editingReq, date: new Date(e.target.value).toISOString() })}
                  />
                </div>
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label>Priority</label>
                  <select
                    value={editingReq?.priority || 'Medium'}
                    onChange={(e) => setEditingReq({ ...editingReq, priority: e.target.value })}
                  >
                    {['High', 'Medium', 'Low'].map(option => (
                      <option key={option} value={option}>{option}</option>
                    ))}
                  </select>
                </div>
                <div className="form-group">
                  <label>Complexity</label>
                  <select
                    value={editingReq?.complexity || 'Moderate'}
                    onChange={(e) => setEditingReq({ ...editingReq, complexity: e.target.value })}
                  >
                    {['High', 'Moderate', 'Low'].map(option => (
                      <option key={option} value={option}>{option}</option>
                    ))}
                  </select>
                </div>
                <div className="form-group">
                  <label>Estimated Hours</label>
                  <input
                    type="number"
                    value={editingReq?.estimated_time || 4}
                    onChange={(e) => setEditingReq({ ...editingReq, estimated_time: parseInt(e.target.value) })}
                  />
                </div>
              </div>

              <div className="form-group">
                <label>Categories</label>
                <div className="category-grid">
                  {['Functional', 'Non-Functional', 'UI', 'Security', 'Performance'].map(cat => (
                    <label key={cat} className="category-option">
                      <input
                        type="checkbox"
                        checked={editingReq?.categories?.includes(cat)}
                        onChange={(e) => {
                          const categories = e.target.checked
                            ? [...editingReq.categories, cat]
                            : editingReq.categories.filter(c => c !== cat);
                          setEditingReq({ ...editingReq, categories });
                        }}
                      />
                      {cat}
                    </label>
                  ))}
                </div>
              </div>
            </div>

            <div className="form-actions">
              <button onClick={() => setShowEditModal(false)}>Cancel</button>
              <button className="save-btn" onClick={handleUpdateRequirement}>
                Save Changes
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Create Modal */}
      {showCreateModal && (
        <div className="modal-overlay">
          <div className="modal-content">
            <h2>Create New Requirement</h2>
            <div className="form-grid">
              <div className="form-group">
                <label>Requirement Text</label>
                <textarea
                  value={newRequirement.requirement}
                  onChange={(e) => setNewRequirement({ ...newRequirement, requirement: e.target.value })}
                />
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label>Author</label>
                  <input
                    type="text"
                    value={newRequirement.author}
                    onChange={(e) => setNewRequirement({ ...newRequirement, author: e.target.value })}
                  />
                </div>
                <div className="form-group">
                  <label>Date & Time</label>
                  <input
                    type="datetime-local"
                    value={new Date(newRequirement.date).toISOString().slice(0, 16)}
                    onChange={(e) => setNewRequirement({ ...newRequirement, date: new Date(e.target.value).toISOString() })}
                  />
                </div>
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label>Priority</label>
                  <select
                    value={newRequirement.priority}
                    onChange={(e) => setNewRequirement({ ...newRequirement, priority: e.target.value })}
                  >
                    {['High', 'Medium', 'Low'].map(option => (
                      <option key={option} value={option}>{option}</option>
                    ))}
                  </select>
                </div>
                <div className="form-group">
                  <label>Complexity</label>
                  <select
                    value={newRequirement.complexity}
                    onChange={(e) => setNewRequirement({ ...newRequirement, complexity: e.target.value })}
                  >
                    {['High', 'Moderate', 'Low'].map(option => (
                      <option key={option} value={option}>{option}</option>
                    ))}
                  </select>
                </div>
                <div className="form-group">
                  <label>Estimated Hours</label>
                  <input
                    type="number"
                    value={newRequirement.estimated_time}
                    onChange={(e) => setNewRequirement({ ...newRequirement, estimated_time: parseInt(e.target.value) })}
                  />
                </div>
              </div>

              <div className="form-group">
                <label>Categories</label>
                <div className="category-grid">
                  {['Functional', 'Non-Functional', 'UI', 'Security', 'Performance'].map(cat => (
                    <label key={cat} className="category-option">
                      <input
                        type="checkbox"
                        checked={newRequirement.categories?.includes(cat)}
                        onChange={(e) => {
                          const categories = e.target.checked
                            ? [...(newRequirement.categories || []), cat]
                            : (newRequirement.categories || []).filter(c => c !== cat);
                          setNewRequirement({ ...newRequirement, categories });
                        }}
                      />
                      {cat}
                    </label>
                  ))}
                </div>
              </div>
            </div>

            <div className="form-actions">
              <button onClick={() => setShowCreateModal(false)}>Cancel</button>
              <button className="create-btn" onClick={handleCreateRequirement}>
                Create Requirement
              </button>
            </div>
          </div>
        </div>
      )}

      {loading && <div className="loading-overlay">Processing...</div>}
    </div>
  );
}

export default Upload;