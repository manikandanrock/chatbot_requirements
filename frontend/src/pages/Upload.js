import React, { useState, useEffect } from 'react';
import axios from 'axios';
import DOMPurify from 'dompurify';
import './Upload.css';

const API_BASE_URL = 'http://localhost:5000';

function Upload() {
  const [requirements, setRequirements] = useState([]);
  const [loading, setLoading] = useState({ general: false, upload: false, project: false });
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState(localStorage.getItem('activeTab') || 'projects');
  const [selectedFile, setSelectedFile] = useState(null);
  const [showEditModal, setShowEditModal] = useState(false);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newProjectHourlyRate, setNewProjectHourlyRate] = useState(30);
  const [renameProjectHourlyRate, setRenameProjectHourlyRate] = useState(30);
  const [newRequirement, setNewRequirement] = useState({
    requirement: '',
    author: '',
    priority: 'Medium',
    complexity: 'Moderate',
    estimated_time: 4,
    date: new Date().toISOString(),
    categories: [],
  });
  const [editingReq, setEditingReq] = useState({
    requirement: '',
    author: '',
    priority: 'Medium',
    complexity: 'Moderate',
    estimated_time: 4,
    date: new Date().toISOString(),
    categories: [],
  });
  const [projects, setProjects] = useState([]);
  const [selectedProjectId, setSelectedProjectId] = useState(() => {
    const savedId = localStorage.getItem('selectedProjectId');
    return savedId ? parseInt(savedId, 10) : null;
  });
  const [showProjectModal, setShowProjectModal] = useState(false);
  const [newProjectName, setNewProjectName] = useState('');
  const [newProjectDescription, setNewProjectDescription] = useState('');
  const [showRenameModal, setShowRenameModal] = useState(false);
  const [renameProjectId, setRenameProjectId] = useState(null);
  const [renameProjectName, setRenameProjectName] = useState('');
  const [renameProjectDescription, setRenameProjectDescription] = useState('');

  useEffect(() => {
    if (selectedProjectId) {
      localStorage.setItem('selectedProjectId', selectedProjectId);
    } else {
      localStorage.removeItem('selectedProjectId');
    }
  }, [selectedProjectId]);

  

  useEffect(() => {
    localStorage.setItem('activeTab', activeTab);
  }, [activeTab]);

  useEffect(() => {
    const fetchProjectsAndRequirements = async () => {
      setLoading(prev => ({ ...prev, general: true }));
      try {
        const projectsResponse = await axios.get(`${API_BASE_URL}/api/projects`);
        setProjects(projectsResponse.data);

        if (selectedProjectId) {
          const requirementsResponse = await axios.get(
            `${API_BASE_URL}/api/projects/${selectedProjectId}/requirements`
          );
          const requirements = requirementsResponse.data.map(req => ({
            ...req,
            categories: typeof req.categories === 'string' ? req.categories.split(', ') : req.categories || [],
          }));
          setRequirements(requirements);
        }
      } catch (err) {
        setError('Failed to fetch data');
      } finally {
        setLoading(prev => ({ ...prev, general: false }));
      }
    };
    fetchProjectsAndRequirements();
  }, [selectedProjectId]);

  useEffect(() => {
    if (!selectedProjectId) {
      setRequirements([]);
      setActiveTab('projects');
    }
  }, [selectedProjectId]);

  const handleFileUpload = async () => {
    if (!selectedFile) {
      setError('Please select a file to upload');
      return;
    }
    if (!selectedProjectId) {
      setError('Please select a project first');
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('project_id', selectedProjectId);

    setLoading(prev => ({ ...prev, upload: true }));
    try {
      const response = await axios.post(`${API_BASE_URL}/api/analyze`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      const requirements = response.data.requirements.map(req => ({
        ...req,
        categories: typeof req.categories === 'string' ? req.categories.split(', ') : req.categories || [],
      }));
      setRequirements(requirements);
      setSelectedFile(null);
      setError(null);
      alert('File uploaded successfully!');
    } catch (err) {
      setError('File upload failed');
    } finally {
      setLoading(prev => ({ ...prev, upload: false }));
    }
  };

  const createProject = async () => {
    if (!newProjectName.trim()) {
      setError('Project name is required');
      return;
    }

    setLoading(prev => ({ ...prev, project: true }));
    try {
      const response = await axios.post(`${API_BASE_URL}/api/projects`, {
        name: newProjectName,
        description: newProjectDescription,
        hourly_rate: newProjectHourlyRate,
      });
      setProjects([...projects, response.data]);
      setShowProjectModal(false);
      setNewProjectName('');
      setNewProjectDescription('');
      setSelectedProjectId(response.data.id);
    } catch (err) {
      setError('Failed to create project');
    } finally {
      setLoading(prev => ({ ...prev, project: false }));
    }
  };

  const renameProject = async () => {
    if (!renameProjectName.trim()) {
      setError('Project name is required');
      return;
    }

    setLoading(prev => ({ ...prev, project: true }));
    try {
      await axios.put(`${API_BASE_URL}/api/projects/${renameProjectId}`, {
        name: renameProjectName,
        description: renameProjectDescription,
        hourly_rate: renameProjectHourlyRate,
      });

      const updatedProjects = projects.map(project =>
        project.id === renameProjectId
          ? { 
              ...project, 
              name: renameProjectName,
              description: renameProjectDescription,
              hourly_rate: renameProjectHourlyRate
            }
          : project
      );
      setProjects(updatedProjects);
      setShowRenameModal(false);
    } catch (err) {
      setError('Failed to update project');
    } finally {
      setLoading(prev => ({ ...prev, project: false }));
    }
  };

  const deleteProject = async (projectId) => {
    if (!window.confirm('Delete this project and all requirements?')) return;

    setLoading(prev => ({ ...prev, project: true }));
    try {
      await axios.delete(`${API_BASE_URL}/api/projects/${projectId}`);
      setProjects(projects.filter(project => project.id !== projectId));
      setSelectedProjectId(null);
    } catch (err) {
      setError('Failed to delete project');
    } finally {
      setLoading(prev => ({ ...prev, project: false }));
    }
  };

  const deleteAllRequirements = async () => {
    if (!window.confirm('Delete all requirements for this project?')) return;

    setLoading(prev => ({ ...prev, general: true }));
    try {
      await axios.delete(`${API_BASE_URL}/api/projects/${selectedProjectId}/requirements`);
      setRequirements([]);
    } catch (err) {
      setError('Failed to delete requirements');
    } finally {
      setLoading(prev => ({ ...prev, general: false }));
    }
  };

  const handleStatusUpdate = async (id, status) => {
    try {
      await axios.patch(`${API_BASE_URL}/api/requirements/${id}/status`, { status });
      setRequirements(prev => prev.map(req => req.id === id ? { ...req, status } : req));
    } catch (err) {
      setError('Status update failed');
    }
  };

  const handleCreateRequirement = async () => {
    if (!newRequirement.requirement.trim()) {
      setError('Requirement text required');
      return;
    }

    try {
      const aiResponse = await axios.post(`${API_BASE_URL}/api/classify`, {
        text: newRequirement.requirement,
      });

      const { categories, priority, complexity } = aiResponse.data;
      const response = await axios.post(
        `${API_BASE_URL}/api/projects/${selectedProjectId}/requirements`,
        {
          ...newRequirement,
          categories: categories || [],
          priority: priority || 'Medium',
          complexity: complexity || 'Moderate',
          date: new Date(newRequirement.date).toISOString(),
        }
      );

      const newReq = {
        ...response.data,
        categories: typeof response.data.categories === 'string' 
          ? response.data.categories.split(', ') 
          : response.data.categories || [],
      };

      setRequirements(prev => [...prev, newReq]);
      setShowCreateModal(false);
      setNewRequirement({
        requirement: '',
        author: '',
        priority: 'Medium',
        complexity: 'Moderate',
        estimated_time: 4,
        date: new Date().toISOString(),
        categories: [],
      });
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to create requirement');
    }
  };

  const handleUpdateRequirement = async () => {
    if (!editingReq.requirement.trim()) {
      setError('Requirement text required');
      return;
    }

    try {
      await axios.put(`${API_BASE_URL}/api/requirements/${editingReq.id}`, {
        ...editingReq,
        date: new Date(editingReq.date).toISOString(),
        categories: editingReq.categories.join(', '),
      });

      setRequirements(prev => prev.map(req => req.id === editingReq.id ? editingReq : req));
      setShowEditModal(false);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to update requirement');
    }
  };

  const handleDeleteRequirement = async (id) => {
    if (!window.confirm('Delete this requirement permanently?')) return;

    try {
      await axios.delete(`${API_BASE_URL}/api/requirements/${id}`);
      setRequirements(prev => prev.filter(req => req.id !== id));
    } catch (err) {
      setError('Deletion failed');
    }
  };

  const sanitizeHTML = (text) => ({
    __html: DOMPurify.sanitize(text),
  });

  const selectedProject = projects.find(p => p.id === selectedProjectId);
  const projectHourlyRate = selectedProject?.hourly_rate || 30;

  return (
    <div className="container">
      <h1 className="page-title">Upload & Review</h1>

      <div className="tab-container">
        <button className={`tab ${activeTab === 'projects' ? 'active' : ''}`} onClick={() => setActiveTab('projects')}>
          Projects
        </button>
        {selectedProjectId && (
          <>
            <button className={`tab ${activeTab === 'upload' ? 'active' : ''}`} onClick={() => setActiveTab('upload')}>
              Upload Documents
            </button>
            <button className={`tab ${activeTab === 'review' ? 'active' : ''}`} onClick={() => setActiveTab('review')}>
              Review Requirements ({requirements.length})
            </button>
          </>
        )}
        {activeTab !== 'projects' && (
          <button className="floating-action-btn" onClick={() => setShowCreateModal(true)}>
            +
          </button>
        )}
      </div>

      {activeTab === 'projects' && (
  <div className="projects-section">
    <button onClick={() => setShowProjectModal(true)}>Create New Project</button>
    {projects.map(project => {
      // Calculate requirements count and cost dynamically
      const projectRequirements = requirements.filter(req => req.project_id === project.id);
      const totalCost = projectRequirements.reduce((sum, req) => 
        sum + (req.estimated_time * project.hourly_rate), 0);
      
      return (
        <div
          key={project.id}
          className={`project-item ${selectedProjectId === project.id ? 'selected' : ''}`}
          onClick={() => setSelectedProjectId(project.id)}
        >
          <div className="project-header">
            <div className="project-name">{project.name}</div>
            <div className="project-rate">${project.hourly_rate}/h</div>
          </div>
          

          <div className="project-description">{project.description}</div>
          {/*
          <div className="project-meta">
            <span>Requirements: {projectRequirements.length}</span>
            <span>Total Cost: ${totalCost.toFixed(2)}</span>
          </div>
          */}
          <div className="project-actions">
            <button onClick={(e) => {
              e.stopPropagation();
              setRenameProjectId(project.id);
              setRenameProjectName(project.name);
              setRenameProjectDescription(project.description || '');
              setRenameProjectHourlyRate(project.hourly_rate);
              setShowRenameModal(true);
            }}>
              Edit
            </button>
            <button onClick={(e) => {
              e.stopPropagation();
              deleteProject(project.id);
            }}>
              Delete
            </button>
          </div>
        </div>
      );
    })}
  </div>
)}

      {activeTab === 'upload' && selectedProjectId && (
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
            <button onClick={handleFileUpload} disabled={!selectedFile || loading.upload}>
              {loading.upload ? 'Processing...' : 'Upload & Analyze'}
            </button>
            {error && <div className="error-message">{error}</div>}
          </div>
        </div>
      )}

      {activeTab === 'review' && selectedProjectId && (
        <div className="requirements-grid">
          <div className="project-summary">
            <div className="total-cost">
              Total Project Cost: $
              {requirements
                .reduce((sum, req) => sum + (req.estimated_time * projectHourlyRate), 0)
                .toFixed(2)}
            </div>
            <div className="hourly-rate">Hourly Rate: ${projectHourlyRate.toFixed(2)}</div>
          </div>
          
          {requirements.map(req => (
            <div key={req.id} className="requirement-card">
              <div className="card-header">
                <span className="requirement-id">ID: {req.id}</span>
                <span className="status-badge" style={{
                  backgroundColor: {
                    Approved: '#4CAF50',
                    Disapproved: '#f44336',
                    Review: '#ff9800',
                    Draft: '#607d8b',
                  }[req.status],
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
                  <span>💰 ${(req.estimated_time * projectHourlyRate).toFixed(2)}</span>
                </div>
                <div className="categories">
                  {(req.categories ? (typeof req.categories === 'string' ? 
                    req.categories.split(', ') : req.categories) : []).map(cat => (
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
                    categories: typeof req.categories === 'string' ? 
                      req.categories.split(', ') : req.categories || [],
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
      )}

{showProjectModal && (
  <div className="modal-overlay">
    <div className="modal-content">
      <h2>Create New Project</h2>
      <div className="form-grid">
        <div className="form-group">
          <label>Project Name</label>
          <input
            type="text"
            value={newProjectName}
            onChange={(e) => setNewProjectName(e.target.value)}
          />
        </div>
        <div className="form-group">
          <label>Description</label>
          <textarea
            value={newProjectDescription}
            onChange={(e) => setNewProjectDescription(e.target.value)}
          />
        </div>
        <div className="form-group">
          <label>Hourly Rate ($)</label>
          <input
            type="number"
            step="0.01"
            min="1"
            value={newProjectHourlyRate}
            onChange={(e) => setNewProjectHourlyRate(parseFloat(e.target.value) || 30)}
          />
        </div>
      </div>
      <div className="form-actions">
        <button onClick={() => setShowProjectModal(false)}>Cancel</button>
        <button className="create-btn" onClick={createProject}>
          Create Project
        </button>
      </div>
    </div>
  </div>
)}

{showRenameModal && (
  <div className="modal-overlay">
    <div className="modal-content">
      <h2>Edit Project</h2>
      <div className="form-grid">
        <div className="form-group">
          <label>Project Name</label>
          <input
            type="text"
            value={renameProjectName}
            onChange={(e) => setRenameProjectName(e.target.value)}
          />
        </div>
        <div className="form-group">
          <label>Description</label>
          <textarea
            value={renameProjectDescription}
            onChange={(e) => setRenameProjectDescription(e.target.value)}
          />
        </div>
        <div className="form-group">
          <label>Hourly Rate ($)</label>
          <input
            type="number"
            step="0.01"
            min="1"
            value={renameProjectHourlyRate}
            onChange={(e) => setRenameProjectHourlyRate(parseFloat(e.target.value))}
          />
        </div>
      </div>
      <div className="form-actions">
        <button onClick={() => setShowRenameModal(false)}>Cancel</button>
        <button className="update-btn" onClick={renameProject}>
          Update Project
        </button>
      </div>
    </div>
  </div>
)}

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
              value={new Date(editingReq?.date || new Date()).toISOString().slice(0, 16)}
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
              min="1"
              value={editingReq?.estimated_time || 4}
              onChange={(e) => setEditingReq({ 
                ...editingReq, 
                estimated_time: Math.max(1, parseInt(e.target.value) || 1)
              })}
            />
            <div className="cost-preview">
              Estimated Cost: $
              {((editingReq?.estimated_time || 0) * 
               (projects.find(p => p.id === selectedProjectId)?.hourly_rate || 30)).toFixed(2)}
            </div>
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
                      ? [...(editingReq.categories || []), cat]
                      : (editingReq.categories || []).filter(c => c !== cat);
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
        <button className="update-btn" onClick={handleUpdateRequirement}>
          Update Requirement
        </button>
      </div>
    </div>
  </div>
)}

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
              min="1"
              value={newRequirement.estimated_time}
              onChange={(e) => setNewRequirement({ 
                ...newRequirement, 
                estimated_time: Math.max(1, parseInt(e.target.value) || 1)
              })}
            />
            <div className="cost-preview">
              Estimated Cost: $
              {(newRequirement.estimated_time * 
               (projects.find(p => p.id === selectedProjectId)?.hourly_rate || 30)).toFixed(2)}
            </div>
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

{loading.general && <div className="loading-overlay">Processing...</div>}
</div>
);
}

export default Upload;
