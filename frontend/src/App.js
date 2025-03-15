// src/App.js
import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, Outlet } from 'react-router-dom';
import Home from './pages/Home';
import Dashboard from './pages/Dashboard';
import Upload from './pages/Upload';
import Loading from './Components/Loading';
import Demo from './pages/Demo';
import Chatbot from './pages/Chatbot';
import { ChatProvider } from './pages/ChatContext'; // Import ChatProvider
import './App.css';

// Shared layout component with navigation
const Layout = () => {
  return (
    <div className="app-container">
      <Navigation />
      <div className="content-container">
        <React.Suspense fallback={<Loading />}>
          <Outlet />
        </React.Suspense>
      </div>
    </div>
  );
};

// Navigation component
const Navigation = () => {
  return (
    <nav className="main-nav">
      <div className="nav-brand">Requirements Manager</div>
      <ul className="nav-links">
        <li>
          <Link to="/" className="nav-link">Home</Link>
        </li>
        <li>
          <Link to="/upload" className="nav-link">Upload & Review</Link>
        </li>
        <li>
          <Link to="/dashboard" className="nav-link">Dashboard</Link>
        </li>
        <li>
          <Link to="/demo" className="nav-link">Demo</Link>
        </li>
        <li>
          <Link to="/chatbot" className="nav-link">Chatbot</Link>
        </li>
      </ul>
    </nav>
  );
};

function App() {
  return (
    <ChatProvider> {/* Wrap the entire app with ChatProvider */}
      <Router>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Home />} />
            <Route path="upload" element={<Upload />} />
            <Route path="dashboard" element={<Dashboard />} />
            <Route path="demo" element={<Demo />} />
            <Route path="chatbot" element={<Chatbot />} />
          </Route>
        </Routes>
      </Router>
    </ChatProvider>
  );
}

export default App;