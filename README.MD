# AI-Powered Requirements Analysis System

## Overview

An intelligent system for automated requirement extraction, classification, and analysis. Processes documents (PDF/TXT) to identify and categorize software requirements with AI-powered insights and chatbot integration.

## Key Features

- **Document Analysis**  
  Upload and analyze requirement documents (PDF/TXT/MD)
- **AI Classification**  
  Automatic categorization using NLP models (BART + spaCy)
- **Smart Metadata Extraction**  
  Detect authors, dates, and technical specifications
- **Requirement Management**  
  Database storage with filtering/search capabilities
- **Analytics Dashboard**  
  Real-time statistics and reporting

## Enhanced Tech Stack

### Backend Services:
- **Python 3.12** (Flask REST API)
- **SQLAlchemy** (Database ORM)
- **spaCy** (NLP processing)
- **Hugging Face Transformers** (BART classification)
- **pdfminer.six** (PDF text extraction)

### Frontend Interface:
- **React.js** (UI Framework)
- **Material UI** (Design System)
- **Axios** (API Client)

## System Setup

### Prerequisites
- Python 3.12+
- Node.js 18+
- Git

### Prerequisites

- Python 3.x
- Node.js & npm
- Git

### Creating virtual environmnet and installing necessary packages

```sh
# New Terminal 1   

> python -m venv venv
> venv\Scripts\activate
> pip install -r requirements.txt
```
### Backend Setup

```sh
# New Terminal 2

> venv\Scripts\activate
> cd backend
> cd ai_service
> flask run --host=0.0.0.0 --port=5000
```

### Frontend Setup

```sh
#New Terminal 3

> cd frontend
> npm install
> npm start
```
### Our System Specifications
```sh
Python- 3.12.6
Node.js- 22.14.0
```
### API Endpoints
- /api/upload	POST	File upload and processing
- /api/analyze	POST	Document analysis pipeline
- /api/chat	POST	AI chatbot interface (yet to be implemented)
- /api/requirements	GET	Retrieve requirements
- /api/health	GET	System status check

### Key Enhancements

- Context-aware requirement analysis

- Natural language query interface

- Real-time system statistics in responses

- Improved Security

- Environment variable configuration

- Input sanitization

- Safety filters for AI responses

- Database Features

- SQLAlchemy ORM

- Advanced filtering/search

- Automated metadata extraction

### System Requirements
- **Component	Minimum Version**
- Python	3.12
- Node.js	18.x
- RAM	4GB
- Disk	500MB



