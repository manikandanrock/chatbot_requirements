# AI-Powered Requirements Management System

This project is an AI-powered requirements management system designed to help teams classify, prioritize, and manage software requirements efficiently. It uses the **Gemini API** for natural language processing (NLP) to classify requirements into categories such as Functional, Non-Functional, UI, Security, and Performance. The system also estimates priority, complexity, and development time for each requirement.

---

## Features

- **File Upload**: Upload requirements in `.txt`, `.md`, or `.pdf` format.
- **AI Classification**: Automatically classify requirements into categories using the Gemini API.
- **Priority and Complexity Estimation**: Estimate the priority (High, Medium, Low) and complexity (High, Moderate, Low) of each requirement.
- **Database Storage**: Store requirements in a SQLite database for easy retrieval and management.
- **Search and Filter**: Search and filter requirements by category, status, priority, or complexity.
- **Statistics**: View system-wide statistics, including the total number of requirements, approved requirements, and requirements in review.

---

## Technologies Used

- **Backend**: Flask (Python)
- **Frontend**: React (if applicable)
- **Database**: SQLite
- **AI/ML**: Gemini API for NLP-based classification
- **File Processing**: `pdfminer` for PDF text extraction
- **Logging**: Python `logging` module
- **Environment Management**: `python-dotenv`

---

## Setup Instructions

### Prerequisites

1. **Python 3.8+**: Ensure Python is installed on your system.
2. **Gemini API Key**: Obtain an API key from [Google Cloud Console](https://console.cloud.google.com/).
3. **Node.js (Optional)**: If you're using a React frontend, ensure Node.js is installed.

---
### Set Up Environment Variables
Create a `.env` file in the root directory and add your Gemini API key:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```
---

### Installation

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
## Usage

### Upload Requirements
1. Navigate to the file upload page.
2. Upload a `.txt`, `.md`, or `.pdf` file containing your requirements.
3. The system will automatically classify the requirements and display the results.

### View Requirements
1. Navigate to the requirements page.
2. Use the search and filter options to find specific requirements.
3. View details such as category, priority, complexity, and estimated time.

### Update Requirements
1. Click on a requirement to view its details.
2. Update the status, priority, or complexity as needed.
3. Save your changes.

### View Statistics
1. Navigate to the statistics page.
2. View system-wide statistics, including the total number of requirements, approved requirements, and requirements in review.

### Chatbot
1. Ask the Gemini Ai powered chatbot to get insights about the requiremnts.
2. It provieds the needed domain knowledge to work with the project and requirements.

---

## API Endpoints

### Backend API
- **POST `/api/upload`**: Upload a file for analysis.
- **POST `/api/analyze`**: Analyze the uploaded file and classify requirements.
- **GET `/api/requirements`**: Retrieve all requirements.
- **POST `/api/requirements`**: Create a new requirement.
- **GET `/api/requirements/<req_id>`**: Retrieve a specific requirement.
- **PUT `/api/requirements/<req_id>`**: Update a specific requirement.
- **DELETE `/api/requirements/<req_id>`**: Delete a specific requirement.
- **PATCH `/api/requirements/<req_id>/status`**: Update the status of a requirement.
- **GET `/api/requirements/stats`**: Retrieve system-wide statistics.

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



