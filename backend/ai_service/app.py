import os
import re
import logging
import traceback
from typing import Self
from venv import logger
import requests
import torch
import werkzeug
from flask import Flask, request, jsonify, abort, session
from pdfminer.high_level import extract_text
from flask_cors import CORS
from transformers import pipeline
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from dateutil import parser as dparser
from enum import Enum
from sqlalchemy import or_, func, case
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Load environment variables
load_dotenv()

# Flask App Initialization
app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": "http://localhost:3000",
        "methods": ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True  # Allow credentials (cookies)
    }
})

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///requirements.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')  # Required for session security
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True

# Initialize database
db = SQLAlchemy(app)

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Constants
MISSING_FIELDS_ERROR = "Missing required fields"
JIRA_CONNECTION_ERROR = "Failed to connect to Jira. Please check your credentials and try again."
JIRA_PUSH_ERROR = "Failed to create Jira issue. Please check your settings and try again."

# Enums for requirement attributes
class PriorityEnum(str, Enum):
    HIGH = 'High'
    MEDIUM = 'Medium'
    LOW = 'Low'

class ComplexityEnum(str, Enum):
    HIGH = 'High'
    MODERATE = 'Moderate'
    LOW = 'Low'

class StatusEnum(str, Enum):
    APPROVED = 'Approved'
    DISAPPROVED = 'Disapproved'
    REVIEW = 'Review'
    DRAFT = 'Draft'

# Project model
class Project(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    description = db.Column(db.Text, nullable=True)
    hourly_rate = db.Column(db.Float, default=30.0)  # Add this line
    created_at = db.Column(db.DateTime, default=datetime.now)
    requirements = db.relationship('Requirement', backref='project', lazy=True)

# Requirement model
class Requirement(db.Model):
    id = db.Column(db.String(20), primary_key=True)  # Only one ID column
    requirement = db.Column(db.Text, nullable=False)
    categories = db.Column(db.Text, nullable=False)
    status = db.Column(db.Enum(StatusEnum), default=StatusEnum.REVIEW)
    priority = db.Column(db.Enum(PriorityEnum), default=PriorityEnum.MEDIUM)
    author = db.Column(db.String(100), default='System')
    ddate = db.Column(db.DateTime, default=datetime.now)
    complexity = db.Column(db.Enum(ComplexityEnum), default=ComplexityEnum.MODERATE)
    estimated_time = db.Column(db.Integer)
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=True)


    __table_args__ = (
        db.Index('idx_ddate', 'ddate'),
        db.Index('idx_status', 'status'),
        db.Index('idx_priority', 'priority'),
        db.Index('idx_author', 'author'),
    )

# Example using a dedicated counter table with atomic update
def generate_requirement_id(project_id):
    for _ in range(3):  # Retry up to 3 times
        try:
            # Start a nested transaction
            db.session.begin_nested()
            
            # Get or create counter with lock
            counter = db.session.query(RequirementCounter).filter_by(
                project_id=project_id
            ).with_for_update().first()
            
            if not counter:
                counter = RequirementCounter(project_id=project_id, count=0)
                db.session.add(counter)
                db.session.flush()  # Ensure counter exists before incrementing
            
            # Increment and get the new count
            counter.count += 1
            db.session.flush()
            
            # Format the ID
            req_id = f"p{project_id}_r{counter.count}"
            
            # Verify the ID doesn't exist (just to be safe)
            if not db.session.query(Requirement.id).filter_by(id=req_id).first():
                db.session.commit()  # Commit the nested transaction
                return req_id
            
            # If we get here, the ID exists (very unlikely but possible)
            db.session.rollback()  # Rollback the nested transaction
            continue
            
        except Exception as e:
            db.session.rollback()  # Rollback the nested transaction on error
            logging.error(f"ID generation attempt failed: {str(e)}")
            continue
    
    logging.error("Failed to generate unique requirement ID after 3 attempts")
    raise RuntimeError("Failed to generate unique requirement ID")

class RequirementCounter(db.Model):
    __tablename__ = 'requirement_counter'
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, nullable=False, unique=True)
    count = db.Column(db.Integer, default=0, nullable=False)

# Create database tables
with app.app_context():
    db.create_all()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def generate_requirement_id(project_id):
    """
    Generate a project-specific requirement ID in the format p<project_id>_r<X>,
    where X is a sequential number.
    """
    try:
        # Get the highest existing sequence number for the project
        last_requirement = (
            Requirement.query.filter_by(project_id=project_id)
            .order_by(Requirement.id.desc())
            .first()
        )
        
        if last_requirement:
            # Extract the sequence number from the last requirement ID
            last_sequence = int(last_requirement.id.split('_r')[-1])  # Extract the number after '_r'
            return f"p{project_id}_r{last_sequence + 1}"  # Corrected format
        else:
            return f"p{project_id}_r1"  # Start from 1 if no requirements exist for the project
    except Exception as e:
        logging.error(f"ID generation failed: {str(e)}")
        raise RuntimeError("Failed to generate requirement ID")

# Configure and load the zero-shot classification pipeline with optimized settings
zero_shot_pipeline = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0 if torch.cuda.is_available() else -1,  # Use GPU if available
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    batch_size=8  # Process multiple items at once when possible
)

# Pre-defined time labels and templates for consistent predictions
TIME_LABELS = ["2 hours", "4 hours", "6 hours", "8 hours", "10 hours", 
               "12 hours", "24 hours", "48 hours", "50 hours", "72 hours"]
TIME_HYPOTHESIS = "This requirement will take {} to implement."

# Cache for frequent predictions to avoid redundant computations
_prediction_cache = {}
CACHE_SIZE = 1000  # Limit cache size to prevent memory issues

def predict_estimated_time(requirement_text, complexity=None, priority=None):
    """
    Optimized prediction of implementation time for requirements using zero-shot learning.
    
    Args:
        requirement_text (str): Text of the requirement to analyze
        complexity (str, optional): Complexity level (High/Medium/Low)
        priority (str, optional): Priority level (High/Medium/Low)
    
    Returns:
        int: Predicted hours for implementation (defaults to 4 if prediction fails)
    """
    # Return cached result if available
    cache_key = hash(requirement_text[:200])  # Use first 200 chars for cache key
    if cache_key in _prediction_cache:
        return _prediction_cache[cache_key]
    
    try:
        # Batch processing when possible (modify if you process multiple reqs at once)
        results = zero_shot_pipeline(
            [requirement_text],
            candidate_labels=TIME_LABELS,
            hypothesis_template=TIME_HYPOTHESIS,
            multi_label=False
        )
        
        # Handle single or batch results
        if isinstance(results, list):
            result = results[0]  # Get first result if batch processing
        else:
            result = results
            
        # Extract the predicted time
        top_label = result['labels'][0]
        predicted_time = int(top_label.split()[0])
        
        # Apply complexity/priority adjustments if provided
        if complexity and priority:
            adjustment = 1.0
            if complexity.lower() == 'high':
                adjustment *= 1.5
            elif complexity.lower() == 'low':
                adjustment *= 0.75
                
            if priority.lower() == 'high':
                adjustment *= 0.8  # High priority might mean more resources allocated
            elif priority.lower() == 'low':
                adjustment *= 1.2
                
            predicted_time = max(2, round(predicted_time * adjustment))
        
        # Update cache and maintain size
        if len(_prediction_cache) >= CACHE_SIZE:
            _prediction_cache.popitem()  # Remove oldest entry
        _prediction_cache[cache_key] = predicted_time
        
        return predicted_time
        
    except Exception as e:
        logging.error(f"Error predicting time for '{requirement_text[:50]}...': {str(e)}")
        
        # Return complexity/priority based defaults if prediction fails
        default_time = 4
        if complexity and priority:
            if complexity.lower() == 'high':
                default_time = 8
            elif complexity.lower() == 'low':
                default_time = 2
                
            if priority.lower() == 'high':
                default_time = max(2, default_time - 2)
            elif priority.lower() == 'low':
                default_time += 2
                
        return default_time

# Load NLP models
try:
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        multi_label=True
    )
    logging.info("✅ AI models loaded successfully")
except Exception as e:
    logging.error(f"❌ Error loading models: {e}")
    classifier = None

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

# Uploads Directory
UPLOAD_DIR = "uploads/"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Classification Labels
candidate_labels = ["Functional", "Non-Functional", "UI", "Security", "Performance"]
complexity_labels = ["High", "Moderate", "Low"]
priority_labels = ["High priority", "Medium priority", "Low priority"]

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'md'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_metadata(text):
    metadata = {'author': 'System', 'date': datetime.now()}
    try:
        date_match = dparser.parse(text, fuzzy=True)
        metadata['date'] = date_match
    except Exception:
        pass
    
    author_patterns = [
        r"Prepared by:\s*(.+)",
        r"Author:\s*(.+)",
        r"By\s+(.+)",
        r"Created by:\s*(.+)",
        r"Submitted by:\s*(.+)"
    ]
    for pattern in author_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            metadata['author'] = match.group(1).strip()
            break
    return metadata

def clean_text(text):
    return re.sub(r"\s+", " ", re.sub(r"[•\t\n]+", " ", text)).strip()

# Project Endpoints
@app.route('/api/projects', methods=['GET', 'POST'])
@limiter.limit("50 per hour")
def handle_projects():
    if request.method == 'GET':
        try:
            projects = Project.query.all()
            return jsonify([{
                "id": project.id,
                "name": project.name,
                "description": project.description,  # Add this line
                "hourly_rate": project.hourly_rate,
                "created_at": project.created_at.isoformat(),
                "requirements_count": len(project.requirements)
            } for project in projects])
        except Exception as e:
            logging.error(f"Error fetching projects: {str(e)}")
            return jsonify({"error": "Failed to fetch projects"}), 500

    elif request.method == 'POST':
        try:
            data = request.get_json()
            new_project = Project(
                name=data['name'],
                description=data.get('description', '')  # Add this line
            )
            db.session.add(new_project)
            db.session.commit()
            return jsonify({
                "id": new_project.id,
                "name": new_project.name,
                "description": new_project.description,
                "message": "Project created successfully"
            }), 201
        except Exception as e:
            db.session.rollback()
            return jsonify({"error": str(e)}), 500
        
@app.route('/api/projects/<int:project_id>/requirements', methods=['DELETE'])
@limiter.limit("50 per hour")
def delete_all_requirements(project_id):
    try:
        # Delete all requirements for the project
        Requirement.query.filter_by(project_id=project_id).delete()
        db.session.commit()
        return jsonify({"message": "All requirements deleted successfully"}), 200
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error deleting requirements: {str(e)}")
        return jsonify({"error": "Failed to delete requirements"}), 500

@app.route('/api/projects/<int:project_id>', methods=['PUT'])
@limiter.limit("50 per hour")
def rename_project(project_id):
    try:
        data = request.get_json()
        project = Project.query.get(project_id)
        if not project:
            return jsonify({"error": "Project not found"}), 404
        
        if 'hourly_rate' in data:
            project.hourly_rate = float(data['hourly_rate'])

        # Update project name and description
        if 'name' in data:
            project.name = data['name']
        if 'description' in data:
            project.description = data.get('description', '')

        db.session.commit()
        return jsonify({
            "message": "Project updated successfully",
            "project": {
                "id": project.id,
                "name": project.name,
                "description": project.description,
            }
        }), 200
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error updating project: {str(e)}")
        return jsonify({"error": "Failed to update project"}), 500

@app.route('/api/projects/<int:project_id>', methods=['DELETE'])
@limiter.limit("50 per hour")
def delete_project(project_id):
    try:
        project = Project.query.get(project_id)
        if not project:
            return jsonify({"error": "Project not found"}), 404

        # Delete all requirements for the project
        Requirement.query.filter_by(project_id=project_id).delete()

        # Delete the project
        db.session.delete(project)
        db.session.commit()
        return jsonify({"message": "Project and its requirements deleted successfully"}), 200
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error deleting project: {str(e)}")
        return jsonify({"error": "Failed to delete project"}), 500
    
@app.route('/api/projects/<int:project_id>/requirements', methods=['GET', 'POST'])
@limiter.limit("50 per hour")
def handle_project_requirements(project_id):
    project = Project.query.get(project_id)
    if not project:
        return jsonify({"error": "Project not found"}), 404

    if request.method == 'GET':
        try:
            requirements = Requirement.query.filter_by(project_id=project_id).all()
            return jsonify([{
                "id": req.id,
                "requirement": req.requirement,
                "categories": req.categories,
                "status": req.status.value,
                "priority": req.priority.value,
                "complexity": req.complexity.value,
                "estimated_time": req.estimated_time,
                "author": req.author,
                "date": req.ddate.isoformat()
            } for req in requirements])
        except Exception as e:
            logging.error(f"Error fetching project requirements: {str(e)}")
            return jsonify({"error": "Failed to fetch project requirements"}), 500
        
    elif request.method == 'POST':
        try:
            data = request.get_json()

            # Validate required fields
            if not data.get('requirement'):
                return jsonify({"error": "Requirement text is required"}), 400

            # Clean the requirement text
            cleaned = clean_text(data['requirement'])

            # Classify the requirement using AI
            classification = classifier(cleaned, candidate_labels)
            categories = ', '.join(classification['labels'][:3])

            # Validate and set status, priority, and complexity
            status = data.get('status', 'Review')
            if status not in [e.value for e in StatusEnum]:
                return jsonify({"error": f"Invalid status: {status}"}), 400

            priority = data.get('priority', 'Medium')
            if priority not in [e.value for e in PriorityEnum]:
                return jsonify({"error": f"Invalid priority: {priority}"}), 400

            complexity = data.get('complexity', 'Moderate')
            if complexity not in [e.value for e in ComplexityEnum]:
                return jsonify({"error": f"Invalid complexity: {complexity}"}), 400

            # Create a new requirement
            new_req = Requirement(
                id=generate_requirement_id(project_id),  # Generate a unique ID
                requirement=cleaned,
                categories=categories,
                status=StatusEnum(status),
                priority=PriorityEnum(priority),
                complexity=ComplexityEnum(complexity),
                estimated_time=data.get('estimated_time', 4),
                author=data.get('author', 'System'),
                ddate=datetime.now(),
                project_id=project_id
            )

            # Add and commit the new requirement to the database
            db.session.add(new_req)
            db.session.commit()

            # Return success response
            return jsonify({
                "id": new_req.id,
                "message": "Requirement created and assigned to project successfully"
            }), 201
        
        except ValueError as e:
            db.session.rollback()
            logging.error(f"ValueError: {str(e)}")
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            db.session.rollback()
            logging.error(f"Unexpected error: {str(e)}")
            return jsonify({"error": "Failed to create requirement"}), 500
        
@app.route('/api/requirements/<string:req_id>/assign', methods=['PATCH'])
@limiter.limit("50 per hour")
def assign_requirement_to_project(req_id):
    requirement = Requirement.query.get(req_id)
    if not requirement:
        return jsonify({"error": "Requirement not found"}), 404

    try:
        data = request.get_json()
        project_id = data.get('project_id')
        if project_id:
            project = Project.query.get(project_id)
            if not project:
                return jsonify({"error": "Project not found"}), 404
            requirement.project_id = project_id
            db.session.commit()
            return jsonify({
                "message": "Requirement assigned to project successfully",
                "project_id": project_id
            })
        else:
            return jsonify({"error": "Project ID is required"}), 400
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@app.route('/api/upload', methods=['POST'])
@limiter.limit("50 per hour")
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        filename = werkzeug.utils.secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_DIR, filename)
        file.save(file_path)
        return jsonify({'message': 'File uploaded successfully', 'file_path': file_path}), 200
    except Exception as e:
        logging.error(f"File upload error: {e}")
        return jsonify({'error': 'File upload failed'}), 500

@app.route("/api/analyze", methods=["POST"])
@limiter.limit("50 per hour")
def analyze_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    # Get project_id from the request
    project_id = request.form.get('project_id')
    if not project_id:
        return jsonify({"error": "Project ID is required"}), 400

    # Check if the project exists
    project = Project.query.get(project_id)
    if not project:
        return jsonify({"error": "Project not found"}), 404

    # Process the file and create requirements
    file_path = None
    try:
        filename = werkzeug.utils.secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_DIR, filename)
        file.save(file_path)

        if filename.endswith(".pdf"):
            text = extract_text(file_path)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

        if not text.strip():
            return jsonify({"error": "No text extracted from file"}), 500

        metadata = extract_metadata(text)
        requirements = []
        sentences = text.split('.')  # Simple sentence splitting

        # Get the highest existing sequence number for the project first
        last_requirement = (
            Requirement.query.filter_by(project_id=project_id)
            .order_by(Requirement.id.desc())
            .first()
        )
        
        if last_requirement:
            # Extract the sequence number from the last requirement ID
            last_sequence = int(last_requirement.id.split('_r')[-1])
        else:
            last_sequence = 0

        for sent in sentences:
            cleaned = clean_text(sent)
            if len(cleaned.split()) < 3:
                continue

            try:
                # Generate sequential IDs locally
                last_sequence += 1
                requirement_id = f"p{project_id}_r{last_sequence}"

                # Classify the requirement
                classification = zero_shot_pipeline(cleaned, candidate_labels)
                categories = ', '.join(classification['labels'][:3])

                # Predict priority
                priority_result = zero_shot_pipeline(
                    cleaned, 
                    candidate_labels=priority_labels,
                    hypothesis_template="This requirement has {} priority."
                )
                priority = PriorityEnum(priority_result['labels'][0].split()[0])

                # Predict complexity
                complexity_result = zero_shot_pipeline(
                    cleaned,
                    candidate_labels=complexity_labels,
                    hypothesis_template="This requirement has {} complexity."
                )
                complexity = ComplexityEnum(complexity_result['labels'][0])

                # Predict estimated time
                estimated_time = predict_estimated_time(cleaned, complexity.value, priority.value)
                if not estimated_time:
                    estimated_time = 4  # Default fallback value

                # Create the requirement
                requirement = Requirement(
                    id=requirement_id,
                    requirement=cleaned,
                    categories=categories,
                    status=StatusEnum.REVIEW,
                    priority=priority,
                    complexity=complexity,
                    estimated_time=estimated_time,
                    author=metadata['author'],
                    ddate=metadata['date'],
                    project_id=project_id
                )

                db.session.add(requirement)
                requirements.append({
                    "id": requirement.id,
                    "requirement": cleaned,
                    "categories": categories,
                    "status": requirement.status.value,
                    "priority": requirement.priority.value,
                    "complexity": requirement.complexity.value,
                    "estimated_time": estimated_time,
                    "author": requirement.author,
                    "date": requirement.ddate.isoformat()
                })

            except Exception as e:
                logging.error(f"Error processing requirement: {str(e)}")
                continue  # Skip this requirement but continue with others

        db.session.commit()
        
        # Clean up the uploaded file
        if file_path and os.path.exists(file_path):
            os.remove(file_path)

        return jsonify({
            "requirements": requirements,
            "total": len(requirements),
            "project_id": project_id,
            "project_name": project.name
        })

    except Exception as e:
        db.session.rollback()
        logging.error(f"Analysis error: {str(e)}", exc_info=True)
        
        # Clean up file if error occurred
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            
        return jsonify({"error": "Failed to analyze file"}), 500
    
@app.route('/api/projects/<int:project_id>/requirements', methods=['POST'])
def create_requirement(project_id):
    project = Project.query.get(project_id)
    if not project:
        return jsonify({"error": "Project not found"}), 404

    try:
        data = request.get_json()

        # Validate required fields
        if not data.get('requirement'):
            return jsonify({"error": "Requirement text is required"}), 400

        # Call the AI model to classify the requirement
        classification = classifier(data['requirement'], candidate_labels)
        categories = ', '.join(classification['labels'][:3])  # Top 3 categories
        # Predict priority using zero-shot classification
        priority_result = zero_shot_pipeline(
            data['requirement'], 
            candidate_labels=priority_labels,
            hypothesis_template="This requirement has {} priority."
        )
        priority = PriorityEnum(priority_result['labels'][0].split()[0])

        # Predict complexity using zero-shot classification
        complexity_result = zero_shot_pipeline(
            data['requirement'],
            candidate_labels=complexity_labels,
            hypothesis_template="This requirement has {} complexity."
        )
        complexity = ComplexityEnum(complexity_result['labels'][0])

        # Create a new requirement
        new_req = Requirement(
            id=generate_requirement_id(project_id),  # Generate a unique ID
            requirement=data['requirement'],
            categories=categories,
            status=StatusEnum(data.get('status', 'Review')),
            priority=PriorityEnum(priority),
            complexity=ComplexityEnum(complexity),
            estimated_time=data.get('estimated_time', 4),
            author=data.get('author', 'System'),
            ddate=datetime.now(),
            project_id=project_id
        )

        # Add and commit the new requirement to the database
        db.session.add(new_req)
        db.session.commit()

        # Return success response
        return jsonify({
            "id": new_req.id,
            "message": "Requirement created and assigned to project successfully"
        }), 201

    except ValueError as e:
        db.session.rollback()
        logging.error(f"ValueError: {str(e)}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        db.session.rollback()
        logging.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": "Failed to create requirement"}), 500


@app.route("/api/requirements", methods=["GET", "POST"])
@limiter.limit("50 per hour")
def handle_requirements():
    if request.method == "GET":
        try:
            search_query = request.args.get('search', '')
            types = request.args.getlist('type')
            statuses = request.args.getlist('status')
            complexities = request.args.getlist('complexity')
            priorities = request.args.getlist('priority')
            project_id = request.args.get('project')
            page = request.args.get('page', 1, type=int)
            per_page = request.args.get('per_page', 10, type=int)

            # Validate project ID format if present
            if project_id:
                try:
                    project_id = int(project_id)
                except ValueError:
                    return jsonify({"error": "Invalid project ID format"}), 400

            # Validate enum filters
            for status in statuses:
                if status not in [e.value for e in StatusEnum]:
                    raise ValueError(f"Invalid status filter: {status}")
            for priority in priorities:
                if priority not in [e.value for e in PriorityEnum]:
                    raise ValueError(f"Invalid priority filter: {priority}")
            for complexity in complexities:
                if complexity not in [e.value for e in ComplexityEnum]:
                    raise ValueError(f"Invalid complexity filter: {complexity}")

            query = Requirement.query

            # Apply project filter if ID is valid
            if project_id:
                query = query.filter_by(project_id=project_id)

            if search_query:
                query = query.filter(
                    Requirement.requirement.ilike(f'%{search_query}%') |
                    Requirement.categories.ilike(f'%{search_query}%')
                )

            if types:
                type_filters = [Requirement.categories.ilike(f'%{t}%') for t in types]
                query = query.filter(or_(*type_filters))

            if statuses:
                query = query.filter(Requirement.status.in_([StatusEnum(s) for s in statuses]))

            if complexities:
                query = query.filter(Requirement.complexity.in_([ComplexityEnum(c) for c in complexities]))

            if priorities:
                query = query.filter(Requirement.priority.in_([PriorityEnum(p) for p in priorities]))

            # Calculate statistics
            stats_query = query.with_entities(
                func.count(Requirement.id),
                func.count(case((Requirement.status == StatusEnum.APPROVED, 1))),
                func.count(case((Requirement.status == StatusEnum.REVIEW, 1))),
                func.count(case((Requirement.status == StatusEnum.DISAPPROVED, 1)))
            ).one()

            stats = {
                "total": stats_query[0],
                "approved": stats_query[1],
                "inReview": stats_query[2],
                "disapproved": stats_query[3]
            }

            # Pagination
            pagination = query.paginate(
                page=page,
                per_page=per_page,
                error_out=False
            )

            return jsonify({
                "requirements": [{
                    "id": req.id,
                    "requirement": req.requirement,
                    "categories": req.categories,
                    "status": req.status.value,
                    "priority": req.priority.value,
                    "complexity": req.complexity.value,
                    "estimated_time": req.estimated_time,
                    "author": req.author,
                    "date": req.ddate.isoformat()
                } for req in pagination.items],
                "stats": stats,
                "total": pagination.total,
                "page": pagination.page,
                "pages": pagination.pages
            })

        except ValueError as ve:
            logging.error(f"Validation error: {str(ve)}")
            return jsonify({"error": str(ve)}), 400
        except Exception as e:
            logging.error(f"Error fetching requirements: {str(e)}")
            return jsonify({"error": "Failed to fetch requirements"}), 500

    elif request.method == "POST":
        try:
            data = request.get_json()
            
            # Validate required field
            if 'requirement' not in data or not data['requirement'].strip():
                return jsonify({"error": "Requirement text is required"}), 400

            cleaned = clean_text(data['requirement'])
            
            # Classify requirement
            classification = zero_shot_pipeline(cleaned, candidate_labels)
            categories = ', '.join(classification['labels'][:3])

            # Predict priority
            priority_result = zero_shot_pipeline(
                cleaned, 
                candidate_labels=priority_labels,
                hypothesis_template="This requirement has {} priority."
            )
            priority = PriorityEnum(priority_result['labels'][0].split()[0])

            # Predict complexity
            complexity_result = zero_shot_pipeline(
                cleaned,
                candidate_labels=complexity_labels,
                hypothesis_template="This requirement has {} complexity."
            )
            complexity = ComplexityEnum(complexity_result['labels'][0])

            # Generate requirement ID with project context
            project_id = data.get('project_id')
            requirement_id = generate_requirement_id(project_id) if project_id else generate_requirement_id()

            # Create new requirement
            new_req = Requirement(
                id=requirement_id,
                requirement=cleaned,
                categories=categories,
                status=StatusEnum(data.get('status', 'Review')),
                priority=priority,
                complexity=complexity,
                estimated_time=predict_estimated_time(cleaned, complexity.value, priority.value) or 4,
                author=data.get('author', 'System'),
                ddate=datetime.now(),
                project_id=project_id
            )

            db.session.add(new_req)
            db.session.commit()

            return jsonify({
                "id": new_req.id,
                "message": "Requirement created successfully"
            }), 201

        except ValueError as ve:
            db.session.rollback()
            return jsonify({"error": str(ve)}), 400
        except Exception as e:
            db.session.rollback()
            logging.error(f"Error creating requirement: {str(e)}")
            return jsonify({"error": "Failed to create requirement"}), 500

@app.route("/api/requirements/<string:req_id>", methods=["GET", "PUT", "DELETE"])
@limiter.limit("50 per hour")
def handle_single_requirement(req_id):
    requirement = Requirement.query.get(req_id)
    if not requirement:
        return jsonify({"error": "Requirement not found"}), 404

    if request.method == "GET":
        return jsonify({
            "id": requirement.id,
            "requirement": requirement.requirement,
            "categories": requirement.categories,
            "status": requirement.status.value,
            "priority": requirement.priority.value,
            "complexity": requirement.complexity.value,
            "estimated_time": requirement.estimated_time,
            "author": requirement.author,
            "date": requirement.ddate.isoformat()
        })
    
    elif request.method == "PUT":
        try:
            data = request.get_json()
            if 'requirement' in data:
                requirement.requirement = clean_text(data['requirement'])
            if 'categories' in data:
                requirement.categories = data['categories']
            if 'status' in data:
                requirement.status = StatusEnum(data['status'])
            if 'priority' in data:
                requirement.priority = PriorityEnum(data['priority'])
            if 'complexity' in data:
                requirement.complexity = ComplexityEnum(data['complexity'])
            if 'estimated_time' in data:
                requirement.estimated_time = int(data['estimated_time'])
            if 'author' in data:
                requirement.author = data['author']
            
            db.session.commit()
            return jsonify({
                "message": "Requirement updated successfully",
                "requirement": {
                    "id": requirement.id,
                    "author": requirement.author,
                    "status": requirement.status.value
                }
            })
        
        except ValueError as e:
            db.session.rollback()
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            db.session.rollback()
            return jsonify({"error": str(e)}), 500
    
    elif request.method == "DELETE":
        db.session.delete(requirement)
        db.session.commit()
        return jsonify({"message": "Requirement deleted successfully"})

@app.route("/api/requirements/<string:req_id>/status", methods=["PATCH"])
@limiter.limit("50 per hour")
def update_status(req_id):
    requirement = Requirement.query.get(req_id)
    if not requirement:
        return jsonify({"error": "Requirement not found"}), 404
    
    try:
        data = request.get_json()
        new_status = StatusEnum(data['status'])
        requirement.status = new_status
        db.session.commit()
        return jsonify({
            "message": "Status updated successfully",
            "new_status": new_status.value
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": "Status update failed"}), 500
    
def get_system_stats():
    """Get current system statistics from database"""
    try:
        return {
            "total": Requirement.query.count(),
            "approved": Requirement.query.filter_by(status=StatusEnum.APPROVED).count(),
            "inReview": Requirement.query.filter_by(status=StatusEnum.REVIEW).count(),
            "disapproved": Requirement.query.filter_by(status=StatusEnum.DISAPPROVED).count(),
        }
    except Exception as e:
        logging.error(f"Error getting system stats: {str(e)}")
        return {
            "total": 0,
            "approved": 0,
            "inReview": 0,
            "disapproved": 0
        }
    
@app.route('/api/requirements/stats', methods=['GET'])
@limiter.limit("50 per hour")
def get_stats():
    try:
        project_id = request.args.get('project')
        query = Requirement.query

        if project_id:
            try:
                project_id = int(project_id)
                query = query.filter_by(project_id=project_id)
            except ValueError:
                return jsonify({"error": "Invalid project ID format"}), 400

        total = query.count()
        approved = query.filter_by(status=StatusEnum.APPROVED).count()
        in_review = query.filter_by(status=StatusEnum.REVIEW).count()
        disapproved = query.filter_by(status=StatusEnum.DISAPPROVED).count()

        return jsonify({
            "total": total,
            "approved": approved,
            "inReview": in_review,
            "disapproved": disapproved
        })
    except Exception as e:
        logging.error(f"Error fetching stats: {e}")
        return jsonify({"error": "Failed to fetch stats"}), 500
    
    
@app.route('/api/classify', methods=['POST'])
def classify_requirement():
    try:
        data = request.get_json()

        # Validate required fields
        if not data.get('text'):
            return jsonify({"error": "Text is required"}), 400

        # Call the AI model to classify the text
        classification = classifier(data['text'], candidate_labels)

        # Extract the top categories, priority, and complexity
        categories = classification['labels'][:3]  # Top 3 categories
        # Predict priority using the AI model
        priority_result = zero_shot_pipeline(
            data['text'], 
            candidate_labels=priority_labels,
            hypothesis_template="This requirement has {} priority."
        )
        priority = PriorityEnum(priority_result['labels'][0].split()[0])

        # Predict complexity using the AI model
        complexity_result = zero_shot_pipeline(
            data['text'],
            candidate_labels=complexity_labels,
            hypothesis_template="This requirement has {} complexity."
        )
        complexity = ComplexityEnum(complexity_result['labels'][0])

        return jsonify({
            "categories": categories,
            "priority": priority,
            "complexity": complexity,
        }), 200
    except Exception as e:
        logging.error(f"Error classifying requirement: {str(e)}")
        return jsonify({"error": "Failed to classify requirement"}), 500
        
@app.route('/api/chat', methods=['POST'])
@limiter.limit("50 per hour")
def handle_chat():
    try:
        # Parse the request data
        data = request.get_json()
        user_message = data.get('message', '').strip()
        project_id = data.get('project_id')  # Get project_id from the request

        # Validate the user message
        if not user_message:
            return jsonify({'error': 'Empty message received'}), 400

        # Validate the project_id
        if not project_id:
            return jsonify({'error': 'Project ID is required'}), 400

        # Check if the project exists
        project = Project.query.get(project_id)
        if not project:
            return jsonify({'error': 'Project not found'}), 404

        try:
            # Fetch requirements for the selected project only
            requirements = (
                Requirement.query.filter_by(project_id=project_id)
                .order_by(Requirement.ddate.desc())
                .all()
            )

            # Calculate statistics for the project
            stats = {
                "total": len(requirements),
                "approved": sum(1 for req in requirements if req.status == StatusEnum.APPROVED),
                "inReview": sum(1 for req in requirements if req.status == StatusEnum.REVIEW),
                "disapproved": sum(1 for req in requirements if req.status == StatusEnum.DISAPPROVED),
            }

            # Format requirements data for AI context
            requirements_context = []
            for req in requirements:
                requirements_context.append(
                    f"Requirement ID: {req.id}\n"
                    f"Text: {req.requirement}\n"
                    f"Categories: {req.categories}\n"
                    f"Status: {req.status.value}\n"
                    f"Priority: {req.priority.value}\n"
                    f"Complexity: {req.complexity.value}\n"
                    f"Estimated Time: {req.estimated_time} hours\n"
                    f"Author: {req.author}\n"
                    f"Date: {req.ddate.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"{'-'*40}"
                )

            # Build context-aware prompt
            prompt = f"""
            You are an AI assistant with two primary roles:
            1. **General Conversationalist**: You function like ChatGPT, capable of engaging in natural, friendly, and informative conversations on a wide range of topics.
            2. **Expert in Requirements Management**: You specialize in analyzing, managing, and improving requirements for systems and projects, providing professional and technical insights.

            ---

            **HOW TO HANDLE USER QUERIES:**
            - If the query is related to **requirements management**, respond as a professional expert in the field.
            - If the query is **unrelated to requirements management**, respond as a general conversationalist (like ChatGPT).
            - If the query is **ambiguous**, ask clarifying questions to determine the appropriate role.

            ---

            **CURRENT CONTEXT: REQUIREMENTS MANAGEMENT**

            SYSTEM STATISTICS:
            - Total Requirements: {stats['total']}
            - Approved: {stats['approved']}
            - In Review: {stats['inReview']}
            - Disapproved: {stats['disapproved']}

            ALL REQUIREMENTS ({len(requirements)} total):
            {'\n\n'.join(requirements_context) if requirements else 'No requirements found'}

            USER QUERY: {user_message}

            ---

            **GUIDELINES FOR REQUIREMENTS MANAGEMENT RESPONSES:**
            1. Reference specific Requirement IDs when possible.
            2. Analyze requirement text for conflicts, duplicates, or ambiguities.
            3. Consider status, priority, and complexity in your analysis.
            4. Suggest specific improvements or optimizations.
            5. Highlight potential issues or risks.
            6. Keep the response concise and under 800 characters.

            ---

            **GUIDELINES FOR GENERAL CONVERSATION RESPONSES:**
            1. Respond in a friendly, engaging, and professional tone.
            2. Provide accurate and helpful information on a wide range of topics.
            3. If you don’t know the answer, admit it and offer to help find a solution.
            4. Maintain a natural and conversational flow.

            ---

            **RESPONSE:**
            """

            # Send the prompt to Gemini API
            payload = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }]
            }

            headers = {"Content-Type": "application/json"}
            response = requests.post(GEMINI_API_URL, json=payload, headers=headers)

            # Handle Gemini API errors
            if response.status_code != 200:
                logging.error(f"Gemini API Error: {response.text}")
                return jsonify({'error': 'API request failed', 'details': response.text}), 500

            # Parse Gemini's response
            gemini_response = response.json()
            bot_reply = gemini_response.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "I couldn't process that.")

            # Return the response
            return jsonify({
                "response": bot_reply,
                "stats": stats,
                "requirements_analyzed": len(requirements),
                "oldest_requirement": requirements[-1].ddate.isoformat() if requirements else None,
                "newest_requirement": requirements[0].ddate.isoformat() if requirements else None
            })

        except Exception as api_error:
            logging.error(f"API processing error: {str(api_error)}")
            return jsonify({"error": "Failed to process AI response"}), 500

    except Exception as e:
        logging.error(f"Chat error: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Failed to process request",
            "details": str(e)
        }), 500
    
@app.route('/api/requirements/<string:req_id>/status', methods=['PATCH'])
@limiter.limit("50 per hour")
def update_requirement_status(req_id):
    """
    Update the status of a requirement.
    """
    try:
        data = request.get_json()
        new_status = data.get('status')

        # Validate the new status
        if new_status not in [e.value for e in StatusEnum]:
            return jsonify({"error": "Invalid status"}), 400

        # Find the requirement by ID
        requirement = Requirement.query.get(req_id)
        if not requirement:
            return jsonify({"error": "Requirement not found"}), 404

        # Update the status
        requirement.status = StatusEnum(new_status)
        db.session.commit()

        return jsonify({
            "message": "Status updated successfully",
            "new_status": new_status
        }), 200
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error updating status: {str(e)}")
        return jsonify({"error": "Failed to update status"}), 500
    
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok",
        "database": "connected" if db.session.connection() else "disconnected",
        "ai_models": "loaded" if classifier else "unavailable",
        "gemini": "ready" if GEMINI_API_KEY else "missing_api_key"
    })

def validate_jira_credentials(site_url, email, api_token):
    """Validate Jira credentials by fetching user details."""
    try:
        response = requests.get(
            f"{site_url}/rest/api/3/myself",
            auth=HTTPBasicAuth(email, api_token),
            headers={"Accept": "application/json"}
        )
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Jira validation error: {str(e)}")
        return False

@app.route('/api/jira/connect', methods=['POST'])
def connect_to_jira():
    """Connect to Jira and validate credentials"""
    try:
        data = request.get_json()
        if not all(key in data for key in ['siteUrl', 'email', 'apiToken', 'projectKey']):
            return jsonify({"success": False, "error": "Missing required fields"}), 400

        # Validate credentials with Jira
        auth = HTTPBasicAuth(data['email'], data['apiToken'])
        test_response = requests.get(
            f"{data['siteUrl']}/rest/api/3/myself",
            auth=auth,
            timeout=10
        )
        test_response.raise_for_status()

        # Verify project exists
        project_response = requests.get(
            f"{data['siteUrl']}/rest/api/3/project/{data['projectKey']}",
            auth=auth,
            timeout=10
        )
        project_response.raise_for_status()

        # Store validated settings in session
        session['jira_settings'] = {
            'siteUrl': data['siteUrl'].rstrip('/'),
            'email': data['email'],
            'apiToken': data['apiToken'],
            'projectKey': data['projectKey']
        }
        
        return jsonify({
            "success": True,
            "user": test_response.json(),
            "project": project_response.json()
        })

    except requests.exceptions.RequestException as e:
        error_msg = "Jira connection failed: "
        if e.response:
            if e.response.status_code == 401:
                error_msg += "Invalid credentials"
            elif e.response.status_code == 404:
                error_msg += "Project not found"
            else:
                error_msg += f"HTTP {e.response.status_code}: {e.response.text}"
        else:
            error_msg += str(e)
        return jsonify({"success": False, "error": error_msg}), 401
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/jira/push', methods=['POST'])
def push_to_jira():
    """Push requirement to Jira with comprehensive validation"""
    try:
        # Session validation
        jira_settings = session.get('jira_settings')
        if not jira_settings:
            logging.error("Jira session missing")
            return jsonify({"success": False, "error": "Jira connection expired. Please reconnect."}), 401

        # Validate request data
        data = request.get_json()
        if not data or 'requirementId' not in data:
            return jsonify({"success": False, "error": "Missing requirement ID"}), 400
            
        requirement = Requirement.query.get(data['requirementId'])
        if not requirement:
            return jsonify({"success": False, "error": "Requirement not found"}), 404

        if requirement.status != StatusEnum.APPROVED:
            return jsonify({"success": False, "error": "Requirement must be approved first"}), 400

        # Configure Jira client
        auth = HTTPBasicAuth(jira_settings['email'], jira_settings['apiToken'])
        headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
        base_url = jira_settings['siteUrl'].rstrip('/')
        project_key = jira_settings['projectKey']

        # 1. Validate project exists and is accessible
        try:
            proj_response = requests.get(
                f"{base_url}/rest/api/3/project/{project_key}",
                auth=auth,
                headers=headers,
                timeout=10
            )
            proj_response.raise_for_status()
            project_id = proj_response.json().get('id')
        except requests.exceptions.HTTPError as e:
            error_msg = f"Project validation failed ({e.response.status_code}): "
            if e.response.status_code == 404:
                error_msg += f"Project '{project_key}' not found"
            else:
                error_msg += e.response.text
            return jsonify({"success": False, "error": error_msg}), 400

        # 2. Fetch available issue types with proper error handling
        try:
            issue_types_response = requests.get(
                f"{base_url}/rest/api/3/issuetype/project?projectId={project_id}",
                auth=auth,
                headers=headers,
                timeout=10
            )
            issue_types_response.raise_for_status()
            issue_types = [it['name'] for it in issue_types_response.json()]
            
            if not issue_types:
                return jsonify({
                    "success": False,
                    "error": f"No issue types available in project {project_key}. Create one in Jira first."
                }), 400
                
        except requests.exceptions.HTTPError as e:
            error_msg = f"Issue type fetch failed ({e.response.status_code}): {e.response.text}"
            return jsonify({"success": False, "error": error_msg}), 500

        # 3. Build payload with fallback values
        try:
            payload = {
                "fields": {
                    "project": {"key": project_key},
                    "summary": requirement.requirement[:250] or "New Requirement",
                    "issuetype": {"name": issue_types[0]},
                    "description": {
                        "type": "doc",
                        "version": 1,
                        "content": [{
                            "type": "paragraph",
                            "content": [{
                                "type": "text",
                                "text": requirement.requirement[:2000] or "Requirement description"
                            }]
                        }]
                    }
                }
            }

            # Add priority if configured in Jira
            if requirement.priority:
                jira_priority = Self.get_jira_priority(requirement.priority.value)
                if jira_priority:
                    payload['fields']['priority'] = {"name": jira_priority}

        except Exception as e:
            logging.error(f"Payload construction failed: {str(e)}")
            return jsonify({"success": False, "error": "Failed to build Jira payload"}), 500

        # 4. Create Jira issue with detailed error parsing
        try:
            response = requests.post(
                f"{base_url}/rest/api/3/issue",
                json=payload,
                auth=auth,
                headers=headers,
                timeout=20
            )
            response.raise_for_status()
            issue_data = response.json()
            return jsonify({
                "success": True,
                "issue": {
                    "key": issue_data['key'],
                    "url": f"{base_url}/browse/{issue_data['key']}"
                }
            })

        except requests.exceptions.HTTPError as e:
            error_msg = f"Jira API Error ({e.response.status_code}): "
            if e.response.status_code == 400:
                error_msg += "Invalid request format. Check:\n- Field values\n- Custom field requirements"
            elif e.response.status_code == 403:
                error_msg += "Permission denied. Verify user has 'Create Issues' permission."
            else:
                error_msg += e.response.text
            return jsonify({"success": False, "error": error_msg}), e.response.status_code

    except Exception as e:
        logging.error(f"Unexpected error: {traceback.format_exc()}")
        return jsonify({"success": False, "error": "Internal server error. Check logs."}), 500

def get_jira_priority(self, priority):
    """Dynamic priority mapping with fallback"""
    priority_map = {
        'High': 'Highest',
        'Medium': 'Medium',
        'Low': 'Low'
    }
    try:
        # Verify priority exists in Jira instance
        priorities_response = requests.get(
            f"{self.jira_settings['siteUrl']}/rest/api/3/priority",
            auth=HTTPBasicAuth(self.jira_settings['email'], self.jira_settings['apiToken'])
        )
        valid_priorities = [p['name'] for p in priorities_response.json()]
        return priority_map[priority] if priority_map.get(priority) in valid_priorities else None
    except:
        return None
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
