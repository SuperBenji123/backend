from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import json
import time
import uuid
import os
import sys
from typing import Dict, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import the agent system
try:
    from agent_system import process_query, process_query_with_human_feedback, classify_user_intent
    logger.info("Successfully imported agent_system module")
except ImportError as e:
    logger.error(f"Failed to import agent_system module: {e}")
    raise

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# In-memory storage for sessions and pending approvals
# In a production app, this would be a database
sessions = {}
pending_approvals = {}

# Set to True to enable human approval for critical operations
HUMAN_APPROVAL_REQUIRED = True

# Request timing middleware
@app.before_request
def before_request():
    request.start_time = time.time()
    logger.info(f"Request received: {request.method} {request.path} from {request.remote_addr}")
    if request.is_json:
        # Log request data without sensitive information
        safe_data = {k: v for k, v in request.json.items() if k.lower() not in ('password', 'token', 'key', 'secret')}
        logger.debug(f"Request data: {json.dumps(safe_data)}")

@app.after_request
def after_request(response):
    if hasattr(request, 'start_time'):
        duration = time.time() - request.start_time
        logger.info(f"Request completed in {duration:.4f} seconds with status {response.status_code}")
    return response

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "message": "API is running"
    })

@app.route('/api/session', methods=['POST'])
def create_session():
    """Create a new user session"""
    try:
        # Generate a unique session ID
        session_id = str(uuid.uuid4())
        
        # Get user ID from request, or generate one
        data = request.json or {}
        user_id = data.get('user_id', f"user_{int(time.time())}")
        
        # Store session information
        sessions[session_id] = {
            "user_id": user_id,
            "created_at": time.time(),
            "last_active": time.time(),
            "assistant_id": None,
            "thread_id": None,
            "message_history": []
        }
        
        logger.info(f"Created new session {session_id} for user {user_id}")
        
        return jsonify({
            "success": True,
            "status": "approved",
            "result": result,
            "message": "Query was approved and processed."
        })
    
    except Exception as e:
        logger.error(f"Error handling approval: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    """List all active sessions (admin only)"""
    try:
        # In a real app, you would authenticate the admin here
        
        active_sessions = []
        for session_id, session_data in sessions.items():
            # Skip sessions that have been inactive for more than 24 hours
            if time.time() - session_data.get('last_active', 0) > 86400:  # 24 hours
                continue
                
            active_sessions.append({
                "session_id": session_id,
                "user_id": session_data.get("user_id"),
                "created_at": session_data.get("created_at"),
                "last_active": session_data.get("last_active"),
                "assistant_id": session_data.get("assistant_id"),
                "thread_id": session_data.get("thread_id"),
                "message_count": len(session_data.get("message_history", []))
            })
        
        return jsonify({
            "success": True,
            "sessions": active_sessions
        })
    
    except Exception as e:
        logger.error(f"Error listing sessions: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

# Clean up old sessions periodically (would be better as a background task)
@app.route('/api/cleanup', methods=['POST'])
def cleanup_sessions():
    """Clean up old sessions and pending approvals"""
    try:
        # In a real app, you would authenticate the admin here
        
        # Clear old sessions (inactive for more than 24 hours)
        now = time.time()
        removed_sessions = 0
        session_ids = list(sessions.keys())
        
        for session_id in session_ids:
            last_active = sessions[session_id].get('last_active', 0)
            if now - last_active > 86400:  # 24 hours
                del sessions[session_id]
                removed_sessions += 1
        
        # Clear old pending approvals (older than 48 hours)
        removed_approvals = 0
        approval_ids = list(pending_approvals.keys())
        
        for approval_id in approval_ids:
            timestamp = pending_approvals[approval_id].get('timestamp', 0)
            if now - timestamp > 172800:  # 48 hours
                del pending_approvals[approval_id]
                removed_approvals += 1
        
        return jsonify({
            "success": True,
            "removed_sessions": removed_sessions,
            "removed_approvals": removed_approvals
        })
    
    except Exception as e:
        logger.error(f"Error cleaning up resources: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

# Development-only endpoint for viewing logs
@app.route('/api/logs', methods=['GET'])
def view_logs():
    """View recent logs (admin only, development only)"""
    try:
        # In a real app, you would authenticate the admin here
        # And should only allow this in development mode
        
        # Check if we're in debug mode before allowing access
        if not app.debug:
            return jsonify({"error": "This endpoint is only available in debug mode"}), 403
            
        with open('api.log', 'r') as f:
            logs = f.readlines()
        
        # Get the most recent logs (up to 100 lines)
        recent_logs = logs[-100:]
        
        return jsonify({
            "success": True,
            "logs": recent_logs
        })
    
    except Exception as e:
        logger.error(f"Error viewing logs: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development' or os.environ.get('DEBUG') == 'true'
    
    logger.info(f"Starting Flask API on port {port} with debug={debug_mode}")
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
