from flask import Flask, request, jsonify
from flask_cors import CORS 
from agent_system import process_query

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "https://superbenji-webapp.vercel.app"}})

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok",
        "message": "API is running"
    })

@app.route('/api/query', methods=['POST'])
def query_agent():
    try:
        data = request.json
        user_query = data.get('query', '')
        
        if not user_query:
            return jsonify({"error": "No query provided"}), 400
            
        # Process the query using our agent system
        result = process_query(user_query)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/agent/supervisor', methods=['POST'])
def query_supervisor():
    payload = request.get_json()  # Extract JSON data from the request

    return jsonify({
        "Status": "Query Received",
        "ReceivedData": payload  # Echo back the received payload for debugging
    })
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
