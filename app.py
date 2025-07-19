from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import threading
import time
from services.resume_processor import ResumeProcessor
from services.screen_reader import ScreenReader
from services.voice_handler import VoiceHandler
from services.ai_assistant import AIAssistant
from services.job_analyzer import JobAnalyzer
import json
import logging
from werkzeug.exceptions import RequestEntityTooLarge
from flask_socketio import SocketIO, emit
from flask import request
import warnings

warnings.filterwarnings("ignore", message="resource_tracker: There appear to be")

app = Flask(__name__)
CORS(app)

# Set max upload size to 16 MB
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# Configuration
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize services
resume_processor = ResumeProcessor()
screen_reader = ScreenReader()
voice_handler = VoiceHandler()
ai_assistant = AIAssistant()
job_analyzer = JobAnalyzer()

# Global state
app_state = {
    "resume_data": None,
    "job_description": None,
    "interview_active": False,
    "listening": False,
    "screen_monitoring": False,
    "responses": [],  # Store AI responses
    "microphone_thread": None,  # Track microphone monitoring thread
    "ai_speech_enabled": False,  # Track AI speech setting - disabled by default
}

# Add after app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


@app.errorhandler(Exception)
def handle_exception(e):
    logging.error(f"Unhandled Exception: {e}")
    return jsonify({"error": str(e)}), 500


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    logging.error("File too large")
    return jsonify({"error": "File too large. Max size is 16 MB."}), 413


@app.route("/upload-resume", methods=["POST"])
def upload_resume():
    try:
        if "resume" not in request.files:
            logging.warning("No file provided in upload-resume")
            return jsonify({"error": "No file provided"}), 400

        file = request.files["resume"]
        if file.filename == "":
            logging.warning("No file selected in upload-resume")
            return jsonify({"error": "No file selected"}), 400

        # Save file
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        logging.info(f"Resume file saved: {filepath}")

        # Process resume
        resume_text = resume_processor.extract_text(filepath)
        resume_data = resume_processor.parse_resume(resume_text)

        app_state["resume_data"] = resume_data
        logging.info("Resume processed and data stored in app_state")

        return jsonify(
            {"message": "Resume uploaded successfully", "resume_data": resume_data}
        )

    except Exception as e:
        logging.error(f"Error in upload_resume: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/set-job-description", methods=["POST"])
def set_job_description():
    try:
        data = request.json
        job_description = data.get("job_description")

        if not job_description:
            logging.warning("No job description provided")
            return jsonify({"error": "Job description is required"}), 400

        app_state["job_description"] = job_description
        logging.info("Job description set in app_state")

        # Analyze job requirements
        job_analysis = job_analyzer.analyze_job(job_description)
        logging.info("Job description analyzed")

        return jsonify(
            {
                "message": "Job description set successfully",
                "job_analysis": job_analysis,
            }
        )

    except Exception as e:
        logging.error(f"Error in set_job_description: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/start-interview", methods=["POST"])
def start_interview():
    try:
        if not app_state["resume_data"] or not app_state["job_description"]:
            logging.warning(
                "Attempt to start interview without resume or job description"
            )
            return jsonify({"error": "Resume and job description are required"}), 400

        app_state["interview_active"] = True
        app_state["listening"] = True
        app_state["screen_monitoring"] = False  # Disabled by default
        logging.info("Interview started")

        # Start background services
        threading.Thread(target=monitor_interview, daemon=True).start()

        # Start microphone monitoring if not already running
        if (
            app_state["microphone_thread"] is None
            or not app_state["microphone_thread"].is_alive()
        ):
            app_state["microphone_thread"] = threading.Thread(
                target=monitor_microphone, daemon=True
            )
            app_state["microphone_thread"].start()
            logging.info("Microphone monitoring thread started for interview")

        return jsonify({"message": "Interview started successfully"})

    except Exception as e:
        logging.error(f"Error in start_interview: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/stop-interview", methods=["POST"])
def stop_interview():
    try:
        app_state["interview_active"] = False
        app_state["listening"] = False
        app_state["screen_monitoring"] = False
        logging.info("Interview stopped")

        return jsonify({"message": "Interview stopped successfully"})

    except Exception as e:
        logging.error(f"Error in stop_interview: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/get-status")
def get_status():
    logging.debug("Status requested")
    return jsonify(
        {
            "interview_active": app_state["interview_active"],
            "listening": app_state["listening"],
            "screen_monitoring": app_state["screen_monitoring"],
            "has_resume": app_state["resume_data"] is not None,
            "has_job_description": app_state["job_description"] is not None,
            "microphone_status": (
                "Listening..." if app_state["listening"] else "Stopped"
            ),
        }
    )


@app.route("/get-available-models")
def get_available_models():
    """Get list of available AI models from Ollama"""
    try:
        models = ai_assistant.get_available_models()
        return jsonify({"models": models})
    except Exception as e:
        logging.error(f"Error getting available models: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/set-model", methods=["POST"])
def set_model():
    """Set the current AI model"""
    try:
        data = request.json
        model_name = data.get("model_name")

        if not model_name:
            return jsonify({"error": "Model name is required"}), 400

        success = ai_assistant.set_model(model_name)
        if success:
            return jsonify({"message": f"Model set to {model_name} successfully"})
        else:
            return jsonify({"error": "Failed to set model"}), 400

    except Exception as e:
        logging.error(f"Error setting model: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/get-current-model")
def get_current_model():
    """Get the current AI model name"""
    try:
        model_name = ai_assistant.get_current_model()
        return jsonify({"model_name": model_name})
    except Exception as e:
        logging.error(f"Error getting current model: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/get-responses")
def get_responses():
    logging.debug("Responses requested")
    return jsonify({"responses": app_state["responses"]})


@app.route("/get-speaker-history")
def get_speaker_history():
    """Get recent speaker history with timestamps"""
    try:
        history = voice_handler.get_speaker_history(limit=20)
        return jsonify(
            {
                "speaker_history": history,
                "current_speaker": voice_handler.get_current_speaker(),
            }
        )
    except Exception as e:
        logging.error(f"Error getting speaker history: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/set-interviewer-voice", methods=["POST"])
def set_interviewer_voice():
    """Set interviewer voice properties"""
    try:
        data = request.json
        rate = data.get("rate", 140)
        volume = data.get("volume", 0.9)
        voice_id = data.get("voice_id")

        if voice_id:
            voice_handler.set_interviewer_voice(voice_id)

        voice_handler.set_interviewer_voice_properties(rate, volume)

        return jsonify({"message": "Interviewer voice updated successfully"})
    except Exception as e:
        logging.error(f"Error setting interviewer voice: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/get-available-voices")
def get_available_voices():
    """Get list of available TTS voices"""
    try:
        voices = voice_handler.list_available_voices()
        return jsonify({"voices": voices})
    except Exception as e:
        logging.error(f"Error getting available voices: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/get-ai-speech-settings")
def get_ai_speech_settings():
    """Get current AI speech settings"""
    try:
        return jsonify({"ai_speech_enabled": app_state["ai_speech_enabled"]})
    except Exception as e:
        logging.error(f"Error getting AI speech settings: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/set-ai-speech-settings", methods=["POST"])
def set_ai_speech_settings():
    """Set AI speech settings"""
    try:
        data = request.json
        ai_speech_enabled = data.get("ai_speech_enabled", True)

        app_state["ai_speech_enabled"] = ai_speech_enabled
        logging.info(f"AI speech setting updated: {ai_speech_enabled}")

        return jsonify({"message": "AI speech settings updated successfully"})
    except Exception as e:
        logging.error(f"Error setting AI speech settings: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/test-connection")
def test_connection():
    """Test endpoint to verify server is running"""
    return jsonify({"status": "ok", "message": "Server is running"})


# Removed start_microphone endpoint - now handled by WebSocket


# Removed stop_microphone endpoint - now handled by WebSocket


@app.route("/toggle-screen-monitoring", methods=["POST"])
def toggle_screen_monitoring():
    """Toggle screen monitoring on/off"""
    try:
        current_state = app_state["screen_monitoring"]
        app_state["screen_monitoring"] = not current_state

        logging.info(
            f"Screen monitoring toggled {'ON' if app_state['screen_monitoring'] else 'OFF'}"
        )

        return jsonify(
            {
                "message": f"Screen monitoring {'started' if app_state['screen_monitoring'] else 'stopped'} successfully",
                "screen_monitoring": app_state["screen_monitoring"],
            }
        )
    except Exception as e:
        logging.error(f"Error toggling screen monitoring: {e}")
        return jsonify({"error": str(e)}), 500


# Removed clear-responses endpoint - clearing is now UI-only


@app.route("/test-voice", methods=["POST"])
def test_voice():
    """Test the interviewer voice by speaking the provided text (if AI speech is enabled)"""
    try:
        data = request.json
        text = data.get("text", "This is a test of the interviewer voice settings.")
        if app_state["ai_speech_enabled"]:
            voice_handler.speak_interviewer_response(text)
        return jsonify(
            {
                "message": "Test voice triggered",
                "spoken": app_state["ai_speech_enabled"],
            }
        )
    except Exception as e:
        logging.error(f"Error in test_voice: {e}")
        return jsonify({"error": str(e)}), 500


def monitor_microphone():
    """Background thread to monitor microphone independently"""
    logging.info("Microphone monitoring started")
    while True:
        try:
            # Only listen if listening is enabled
            if app_state["listening"]:
                audio_text = voice_handler.listen_for_speech()
                if audio_text:
                    logging.info(f"Audio detected: {audio_text}")

                    # Check if it's a question
                    if is_question(audio_text):
                        # Generate and speak response
                        response = ai_assistant.generate_response(
                            question=audio_text,
                            resume_data=app_state["resume_data"],
                            job_description=app_state["job_description"],
                        )

                        logging.info(f"AI Response: {response}")

                        # Always add to speaker history (even if not spoken)
                        voice_handler._record_speaker("interviewer", response)

                        # Only speak if AI speech is enabled
                        if app_state["ai_speech_enabled"]:
                            voice_handler.speak_interviewer_response(response)

                        app_state["responses"].append(response)  # Store the response

            time.sleep(0.5)  # Check more frequently for better responsiveness

        except Exception as e:
            logging.error(f"Error in monitor_microphone: {e}")
            time.sleep(1)


def monitor_interview():
    """Background thread to monitor interview"""
    while app_state["interview_active"]:
        try:
            # Monitor screen for questions
            if app_state["screen_monitoring"]:
                screen_text = screen_reader.capture_and_read()
                if screen_text:
                    # Process screen text for potential questions
                    questions = extract_questions_from_text(screen_text)
                    for question in questions:
                        logging.info(f"Screen question detected: {question}")

                        # Add screen-detected question to speaker history
                        voice_handler._record_speaker("user", question)

                        # Generate response for screen questions too
                        response = ai_assistant.generate_response(
                            question=question,
                            resume_data=app_state["resume_data"],
                            job_description=app_state["job_description"],
                        )
                        logging.info(f"AI Response: {response}")

                        # Always add to speaker history (even if not spoken)
                        voice_handler._record_speaker("interviewer", response)

                        # Only speak if AI speech is enabled
                        if app_state["ai_speech_enabled"]:
                            voice_handler.speak_interviewer_response(response)

                        app_state["responses"].append(response)

            time.sleep(1)  # Prevent excessive CPU usage

        except Exception as e:
            logging.error(f"Error in monitor_interview: {e}")
            time.sleep(1)


def extract_questions_from_text(text):
    """Extract potential questions from screen text"""
    questions = []
    sentences = text.split(".")

    for sentence in sentences:
        sentence = sentence.strip()
        if sentence.endswith("?") or any(
            word in sentence.lower()
            for word in [
                "what",
                "how",
                "why",
                "when",
                "where",
                "tell me",
                "describe",
                "explain",
            ]
        ):
            questions.append(sentence)

    return questions


def is_question(text):
    """Check if text contains a question"""
    question_indicators = [
        "?",
        "what",
        "how",
        "why",
        "when",
        "where",
        "tell me",
        "describe",
        "explain",
        "can you",
    ]
    return any(indicator in text.lower() for indicator in question_indicators)


# Add WebSocket event handlers at the end of the file:
@socketio.on("audio_stream")
def handle_audio_stream(data):
    """Handle real-time audio streaming for speech-to-text"""
    print(f"audio_stream from socket id: {request.sid}")
    try:
        audio_chunk = data.get("audio_chunk", [])
        if not audio_chunk:
            return

        # Convert audio chunk back to proper format for processing
        import numpy as np

        audio_data = np.array(audio_chunk, dtype=np.int16).astype(np.float32) / 32767.0

        # Process audio chunk for transcription
        # For now, we'll simulate real-time transcription
        # In a full implementation, you'd use a streaming STT service
        transcription_text = voice_handler.process_audio_chunk(audio_data)
        print(f"Transcription result: {transcription_text}")  # Debug log
        if transcription_text:
            socketio.emit("transcription", {"text": transcription_text}, to=request.sid)
            print(f"Emitting transcription: {transcription_text}")

    except Exception as e:
        print(f"Error processing audio stream: {e}")


# Manual test event for debugging
@socketio.on("test_transcription")
def handle_test_transcription():
    emit("transcription", {"text": "This is a test transcription!"})


@socketio.on("question")
def handle_question(data):
    """Handle questions and generate streaming AI responses"""
    try:
        question = data.get("question", "")
        if not question:
            return

        # Emit user question for real-time UI update
        emit("user_question", {"question": question}, broadcast=True)

        logging.info(f"Processing question via WebSocket: {question}")

        # Generate AI response with streaming
        response_generator = ai_assistant.generate_streaming_response(
            question=question,
            resume_data=app_state.get("resume_data"),
            job_description=app_state.get("job_description"),
        )

        # Stream the response in chunks
        full_response = ""
        try:
            for chunk in response_generator:
                if chunk:
                    emit("ai_response", {"text": chunk})
                    full_response += chunk
            # Send completion event
            emit("ai_response_complete", {"status": "success"})
            # Add to speaker history
            voice_handler._record_speaker("user", question)
            voice_handler._record_speaker("interviewer", full_response)
        except Exception as e:
            logging.error(f"Error during response streaming: {e}")
            emit("ai_response_error", {"error": str(e)})
    except Exception as e:
        logging.error(f"Error processing question: {e}")
        emit("ai_response_error", {"error": "Failed to generate response"})


if __name__ == "__main__":
    socketio.run(app, debug=True, host="0.0.0.0", port=5001)
