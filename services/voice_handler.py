import speech_recognition as sr
import pyttsx3
import threading
import time
import logging
import os
from vosk import Model, KaldiRecognizer
import json

logger = logging.getLogger(__name__)


class VoiceHandler:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Initialize TTS engines for different speakers
        self.interviewer_tts = pyttsx3.init()
        self.user_tts = pyttsx3.init()  # For user feedback if needed

        # Configure interviewer voice (professional, slightly slower)
        self.interviewer_tts.setProperty("rate", 140)  # Slightly slower for clarity
        self.interviewer_tts.setProperty("volume", 0.9)

        # Configure user voice (if needed for feedback)
        self.user_tts.setProperty("rate", 160)
        self.user_tts.setProperty("volume", 0.8)

        # Store selected voice id
        self.selected_interviewer_voice_id = None

        # Set different voices for interviewer and user
        self._configure_speaker_voices()

        # Speaker identification
        self.current_speaker = "interviewer"  # Default speaker
        self.speaker_history = []  # Track who spoke when

        # Adjust for ambient noise
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

        # Vosk model setup
        model_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../vosk-model-small-en-us-0.15")
        )
        self.vosk_model = Model(model_path)
        self.vosk_recognizer = None  # Will be created per stream

        logger.info("VoiceHandler initialized with multi-speaker support")

    def _configure_speaker_voices(self):
        """Configure different voices for interviewer and user"""
        try:
            voices = self.interviewer_tts.getProperty("voices")

            if voices and hasattr(voices, "__len__") and len(voices) >= 2:
                # Set interviewer to first voice (usually male)
                if hasattr(voices[0], "id"):
                    self.interviewer_tts.setProperty("voice", voices[0].id)
                    self.selected_interviewer_voice_id = voices[0].id
                # Set user to second voice (usually female) if available
                if hasattr(voices[1], "id"):
                    self.user_tts.setProperty("voice", voices[1].id)

                # Log voice names if available
                interviewer_name = (
                    getattr(voices[0], "name", "Unknown")
                    if hasattr(voices[0], "name")
                    else "Unknown"
                )
                user_name = (
                    getattr(voices[1], "name", "Unknown")
                    if hasattr(voices[1], "name")
                    else "Unknown"
                )
                logger.info(
                    f"Configured voices: Interviewer={interviewer_name}, User={user_name}"
                )
            else:
                # Fallback: use same voice but different properties
                logger.warning(
                    "Only one voice available, using different properties for speakers"
                )
                self.interviewer_tts.setProperty("rate", 140)
                self.user_tts.setProperty("rate", 160)
                self.selected_interviewer_voice_id = None

        except Exception as e:
            logger.error(f"Error configuring speaker voices: {e}")

    def listen_for_speech(self, timeout=1, phrase_time_limit=5):
        """Listen for speech and convert to text"""
        try:
            with self.microphone as source:
                # Listen for audio with timeout
                audio = self.recognizer.listen(
                    source, timeout=timeout, phrase_time_limit=phrase_time_limit
                )

            # Convert speech to text using Google's speech recognition
            try:
                text = self.recognizer.recognize_google(audio)
            except AttributeError:
                # Fallback if recognize_google is not available
                logger.warning("Google speech recognition not available")
                return None

            # Mark this as user speech
            self._record_speaker("user", text)

            return text.lower()

        except sr.WaitTimeoutError:
            # No speech detected within timeout
            return None
        except sr.UnknownValueError:
            # Speech was unintelligible
            return None
        except sr.RequestError as e:
            logger.error(f"Error with speech recognition service: {e}")
            return None

    def speak_interviewer_response(self, text):
        """Speak as the interviewer (AI)"""
        try:
            self.current_speaker = "interviewer"
            self._record_speaker("interviewer", text)

            # Use threading to avoid blocking
            speech_thread = threading.Thread(
                target=self._speak_interviewer_async, args=(text,)
            )
            speech_thread.daemon = True
            speech_thread.start()

            logger.info(f"Interviewer speaking: {text[:50]}...")

        except Exception as e:
            logger.error(f"Error with interviewer text-to-speech: {e}")

    def speak_user_feedback(self, text):
        """Speak as the user (for feedback/confirmation)"""
        try:
            self.current_speaker = "user"
            self._record_speaker("user", text)

            # Use threading to avoid blocking
            speech_thread = threading.Thread(
                target=self._speak_user_async, args=(text,)
            )
            speech_thread.daemon = True
            speech_thread.start()

            logger.info(f"User feedback: {text[:50]}...")

        except Exception as e:
            logger.error(f"Error with user text-to-speech: {e}")

    def _speak_interviewer_async(self, text):
        """Asynchronous speech function for interviewer"""
        try:
            # Always set the voice property before speaking
            if self.selected_interviewer_voice_id:
                self.interviewer_tts.setProperty(
                    "voice", self.selected_interviewer_voice_id
                )
            self.interviewer_tts.say(text)
            self.interviewer_tts.runAndWait()
        except Exception as e:
            logger.error(f"Error in interviewer TTS engine: {e}")

    def _speak_user_async(self, text):
        """Asynchronous speech function for user"""
        try:
            self.user_tts.say(text)
            self.user_tts.runAndWait()
        except Exception as e:
            logger.error(f"Error in user TTS engine: {e}")

    def _record_speaker(self, speaker, text):
        """Record who spoke and when"""
        timestamp = time.time()
        self.speaker_history.append(
            {
                "speaker": speaker,
                "text": text,
                "timestamp": timestamp,
                "time_str": time.strftime("%H:%M:%S", time.localtime(timestamp)),
            }
        )

        # Keep only last 100 entries to prevent memory bloat
        if len(self.speaker_history) > 100:
            self.speaker_history = self.speaker_history[-100:]

    def get_speaker_history(self, limit=20):
        """Get recent speaker history"""
        return self.speaker_history[-limit:] if self.speaker_history else []

    def get_current_speaker(self):
        """Get current speaker"""
        return self.current_speaker

    # Removed clear_speaker_history method - clearing is now UI-only

    def set_interviewer_voice_properties(self, rate=140, volume=0.9):
        """Set interviewer TTS voice properties"""
        self.interviewer_tts.setProperty("rate", rate)
        self.interviewer_tts.setProperty("volume", volume)
        logger.info(
            f"Interviewer voice properties updated: rate={rate}, volume={volume}"
        )

    def set_user_voice_properties(self, rate=160, volume=0.8):
        """Set user TTS voice properties"""
        self.user_tts.setProperty("rate", rate)
        self.user_tts.setProperty("volume", volume)
        logger.info(f"User voice properties updated: rate={rate}, volume={volume}")

    def list_available_voices(self):
        """List available TTS voices"""
        try:
            voices = self.interviewer_tts.getProperty("voices")
            if voices and hasattr(voices, "__iter__"):
                return [
                    (voice.id, voice.name)
                    for voice in voices
                    if hasattr(voice, "id") and hasattr(voice, "name")
                ]
            else:
                return []
        except Exception as e:
            logger.error(f"Error listing voices: {e}")
            return []

    def set_interviewer_voice(self, voice_id):
        """Set voice by ID"""
        try:
            self.interviewer_tts.setProperty("voice", voice_id)
            self.selected_interviewer_voice_id = voice_id
            logger.info(f"Voice set to: {voice_id}")
        except Exception as e:
            logger.error(f"Error setting voice: {e}")

    def set_user_voice(self, voice_id):
        """Set a specific voice for user"""
        try:
            self.user_tts.setProperty("voice", voice_id)
            logger.info(f"User voice set to: {voice_id}")
        except Exception as e:
            logger.error(f"Error setting user voice: {e}")

    def continuous_listen(self, callback, stop_event):
        """Continuously listen for speech and call callback with recognized text"""
        while not stop_event.is_set():
            try:
                text = self.listen_for_speech(timeout=1)
                if text:
                    # Pass speaker info to callback
                    callback(text, "user")
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage
            except Exception as e:
                logger.error(f"Error in continuous listening: {e}")
                time.sleep(1)  # Wait before retrying

    def speak_response(self, text):
        """Legacy method - now defaults to interviewer speaking"""
        self.speak_interviewer_response(text)

    def set_voice_properties(self, rate=150, volume=0.9):
        """Legacy method - now sets interviewer properties"""
        self.set_interviewer_voice_properties(rate, volume)

    def set_voice(self, voice_id):
        """Legacy method - now sets interviewer voice"""
        self.set_interviewer_voice(voice_id)

    def process_audio_chunk(self, audio_data):
        """Process audio chunk for real-time transcription using Vosk"""
        try:
            import numpy as np

            # Vosk expects 16kHz mono PCM 16-bit signed int
            # Convert float32 audio_data to int16
            int16_audio = (audio_data * 32767).astype(np.int16).tobytes()
            # Create recognizer if not exists
            if self.vosk_recognizer is None:
                self.vosk_recognizer = KaldiRecognizer(self.vosk_model, 16000)
            # Feed audio to recognizer
            if self.vosk_recognizer.AcceptWaveform(int16_audio):
                result = self.vosk_recognizer.Result()
                text = json.loads(result).get("text", "")
                return text if text else None
            else:
                partial = self.vosk_recognizer.PartialResult()
                text = json.loads(partial).get("partial", "")
                return text if text else None
        except Exception as e:
            logger.error(f"Error processing audio chunk with Vosk: {e}")
            return None
