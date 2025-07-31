import wave
import pyaudio
import speech_recognition as sr
import pyttsx3
import threading
import time
import logging
import os
import numpy as np
from faster_whisper import WhisperModel

# Suppress faster_whisper library logs
logging.getLogger("faster_whisper").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class VoiceHandler:
    def __init__(self, whisper_model_size=None, device=None):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Use environment variables or default values
        whisper_model_size = whisper_model_size or os.getenv(
            "WHISPER_MODEL_SIZE", "base"
        )
        device = device or os.getenv("WHISPER_DEVICE", "cpu")

        # Initialize faster-whisper model
        self.whisper_model = WhisperModel(
            whisper_model_size,
            device=device,
            compute_type="int8" if device == "cpu" else "float16",
        )

        logger.info(
            f"Whisper model loaded with size: {whisper_model_size}, device: {device}"
        )

        # Buffer for streaming ASR
        self._asr_buffer = bytes()
        self._sample_rate = 16000  # faster-whisper default

        # Rolling buffer for maintaining context
        self._rolling_buffer = bytes()
        self._last_transcription_time = time.time()
        self._context_window = 30.0  # Keep 30 seconds of context

        # Silence detection parameters
        self._silence_threshold = 0.01  # Amplitude threshold for silence
        self._silence_duration = 1.0  # Seconds of silence to trigger transcription
        self._last_speech_time = time.time()
        self._is_speaking = False
        self._max_chunk_duration = 20.0  # Maximum chunk duration in seconds

        # Audio recording parameters
        self.chunk_size = 4096  # Increased from 1024 to 4096
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000

        # Initialize TTS engines for different speakers
        try:
            self.interviewer_tts = pyttsx3.init()
            self.user_tts = pyttsx3.init()  # For user feedback if needed
        except Exception as e:
            logger.warning(
                f"TTS initialization failed: {e}. TTS features will be disabled."
            )
            self.interviewer_tts = None
            self.user_tts = None

        # Configure interviewer voice (professional, slightly slower)
        if self.interviewer_tts:
            self.interviewer_tts.setProperty("rate", 140)  # Slightly slower for clarity
            self.interviewer_tts.setProperty("volume", 0.9)

        # Configure user voice (if needed for feedback)
        if self.user_tts:
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

        logger.info(
            "VoiceHandler initialized with multi-speaker support and Whisper integration"
        )

    def _configure_speaker_voices(self):
        """Configure different voices for interviewer and user"""
        if not self.interviewer_tts or not self.user_tts:
            logger.warning("TTS engines not available, skipping voice configuration")
            return

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
        """Listen for speech and convert to text using Google Speech Recognition"""
        try:
            with self.microphone as source:
                # Listen for audio with timeout
                audio = self.recognizer.listen(
                    source, timeout=timeout, phrase_time_limit=phrase_time_limit
                )

            # Convert speech to text using Google's speech recognition
            try:
                text = self.recognizer.recognize_google(audio)
                logger.info(f"Speech recognition completed: {text[:50]}...")
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

    def listen_with_whisper(self, timeout=1, phrase_time_limit=5):
        """Listen for speech and convert to text using Whisper"""
        try:
            with self.microphone as source:
                # Listen for audio with timeout
                audio = self.recognizer.listen(
                    source, timeout=timeout, phrase_time_limit=phrase_time_limit
                )

            # Convert audio to numpy array
            audio_data = np.frombuffer(audio.frame_data, dtype=np.int16)
            audio_data = audio_data.astype(np.float32) / 32767.0

            # Transcribe using Whisper
            try:
                segments, _ = self.whisper_model.transcribe(
                    audio_data, language="en", vad_filter=True, word_timestamps=False
                )

                text = " ".join([seg.text for seg in segments]).strip()

                if text:
                    logger.info(f"Whisper transcription completed: {text[:50]}...")
                    # Mark this as user speech
                    self._record_speaker("user", text)
                    return text.lower()
                # else:
                #     logger.debug("Whisper transcription completed but no text found")
                #     return None
                return None

            except Exception as e:
                logger.error(f"Error with Whisper transcription: {e}")
                return None

        except sr.WaitTimeoutError:
            # No speech detected within timeout
            return None
        except sr.UnknownValueError:
            # Speech was unintelligible
            return None
        except Exception as e:
            logger.error(f"Error with speech recognition: {e}")
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
        if not self.interviewer_tts:
            logger.warning("TTS engine not available, skipping speech")
            return

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
        if not self.user_tts:
            logger.warning("User TTS engine not available, skipping speech")
            return

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

    def set_interviewer_voice_properties(self, rate=140, volume=0.9):
        """Set interviewer TTS voice properties"""
        if not self.interviewer_tts:
            logger.warning("Interviewer TTS engine not available")
            return

        self.interviewer_tts.setProperty("rate", rate)
        self.interviewer_tts.setProperty("volume", volume)
        logger.info(
            f"Interviewer voice properties updated: rate={rate}, volume={volume}"
        )

    def set_user_voice_properties(self, rate=160, volume=0.8):
        """Set user TTS voice properties"""
        if not self.user_tts:
            logger.warning("User TTS engine not available")
            return

        self.user_tts.setProperty("rate", rate)
        self.user_tts.setProperty("volume", volume)
        logger.info(f"User voice properties updated: rate={rate}, volume={volume}")

    def list_available_voices(self):
        """List available TTS voices"""
        if not self.interviewer_tts:
            logger.warning("TTS engine not available")
            return []

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
        if not self.interviewer_tts:
            logger.warning("Interviewer TTS engine not available")
            return

        try:
            self.interviewer_tts.setProperty("voice", voice_id)
            self.selected_interviewer_voice_id = voice_id
            logger.info(f"Voice set to: {voice_id}")
        except Exception as e:
            logger.error(f"Error setting voice: {e}")

    def set_user_voice(self, voice_id):
        """Set a specific voice for user"""
        if not self.user_tts:
            logger.warning("User TTS engine not available")
            return

        try:
            self.user_tts.setProperty("voice", voice_id)
            logger.info(f"User voice set to: {voice_id}")
        except Exception as e:
            logger.error(f"Error setting user voice: {e}")

    def continuous_listen(self, callback, stop_event, use_whisper=True):
        """Continuously listen for speech and call callback with recognized text"""
        while not stop_event.is_set():
            try:
                if use_whisper:
                    text = self.listen_with_whisper(timeout=1)
                else:
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

    def configure_silence_detection(
        self,
        silence_threshold=0.01,
        silence_duration=1.0,
        max_chunk_duration=20.0,
        context_window=30.0,
    ):
        """Configure silence detection parameters for better sentence completion

        Args:
            silence_threshold (float): Audio amplitude threshold for silence detection (0.0-1.0)
            silence_duration (float): Seconds of silence to trigger transcription
            max_chunk_duration (float): Maximum chunk duration in seconds before forced transcription
            context_window (float): Seconds of audio context to maintain for better transcription
        """
        self._silence_threshold = silence_threshold
        self._silence_duration = silence_duration
        self._max_chunk_duration = max_chunk_duration
        self._context_window = context_window
        logger.info(
            f"Silence detection configured: threshold={silence_threshold}, duration={silence_duration}s, max_chunk={max_chunk_duration}s, context={context_window}s"
        )

    def process_audio_chunk(self, audio_data):
        """Process audio chunk using rolling buffer to maintain context

        Args:
            audio_data (numpy.ndarray): NumPy float32 mono block audio data

        Returns:
            str or None: Transcribed text if complete phrase is available, None otherwise
        """
        try:
            # Convert NumPy float32 audio to 16-bit PCM bytes
            int16_audio = (audio_data * 32767).astype(np.int16).tobytes()
            current_time = time.time()

            # Add to rolling buffer (maintains context)
            self._rolling_buffer += int16_audio

            # Calculate audio amplitude for silence detection
            audio_amplitude = np.abs(audio_data).mean()

            # Check if there's speech activity
            if audio_amplitude > self._silence_threshold:
                # Speech detected
                if not self._is_speaking:
                    self._is_speaking = True
                    logger.debug("Speech started")
                self._last_speech_time = current_time
            else:
                # Silence detected
                if self._is_speaking:
                    # Check if we've had enough silence to consider the sentence complete
                    silence_elapsed = current_time - self._last_speech_time
                    if silence_elapsed >= self._silence_duration:
                        self._is_speaking = False
                        logger.debug(
                            f"Speech ended after {silence_elapsed:.2f}s of silence"
                        )

            # Only transcribe if we have enough audio and speech has ended
            if (
                len(self._rolling_buffer)
                > self._sample_rate * 2  # At least 2 seconds of audio
                and not self._is_speaking
                and current_time - self._last_speech_time >= self._silence_duration
            ):

                try:
                    # Convert rolling buffer to numpy array for Whisper
                    audio_array = (
                        np.frombuffer(self._rolling_buffer, dtype=np.int16).astype(
                            np.float32
                        )
                        / 32767.0
                    )

                    # Transcribe using faster-whisper
                    segments, _ = self.whisper_model.transcribe(
                        audio_array,
                        language="en",
                        vad_filter=True,
                        word_timestamps=False,
                    )

                    # Extract text from segments
                    text = " ".join([seg.text for seg in segments]).strip()

                    if text:
                        logger.info(f"Transcription completed: {text[:50]}...")
                        # Update last transcription time
                        self._last_transcription_time = current_time
                        # Keep recent audio for context (last 30 seconds)
                        self._maintain_context_window()
                        return text
                    else:
                        # No text found, maintain context and continue
                        self._maintain_context_window()
                        return None

                except Exception as transcription_error:
                    logger.error(f"Error during transcription: {transcription_error}")
                    # Maintain context even on error
                    self._maintain_context_window()
                    return None

            # Check for buffer overrun (e.g., > 20 seconds of audio)
            buffer_overrun_threshold = (
                self._sample_rate * self._max_chunk_duration * 2
            )  # 20 seconds
            if len(self._rolling_buffer) > buffer_overrun_threshold:
                logger.warning(
                    f"Buffer overrun: processing {len(self._rolling_buffer)} bytes to prevent memory issues"
                )
                # Force transcription to prevent memory issues
                try:
                    audio_array = (
                        np.frombuffer(self._rolling_buffer, dtype=np.int16).astype(
                            np.float32
                        )
                        / 32767.0
                    )
                    segments, _ = self.whisper_model.transcribe(
                        audio_array,
                        language="en",
                        vad_filter=True,
                        word_timestamps=False,
                    )
                    text = " ".join([seg.text for seg in segments]).strip()
                    self._last_transcription_time = current_time
                    self._maintain_context_window()
                    return text if text else None
                except Exception as e:
                    logger.error(f"Error during forced transcription: {e}")
                    self._maintain_context_window()
                    return None

            # No transcription needed yet
            return None

        except Exception as e:
            logger.error(f"Error in process_audio_chunk: {e}")
            return None

    def _maintain_context_window(self):
        """Maintain a rolling context window of recent audio"""
        try:
            current_time = time.time()
            time_since_last_transcription = current_time - self._last_transcription_time

            # If we haven't transcribed recently, keep more context
            if (
                time_since_last_transcription > 5.0
            ):  # 5 seconds since last transcription
                # Keep last 30 seconds of audio for context
                max_context_bytes = int(self._sample_rate * self._context_window * 2)
                if len(self._rolling_buffer) > max_context_bytes:
                    # Keep the most recent audio
                    self._rolling_buffer = self._rolling_buffer[-max_context_bytes:]
                    logger.debug(
                        f"Maintained context window: {len(self._rolling_buffer)} bytes"
                    )
            else:
                # Recent transcription, can be more aggressive with cleanup
                # Keep last 10 seconds for immediate context
                max_context_bytes = int(self._sample_rate * 10 * 2)
                if len(self._rolling_buffer) > max_context_bytes:
                    self._rolling_buffer = self._rolling_buffer[-max_context_bytes:]
                    logger.debug(
                        f"Reduced context window: {len(self._rolling_buffer)} bytes"
                    )

        except Exception as e:
            logger.error(f"Error maintaining context window: {e}")
            # Fallback: keep last 10 seconds
            max_context_bytes = int(self._sample_rate * 10 * 2)
            if len(self._rolling_buffer) > max_context_bytes:
                self._rolling_buffer = self._rolling_buffer[-max_context_bytes:]

    def clear_audio_buffers(self):
        """Clear all audio buffers to reset context for next question"""
        try:
            self._rolling_buffer = b""
            self._asr_buffer = b""
            self._is_speaking = False
            self._last_speech_time = time.time()
            self._last_transcription_time = time.time()
            logger.info("Audio buffers cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing audio buffers: {e}")

    def record_chunk(self, p, stream, file_path, chunk_length=1):
        """Record audio chunk to WAV file"""
        try:
            frames = []
            for _ in range(0, int(self.rate / self.chunk_size * chunk_length)):
                data = stream.read(self.chunk_size)
                frames.append(data)

            wf = wave.open(file_path, "wb")
            wf.setnchannels(self.channels)
            wf.setsampwidth(p.get_sample_size(self.audio_format))  # Fixed typo
            wf.setframerate(self.rate)
            wf.writeframes(b"".join(frames))
            wf.close()

            # logger.debug(f"Audio chunk recorded to {file_path}")

        except Exception as e:
            logger.error(f"Error recording audio chunk: {e}")

    def transcribe_chunk(self, model, file_path):
        """Transcribe audio chunk using Whisper"""
        try:
            segments, _ = model.transcribe(
                file_path, language="en", vad_filter=True, word_timestamps=False
            )

            text = " ".join([seg.text for seg in segments]).strip()
            return text if text else None

        except Exception as e:
            logger.error(f"Error transcribing chunk: {e}")
            return None

    def transcribe_audio_data(self, audio_data):
        """Transcribe audio data (bytes, numpy array, or file path) using Whisper"""
        try:
            # Handle different input types
            if isinstance(audio_data, bytes):
                # Convert bytes to numpy array
                audio_array = (
                    np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                    / 32767.0
                )
            elif isinstance(audio_data, np.ndarray):
                # Ensure it's float32 and normalized
                if audio_data.dtype != np.float32:
                    audio_array = audio_data.astype(np.float32)
                else:
                    audio_array = audio_data
                # Normalize if not already
                if audio_array.max() > 1.0 or audio_array.min() < -1.0:
                    audio_array = audio_array / 32767.0
            elif isinstance(audio_data, str):
                # Treat as file path
                return self.transcribe_chunk(self.whisper_model, audio_data)
            else:
                logger.error(f"Unsupported audio data type: {type(audio_data)}")
                return None

            # Transcribe using Whisper
            segments, _ = self.whisper_model.transcribe(
                audio_array, language="en", vad_filter=True, word_timestamps=False
            )

            text = " ".join([seg.text for seg in segments]).strip()
            return text if text else None

        except Exception as e:
            logger.error(f"Error transcribing audio data: {e}")
            return None

    def transcribe_websocket_audio(self, audio_data, socket_id=None):
        """Transcribe audio data from websocket connections"""
        try:
            # if socket_id:
            #     logger.debug(f"Processing audio from socket: {socket_id}")

            # Handle websocket audio data (usually bytes)
            if isinstance(audio_data, bytes):
                # Try to convert to numpy array
                try:
                    # Assume 16-bit PCM audio
                    audio_array = (
                        np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                        / 32767.0
                    )
                except Exception as e:
                    logger.error(f"Error converting websocket audio bytes: {e}")
                    return None
            elif isinstance(audio_data, np.ndarray):
                audio_array = audio_data.astype(np.float32)
                if audio_array.max() > 1.0 or audio_array.min() < -1.0:
                    audio_array = audio_array / 32767.0
            else:
                logger.error(
                    f"Unsupported websocket audio data type: {type(audio_data)}"
                )
                return None

            # Check if audio array is valid
            if len(audio_array) == 0:
                # logger.debug("Empty audio array received")
                return None

            # Transcribe using Whisper
            segments, _ = self.whisper_model.transcribe(
                audio_array, language="en", vad_filter=True, word_timestamps=False
            )

            text = " ".join([seg.text for seg in segments]).strip()

            if text:
                logger.info(f"Websocket transcription: {text[:50]}...")
                # Mark this as user speech
                self._record_speaker("user", text)

            return text if text else None

        except Exception as e:
            logger.error(f"Error transcribing websocket audio: {e}")
            return None

    def start_live_transcription(self, callback, stop_event):
        """Start live transcription using PyAudio and Whisper"""
        try:
            p = pyaudio.PyAudio()
            stream = p.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk_size,
            )

            logger.info("Live transcription started")

            try:
                while not stop_event.is_set():
                    chunk_file = "temp_chunk.wav"
                    self.record_chunk(
                        p, stream, chunk_file, chunk_length=20
                    )  # Increased to 20 seconds for longer sentences

                    # Transcribe the chunk
                    transcription = self.transcribe_chunk(
                        self.whisper_model, chunk_file
                    )

                    if transcription:
                        logger.info(f"Live transcription: {transcription[:50]}...")
                        # Mark this as user speech
                        self._record_speaker("user", transcription)
                        # Call the callback with the transcribed text
                        callback(transcription.lower(), "user")

                    # Clean up temporary file
                    try:
                        os.remove(chunk_file)
                    except OSError:
                        pass

            except KeyboardInterrupt:
                logger.info("Live transcription stopped by user")
            except Exception as e:
                logger.error(f"Error in live transcription loop: {e}")
            finally:
                # Clean up resources
                stream.stop_stream()
                stream.close()
                p.terminate()
                logger.info("Live transcription resources cleaned up")

        except Exception as e:
            logger.error(f"Error starting live transcription: {e}")

    def test_whisper_transcription(self):
        """Test Whisper transcription with a short audio recording"""
        try:
            logger.info("Testing Whisper transcription...")

            # Record a short audio sample
            p = pyaudio.PyAudio()
            stream = p.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk_size,
            )

            test_file = "test_whisper.wav"
            self.record_chunk(p, stream, test_file, chunk_length=3)  # 3 seconds

            stream.stop_stream()
            stream.close()
            p.terminate()

            # Transcribe the test file
            transcription = self.transcribe_chunk(self.whisper_model, test_file)

            # Clean up
            try:
                os.remove(test_file)
            except OSError:
                pass

            if transcription:
                logger.info(f"Whisper test successful: {transcription}")
                return transcription
            else:
                logger.warning("Whisper test completed but no text was transcribed")
                return None

        except Exception as e:
            logger.error(f"Error testing Whisper transcription: {e}")
            return None
