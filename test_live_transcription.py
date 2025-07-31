#!/usr/bin/env python3
"""
Test script for live transcription using Whisper models
"""

import time
import threading
import logging
from services.voice_handler import VoiceHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def transcription_callback(text, speaker):
    """Callback function for transcription results"""
    print(f"\n🎤 [{speaker.upper()}] {text}")
    print("=" * 50)


def test_whisper_transcription():
    """Test Whisper transcription functionality"""
    print("🧪 Testing Whisper Transcription...")

    # Initialize voice handler with Whisper
    voice_handler = VoiceHandler(whisper_model_size="base", device="cpu")

    # Test basic Whisper transcription
    print("\n📝 Testing basic Whisper transcription...")
    print("Please speak something for 3 seconds...")

    result = voice_handler.test_whisper_transcription()
    if result:
        print(f"✅ Whisper test successful: {result}")
    else:
        print("❌ Whisper test failed - no text transcribed")
        return False

    return True


def test_live_transcription():
    """Test live transcription functionality"""
    print("\n🎙️ Testing Live Transcription...")
    print("Press Ctrl+C to stop")

    # Initialize voice handler
    voice_handler = VoiceHandler(whisper_model_size="base", device="cpu")

    # Create stop event for threading
    stop_event = threading.Event()

    try:
        # Start live transcription in a separate thread
        transcription_thread = threading.Thread(
            target=voice_handler.start_live_transcription,
            args=(transcription_callback, stop_event),
        )
        transcription_thread.daemon = True
        transcription_thread.start()

        print("🎤 Live transcription started! Speak now...")
        print("Press Ctrl+C to stop")

        # Keep the main thread alive
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Stopping live transcription...")
        stop_event.set()
        time.sleep(2)  # Give time for cleanup
        print("✅ Live transcription stopped")


def test_continuous_listen():
    """Test continuous listening with Whisper"""
    print("\n👂 Testing Continuous Listening with Whisper...")
    print("Press Ctrl+C to stop")

    # Initialize voice handler
    voice_handler = VoiceHandler(whisper_model_size="base", device="cpu")

    # Create stop event for threading
    stop_event = threading.Event()

    try:
        # Start continuous listening in a separate thread
        listen_thread = threading.Thread(
            target=voice_handler.continuous_listen,
            args=(transcription_callback, stop_event, True),  # use_whisper=True
        )
        listen_thread.daemon = True
        listen_thread.start()

        print("🎤 Continuous listening started! Speak now...")
        print("Press Ctrl+C to stop")

        # Keep the main thread alive
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Stopping continuous listening...")
        stop_event.set()
        time.sleep(2)  # Give time for cleanup
        print("✅ Continuous listening stopped")


def main():
    """Main test function"""
    print("🚀 Voice Handler Whisper Integration Test")
    print("=" * 50)

    # Test 1: Basic Whisper transcription
    if not test_whisper_transcription():
        print("❌ Basic Whisper test failed. Exiting.")
        return

    print("\n" + "=" * 50)

    # Test 2: Live transcription
    try:
        test_live_transcription()
    except Exception as e:
        print(f"❌ Live transcription test failed: {e}")

    print("\n" + "=" * 50)

    # Test 3: Continuous listening
    try:
        test_continuous_listen()
    except Exception as e:
        print(f"❌ Continuous listening test failed: {e}")

    print("\n✅ All tests completed!")


if __name__ == "__main__":
    main()
