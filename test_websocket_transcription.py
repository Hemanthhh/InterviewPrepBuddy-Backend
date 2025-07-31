#!/usr/bin/env python3
"""
Test script for websocket audio transcription using Whisper
"""

import time
import logging
import numpy as np
from services.voice_handler import VoiceHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def test_websocket_audio_transcription():
    """Test websocket audio transcription with different data types"""
    print("🧪 Testing Websocket Audio Transcription...")

    # Initialize voice handler
    voice_handler = VoiceHandler(whisper_model_size="base", device="cpu")

    # Test 1: Simulate websocket audio data as bytes
    print("\n📝 Test 1: Audio data as bytes...")

    # Create dummy audio data (1 second of silence at 16kHz)
    sample_rate = 16000
    duration = 1  # seconds
    samples = int(sample_rate * duration)

    # Create some test audio (sine wave)
    t = np.linspace(0, duration, samples, False)
    test_audio = np.sin(2 * np.pi * 440 * t) * 0.1  # 440 Hz sine wave, low volume

    # Convert to 16-bit PCM bytes
    audio_bytes = (test_audio * 32767).astype(np.int16).tobytes()

    # Test transcription
    result = voice_handler.transcribe_websocket_audio(audio_bytes, "test_socket_1")
    if result:
        print(f"✅ Bytes transcription: {result}")
    else:
        print("❌ Bytes transcription failed (expected for silence/sine wave)")

    # Test 2: Audio data as numpy array
    print("\n📝 Test 2: Audio data as numpy array...")

    # Create test audio with some speech-like characteristics
    # This is just a test - real speech would be much more complex
    speech_like = np.random.randn(samples) * 0.01  # Low amplitude noise

    result = voice_handler.transcribe_websocket_audio(speech_like, "test_socket_2")
    if result:
        print(f"✅ Numpy array transcription: {result}")
    else:
        print("❌ Numpy array transcription failed (expected for noise)")

    # Test 3: Invalid data type
    print("\n📝 Test 3: Invalid data type...")

    invalid_data = "not audio data"
    result = voice_handler.transcribe_websocket_audio(invalid_data, "test_socket_3")
    if result is None:
        print("✅ Invalid data type handled correctly")
    else:
        print("❌ Invalid data type not handled correctly")

    # Test 4: Empty data
    print("\n📝 Test 4: Empty data...")

    empty_data = b""
    result = voice_handler.transcribe_websocket_audio(empty_data, "test_socket_4")
    if result is None:
        print("✅ Empty data handled correctly")
    else:
        print("❌ Empty data not handled correctly")

    print("\n✅ Websocket transcription tests completed!")


def test_real_audio_transcription():
    """Test with real audio recording"""
    print("\n🎙️ Testing Real Audio Transcription...")

    voice_handler = VoiceHandler(whisper_model_size="base", device="cpu")

    print("Please speak something for 3 seconds...")
    result = voice_handler.test_whisper_transcription()

    if result:
        print(f"✅ Real audio transcription: {result}")

        # Test the same audio as websocket data
        print("\n📝 Testing same audio as websocket data...")

        # This would normally come from a websocket
        # For testing, we'll simulate it
        try:
            # Read the test file if it exists
            import os

            if os.path.exists("test_whisper.wav"):
                with open("test_whisper.wav", "rb") as f:
                    audio_bytes = f.read()

                result2 = voice_handler.transcribe_websocket_audio(
                    audio_bytes, "real_audio_test"
                )
                if result2:
                    print(f"✅ Websocket transcription of real audio: {result2}")
                else:
                    print("❌ Websocket transcription of real audio failed")
        except Exception as e:
            print(f"❌ Error testing real audio as websocket data: {e}")
    else:
        print("❌ Real audio transcription failed")


def main():
    """Main test function"""
    print("🚀 Websocket Audio Transcription Test")
    print("=" * 50)

    # Test websocket audio transcription
    test_websocket_audio_transcription()

    # Test with real audio
    test_real_audio_transcription()

    print("\n✅ All tests completed!")


if __name__ == "__main__":
    main()
