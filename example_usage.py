#!/usr/bin/env python3
"""
Example usage of VoiceHandler with Whisper integration
"""

import time
import threading
import logging
from services.voice_handler import VoiceHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def handle_transcription(text, speaker):
    """Handle transcribed text"""
    print(f"\n🎤 [{speaker.upper()}] {text}")

    # Example: Echo back the transcribed text
    if speaker == "user":
        # Simulate AI response
        response = f"I heard you say: {text}"
        print(f"🤖 [AI] {response}")


def main():
    """Main example function"""
    print("🎙️ Voice Handler with Whisper - Example Usage")
    print("=" * 60)

    # Initialize voice handler with Whisper
    print("🔧 Initializing VoiceHandler with Whisper...")
    voice_handler = VoiceHandler(
        whisper_model_size="base",  # Options: tiny, base, small, medium, large
        device="cpu",  # Options: cpu, cuda, auto
    )

    print("✅ VoiceHandler initialized successfully!")
    print("\nAvailable features:")
    print("1. Live transcription with Whisper")
    print("2. Continuous listening with Whisper")
    print("3. Text-to-speech for interviewer and user")
    print("4. Speaker history tracking")

    # Test basic transcription
    print("\n🧪 Testing basic transcription...")
    print("Please speak something for 3 seconds...")

    result = voice_handler.test_whisper_transcription()
    if result:
        print(f"✅ Transcription successful: {result}")
    else:
        print("❌ Transcription failed")
        return

    # Start live transcription
    print("\n🎙️ Starting live transcription...")
    print("Speak naturally and press Ctrl+C to stop")

    # Create stop event for threading
    stop_event = threading.Event()

    try:
        # Start live transcription in a separate thread
        transcription_thread = threading.Thread(
            target=voice_handler.start_live_transcription,
            args=(handle_transcription, stop_event),
        )
        transcription_thread.daemon = True
        transcription_thread.start()

        # Keep the main thread alive
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Stopping transcription...")
        stop_event.set()
        time.sleep(2)  # Give time for cleanup

        # Show speaker history
        print("\n📊 Speaker History:")
        history = voice_handler.get_speaker_history(limit=10)
        for entry in history:
            print(f"  [{entry['time_str']}] {entry['speaker']}: {entry['text']}")

        print("\n✅ Example completed!")


if __name__ == "__main__":
    main()
