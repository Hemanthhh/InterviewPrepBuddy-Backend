# Logging and Configuration Improvements

This document describes the enhanced logging and configuration features for the Whisper transcription system in the Interview Prep Buddy application.

## Enhanced Logging

The VoiceHandler class now includes comprehensive logging for better monitoring and debugging:

### 1. Whisper Model Load Logging
- **Location**: `VoiceHandler.__init__()`
- **Log Level**: `INFO`
- **Message Format**: "Whisper model loaded with size: {model_size}, device: {device}"
- **Purpose**: Track which Whisper model size and device are being used during initialization

### 2. Transcription Event Logging
- **Location**: `VoiceHandler.process_audio_chunk()` and `VoiceHandler.listen_for_speech()`
- **Log Level**: `INFO`
- **Message Format**: "Transcription completed: {first_50_chars}..."
- **Purpose**: Log each successful transcription with a preview of the transcribed text

### 3. Buffer Management Warnings
- **Buffer Underrun**: 
  - **Log Level**: `DEBUG`
  - **Message**: "Buffer underrun: current size {size}, threshold {threshold}"
  - **Purpose**: Track when audio buffer doesn't have enough data for transcription
  
- **Buffer Overrun**:
  - **Log Level**: `WARNING`
  - **Message**: "Buffer overrun: buffer size {size} bytes exceeds threshold"
  - **Purpose**: Alert when audio buffer accumulates too much data (>5 seconds)

### 4. Additional Logging
- Voice configuration logging at startup
- TTS voice property changes
- Speech recognition errors and warnings
- Transcription errors and recovery

## Configuration Options

The Whisper model can now be configured through environment variables or constructor parameters:

### Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# Whisper Model Configuration
WHISPER_MODEL_SIZE=base
WHISPER_DEVICE=cpu
```

### Available Model Sizes
- `tiny`: Fastest, least accurate (~39 MB)
- `base`: Good balance of speed and accuracy (~74 MB) - **Default**
- `small`: Better accuracy, slower (~244 MB)
- `medium`: High accuracy (~769 MB)
- `large-v1`: Highest accuracy, slowest (~1550 MB)
- `large-v2`: Latest large model (~1550 MB)
- `large-v3`: Most recent large model (~1550 MB)

### Available Devices
- `cpu`: Use CPU for inference - **Default**
- `cuda`: Use NVIDIA GPU (requires CUDA-compatible GPU)
- `auto`: Automatically select best available device

### Constructor Parameters

You can also pass configuration directly to the VoiceHandler constructor:

```python
# Use specific model size and device
voice_handler = VoiceHandler(whisper_model_size="small", device="cuda")

# Use environment variables (default behavior)
voice_handler = VoiceHandler()
```

## Performance Considerations

### Model Size vs Performance
- **tiny/base**: Good for real-time applications, lower CPU usage
- **small/medium**: Better for accuracy-critical applications
- **large models**: Best accuracy but require more RAM and processing time

### Device Selection
- **CPU**: Universal compatibility, moderate performance
- **CUDA**: Significantly faster on compatible NVIDIA GPUs
- **auto**: Automatically selects CUDA if available, falls back to CPU

## Monitoring and Debugging

### Log Level Configuration
The application uses Python's standard logging module. You can adjust log levels in `app.py`:

```python
# Current configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

# For more detailed debugging, use:
logging.basicConfig(level=logging.DEBUG)
```

### Key Log Messages to Monitor
1. **Model Loading**: Confirm correct model size and device
2. **Transcription Events**: Monitor transcription frequency and content
3. **Buffer Warnings**: Watch for performance issues
4. **Error Messages**: Identify and resolve transcription failures

## Example Configuration Files

### .env (Production)
```bash
WHISPER_MODEL_SIZE=base
WHISPER_DEVICE=cpu
```

### .env (High-accuracy setup)
```bash
WHISPER_MODEL_SIZE=medium
WHISPER_DEVICE=cuda
```

### .env (Low-resource setup)
```bash
WHISPER_MODEL_SIZE=tiny
WHISPER_DEVICE=cpu
```

## Troubleshooting

### Common Issues
1. **Model loading errors**: Check if specified model size is valid
2. **CUDA errors**: Ensure NVIDIA drivers and CUDA toolkit are installed
3. **Buffer overruns**: May indicate audio processing bottlenecks
4. **Transcription failures**: Check microphone permissions and audio quality

### Debug Steps
1. Enable DEBUG logging to see detailed buffer information
2. Monitor model loading messages for configuration verification
3. Check transcription event logs for processing frequency
4. Watch for buffer warning patterns indicating performance issues
