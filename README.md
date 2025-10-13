# AI Guard Agent - Room Security System

## Overview

AI Guard Agent is an intelligent room security system that utilizes pre-trained machine learning models to identify intruders and prevent privacy breaches. The system employs facial recognition technology to distinguish between authorized users and potential intruders, implementing escalation scenarios based on threat levels.

## Features

- **Face Recognition**: Uses pre-trained face recognition models to identify authorized users
- **Real-time Monitoring**: Continuous surveillance through webcam/camera feed
- **Intruder Detection**: Automatic detection of unauthorized individuals
- **Escalation Logic**: Implements security escalation based on threat assessment
- **User Registration**: Easy interface to register authorized faces

## System Architecture

The project consists of two main components:

1. **Face Registration Module** (`Face_recognition.py`): Allows users to register their faces in the system
2. **AI Guard Integration** (`AI_guard_integrated.py`): Main security monitoring system with intruder detection and response

## Prerequisites

### System Requirements

- **Python Version**: Python 3.11 (Required for face_recognition module compatibility)
- **Operating System**: Windows/Linux/macOS
- **Hardware**: 
  - Webcam or compatible camera device
  - Microphone for voice input
  - Speakers for audio output
- **FFmpeg**: Required for audio processing
- **Internet Connection**: Required for Google Gemini API and gTTS

### Dependencies

Core libraries used in this project:
- **faster-whisper**: Automatic Speech Recognition (ASR)
- **PyTorch**: Deep learning framework (CPU version)
- **face_recognition**: Face detection and recognition
- **OpenCV (cv2)**: Computer vision and video processing
- **dlib**: Machine learning toolkit for face recognition
- **google-generativeai**: Gemini LLM for intelligent conversation
- **gTTS**: Google Text-to-Speech
- **pydub**: Audio manipulation
- **pyaudio**: Audio recording
- **numpy**: Numerical operations
- Additional packages listed in `requirements.txt`

## Installation

### Step 1: Install Python 3.11

Download and install Python 3.11 from the official website:
[Python 3.11.8 Download](https://www.python.org/downloads/release/python-3118)

**Important**: Make sure to check "Add Python to PATH" during installation.

### Step 2: Install FFmpeg

FFmpeg is required for audio processing capabilities.

Follow the installation guide in this video tutorial:
[FFmpeg Installation Guide](https://youtu.be/K7znsMo_48I?si=-HXmmoZVc5bOmCEC)

To verify FFmpeg installation, open command prompt/terminal and run:
```bash
ffmpeg -version
```

### Step 3: Create Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

#### Using venv (Standard Python):
```bash
python -m venv ai_guard_env
```

#### Using Conda (Alternative):
```bash
conda create -n ai_guard_env python=3.11
```

### Step 4: Activate Virtual Environment

#### Windows:
```bash
ai_guard_env\Scripts\activate
```

#### macOS/Linux:
```bash
source ai_guard_env/bin/activate
```

#### Conda:
```bash
conda activate ai_guard_env
```

### Step 5: Install Dependencies

Execute the following commands in order:

```bash
# Upgrade pip to latest version
pip install --upgrade pip

# Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r requirements.txt
```

### Step 6: Install dlib (If Required)

If you encounter issues installing dlib through `requirements.txt`, use the pre-built wheel file:

```bash
pip install dlib-19.24.1-cp311-cp311-win_amd64.whl
```

**Note**: Ensure the wheel file path matches your actual file location.

## Configuration

### API Key Setup

Before running the system, you need a Google Gemini API key:

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Either:
   - Set it as an environment variable: `export GEMINI_API_KEY="your-key-here"`
   - Or enter it when prompted during first run
   - Or hardcode it in `guard_functions.py` (line 88)

## Usage

### Running the AI Guard System

Start the complete AI Guard system:

```bash
python main.py
```

The system will guide you through:

#### Phase 1: Face Enrollment
- If no faces are enrolled, you'll be prompted to enroll at least one trusted person
- Position yourself in front of the camera
- Press SPACE to capture (5 samples required)
- Press ESC to cancel
- Ensure good lighting conditions for best results

#### Phase 2: Voice Activation
- The system will ask you to speak the activation command
- Say: **"guard my room"**
- You have 3 attempts to activate the system

#### Phase 3: Monitoring Mode
- System continuously monitors the room via webcam
- Recognizes trusted persons (green boxes)
- Detects unknown persons (red boxes)
- Initiates conversation with unknown persons
- Implements intelligent escalation if needed

### Quick Test Mode

Skip activation and go straight to monitoring:

```bash
python main.py --test
```

This is useful for testing without voice activation each time.

### Keyboard Controls During Monitoring

- **'q'**: Quit the system completely
- **'s'**: Save a screenshot of the current frame

### System Behavior

**When Known Person Detected:**
- System welcomes them
- Pauses monitoring
- Returns to main menu

**When Unknown Person Detected:**
- System initiates conversation after detecting unknown face for ~2 seconds
- Conversation follows escalation levels:
  - **Level 1 (Polite)**: "Who are you? Why are you here?"
  - **Level 2 (Firm)**: "Please leave immediately."
  - **Level 3 (Warning)**: "Security has been alerted."
- Conversation log saved to `logs/` directory
- 60-second cooldown between conversations

## Project Structure

```
AI_Guard_agent_EE782/
│
├── main.py                      # Main entry point - orchestrates entire system
├── guard_functions.py           # Core functions (ASR, FR, TTS, LLM integration)
├── requirements.txt             # Python dependencies
├── dlib-19.24.1-cp311-cp311-win_amd64.whl  # dlib wheel file (Windows)
├── face_encodings.pkl          # Saved face encodings (generated)
├── README.md                    # Project documentation
│
├── known_faces/                # Images of enrolled persons (auto-created)
│   ├── John_1.jpg
│   ├── John_2.jpg
│   └── ...
│
├── logs/                       # Conversation logs (auto-created)
│   ├── conversation_20251012_143022.txt
│   └── ...
│
└── audio_recordings/           # Recorded audio files (auto-created)
    ├── recording_20251012_143025.wav
    └── ...
```

### File Descriptions

**main.py**
- Entry point of the application
- Manages workflow: setup → enrollment → activation → monitoring
- Implements milestone structure from assignment

**guard_functions.py**
- Contains all core functionality
- ASR functions (Whisper integration)
- TTS functions (gTTS integration)
- Face recognition functions
- LLM integration (Gemini)
- Conversation and escalation logic

**face_encodings.pkl**
- Serialized face encodings (automatically generated)
- Contains 128-dimensional vectors for each enrolled person
- Loaded at startup for face recognition

## How It Works

### System Architecture

The AI Guard Agent integrates four major AI components:

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERACTION                         │
│              (Voice Commands, Face Enrollment)               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   MAIN.PY (Orchestrator)                     │
│              Controls workflow and user interface            │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│     ASR     │ │     FR      │ │    LLM      │
│  (Whisper)  │ │(face_recog) │ │  (Gemini)   │
│             │ │             │ │             │
│ Speech→Text │ │ Face→Vector │ │ Context→    │
│             │ │ Compare     │ │ Response    │
└─────────────┘ └─────────────┘ └─────────────┘
        │              │              │
        └──────────────┼──────────────┘
                       │
                       ▼
              ┌────────────────┐
              │      TTS       │
              │    (gTTS)      │
              │                │
              │  Text→Speech   │
              └────────────────┘
```

### Component Interaction

**1. ASR (Automatic Speech Recognition)**
- **Technology**: Faster-Whisper (OpenAI's Whisper model)
- **Purpose**: Convert spoken words to text
- **Used For**: 
  - Voice activation ("guard my room")
  - Recording responses from unknown persons
- **Model**: Tiny (fast, optimized for real-time)

**2. Face Recognition**
- **Technology**: face_recognition library (dlib-based)
- **Purpose**: Identify persons by facial features
- **Process**:
  1. Detect faces in video frame
  2. Extract 128-dimensional face encoding
  3. Compare with stored encodings
  4. Match if distance < threshold
- **Used For**:
  - Distinguishing trusted vs unknown persons
  - Triggering appropriate responses

**3. LLM (Large Language Model)**
- **Technology**: Google Gemini 2.0 Flash Experimental
- **Purpose**: Intelligent conversation and analysis
- **Used For**:
  - Generating contextual responses to unknown persons
  - Analyzing cooperativeness of responses
  - Determining escalation level
  - Making threat assessments

**4. TTS (Text-to-Speech)**
- **Technology**: Google Text-to-Speech
- **Purpose**: Convert AI responses to speech
- **Used For**:
  - Speaking to users and unknown persons
  - Providing audio feedback
  - Announcements and warnings

### Workflow Details

#### Phase 1: Enrollment
1. Load existing face encodings from `face_encodings.pkl`
2. Display list of enrolled persons
3. Allow adding new trusted persons:
   - Capture 5 face samples via webcam
   - Extract face encodings from each sample
   - Average encodings for robustness
   - Save to disk

#### Phase 2: Voice Activation
1. System announces readiness via TTS
2. Record 3 seconds of audio
3. Transcribe with Whisper
4. Check for keyword "guard my room"
5. Activate if detected (max 3 attempts)

#### Phase 3: Monitoring
1. Capture video frames continuously
2. Every 2nd frame:
   - Detect faces using HOG/CNN
   - Extract face encodings
   - Compare with known encodings
   - Draw bounding boxes and labels
3. Track consecutive frames with unknown faces
4. Trigger conversation after threshold (10 frames)

#### Phase 4: Escalation Dialogue
1. **Level 1 - Polite Inquiry**:
   - "Who are you? Why are you here?"
   - Listen for response
   - LLM analyzes: cooperative? valid reason? leaving?
   - If cooperative with valid reason → inform and ask to leave
   - If uncooperative 3 times → escalate to Level 2

2. **Level 2 - Firm Request**:
   - "Please leave immediately."
   - More direct tone
   - Continue analyzing responses
   - If still uncooperative 3 times → escalate to Level 3

3. **Level 3 - Final Warning**:
   - "Security has been alerted."
   - Log incident
   - Mark as security alert
   - End conversation

### Face Encoding Process

Face recognition works by converting faces into numerical vectors:

```
Face Image → Face Detection → Landmark Detection → Encoding
              (Find face)      (68 facial points)   (128-D vector)

Example encoding: [0.234, -0.567, 0.123, ..., 0.891]
                  (128 floating-point numbers)
```

**Comparison Process**:
- Calculate Euclidean distance between encodings
- Distance < 0.6 = Match (same person)
- Distance > 0.6 = Different person

**Why average multiple samples?**
- Different angles, expressions, lighting
- Averaging reduces noise and improves accuracy
- More robust to variations

### Conversation Analysis

The LLM analyzes responses for:

1. **Cooperativeness**: Tone, willingness to engage
2. **Intent to Leave**: Explicit or implicit agreement
3. **Valid Reason**: Legitimate purpose (visitor, delivery, etc.)

Example Analysis:
```
User: "I'm here to see John, is he home?"
→ Cooperative: True
→ Valid Reason: True
→ Leaving: False
Response: "John isn't here. Please come back later."

User: "None of your business!"
→ Cooperative: False
→ Valid Reason: False
→ Leaving: False
Response: [Escalate to next level]
```

## Escalation Scenarios

The system implements various escalation levels based on intruder detection:

- **Level 1**: Initial detection and logging
- **Level 2**: Alert notification
- **Level 3**: Advanced response (implementation in progress)

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'google.generativeai'`
- **Solution**: Install the package: `pip install google-generativeai`

**Issue**: dlib installation fails with compilation errors
- **Solution**: Use the provided wheel file: `pip install dlib-19.24.1-cp311-cp311-win_amd64.whl`
- **Note**: Ensure you're using Python 3.11

**Issue**: Camera not detected or `Failed to capture frame`
- **Solution**: 
  - Check camera permissions in system settings
  - Ensure no other application is using the camera
  - Try specifying different camera index: modify `video_source=0` to `video_source=1` in monitor_room call

**Issue**: Face recognition accuracy is low
- **Solution**: 
  - Ensure good lighting (avoid backlighting)
  - Register multiple samples from different angles
  - Adjust recognition tolerance in config (lower = stricter)
  - Clean camera lens
  - Position face 2-3 feet from camera

**Issue**: No speech detected during voice activation
- **Solution**:
  - Check microphone permissions
  - Speak clearly and loudly
  - Reduce background noise
  - Verify microphone is working: `python -m pyaudio`
  - Try increasing recording duration

**Issue**: Audio playback not working (TTS)
- **Solution**:
  - Check speaker volume
  - Install audio backend: `sudo apt-get install ffmpeg` (Linux)
  - On Windows, ensure you have audio codec support
  - Error will be printed but won't crash the program

**Issue**: `KeyError` or `ImportError` after installation
- **Solution**: 
  - Verify virtual environment is activated
  - Reinstall all dependencies: `pip install -r requirements.txt --force-reinstall`
  - Check Python version: `python --version` (should be 3.11.x)

**Issue**: Gemini API key error
- **Solution**:
  - Verify API key is correct
  - Check API quota at [Google AI Studio](https://makersuite.google.com/)
  - Ensure internet connection is active
  - Set as environment variable: `export GEMINI_API_KEY="your-key"`

**Issue**: FFmpeg not found error
- **Solution**:
  - Verify installation: `ffmpeg -version`
  - Add FFmpeg to system PATH
  - Restart terminal/command prompt after installation
  - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

**Issue**: High CPU usage during monitoring
- **Solution**:
  - Increase `PROCESS_EVERY_N_FRAMES` in config (skip more frames)
  - Use smaller Whisper model size (already using 'tiny')
  - Close other applications
  - Consider reducing video resolution

**Issue**: Face encodings file not found on first run
- **Solution**: This is normal - file is created during first enrollment
- If error persists, check write permissions in project directory

**Issue**: Person not recognized despite enrollment
- **Solution**:
  - Re-enroll with better lighting
  - Capture more samples (5 minimum, 10 recommended)
  - Check if glasses/mask affecting recognition
  - Lower tolerance value: `FACE_RECOGNITION_TOLERANCE = 0.5`

**Issue**: Conversation logs contain garbled text
- **Solution**: Logs use UTF-8 encoding automatically
- If viewing in Notepad, use Notepad++ or VS Code
- Ensure terminal supports UTF-8

**Issue**: System freezes during enrollment
- **Solution**:
  - Press ESC to cancel
  - Ensure camera is working
  - Check for error messages in console
  - Restart the program

### Debug Mode

To enable detailed debug output, modify `guard_functions.py`:
- Add print statements before/after each major function
- Check `logs/` directory for conversation logs
- Review `audio_recordings/` for saved audio files
- Use `--test` mode to skip activation: `python main.py --test`

### Performance Optimization Tips

1. **Reduce frame processing**: Increase `PROCESS_EVERY_N_FRAMES` to 3 or 4
2. **Use smaller Whisper model**: Already using 'tiny' (fastest)
3. **Close unnecessary applications**: Free up CPU/RAM
4. **Improve lighting**: Better lighting = faster face detection
5. **Reduce video resolution**: Modify `cv2.resize()` factor in recognition function

## Future Enhancements

- Integration with Large Language Models (LLM) for intelligent escalation logic
- Multi-camera support
- Cloud storage for face encodings
- Mobile app notifications
- Advanced threat assessment algorithms
- Integration with smart home systems

## Technical Details

### Models Used
- Pre-trained face detection models
- Deep learning-based face recognition (dlib/face_recognition)
- Computer vision processing (OpenCV)

### Performance
- Real-time processing capability
- Low latency face detection
- Optimized for CPU execution

## Advanced Features

### Escalation System

The AI Guard implements a sophisticated three-level escalation system:

**Level 1: Initial Contact (Polite)**
- Objective: Identify the person and their purpose
- Tone: Professional and courteous
- Questions: "Who are you? Why are you here?"
- Timeout: 3 uncooperative responses before escalation

**Level 2: Firm Request (Stern)**
- Objective: Direct the person to leave
- Tone: Authoritative and serious
- Statement: "You are not authorized. Please leave immediately."
- Timeout: 3 uncooperative responses before final escalation

**Level 3: Final Warning (Alert)**
- Objective: Inform of consequences
- Tone: Very firm, no-nonsense
- Statement: "Security has been alerted. This incident is being logged."
- Action: Creates security alert in log

### Intelligent Response Analysis

The system uses Google Gemini LLM to analyze responses on multiple dimensions:
- **Cooperativeness**: Is the person being respectful and responsive?
- **Intent to Leave**: Are they agreeing to depart?
- **Valid Reason**: Do they have a legitimate purpose (visitor, delivery)?
- **Threat Level**: Implicit assessment of potential threat

### Conversation Cooldown

To prevent harassment, the system implements a 60-second cooldown between conversations with unknown persons. This ensures the system doesn't continuously interrogate someone who refuses to leave.

### Visual Feedback

During monitoring, the system provides real-time visual feedback:
- **Green boxes**: Known/trusted persons
- **Red boxes**: Unknown persons
- **Confidence scores**: Displayed for all recognitions (e.g., "John (0.87)")
- **Status text**: "All Clear", "Welcome - Known Person", "ALERT: Unknown Person"

## Example Usage Scenarios

### Scenario 1: Trusted Person Returns Home

```
1. System monitoring room
2. John (enrolled user) enters
3. Face detected and recognized: "John (0.91)"
4. System: "Welcome back! Room monitoring paused."
5. System returns to main menu
```

### Scenario 2: Delivery Person at Door

```
1. Unknown face detected for 2 seconds
2. System: "Hello, I don't recognize you. Who are you and why are you here?"
3. Person: "I have a package for Sarah"
4. System analyzes: cooperative=true, valid_reason=true
5. System: "Sarah isn't here. Please leave it at the door or come back later."
6. Person: "Okay, I'll leave it here"
7. System: "Thank you. Goodbye."
8. Conversation logged, monitoring resumes
```

### Scenario 3: Uncooperative Intruder

```
1. Unknown face detected
2. System: "Hello, I don't recognize you..."
3. Person: "None of your business"
4. System analyzes: cooperative=false
5. [Strike 1] System continues politely
6. Person: refuses to respond
7. [Strike 2] 
8. Person: hostile response
9. [Strike 3] ESCALATE TO LEVEL 2
10. System: "I'm not authorized to let you in. Leave immediately."
11. Person: continues uncooperative behavior
12. [3 more strikes] ESCALATE TO LEVEL 3
13. System: "This is your final warning. Security has been alerted."
14. Incident logged with security flag
```

## Project Statistics

- **Total Lines of Code**: ~1,200 lines
- **Main Components**: 2 files (main.py, guard_functions.py)
- **Functions**: 15+ core functions
- **AI Models Used**: 4 (Whisper, face_recognition, Gemini, gTTS)
- **Configuration Parameters**: 20+
- **Supported Languages**: English (expandable)

## Credits and Acknowledgments

### Technologies Used

- **OpenAI Whisper**: Speech recognition model
- **face_recognition by Adam Geitgey**: Face recognition library
- **dlib by Davis King**: Machine learning toolkit
- **Google Gemini**: Large language model
- **Google TTS**: Text-to-speech engine
- **OpenCV**: Computer vision library
- **PyTorch**: Deep learning framework

### Academic Context

- **Course**: EE782 - Advanced Topics in Machine Learning
- **Assignment**: Programming Assignment 2
- **Focus**: Integration of multiple AI technologies for practical application

### Learning Outcomes

Through this project, the following concepts were explored:
- Multi-modal AI system integration
- Real-time computer vision processing
- Natural language understanding and generation
- Speech recognition and synthesis
- State machine design for escalation logic
- Software architecture and modularity
- Error handling and graceful degradation

## Version History

**v1.0** (Current)
- Initial release
- Voice activation system
- Face recognition and enrollment
- Three-level escalation dialogue
- LLM integration
- Comprehensive logging

## Contact and Support

For questions, issues, or contributions related to this project:
- Check the troubleshooting section first
- Review the project report for technical details
- Examine conversation logs for debugging
- Contact project maintainer for specific clarifications

---

**Disclaimer**: This system is designed for educational purposes as part of EE782 coursework. It demonstrates the integration of multiple AI technologies but should not be considered a production-ready security system. Always comply with privacy laws and regulations when deploying surveillance systems.
