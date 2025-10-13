# AI Guard Agent - Quick Start Guide

Get up and running with AI Guard in 15 minutes!

## Prerequisites Checklist

Before starting, ensure you have:

- [ ] Python 3.11 installed
- [ ] Webcam connected and working
- [ ] Microphone connected and working
- [ ] Speakers/headphones connected
- [ ] Internet connection (for Gemini API and gTTS)
- [ ] Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

## Installation (5 minutes)

### Step 1: Download and Extract

```bash
# Clone or download the repository
git clone https://github.com/AdiTheGeek/AI_Guard_agent_EE782.git
cd AI_Guard_agent_EE782
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv ai_guard_env

# Activate it
# Windows:
ai_guard_env\Scripts\activate

# macOS/Linux:
source ai_guard_env/bin/activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt

# Install dlib (if needed)
pip install dlib-19.24.1-cp311-cp311-win_amd64.whl
```

### Step 4: Install FFmpeg

**Windows**: 
1. Download from [ffmpeg.org](https://ffmpeg.org/download.html)
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to PATH
4. Verify: `ffmpeg -version`

**macOS**: 
```bash
brew install ffmpeg
```

**Linux**: 
```bash
sudo apt update
sudo apt install ffmpeg
```

## First Run (10 minutes)

### Step 1: Set Up API Key

Option A - Environment Variable (Recommended):
```bash
# Windows (PowerShell):
$env:GEMINI_API_KEY="your-api-key-here"

# macOS/Linux:
export GEMINI_API_KEY="your-api-key-here"
```

Option B - Enter When Prompted:
The system will ask for your API key on first run.

### Step 2: Run the System

```bash
python main.py
```

### Step 3: Enroll Your Face (First Time)

1. When prompted, enter your name
2. Press SPACE to capture images (5 times)
3. Look at different angles for each capture
4. Wait for "Successfully enrolled" message

**Tips for good enrollment**:
- Face the camera directly
- Ensure good lighting
- Stay 2-3 feet from camera
- Capture different angles (straight, left, right)
- Remove glasses, then add one capture with glasses

### Step 4: Voice Activation

1. System will say: "Do you want me to start guarding the room?"
2. Say clearly: **"guard my room"**
3. You have 3 attempts
4. If activated, system announces: "Guard mode activated"

### Step 5: Start Monitoring

1. Press Enter when ready
2. System starts monitoring via webcam
3. You'll see:
   - Your face with green box: "Welcome - Known Person"
   - Status text showing "All Clear"

### Step 6: Test the System

**Test Unknown Face Detection**:
1. Step out of frame
2. Have someone else enter (or show a photo)
3. System detects unknown face (red box)
4. After ~2 seconds, conversation begins

**Test Keyboard Controls**:
- Press **'s'** to save a screenshot
- Press **'q'** to quit

## Quick Command Reference

```bash
# Normal mode (full workflow)
python main.py

# Test mode (skip activation)
python main.py --test

# Check installations
python --version        # Should show 3.11.x
pip list               # Show installed packages
ffmpeg -version        # Verify FFmpeg
```

## File Structure After First Run

```
AI_Guard_agent_EE782/
├── main.py
├── guard_functions.py
├── requirements.txt
├── face_encodings.pkl          ← Created after enrollment
├── known_faces/                ← Created, contains face images
│   ├── YourName_1.jpg
│   ├── YourName_2.jpg
│   └── ...
├── logs/                       ← Created when conversation happens
│   └── conversation_20251012_143022.txt
└── audio_recordings/           ← Created during conversations
    └── recording_20251012_143025.wav
```

## Common First-Run Issues

### "No module named 'faster_whisper'"
```bash
pip install faster-whisper
```

### "Camera not found" or Black Screen
- Close other apps using camera (Zoom, Teams, etc.)
- Check camera permissions in system settings
- Try unplugging and replugging camera

### Voice Activation Not Working
- Check microphone permissions
- Speak louder and clearer
- Reduce background noise
- Say "guard my room" (not "God my room" or similar)

### Face Not Recognized
- Re-enroll with better lighting
- Ensure you're 2-3 feet from camera
- Capture more varied samples

### "API key error" or Gemini Issues
- Verify your API key is correct
- Check internet connection
- Verify API quota at Google AI Studio

## Testing Escalation (Optional)

Want to see the full escalation system in action?

### Simulate Unknown Person Conversation:

1. Enroll yourself first
2. Start monitoring
3. Step away from camera
4. Have friend/family member enter
5. System initiates conversation
6. Have them respond uncooperatively 3 times
7. Watch escalation to Level 2, then Level 3

**Example Uncooperative Responses**:
- "Why should I tell you?"
- "None of your business"
- "I'm not leaving"
- [Silence/no response]

### Check Conversation Logs:

After conversation ends:
```bash
# View logs
cd logs
cat conversation_*.txt  # Linux/Mac
type conversation_*.txt # Windows

# Or open in text editor
notepad logs/conversation_20251012_143022.txt
```

## Tips for Best Results

### Face Recognition:
✓ Good, even lighting (avoid backlighting)
✓ Clean camera lens
✓ Face camera directly during enrollment
✓ Stay 2-3 feet from camera
✓ Enroll without glasses, then add one with glasses

### Voice Activation:
✓ Speak clearly and at normal volume
✓ Reduce background noise
✓ Say complete phrase: "guard my room"
✓ Wait for recording prompt before speaking

### System Performance:
✓ Close unnecessary applications
✓ Use good lighting (helps face detection speed)
✓ Allow 1-2 seconds for LLM responses
✓ Don't move too quickly (gives system time to recognize)

## Next Steps

Once you have the basic system running:

1. **Add More Users**: 
   - Run `python main.py`
   - When asked "Add another trusted person?", say yes
   - Enroll family members or roommates

2. **Review Logs**:
   - Check `logs/` for conversation records
   - Understand escalation patterns
   - Identify areas for improvement

3. **Customize Settings**:
   - Edit `guard_functions.py`
   - Adjust `GuardConfig` parameters:
     - `FACE_RECOGNITION_TOLERANCE` (default: 0.6)
     - `UNKNOWN_FACE_THRESHOLD` (default: 10 frames)
     - `UNCOOPERATIVE_THRESHOLD` (default: 3 strikes)

4. **Experiment**:
   - Test different lighting conditions
   - Try different voice commands
   - Test conversation scenarios
   - Adjust escalation timing

## Performance Optimization

If system is slow:

```python
# In guard_functions.py, increase these values:

PROCESS_EVERY_N_FRAMES = 3  # Skip more frames (default: 2)
WHISPER_MODEL_SIZE = "tiny"  # Already optimal
```

If recognition is inaccurate:

```python
# In guard_functions.py:

FACE_RECOGNITION_TOLERANCE = 0.5  # Stricter (default: 0.6)
# Or
FACE_RECOGNITION_TOLERANCE = 0.7  # More lenient
```

## Getting Help

1. **Check Documentation**:
   - README.md - Full documentation
   - Project Report - Technical details
   - This file - Quick start

2. **Common Issues**:
   - See "Troubleshooting" in README.md
   - Check logs in `logs/` directory

3. **Debug Mode**:
   - Watch console output for error messages
   - Check audio_recordings/ for transcription quality
   - Verify face_encodings.pkl exists

## Useful Commands Summary

```bash
# Environment Management
python -m venv ai_guard_env
ai_guard_env\Scripts\activate    # Windows
source ai_guard_env/bin/activate # Mac/Linux

# Installation
pip install --upgrade pip
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Running
python main.py              # Full system
python main.py --test       # Skip activation

# Verification
python --version            # Check Python version
pip list                    # List installed packages
ffmpeg -version             # Check FFmpeg
```

## What to Expect

**Normal Operation Timeline**:
```
[0:00] Start program
[0:05] Models loaded (Whisper, Gemini)
[0:10] Face enrollment complete
[0:15] Voice activation successful
[0:16] Monitoring begins

[Real-time] Known face detected → Welcome message
[Real-time] Unknown face → Wait 2 seconds → Conversation
[Conversation] 3-5 exchanges, 15-30 seconds total
[End] Logs saved, system continues monitoring
```

**Resource Usage**:
- CPU: 25-40% during monitoring
- RAM: ~600MB
- Disk: Minimal (logs are small text files)
- Network: Only during LLM API calls

## Success Indicators

You'll know the system is working correctly when:

✓ Models load without errors (~5 seconds)
✓ Face enrollment succeeds (green checkmark)
✓ Voice activation works (1-3 attempts)
✓ Your face shows green box with name
✓ Unknown faces show red box
✓ Conversations are logged in `logs/`
✓ System speaks responses audibly

## Emergency Stop

If something goes wrong:

1. **Press 'q'** during monitoring
2. **Press Ctrl+C** in terminal (emergency stop)
3. **Close the video window**
4. **Kill Python process** (Task Manager/Activity Monitor)

The system will clean up and save logs before exiting.

## Ready to Start!

You're now ready to run the AI Guard Agent system. Follow the steps above, and you should have a working intelligent room monitoring system in about 15 minutes.

**Remember**: This is an educational project. Use responsibly and ensure compliance with privacy laws!

---

**Quick Help**: If stuck at any step, check the full README.md or the project report for detailed explanations.
