"""
AI Guard Agent - Integration Functions
============================================
This module provides the core functions that integrate ASR (Automatic Speech Recognition),
FR (Face Recognition), TTS (Text-to-Speech), and LLM (Large Language Model) components.
Each function is designed to be called from main.py for clean separation of concerns.

Components:
- Audio Recording & Speech Recognition (Whisper)
- Text-to-Speech (gTTS)
- Face Recognition (face_recognition library)
- LLM Integration (Google Gemini)
- Conversation & Escalation Logic
"""

# ============================================================================
# IMPORTS
# ============================================================================

import cv2  # OpenCV - for video capture and image processing
import numpy as np  # NumPy - for numerical operations on arrays
import pickle  # For serializing/deserializing face encodings
import os  # For file and directory operations
from pathlib import Path  # For modern path handling
from datetime import datetime  # For timestamps in logs
import time  # For delays and timing operations
import pyaudio  # For audio recording from microphone
import wave  # For saving audio in WAV format
from faster_whisper import WhisperModel  # ASR model for speech-to-text
from gtts import gTTS  # Google Text-to-Speech
from pydub import AudioSegment  # For audio file manipulation
from pydub.playback import play  # For playing audio files
import tempfile  # For creating temporary files

# Import face recognition components
import face_recognition  # High-level face recognition library

# LLM imports - Google Gemini for intelligent conversation
try:
    import google.generativeai as genai
except ImportError:
    # If not installed, set to None and handle later
    genai = None


# ============================================================================
# CONFIGURATION & GLOBALS
# ============================================================================

class GuardConfig:
    """
    Configuration class for AI Guard system
    
    This class centralizes all configuration parameters for the system,
    making it easy to adjust settings without modifying code throughout
    the application.
    """
    def __init__(self):
        # ====================================================================
        # Directory Configuration
        # ====================================================================
        
        # Directory to store images of known/trusted faces
        self.KNOWN_FACES_DIR = "known_faces"
        
        # File to store serialized face encodings (pickled data)
        self.ENCODINGS_FILE = "face_encodings.pkl"
        
        # Directory for conversation and incident logs
        self.LOGS_DIR = "logs"
        
        # Directory for storing recorded audio files
        self.AUDIO_DIR = "audio_recordings"
        
        # ====================================================================
        # API Configuration
        # ====================================================================
        
        # Google Gemini API key for LLM functionality
        # Can be set via environment variable or hardcoded here
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDl4meToV6a8z6oooixUAH6aiDE2XeCADo")
        
        # ====================================================================
        # Audio Recording Settings
        # ====================================================================
        
        # Audio format: 16-bit PCM (standard for speech)
        self.AUDIO_FORMAT = pyaudio.paInt16
        
        # Number of audio channels: 1 (mono) - sufficient for speech
        self.CHANNELS = 1
        
        # Sample rate: 16kHz - optimal for speech recognition
        self.RATE = 16000
        
        # Audio buffer size: number of frames per buffer
        self.CHUNK = 1024
        
        # Default recording duration in seconds
        self.RECORD_DURATION = 5
        
        # ====================================================================
        # Whisper ASR Configuration
        # ====================================================================
        
        # Whisper model size: tiny (fastest), base, small, medium, large
        # 'tiny' is used for real-time performance on CPU
        self.WHISPER_MODEL_SIZE = "tiny"
        
        # Device to run Whisper on: 'cpu' or 'cuda' (GPU)
        self.WHISPER_DEVICE = "cpu"
        
        # Compute type for quantization: int8 for speed, float16 for accuracy
        self.WHISPER_COMPUTE_TYPE = "int8"
        
        # ====================================================================
        # Face Recognition Settings
        # ====================================================================
        
        # Recognition tolerance: lower = stricter matching (0.0-1.0)
        # 0.6 is a good balance between accuracy and false positives
        self.FACE_RECOGNITION_TOLERANCE = 0.6
        
        # Number of consecutive frames with unknown face before triggering alert
        self.UNKNOWN_FACE_THRESHOLD = 10
        
        # Process every Nth frame for efficiency (skip frames)
        # Processing every frame is computationally expensive
        self.PROCESS_EVERY_N_FRAMES = 2
        
        # ====================================================================
        # Escalation & Conversation Settings
        # ====================================================================
        
        # Cooldown period (seconds) between conversations with unknown persons
        # Prevents continuous harassment if person refuses to leave
        self.CONVERSATION_COOLDOWN = 60
        
        # Maximum escalation level (1=polite, 2=firm, 3=warning)
        self.MAX_ESCALATION_LEVEL = 3
        
        # Maximum conversation turns before giving up
        self.MAX_CONVERSATION_TURNS = 5
        
        # Number of uncooperative responses before escalating
        self.UNCOOPERATIVE_THRESHOLD = 3
        
        # ====================================================================
        # Directory Creation
        # ====================================================================
        
        # Create all required directories if they don't exist
        Path(self.KNOWN_FACES_DIR).mkdir(exist_ok=True)
        Path(self.LOGS_DIR).mkdir(exist_ok=True)
        Path(self.AUDIO_DIR).mkdir(exist_ok=True)


# ============================================================================
# AUDIO/ASR FUNCTIONS
# ============================================================================

def initialize_whisper(config):
    """
    Initialize Whisper ASR (Automatic Speech Recognition) model
    
    Whisper is an open-source speech recognition model by OpenAI.
    It converts spoken words into text with high accuracy.
    
    Args:
        config: GuardConfig object containing model settings
    
    Returns:
        WhisperModel instance ready for transcription
    """
    print(f"Loading Whisper model ({config.WHISPER_MODEL_SIZE})...")
    
    # Load the Whisper model with specified settings
    # - model_size: determines accuracy vs speed tradeoff
    # - device: cpu or cuda (GPU)
    # - compute_type: quantization for performance
    model = WhisperModel(
        config.WHISPER_MODEL_SIZE, 
        device=config.WHISPER_DEVICE, 
        compute_type=config.WHISPER_COMPUTE_TYPE
    )
    
    print("✓ Whisper model loaded")
    return model


def record_audio(config, duration=None, filename=None):
    """
    Record audio from the default microphone
    
    This function captures audio input from the system's default microphone
    and saves it as a WAV file for later transcription.
    
    Args:
        config: GuardConfig object with audio settings
        duration: Recording duration in seconds (uses config default if None)
        filename: Output filename (auto-generated with timestamp if None)
    
    Returns:
        Path to the recorded audio file (string)
    """
    # Use default duration from config if not specified
    if duration is None:
        duration = config.RECORD_DURATION
    
    # Generate filename with timestamp if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(config.AUDIO_DIR, f"recording_{timestamp}.wav")
    
    print(f"🎤 Recording for {duration} seconds...")
    
    # Initialize PyAudio interface
    p = pyaudio.PyAudio()
    
    # Open audio stream with configured parameters
    stream = p.open(
        format=config.AUDIO_FORMAT,      # 16-bit PCM
        channels=config.CHANNELS,        # Mono
        rate=config.RATE,                # 16kHz sample rate
        input=True,                      # Input stream (recording)
        frames_per_buffer=config.CHUNK   # Buffer size
    )
    
    # List to store audio frames
    frames = []
    
    # Calculate number of chunks to record based on duration
    # rate/chunk = chunks per second, multiply by duration
    num_chunks = int(config.RATE / config.CHUNK * duration)
    
    # Record audio in chunks
    for _ in range(num_chunks):
        data = stream.read(config.CHUNK)
        frames.append(data)
    
    print("✓ Recording finished")
    
    # Clean up audio stream
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Save recorded audio to WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(config.CHANNELS)
    wf.setsampwidth(p.get_sample_size(config.AUDIO_FORMAT))
    wf.setframerate(config.RATE)
    wf.writeframes(b''.join(frames))  # Concatenate all frames
    wf.close()
    
    return filename


def transcribe_audio(whisper_model, audio_file, language="en"):
    """
    Transcribe audio file to text using Whisper model
    
    This function takes an audio file and converts the speech in it
    to text using the pre-loaded Whisper model.
    
    Args:
        whisper_model: Loaded WhisperModel instance
        audio_file: Path to audio file (WAV format)
        language: Language code (default "en" for English)
    
    Returns:
        Transcribed text as string (empty string if no speech detected)
    """
    print("🔄 Transcribing audio...")
    
    # Transcribe audio with Whisper
    # - beam_size: number of beams for beam search (higher = more accurate but slower)
    # - language: expected language for better accuracy
    segments, info = whisper_model.transcribe(
        audio_file, 
        beam_size=5, 
        language=language
    )
    
    # Concatenate all segments into single transcript
    transcript = ""
    for segment in segments:
        transcript += segment.text + " "
    
    # Remove trailing whitespace
    transcript = transcript.strip()
    
    # Print result
    if transcript:
        print(f"👤 User said: {transcript}")
    else:
        print("(!) No speech detected")
    
    return transcript


def listen_for_command(whisper_model, config, activation_keyword="guard my room"):
    """
    Listen for a specific activation command
    
    This function records a short audio clip and checks if it contains
    the activation keyword to trigger guard mode.
    
    Args:
        whisper_model: Loaded WhisperModel instance
        config: GuardConfig object
        activation_keyword: Phrase to activate guard mode
    
    Returns:
        True if activation keyword detected, False otherwise
    """
    print(f"\n🎧 Listening for activation command: '{activation_keyword}'")
    
    # Record 3 seconds of audio
    audio_file = record_audio(config, duration=3)
    
    # Transcribe the recording
    transcript = transcribe_audio(whisper_model, audio_file)
    
    # Clean up temporary audio file
    if os.path.exists(audio_file):
        os.remove(audio_file)
    
    # Check for activation keyword (case-insensitive, fuzzy match)
    # Using 'in' allows partial matches (more flexible)
    if activation_keyword.lower() in transcript.lower():
        print("✓ Activation command detected!")
        return True
    
    return False


# ============================================================================
# TTS (TEXT-TO-SPEECH) FUNCTIONS
# ============================================================================

def speak(text, lang='en', slow=False):
    """
    Convert text to speech and play it
    
    This function uses Google Text-to-Speech (gTTS) to convert text
    into spoken audio and plays it through the system speakers.
    
    Args:
        text: Text to speak (string)
        lang: Language code (default 'en' for English)
        slow: Speak slowly if True (default False)
    """
    print(f"🔊 AI Guard: {text}")
    
    try:
        # Generate speech using gTTS
        tts = gTTS(text=text, lang=lang, slow=slow)
        
        # Save to temporary file
        # Using NamedTemporaryFile ensures proper cleanup
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            temp_filename = fp.name
            tts.save(temp_filename)
        
        # Play audio using pydub
        try:
            audio = AudioSegment.from_mp3(temp_filename)
            play(audio)
        except Exception as e:
            # If playback fails, just print warning (don't crash)
            print(f"(!) Audio playback warning: {e}")
        
        # Clean up temporary audio file
        if os.path.exists(audio_file):
            os.remove(audio_file)
        
        # Log this exchange
        conversation_log['exchanges'].append({
            'turn': turn + 1,
            'escalation_level': escalation_level,
            'ai_response': ai_response,
            'user_input': user_input
        })
        
        # Update conversation context for LLM
        conversation_context += f"Turn {turn+1} - AI: {ai_response} | User: {user_input}\n"
        
        # Check for valid response
        if not user_input or len(user_input) < 3:
            # No response or very short response - likely uncooperative
            print("(!) No valid response received")
            uncooperative_count += 1
        else:
            # Analyze response with LLM
            analysis = analyze_response_with_llm(llm_client, user_input, escalation_level)
            
            print(f"Analysis: {analysis['summary']}")
            
            # Check if person agreed to leave
            if analysis['leaving']:
                # Person is leaving peacefully
                farewell = "Thank you for cooperating. Have a good day."
                speak(farewell)
                conversation_log['exchanges'].append({
                    'turn': turn + 2,
                    'escalation_level': escalation_level,
                    'ai_response': farewell,
                    'user_input': ''
                })
                conversation_log['person_left_peacefully'] = True
                break
            
            # Check if they have a valid reason (at level 1 only)
            elif escalation_level == 1 and analysis['has_valid_reason']:
                # They have a legitimate reason at level 1
                if analysis['cooperative']:
                    response = "I understand. The person you're looking for is not here right now. You can come back later or I can take a message."
                    speak(response)
                    conversation_log['exchanges'].append({
                        'turn': turn + 2,
                        'escalation_level': escalation_level,
                        'ai_response': response,
                        'user_input': ''
                    })
                    
                    # Ask if they're leaving
                    audio_file = record_audio(config)
                    user_response = transcribe_audio(whisper_model, audio_file)
                    if os.path.exists(audio_file):
                        os.remove(audio_file)
                    
                    # Check if they accept and are leaving
                    leaving_check = analyze_response_with_llm(llm_client, user_response, escalation_level)
                    if leaving_check['leaving']:
                        farewell = "Goodbye. Feel free to return later."
                        speak(farewell)
                        conversation_log['person_left_peacefully'] = True
                        break
                    else:
                        # They're not leaving despite valid reason
                        uncooperative_count += 1
                else:
                    # Valid reason but not cooperative
                    uncooperative_count += 1
            
            # Check if response was uncooperative
            elif not analysis['cooperative']:
                uncooperative_count += 1
                print(f"Uncooperative count: {uncooperative_count}/{config.UNCOOPERATIVE_THRESHOLD}")
        
        # Check if should escalate based on uncooperative responses
        if uncooperative_count >= config.UNCOOPERATIVE_THRESHOLD:
            escalation_level += 1
            uncooperative_count = 0  # Reset counter
            print(f"!!! ESCALATING TO LEVEL {escalation_level} !!!")
        
        # Move to next turn
        turn += 1
        time.sleep(0.5)  # Brief pause between turns
    
    # Record final escalation level
    conversation_log['final_escalation_level'] = escalation_level
    
    # Take final action based on escalation level
    if escalation_level >= config.MAX_ESCALATION_LEVEL:
        # Maximum escalation reached - alert security
        final_warning = "Security alert! This incident has been logged and authorities notified."
        speak(final_warning)
        conversation_log['security_alerted'] = True
    
    return conversation_log


def save_conversation_log(config, conversation_log):
    """
    Save conversation log to file (with UTF-8 encoding to avoid Unicode errors)
    
    This function creates a human-readable log file of the entire
    conversation with an unknown person, including timestamps,
    all exchanges, and final outcomes.
    
    Args:
        config: GuardConfig object
        conversation_log: Dictionary containing conversation data
    """
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(config.LOGS_DIR, f"conversation_{timestamp}.txt")
    
    # Use UTF-8 encoding to handle all Unicode characters properly
    # This prevents errors with special characters in speech transcripts
    with open(log_file, 'w', encoding='utf-8') as f:
        # Write header
        f.write("="*60 + "\n")
        f.write(f"AI GUARD CONVERSATION LOG\n")
        f.write(f"Timestamp: {conversation_log['timestamp']}\n")
        f.write(f"Final Escalation Level: {conversation_log['final_escalation_level']}\n")
        f.write("="*60 + "\n\n")
        
        # Write each conversation exchange
        for exchange in conversation_log['exchanges']:
            f.write(f"--- Turn {exchange['turn']} (Level {exchange['escalation_level']}) ---\n")
            f.write(f"AI Guard: {exchange['ai_response']}\n")
            f.write(f"Person: {exchange['user_input']}\n\n")
        
        # Write final outcome
        if conversation_log.get('security_alerted'):
            f.write("\n[!] SECURITY ALERT TRIGGERED\n")
        
        if conversation_log.get('person_left_peacefully'):
            f.write("\n[OK] Person left peacefully\n")
    
    print(f"✓ Conversation logged: {log_file}")


# ============================================================================
# MAIN MONITORING LOOP (Core Function)
# ============================================================================

def monitor_room(whisper_model, llm_client, config, known_encodings, known_names, video_source=0):
    """
    Main monitoring loop - recognizes faces and handles unknowns
    
    This is the core function that continuously monitors the room via webcam.
    It performs the following tasks:
    1. Captures video frames from webcam
    2. Detects and recognizes faces in each frame
    3. Tracks known vs unknown faces
    4. Welcomes known faces
    5. Initiates conversation with unknown faces
    6. Handles escalation if needed
    
    The function implements a state machine that tracks:
    - Frames with unknown faces (triggers conversation after threshold)
    - Frames with known faces (pauses monitoring to return to menu)
    - Cooldown periods between conversations
    
    Args:
        whisper_model: Loaded WhisperModel instance
        llm_client: Gemini model instance
        config: GuardConfig object
        known_encodings: List of known face encodings
        known_names: List of corresponding names
        video_source: Video source (0 for default webcam, or path to video file)
    
    Returns:
        bool: True to continue (return to enrollment menu)
              False to exit program completely
    """
    # Sanity check: ensure we have enrolled faces
    if not known_encodings:
        print("(!) No known faces enrolled! System may not work properly.")
        return False
    
    # Print monitoring status
    print("\n" + "="*60)
    print("AI GUARD ACTIVE - MONITORING ROOM")
    print("="*60)
    print(f"Known faces: {', '.join(known_names)}")
    print("Press 'q' to quit, 's' to save screenshot")
    print("="*60 + "\n")
    
    # Open video capture device
    video_capture = cv2.VideoCapture(video_source)
    
    # Initialize state variables
    frame_count = 0                    # Total frames processed
    unknown_detected_frames = 0        # Consecutive frames with unknown face
    known_detected_frames = 0          # Consecutive frames with known face
    last_conversation_time = 0         # Timestamp of last conversation
    known_face_welcomed = False        # Flag to prevent repeated welcomes
    
    # Main monitoring loop
    while True:
        # Capture frame from webcam
        ret, frame = video_capture.read()
        if not ret:
            print("(!) Failed to capture frame")
            break
        
        frame_count += 1
        
        # Process every Nth frame for efficiency
        # Face recognition is computationally expensive, so we skip frames
        if frame_count % config.PROCESS_EVERY_N_FRAMES == 0:
            # Recognize faces in current frame
            face_locations, face_names, has_unknown, has_known = recognize_faces_in_frame(
                frame, known_encodings, known_names, config.FACE_RECOGNITION_TOLERANCE
            )
            
            # Update face tracking counters
            if has_unknown:
                # Unknown face detected
                unknown_detected_frames += 1
                known_detected_frames = 0
            elif has_known:
                # Known face detected
                known_detected_frames += 1
                unknown_detected_frames = 0
            else:
                # No faces detected
                unknown_detected_frames = 0
                known_detected_frames = 0
            
            # Draw bounding boxes and labels on frame
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale coordinates back up (we downscaled for processing)
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                # Choose color based on recognition
                # Green for known, red for unknown
                color = (0, 255, 0) if "Unknown" not in name else (0, 0, 255)
                
                # Draw bounding box around face
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                
                # Draw label background
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                
                # Draw name label
                cv2.putText(frame, name, (left + 6, bottom - 6), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        # Display status text on frame
        if unknown_detected_frames > 0:
            status = "ALERT: Unknown Person"
            status_color = (0, 0, 255)  # Red
        elif known_detected_frames > 0:
            status = "Welcome - Known Person"
            status_color = (0, 255, 0)  # Green
        else:
            status = "All Clear"
            status_color = (0, 255, 0)  # Green
        
        # Draw status text
        cv2.putText(frame, status, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # Display frame in window
        cv2.imshow('AI Guard - Room Monitor', frame)
        
        # Handle known face detection
        current_time = time.time()
        if (known_detected_frames >= 30 and not known_face_welcomed):
            # Known face has been present for ~1 second (30 frames)
            known_face_welcomed = True
            print("\n✓ Known person detected - Welcome!")
            speak(f"Welcome back! Room monitoring paused.")
            
            # Release video capture temporarily
            video_capture.release()
            cv2.destroyAllWindows()
            
            print("\nReturning to main menu...")
            time.sleep(2)
            return True  # Return to enrollment/activation menu
        
        # Check if should initiate conversation with unknown person
        if (unknown_detected_frames >= config.UNKNOWN_FACE_THRESHOLD and
            current_time - last_conversation_time > config.CONVERSATION_COOLDOWN):
            
            # Unknown face detected for threshold number of frames
            # and cooldown period has elapsed
            print(f"\n(!) Unknown person detected for {unknown_detected_frames} frames")
            last_conversation_time = current_time
            unknown_detected_frames = 0  # Reset counter
            
            # Release video temporarily during conversation
            video_capture.release()
            cv2.destroyAllWindows()
            
            # Handle conversation with unknown person
            conversation_log = handle_unknown_person(whisper_model, llm_client, config)
            
            # Save conversation log to file
            save_conversation_log(config, conversation_log)
            
            # Resume monitoring after conversation
            print("\n📹 Resuming monitoring...")
            video_capture = cv2.VideoCapture(video_source)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            # Quit command
            print("\n👋 Shutting down AI Guard...")
            video_capture.release()
            cv2.destroyAllWindows()
            return False  # Exit completely
        
        elif key == ord('s'):
            # Save screenshot command
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Screenshot saved: {filename}")
    
    # Cleanup (if loop exits normally)
    video_capture.release()
    cv2.destroyAllWindows()
    return False up temporary file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            
    except Exception as e:
        print(f"TTS Error: {e}")


# ============================================================================
# FACE RECOGNITION FUNCTIONS
# ============================================================================

def load_face_encodings(config):
    """
    Load known face encodings from pickle file
    
    Face encodings are 128-dimensional vectors that uniquely represent
    a person's face. This function loads previously saved encodings.
    
    Args:
        config: GuardConfig object
    
    Returns:
        tuple: (list of encodings, list of names)
               Empty lists if no encodings file exists
    """
    encodings_path = config.ENCODINGS_FILE
    
    # Check if encodings file exists
    if os.path.exists(encodings_path):
        # Load pickled data
        with open(encodings_path, 'rb') as f:
            data = pickle.load(f)
            encodings = data['encodings']
            names = data['names']
        
        print(f"✓ Loaded {len(names)} known face(s): {', '.join(names)}")
        return encodings, names
    
    # No encodings file found
    print("(!) No face encodings found")
    return [], []


def save_face_encodings(config, encodings, names):
    """
    Save face encodings to pickle file
    
    Serializes face encodings and associated names to disk for
    persistence across program runs.
    
    Args:
        config: GuardConfig object
        encodings: List of face encodings (numpy arrays)
        names: List of corresponding names (strings)
    """
    # Package data into dictionary
    data = {
        'encodings': encodings,
        'names': names
    }
    
    # Save as pickle file
    with open(config.ENCODINGS_FILE, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"✓ Saved {len(names)} face encoding(s)")


def enroll_face_from_webcam(config, name, num_samples=5):
    """
    Enroll a new face by capturing images from webcam
    
    This function captures multiple images of a person's face from different
    angles/expressions and generates an average face encoding for robust
    recognition.
    
    Args:
        config: GuardConfig object
        name: Name of the person to enroll
        num_samples: Number of sample images to capture (default 5)
    
    Returns:
        Face encoding (numpy array) or None if enrollment failed
    """
    print(f"\n📸 Enrolling face for: {name}")
    print(f"   Will capture {num_samples} samples")
    print("   Press SPACE to capture, ESC to cancel")
    
    # Open default webcam (device 0)
    video_capture = cv2.VideoCapture(0)
    
    # Track enrollment progress
    samples_captured = 0
    temp_encodings = []  # Store encodings from each sample
    
    # Continue until we have enough samples
    while samples_captured < num_samples:
        # Read frame from webcam
        ret, frame = video_capture.read()
        if not ret:
            print("(!) Failed to capture frame")
            break
        
        # Create display frame with instructions
        display_frame = frame.copy()
        
        # Show progress
        cv2.putText(display_frame, f"Samples: {samples_captured}/{num_samples}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show instruction
        cv2.putText(display_frame, "Press SPACE to capture", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display frame
        cv2.imshow('Enroll Face', display_frame)
        
        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC key
            print("Enrollment cancelled")
            video_capture.release()
            cv2.destroyAllWindows()
            return None
        
        elif key == 32:  # SPACE key
            # Convert BGR (OpenCV format) to RGB (face_recognition format)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect and encode faces in the frame
            face_encodings = face_recognition.face_encodings(rgb_frame)
            
            # Check if a face was detected
            if len(face_encodings) > 0:
                # Store the first face encoding found
                temp_encodings.append(face_encodings[0])
                samples_captured += 1
                print(f"   ✓ Captured sample {samples_captured}/{num_samples}")
                
                # Save image to disk for reference
                img_path = Path(config.KNOWN_FACES_DIR) / f"{name}_{samples_captured}.jpg"
                cv2.imwrite(str(img_path), frame)
            else:
                print("   (!) No face detected, try again")
    
    # Clean up
    video_capture.release()
    cv2.destroyAllWindows()
    
    # If we successfully captured samples, average the encodings
    if temp_encodings:
        # Average all encodings to create a robust representation
        # This reduces the effect of lighting/angle variations
        avg_encoding = np.mean(temp_encodings, axis=0)
        print(f"✓ Successfully enrolled {name}")
        return avg_encoding
    
    return None


def recognize_faces_in_frame(frame, known_encodings, known_names, tolerance=0.6):
    """
    Recognize faces in a single video frame
    
    This function detects all faces in a frame, extracts their encodings,
    and compares them against known encodings to identify people.
    
    Args:
        frame: OpenCV frame (BGR format)
        known_encodings: List of known face encodings
        known_names: List of corresponding names
        tolerance: Recognition tolerance (lower = stricter, 0.0-1.0)
    
    Returns:
        tuple: (face_locations, face_names, has_unknown, has_known)
            - face_locations: list of (top, right, bottom, left) coordinates
            - face_names: list of names/labels for each face
            - has_unknown: boolean, True if any unknown face detected
            - has_known: boolean, True if any known face detected
    """
    # Resize frame to 1/4 size for faster processing
    # Face recognition is computationally expensive, so we trade
    # resolution for speed in real-time applications
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
    # Convert BGR (OpenCV) to RGB (face_recognition library)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces in the frame
    # Returns list of (top, right, bottom, left) tuples
    face_locations = face_recognition.face_locations(rgb_small_frame)
    
    # Extract face encodings for all detected faces
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    # Initialize result variables
    face_names = []
    has_unknown = False
    has_known = False
    
    # Process each detected face
    for face_encoding in face_encodings:
        # Compare this face encoding against all known encodings
        # Returns list of True/False for each known face
        matches = face_recognition.compare_faces(
            known_encodings, 
            face_encoding, 
            tolerance=tolerance
        )
        
        # Default to unknown
        name = "Unknown"
        
        # If we have known encodings to compare against
        if len(known_encodings) > 0:
            # Calculate Euclidean distance to each known face
            # Lower distance = more similar
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            
            # Find the closest match
            best_match_index = np.argmin(face_distances)
            
            # If the closest match is within tolerance
            if matches[best_match_index]:
                name = known_names[best_match_index]
                
                # Calculate confidence score (1 - distance)
                confidence = 1 - face_distances[best_match_index]
                name = f"{name} ({confidence:.2f})"
                has_known = True
        
        # Check if face is unknown
        if "Unknown" in name:
            has_unknown = True
        
        face_names.append(name)
    
    return face_locations, face_names, has_unknown, has_known


# ============================================================================
# LLM INTEGRATION
# ============================================================================

def initialize_llm(config):
    """
    Initialize LLM (Large Language Model) - Google Gemini
    
    The LLM provides intelligent conversation capabilities, allowing
    the system to have natural dialogues with unknown persons and
    make contextual decisions about threats.
    
    Args:
        config: GuardConfig object
    
    Returns:
        Gemini model instance
    """
    # Check if API key is configured
    if not config.GEMINI_API_KEY:
        # Prompt user for API key if not set
        api_key = input("\nEnter your Gemini API key: ").strip()
        config.GEMINI_API_KEY = api_key
    
    # Check if library is installed
    if genai is None:
        raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
    
    # Configure Gemini with API key
    genai.configure(api_key=config.GEMINI_API_KEY)
    
    # Create model instance (using Gemini 2.0 Flash Experimental)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    print("✓ Conversational AI initialized (Gemini)")
    return model


def analyze_response_with_llm(llm_client, user_input, escalation_level):
    """
    Use LLM to analyze if user's response is cooperative
    
    This function sends the user's response to the LLM for analysis,
    determining if they're being cooperative, if they're leaving,
    and if they have a valid reason for being there.
    
    Args:
        llm_client: Gemini model instance
        user_input: User's transcribed response (string)
        escalation_level: Current escalation level (1-3)
    
    Returns:
        dict with keys:
            - 'cooperative': bool (is the person cooperative?)
            - 'leaving': bool (is the person agreeing to leave?)
            - 'has_valid_reason': bool (legitimate reason for being there?)
            - 'summary': string (brief summary of their intent)
    """
    # Construct prompt for LLM
    # We want structured JSON output for easy parsing
    prompt = f"""You are an AI security guard analyzing a visitor's response. 

Current escalation level: {escalation_level}
Visitor said: "{user_input}"

Analyze this response and return ONLY a JSON object (no markdown, no extra text) with these fields:
{{
    "cooperative": true/false (are they being cooperative and respectful?),
    "leaving": true/false (are they agreeing to leave?),
    "has_valid_reason": true/false (did they provide a legitimate reason for being there?),
    "summary": "brief summary of their intent"
}}

Examples:
- "I'm John's friend, here to drop off his keys" -> cooperative: true, has_valid_reason: true
- "None of your business" -> cooperative: false
- "Fine, I'm leaving" -> cooperative: true, leaving: true
- "Why should I tell you?" -> cooperative: false
"""
    
    try:
        # Generate response from LLM
        response = llm_client.generate_content(prompt)
        response_text = response.text.strip()
        
        # Remove markdown code blocks if present
        # Some models wrap JSON in ```json ... ```
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        
        # Parse JSON response
        import json
        analysis = json.loads(response_text)
        return analysis
    
    except Exception as e:
        print(f"(!) LLM analysis error: {e}")
        
        # Fallback: simple keyword-based analysis if LLM fails
        user_lower = user_input.lower()
        return {
            "cooperative": not any(word in user_lower for word in ["no", "why", "none", "leave me"]),
            "leaving": any(word in user_lower for word in ["leave", "leaving", "going", "bye"]),
            "has_valid_reason": any(word in user_lower for word in ["friend", "delivery", "visitor", "looking for"]),
            "summary": "Unable to analyze properly"
        }


def get_llm_response(llm_client, user_input, escalation_level, conversation_context=""):
    """
    Get contextual response from LLM
    
    This function generates an appropriate response based on:
    - What the user said
    - Current escalation level (1=polite, 2=firm, 3=warning)
    - Previous conversation context
    
    Args:
        llm_client: Gemini model instance
        user_input: User's transcribed speech
        escalation_level: Current escalation level (1-3)
        conversation_context: Previous conversation summary (string)
    
    Returns:
        AI response as string
    """
    # Define system prompts for each escalation level
    # These guide the LLM's tone and goals
    system_prompts = {
        1: """You are a polite but firm AI security guard. An unknown person has been detected. 
Your goal: Find out who they are and why they're here. Be professional and brief (1-2 sentences).
If they give a valid reason (visiting someone, delivery, etc.), inform them the person isn't here and suggest they come back later.""",
        
        2: """You are a stern AI security guard. The person has been uncooperative or suspicious.
Your goal: Firmly but politely ask them to leave. Be direct and serious (1-2 sentences).
Make it clear they are not authorized to be here.""",
        
        3: """You are an AI security guard issuing a final warning. The person has been repeatedly uncooperative.
Your goal: Issue a stern final warning that security/authorities are being notified. Be very firm and direct (1-2 sentences)."""
    }
    
    # Construct full prompt
    prompt = f"""{system_prompts[escalation_level]}

Conversation context: {conversation_context}

Person said: "{user_input}"

Respond appropriately as the AI guard (keep it brief, 1-2 sentences):"""
    
    try:
        # Generate response from LLM
        response = llm_client.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=100,  # Keep responses short
                temperature=0.7,         # Some creativity, but not too much
            )
        )
        return response.text.strip()
    
    except Exception as e:
        print(f"(!) LLM error: {e}")
        
        # Fallback responses if LLM fails
        if escalation_level == 1:
            return "I don't recognize you. Could you please tell me who you are and why you're here?"
        elif escalation_level == 2:
            return "I'm not authorized to let you in. Please leave the premises immediately."
        else:
            return "This is your final warning. Security has been alerted."


# ============================================================================
# CONVERSATION & ESCALATION LOGIC
# ============================================================================

def handle_unknown_person(whisper_model, llm_client, config):
    """
    Handle conversation with unknown person with intelligent escalation
    
    This is the core conversation function that:
    1. Initiates dialogue with unknown person
    2. Listens to their responses
    3. Analyzes cooperativeness using LLM
    4. Escalates if they're uncooperative
    5. Logs the entire conversation
    
    The conversation follows this escalation path:
    Level 1: Polite inquiry (who are you? why are you here?)
    Level 2: Firm request to leave (you need to leave now)
    Level 3: Final warning (security has been alerted)
    
    Args:
        whisper_model: Loaded WhisperModel instance
        llm_client: Gemini model instance
        config: GuardConfig object
    
    Returns:
        dict: Conversation log with metadata including:
            - timestamp
            - all conversation exchanges
            - final escalation level
            - whether security was alerted
            - whether person left peacefully
    """
    print("\n" + "="*60)
    print("UNKNOWN PERSON DETECTED - INITIATING CONVERSATION")
    print("="*60)
    
    # Initialize conversation log
    conversation_log = {
        'timestamp': datetime.now().isoformat(),
        'exchanges': [],
        'final_escalation_level': 1,
        'security_alerted': False
    }
    
    # Initialize conversation state
    escalation_level = 1           # Start at polite level
    turn = 0                       # Conversation turn counter
    uncooperative_count = 0        # Track uncooperative responses
    conversation_context = ""      # Running context for LLM
    
    # Continue conversation until max turns or max escalation
    while turn < config.MAX_CONVERSATION_TURNS and escalation_level <= config.MAX_ESCALATION_LEVEL:
        print(f"\n--- Turn {turn + 1} | Escalation Level {escalation_level} ---")
        
        # Generate AI response
        if turn == 0:
            # First turn: standard greeting/inquiry
            ai_response = "Hello, I don't recognize you. Could you please tell me who you are and why you're here?"
        else:
            # Subsequent turns: contextual response from LLM
            ai_response = get_llm_response(llm_client, user_input, escalation_level, conversation_context)
        
        # Speak the response via TTS
        speak(ai_response)
        
        # Record user's audio response
        audio_file = record_audio(config)
        user_input = transcribe_audio(whisper_model, audio_file)
        
        # Clean
