"""
AI Guard Agent - Integration Functions
This module provides the core functions that integrate ASR, FR, TTS, and LLM components.
Each function is designed to be called from main.py for clean separation of concerns.
"""

import cv2
import numpy as np
import pickle
import os
from pathlib import Path
from datetime import datetime
import time
import pyaudio
import wave
from faster_whisper import WhisperModel
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import tempfile

# Import face recognition components
import face_recognition

# LLM imports
try:
    import google.generativeai as genai
except ImportError:
    genai = None


# ============================================================================
# CONFIGURATION & GLOBALS
# ============================================================================

class GuardConfig:
    """Configuration class for AI Guard system"""
    def __init__(self):
        # Directories
        self.KNOWN_FACES_DIR = "known_faces"
        self.ENCODINGS_FILE = "face_encodings.pkl"
        self.LOGS_DIR = "logs"
        self.AUDIO_DIR = "audio_recordings"
        
        # API Keys (set your API key here or via environment variable)
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDl4meToV6a8z6oooixUAH6aiDE2XeCADo")  # Add your key here
        
        # Audio settings
        self.AUDIO_FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = 1024
        self.RECORD_DURATION = 5  # seconds
        
        # Whisper settings
        self.WHISPER_MODEL_SIZE = "tiny"  # tiny, base, small, medium
        self.WHISPER_DEVICE = "cpu"
        self.WHISPER_COMPUTE_TYPE = "int8"
        
        # Face recognition settings
        self.FACE_RECOGNITION_TOLERANCE = 0.6
        self.UNKNOWN_FACE_THRESHOLD = 10  # frames before triggering
        self.PROCESS_EVERY_N_FRAMES = 2
        
        # Escalation settings
        self.CONVERSATION_COOLDOWN = 60  # seconds between conversations
        self.MAX_ESCALATION_LEVEL = 3
        self.MAX_CONVERSATION_TURNS = 5
        self.UNCOOPERATIVE_THRESHOLD = 3  # strikes before escalation
        
        # Create directories
        Path(self.KNOWN_FACES_DIR).mkdir(exist_ok=True)
        Path(self.LOGS_DIR).mkdir(exist_ok=True)
        Path(self.AUDIO_DIR).mkdir(exist_ok=True)


# ============================================================================
# AUDIO/ASR FUNCTIONS
# ============================================================================

def initialize_whisper(config):
    """
    Initialize Whisper ASR model
    
    Args:
        config: GuardConfig object
    
    Returns:
        WhisperModel instance
    """
    print(f"Loading Whisper model ({config.WHISPER_MODEL_SIZE})...")
    model = WhisperModel(
        config.WHISPER_MODEL_SIZE, 
        device=config.WHISPER_DEVICE, 
        compute_type=config.WHISPER_COMPUTE_TYPE
    )
    print("✓ Whisper model loaded")
    return model


def record_audio(config, duration=None, filename=None):
    """
    Record audio from microphone
    
    Args:
        config: GuardConfig object
        duration: Recording duration in seconds (default from config)
        filename: Output filename (auto-generated if None)
    
    Returns:
        Path to recorded audio file
    """
    if duration is None:
        duration = config.RECORD_DURATION
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(config.AUDIO_DIR, f"recording_{timestamp}.wav")
    
    print(f"🎤 Recording for {duration} seconds...")
    
    p = pyaudio.PyAudio()
    stream = p.open(
        format=config.AUDIO_FORMAT,
        channels=config.CHANNELS,
        rate=config.RATE,
        input=True,
        frames_per_buffer=config.CHUNK
    )
    
    frames = []
    for _ in range(0, int(config.RATE / config.CHUNK * duration)):
        data = stream.read(config.CHUNK)
        frames.append(data)
    
    print("✓ Recording finished")
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Save to file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(config.CHANNELS)
    wf.setsampwidth(p.get_sample_size(config.AUDIO_FORMAT))
    wf.setframerate(config.RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    return filename


def transcribe_audio(whisper_model, audio_file, language="en"):
    """
    Transcribe audio file using Whisper
    
    Args:
        whisper_model: Loaded WhisperModel instance
        audio_file: Path to audio file
        language: Language code (default "en")
    
    Returns:
        Transcribed text as string
    """
    print("🔄 Transcribing audio...")
    
    segments, info = whisper_model.transcribe(
        audio_file, 
        beam_size=5, 
        language=language
    )
    
    transcript = ""
    for segment in segments:
        transcript += segment.text + " "
    
    transcript = transcript.strip()
    
    if transcript:
        print(f"👤 User said: {transcript}")
    else:
        print("(!) No speech detected")
    
    return transcript


def listen_for_command(whisper_model, config, activation_keyword="guard my room"):
    """
    Listen for activation command
    
    Args:
        whisper_model: Loaded WhisperModel instance
        config: GuardConfig object
        activation_keyword: Phrase to activate guard mode
    
    Returns:
        True if activation keyword detected, False otherwise
    """
    print(f"\n🎧 Listening for activation command: '{activation_keyword}'")
    
    audio_file = record_audio(config, duration=3)
    transcript = transcribe_audio(whisper_model, audio_file)
    
    # Clean up audio file
    if os.path.exists(audio_file):
        os.remove(audio_file)
    
    # Check for activation keyword (case-insensitive, fuzzy match)
    if activation_keyword.lower() in transcript.lower():
        print("✓ Activation command detected!")
        return True
    
    return False


# ============================================================================
# TTS FUNCTIONS
# ============================================================================

def speak(text, lang='en', slow=False):
    """
    Convert text to speech and play it
    
    Args:
        text: Text to speak
        lang: Language code (default 'en')
        slow: Speak slowly if True
    """
    print(f"🔊 AI Guard: {text}")
    
    try:
        tts = gTTS(text=text, lang=lang, slow=slow)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            temp_filename = fp.name
            tts.save(temp_filename)
        
        # Play audio
        try:
            audio = AudioSegment.from_mp3(temp_filename)
            play(audio)
        except Exception as e:
            print(f"(!) Audio playback warning: {e}")
        
        # Clean up
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            
    except Exception as e:
        print(f"TTS Error: {e}")


# ============================================================================
# FACE RECOGNITION FUNCTIONS
# ============================================================================

def load_face_encodings(config):
    """
    Load known face encodings from file
    
    Args:
        config: GuardConfig object
    
    Returns:
        tuple: (list of encodings, list of names)
    """
    encodings_path = config.ENCODINGS_FILE
    
    if os.path.exists(encodings_path):
        with open(encodings_path, 'rb') as f:
            data = pickle.load(f)
            encodings = data['encodings']
            names = data['names']
        print(f"✓ Loaded {len(names)} known face(s): {', '.join(names)}")
        return encodings, names
    
    print("(!) No face encodings found")
    return [], []


def save_face_encodings(config, encodings, names):
    """
    Save face encodings to file
    
    Args:
        config: GuardConfig object
        encodings: List of face encodings
        names: List of corresponding names
    """
    data = {
        'encodings': encodings,
        'names': names
    }
    with open(config.ENCODINGS_FILE, 'wb') as f:
        pickle.dump(data, f)
    print(f"✓ Saved {len(names)} face encoding(s)")


def enroll_face_from_webcam(config, name, num_samples=5):
    """
    Enroll a new face by capturing from webcam
    
    Args:
        config: GuardConfig object
        name: Name of the person to enroll
        num_samples: Number of sample images to capture
    
    Returns:
        Face encoding (numpy array) or None if failed
    """
    print(f"\n📸 Enrolling face for: {name}")
    print(f"   Will capture {num_samples} samples")
    print("   Press SPACE to capture, ESC to cancel")
    
    video_capture = cv2.VideoCapture(0)
    samples_captured = 0
    temp_encodings = []
    
    while samples_captured < num_samples:
        ret, frame = video_capture.read()
        if not ret:
            print("(!) Failed to capture frame")
            break
        
        # Display instructions
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Samples: {samples_captured}/{num_samples}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, "Press SPACE to capture", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('Enroll Face', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            print("Enrollment cancelled")
            video_capture.release()
            cv2.destroyAllWindows()
            return None
        
        elif key == 32:  # SPACE
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(rgb_frame)
            
            if len(face_encodings) > 0:
                temp_encodings.append(face_encodings[0])
                samples_captured += 1
                print(f"   ✓ Captured sample {samples_captured}/{num_samples}")
                
                # Save image
                img_path = Path(config.KNOWN_FACES_DIR) / f"{name}_{samples_captured}.jpg"
                cv2.imwrite(str(img_path), frame)
            else:
                print("   (!) No face detected, try again")
    
    video_capture.release()
    cv2.destroyAllWindows()
    
    if temp_encodings:
        # Average the encodings
        avg_encoding = np.mean(temp_encodings, axis=0)
        print(f"✓ Successfully enrolled {name}")
        return avg_encoding
    
    return None


def recognize_faces_in_frame(frame, known_encodings, known_names, tolerance=0.6):
    """
    Recognize faces in a single frame
    
    Args:
        frame: OpenCV frame (BGR)
        known_encodings: List of known face encodings
        known_names: List of corresponding names
        tolerance: Recognition tolerance (lower = stricter)
    
    Returns:
        tuple: (face_locations, face_names, has_unknown, has_known)
    """
    # Resize for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    face_names = []
    has_unknown = False
    has_known = False
    
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(
            known_encodings, 
            face_encoding, 
            tolerance=tolerance
        )
        name = "Unknown"
        
        if len(known_encodings) > 0:
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            if matches[best_match_index]:
                name = known_names[best_match_index]
                confidence = 1 - face_distances[best_match_index]
                name = f"{name} ({confidence:.2f})"
                has_known = True
        
        if "Unknown" in name:
            has_unknown = True
        
        face_names.append(name)
    
    return face_locations, face_names, has_unknown, has_known


# ============================================================================
# LLM INTEGRATION
# ============================================================================

def initialize_llm(config):
    """
    Initialize LLM (Google Gemini)
    
    Args:
        config: GuardConfig object
    
    Returns:
        Gemini model instance
    """
    if not config.GEMINI_API_KEY:
        api_key = input("\nEnter your Gemini API key: ").strip()
        config.GEMINI_API_KEY = api_key
    
    if genai is None:
        raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
    
    genai.configure(api_key=config.GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    print("✓ Conversational AI initialized (Gemini)")
    return model


def analyze_response_with_llm(llm_client, user_input, escalation_level):
    """
    Use LLM to analyze if user's response is cooperative
    
    Args:
        llm_client: Gemini model instance
        user_input: User's transcribed response
        escalation_level: Current escalation level
    
    Returns:
        dict with 'cooperative', 'leaving', 'reason'
    """
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
        response = llm_client.generate_content(prompt)
        response_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        
        # Parse JSON
        import json
        analysis = json.loads(response_text)
        return analysis
    
    except Exception as e:
        print(f"(!) LLM analysis error: {e}")
        # Fallback: simple keyword analysis
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
    
    Args:
        llm_client: Gemini model instance
        user_input: User's transcribed speech
        escalation_level: Current escalation level (1-3)
        conversation_context: Previous conversation summary
    
    Returns:
        AI response as string
    """
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
    
    prompt = f"""{system_prompts[escalation_level]}

Conversation context: {conversation_context}

Person said: "{user_input}"

Respond appropriately as the AI guard (keep it brief, 1-2 sentences):"""
    
    try:
        response = llm_client.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=100,
                temperature=0.7,
            )
        )
        return response.text.strip()
    
    except Exception as e:
        print(f"(!) LLM error: {e}")
        # Fallback responses
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
    
    Args:
        whisper_model: Loaded WhisperModel instance
        llm_client: Gemini model instance
        config: GuardConfig object
    
    Returns:
        dict: Conversation log with metadata
    """
    print("\n" + "="*60)
    print("UNKNOWN PERSON DETECTED - INITIATING CONVERSATION")
    print("="*60)
    
    conversation_log = {
        'timestamp': datetime.now().isoformat(),
        'exchanges': [],
        'final_escalation_level': 1,
        'security_alerted': False
    }
    
    escalation_level = 1
    turn = 0
    uncooperative_count = 0
    conversation_context = ""
    
    while turn < config.MAX_CONVERSATION_TURNS and escalation_level <= config.MAX_ESCALATION_LEVEL:
        print(f"\n--- Turn {turn + 1} | Escalation Level {escalation_level} ---")
        
        # Get AI response
        if turn == 0:
            ai_response = "Hello, I don't recognize you. Could you please tell me who you are and why you're here?"
        else:
            ai_response = get_llm_response(llm_client, user_input, escalation_level, conversation_context)
        
        # Speak the response
        speak(ai_response)
        
        # Record user's response
        audio_file = record_audio(config)
        user_input = transcribe_audio(whisper_model, audio_file)
        
        # Clean up audio
        if os.path.exists(audio_file):
            os.remove(audio_file)
        
        # Log exchange
        conversation_log['exchanges'].append({
            'turn': turn + 1,
            'escalation_level': escalation_level,
            'ai_response': ai_response,
            'user_input': user_input
        })
        
        # Update context
        conversation_context += f"Turn {turn+1} - AI: {ai_response} | User: {user_input}\n"
        
        # Check for valid response
        if not user_input or len(user_input) < 3:
            print("(!) No valid response received")
            uncooperative_count += 1
        else:
            # Analyze response with LLM
            analysis = analyze_response_with_llm(llm_client, user_input, escalation_level)
            
            print(f"Analysis: {analysis['summary']}")
            
            if analysis['leaving']:
                # Person agreed to leave
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
            
            elif escalation_level == 1 and analysis['has_valid_reason']:
                # They have a valid reason at level 1
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
                    
                    leaving_check = analyze_response_with_llm(llm_client, user_response, escalation_level)
                    if leaving_check['leaving']:
                        farewell = "Goodbye. Feel free to return later."
                        speak(farewell)
                        conversation_log['person_left_peacefully'] = True
                        break
                    else:
                        uncooperative_count += 1
                else:
                    uncooperative_count += 1
            
            elif not analysis['cooperative']:
                uncooperative_count += 1
                print(f"Uncooperative count: {uncooperative_count}/{config.UNCOOPERATIVE_THRESHOLD}")
        
        # Check if should escalate
        if uncooperative_count >= config.UNCOOPERATIVE_THRESHOLD:
            escalation_level += 1
            uncooperative_count = 0
            print(f"!!! ESCALATING TO LEVEL {escalation_level} !!!")
        
        turn += 1
        time.sleep(0.5)
    
    conversation_log['final_escalation_level'] = escalation_level
    
    # Final action based on escalation
    if escalation_level >= config.MAX_ESCALATION_LEVEL:
        final_warning = "Security alert! This incident has been logged and authorities notified."
        speak(final_warning)
        conversation_log['security_alerted'] = True
    
    return conversation_log


def save_conversation_log(config, conversation_log):
    """
    Save conversation log to file (with UTF-8 encoding to avoid Unicode errors)
    
    Args:
        config: GuardConfig object
        conversation_log: Dictionary containing conversation data
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(config.LOGS_DIR, f"conversation_{timestamp}.txt")
    
    # Use UTF-8 encoding to handle all Unicode characters
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write(f"AI GUARD CONVERSATION LOG\n")
        f.write(f"Timestamp: {conversation_log['timestamp']}\n")
        f.write(f"Final Escalation Level: {conversation_log['final_escalation_level']}\n")
        f.write("="*60 + "\n\n")
        
        for exchange in conversation_log['exchanges']:
            f.write(f"--- Turn {exchange['turn']} (Level {exchange['escalation_level']}) ---\n")
            f.write(f"AI Guard: {exchange['ai_response']}\n")
            f.write(f"Person: {exchange['user_input']}\n\n")
        
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
    
    Args:
        whisper_model: Loaded WhisperModel instance
        llm_client: Gemini model instance
        config: GuardConfig object
        known_encodings: List of known face encodings
        known_names: List of corresponding names
        video_source: Video source (0 for default webcam)
    
    Returns:
        bool: True to continue (return to enrollment), False to exit
    """
    if not known_encodings:
        print("(!) No known faces enrolled! System may not work properly.")
        return False
    
    print("\n" + "="*60)
    print("AI GUARD ACTIVE - MONITORING ROOM")
    print("="*60)
    print(f"Known faces: {', '.join(known_names)}")
    print("Press 'q' to quit, 's' to save screenshot")
    print("="*60 + "\n")
    
    video_capture = cv2.VideoCapture(video_source)
    
    frame_count = 0
    unknown_detected_frames = 0
    known_detected_frames = 0
    last_conversation_time = 0
    known_face_welcomed = False
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("(!) Failed to capture frame")
            break
        
        frame_count += 1
        
        # Process every Nth frame for efficiency
        if frame_count % config.PROCESS_EVERY_N_FRAMES == 0:
            face_locations, face_names, has_unknown, has_known = recognize_faces_in_frame(
                frame, known_encodings, known_names, config.FACE_RECOGNITION_TOLERANCE
            )
            
            # Track faces
            if has_unknown:
                unknown_detected_frames += 1
                known_detected_frames = 0
            elif has_known:
                known_detected_frames += 1
                unknown_detected_frames = 0
            else:
                unknown_detected_frames = 0
                known_detected_frames = 0
            
            # Draw results on frame
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                color = (0, 255, 0) if "Unknown" not in name else (0, 0, 255)
                
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        # Display status
        if unknown_detected_frames > 0:
            status = "ALERT: Unknown Person"
            status_color = (0, 0, 255)
        elif known_detected_frames > 0:
            status = "Welcome - Known Person"
            status_color = (0, 255, 0)
        else:
            status = "All Clear"
            status_color = (0, 255, 0)
        
        cv2.putText(frame, status, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        cv2.imshow('AI Guard - Room Monitor', frame)
        
        # Handle known face detection
        current_time = time.time()
        if (known_detected_frames >= 30 and not known_face_welcomed):
            known_face_welcomed = True
            print("\n✓ Known person detected - Welcome!")
            speak(f"Welcome back! Room monitoring paused.")
            
            # Release video temporarily
            video_capture.release()
            cv2.destroyAllWindows()
            
            print("\nReturning to main menu...")
            time.sleep(2)
            return True  # Return to enrollment/activation
        
        # Check if should initiate conversation with unknown person
        if (unknown_detected_frames >= config.UNKNOWN_FACE_THRESHOLD and
            current_time - last_conversation_time > config.CONVERSATION_COOLDOWN):
            
            print(f"\n(!) Unknown person detected for {unknown_detected_frames} frames")
            last_conversation_time = current_time
            unknown_detected_frames = 0
            
            # Release video temporarily
            video_capture.release()
            cv2.destroyAllWindows()
            
            # Handle conversation
            conversation_log = handle_unknown_person(whisper_model, llm_client, config)
            save_conversation_log(config, conversation_log)
            
            # Resume monitoring
            print("\n📹 Resuming monitoring...")
            video_capture = cv2.VideoCapture(video_source)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n👋 Shutting down AI Guard...")
            video_capture.release()
            cv2.destroyAllWindows()
            return False  # Exit completely
        elif key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Screenshot saved: {filename}")
    
    video_capture.release()
    cv2.destroyAllWindows()
    return False
