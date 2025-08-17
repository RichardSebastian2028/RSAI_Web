import speech_recognition as sr
import pyttsx3
from googletrans import Translator
import random

# CONFIG
QUIT_COMMANDS = ['exit', 'quit', 'stop', 'bye']
JOKES = [
    "Why don't scientists trust atoms? Because they make up everything!",
    "Why did the math book look sad? Because it had too many problems.",
    "Why don't programmers like nature? It has too many bugs.",
    "Why do we tell actors to 'break a leg?' Because every play has a cast.",
    "Why did the scarecrow win an award? Because he was outstanding in his field."
]
UNKNOWN_RESPONSES = [
    "I'm sorry, I didn't understand that. Could you please repeat?", 
    "Could you please clarify what you mean?", 
    "I'm not sure I follow. Can you explain a bit more?"
]

# Speak function that tries to match voice to language
def speak(text, lang_code='en'):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    voices = engine.getProperty('voices')

    # Try to find a matching voice
    voice_found = False
    for voice in voices:
        if lang_code.lower() in voice.id.lower():
            engine.setProperty('voice', voice.id)
            voice_found = True
            break

    if not voice_found:
        engine.setProperty('voice', voices[0].id)  # fallback to first voice
    
    engine.say(text)
    engine.runAndWait()

# Recognize speech
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Speak now (say 'quit' to exit)...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.listen(source)
    
    try:
        text = recognizer.recognize_google(audio, language='en-US')
        print(f"🗣 You said: {text}")
        return text.lower()
    except sr.UnknownValueError:
        print(f"❌ {random.choice(UNKNOWN_RESPONSES)}")
    except sr.RequestError as e:
        print(f"API ERROR: {e}")
    return ""

# Translate
def translate_text(text, dest_language='es'):
    translator = Translator()
    # Force translation by adding context for short words
    if len(text.split()) == 1:
        text = f"{text}!"
    translation = translator.translate(text, dest=dest_language)
    print(f"🌎 Translated Text: {translation.text}")
    return translation.text

# Language selection
def display_supported_languages():
    print("🌐 Supported Languages:")
    print("1. Hindi (hi)")
    print("2. Tamil (ta)")
    print("3. Telugu (te)")
    print("4. Kannada (kn)")
    print("5. Malayalam (ml)")
    print("6. Bengali (bn)")
    print("7. Marathi (mr)")
    print("8. Gujarati (gu)")
    print("9. Punjabi (pa)")
    print("10. Spanish (es)")
    choice = input("Select a language number: ")
    language_dict = {
        '1': 'hi', '2': 'ta', '3': 'te', '4': 'kn', '5': 'ml',
        '6': 'bn', '7': 'mr', '8': 'gu', '9': 'pa', '10': 'es'
    }
    return language_dict.get(choice, 'es')

# Main
def main():
    target_language = display_supported_languages()
    while True:
        original_text = speech_to_text()

        if original_text in QUIT_COMMANDS:
            print("👋 Goodbye!")
            break

        if "joke" in original_text:
            joke = random.choice(JOKES)
            print(f"😂 {joke}")
            speak(joke, lang_code='en')
            continue

        if original_text:
            translated_text = translate_text(original_text, dest_language=target_language)
            speak(translated_text, lang_code=target_language)
        else:
            print(f"❌ {random.choice(UNKNOWN_RESPONSES)}")

if __name__ == "__main__":
    main()

