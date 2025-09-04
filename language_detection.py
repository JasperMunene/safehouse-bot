import google.generativeai as genai
from config import Config


class LanguageDetector:
    def __init__(self):
        genai.configure(api_key=Config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def detect_language(self, text):
        """Detect the language of the given text"""
        if not text.strip():
            return Config.DEFAULT_LANGUAGE

        try:
            prompt = Config.LANGUAGE_DETECTION_PROMPT.format(text=text)
            response = self.model.generate_content(prompt)
            detected_lang = response.text.strip().lower()

            # Validate the detected language
            if detected_lang in Config.SUPPORTED_LANGUAGES:
                return detected_lang
            else:
                return Config.DEFAULT_LANGUAGE

        except Exception as e:
            print(f"Language detection error: {e}")
            return Config.DEFAULT_LANGUAGE

    def is_language_settled(self, conversation_history):
        """Check if the language has been consistently used in recent messages"""
        if len(conversation_history) < 3:
            return False

        # Get the last few user messages
        user_messages = [msg for i, msg in enumerate(conversation_history) if i % 2 == 0]
        recent_messages = user_messages[-3:]

        # Detect language for each recent message
        detected_languages = []
        for msg in recent_messages:
            lang = self.detect_language(msg)
            detected_languages.append(lang)

        # If all recent messages are in the same language, consider it settled
        if len(set(detected_languages)) == 1:
            return detected_languages[0]

        return False