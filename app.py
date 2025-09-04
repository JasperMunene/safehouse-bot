from flask import Flask, request, jsonify, session
from flask_restful import Api, Resource
from flask_cors import CORS
from gemini_integration import GeminiChat
from language_detection import LanguageDetector
from config import Config
import re

app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = Config.SECRET_KEY
api = Api(app)
CORS(app)  # Enable CORS for all routes

# Initialize services
gemini_chat = GeminiChat()
language_detector = LanguageDetector()


class ChatResource(Resource):
    def post(self):
        try:
            data = request.get_json()
            user_message = data.get('message', '').strip()

            if not user_message:
                return jsonify({
                    'response': Config.INITIAL_GREETING,
                    'escalate': False,
                    'language_settled': False
                })

            # Get or initialize session data
            if 'conversation_history' not in session:
                session['conversation_history'] = []
                session['language'] = None
                session['language_settled'] = False

            conversation_history = session['conversation_history']
            current_language = session.get('language')
            language_settled = session.get('language_settled', False)

            # Add user message to conversation history
            conversation_history.append(user_message)

            # Detect language if not already settled
            if not language_settled:
                detected_language = language_detector.detect_language(user_message)

                # Check if language is consistently used
                settled_language = language_detector.is_language_settled(conversation_history)

                if settled_language:
                    current_language = settled_language
                    language_settled = True
                    session['language'] = current_language
                    session['language_settled'] = True
                else:
                    # Use detected language for this response, but don't settle yet
                    current_language = detected_language

            # Check for escalation keywords
            should_escalate = self.check_escalation(user_message, current_language)

            if should_escalate:
                response = Config.HANDOVER_MESSAGES.get(
                    current_language,
                    Config.HANDOVER_MESSAGES['en']
                )
                # Clear conversation history for privacy
                session.pop('conversation_history', None)
                session.pop('language', None)
                session.pop('language_settled', None)
                return jsonify({
                    'response': response,
                    'escalate': True,
                    'language_settled': language_settled
                })

            # Generate empathetic response
            bot_response = gemini_chat.generate_response(
                user_message,
                current_language,
                conversation_history
            )

            # Update conversation history
            conversation_history.append(bot_response)
            session['conversation_history'] = conversation_history[-6:]  # Keep last 3 exchanges

            return jsonify({
                'response': bot_response,
                'escalate': False,
                'language_settled': language_settled,
                'current_language': current_language
            })

        except Exception as e:
            return jsonify({
                'response': 'I hear you. Would you like to share more?',
                'escalate': False,
                'language_settled': False
            })

    def check_escalation(self, message, language):
        """Check if message contains escalation keywords"""
        keywords = Config.ESCALATION_KEYWORDS.get(
            language,
            Config.ESCALATION_KEYWORDS['en']
        )
        message_lower = message.lower()

        for keyword in keywords:
            if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', message_lower):
                return True
        return False


class SessionResource(Resource):
    def delete(self):
        """Clear the current session"""
        session.clear()
        return jsonify({'message': 'Session cleared'})


# Add resources
api.add_resource(ChatResource, '/api/chat')
api.add_resource(SessionResource, '/api/session')

if __name__ == '__main__':
    app.run(debug=True, port=5000)