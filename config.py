import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-please-change-in-production'
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

    # Supported languages with codes and names
    SUPPORTED_LANGUAGES = {
        'en': 'English',
        'am': 'Amharic',
        'om': 'Oromifa',
        'ti': 'Tigrigna'
    }

    DEFAULT_LANGUAGE = 'en'

    # Language detection prompts
    LANGUAGE_DETECTION_PROMPT = """
    Determine the language of the following text. 
    Respond ONLY with the language code: 'en' for English, 'am' for Amharic, 'om' for Oromifa, or 'ti' for Tigrigna.
    If uncertain, respond with 'en'.

    Text: "{text}"
    """

    # Initial greeting that prompts for language naturally
    INITIAL_GREETING = "Hello! I'm here to support you. Please feel free to speak in whichever language you're most comfortable with."

    # Escalation keywords in different languages
    ESCALATION_KEYWORDS = {
        'en': ['help', 'emergency', 'danger', 'unsafe', 'speak to someone', 'human', 'representative'],
        'am': ['እገዛ', 'አደጋ', 'አጋዥ', 'አስፈላጊ', 'ወከል', 'ሰው'],
        'om': ['gargaarsa', 'rakkoo', 'namatti himi', 'naaf bilisaa', 'namni'],
        'ti': ['ሓገዝ', 'ኣደጋ', 'ሰብ ክዛረብ', 'እገዳይ', 'ወኪል']
    }

    # Safehouse handover messages
    HANDOVER_MESSAGES = {
        'en': "I'm connecting you with a safehouse representative who can provide further support. Please wait a moment.",
        'am': "አሁን በተጨማሪ ድጋፍ ሊሰጡዎት የሚችሉ የደህንነት ቤት ተወካዮች ከግንኙነት ይፈጠራሉ። እባክዎ ይጠብቁ።",
        'om': "Ani namoota deeggarsa kanaan si gaafachuuf namoota safehouse waliin si qunnamuu. Tursaasaa ta'i.",
        'ti': "ኣነ ምስ ሰብ ሓገዝ ክህበካ እኽእል ዝኮነ ናይ ጸጥታ ገዛ ሰለድቲ ክራንቅ እየ። በጃኻ ተጸበይ።"
    }