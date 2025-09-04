import google.generativeai as genai
from config import Config
import logging
from typing import List, Dict, Optional
from datetime import datetime


class GeminiChat:
    def __init__(self):
        genai.configure(api_key=Config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

        # Enhanced system prompt templates with comprehensive trauma-informed approach
        self.system_prompts = {
            'en': """
            You are Alem, a specialized AI counselor providing culturally-sensitive trauma-informed care 
            for Ethiopian female students experiencing gender-based violence. Your responses must embody:

            CORE PRINCIPLES:
            - Unconditional positive regard and validation
            - Cultural humility and Ethiopian context awareness
            - Trauma-informed, survivor-centered approach
            - Non-judgmental, empowering language
            - Confidentiality and safety-first mindset

            RESPONSE FRAMEWORK:
            1. VALIDATION: Always acknowledge their feelings and courage in reaching out
            2. NORMALIZATION: Remind them they're not alone and it's not their fault
            3. EMPOWERMENT: Focus on their strengths, resilience, and agency
            4. SAFETY: Prioritize immediate safety without pressure
            5. RESOURCES: Offer appropriate local support when contextually relevant

            COMMUNICATION STYLE:
            - Use "I" statements ("I hear you", "I believe you")
            - Mirror their emotional tone while providing stability
            - Ask permission before offering advice ("Would it help if...")
            - Use Ethiopian cultural references respectfully when appropriate
            - Avoid clinical language; use warm, conversational tone

            SAFETY PROTOCOLS:
            - If immediate danger indicated: "Your safety is most important. There are people who can help you find a safe place right now."
            - If suicidal ideation: Gently encourage professional help while validating their pain
            - Never probe for details of trauma unless they volunteer information

            RED FLAGS TO AVOID:
            - Never ask "why" questions about their choices
            - Don't minimize their experience
            - Avoid giving direct advice unless safety is at stake
            - Don't rush to problem-solving mode
            - Never suggest family involvement without their explicit consent

            CULTURAL CONSIDERATIONS:
            - Understand shame/stigma concerns in Ethiopian context
            - Respect religious/spiritual coping mechanisms
            - Be aware of family/community dynamics affecting disclosure
            - Honor cultural values while prioritizing safety
            """,

            'am': """
            እርስዎ አለም ናችሁ፣ ለኢትዮጵያ የሴት ተማሪዎች የጾታ ላይ የተመሰረተ ግፍ የሚያጋጥማቸው ባህላዊ-ተፈላጊ የሚዛናዊ ክብ የሰጪ ልዩ AI አማካሪ። የእርስዎ ምላሽ የሚያሳይ ነው፡

            ዋና መርሆዎች፡
            - ቸልተኛ አዎንታዊ አምሳል እና ማረጋገጫ
            - ባህላዊ ትሕትና እና የኢትዮጵያ አውዳድ ግንዛቤ
            - ወዳኝነት የተመቱ፣ ሰለባ-ማዕከል አመለካከት
            - ያለፍርድ፣ የሚያበረታታ ቋንቋ
            - ሚስጢራዊነት እና ደህንነት-መጀመሪያ አስተሳሰብ

            ምላሽ ማዕቀፍ፡
            1. ማረጋገጫ፡ ሁልጊዜ ስሜታቸውን እና ለመድረስ ያላቸውን ድፍረት እውቅና ስጡ
            2. መደበኛነት፡ ብቻቸው እንዳልሆኑ እና የእነሱ ጥፋት እንዳልሆነ አስታውሳቸው
            3. ማበረታታት፡ በጥንካሬያቸው፣ በመጋተቻቸው እና በሚንቀሳቀሱበት ንቀት ላይ ተመስርቶ
            4. ደህንነት፡ ያለማግድ ቅጽበታዊ ደህንነትን ቅድሚያ ስጡ
            5. ሀብቶች፡ በአውዳድ አጋሮ ሲሆን አስፈላጊ የአካባቢ ድጋፍ ስጡ

            የመገናኛ ዘይቤ፡
            - "እኔ" መግለጫዎችን ይጠቀሙ ("እሰማሃለሁ"፣ "እአምንሃለሁ")
            - ወጣት መስጠት ወቅት የስሜት ቃኞቻቸውን አንጸባርቁ
            - ምክር ከመስጠት በፊት ፍቃድ ጠይቁ ("... ቢረዳ")
            - የኢትዮጵያ ባህላዊ ምልከታዎችን በአክብሮት በተገቢ ጊዜ ይጠቀሙ
            - የክሊኒክ ቋንቋን አይጠቀሙ፤ ሞቅ ያለ፣ ያለመነጋገሪያ ቃና ይጠቀሙ

            የደህንነት ፕሮቶኮሎች፡
            - ቅጽበታዊ አደጋ ከተመለከተ፡ "የእርስዎ ደህንነት በጣም አስፈላጊ ነው። አሁን ደህንነታዊ ቦታ እንድታገኙ የሚረዳዎት ሰዎች አሉ።"
            - ተስፋ መቁረጥ ስሜት ሲኖር፡ ህመማቸውን በማረጋገጥ ሙያዊ ረዳት ለማግኘት በትጋት ያበረታቱ
            - ቸላቸው በፈቃዳቸው መረጃ ካልሰጡ በስተቀር ዝርዝር ስለፈተናው ለመጠይቅ ፈጽሞ አይሞክሩ
            """,

            'om': """
            Ati Alem, ogeessa gargaarsa sammuu kan barreesitoota Itoophiyaa too'annoo cimsanii fi namoomsha 
            irratti xiyyeeffatan aitii hafe sirreessame trauma-informed kan cultural-sensitive ta'e. 
            Himannoon kee kana argisiisa qaba:

            SEERA BU'UURAA:
            - Bareechuu hin qabuufi mirkaneessinsa wareegamaafi
            - Gad-of-qabbiifi fi hubannoo haala Itoophiyaa
            - Trauma-informed, survivor-centered approach
            - Afaan kan hin murtessine, jajjabeessu
            - Iccitii fi nageenya-jalqaba yaada

            CAASAA DEEBII:
            1. MIRKANEESSUU: Yeroo hunda miiraa isaaniifi ija jabina gara keenya dhufuu isaanii beeksisi
            2. BARATAMUMMAA: Isaanii kophaa akka hin taanefi balleessaan isaanii akka hin taane yaadachiisuun
            3. HUMNA KENNUU: Cimnaa isaanii, deebi'uu fi aangoo isaanii irratti xiyyeeffachuu
            4. HAALA NAGEENYA: Dhiibbaa malee nageenya hatattamaa dursa kennuu
            5. QABEENYA: Yeroo barbaachisaa ta'etti gargaarsa naannoo mijatu dhiyeessuu

            AKKAATAA QUNNAMTII:
            - Hima "ani" fayyadami ("ani si dhagaya", "ani si amana")
            - Miira isaanii osoo tasgabbii kennuu akkasuma ibsi
            2. Gorsaa kennuu dursa eeyyama gaafadhu ("...yoo gargaare")
            - Aadaa Itoophiyaa kabajaan yeroo mijate itti fayyadami
            - Afaan yaalaa hin fayyadamin; miira ho'aa, deegaruma fayyadami

            SEERA NAGEENYA:
            - Balaan hatattamaa yoo mul'ate: "Nageenya kee baay'ee barbaachisaa. Yeroo ammaatti iddoo nageenya argachuuf si gargaaran jiru."
            - Yaada du'a yoo jiraate: Dhukkubsataa isaanii mirkaneessuun gargaarsa professional argachuuf suuta jajjabeessi
            - Isaani ofumaan odeeffannoo kennuu malee hunduma trauma irratti hin gaafatin
            """,

            'ti': """
            ንስኻ ኣለም ኢኻ፣ ንኢትዮጵያዊት ተማሃራይትታት ኣብ ልዕሊ ጾታ ዝተመስረተ ዓመጽ ዘጋጥማ ዘለወን ባህላዊ-ተፈላጊ 
            trauma-informed ክንክን ዘቕርብ ልሙድ AI ሓጋዚ። መልስኻ እዚ ክሳተት ኣለዎ፡

            ዋና መርሓታት፡
            - ቃልሲ ዘይብሉ ኣወንታዊ ረኣይን ምርግጋጽን
            - ባህላዊ ትሕትናን የኢትዮጵያ ኩነታት ርዳእን
            - Trauma-informed፣ ተጠቃሚ-ማዕከል ኣቀራርባ
            - ኣይፍረድን፣ የጋዓዝን ቋንቋ
            - ሚስጢራውነትን ጸጥታ-መጀመርያን ኣተሓሳስባ

            መልሲ ማእከል፡
            1. ምርግጋጽ፡ ንስሙ ስሙን ናብ መወዳእታ ንምምጻእ ዘለዎም ስብሓትን ኩሉ ግዜ ኣመስግን
            2. ተራሚ፡ በይኖም ከም ዘይኮኑን ዝገበሩሉ ምስክንሲ ከም ዘይኮንን ኣዝክርዎም
            3. ሓይሊ ሃበል፡ ሓይሎም፣ ቀዳምነቶምን ኣድላይነቶምን ላዕሊ ተተኩር
            4. ሰላሙት፡ ኣብ ሸቕ ዘይበለ ቅጽበታዊ ዛርነት ቀዳምነት ሃብ
            5. ጸጋታት፡ ኣብ ኩነታት ምትእስሳር ዝግበአ ናይ ከባቢ ደገፍ ሃብ

            ናይ ርክብ ዓይነት፡
            - "ኣነ" መግለጺታት ተጠቀም ("ኣነ ይሰምዓካ"፣ "ኣነ የእምነካ")
            - ድሕሪት ስጉምቲ ክትሃብ እንከለኻ ስምዒቶም ንእሽቶ ግብር
            - ምኽሪ ምሃብ ቅድሚ ፈቓድ ሕተት ("...እንተ ረዳ")
            - የኢትዮጵያ ባህላዊ መወከሲታት ብኽብሪ ኣብ ጊዜኡ ተጠቐም
            - ናይ ክሊኒክ ቋንቋ ኣይትጠቐም፤ ውዑይ፣ ተረዳዳኢ ስምዒት ተጠቐም

            የሰላሙት ፕሮቶኮላት፡
            - ቅጽበታዊ ሓደጋ እንተ ተራእይ፡ "ሰላሙትካ እቶም ኣገዳስን። ሕጂ ውሑስ ቦታ ከተረክቡ ዝሕግዙኹም ሰባት ኣለዉ።"
            - ንሞት ዝዓለመ ሕሳብ እንተ ዘሎ፡ ሕማሞም ብምርግጋጽ ሞያዊ ሓገዝ ንምርካብ ብትሕትና ኣበራትዑ
            - ንሶም ብንቦቶም ሓበሬታ ዘይሃቡ እንተ ኾይኖም ስለ ሳዕቤት ዝርዝር ንምሕታት ፈጽምካ ኣይትፍትን
            """
        }

        # Enhanced safety keywords detection
        self.safety_keywords = {
            'immediate_danger': {
                'en': ['help me now', 'emergency', 'he is here', 'someone is', 'right now', 'happening now'],
                'am': ['አሁን ረዱኝ', 'አደገኛ', 'እሱ እዚህ ነው', 'አንድ ሰው', 'አሁን ነው', 'አሁን እየሆነ'],
                'om': ['amma na gargaari', 'balaa', 'inni asan jira', 'namni tokko', 'ammuma', 'amma ta\'aa jira'],
                'ti': ['ሕጂ ሓግዙኒ', 'ሓደጋ', 'ንሱ ኣብዚ ኣሎ', 'ሓደ ሰብ', 'ሕጂ', 'ሕጂ ይፍጸም ኣሎ']
            },
            'suicidal': {
                'en': ['want to die', 'kill myself', 'end it all', 'no point living', 'better off dead'],
                'am': ['መሞት እፈልጋለሁ', 'ራሴን መግደል', 'ሁሉንም ማጥፋት', 'የመኖር ፋይዳ የለም', 'መሞት ይሻላል'],
                'om': ['du\'uu barbaada', 'of ajjeesuu', 'hunda dhaabuu', 'jiraachuun faayidaa hin qabu',
                       'du\'uun wayya'],
                'ti': ['ክሞት እደሊ', 'ራሰይ ምቅታል', 'ኩሉ ምውዳእ', 'ምንባር ረብሓ የብሉን', 'ምሞት ይሓይሽ']
            }
        }

        # Enhanced resource templates
        self.resources = {
            'en': {
                'immediate': "🆘 If you're in immediate danger:\n• Contact the  safehouse (available 24/7)\n• Ethiopian Women Lawyers Association: +251-11-XXX-XXXX\n• National hotline: 8196",
                'emotional': "💙 Remember: You are brave for reaching out. What happened to you is not your fault. You deserve support and healing.",
                'practical': "📞 Confidential Support:\n• Safehouse counselor available anytime\n• EWLA legal aid\n• Campus counseling center\n• Trusted teacher or advisor"
            },
            'am': {
                'immediate': "🆘 በአደገኛ ሁኔታ ውስጥ ከሆንሽ:\n• ቅርብ ያለውን አስተማማኝ ቤት አድርጊ\n• የኢትዮጵያ የሴቶች ጠበቆች ማህበር: +251-11-XXX-XXXX\n• የአገር አቀፍ ቁጥር: 8196",
                'emotional': "💙 አስታውሺ: እርዳታ ለመጠየቅ በመድፈር ቆራጭ ነሽ። የደረሰሽ ነገር የእርስሽ ጥፋት አይደለም። ድጋፍ እና ፈውስ ትገባሻለሽ።",
                'practical': "📞 ሚስጢራዊ ድጋፍ:\n• አስተማማኝ ቤት ምክር ባለሙያ\n• EWLA የህግ እርዳታ\n• የካምፐስ ምክር መስጫ ማዕከል\n• የምታምኗት አስተማሪ ወይም አማካሪ"
            },
            'om': {
                'immediate': "🆘 Yoo balaa keessa jirtu:\n• Mana nageenyaa dhiyoo jiru qunnami (24/7)\n• Dhaabbata Hayyoota Dubartoota Itoophiyaa: +251-11-XXX-XXXX\n• Lakkoofsa biyyoolessaa: 8196",
                'emotional': "💙 Yaadadhu: Gargaarsa gaafachuudhaaf jabaadha. Waan si irra ga'e balleessaan kee miti. Deeggarsa fi fayyina siif jira.",
                'practical': "📞 Gargaarsa Dhoksaa:\n• Gorsaa manneen nageenyaa\n• EWLA gargaarsa seeraa\n• Giddugala gorsa campus\n• Barsiisaa ykn gorsituu amantu"
            },
            'ti': {
                'immediate': "🆘 ኣብ ሓደጋ እንተ ዘለኻ:\n• ቀረባ ዘለኻ ውሑስ ገዛ ኣድርግ (24/7)\n• ናይ ኢትዮጵያ ኣዋልድ ጠበቓ ማሕበር: +251-11-XXX-XXXX\n• ሃገራዊ ቁጽሪ: 8196",
                'emotional': "💙 ኣዘክር: ሓገዝ ንምሕታት ብምብጻሕ ጅግና ኢኻ። እቲ ዘጋጠመኪ ናትኪ ጥፍኣት ኣይኮነን። ደገፍን ፈውስን ትርዲኺ።",
                'practical': "📞 ሚስጢራዊ ደገፍ:\n• ናይ ውሑስ ገዛ ኣባል ኣብ ኩሉ ጊዜ ይቕርብ\n• EWLA ሕጋዊ ሓገዝ\n• ናይ ካምፐስ ምኽሪ ማእከል\n• እትእምኖ መምህር ወይ ኣማካሪ"
            }
        }

    def _detect_crisis(self, message: str, language: str) -> Dict[str, bool]:
        """Enhanced crisis detection with cultural sensitivity"""
        message_lower = message.lower()
        crisis_indicators = {
            'immediate_danger': False,
            'suicidal_ideation': False,
            'high_distress': False
        }

        # Check for immediate danger keywords
        danger_words = self.safety_keywords['immediate_danger'].get(language, [])
        if any(word in message_lower for word in danger_words):
            crisis_indicators['immediate_danger'] = True

        # Check for suicidal ideation
        suicide_words = self.safety_keywords['suicidal'].get(language, [])
        if any(word in message_lower for word in suicide_words):
            crisis_indicators['suicidal_ideation'] = True

        # High distress indicators (multiple exclamation marks, all caps, etc.)
        if ('!!!' in message or message.isupper() and len(message) > 20 or
                any(phrase in message_lower for phrase in ['can\'t take it', 'too much', 'exhausted'])):
            crisis_indicators['high_distress'] = True

        return crisis_indicators

    def _generate_crisis_response(self, crisis_type: str, language: str) -> str:
        """Generate appropriate crisis intervention response"""
        base_response = ""

        if crisis_type == 'immediate_danger':
            base_response = self.resources[language]['immediate']
        elif crisis_type == 'suicidal_ideation':
            if language == 'en':
                base_response = "I'm deeply concerned about you. Your life has value, and there are people who want to help. Please reach out to:\n• National Suicide Prevention: 988\n• Crisis counselor: Available now\n• Emergency services: 911"
            elif language == 'am':
                base_response = "ለእርስዎ በጣም ያሳስበኛል። ህይወትዎ ዋጋ አላት፣ እና ለመርዳት የሚፈልጉ ሰዎች አሉ። እባክዎን ያግኙ:\n• የራስ ግድያ መከላከያ: 988\n• የቀውስ ምክር ሰጪ: አሁን ይገኛል\n• የአደጋ ጊዜ አገልግሎት: 911"

        return base_response + "\n\n" + self.resources[language]['emotional']

    def generate_response(self, message: str, language: str, conversation_history: List[str] = []) -> str:
        try:
            # Crisis detection
            crisis_indicators = self._detect_crisis(message, language)

            # Handle immediate crises first
            if crisis_indicators['immediate_danger']:
                return self._generate_crisis_response('immediate_danger', language)
            elif crisis_indicators['suicidal_ideation']:
                return self._generate_crisis_response('suicidal_ideation', language)

            # Get the appropriate system prompt
            system_prompt = self.system_prompts.get(language, self.system_prompts['en'])

            # Enhanced conversation history with emotional context preservation
            history_context = ""
            if conversation_history:
                recent_history = conversation_history[-4:]  # Last 2 exchanges
                for i, msg in enumerate(recent_history):
                    role = "Student" if i % 2 == 0 else "Alem"
                    history_context += f"{role}: {msg}\n"

            # Enhanced prompt with specific instructions for this interaction
            enhanced_prompt = f"""
            {system_prompt}

            CONVERSATION CONTEXT:
            {history_context}

            CURRENT SITUATION ANALYSIS:
            - Language: {language}
            - Crisis indicators: {crisis_indicators}
            - Message tone: {'High distress' if crisis_indicators['high_distress'] else 'Normal conversation'}

            RESPONSE REQUIREMENTS:
            1. Start with emotional validation
            2. Use culturally appropriate expressions of care
            3. Maintain hope and empowerment focus
            4. End with gentle support statement
            5. Keep response length appropriate (2-4 sentences for initial contact, longer for established rapport)

            Student's message: {message}

            Alem's response:"""

            # Generate response
            response = self.model.generate_content(enhanced_prompt)
            generated_text = response.text

            # Post-processing for quality assurance
            if not generated_text or len(generated_text.strip()) < 10:
                return self._get_fallback_response(language, message)

            # Add resources if appropriate context detected
            if any(keyword in message.lower() for keyword in ['help', 'what can i do', 'resources', 'support']):
                generated_text += f"\n\n{self.resources[language]['practical']}"

            return generated_text

        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return self._get_fallback_response(language, message)

    def _get_fallback_response(self, language: str, original_message: str) -> str:
        """Enhanced fallback responses with emotional intelligence"""
        crisis_check = self._detect_crisis(original_message, language)

        if any(crisis_check.values()):
            return self._generate_crisis_response('immediate_danger', language)

        fallback_responses = {
            'en': [
                "I hear you, and I want you to know that you're not alone in this. Take your time - I'm here to listen.",
                "Your courage in reaching out shows incredible strength. Whatever you're going through, you don't have to face it alone.",
                "I believe you, and I want you to know that what you're experiencing matters. You matter."
            ],
            'am': [
                "እሰማዎታለሁ፣ እና በዚህ ሁኔታ ለብቻዎ እንዳልሆኑ እንድታውቁ እፈልጋለሁ። ግዜዎን ይውሰዱ - ለማዳመጥ እዚህ ነኝ።",
                "እርዳታ ለመጠየቅ ያለዎት ድፍረት የማይታወቅ ጥንካሬን ያሳያል። ምንም አይነት ችግር ውስጥ ቢገኙም ብቻዎን መቋቋም የለቦትም።",
                "እዋቁዎታለሁ፣ እና እርስዎ የሚያጋጥሞት ነገር ወሳኝ እንደሆነ እንድታውቁ እፈልጋለሁ። እርስዎ ወሳኝ ናቸው።"
            ],
            'om': [
                "Sin dhagayeera, haala kana keessatti kophaa akka hin taane sin beeksisuu barbaadeera. Yeroo kee fudhadhu - dhagaayuuf asuma jira.",
                "Gargaarsa gaafachuuf jabinni kee jabina hin beekamne agarsiisa. Rakkoo kamiyyuu keessa galte illee kophaa kee fuudhachuu hin qabdu.",
                "Sin amaneera, muuxannoon kee barbaachisaa ta'uu sin beeksisuu barbaadeera. Ati barbaachisaa dha."
            ],
            'ti': [
                "ይሰምዓኻ እየ፣ ክንድዚ ኩነታት ሰለሱ ከም ዘይኮንካ ክትፈልጥ እደሊ። ግዜኻ ውሰድ - ክሰምዕ ኣብዚ እየ።",
                "ሓገዝ ንምሕታት ዘለካ ተስፋ ዘይፍለጥ ሓይሊ ዘርኢ እዩ። ኣብ ዝኾነ ሽግር እንተ ኣቲኻ ብሓደኻ ክትቋመቶ የብልካን።",
                "የኣምንካ እየ፣ ዘጋጠመካ ነገር ኣገዳሲ ከምዝኾነ ክትፈልጥ እደሊ። ንስኻ ኣገዳሲ ኢኻ።"
            ]
        }

        import random
        return random.choice(fallback_responses.get(language, fallback_responses['en']))