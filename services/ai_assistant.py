import requests
import json
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("ai_assistant.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


class AIAssistant:
    def __init__(self):
        logger.info("Initializing AIAssistant")
        # Configuration for local AI model
        self.local_model_url = "http://localhost:11434/api/generate"  # Ollama default
        self.model_name = (
            "chevalblanc/gpt-4o-mini:latest"  # Default model, can be changed
        )
        self.max_tokens = 5000  # Reduced for faster responses

        # Fallback to OpenAI if local model is not available
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.use_openai = False

        logger.info(f"Local model URL: {self.local_model_url}")
        logger.info(f"Model name: {self.model_name}")
        logger.info(f"OpenAI API key available: {bool(self.openai_api_key)}")

    def set_local_model(
        self, model_name=None, url="http://localhost:11434/api/generate"
    ):
        """Configure local AI model settings"""
        logger.info(f"Setting local model - Model: {model_name}, URL: {url}")
        if model_name is not None:
            self.model_name = model_name
        self.local_model_url = url
        logger.info(
            f"Local model updated - Model: {self.model_name}, URL: {self.local_model_url}"
        )

    def generate_response(
        self, question, resume_data=None, job_description=None, context=""
    ):
        """Generate AI response to interview question"""
        logger.info(f"Generating response for question: {question[:100]}...")
        logger.debug(f"Resume data available: {bool(resume_data)}")
        logger.debug(f"Job description available: {bool(job_description)}")
        logger.debug(f"Context length: {len(context)}")

        try:
            # Try local model first
            logger.info("Attempting to generate response using local model")
            response = self._generate_local_response(
                question, resume_data, job_description, context
            )
            if response:
                logger.info("Successfully generated response using local model")
                return response

            # Fallback to OpenAI if local model fails
            if self.openai_api_key:
                logger.info("Local model failed, attempting OpenAI fallback")
                response = self._generate_openai_response(
                    question, resume_data, job_description, context
                )
                if response:
                    logger.info("Successfully generated response using OpenAI")
                    return response

            # Final fallback to template response
            logger.info("AI models failed, using template response")
            response = self._generate_template_response(
                question, resume_data, job_description
            )
            logger.info("Generated template response")
            return response

        except Exception as e:
            logger.error(f"Error generating AI response: {e}", exc_info=True)
            return "I apologize, but I'm having trouble generating a response right now. Could you please repeat the question?"


    def _generate_local_response(self, question, resume_data, job_description, context):
        """Generate response using local AI model (Ollama)"""
        try:
            logger.debug("Building prompt for local model")
            # Construct prompt
            prompt = self._build_prompt(question, resume_data, job_description, context)
            logger.debug(f"Prompt length: {len(prompt)}")

            # Prepare request payload
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Lower temperature for faster, more focused responses
                    "max_tokens": self.max_tokens,
                    "top_p": 0.8,  # Slightly lower for more focused responses
                    "top_k": 40,  # Add top_k for better response quality
                    "repeat_penalty": 1.1,  # Prevent repetition
                },
            }

            logger.info(f"Making request to local model: {self.local_model_url}")
            logger.info(f"Using model: {self.model_name}")
            # Make request to local model
            response = requests.post(self.local_model_url, json=payload, timeout=30)

            if response.status_code == 200:
                result = response.json()
                ai_response = result.get("response", "").strip()
                if ai_response[:7] == "<think>":
                    ai_response = ai_response.split("</think>")[1]
                logger.info(ai_response)
                logger.info("Local model request successful")
                logger.debug(f"Raw response length: {len(ai_response)}")
                return ai_response
            else:
                logger.warning(
                    f"Local model request failed with status {response.status_code}"
                )
                logger.debug(f"Response content: {response.text}")
                return None

        except requests.RequestException as e:
            logger.error(f"Error connecting to local model: {e}")
            return None
        except Exception as e:
            logger.error(f"Error with local model: {e}", exc_info=True)
            return None


    def _generate_openai_response(
        self, question, resume_data, job_description, context
    ):
        """Generate response using OpenAI API (fallback)"""
        try:
            logger.debug("Importing OpenAI client")
            from openai import OpenAI

            # Initialize OpenAI client
            client = OpenAI(api_key=self.openai_api_key)
            logger.debug("OpenAI client initialized")

            # Construct prompt
            prompt = self._build_prompt(question, resume_data, job_description, context)
            logger.debug(f"OpenAI prompt length: {len(prompt)}")

            logger.info("Making request to OpenAI API")
            # Make request to OpenAI
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI assistant helping me to prepare for job interviews. \
                         I want to know how to answer concisely and clearly to point in an interview. Provide professional, relevant responses.\
                            Answer everything in 100 to 150 words only. ALWAYS respond in English only, regardless of the language of the question.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=0.3,  # Lower temperature for faster, more focused responses
            )

            content = response.choices[0].message.content
            logger.info("OpenAI API request successful")
            logger.debug(f"OpenAI response length: {len(content) if content else 0}")
            return content.strip() if content else None

        except Exception as e:
            logger.error(f"Error with OpenAI API: {e}", exc_info=True)
            return None

    def _build_prompt(self, question, resume_data, job_description, context):
        """Build prompt for AI model"""
        logger.debug("Building AI prompt")
        prompt = f"""You are an AI assistant helping with a job interview. You are the candidate being interviewed.

CRITICAL: Respond in 100-200 words only. Be direct and professional.
IMPORTANT: Always respond in English only, regardless of the language of the question.

Question: {question}

Your Background:"""

        if resume_data:
            logger.debug("Adding resume data to prompt")
            prompt += f"""
Name: {resume_data.get('name', 'N/A')}
Skills: {', '.join(resume_data.get('skills', [])[:5])}  # Limit to top 5 skills
Experience: {resume_data.get('experience', 'N/A')[:200]}  # Limit experience length
Education: {resume_data.get('education', 'N/A')}
"""

        if job_description:
            logger.debug("Adding job description to prompt")
            prompt += f"""
Job Context: {job_description[:300]}  # Limit job description length
"""

        if context:
            logger.debug("Adding additional context to prompt")
            prompt += f"""
Context: {context[:200]}  # Limit context length
"""

        prompt += """
Instructions:
1. Answer directly and professionally
2. Use your background as reference
3. Be confident and authentic
4. Stay within 100-200 words
5. Focus on the specific question
6. No filler words or sentences
7. ALWAYS respond in English only, even if the question is in another language
8. Always answer in First Person point of view

Response:"""

        logger.debug(f"Final prompt length: {len(prompt)}")
        return prompt

    def _generate_template_response(self, question, resume_data, job_description):
        """Generate template response when AI models are not available"""
        logger.info("Generating template response")
        question_lower = question.lower()
        logger.debug(f"Question type: {question_lower[:50]}...")

        # Common interview questions and template responses
        if any(
            word in question_lower
            for word in ["tell me about yourself", "introduce yourself"]
        ):
            logger.debug("Generating self-introduction template")
            if resume_data:
                skills = resume_data.get("skills", [])
                response = f" "
                if skills:
                    response += f"I have experience with {', '.join(skills[:3])}. "
                response += "I'm excited about this opportunity and believe my skills align well with your requirements."
                logger.debug("Generated self-introduction response")
                return response

        elif any(word in question_lower for word in ["strengths", "strong points"]):
            logger.debug("Generating strengths template")
            if resume_data and resume_data.get("skills"):
                skills = resume_data.get("skills", [])
                response = f"My key strengths include {', '.join(skills[:3])}. I'm particularly strong in problem-solving and adapting to new technologies."
                logger.debug("Generated strengths response")
                return response

        elif any(
            word in question_lower for word in ["weakness", "areas for improvement"]
        ):
            logger.debug("Generating weakness template")
            response = "I'm always looking to improve my skills. I believe in continuous learning and staying updated with the latest technologies and best practices in my field."
            logger.debug("Generated weakness response")
            return response

        elif any(word in question_lower for word in ["experience", "background"]):
            logger.debug("Generating experience template")
            if resume_data and resume_data.get("experience"):
                response = "I have experience in various roles where I've developed strong technical and problem-solving skills. I'm always eager to take on new challenges and contribute to team success."
                logger.debug("Generated experience response")
                return response

        elif any(word in question_lower for word in ["why", "interested", "company"]):
            logger.debug("Generating interest template")
            response = "I'm interested in this role because it aligns with my skills and career goals. I believe I can contribute meaningfully to your team while continuing to grow professionally."
            logger.debug("Generated interest response")
            return response

        else:
            logger.debug("Generating generic template")
            response = "That's a great question. Based on my experience and background, I believe I can provide valuable insights and contribute effectively to your team."
            logger.debug("Generated generic response")
            return response

    def get_available_models(self):
        """Get list of available models from Ollama"""
        try:
            # Query Ollama for available models
            models_url = self.local_model_url.replace("/api/generate", "/api/tags")
            logger.info(f"Fetching available models from: {models_url}")

            response = requests.get(models_url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                logger.info(f"Found {len(models)} available models")

                # Format models for frontend
                formatted_models = []
                for model in models:
                    formatted_models.append(
                        {
                            "name": model.get("name", ""),
                            "modified_at": model.get("modified_at", ""),
                            "size": model.get("size", 0),
                        }
                    )

                return formatted_models
            else:
                logger.warning(f"Failed to get models, status: {response.status_code}")
                return []

        except requests.RequestException as e:
            logger.error(f"Error connecting to Ollama for models: {e}")
            return []
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return []

    def set_model(self, model_name):
        """Set the current model to use"""
        logger.info(f"set_model called with: {model_name}")
        if model_name:
            self.model_name = model_name
            logger.info(f"Model set to: {self.model_name}")
            return True
        logger.warning("set_model called with empty model_name")
        return False

    def get_current_model(self):
        """Get the current model name"""
        return self.model_name