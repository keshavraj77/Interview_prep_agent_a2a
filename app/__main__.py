import sys
import os
import logging

import click
import uvicorn
import httpx

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import BasePushNotificationSender, InMemoryTaskStore, InMemoryPushNotificationConfigStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from dotenv import load_dotenv

from .interview_prep_executor import InterviewPrepAgentExecutor
from .interview_prep_agent import InterviewPrepAgent

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Exception for missing API key."""


@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=10001)
def main(host, port):
    """Starts the Interview Preparation Agent server."""
    try:
        # Validate Google API key
        if not os.getenv('GOOGLE_API_KEY'):
            raise MissingAPIKeyError(
                'GOOGLE_API_KEY environment variable not set.'
            )

        logger.info("Starting Interview Preparation Agent server")
        logger.info(f"Google API Key configured: {'Yes' if os.getenv('GOOGLE_API_KEY') else 'No'}")
        logger.info(f"Web search enabled: {os.getenv('ENABLE_WEB_SEARCH', 'true')}")
        logger.info(f"Push notifications enabled: {os.getenv('ENABLE_PUSH_NOTIFICATIONS', 'true')}")

        # Define agent capabilities
        capabilities = AgentCapabilities(
            streaming=True,
            pushNotifications=True
        )

        # Define agent skills
        skills = [
            AgentSkill(
                id='interview_preparation',
                name='Interview Preparation Planning',
                description='Creates personalized interview preparation plans with multi-turn conversation',
                tags=['interview', 'preparation', 'career', 'coaching', 'personalized'],
                examples=[
                    'I want to prepare for software engineering interviews',
                    'Help me get ready for algorithms and system design interviews',
                    'Create a study plan for technical interviews at FAANG companies'
                ]
            ),
            AgentSkill(
                id='domain_specific_prep',
                name='Domain-Specific Interview Guidance',
                description='Provides targeted preparation for specific interview domains',
                tags=['algorithms', 'system_design', 'databases', 'machine_learning'],
                examples=[
                    'I need help with system design interviews',
                    'Focus on machine learning interview preparation',
                    'Help me with database and SQL interview questions'
                ]
            ),
            AgentSkill(
                id='company_research',
                name='Company-Specific Interview Research',
                description='Researches interview patterns and preparation for specific companies',
                tags=['company_research', 'interview_patterns', 'targeted_prep'],
                examples=[
                    'Help me prepare for interviews at Google',
                    'What should I expect in Amazon interviews?',
                    'Research interview process for startup companies'
                ]
            )
        ]

        # Create agent card
        agent_card = AgentCard(
            name='Interview Preparation Agent',
            description='A comprehensive interview preparation coach that provides personalized study plans, domain-specific guidance, and company research through multi-turn conversations',
            url=f'http://{host}:{port}/',
            version='1.0.0',
            defaultInputModes=InterviewPrepAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=InterviewPrepAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=skills,
        )

        # Initialize HTTP client
        httpx_client = httpx.AsyncClient()

        # Initialize stores
        push_config_store = InMemoryPushNotificationConfigStore()
        task_store = InMemoryTaskStore()

        # Create request handler with interview prep executor
        request_handler = DefaultRequestHandler(
            agent_executor=InterviewPrepAgentExecutor(httpx_client),
            task_store=task_store,
            push_config_store=push_config_store,
            push_sender=BasePushNotificationSender(httpx_client, push_config_store),
        )

        # Create and configure the server
        server = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler
        )

        logger.info(f"Interview Preparation Agent server starting on http://{host}:{port}")
        logger.info("Agent capabilities:")
        logger.info(f"  - Streaming: {capabilities.streaming}")
        logger.info(f"  - Push Notifications: {capabilities.push_notifications}")
        logger.info(f"  - Skills: {len(skills)}")

        # Start the server
        uvicorn.run(server.build(), host=host, port=port)

    except MissingAPIKeyError as e:
        logger.error(f'Error: {e}')
        logger.error('Please set your GOOGLE_API_KEY in the environment or .env file')
        sys.exit(1)
    except Exception as e:
        logger.error(f'An error occurred during server startup: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()


# Additional configuration and testing utilities

def create_test_client():
    """Create a test client for the Interview Preparation Agent."""
    try:
        # Ensure required environment variables are set
        if not os.getenv('GOOGLE_API_KEY'):
            logger.warning("GOOGLE_API_KEY not set. Some functionality may not work.")

        # Create HTTP client
        httpx_client = httpx.AsyncClient()

        # Create stores
        push_config_store = InMemoryPushNotificationConfigStore()
        task_store = InMemoryTaskStore()

        # Create agent executor
        agent_executor = InterviewPrepAgentExecutor(httpx_client)

        return {
            'agent_executor': agent_executor,
            'task_store': task_store,
            'push_config_store': push_config_store,
            'httpx_client': httpx_client
        }

    except Exception as e:
        logger.error(f"Error creating test client: {e}")
        return None


def validate_environment():
    """Validate that all required environment variables are set."""
    required_vars = ['GOOGLE_API_KEY']
    optional_vars = {
        'ENABLE_WEB_SEARCH': 'true',
        'ENABLE_PUSH_NOTIFICATIONS': 'true',
        'BASE_API_URL': 'http://localhost:8000',
        'PROCESSING_DELAY_SECONDS': '5',
        'CALLBACK_TIMEOUT_SECONDS': '60'
    }

    missing_required = []
    for var in required_vars:
        if not os.getenv(var):
            missing_required.append(var)

    if missing_required:
        logger.error(f"Missing required environment variables: {', '.join(missing_required)}")
        logger.error("Please set these variables in your .env file or environment")
        return False

    # Log optional variables
    logger.info("Environment configuration:")
    for var, default in optional_vars.items():
        value = os.getenv(var, default)
        logger.info(f"  {var}: {value}")

    return True


# Export key components for testing
__all__ = [
    'main',
    'create_test_client',
    'validate_environment',
    'InterviewPrepAgentExecutor',
    'InterviewPrepAgent'
]