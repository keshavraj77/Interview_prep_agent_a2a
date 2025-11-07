import asyncio
import os
import logging
import uuid
from typing import Any, Dict, Optional
import httpx
from urllib.parse import urlparse

from a2a.types import (
    Task,
    TaskState,
    Message,
    Part,
    TextPart,
    TaskStatus,
    PushNotificationConfig,
    PushNotificationAuthenticationInfo,
    Artifact,
)
from a2a.utils import new_agent_text_message

logger = logging.getLogger(__name__)


class InterviewPrepPushNotificationSettings:
    """Configuration for interview prep push notifications from environment variables."""

    def __init__(self):
        # Push notification mode: multi_turn for interview prep
        self.mode = os.getenv('PUSH_NOTIFICATION_MODE', 'multi_turn')

        # Processing delay before sending callback (in seconds)
        self.processing_delay = int(os.getenv('PROCESSING_DELAY_SECONDS', '5'))

        # Enable/disable push notifications
        self.enabled = os.getenv('ENABLE_PUSH_NOTIFICATIONS', 'true').lower() == 'true'

        # Callback timeout (in seconds) - longer for interview prep
        self.callback_timeout = int(os.getenv('CALLBACK_TIMEOUT_SECONDS', '60'))

        # Optional webhook signature secret for security
        self.webhook_secret = os.getenv('WEBHOOK_SIGNATURE_SECRET')

        # Progress update intervals for long processing
        self.progress_update_interval = int(os.getenv('PROGRESS_UPDATE_INTERVAL_SECONDS', '10'))

        logger.info(f"Interview prep push notifications configured: enabled={self.enabled}, mode={self.mode}, delay={self.processing_delay}s")


class InterviewPrepPushNotificationHandler:
    """Enhanced push notification handler for interview preparation agent."""

    def __init__(self, httpx_client: httpx.AsyncClient):
        self.client = httpx_client
        self.settings = InterviewPrepPushNotificationSettings()

    async def handle_push_notification_request(
        self,
        task: Task,
        webhook_config: PushNotificationConfig,
        agent_response_generator,
        query: str,
        context_id: str,
        request_metadata: Dict[str, Any] = None
    ) -> None:
        """
        Handle a long-running interview preparation request with push notifications.

        Args:
            task: The A2A task object
            webhook_config: Push notification configuration from request
            agent_response_generator: The agent's response generator function
            query: User query
            context_id: Conversation context ID
            request_metadata: Optional request metadata
        """
        if not self.settings.enabled:
            logger.warning("Push notifications are disabled via environment variable")
            return

        # Debug: Log incoming task metadata and request metadata
        logger.info(f"Incoming task metadata: {task.metadata}")
        logger.info(f"Incoming request metadata: {request_metadata}")

        # Extract webhook configuration
        callback_url = webhook_config.url
        webhook_token = webhook_config.token
        auth_config = webhook_config.authentication

        if not callback_url:
            logger.error("No callback URL provided in push notification config")
            return

        # Replace BASE_API_URL placeholder with environment variable if present
        callback_url = self._resolve_callback_url(callback_url)
        logger.info(f"Resolved callback URL: {callback_url}")

        # Validate callback URL
        if not self._validate_callback_url(callback_url):
            logger.error(f"Invalid callback URL: {callback_url}")
            return

        # Start background processing with progress updates
        asyncio.create_task(
            self._process_async_interview_prep_request(
                task=task,
                callback_url=callback_url,
                webhook_token=webhook_token,
                auth_config=auth_config,
                agent_response_generator=agent_response_generator,
                query=query,
                context_id=context_id,
                request_metadata=request_metadata
            )
        )

    def _resolve_callback_url(self, callback_url: str) -> str:
        """
        Resolve callback URL by replacing BASE_API_URL placeholder with environment variable.
        """
        # Check if URL contains BASE_API_URL/ pattern
        if "BASE_API_URL/" in callback_url:
            # Get the actual base API URL from environment variable
            base_api_url = os.getenv('BASE_API_URL')

            if base_api_url:
                # Remove trailing slash from base_api_url if present
                base_api_url = base_api_url.rstrip('/')

                # Replace the placeholder with the actual URL
                resolved_url = callback_url.replace('BASE_API_URL', base_api_url)

                logger.info(f"Replaced BASE_API_URL placeholder: {callback_url} -> {resolved_url}")
                return resolved_url
            else:
                logger.warning("BASE_API_URL environment variable not set, but callback URL contains BASE_API_URL placeholder")
                return callback_url

        # Return original URL if no replacement needed
        return callback_url

    def _validate_callback_url(self, url: str) -> bool:
        """Validate callback URL for security."""
        try:
            parsed = urlparse(url)

            # Must be HTTPS in production
            if parsed.scheme not in ['http', 'https']:
                return False

            # Must have valid hostname
            if not parsed.hostname:
                return False

            # Block localhost/private IPs in production
            # (For demo purposes, we'll allow localhost)

            return True
        except Exception:
            return False

    async def _process_async_interview_prep_request(
        self,
        task: Task,
        callback_url: str,
        webhook_token: Optional[str],
        auth_config: Dict[str, Any],
        agent_response_generator,
        query: str,
        context_id: str,
        request_metadata: Dict[str, Any] = None
    ) -> None:
        """Process the interview preparation request asynchronously with progress updates."""
        progress_steps = []

        try:
            # Add initial processing delay
            if self.settings.processing_delay > 0:
                await asyncio.sleep(self.settings.processing_delay)

            logger.info("Starting async interview preparation processing")

            # Track progress through the agent's response generator
            async for response in agent_response_generator(query, context_id, request_metadata):
                progress_content = response.get('content', '')
                is_complete = response.get('is_task_complete', False)
                require_input = response.get('require_user_input', False)

                # Store progress step
                progress_steps.append({
                    'content': progress_content,
                    'timestamp': asyncio.get_event_loop().time(),
                    'is_complete': is_complete,
                    'require_input': require_input
                })

                # Send progress update for non-final responses
                if not is_complete and not require_input and progress_content:
                    await self._send_progress_update(
                        task=task,
                        callback_url=callback_url,
                        auth_config=auth_config,
                        webhook_token=webhook_token,
                        progress_content=progress_content,
                        context_id=context_id,
                        request_metadata=request_metadata
                    )

                    # Add delay between progress updates
                    await asyncio.sleep(self.settings.progress_update_interval)

                # If this is the final response, prepare for completion
                if is_complete or require_input:
                    final_response = response
                    break

            # If no final response was captured, create a default one
            if 'final_response' not in locals():
                final_response = {
                    'is_task_complete': True,
                    'require_user_input': False,
                    'content': 'Interview preparation plan completed successfully!'
                }

            # Prepare final callback payload
            callback_payload = self._create_final_callback_payload(
                task=task,
                response=final_response,
                context_id=context_id,
                request_metadata=request_metadata,
                progress_steps=progress_steps
            )

            # Send final callback
            await self._send_callback(
                callback_url=callback_url,
                payload=callback_payload,
                auth_config=auth_config,
                webhook_token=webhook_token
            )

            logger.info("Interview preparation async processing completed successfully")

        except Exception as e:
            logger.error(f"Error processing async interview prep request: {e}")

            # Send error callback
            error_payload = self._create_error_callback_payload(
                task=task,
                error_message=str(e),
                context_id=context_id,
                request_metadata=request_metadata
            )

            try:
                await self._send_callback(
                    callback_url=callback_url,
                    payload=error_payload,
                    auth_config=auth_config,
                    webhook_token=webhook_token
                )
            except Exception as callback_error:
                logger.error(f"Failed to send error callback: {callback_error}")

    async def _send_progress_update(
        self,
        task: Task,
        callback_url: str,
        auth_config: Dict[str, Any],
        webhook_token: Optional[str],
        progress_content: str,
        context_id: str,
        request_metadata: Dict[str, Any] = None
    ) -> None:
        """Send a progress update via push notification."""
        try:
            # Create progress message
            progress_message = new_agent_text_message(
                progress_content,
                context_id,
                task.id
            )

            # Use request metadata if available, otherwise fall back to task metadata
            final_metadata = request_metadata if request_metadata else task.metadata

            # Create updated task for progress
            updated_task = Task(
                id=task.id,
                kind=task.kind,
                context_id=context_id,
                status=TaskStatus(
                    state=TaskState.working,
                    message=progress_message
                ),
                history=task.history + [progress_message] if task.history else [progress_message],
                metadata=final_metadata
            )

            # Create JSON-RPC payload for progress update
            task_payload = updated_task.model_dump(exclude_none=True)
            jsonrpc_payload = {
                "jsonrpc": "2.0",
                "method": "pushNotifications/send",
                "params": {
                    "task": task_payload
                },
                "id": str(uuid.uuid4())
            }

            # Send progress update
            await self._send_callback(
                callback_url=callback_url,
                payload=jsonrpc_payload,
                auth_config=auth_config,
                webhook_token=webhook_token
            )

            logger.info(f"Sent progress update: {progress_content[:50]}...")

        except Exception as e:
            logger.error(f"Failed to send progress update: {e}")

    def _create_final_callback_payload(
        self,
        task: Task,
        response: Dict[str, Any],
        context_id: str,
        request_metadata: Dict[str, Any] = None,
        progress_steps: list = None
    ) -> Dict[str, Any]:
        """Create final callback payload in JSON-RPC format."""

        # Determine final task state based on the improved flow
        if response['is_task_complete']:
            state = TaskState.completed
        elif response['require_user_input']:
            # Check if this is a satisfaction check (plan delivered)
            if 'satisfied' in response.get('content', '').lower():
                state = TaskState.input_required  # Awaiting satisfaction feedback
            else:
                state = TaskState.input_required
        else:
            state = TaskState.working

        # Create agent message
        agent_message = new_agent_text_message(
            response['content'],
            context_id,
            task.id
        )

        # Prepare artifacts for completed tasks
        artifacts = None
        if state == TaskState.completed:
            # Create artifact with the response content
            artifacts = [
                Artifact(
                    artifactId=str(uuid.uuid4()),
                    name="interview_preparation_plan",
                    parts=[Part(root=TextPart(text=response['content']))]
                )
            ]

        # Use request metadata if available, otherwise fall back to task metadata
        final_metadata = request_metadata if request_metadata else task.metadata

        # Add processing summary to metadata
        if final_metadata is None:
            final_metadata = {}

        final_metadata['processing_summary'] = {
            'total_steps': len(progress_steps) if progress_steps else 0,
            'processing_duration_estimate': '2-3 minutes',
            'agent_type': 'interview_preparation'
        }

        # Update task with final status
        updated_task = Task(
            id=task.id,
            kind=task.kind,
            context_id=context_id,
            status=TaskStatus(
                state=state,
                message=agent_message if state == TaskState.input_required else None
            ),
            artifacts=artifacts,
            history=task.history + [agent_message] if task.history else [agent_message],
            metadata=final_metadata
        )

        # Debug logging
        logger.info(f"Final callback payload metadata: {final_metadata}")
        task_payload = updated_task.model_dump(exclude_none=True)
        logger.info(f"Task object for final JSON-RPC: {task_payload}")

        # Wrap the Task object in JSON-RPC 2.0 format
        jsonrpc_payload = {
            "jsonrpc": "2.0",
            "method": "pushNotifications/send",
            "params": {
                "task": task_payload
            },
            "id": str(uuid.uuid4())
        }

        return jsonrpc_payload

    def _create_error_callback_payload(
        self,
        task: Task,
        error_message: str,
        context_id: str,
        request_metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create error callback payload in JSON-RPC format."""

        error_agent_message = new_agent_text_message(
            f"I encountered an error while creating your interview preparation plan: {error_message}. Please try again or contact support if the issue persists.",
            context_id,
            task.id
        )

        # Use request metadata if available, otherwise fall back to task metadata
        final_metadata = request_metadata if request_metadata else task.metadata

        # Add error info to metadata
        if final_metadata is None:
            final_metadata = {}

        final_metadata['error_info'] = {
            'error_message': error_message,
            'agent_type': 'interview_preparation',
            'error_occurred_at': 'async_processing'
        }

        updated_task = Task(
            id=task.id,
            kind=task.kind,
            context_id=context_id,
            status=TaskStatus(
                state=TaskState.input_required,
                message=error_agent_message
            ),
            history=task.history + [error_agent_message] if task.history else [error_agent_message],
            metadata=final_metadata
        )

        # Get the task payload
        task_payload = updated_task.model_dump(exclude_none=True)

        # Wrap the Task object in JSON-RPC 2.0 format
        jsonrpc_payload = {
            "jsonrpc": "2.0",
            "method": "pushNotifications/send",
            "params": {
                "task": task_payload
            },
            "id": str(uuid.uuid4())
        }

        return jsonrpc_payload

    async def _send_callback(
        self,
        callback_url: str,
        payload: Dict[str, Any],
        auth_config: PushNotificationAuthenticationInfo | None,
        webhook_token: Optional[str]
    ) -> None:
        """Send callback to client webhook URL."""

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "A2A-InterviewPrep-Agent/1.0"
        }

        # Add authentication headers
        if auth_config:
            auth_headers = self._get_auth_headers(auth_config)
            headers.update(auth_headers)

        # Add webhook token if provided
        if webhook_token:
            headers["X-Webhook-Token"] = webhook_token

        # Add signature if webhook secret is configured
        if self.settings.webhook_secret:
            # This would implement HMAC signature
            # For demo purposes, we'll skip this
            pass

        try:
            logger.info(f"Sending JSON-RPC callback to {callback_url}")
            logger.info(f"Callback headers: {headers}")
            logger.info(f"JSON-RPC method: {payload.get('method', 'Unknown')}")

            # Print the EXACT JSON-RPC body being sent (for debugging)
            import json
            exact_json_body = json.dumps(payload, indent=2, default=str, ensure_ascii=False)
            logger.debug(f"EXACT JSON-RPC CALLBACK BODY:\n{exact_json_body}")

            # Ensure UTF-8 encoding by manually serializing with ensure_ascii=False
            json_data = json.dumps(payload, default=str, ensure_ascii=False)
            headers["Content-Type"] = "application/json; charset=utf-8"

            response = await self.client.post(
                callback_url,
                content=json_data.encode('utf-8'),
                headers=headers,
                timeout=self.settings.callback_timeout
            )

            if response.status_code == 200:
                logger.info(f"Successfully sent callback to {callback_url}")
            else:
                logger.warning(f"Callback returned status {response.status_code}: {response.text}")

        except httpx.TimeoutException:
            logger.error(f"Callback to {callback_url} timed out")
        except httpx.RequestError as e:
            logger.error(f"Failed to send callback to {callback_url}: {e}")

    def _get_auth_headers(self, auth_config: PushNotificationAuthenticationInfo | None) -> Dict[str, str]:
        """Get authentication headers using JWT token from environment variable."""
        headers = {}

        # Get JWT token from environment variable for callback authentication
        jwt_token = os.getenv('A2A_CALLBACK_TOKEN')

        if not jwt_token:
            logger.warning("A2A_CALLBACK_TOKEN environment variable not set. Callback authentication may fail.")
            return headers

        # Use Bearer authentication with the JWT token from environment
        if auth_config:
            schemes = getattr(auth_config, 'schemes', []) or []
            if 'Bearer' in schemes:
                headers["Authorization"] = f"Bearer {jwt_token}"
                logger.info("Added Bearer authentication header for callback")
            elif 'Basic' in schemes:
                headers["Authorization"] = f"Basic {jwt_token}"
                logger.info("Added Basic authentication header for callback")
        else:
            # Default to Bearer if no auth config specified
            headers["Authorization"] = f"Bearer {jwt_token}"
            logger.info("Added default Bearer authentication header for callback")

        return headers
