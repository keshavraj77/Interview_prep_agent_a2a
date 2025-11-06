import logging
import asyncio
from typing import Dict, Any, Optional, List
import httpx

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InternalError,
    InvalidParamsError,
    Part,
    Task,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import (
    new_agent_text_message,
    new_task,
)
from a2a.utils.errors import ServerError

from .interview_prep_agent import InterviewPrepAgent
from .conversation_state import ConversationPhase, ConversationState, UserInputs
from .web_search_tools import WebSearchManager

logger = logging.getLogger(__name__)


class InterviewPrepAgentExecutor(AgentExecutor):
    """
    Enhanced AgentExecutor for multi-turn Interview Preparation Agent with A2A Protocol.
    """

    def __init__(self, httpx_client: httpx.AsyncClient):
        self.agent = InterviewPrepAgent()
        self.search_manager = WebSearchManager()
        self.httpx_client = httpx_client

        # Import push notification handler
        try:
            from .push_notification_handler import InterviewPrepPushNotificationHandler
            self.push_notification_handler = InterviewPrepPushNotificationHandler(httpx_client)
        except ImportError as e:
            logger.warning(f"Push notification handler not available: {e}")
            self.push_notification_handler = None

        logger.info("InterviewPrepAgentExecutor initialized")

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """
        Execute the interview preparation agent with A2A protocol support.
        """
        try:
            # Validate request
            error = self._validate_request(context)
            if error:
                raise ServerError(error=InvalidParamsError())

            query = context.get_user_input()
            task = context.current_task

            if not task:
                task = new_task(context.message)
                await event_queue.enqueue_event(task)

            logger.info(f"Processing query: {query}")
            logger.info(f"Task ID: {task.id}, Context ID: {task.context_id}")

            # Check for push notification configuration (but don't use it immediately)
            push_notification_config = None
            if (context.configuration and
                context.configuration.push_notification_config and
                self.push_notification_handler and
                self.push_notification_handler.settings.enabled):

                push_notification_config = context.configuration.push_notification_config

            # Check if this query should trigger async processing
            should_start_async = await self._should_start_async_processing(task.context_id, query)

            if should_start_async and push_notification_config:
                logger.info(f"Starting async processing with push notifications for task {task.id}")

                # Handle with push notifications for async processing
                await self._handle_with_push_notifications(
                    context, event_queue, task, push_notification_config, query
                )
                return

            # Standard multi-turn processing (for input gathering)
            await self._handle_standard_processing(context, event_queue, task, query)

        except Exception as e:
            logger.error(f'An error occurred while processing: {e}')
            raise ServerError(error=InternalError()) from e

    async def _handle_standard_processing(
        self,
        context: RequestContext,
        event_queue: EventQueue,
        task: Task,
        query: str
    ) -> None:
        """Handle standard multi-turn processing without push notifications."""
        updater = TaskUpdater(event_queue, task.id, task.context_id)

        try:
            async for item in self.agent.stream(query, task.context_id):
                is_task_complete = item.get('is_task_complete', False)
                require_user_input = item.get('require_user_input', False)
                content = item.get('content', '')
                phase = item.get('phase', '')

                # Check if we need to start async processing
                if item.get('trigger_async_processing') or item.get('trigger_push_notification'):
                    logger.info("User confirmed processing - transitioning to async mode")

                    # Return "submitted" status immediately to the user
                    await updater.update_status(
                        TaskState.submitted,
                        new_agent_text_message(
                            "Your interview preparation request has been submitted for processing. You'll receive updates via push notifications.",
                            task.context_id,
                            task.id,
                        ),
                        final=True
                    )
                    return  # End standard processing, let async mode take over

                # Check if we need to start refinement processing
                elif item.get('trigger_refinement_processing'):
                    logger.info("Starting refinement processing")

                    # Update status to working
                    await updater.update_status(
                        TaskState.working,
                        new_agent_text_message(
                            "🔧 Processing your refinement request...",
                            task.context_id,
                            task.id,
                        ),
                    )

                    # Perform refinement
                    refined_plan = await self._generate_refined_plan(task.context_id)

                    # Complete with the refined plan
                    await updater.add_artifact(
                        [Part(root=TextPart(text=refined_plan))],
                        name='refined_interview_preparation_plan',
                    )
                    await updater.complete()
                    return

                # Handle regular conversation flow
                if not is_task_complete and not require_user_input:
                    # Working status update
                    await updater.update_status(
                        TaskState.working,
                        new_agent_text_message(
                            content,
                            task.context_id,
                            task.id,
                        ),
                    )
                elif require_user_input:
                    # Need more input from user
                    await updater.update_status(
                        TaskState.input_required,
                        new_agent_text_message(
                            content,
                            task.context_id,
                            task.id,
                        ),
                        final=True,
                    )
                    break
                else:
                    # Task completed
                    await updater.add_artifact(
                        [Part(root=TextPart(text=content))],
                        name='interview_preparation_result',
                    )
                    await updater.complete()
                    break

        except Exception as e:
            logger.error(f'Error in standard processing: {e}')
            raise

    async def _handle_with_push_notifications(
        self,
        context: RequestContext,
        event_queue: EventQueue,
        task: Task,
        push_notification_config,
        query: str
    ) -> None:
        """Handle processing with push notification support for async operations."""
        if not self.push_notification_handler:
            logger.error("Push notification handler not available")
            await self._handle_standard_processing(context, event_queue, task, query)
            return

        updater = TaskUpdater(event_queue, task.id, task.context_id)

        # Check if this query should trigger async processing
        should_start_async = await self._should_start_async_processing(task.context_id, query)

        if should_start_async:
            logger.info("Starting async processing with push notifications")

            # Send "submitted" status
            submitted_message = new_agent_text_message(
                "Great! I'm working on your personalized interview prep plan. This will take about 1-2 minutes.",
                task.context_id,
                task.id,
            )
            await updater.update_status(TaskState.submitted, submitted_message)

            # Send "working" status
            working_message = new_agent_text_message(
                "..Searching for the best interview resources and study materials for you...",
                task.context_id,
                task.id,
            )
            await updater.update_status(TaskState.working, working_message)

            # Extract metadata from context
            request_metadata = context.metadata
            logger.info(f"Request metadata from context: {request_metadata}")

            if not request_metadata:
                request_metadata = None
                logger.info("Metadata is empty, setting to None")

            # Handle async processing with push notifications
            await self.push_notification_handler.handle_push_notification_request(
                task=task,
                webhook_config=push_notification_config,
                agent_response_generator=self._async_research_and_generate,
                query=query,
                context_id=task.context_id,
                request_metadata=request_metadata
            )
        else:
            # Handle as regular multi-turn conversation
            await self._handle_standard_processing(context, event_queue, task, query)

    async def _should_start_async_processing(self, context_id: str, query: str) -> bool:
        """
        Determine if the current query should trigger async processing.

        Returns True only when user explicitly confirms processing after all inputs are collected.
        """
        try:
            # Get conversation state
            conversation_state = await self.agent._get_conversation_state(context_id)

            # Only trigger async processing when user explicitly confirms processing
            # AND we have all required inputs (ready_to_process phase)
            if conversation_state.phase == ConversationPhase.READY_TO_PROCESS:
                # Check if user is confirming to start processing
                if any(word in query.lower() for word in ['yes', 'start', 'create', 'begin', 'proceed']):
                    logger.info(f"User confirmed processing in ready_to_process phase: {query}")
                    return True

            # Also trigger for refinement processing
            if conversation_state.phase == ConversationPhase.REFINEMENT_PROCESSING:
                return True

            return False

        except Exception as e:
            logger.error(f"Error checking async processing condition: {e}")
            return False

    async def _async_research_and_generate(self, query: str, context_id: str):
        """
        Async generator for research and plan generation with progress updates.
        This mimics the agent.stream interface but performs long-running research.
        """
        try:
            logger.info(f"Starting async research and generation for context: {context_id}")

            # Get conversation state
            conversation_state = await self.agent._get_conversation_state(context_id)

            # Step 1: Research phase
            yield {
                'is_task_complete': False,
                'require_user_input': False,
                'content': '🔍 Collecting latest interview resources and trends...',
            }

            # Simulate research delay and perform actual research
            await asyncio.sleep(10)  # 10 seconds for domain research

            research_data = {}
            if conversation_state.user_inputs.domains:
                research_data = await self.search_manager.comprehensive_research(
                    domains=conversation_state.user_inputs.domains,
                    skill_level=conversation_state.user_inputs.skill_level or 'intermediate',
                    companies=conversation_state.user_inputs.specific_companies
                )

            # Step 2: Plan generation
            yield {
                'is_task_complete': False,
                'require_user_input': False,
                'content': '📋 Generating personalized study plan based on your preferences...',
            }

            await asyncio.sleep(15)  # 15 seconds for plan generation

            # Generate the preparation plan
            if research_data.get('success'):
                plan_content = await self.agent.create_preparation_plan(
                    conversation_state.user_inputs,
                    research_data.get('research_data', {})
                )
            else:
                plan_content = await self._generate_fallback_plan(conversation_state.user_inputs)

            # Step 3: Finalization
            yield {
                'is_task_complete': False,
                'require_user_input': False,
                'content': '✨ Finalizing your interview preparation roadmap...',
            }

            await asyncio.sleep(5)  # 5 seconds for finalization

            # Update conversation state with the plan
            conversation_state.plan_content = plan_content
            conversation_state.plan_generated = True
            conversation_state.advance_phase(ConversationPhase.PLAN_DELIVERED)
            await self.agent._save_conversation_state(context_id, conversation_state)

            # Final response with complete plan and satisfaction check
            yield {
                'is_task_complete': False,  # Not fully complete yet - need satisfaction check
                'require_user_input': True,  # Need user feedback
                'content': f"""🎉 **Your Interview Preparation Plan is Ready!**

{plan_content}

---

**Are you satisfied with this preparation plan, or would you like me to make any adjustments?**

You can say:
- **"I'm satisfied"** or **"This looks perfect!"** to complete
- **"I want to adjust..."** to request specific changes

What would you like to do next?"""
            }

        except Exception as e:
            logger.error(f"Error in async research and generation: {e}")
            yield {
                'is_task_complete': False,
                'require_user_input': True,
                'content': f"I encountered an error while creating your plan: {str(e)}. Please try again."
            }

    async def _generate_preparation_plan(self, context_id: str) -> str:
        """Generate preparation plan synchronously for standard processing."""
        try:
            conversation_state = await self.agent._get_conversation_state(context_id)

            # Perform quick research
            research_data = {}
            if conversation_state.user_inputs.domains:
                research_data = await self.search_manager.comprehensive_research(
                    domains=conversation_state.user_inputs.domains,
                    skill_level=conversation_state.user_inputs.skill_level or 'intermediate'
                )

            # Generate plan
            if research_data.get('success'):
                plan_content = await self.agent.create_preparation_plan(
                    conversation_state.user_inputs,
                    research_data.get('research_data', {})
                )
            else:
                plan_content = await self._generate_fallback_plan(conversation_state.user_inputs)

            return plan_content

        except Exception as e:
            logger.error(f"Error generating preparation plan: {e}")
            return f"I encountered an error creating your plan: {str(e)}. Please try again."

    async def _generate_refined_plan(self, context_id: str) -> str:
        """Generate refined preparation plan based on user feedback."""
        try:
            conversation_state = await self.agent._get_conversation_state(context_id)

            # Get the refinement requests
            refinements = conversation_state.refinement_requests
            original_plan = conversation_state.plan_content or "Previous plan"

            # Perform research based on refinements
            research_data = {}
            if conversation_state.user_inputs.domains:
                research_data = await self.search_manager.comprehensive_research(
                    domains=conversation_state.user_inputs.domains,
                    skill_level=conversation_state.user_inputs.skill_level or 'intermediate'
                )

            # Generate refined plan incorporating feedback
            refinement_summary = "\n".join([f"- {req}" for req in refinements])

            refined_plan = f"""# 🎯 Your Refined Interview Preparation Plan

## 📝 Refinements Applied
Based on your feedback:
{refinement_summary}

## 📋 Updated Overview
- **Domains:** {", ".join([d.replace('_', ' ').title() for d in conversation_state.user_inputs.domains])}
- **Skill Level:** {conversation_state.user_inputs.skill_level.replace('_', ' ').title() if conversation_state.user_inputs.skill_level else 'Intermediate'}
- **Learning Style:** {conversation_state.user_inputs.preference.replace('_', ' ').title() if conversation_state.user_inputs.preference else 'Balanced'}

## 🔄 Adjusted 12-Week Preparation Schedule

### Weeks 1-3: Foundation Building (Updated)
"""

            # Add domain-specific content based on research and refinements
            for domain in conversation_state.user_inputs.domains:
                refined_plan += f"\n#### {domain.replace('_', ' ').title()}\n"

                if domain in research_data.get('domains', {}):
                    domain_data = research_data['domains'][domain]

                    # Add learning resources
                    if domain_data.get('learning_resources', {}).get('success'):
                        resources = domain_data['learning_resources']['results'][:3]
                        for resource in resources:
                            refined_plan += f"- 📚 {resource.get('title', 'Resource')}\n"

            refined_plan += f"""

### Weeks 4-6: Enhanced Skill Development
- Targeted practice based on your feedback
- Advanced problem-solving techniques
- Specialized resources and materials

### Weeks 7-9: Advanced Topics (Refined)
- Complex scenarios and case studies
- Industry-specific preparation
- Mock interview intensification

### Weeks 10-12: Final Preparation (Customized)
- Personalized company preparation
- Last-minute optimization
- Confidence building exercises

## 🎯 Refinement-Specific Recommendations

{self._generate_refinement_recommendations(refinements)}

## 📈 Updated Progress Tracking
- Weekly milestone checks
- Refined success metrics
- Adjusted timeline based on feedback

---
*This refined plan incorporates your specific feedback and preferences. Further adjustments can be made as needed.*
"""

            # Store the refined plan
            conversation_state.plan_content = refined_plan
            await self.agent._save_conversation_state(context_id, conversation_state)

            return refined_plan

        except Exception as e:
            logger.error(f"Error generating refined plan: {e}")
            return f"I encountered an error refining your plan: {str(e)}. Please try again."

    def _generate_refinement_recommendations(self, refinements: List[str]) -> str:
        """Generate specific recommendations based on refinement requests."""
        recommendations = []

        for refinement in refinements:
            refinement_lower = refinement.lower()

            if 'system design' in refinement_lower:
                recommendations.append("- 🏗️ Additional system design practice with real-world scenarios")
            elif 'algorithm' in refinement_lower:
                recommendations.append("- 🧮 Enhanced algorithm problem-solving with complexity analysis")
            elif 'timeline' in refinement_lower or 'time' in refinement_lower:
                recommendations.append("- ⏰ Adjusted timeline to better fit your schedule")
            elif 'company' in refinement_lower:
                recommendations.append("- 🏢 Company-specific interview preparation and insights")
            else:
                recommendations.append(f"- 🔧 Customization based on: {refinement}")

        return "\n".join(recommendations) if recommendations else "- 🎯 General optimizations applied"

    async def _generate_fallback_plan(self, user_inputs: UserInputs) -> str:
        """Generate a fallback plan when web search fails."""
        domains_str = ", ".join([d.replace('_', ' ').title() for d in user_inputs.domains])

        return f"""# 🎯 Your Interview Preparation Plan

## 📋 Overview
- **Domains:** {domains_str}
- **Skill Level:** {user_inputs.skill_level.replace('_', ' ').title() if user_inputs.skill_level else 'Intermediate'}
- **Learning Style:** {user_inputs.preference.replace('_', ' ').title() if user_inputs.preference else 'Balanced'}

## 📅 8-Week Preparation Schedule

### Weeks 1-2: Foundation Building
- Review fundamental concepts
- Set up practice environment
- Begin daily coding practice

### Weeks 3-4: Core Skills Development
- Focus on key algorithms and data structures
- Practice system design basics
- Mock interview sessions

### Weeks 5-6: Advanced Topics
- Complex problem solving
- In-depth system design
- Domain-specific deep dives

### Weeks 7-8: Final Preparation
- Company research and preparation
- Final mock interviews
- Review and polish

## 🔗 Essential Resources
- **LeetCode** - Algorithm practice
- **System Design Primer** - Architecture concepts
- **Pramp** - Mock interviews
- **InterviewBit** - Comprehensive prep

## 💡 Daily Schedule Recommendation
- 1 hour: Coding practice
- 30 minutes: System design study
- 30 minutes: Domain-specific learning

*Note: This is a basic plan. For a more detailed, research-backed plan, please ensure web search is enabled.*"""

    def _validate_request(self, context: RequestContext) -> bool:
        """Validate the incoming request."""
        # For interview prep, we accept all text-based requests
        return False

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Cancel the current operation."""
        raise ServerError(error=UnsupportedOperationError())
