import os
import asyncio
import logging
from typing import Dict, Any, AsyncIterable, List, Optional
from collections.abc import AsyncIterable as AsyncIterableType
from pydantic import BaseModel
from typing import Annotated, Literal
import operator

from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from .conversation_state import (
    ConversationState,
    ConversationPhase,
    InterviewPrepResponse,
    ResponseStatus,
    InterviewDomain,
    SkillLevel,
    PrepPreference,
    UserInputs
)
from .web_search_tools import (
    search_interview_resources,
    search_company_interview_info,
    search_learning_resources,
    search_leetcode_problems,
    search_youtube_channels,
    search_current_interview_guides,
    WebSearchManager
)

logger = logging.getLogger(__name__)


class InterviewPrepResponseFormat(BaseModel):
    """Response format for the interview prep agent."""
    status: Literal['input_required', 'processing', 'completed', 'error'] = 'input_required'
    message: str
    phase: str = 'initial'
    collected_inputs: Optional[Dict[str, Any]] = None
    trigger_push_notification: bool = False


class InterviewPrepState(TypedDict):
    """State schema for the interview prep agent."""
    messages: Annotated[list, add_messages]
    conversation_state: Dict[str, Any]
    structured_response: InterviewPrepResponseFormat


class InterviewPrepAgent:
    """
    Multi-turn Interview Preparation Agent with Google Gemini and web search capabilities.
    """

    SYSTEM_INSTRUCTION = """
You are an expert Interview Preparation Coach. Your role is to guide users through a comprehensive interview preparation process.

Your responsibilities:
1. Gather user requirements through multi-turn conversation
2. Research relevant resources using web search tools
3. Create personalized interview preparation plans
4. Provide domain-specific guidance and resources

Conversation Flow:
1. Initial: Understand user's interview preparation goals
2. Domain Selection: Help choose interview domains (algorithms, system design, databases, ML, etc.)
3. Level Assessment: Determine user's current skill level
4. Preference Gathering: Understand learning preferences and constraints
5. Processing: Research and create personalized preparation plan
6. Interactive Processing: Refine plan based on additional user input

Always be encouraging, professional, and provide actionable advice. Use web search tools to find current, relevant resources.
"""

    FORMAT_INSTRUCTION = """
Respond using the InterviewPrepResponse format:
- Set status to 'input_required' when you need more information from the user
- Set status to 'processing' when performing research or plan generation
- Set status to 'completed' when the preparation plan is finalized
- Set status to 'error' if there are any errors

Include the current conversation phase and any collected inputs.
For processing status, indicate if web search is required and provide search queries.
"""

    def __init__(self):
        """Initialize the Interview Preparation Agent."""
        # Initialize Google Gemini model
        model_source = os.getenv('MODEL_SOURCE', 'google')
        if model_source == 'google':
            self.model = ChatGoogleGenerativeAI(
                model='gemini-2.0-flash',
                temperature=0.3
            )
        else:
            raise ValueError("Only Google Gemini is supported in this implementation")

        # Initialize web search manager
        self.search_manager = WebSearchManager()

        # Setup tools
        self.tools = [
            search_interview_resources,
            search_company_interview_info,
            search_learning_resources,
            search_leetcode_problems,
            search_youtube_channels,
            search_current_interview_guides
        ]

        # Initialize memory for conversation persistence
        self.memory = MemorySaver()

        # Simple in-memory state store (for demo purposes)
        self.conversation_states: Dict[str, ConversationState] = {}

        # Create the LangGraph agent
        self.graph = create_react_agent(
            self.model,
            tools=self.tools,
            checkpointer=self.memory,
            prompt=f"{self.SYSTEM_INSTRUCTION}\n\n{self.FORMAT_INSTRUCTION}"
        )

        logger.info("InterviewPrepAgent initialized with Google Gemini and web search tools")

    async def stream(self, query: str, context_id: str) -> AsyncIterableType[Dict[str, Any]]:
        """
        Process user input and stream responses.

        Args:
            query: User's input message
            context_id: Conversation context identifier

        Yields:
            Dictionary with response information for A2A protocol
        """
        try:
            # Get current conversation state
            conversation_state = await self._get_conversation_state(context_id)

            # Add user message to history
            conversation_state.add_message("user", query)

            # Process based on current phase
            if conversation_state.phase == ConversationPhase.INITIAL:
                async for item in self._handle_initial_phase(query, conversation_state, context_id):
                    yield item

            elif conversation_state.phase == ConversationPhase.DOMAIN_SELECTION:
                async for item in self._handle_domain_selection(query, conversation_state, context_id):
                    yield item

            elif conversation_state.phase == ConversationPhase.LEVEL_ASSESSMENT:
                async for item in self._handle_level_assessment(query, conversation_state, context_id):
                    yield item

            elif conversation_state.phase == ConversationPhase.PREFERENCE_GATHERING:
                async for item in self._handle_preference_gathering(query, conversation_state, context_id):
                    yield item

            elif conversation_state.phase == ConversationPhase.READY_TO_PROCESS:
                async for item in self._handle_ready_to_process(query, conversation_state, context_id):
                    yield item

            elif conversation_state.phase == ConversationPhase.ASYNC_PROCESSING:
                async for item in self._handle_async_processing(query, conversation_state, context_id):
                    yield item

            elif conversation_state.phase == ConversationPhase.PLAN_DELIVERED:
                async for item in self._handle_plan_delivered(query, conversation_state, context_id):
                    yield item

            elif conversation_state.phase == ConversationPhase.REFINEMENT_INPUT:
                async for item in self._handle_refinement_input(query, conversation_state, context_id):
                    yield item

            elif conversation_state.phase == ConversationPhase.REFINEMENT_PROCESSING:
                async for item in self._handle_refinement_processing(query, conversation_state, context_id):
                    yield item

            elif conversation_state.phase == ConversationPhase.COMPLETED:
                async for item in self._handle_completed_phase(query, conversation_state, context_id):
                    yield item

            # Save updated state
            await self._save_conversation_state(context_id, conversation_state)

        except Exception as e:
            logger.error(f"Error in stream method: {e}")
            yield {
                'is_task_complete': False,
                'require_user_input': True,
                'content': f"I encountered an error: {str(e)}. Please try again or rephrase your request.",
                'phase': 'error'
            }

    async def _get_conversation_state(self, context_id: str) -> ConversationState:
        """Retrieve or create conversation state for the given context."""
        if context_id in self.conversation_states:
            existing_state = self.conversation_states[context_id]
            logger.info(f"Retrieved existing state for {context_id}: phase={existing_state.phase}, domains={existing_state.user_inputs.domains}")
            return existing_state
        else:
            logger.info(f"Creating new state for {context_id}")
            new_state = ConversationState()
            self.conversation_states[context_id] = new_state
            return new_state

    async def _save_conversation_state(self, context_id: str, state: ConversationState) -> None:
        """Save conversation state."""
        self.conversation_states[context_id] = state
        logger.info(f"Saved state for {context_id}: phase={state.phase}")

    async def _handle_initial_phase(
        self,
        query: str,
        state: ConversationState,
        context_id: str
    ) -> AsyncIterableType[Dict[str, Any]]:
        """Handle the initial phase of conversation."""
        logger.info("Handling initial phase")

        # Check if user is asking for interview preparation
        if any(keyword in query.lower() for keyword in ['interview', 'prepare', 'job', 'coding']):
            state.advance_phase(ConversationPhase.DOMAIN_SELECTION)
            state.add_message("agent", "Great! I'll help you prepare for interviews.")

            yield {
                'is_task_complete': False,
                'require_user_input': True,
                'content': """Great! I'm here to help you prepare for interviews. Let me gather some information to create a personalized preparation plan.

First, which interview domains would you like to focus on? You can choose multiple:

- **Algorithms** - Data structures, algorithms, coding problems
- **System Design** - Scalable system architecture, distributed systems
- **Databases** - SQL, NoSQL, database design
- **Machine Learning** - ML algorithms, data science concepts
- **Behavioral** - Soft skills, situational questions
- **Frontend** - JavaScript, React, UI/UX
- **Backend** - APIs, microservices, server architecture

Please tell me which domains interest you, or type "all" if you want comprehensive preparation.""",
                'phase': ConversationPhase.DOMAIN_SELECTION
            }
        else:
            yield {
                'is_task_complete': False,
                'require_user_input': True,
                'content': """Hello! I'm your Interview Preparation Coach. I help professionals prepare for technical interviews with personalized study plans and resources.

I can help you with:
- Algorithm and coding interview prep
- System design interview guidance
- Domain-specific preparation (ML, databases, etc.)
- Company-specific interview insights
- Personalized study schedules

Would you like to start preparing for interviews? Just say "I want to prepare for interviews" or tell me about your specific goals!""",
                'phase': ConversationPhase.INITIAL
            }

    async def _handle_domain_selection(
        self,
        query: str,
        state: ConversationState,
        context_id: str
    ) -> AsyncIterableType[Dict[str, Any]]:
        """Handle domain selection phase."""
        logger.info("Handling domain selection phase")
        logger.info(f"User query: {query}")

        # Parse domains from user input
        domains = self._parse_domains(query)
        logger.info(f"Parsed domains: {domains}")

        if domains:
            state.user_inputs.domains = domains
            logger.info(f"Set state.user_inputs.domains to: {state.user_inputs.domains}")
            state.advance_phase(ConversationPhase.LEVEL_ASSESSMENT)
            state.add_message("agent", f"Selected domains: {', '.join(domains)}")

            domain_list = ", ".join([domain.replace('_', ' ').title() for domain in domains])

            yield {
                'is_task_complete': False,
                'require_user_input': True,
                'content': f"""Perfect! You've selected: **{domain_list}**

Now, what's your current skill level in these areas?

- **Beginner** - New to the field, learning fundamentals
- **Intermediate** - Some experience, comfortable with basics
- **Advanced** - Experienced, looking to master complex topics

Please tell me your skill level.""",
                'phase': ConversationPhase.LEVEL_ASSESSMENT
            }
        else:
            yield {
                'is_task_complete': False,
                'require_user_input': True,
                'content': """I didn't quite catch which domains you'd like to focus on. Please choose from:

- **Algorithms** (or "algo")
- **System Design** (or "systems")
- **Databases** (or "db")
- **Machine Learning** (or "ml")
- **Behavioral**
- **Frontend**
- **Backend**

You can say something like "I want to focus on algorithms and system design" or just "algorithms, databases".""",
                'phase': ConversationPhase.DOMAIN_SELECTION
            }

    def _parse_domains(self, query: str) -> List[str]:
        """Parse interview domains from user input."""
        # Extract only the last user message (split by newlines and take the last non-empty line)
        lines = [line.strip() for line in query.strip().split('\n') if line.strip()]
        user_input = lines[-1] if lines else query

        query_lower = user_input.lower()
        domains = []

        logger.info(f"_parse_domains called with full query length: {len(query)} chars")
        logger.info(f"Extracted user input: '{user_input}'")
        logger.info(f"query_lower: '{query_lower}'")

        # Check for "all" as a standalone word (not part of other text)
        import re
        if re.search(r'\ball\b', query_lower) or 'everything' in query_lower:
            logger.info("Matched 'all' condition, returning all domains")
            return ['algorithms', 'system_design', 'databases', 'machine_learning', 'behavioral', 'frontend', 'backend']

        # Domain keywords with priorities (check longer/more specific phrases first)
        domain_keywords = {
            'algorithms': ['algorithm', 'algo', 'dsa', 'leetcode'],
            'system_design': ['system design', 'systems design', 'system architecture', 'distributed system'],
            'databases': ['database', 'db', 'sql'],
            'machine_learning': ['machine learning', 'ml', 'data science'],
            'behavioral': ['behavioral', 'behavior', 'soft skill'],
            'frontend': ['frontend', 'front-end', 'react', 'javascript', 'ui/ux'],
            'backend': ['backend', 'back-end', 'server', 'microservice']
        }

        # Check each domain
        for domain, keywords in domain_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    logger.info(f"Matched domain '{domain}' with keyword '{keyword}'")
                    if domain not in domains:
                        domains.append(domain)
                    break  # Found a match, move to next domain

        logger.info(f"Returning domains: {domains}")
        return domains

    async def _handle_level_assessment(
        self,
        query: str,
        state: ConversationState,
        context_id: str
    ) -> AsyncIterableType[Dict[str, Any]]:
        """Handle skill level assessment phase."""
        logger.info("Handling level assessment phase")

        level = self._parse_skill_level(query)

        if level:
            state.user_inputs.skill_level = level
            state.advance_phase(ConversationPhase.PREFERENCE_GATHERING)
            state.add_message("agent", f"Skill level: {level}")

            yield {
                'is_task_complete': False,
                'require_user_input': True,
                'content': f"""Great! I've noted your skill level as **{level.replace('_', ' ').title()}**.

Now, what's your learning preference?

- **Theory-Heavy** - Focus on concepts, principles, and understanding
- **Coding-Heavy** - Emphasis on practice problems and hands-on coding
- **Balanced** - Mix of theory and practical exercises
- **Project-Based** - Learn through building real projects

What approach works best for you?""",
                'phase': ConversationPhase.PREFERENCE_GATHERING
            }
        else:
            yield {
                'is_task_complete': False,
                'require_user_input': True,
                'content': """Please let me know your skill level:

- **Beginner** - New to these topics
- **Intermediate** - Some experience
- **Advanced** - Experienced professional

Just say "beginner", "intermediate", or "advanced".""",
                'phase': ConversationPhase.LEVEL_ASSESSMENT
            }

    def _parse_skill_level(self, query: str) -> Optional[str]:
        """Parse skill level from user input."""
        # Extract only the last user message
        lines = [line.strip() for line in query.strip().split('\n') if line.strip()]
        user_input = lines[-1] if lines else query
        query_lower = user_input.lower()

        if any(word in query_lower for word in ['beginner', 'new', 'start']):
            return 'beginner'
        elif any(word in query_lower for word in ['intermediate', 'some experience', 'mid']):
            return 'intermediate'
        elif any(word in query_lower for word in ['advanced', 'expert', 'experienced']):
            return 'advanced'

        return None

    async def _handle_preference_gathering(
        self,
        query: str,
        state: ConversationState,
        context_id: str
    ) -> AsyncIterableType[Dict[str, Any]]:
        """Handle learning preference gathering phase."""
        logger.info("Handling preference gathering phase")

        preference = self._parse_preference(query)

        if preference:
            state.user_inputs.preference = preference
            state.add_message("agent", f"Preference: {preference}")

            # Check if we have all required inputs
            if state.is_input_complete():
                state.advance_phase(ConversationPhase.READY_TO_PROCESS)
                state.awaiting_processing_confirmation = True

                # Show summary and ask for confirmation
                domains_str = ", ".join([d.replace('_', ' ').title() for d in state.user_inputs.domains])

                yield {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': f"""Perfect! Here's what I've gathered:

**Domains:** {domains_str}
**Skill Level:** {state.user_inputs.skill_level.replace('_', ' ').title()}
**Learning Style:** {preference.replace('_', ' ').title()}

I'm ready to create your personalized interview preparation plan. This will involve:
- Researching latest interview trends and resources
- Creating a customized study schedule
- Finding domain-specific practice materials
- Generating a comprehensive roadmap

**This process takes 2-3 minutes. Would you like me to start creating your plan?**

Reply with **"Yes, create my plan"** to begin processing.""",
                    'phase': ConversationPhase.READY_TO_PROCESS
                }
            else:
                # Ask for any missing information
                yield {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': "I still need some more information. Could you please provide your learning preference?",
                    'phase': ConversationPhase.PREFERENCE_GATHERING
                }
        else:
            yield {
                'is_task_complete': False,
                'require_user_input': True,
                'content': """Please choose your learning preference:

- **Theory-Heavy** - Focus on concepts and understanding
- **Coding-Heavy** - Emphasis on practice and coding
- **Balanced** - Mix of theory and practice
- **Project-Based** - Learn through building projects

Just say something like "I prefer coding-heavy" or "balanced approach".""",
                'phase': ConversationPhase.PREFERENCE_GATHERING
            }

    def _parse_preference(self, query: str) -> Optional[str]:
        """Parse learning preference from user input."""
        # Extract only the last user message
        lines = [line.strip() for line in query.strip().split('\n') if line.strip()]
        user_input = lines[-1] if lines else query
        query_lower = user_input.lower()

        if any(word in query_lower for word in ['theory', 'concept', 'understanding']):
            return 'theory_heavy'
        elif any(word in query_lower for word in ['coding', 'practice', 'hands-on']):
            return 'coding_heavy'
        elif any(word in query_lower for word in ['balanced', 'mix', 'both']):
            return 'balanced'
        elif any(word in query_lower for word in ['project', 'build', 'real']):
            return 'project_based'

        return None

    async def _handle_ready_to_process(
        self,
        query: str,
        state: ConversationState,
        context_id: str
    ) -> AsyncIterableType[Dict[str, Any]]:
        """Handle processing confirmation and trigger push notifications."""
        logger.info("Handling ready to process phase")

        if any(word in query.lower() for word in ['yes', 'start', 'create', 'begin', 'proceed']):
            # Don't advance phase here - let the executor handle async processing
            state.awaiting_processing_confirmation = False

            yield {
                'is_task_complete': False,
                'require_user_input': False,
                'content': 'Processing your interview preparation request...',
                'phase': ConversationPhase.READY_TO_PROCESS,
                'trigger_async_processing': True  # Signal for executor to start async processing
            }
        else:
            yield {
                'is_task_complete': False,
                'require_user_input': True,
                'content': 'Please confirm if you want me to create your preparation plan by saying **"Yes, create my plan"**.',
                'phase': ConversationPhase.READY_TO_PROCESS
            }

    async def _handle_async_processing(
        self,
        query: str,
        state: ConversationState,
        context_id: str
    ) -> AsyncIterableType[Dict[str, Any]]:
        """Handle the async processing phase - should not be called directly."""
        logger.info("Handling async processing phase")

        yield {
            'is_task_complete': False,
            'require_user_input': False,
            'content': "I'm currently processing your request. Please wait for the results via push notification.",
            'phase': ConversationPhase.ASYNC_PROCESSING
        }

    async def _handle_plan_delivered(
        self,
        query: str,
        state: ConversationState,
        context_id: str
    ) -> AsyncIterableType[Dict[str, Any]]:
        """Handle satisfaction check after plan delivery."""
        logger.info("Handling plan delivered phase")

        if any(word in query.lower() for word in ['satisfied', 'good', 'perfect', 'thanks', 'done', 'complete']):
            state.advance_phase(ConversationPhase.COMPLETED)
            state.satisfaction_confirmed = True

            yield {
                'is_task_complete': True,
                'require_user_input': False,
                'content': """Excellent! Your interview preparation plan is complete.

**Next Steps:**
1. Save your preparation plan for reference
2. Start with Week 1 activities
3. Track your progress regularly
4. Come back anytime for plan updates

Good luck with your interview preparation!""",
                'phase': ConversationPhase.COMPLETED
            }

        elif any(word in query.lower() for word in ['adjust', 'change', 'modify', 'refine', 'update', 'improve']):
            state.advance_phase(ConversationPhase.REFINEMENT_INPUT)

            yield {
                'is_task_complete': False,
                'require_user_input': True,
                'content': """I'd be happy to refine your preparation plan!

What would you like me to adjust? For example:
- Add more focus on specific domains
- Change the timeline or intensity
- Include specific companies or roles
- Modify learning resources or style
- Any other specific requirements

Please describe what you'd like me to change.""",
                'phase': ConversationPhase.REFINEMENT_INPUT
            }
        else:
            yield {
                'is_task_complete': False,
                'require_user_input': True,
                'content': """Are you satisfied with your preparation plan, or would you like me to make any adjustments?

You can say:
- **"I'm satisfied"** or **"This looks good"** to complete
- **"I want to adjust..."** to request changes""",
                'phase': ConversationPhase.PLAN_DELIVERED
            }

    async def _handle_refinement_input(
        self,
        query: str,
        state: ConversationState,
        context_id: str
    ) -> AsyncIterableType[Dict[str, Any]]:
        """Handle refinement input gathering."""
        logger.info("Handling refinement input phase")

        # Store the refinement request
        state.refinement_requests.append(query)
        state.advance_phase(ConversationPhase.REFINEMENT_PROCESSING)

        yield {
            'is_task_complete': False,
            'require_user_input': False,
            'content': 'I understand your refinement request. Processing your adjustments...',
            'phase': ConversationPhase.REFINEMENT_PROCESSING,
            'trigger_refinement_processing': True  # Signal for refinement processing
        }

    async def _handle_refinement_processing(
        self,
        query: str,
        state: ConversationState,
        context_id: str
    ) -> AsyncIterableType[Dict[str, Any]]:
        """Handle refinement processing phase."""
        logger.info("Handling refinement processing phase")

        yield {
            'is_task_complete': False,
            'require_user_input': False,
            'content': "I'm processing your refinement request. Please wait for the updated plan.",
            'phase': ConversationPhase.REFINEMENT_PROCESSING
        }


    async def _handle_completed_phase(
        self,
        query: str,
        state: ConversationState,
        context_id: str
    ) -> AsyncIterableType[Dict[str, Any]]:
        """Handle completed phase."""
        logger.info("Handling completed phase")

        yield {
            'is_task_complete': True,
            'require_user_input': False,
            'content': "Your preparation plan has been completed! Is there anything specific you'd like me to adjust or explain further?",
            'phase': ConversationPhase.COMPLETED
        }

    async def create_preparation_plan(
        self,
        user_inputs: UserInputs,
        research_data: Dict[str, Any]
    ) -> str:
        """Create a comprehensive preparation plan based on user inputs and research."""
        try:
            domains_str = ", ".join([d.replace('_', ' ').title() for d in user_inputs.domains])

            # Determine preparation timeline based on skill level
            timeline_weeks = 8 if user_inputs.skill_level == 'advanced' else 12 if user_inputs.skill_level == 'intermediate' else 16

            plan = f"""🎉 **Your Interview Preparation Plan is Ready!**

🎯 **Your Interview Preparation Plan**
📋 **Overview**
- **Domains:** {domains_str}
- **Skill Level:** {user_inputs.skill_level.replace('_', ' ').title()}
- **Learning Style:** {user_inputs.preference.replace('_', ' ').title()}

📅 **{timeline_weeks}-Week Preparation Schedule**

"""

            # Timeline breakdown
            foundation_weeks = timeline_weeks // 4
            skill_weeks = timeline_weeks // 4
            advanced_weeks = timeline_weeks // 4
            final_weeks = timeline_weeks - (foundation_weeks + skill_weeks + advanced_weeks)

            plan += f"""**Weeks 1-{foundation_weeks}: Foundation Building**
- Review fundamental concepts
- Set up practice environment
- Begin daily coding practice

**Weeks {foundation_weeks+1}-{foundation_weeks+skill_weeks}: Core Skills Development**
- Focus on key algorithms and data structures
- Practice system design basics
- Mock interview sessions

**Weeks {foundation_weeks+skill_weeks+1}-{foundation_weeks+skill_weeks+advanced_weeks}: Advanced Topics**
- Complex problem solving
- In-depth system design
- Domain-specific deep dives

**Weeks {foundation_weeks+skill_weeks+advanced_weeks+1}-{timeline_weeks}: Final Preparation**
- Company research and preparation
- Final mock interviews
- Review and polish

🔗 **Essential Resources**

"""

            # Add researched resources from web search
            resource_count = 0

            # Add domain-specific resources from research
            for domain in user_inputs.domains:
                domain_title = domain.replace('_', ' ').title()
                plan += f"**{domain_title} Resources:**\n"

                if domain in research_data.get('domains', {}):
                    domain_data = research_data['domains'][domain]

                    # Add current interview guides
                    if domain_data.get('current_guides', {}).get('success'):
                        current_guides = domain_data['current_guides']['results'][:2]
                        for guide in current_guides:
                            if guide.get('url') and guide.get('title'):
                                plan += f"- [📖 {guide['title'][:50]}...]({guide['url']})\n"
                                resource_count += 1

                    # Add interview preparation resources
                    if domain_data.get('interview_info', {}).get('success'):
                        interview_resources = domain_data['interview_info']['results'][:2]
                        for resource in interview_resources:
                            if resource.get('url') and resource.get('title'):
                                plan += f"- [📚 {resource['title'][:50]}...]({resource['url']})\n"
                                resource_count += 1

                    # Add YouTube resources
                    if domain_data.get('youtube_resources', {}).get('success'):
                        youtube_resources = domain_data['youtube_resources']['results'][:2]
                        for resource in youtube_resources:
                            if resource.get('url') and resource.get('title'):
                                plan += f"- [🎥 {resource['title'][:50]}...]({resource['url']})\n"
                                resource_count += 1

                    # Add LeetCode problems for algorithms domain
                    if domain == 'algorithms' and domain_data.get('leetcode_problems', {}).get('success'):
                        leetcode_problems = domain_data['leetcode_problems']['results'][:3]
                        for problem in leetcode_problems:
                            if problem.get('url') and problem.get('title'):
                                plan += f"- [💻 {problem['title'][:50]}...]({problem['url']})\n"
                                resource_count += 1

                plan += "\n"

            # Add general practice platforms
            plan += """**Popular Practice Platforms:**
- LeetCode - Algorithm practice
- System Design Primer - Architecture concepts
- Pramp - Mock interviews
- InterviewBit - Comprehensive prep

"""

            # Add daily schedule recommendation
            if user_inputs.preference == 'coding_heavy':
                plan += """💡 **Daily Schedule Recommendation**
- 1.5 hours: Coding practice
- 30 minutes: System design study
- 30 minutes: Domain-specific learning
"""
            elif user_inputs.preference == 'theory_heavy':
                plan += """💡 **Daily Schedule Recommendation**
- 1 hour: Theory and concept study
- 45 minutes: System design reading
- 45 minutes: Coding practice
"""
            elif user_inputs.preference == 'project_based':
                plan += """💡 **Daily Schedule Recommendation**
- 1 hour: Project development
- 30 minutes: Code review and optimization
- 1 hour: Related theory study
"""
            else:  # balanced
                plan += """💡 **Daily Schedule Recommendation**
- 1 hour: Coding practice
- 30 minutes: System design study
- 30 minutes: Domain-specific learning
"""

            # Add note about web search
            if resource_count > 0:
                plan += f"\n**Note:** This plan includes {resource_count} current resources found through web search.\n"
            else:
                plan += "\n**Note:** This is a basic plan. For a more detailed, research-backed plan, please ensure web search is enabled.\n"

            plan += """
**Are you satisfied with this preparation plan, or would you like me to make any adjustments?**

You can say:
- **"I'm satisfied"** or **"This looks perfect!"** to complete
- **"I want to adjust..."** to request specific changes

What would you like to do next?"""

            return plan

        except Exception as e:
            logger.error(f"Error creating preparation plan: {e}")
            return f"I encountered an error creating your plan: {str(e)}. Please try again."

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']
