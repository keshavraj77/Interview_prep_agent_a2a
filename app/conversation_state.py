from enum import Enum
from typing import Dict, Any, Optional, List
from pydantic import BaseModel


class ConversationPhase(str, Enum):
    """Phases of the interview preparation conversation."""
    INITIAL = "initial"
    DOMAIN_SELECTION = "domain_selection"
    LEVEL_ASSESSMENT = "level_assessment"
    PREFERENCE_GATHERING = "preference_gathering"
    READY_TO_PROCESS = "ready_to_process"           # NEW: All inputs collected, awaiting confirmation
    PROCESSING_CONFIRMATION = "processing_confirmation"  # NEW: User confirms processing
    ASYNC_PROCESSING = "async_processing"           # NEW: Push notification phase
    PLAN_DELIVERED = "plan_delivered"               # NEW: Plan ready, check satisfaction
    REFINEMENT_INPUT = "refinement_input"           # NEW: Gather refinement requests
    REFINEMENT_PROCESSING = "refinement_processing" # NEW: Process refinements
    COMPLETED = "completed"                         # Final completion


class InterviewDomain(str, Enum):
    """Available interview domains."""
    ALGORITHMS = "algorithms"
    SYSTEM_DESIGN = "system_design"
    DATABASES = "databases"
    MACHINE_LEARNING = "machine_learning"
    BEHAVIORAL = "behavioral"
    FRONTEND = "frontend"
    BACKEND = "backend"


class SkillLevel(str, Enum):
    """User skill levels."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class PrepPreference(str, Enum):
    """Preparation style preferences."""
    THEORY_HEAVY = "theory_heavy"
    CODING_HEAVY = "coding_heavy"
    BALANCED = "balanced"
    PROJECT_BASED = "project_based"


class UserInputs(BaseModel):
    """Collected user inputs during conversation."""
    domains: List[InterviewDomain] = []
    skill_level: Optional[SkillLevel] = None
    preference: Optional[PrepPreference] = None
    target_role: Optional[str] = None
    timeline: Optional[str] = None
    specific_companies: List[str] = []
    additional_requirements: List[str] = []


class ConversationState(BaseModel):
    """State management for multi-turn conversation."""
    phase: ConversationPhase = ConversationPhase.INITIAL
    user_inputs: UserInputs = UserInputs()
    messages_history: List[Dict[str, Any]] = []
    processing_steps: List[str] = []
    current_processing_step: Optional[str] = None
    research_data: Dict[str, Any] = {}
    plan_generated: bool = False
    awaiting_processing_confirmation: bool = False  # NEW: Waiting for user to confirm processing
    plan_content: Optional[str] = None              # NEW: Store generated plan
    refinement_requests: List[str] = []             # NEW: Store user refinement requests
    satisfaction_confirmed: bool = False            # NEW: User satisfaction status

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        self.messages_history.append({
            "role": role,
            "content": content,
            "phase": self.phase
        })

    def advance_phase(self, new_phase: ConversationPhase) -> None:
        """Advance to the next conversation phase."""
        self.phase = new_phase

    def is_input_complete(self) -> bool:
        """Check if all required inputs have been collected."""
        return (
            len(self.user_inputs.domains) > 0 and
            self.user_inputs.skill_level is not None and
            self.user_inputs.preference is not None
        )

    def get_missing_inputs(self) -> List[str]:
        """Get list of missing required inputs."""
        missing = []
        if not self.user_inputs.domains:
            missing.append("interview domains")
        if not self.user_inputs.skill_level:
            missing.append("skill level")
        if not self.user_inputs.preference:
            missing.append("preparation preference")
        return missing

    def add_processing_step(self, step: str) -> None:
        """Add a processing step for progress tracking."""
        self.processing_steps.append(step)
        self.current_processing_step = step


class ResponseStatus(str, Enum):
    """Response status types for A2A protocol."""
    INPUT_REQUIRED = "input_required"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


class InterviewPrepResponse(BaseModel):
    """Structured response format for the interview prep agent."""
    status: ResponseStatus
    message: str
    phase: ConversationPhase
    collected_inputs: Optional[Dict[str, Any]] = None
    progress_info: Optional[Dict[str, Any]] = None
    next_question: Optional[str] = None
    requires_web_search: bool = False
    search_queries: List[str] = []