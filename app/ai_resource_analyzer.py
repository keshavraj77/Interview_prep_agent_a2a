"""
AI-Powered Resource Analysis and Personalization Module

This module provides intelligent analysis and ranking of web search results
using LLM capabilities to create personalized recommendations.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ResourceRecommendation(BaseModel):
    """Structured recommendation for a learning resource."""
    url: str
    title: str
    relevance_score: float = Field(ge=0.0, le=10.0, description="Relevance score from 0-10")
    why_recommended: str = Field(description="Personalized explanation of why this resource is recommended")
    difficulty_match: str = Field(description="How well it matches user's skill level")
    resource_type: str = Field(description="Type of resource: tutorial, guide, video, practice, etc.")


class UserIntent(BaseModel):
    """Parsed user intent from natural language."""
    domains: List[str] = Field(description="List of interview domains user wants to focus on")
    skill_level: Optional[str] = Field(description="User's skill level: beginner, intermediate, or advanced")
    learning_preference: Optional[str] = Field(description="Learning style: theory_heavy, coding_heavy, balanced, or project_based")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the parsing")
    reasoning: str = Field(description="Explanation of how the intent was determined")


class AIResourceAnalyzer:
    """
    AI-powered analyzer for intelligent resource curation and personalization.
    Uses LLM to understand user intent and rank/filter search results.
    """

    def __init__(self):
        """Initialize the AI Resource Analyzer with Gemini model."""
        self.model = ChatGoogleGenerativeAI(
            model='gemini-3-flash-preview',
            temperature=0.2  # Lower temperature for more consistent analysis
        )
        logger.info("AIResourceAnalyzer initialized with Gemini 3 Flash Preview")

    async def parse_user_intent(
        self,
        user_message: str,
        context: str = "domain_selection"
    ) -> UserIntent:
        """
        Use LLM with structured output to understand user intent from natural language.

        Args:
            user_message: The user's natural language input
            context: What we're trying to understand (domain_selection, skill_level, preference)

        Returns:
            Structured UserIntent with parsed information
        """
        system_prompt = """You are an expert at understanding user intent for interview preparation.
Parse the user's message and extract structured information.

Available domains (use EXACTLY these values):
- algorithms (includes: DSA, coding, leetcode, data structures, problem solving)
- system_design (includes: architecture, distributed systems, scalability, HLD, LLD)
- databases (includes: SQL, NoSQL, database design, data modeling)
- machine_learning (includes: ML, AI, deep learning, data science, NLP, CV)
- behavioral (includes: soft skills, leadership, communication, STAR method)
- frontend (includes: JavaScript, React, Vue, Angular, UI/UX, CSS, web development)
- backend (includes: APIs, microservices, server architecture, REST, GraphQL)

Skill levels (use EXACTLY these values or null if not mentioned):
- beginner: New to the field, learning fundamentals, junior, fresh grad
- intermediate: Some experience, comfortable with basics, mid-level
- advanced: Experienced, senior, staff, principal, expert level

Learning preferences (use EXACTLY these values or null if not mentioned):
- theory_heavy: Focus on concepts, reading, understanding principles
- coding_heavy: Emphasis on practice, hands-on coding, solving problems
- balanced: Mix of theory and practice
- project_based: Learn through building real projects

Be intelligent about synonyms and variations:
- "algo" or "DSA" → algorithms
- "systems" or "HLD" → system_design
- "ml" or "AI" → machine_learning
- "I'm new" or "just starting" → beginner
- "some experience" → intermediate
- "senior" or "expert" → advanced

If user says "all" or "everything", include all 7 domains.
If something is not mentioned, use null for that field."""

        user_prompt = f"""Context: {context}
User message: "{user_message}"

Extract the user's intent from this message."""

        try:
            # Use structured output to get UserIntent directly from the LLM
            structured_model = self.model.with_structured_output(UserIntent)

            result = await structured_model.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])

            logger.info(f"AI parsed intent: domains={result.domains}, skill={result.skill_level}, pref={result.learning_preference}")
            return result

        except Exception as e:
            logger.error(f"Error parsing user intent with structured output: {e}")
            # Fallback to simple keyword matching if AI fails
            return self._fallback_parse_intent(user_message)

    def _fallback_parse_intent(self, user_message: str) -> UserIntent:
        """Fallback keyword-based parsing if AI structured output fails."""
        user_lower = user_message.lower()

        domain_map = {
            'algorithms': ['algorithm', 'algo', 'dsa', 'leetcode', 'coding', 'data structure'],
            'system_design': ['system design', 'system', 'architecture', 'distributed', 'scalability'],
            'databases': ['database', 'db', 'sql', 'nosql'],
            'machine_learning': ['machine learning', 'ml', 'ai', 'data science'],
            'behavioral': ['behavioral', 'behavior', 'soft skill', 'leadership'],
            'frontend': ['frontend', 'front-end', 'react', 'javascript', 'ui', 'web'],
            'backend': ['backend', 'back-end', 'api', 'microservice', 'server']
        }

        domains = []
        if any(word in user_lower for word in ['all', 'everything', 'comprehensive']):
            domains = list(domain_map.keys())
        else:
            for domain, keywords in domain_map.items():
                if any(keyword in user_lower for keyword in keywords):
                    domains.append(domain)

        skill_level = None
        if any(word in user_lower for word in ['beginner', 'new', 'starting', 'learning']):
            skill_level = 'beginner'
        elif any(word in user_lower for word in ['intermediate', 'some experience', 'comfortable']):
            skill_level = 'intermediate'
        elif any(word in user_lower for word in ['advanced', 'experienced', 'expert', 'senior']):
            skill_level = 'advanced'

        learning_preference = None
        if any(word in user_lower for word in ['theory', 'concept', 'understanding', 'reading']):
            learning_preference = 'theory_heavy'
        elif any(word in user_lower for word in ['coding', 'practice', 'hands-on', 'doing']):
            learning_preference = 'coding_heavy'
        elif any(word in user_lower for word in ['balanced', 'mix', 'both']):
            learning_preference = 'balanced'
        elif any(word in user_lower for word in ['project', 'build', 'real', 'practical']):
            learning_preference = 'project_based'

        return UserIntent(
            domains=domains,
            skill_level=skill_level,
            learning_preference=learning_preference,
            confidence=0.5,  # Lower confidence for fallback
            reasoning="Parsed using fallback keyword matching"
        )

    async def rank_and_filter_resources(
        self,
        search_results: List[Dict[str, Any]],
        user_profile: Dict[str, Any],
        domain: str,
        max_results: int = 5
    ) -> List[ResourceRecommendation]:
        """
        Use AI to intelligently rank and filter search results based on user profile.
        
        Args:
            search_results: Raw search results from web search
            user_profile: User's skill level, preferences, and goals
            domain: The interview domain being researched
            max_results: Maximum number of recommendations to return
            
        Returns:
            List of personalized resource recommendations with explanations
        """
        if not search_results:
            return []
            
        system_prompt = f"""You are an expert interview preparation coach analyzing learning resources.

User Profile:
- Domain: {domain}
- Skill Level: {user_profile.get('skill_level', 'intermediate')}
- Learning Preference: {user_profile.get('preference', 'balanced')}
- Timeline: {user_profile.get('timeline', 'flexible')}

Your task is to:
1. Analyze each resource for relevance and quality
2. Rank resources by how well they match this specific user's needs
3. Provide personalized explanations for why each resource is recommended
4. Filter out low-quality or irrelevant resources

Consider:
- Does the difficulty match the user's skill level?
- Does the content type match their learning preference?
- Is it from a reputable source?
- Is it current and up-to-date?
- Does it provide practical value for interview preparation?"""

        # Prepare resource summaries for analysis
        resources_text = "\n\n".join([
            f"Resource {i+1}:\nTitle: {r.get('title', 'N/A')}\nURL: {r.get('url', 'N/A')}\nSnippet: {r.get('snippet', 'N/A')[:200]}"
            for i, r in enumerate(search_results[:15])  # Analyze top 15
        ])
        
        user_prompt = f"""Analyze these {len(search_results[:15])} resources and select the top {max_results} most relevant for this user.

Resources:
{resources_text}

For each of your top {max_results} recommendations, provide:
1. Resource number (1-{len(search_results[:15])})
2. Relevance score (0-10)
3. Why it's recommended for THIS specific user
4. How well it matches their skill level
5. Resource type (tutorial, guide, video, practice, etc.)

Format your response as:
RESOURCE: [number]
SCORE: [0-10]
WHY: [personalized explanation]
DIFFICULTY: [how it matches skill level]
TYPE: [resource type]

---

Provide exactly {max_results} recommendations."""

        try:
            response = await self.model.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            # Parse AI response into structured recommendations
            recommendations = []
            content = response.content

            # Handle case where content might be a list (Gemini format)
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and 'text' in item:
                        text_parts.append(item['text'])
                    elif isinstance(item, str):
                        text_parts.append(item)
                    else:
                        text_parts.append(str(item))
                content = ''.join(text_parts)

            # Simple parsing - in production, use structured output
            sections = content.split('---')
            
            for section in sections[:max_results]:
                try:
                    lines = section.strip().split('\n')
                    resource_num = None
                    score = 7.0
                    why = ""
                    difficulty = "Good match"
                    res_type = "guide"
                    
                    for line in lines:
                        if line.startswith('RESOURCE:'):
                            try:
                                resource_num = int(line.split(':')[1].strip()) - 1
                            except:
                                pass
                        elif line.startswith('SCORE:'):
                            try:
                                score = float(line.split(':')[1].strip())
                            except:
                                pass
                        elif line.startswith('WHY:'):
                            why = line.split(':', 1)[1].strip()
                        elif line.startswith('DIFFICULTY:'):
                            difficulty = line.split(':', 1)[1].strip()
                        elif line.startswith('TYPE:'):
                            res_type = line.split(':', 1)[1].strip()
                    
                    if resource_num is not None and 0 <= resource_num < len(search_results):
                        resource = search_results[resource_num]
                        recommendations.append(ResourceRecommendation(
                            url=resource.get('url', ''),
                            title=resource.get('title', 'Resource'),
                            relevance_score=score,
                            why_recommended=why or f"Relevant {domain} resource for {user_profile.get('skill_level', 'your')} level",
                            difficulty_match=difficulty,
                            resource_type=res_type
                        ))
                except Exception as parse_error:
                    logger.warning(f"Error parsing recommendation section: {parse_error}")
                    continue
            
            # If AI parsing failed, fall back to simple ranking
            if not recommendations and search_results:
                for i, resource in enumerate(search_results[:max_results]):
                    recommendations.append(ResourceRecommendation(
                        url=resource.get('url', ''),
                        title=resource.get('title', 'Resource'),
                        relevance_score=8.0 - (i * 0.5),
                        why_recommended=f"Relevant {domain} resource for interview preparation",
                        difficulty_match=f"Suitable for {user_profile.get('skill_level', 'intermediate')} level",
                        resource_type="guide"
                    ))
            
            return recommendations[:max_results]
            
        except Exception as e:
            logger.error(f"Error ranking resources: {e}")
            # Fallback: return top results with basic info
            return [
                ResourceRecommendation(
                    url=r.get('url', ''),
                    title=r.get('title', 'Resource'),
                    relevance_score=8.0,
                    why_recommended=f"Relevant {domain} resource",
                    difficulty_match="Good match",
                    resource_type="guide"
                )
                for r in search_results[:max_results]
            ]

    async def synthesize_personalized_plan(
        self,
        user_profile: Dict[str, Any],
        research_data: Dict[str, Any],
        ranked_resources: Dict[str, List[ResourceRecommendation]]
    ) -> str:
        """
        Use AI to synthesize a personalized study plan with explanations.
        
        Args:
            user_profile: User's complete profile
            research_data: Raw research data from web searches
            ranked_resources: AI-ranked resources per domain
            
        Returns:
            Personalized study plan with AI-generated insights
        """
        system_prompt = """You are an expert interview preparation coach creating personalized study plans.

Your task is to synthesize research findings into a cohesive, actionable plan that:
1. Explains WHY each resource is recommended for this specific user
2. Provides a realistic timeline based on their constraints
3. Offers strategic advice for their skill level
4. Includes motivational and practical guidance

Be specific, actionable, and encouraging. Focus on quality over quantity."""

        domains_str = ", ".join([d.replace('_', ' ').title() for d in user_profile.get('domains', [])])
        
        # Prepare resource summaries
        resource_summary = ""
        for domain, resources in ranked_resources.items():
            resource_summary += f"\n{domain.replace('_', ' ').title()}:\n"
            for r in resources[:3]:
                resource_summary += f"- {r.title} (Score: {r.relevance_score}/10)\n  Why: {r.why_recommended}\n"
        
        user_prompt = f"""Create a personalized interview preparation plan for this user:

Profile:
- Domains: {domains_str}
- Skill Level: {user_profile.get('skill_level', 'intermediate').title()}
- Learning Style: {user_profile.get('preference', 'balanced').replace('_', ' ').title()}
- Timeline: {user_profile.get('timeline', '8')} weeks
- Target Companies: {user_profile.get('companies', 'General')}

Top Recommended Resources:
{resource_summary}

Create a plan that includes:
1. A personalized introduction explaining the strategy
2. Week-by-week breakdown
3. Resource recommendations with WHY they're chosen for this user
4. Daily schedule tailored to their learning style
5. Motivational closing

Make it personal, specific, and actionable."""

        try:
            response = await self.model.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])

            content = response.content

            # Handle case where content might be a list (Gemini format)
            if isinstance(content, list):
                # Extract text from Gemini's response format
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and 'text' in item:
                        text_parts.append(item['text'])
                    elif isinstance(item, str):
                        text_parts.append(item)
                    else:
                        text_parts.append(str(item))
                content = ''.join(text_parts)

            return content

        except Exception as e:
            logger.error(f"Error synthesizing plan: {e}")
            return "Error generating personalized plan. Please try again."
