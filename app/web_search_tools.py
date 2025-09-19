import os
import logging
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from duckduckgo_search import DDGS
import asyncio
import httpx

logger = logging.getLogger(__name__)


@tool
async def search_interview_resources(
    query: str,
    domain: str = "general",
    max_results: int = 5
) -> Dict[str, Any]:
    """
    Search for interview preparation resources and information.

    Args:
        query: The search query for interview resources
        domain: The interview domain (algorithms, system_design, etc.)
        max_results: Maximum number of search results to return

    Returns:
        Dictionary containing search results with titles, snippets, and URLs
    """
    try:
        # Check if web search is enabled
        if not os.getenv('ENABLE_WEB_SEARCH', 'true').lower() == 'true':
            return {
                'success': False,
                'error': 'Web search is disabled',
                'results': []
            }

        # Initialize DuckDuckGo search
        ddgs = DDGS()

        # Enhance query with domain-specific terms
        enhanced_query = f"{query} {domain} interview preparation"

        logger.info(f"Searching for: {enhanced_query}")

        # Perform search
        search_results = []
        try:
            results = ddgs.text(
                keywords=enhanced_query,
                max_results=max_results,
                region='us-en',
                safesearch='moderate'
            )

            for result in results:
                search_results.append({
                    'title': result.get('title', ''),
                    'snippet': result.get('body', ''),
                    'url': result.get('href', ''),
                    'domain': domain
                })

        except Exception as search_error:
            logger.error(f"DuckDuckGo search failed: {search_error}")
            return {
                'success': False,
                'error': f'Search failed: {str(search_error)}',
                'results': []
            }

        return {
            'success': True,
            'query': enhanced_query,
            'domain': domain,
            'results': search_results,
            'total_results': len(search_results)
        }

    except Exception as e:
        logger.error(f"Search tool error: {e}")
        return {
            'success': False,
            'error': str(e),
            'results': []
        }


@tool
async def search_company_interview_info(
    company_name: str,
    role_type: str = "software engineer"
) -> Dict[str, Any]:
    """
    Search for company-specific interview information and patterns.

    Args:
        company_name: Name of the company
        role_type: Type of role being interviewed for

    Returns:
        Dictionary containing company interview information
    """
    try:
        if not os.getenv('ENABLE_WEB_SEARCH', 'true').lower() == 'true':
            return {
                'success': False,
                'error': 'Web search is disabled',
                'results': []
            }

        ddgs = DDGS()

        # Search for company-specific interview info
        query = f"{company_name} {role_type} interview questions process experience"

        logger.info(f"Searching company info for: {query}")

        search_results = []
        try:
            results = ddgs.text(
                keywords=query,
                max_results=5,
                region='us-en',
                safesearch='moderate'
            )

            for result in results:
                search_results.append({
                    'title': result.get('title', ''),
                    'snippet': result.get('body', ''),
                    'url': result.get('href', ''),
                    'company': company_name,
                    'role_type': role_type
                })

        except Exception as search_error:
            logger.error(f"Company search failed: {search_error}")
            return {
                'success': False,
                'error': f'Company search failed: {str(search_error)}',
                'results': []
            }

        return {
            'success': True,
            'query': query,
            'company': company_name,
            'role_type': role_type,
            'results': search_results,
            'total_results': len(search_results)
        }

    except Exception as e:
        logger.error(f"Company search tool error: {e}")
        return {
            'success': False,
            'error': str(e),
            'results': []
        }


@tool
async def search_learning_resources(
    topic: str,
    skill_level: str = "intermediate",
    resource_type: str = "all"
) -> Dict[str, Any]:
    """
    Search for learning resources for specific topics and skill levels.

    Args:
        topic: The specific topic to find resources for
        skill_level: User's skill level (beginner, intermediate, advanced)
        resource_type: Type of resource (courses, books, practice, all)

    Returns:
        Dictionary containing learning resources
    """
    try:
        if not os.getenv('ENABLE_WEB_SEARCH', 'true').lower() == 'true':
            return {
                'success': False,
                'error': 'Web search is disabled',
                'results': []
            }

        ddgs = DDGS()

        # Build search query based on resource type
        if resource_type == "courses":
            query = f"{topic} {skill_level} online courses tutorial"
        elif resource_type == "books":
            query = f"{topic} {skill_level} books programming"
        elif resource_type == "practice":
            query = f"{topic} {skill_level} practice problems exercises"
        else:
            query = f"{topic} {skill_level} learning resources tutorial practice"

        logger.info(f"Searching learning resources for: {query}")

        search_results = []
        try:
            results = ddgs.text(
                keywords=query,
                max_results=6,
                region='us-en',
                safesearch='moderate'
            )

            for result in results:
                search_results.append({
                    'title': result.get('title', ''),
                    'snippet': result.get('body', ''),
                    'url': result.get('href', ''),
                    'topic': topic,
                    'skill_level': skill_level,
                    'resource_type': resource_type
                })

        except Exception as search_error:
            logger.error(f"Learning resources search failed: {search_error}")
            return {
                'success': False,
                'error': f'Learning resources search failed: {str(search_error)}',
                'results': []
            }

        return {
            'success': True,
            'query': query,
            'topic': topic,
            'skill_level': skill_level,
            'resource_type': resource_type,
            'results': search_results,
            'total_results': len(search_results)
        }

    except Exception as e:
        logger.error(f"Learning resources search tool error: {e}")
        return {
            'success': False,
            'error': str(e),
            'results': []
        }


@tool
async def search_leetcode_problems(
    topic: str,
    difficulty: str = "medium",
    max_results: int = 10
) -> Dict[str, Any]:
    """
    Search for specific LeetCode problems and problem lists for a given topic.

    Args:
        topic: The algorithm/data structure topic (e.g., "arrays", "dynamic programming")
        difficulty: Problem difficulty (easy, medium, hard, all)
        max_results: Maximum number of results to return

    Returns:
        Dictionary containing LeetCode problem information
    """
    try:
        if not os.getenv('ENABLE_WEB_SEARCH', 'true').lower() == 'true':
            return {
                'success': False,
                'error': 'Web search is disabled',
                'results': []
            }

        ddgs = DDGS()

        # Create specific LeetCode search queries
        queries = [
            f"site:leetcode.com {topic} {difficulty} problems",
            f"leetcode {topic} problem list {difficulty}",
            f"best leetcode {topic} problems {difficulty} interview"
        ]

        all_results = []

        for query in queries:
            logger.info(f"Searching LeetCode problems: {query}")
            try:
                results = ddgs.text(
                    keywords=query,
                    max_results=max_results // len(queries) + 2,
                    region='us-en',
                    safesearch='moderate'
                )

                for result in results:
                    if 'leetcode.com' in result.get('href', '').lower() or 'leetcode' in result.get('title', '').lower():
                        all_results.append({
                            'title': result.get('title', ''),
                            'snippet': result.get('body', ''),
                            'url': result.get('href', ''),
                            'topic': topic,
                            'difficulty': difficulty,
                            'platform': 'leetcode'
                        })

                await asyncio.sleep(0.5)  # Be respectful to search service

            except Exception as search_error:
                logger.error(f"LeetCode search failed for query '{query}': {search_error}")
                continue

        # Remove duplicates based on URL
        seen_urls = set()
        unique_results = []
        for result in all_results:
            if result['url'] not in seen_urls:
                seen_urls.add(result['url'])
                unique_results.append(result)
                if len(unique_results) >= max_results:
                    break

        return {
            'success': True,
            'topic': topic,
            'difficulty': difficulty,
            'results': unique_results[:max_results],
            'total_results': len(unique_results)
        }

    except Exception as e:
        logger.error(f"LeetCode search tool error: {e}")
        return {
            'success': False,
            'error': str(e),
            'results': []
        }


@tool
async def search_youtube_channels(
    topic: str,
    content_type: str = "tutorial",
    max_results: int = 8
) -> Dict[str, Any]:
    """
    Search for YouTube channels and videos for interview preparation.

    Args:
        topic: The topic to search for (e.g., "system design", "algorithms")
        content_type: Type of content (tutorial, interview, explanation, all)
        max_results: Maximum number of results to return

    Returns:
        Dictionary containing YouTube channel and video information
    """
    try:
        if not os.getenv('ENABLE_WEB_SEARCH', 'true').lower() == 'true':
            return {
                'success': False,
                'error': 'Web search is disabled',
                'results': []
            }

        ddgs = DDGS()

        # Create YouTube-specific search queries
        if content_type == "tutorial":
            query = f"site:youtube.com {topic} tutorial programming interview"
        elif content_type == "interview":
            query = f"site:youtube.com {topic} mock interview coding"
        elif content_type == "explanation":
            query = f"site:youtube.com {topic} explained programming concepts"
        else:
            query = f"site:youtube.com {topic} programming interview preparation"

        logger.info(f"Searching YouTube content: {query}")

        search_results = []
        try:
            results = ddgs.text(
                keywords=query,
                max_results=max_results,
                region='us-en',
                safesearch='moderate'
            )

            for result in results:
                if 'youtube.com' in result.get('href', ''):
                    search_results.append({
                        'title': result.get('title', ''),
                        'snippet': result.get('body', ''),
                        'url': result.get('href', ''),
                        'topic': topic,
                        'content_type': content_type,
                        'platform': 'youtube'
                    })

        except Exception as search_error:
            logger.error(f"YouTube search failed: {search_error}")
            return {
                'success': False,
                'error': f'YouTube search failed: {str(search_error)}',
                'results': []
            }

        return {
            'success': True,
            'query': query,
            'topic': topic,
            'content_type': content_type,
            'results': search_results,
            'total_results': len(search_results)
        }

    except Exception as e:
        logger.error(f"YouTube search tool error: {e}")
        return {
            'success': False,
            'error': str(e),
            'results': []
        }


@tool
async def search_current_interview_guides(
    domain: str,
    year: str = "2024",
    guide_type: str = "comprehensive"
) -> Dict[str, Any]:
    """
    Search for current, up-to-date interview preparation guides and articles.

    Args:
        domain: Interview domain (algorithms, system_design, etc.)
        year: Target year for current guides
        guide_type: Type of guide (comprehensive, quick, roadmap, tips)

    Returns:
        Dictionary containing current interview guide information
    """
    try:
        if not os.getenv('ENABLE_WEB_SEARCH', 'true').lower() == 'true':
            return {
                'success': False,
                'error': 'Web search is disabled',
                'results': []
            }

        ddgs = DDGS()

        # Create queries for current guides
        domain_clean = domain.replace('_', ' ')

        if guide_type == "comprehensive":
            query = f"{domain_clean} interview guide {year} complete preparation"
        elif guide_type == "quick":
            query = f"{domain_clean} interview cheat sheet {year} quick reference"
        elif guide_type == "roadmap":
            query = f"{domain_clean} interview roadmap {year} study plan"
        else:
            query = f"{domain_clean} interview tips {year} latest advice"

        logger.info(f"Searching current guides: {query}")

        search_results = []
        try:
            results = ddgs.text(
                keywords=query,
                max_results=8,
                region='us-en',
                safesearch='moderate'
            )

            for result in results:
                # Prioritize results from known tech/education sites
                url = result.get('href', '').lower()
                is_quality_site = any(site in url for site in [
                    'medium.com', 'dev.to', 'hackernoon.com', 'freecodecamp.org',
                    'towards', 'github.com', 'interviewbit.com', 'geeksforgeeks.org'
                ])

                search_results.append({
                    'title': result.get('title', ''),
                    'snippet': result.get('body', ''),
                    'url': result.get('href', ''),
                    'domain': domain,
                    'year': year,
                    'guide_type': guide_type,
                    'is_quality_site': is_quality_site
                })

        except Exception as search_error:
            logger.error(f"Current guides search failed: {search_error}")
            return {
                'success': False,
                'error': f'Current guides search failed: {str(search_error)}',
                'results': []
            }

        # Sort by quality sites first
        search_results.sort(key=lambda x: x.get('is_quality_site', False), reverse=True)

        return {
            'success': True,
            'query': query,
            'domain': domain,
            'year': year,
            'guide_type': guide_type,
            'results': search_results[:6],  # Return top 6 results
            'total_results': len(search_results)
        }

    except Exception as e:
        logger.error(f"Current guides search tool error: {e}")
        return {
            'success': False,
            'error': str(e),
            'results': []
        }


class WebSearchManager:
    """Manager class for coordinating web searches during interview prep planning."""

    def __init__(self):
        self.search_enabled = os.getenv('ENABLE_WEB_SEARCH', 'true').lower() == 'true'
        self.max_results = int(os.getenv('SEARCH_RESULTS_LIMIT', '5'))

    async def comprehensive_research(
        self,
        domains: List[str],
        skill_level: str,
        companies: List[str] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive research for interview preparation.

        Args:
            domains: List of interview domains to research
            skill_level: User's skill level
            companies: Optional list of target companies

        Returns:
            Comprehensive research results
        """
        if not self.search_enabled:
            return {
                'success': False,
                'error': 'Web search is disabled',
                'research_data': {}
            }

        research_data = {
            'domains': {},
            'companies': {},
            'leetcode_problems': {},
            'youtube_resources': {},
            'current_guides': {}
        }

        try:
            # Research each domain comprehensively
            for domain in domains:
                logger.info(f"Researching domain: {domain}")

                # Search for domain-specific interview resources
                domain_results = await search_interview_resources(
                    query=f"{domain} interview preparation guide",
                    domain=domain,
                    max_results=self.max_results
                )

                # Search for learning resources
                learning_results = await search_learning_resources(
                    topic=domain,
                    skill_level=skill_level,
                    resource_type="all"
                )

                # Search for current interview guides
                current_guides = await search_current_interview_guides(
                    domain=domain,
                    year="2024",
                    guide_type="comprehensive"
                )

                # Search for YouTube content
                youtube_results = await search_youtube_channels(
                    topic=domain,
                    content_type="tutorial",
                    max_results=6
                )

                # For algorithms domain, search for LeetCode problems
                leetcode_results = None
                if domain == 'algorithms':
                    difficulty = 'easy' if skill_level == 'beginner' else 'medium' if skill_level == 'intermediate' else 'hard'
                    leetcode_results = await search_leetcode_problems(
                        topic="algorithms data structures",
                        difficulty=difficulty,
                        max_results=8
                    )

                research_data['domains'][domain] = {
                    'interview_info': domain_results,
                    'learning_resources': learning_results,
                    'current_guides': current_guides,
                    'youtube_resources': youtube_results,
                    'leetcode_problems': leetcode_results
                }

                # Add small delay to be respectful to search service
                await asyncio.sleep(1)

            # Research companies if provided
            if companies:
                for company in companies:
                    logger.info(f"Researching company: {company}")

                    company_results = await search_company_interview_info(
                        company_name=company,
                        role_type="software engineer"
                    )

                    research_data['companies'][company] = company_results
                    await asyncio.sleep(1)

            return {
                'success': True,
                'research_data': research_data,
                'domains_researched': len(domains),
                'companies_researched': len(companies) if companies else 0
            }

        except Exception as e:
            logger.error(f"Comprehensive research failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'research_data': research_data
            }