# Interview Preparation Agent

A comprehensive **AI-Powered** Interview Preparation Agent built with LangGraph, Google Gemini, and the A2A Protocol. This intelligent agent provides personalized interview preparation plans through multi-turn conversations, AI-powered resource analysis, web research capabilities, and async processing with push notifications.

## ✨ Features

- **🤖 AI-Powered Intelligence**: Advanced AI analysis using Google Gemini 2.0 Flash for:
  - Natural language intent understanding (no keyword matching!)
  - Intelligent resource ranking and filtering
  - Personalized plan synthesis tailored to your profile
  - Relevance scoring (0-10) with explanations for every recommendation
- **🎯 Personalized Interview Plans**: Tailored preparation strategies based on your experience level, preferred domains, and learning style
- **🔍 Real-time Web Research**: Searches for the latest interview resources, company-specific insights, and learning materials
- **💬 Multi-turn Conversations**: Intelligent conversation flow that guides you through the preparation planning process
- **🔔 Push Notifications**: Async processing with real-time progress updates (optional)
- **🧠 LangGraph + Google Gemini**: Powered by advanced AI for intelligent responses and structured thinking
- **📋 A2A Protocol Compliant**: Full support for the Agent-to-Agent communication protocol

### AI-Powered Improvements

Unlike traditional keyword-matching systems, this agent uses AI to:

- **Understand Intent**: Parses natural language to extract domains, skill levels, and preferences with confidence scores
- **Rank Resources**: Analyzes ALL search results and ranks them by relevance to your specific profile (not just first N results)
- **Explain Recommendations**: Provides personalized "why recommended" explanations for each resource
- **Synthesize Plans**: Generates truly personalized study plans (not template-based) incorporating your goals and timeline
- **Filter Quality**: Automatically filters out low-quality or irrelevant resources

**Example**: Instead of blindly taking the first 5 search results, the AI analyzes all results, ranks them by relevance to your intermediate skill level and coding-heavy preference, and explains why each resource is perfect for you.

### Supported Interview Domains
- **Algorithms & Data Structures**
- **System Design**
- **Databases** (SQL, NoSQL, database design)
- **Machine Learning**
- **Behavioral Interviews**
- **Frontend Development**
- **Backend Development**

## 🚀 Quick Start

### Prerequisites
- Python 3.12 or higher
- Google API Key (for Gemini model access)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/interview-preparation-agent.git
   cd interview-preparation-agent
   ```

2. **Install dependencies**
   ```bash
   pip install -e .
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```

   Edit `.env` and add your Google API key:
   ```bash
   GOOGLE_API_KEY=your_google_api_key_here
   ```

4. **Start the agent**
   ```bash
   python -m app
   ```

   The agent will start on `http://localhost:10001` by default.

### Getting Your Google API Key

1. Visit the [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Click "Get API key" and create a new API key
4. Copy the API key to your `.env` file

## 🧪 Testing

Test your setup using the included test client:

```bash
python test_agent.py
```

The test client provides several options:
- **Get Agent Info**: Test basic agent capabilities
- **Send Single Message**: Send a custom message to the agent
- **Test Multi-Turn Conversation**: Complete interview preparation flow
- **Test with Push Notifications**: Test async processing capabilities
- **Run Comprehensive Test**: Full feature test suite
- **Custom Query**: Send custom queries with advanced options

### Test AI Improvements

Test the AI-powered features specifically:

```bash
python test_ai_improvements.py
```

This tests:
- Intent parsing with natural language variations
- Resource ranking for different user profiles
- Personalized plan synthesis

### Example Test Flow

1. Start the agent: `python -m app`
2. In another terminal, run: `python test_agent.py`
3. Choose option 3 for "Test Multi-Turn Conversation"
4. Watch as the agent guides you through the complete interview preparation process with AI-powered recommendations

## 💬 Usage Example

Here's what a typical conversation looks like:

```
User: I want to prepare for software engineering interviews

Agent: Great! I'll help you prepare for interviews. Which interview domains would you like to focus on?
🔹 Algorithms - Data structures, coding problems
🔹 System Design - Scalable system architecture
🔹 Databases - SQL, NoSQL, database design
🔹 Machine Learning - ML algorithms, model design
🔹 Behavioral - Soft skills, culture fit
🔹 Frontend - UI/UX, JavaScript, frameworks
🔹 Backend - APIs, servers, architecture

User: I want to focus on algorithms and system design

Agent: Perfect! You've selected: Algorithms, System Design

Now, what's your current skill level?
🟢 Beginner - New to the field
🟡 Intermediate - Some experience
🔴 Advanced - Experienced professional

User: Intermediate

Agent: What's your learning preference?
📚 Theory-Heavy - Focus on concepts and understanding
💻 Coding-Heavy - Emphasis on hands-on practice
⚖️ Balanced - Mix of theory and practice
🏗️ Project-Based - Learn through building projects

User: Balanced approach

Agent: Perfect! I'm now creating your personalized interview preparation plan...
🔍 Starting comprehensive research for algorithms and system design resources...

[The agent then provides a detailed, AI-personalized preparation plan with:
 - Relevance scores (0-10) for each resource
 - Personalized explanations of why each resource is recommended
 - Difficulty matching to your skill level
 - Strategic advice tailored to your learning style]
```

## 🛠️ Configuration Options

### Environment Variables

```bash
# Required
GOOGLE_API_KEY=your_google_api_key_here

# Optional - Web Search
ENABLE_WEB_SEARCH=true
SEARCH_RESULTS_LIMIT=5

# Optional - Push Notifications
ENABLE_PUSH_NOTIFICATIONS=true
PUSH_NOTIFICATION_MODE=multi_turn
PROCESSING_DELAY_SECONDS=5
CALLBACK_TIMEOUT_SECONDS=60

# Optional - A2A Integration
BASE_API_URL=http://localhost:8000
A2A_CALLBACK_TOKEN=your_jwt_token_here
```

### Custom Host and Port

```bash
# Start on custom host/port
python -m app --host 0.0.0.0 --port 8080
```

## 🏗️ Architecture

The agent follows a modular architecture with clear separation of concerns and AI-powered intelligence:

```
┌─────────────────────────────────────────────────────────────┐
│                    A2A Client                               │
└─────────────────────┬───────────────────────────────────────┘
                     │ HTTP/JSON-RPC
┌─────────────────────▼───────────────────────────────────────┐
│              InterviewPrepExecutor                          │
├─────────────────────────────────────────────────────────────┤
│              InterviewPrepAgent                             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                LangGraph ReAct Agent                   │ │
│  │  ┌─────────────┬──────────────┬─────────────────────┐  │ │
│  │  │Google Gemini│ Web Search   │ Conversation State  │  │ │
│  │  │   Model     │    Tools     │   Management       │  │ │
│  │  └─────────────┴──────────────┴─────────────────────┘  │ │
│  │  ┌─────────────────────────────────────────────────┐   │ │
│  │  │         AIResourceAnalyzer (NEW!)              │   │ │
│  │  │  • Intent Understanding (NLP)                  │   │ │
│  │  │  • Resource Ranking (0-10 scores)              │   │ │
│  │  │  • Personalized Plan Synthesis                 │   │ │
│  │  └─────────────────────────────────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

- **AIResourceAnalyzer** ⭐ NEW: AI-powered analysis engine for intent parsing, resource ranking, and plan synthesis
- **ConversationState**: Manages conversation phases and user inputs
- **WebSearchManager**: Handles web research for interview resources
- **InterviewPrepAgent**: Core LangGraph agent with memory and tools
- **PushNotificationHandler**: Manages async processing and callbacks
- **LLMProvider**: Centralized LLM configuration and management

### AI Processing Flow

```
User Input
    ↓
AIResourceAnalyzer.parse_user_intent()
    ↓ (structured intent with confidence scores)
WebSearchManager.comprehensive_research()
    ↓ (raw search results from multiple sources)
AIResourceAnalyzer.rank_and_filter_resources()
    ↓ (ranked recommendations with 0-10 scores + explanations)
AIResourceAnalyzer.synthesize_personalized_plan()
    ↓ (complete personalized plan with strategic advice)
User receives AI-personalized plan
```

## 🔧 Development

### Project Structure

```
interview-preparation-agent/
├── app/                          # Main application package
│   ├── __init__.py              # Package exports
│   ├── __main__.py              # Application entry point
│   ├── ai_resource_analyzer.py  # ⭐ AI-powered analysis engine
│   ├── conversation_state.py    # Conversation management
│   ├── interview_prep_agent.py  # Core LangGraph agent
│   ├── interview_prep_executor.py # A2A executor
│   ├── llm_provider.py          # LLM configuration
│   ├── push_notification_handler.py # Async processing
│   └── web_search_tools.py      # Web research tools
├── test_agent.py                # Comprehensive test client
├── test_ai_improvements.py      # AI features test suite
├── AI_IMPROVEMENTS_SUMMARY.md   # Detailed AI improvements doc
├── requirements.txt             # Python dependencies
├── pyproject.toml              # Project configuration
├── .env.example                # Environment template
└── README.md                   # This file
```

### Running Tests

The `test_agent.py` file provides comprehensive testing capabilities:

```bash
# Interactive test menu
python test_agent.py

# Or test programmatically
python -c "
import asyncio
from test_agent import A2ATestClient

async def test():
    client = A2ATestClient()
    await client.run_comprehensive_test()
    await client.close()

asyncio.run(test())
"
```

### Testing AI Features

```bash
# Test AI-powered improvements
python test_ai_improvements.py
```

### Logging

Enable debug logging for development:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📦 Dependencies

### Core Dependencies
- **a2a-sdk==0.3.2**: A2A Protocol implementation
- **langchain-google-genai>=2.0.10**: Google Gemini integration for AI analysis
- **langgraph>=0.3.18**: Agent orchestration framework
- **langchain-openai>=0.1.0**: OpenAI integration support
- **langchain-community>=0.3.0**: Community integrations

### Web & HTTP
- **uvicorn>=0.34.2**: ASGI server for hosting
- **httpx>=0.28.1**: HTTP client for async operations
- **fastapi>=0.116.1**: Web framework
- **starlette>=0.47.2**: ASGI toolkit
- **sse-starlette>=2.0.0**: Server-Sent Events support

### Search & Data
- **duckduckgo-search>=6.0.0**: Web search capabilities
- **pydantic>=2.10.6**: Data validation
- **python-dotenv>=1.1.0**: Environment management

### Development
- **pytest>=7.0.0**: Testing framework
- **pytest-asyncio>=0.21.0**: Async testing support
- **click>=8.1.8**: CLI utilities

## 🎯 AI-Powered Features Deep Dive

### 1. Intent Understanding
The AI understands natural language variations:
- "I want to focus on algo" → algorithms
- "I'm new to this" → beginner
- "I prefer hands-on coding" → coding_heavy

### 2. Resource Ranking
Each resource gets:
- **Relevance Score**: 0-10 rating based on your profile
- **Why Recommended**: Personalized explanation
- **Difficulty Match**: How it aligns with your skill level
- **Resource Type**: practice/tutorial/guide/course

### 3. Plan Synthesis
AI generates plans that include:
- Strategic advice for your specific situation
- Personalized timelines and schedules
- Motivational guidance
- Adaptive recommendations based on your goals

### 4. Quality Filtering
AI automatically filters out:
- Outdated resources
- Irrelevant content
- Low-quality materials
- Duplicate information

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and test thoroughly
4. Commit your changes: `git commit -m 'Add amazing feature'`
5. Push to the branch: `git push origin feature/amazing-feature`
6. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

**Agent not starting:**
- Ensure Python 3.12+ is installed
- Check that all dependencies are installed: `pip install -e .`
- Verify your Google API key is set in `.env`

**Test client connection errors:**
- Make sure the agent is running: `python -m app`
- Check the correct port (default: 10001)
- Verify no firewall blocking the connection

**API key errors:**
- Ensure your Google API key is valid and has Gemini access
- Check the key is properly set in your `.env` file
- Verify the API key has sufficient quota

**AI analysis not working:**
- Check Google API key has access to Gemini 2.0 Flash
- Verify network connectivity to Google AI services
- Check logs for specific AI-related errors
- System falls back to keyword matching if AI fails

### Getting Help

If you encounter issues:
1. Check the troubleshooting section above
2. Review the logs for error messages
3. Test with the included `test_agent.py` script
4. Test AI features with `test_ai_improvements.py`
5. Review `AI_IMPROVEMENTS_SUMMARY.md` for detailed AI architecture
6. Open an issue on GitHub with detailed error information

## 📚 Documentation

- **AI_IMPROVEMENTS_SUMMARY.md**: Detailed documentation of AI-powered improvements
- **README.md**: This file - general overview and setup
- **.env.example**: Environment configuration template

## 🔄 Recent Updates

### AI-Powered Intelligence (Latest)
- ✅ Added `AIResourceAnalyzer` for intelligent analysis
- ✅ Natural language intent understanding
- ✅ AI-powered resource ranking with relevance scores
- ✅ Personalized plan synthesis
- ✅ Quality filtering and explanations for recommendations
- ✅ Fallback mechanisms for reliability

### Previous Updates
- Multi-turn conversation support
- Push notification integration
- Web search capabilities
- A2A protocol compliance

---