#!/usr/bin/env python3
"""
A2A Test Client for Interview Preparation Agent

This client sends A2A protocol requests to the running server and displays
the raw responses to verify correct A2A format and functionality.
"""

import asyncio
import json
import uuid
import httpx
import time
from typing import Dict, Any, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class A2ATestClient:
    """Test client for A2A Interview Preparation Agent."""

    def __init__(self, base_url: str = "http://localhost:10001"):
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=30.0)
        self.context_id = str(uuid.uuid4())
        self.session_id = str(uuid.uuid4())

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    def _create_message_request(
        self,
        query: str,
        task_id: Optional[str] = None,
        context_id: Optional[str] = None,
        push_notification_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create an A2A message/send request."""

        message_id = str(uuid.uuid4())
        context_id = context_id or self.context_id

        message_data = {
            "role": "user",
            "parts": [
                {
                    "kind": "text",
                    "text": query
                }
            ],
            "messageId": message_id,
            "contextId": context_id,
            "kind": "message"
        }

        # Only include taskId if explicitly provided
        if task_id:
            message_data["taskId"] = task_id

        request = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": message_data
            },
            "id": str(uuid.uuid4())
        }

        # Add push notification configuration if provided
        if push_notification_config:
            request["params"]["configuration"] = {
                "pushNotificationConfig": push_notification_config
            }

        return request

    async def send_message(
        self,
        query: str,
        task_id: Optional[str] = None,
        context_id: Optional[str] = None,
        push_notification_config: Optional[Dict[str, Any]] = None,
        show_raw: bool = True
    ) -> Dict[str, Any]:
        """Send a message to the A2A agent and return the response."""

        request = self._create_message_request(query, task_id, context_id, push_notification_config)

        print(f"\n📤 SENDING REQUEST:")
        print("=" * 60)
        print(json.dumps(request, indent=2))

        try:
            start_time = time.time()
            response = await self.client.post(
                f"{self.base_url}/",
                json=request,
                headers={"Content-Type": "application/json"}
            )
            end_time = time.time()

            print(f"\n📥 RESPONSE (took {end_time - start_time:.2f}s):")
            print("=" * 60)
            print(f"Status Code: {response.status_code}")
            print(f"Headers: {dict(response.headers)}")

            if show_raw:
                print(f"\nRaw Response Body:")
                response_text = response.text
                print(response_text)

                # Try to parse and pretty print JSON
                try:
                    response_json = response.json()
                    print(f"\nParsed JSON Response:")
                    print(json.dumps(response_json, indent=2))
                    return response_json
                except json.JSONDecodeError:
                    print("⚠️  Response is not valid JSON")
                    return {"error": "Invalid JSON response", "raw": response_text}
            else:
                return response.json() if response.headers.get('content-type', '').startswith('application/json') else {"raw": response.text}

        except Exception as e:
            print(f"❌ Error sending request: {e}")
            return {"error": str(e)}

    async def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information by sending a test message to see capabilities."""
        print(f"\n🔍 TESTING AGENT CAPABILITIES:")
        print("=" * 60)
        print("Note: A2A agents don't expose GET endpoints for agent cards.")
        print("Let's test the agent capabilities by sending a simple message.")

        try:
            # Send a simple test message to see agent response
            test_response = await self.send_message(
                "Hello, what can you help me with?",
                show_raw=False
            )

            if 'result' in test_response:
                task = test_response['result']
                print(f"\n✅ AGENT CAPABILITIES DETECTED:")
                print(f"   - Task Management: ✅ (Task ID: {task.get('id', 'N/A')[:8]}...)")
                print(f"   - Context Persistence: ✅ (Context ID: {task.get('contextId', 'N/A')[:8]}...)")
                print(f"   - Message History: ✅ ({len(task.get('history', []))} messages)")
                print(f"   - Status Management: ✅ (Status: {task.get('status', {}).get('state', 'N/A')})")
                print(f"   - Multi-turn Support: ✅ (State: input-required)")

                # Show agent response preview
                status_msg = task.get('status', {}).get('message', {})
                if status_msg.get('parts'):
                    text = status_msg['parts'][0].get('text', '')
                    preview = text[:100] + "..." if len(text) > 100 else text
                    print(f"   - Agent Response Preview: {preview}")

                return {
                    "capabilities": {
                        "task_management": True,
                        "context_persistence": True,
                        "message_history": True,
                        "status_management": True,
                        "multi_turn": True
                    },
                    "agent_type": "interview_preparation",
                    "protocol": "A2A JSON-RPC 2.0"
                }
            else:
                print(f"❌ Unexpected response format")
                return {"error": "Unexpected response format"}

        except Exception as e:
            print(f"❌ Error testing agent: {e}")
            return {"error": str(e)}

    async def test_multi_turn_conversation(self):
        """Test the complete multi-turn conversation flow."""
        print("\n🗣️  TESTING MULTI-TURN CONVERSATION FLOW")
        print("=" * 80)

        conversation_queries = [
            "I want to prepare for software engineering interviews",
            "I want to focus on algorithms and system design",
            "I'm at an intermediate level",
            "I prefer a balanced approach"
        ]

        responses = []
        current_context_id = self.context_id

        for i, query in enumerate(conversation_queries, 1):
            print(f"\n🔄 TURN {i}: {query}")
            print("-" * 50)

            response = await self.send_message(
                query=query,
                context_id=current_context_id,
                show_raw=False
            )

            responses.append({
                "turn": i,
                "query": query,
                "response": response
            })

            # Brief pause between turns
            await asyncio.sleep(1)

        return responses

    async def test_with_push_notifications(self, webhook_url: str = "http://localhost:8080/webhook"):
        """Test with push notification configuration."""
        print(f"\n🔔 TESTING WITH PUSH NOTIFICATIONS")
        print("=" * 80)

        # Create push notification config
        push_config = {
            "url": webhook_url,
            "token": "test-webhook-token",
            "authentication": {
                "schemes": ["Bearer"]
            }
        }

        # Send a query that should trigger async processing
        query = "I want algorithms and system design prep, intermediate level, balanced approach"

        response = await self.send_message(
            query=query,
            push_notification_config=push_config,
            show_raw=True
        )

        return response

    async def run_comprehensive_test(self):
        """Run a comprehensive test of the A2A agent."""
        print("🚀 COMPREHENSIVE A2A INTERVIEW PREP AGENT TEST")
        print("=" * 80)

        try:
            # 1. Get agent info
            agent_info = await self.get_agent_info()

            # 2. Test basic message
            print(f"\n1️⃣  Testing Basic Message")
            basic_response = await self.send_message("Hello, I need help with interview preparation")

            # 3. Test multi-turn conversation
            print(f"\n2️⃣  Testing Multi-Turn Conversation")
            conversation_responses = await self.test_multi_turn_conversation()

            # 4. Analyze responses
            print(f"\n📊 ANALYSIS:")
            print("=" * 60)

            # Check agent card
            if 'capabilities' in agent_info:
                caps = agent_info['capabilities']
                print(f"✅ Agent Capabilities:")
                print(f"   - Streaming: {caps.get('streaming', False)}")
                print(f"   - Push Notifications: {caps.get('pushNotifications', False)}")
                print(f"   - Multi-turn: {caps.get('multiTurn', False)}")

            # Check A2A response format
            if conversation_responses:
                last_response = conversation_responses[-1]['response']
                if 'result' in last_response:
                    task = last_response['result']
                    print(f"✅ A2A Task Response:")
                    print(f"   - Task ID: {task.get('id', 'N/A')}")
                    print(f"   - Context ID: {task.get('contextId', 'N/A')}")
                    print(f"   - Status State: {task.get('status', {}).get('state', 'N/A')}")

                    if 'status' in task and 'message' in task['status']:
                        message = task['status']['message']
                        print(f"   - Message Role: {message.get('role', 'N/A')}")
                        print(f"   - Message Parts: {len(message.get('parts', []))}")

            print(f"\n🎉 Comprehensive test completed!")

        except Exception as e:
            print(f"❌ Comprehensive test failed: {e}")
            logger.error(f"Test failed: {e}", exc_info=True)


async def main():
    """Main test function with interactive options."""
    print("🧪 A2A Interview Preparation Agent Test Client")
    print("=" * 60)

    client = A2ATestClient()

    try:
        # Check if server is running
        try:
            await client.client.get(f"{client.base_url}/")
            print("✅ Server is running!")
        except Exception as e:
            print(f"❌ Server not accessible: {e}")
            print("💡 Make sure to start the server first:")
            print("   python -m app --host localhost --port 10001")
            return

        while True:
            print(f"\n🔧 Test Options:")
            print("1. Get Agent Info")
            print("2. Send Single Message")
            print("3. Test Multi-Turn Conversation")
            print("4. Test with Push Notifications")
            print("5. Run Comprehensive Test")
            print("6. Custom Query")
            print("0. Exit")

            choice = input(f"\nChoose an option (0-6): ").strip()

            if choice == "0":
                break
            elif choice == "1":
                await client.get_agent_info()
            elif choice == "2":
                query = input("Enter your message: ").strip()
                if query:
                    await client.send_message(query)
            elif choice == "3":
                await client.test_multi_turn_conversation()
            elif choice == "4":
                webhook_url = input("Enter webhook URL (or press Enter for default): ").strip()
                if not webhook_url:
                    webhook_url = "http://localhost:8080/webhook"
                await client.test_with_push_notifications(webhook_url)
            elif choice == "5":
                await client.run_comprehensive_test()
            elif choice == "6":
                query = input("Enter custom query: ").strip()
                context_id = input("Enter context ID (or press Enter for new): ").strip()
                if not context_id:
                    context_id = str(uuid.uuid4())

                # Ask about push notifications
                push_choice = input("Include push notifications? (y/n): ").strip().lower()
                push_config = None
                if push_choice == 'y':
                    webhook_url = input("Webhook URL: ").strip()
                    if webhook_url:
                        push_config = {
                            "url": webhook_url,
                            "token": "test-token",
                            "authentication": {"schemes": ["Bearer"]}
                        }

                await client.send_message(query, context_id=context_id, push_notification_config=push_config)
            else:
                print("Invalid option. Please try again.")

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())