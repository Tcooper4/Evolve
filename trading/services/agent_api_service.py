"""
Agent API Service

Provides REST API endpoints for agent orchestration and management.
Compatible with the new async agent interface.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
import uvicorn

# Import agent system components
from trading.agents.agent_registry import AgentRegistry
from trading.agents.agent_manager import AgentManager
from trading.agents.agent_loop_manager import AgentLoopManager
from trading.agents.base_agent_interface import AgentConfig
from trading.utils.logging_utils import log_manager
from trading.services.websocket_service import WebSocketService

logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class AgentRequest(BaseModel):
    """Request model for agent operations."""
    agent_type: str = Field(..., description="Type of agent to create/execute")
    config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Agent configuration")
    task_data: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Task data for agent execution")

class AgentResponse(BaseModel):
    """Response model for agent operations."""
    success: bool
    agent_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    message: str
    timestamp: str

class AgentStatusResponse(BaseModel):
    """Response model for agent status."""
    agent_id: str
    status: str
    capabilities: List[str]
    last_execution: Optional[str] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    timestamp: str

class AgentAPIService:
    """Manages API endpoints for agent orchestration."""
    
    def __init__(self, config_path: str = "config/agent_api.json"):
        """Initialize the agent API service."""
        self.config_path = config_path
        self.setup_logging()
        self.load_config()
        self.setup_app()
        self.setup_routes()
        self.setup_middleware()
        
        # Initialize agent system components
        self.agent_registry = AgentRegistry()
        self.agent_manager = AgentManager()
        self.loop_manager = AgentLoopManager()
        
        # Initialize WebSocket service
        self.websocket_service = WebSocketService(self.agent_manager)
        
        logger.info("Agent API Service initialized")
    
    def setup_logging(self) -> None:
        """Set up logging."""
        log_path = Path("logs/api")
        log_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / "agent_api_service.log"),
                logging.StreamHandler()
            ]
        )
    
    def load_config(self) -> None:
        """Load configuration."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            # Use default config if file doesn't exist
            self.config = {
                "host": "0.0.0.0",
                "port": 8001,
                "secret_key": "your-secret-key-change-in-production",
                "cors_origins": ["*"],
                "max_agents": 100,
                "default_timeout": 30
            }
            logger.warning(f"Config file not found, using default config")
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise
    
    def setup_app(self) -> None:
        """Set up FastAPI application."""
        self.app = FastAPI(
            title="Agent Orchestration API",
            description="REST API for agent orchestration and management",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
    
    def setup_middleware(self) -> None:
        """Set up middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.get("cors_origins", ["*"]),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self) -> None:
        """Set up API routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "service": "agent_api",
                "timestamp": datetime.now().isoformat(),
                "agent_count": len(self.agent_manager.get_all_agents()),
                "websocket_connections": self.websocket_service.get_connection_stats()["total_connections"]
            }
        
        @self.app.get("/agents", response_model=List[Dict[str, Any]])
        async def list_agents():
            """List all available agents."""
            try:
                agents = self.agent_manager.get_all_agents()
                return [
                    {
                        "agent_id": agent_id,
                        "agent_type": agent.agent_type,
                        "status": agent.status,
                        "capabilities": agent.capabilities,
                        "created_at": agent.created_at.isoformat() if hasattr(agent, 'created_at') else None
                    }
                    for agent_id, agent in agents.items()
                ]
            except Exception as e:
                logger.error(f"Error listing agents: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/agents/{agent_id}", response_model=AgentStatusResponse)
        async def get_agent_status(agent_id: str):
            """Get status of a specific agent."""
            try:
                agent = self.agent_manager.get_agent(agent_id)
                if not agent:
                    raise HTTPException(status_code=404, detail="Agent not found")
                
                return AgentStatusResponse(
                    agent_id=agent_id,
                    status=agent.status,
                    capabilities=agent.capabilities,
                    last_execution=agent.last_execution.isoformat() if hasattr(agent, 'last_execution') and agent.last_execution else None,
                    performance_metrics=agent.get_performance_metrics() if hasattr(agent, 'get_performance_metrics') else None,
                    timestamp=datetime.now().isoformat()
                )
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error getting agent status: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/agents", response_model=AgentResponse)
        async def create_agent(request: AgentRequest):
            """Create a new agent."""
            try:
                # Create agent configuration
                config = AgentConfig(
                    agent_type=request.agent_type,
                    **request.config
                )
                
                # Create and register agent
                agent_id = self.agent_manager.create_agent(config)
                
                # Broadcast agent creation to WebSocket subscribers
                await self.websocket_service.broadcast_agent_update(
                    agent_id, "created", {"agent_type": request.agent_type}
                )
                
                return AgentResponse(
                    success=True,
                    agent_id=agent_id,
                    message=f"Agent {request.agent_type} created successfully",
                    timestamp=datetime.now().isoformat()
                )
            except Exception as e:
                logger.error(f"Error creating agent: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/agents/{agent_id}/execute", response_model=AgentResponse)
        async def execute_agent(agent_id: str, request: AgentRequest, background_tasks: BackgroundTasks):
            """Execute an agent with given task data."""
            try:
                agent = self.agent_manager.get_agent(agent_id)
                if not agent:
                    raise HTTPException(status_code=404, detail="Agent not found")
                
                # Broadcast execution start
                await self.websocket_service.broadcast_agent_update(
                    agent_id, "execution_started", {"task_data": request.task_data}
                )
                
                # Execute agent asynchronously
                result = await agent.execute(request.task_data)
                
                # Broadcast execution completion
                await self.websocket_service.broadcast_agent_update(
                    agent_id, "execution_completed", {"result": result}
                )
                
                return AgentResponse(
                    success=True,
                    agent_id=agent_id,
                    result=result,
                    message="Agent execution completed successfully",
                    timestamp=datetime.now().isoformat()
                )
            except HTTPException:
                raise
            except Exception as e:
                # Broadcast execution error
                await self.websocket_service.broadcast_agent_update(
                    agent_id, "execution_failed", {"error": str(e)}
                )
                logger.error(f"Error executing agent: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/agents/{agent_id}", response_model=AgentResponse)
        async def delete_agent(agent_id: str):
            """Delete an agent."""
            try:
                success = self.agent_manager.remove_agent(agent_id)
                if not success:
                    raise HTTPException(status_code=404, detail="Agent not found")
                
                # Broadcast agent deletion
                await self.websocket_service.broadcast_agent_update(
                    agent_id, "deleted", {}
                )
                
                return AgentResponse(
                    success=True,
                    agent_id=agent_id,
                    message="Agent deleted successfully",
                    timestamp=datetime.now().isoformat()
                )
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error deleting agent: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/agents/types", response_model=List[Dict[str, Any]])
        async def list_agent_types():
            """List all available agent types."""
            try:
                agent_types = self.agent_registry.get_available_agent_types()
                return [
                    {
                        "agent_type": agent_type,
                        "capabilities": capabilities,
                        "description": self.agent_registry.get_agent_description(agent_type)
                    }
                    for agent_type, capabilities in agent_types.items()
                ]
            except Exception as e:
                logger.error(f"Error listing agent types: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/agents/batch", response_model=List[AgentResponse])
        async def create_agents_batch(requests: List[AgentRequest]):
            """Create multiple agents in batch."""
            try:
                results = []
                for request in requests:
                    try:
                        config = AgentConfig(
                            agent_type=request.agent_type,
                            **request.config
                        )
                        agent_id = self.agent_manager.create_agent(config)
                        
                        # Broadcast agent creation
                        await self.websocket_service.broadcast_agent_update(
                            agent_id, "created", {"agent_type": request.agent_type}
                        )
                        
                        results.append(AgentResponse(
                            success=True,
                            agent_id=agent_id,
                            message=f"Agent {request.agent_type} created successfully",
                            timestamp=datetime.now().isoformat()
                        ))
                    except Exception as e:
                        results.append(AgentResponse(
                            success=False,
                            message=f"Failed to create agent {request.agent_type}: {str(e)}",
                            timestamp=datetime.now().isoformat()
                        ))
                
                return results
            except Exception as e:
                logger.error(f"Error in batch agent creation: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/system/status")
        async def get_system_status():
            """Get overall system status."""
            try:
                agents = self.agent_manager.get_all_agents()
                active_agents = [a for a in agents.values() if a.status == "active"]
                
                status_data = {
                    "total_agents": len(agents),
                    "active_agents": len(active_agents),
                    "system_status": "healthy" if len(agents) < self.config.get("max_agents", 100) else "at_capacity",
                    "websocket_stats": self.websocket_service.get_connection_stats(),
                    "timestamp": datetime.now().isoformat()
                }
                
                # Broadcast system status update
                await self.websocket_service.broadcast_system_update("status", status_data)
                
                return status_data
            except Exception as e:
                logger.error(f"Error getting system status: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/trigger")
        async def trigger_agentic_action(request: Dict[str, Any]):
            """Universal agentic trigger endpoint: accepts a prompt and returns agentic result."""
            try:
                prompt = request.get('prompt')
                if not prompt:
                    raise HTTPException(status_code=400, detail="Missing 'prompt' in request body")
                # Route prompt through PromptAgent or PromptRouterAgent
                from trading.llm.agent import PromptAgent
                agent = PromptAgent()
                result = agent.process_prompt(prompt)
                return {"success": True, "result": result, "timestamp": datetime.now().isoformat()}
            except Exception as e:
                logger.error(f"Error in /trigger endpoint: {e}")
                return {"success": False, "error": str(e), "timestamp": datetime.now().isoformat()}
        
        # WebSocket endpoints
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time agent updates."""
            await self.websocket_service.connect(websocket)
            try:
                while True:
                    message = await websocket.receive_text()
                    await self.websocket_service.handle_message(websocket, message)
            except WebSocketDisconnect:
                await self.websocket_service.disconnect(websocket)
            except Exception as e:
                logger.error(f"WebSocket error: {str(e)}")
                await self.websocket_service.disconnect(websocket)
        
        @self.app.get("/ws/test")
        async def websocket_test_page():
            """Simple WebSocket test page."""
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>WebSocket Test</title>
            </head>
            <body>
                <h1>Agent WebSocket Test</h1>
                <div id="messages"></div>
                <input type="text" id="messageInput" placeholder="Enter message...">
                <button onclick="sendMessage()">Send</button>
                
                <script>
                    const ws = new WebSocket("ws://localhost:8001/ws");
                    const messagesDiv = document.getElementById("messages");
                    const messageInput = document.getElementById("messageInput");
                    
                    ws.onmessage = function(event) {
                        const message = JSON.parse(event.data);
                        const messageElement = document.createElement("div");
                        messageElement.textContent = JSON.stringify(message, null, 2);
                        messagesDiv.appendChild(messageElement);
                    };
                    
                    function sendMessage() {
                        const message = messageInput.value;
                        if (message) {
                            ws.send(message);
                            messageInput.value = "";
                        }
                    }
                    
                    messageInput.addEventListener("keypress", function(e) {
                        if (e.key === "Enter") {
                            sendMessage();
                        }
                    });
                </script>
            </body>
            </html>
            """
            return HTMLResponse(content=html_content)
    
    async def start(self) -> None:
        """Start the API service."""
        try:
            config = uvicorn.Config(
                self.app,
                host=self.config["host"],
                port=self.config["port"],
                log_level="info"
            )
            server = uvicorn.Server(config)
            await server.serve()
        except Exception as e:
            logger.error(f"Error starting API service: {str(e)}")
            raise

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Agent API service')
    parser.add_argument('--config', default="config/agent_api.json", help='Path to config file')
    args = parser.parse_args()
    
    try:
        service = AgentAPIService(args.config)
        asyncio.run(service.start())
    except KeyboardInterrupt:
        logging.info("Agent API service interrupted")
    except Exception as e:
        logging.error(f"Error in agent API service: {str(e)}")
        raise

if __name__ == '__main__':
    main() 