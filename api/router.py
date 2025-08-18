from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi import Depends
import uuid
import json
import logging

from core.session_manager import session_manager
from agents.example_agent import ExampleAgent
from models.schemas import Request, Response
from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter()
@router.post("/chat", response_model=dict)
async def chat(request: Request):
	# Simple REST chat endpoint using ExampleAgent
	session_id = request.session_id or str(uuid.uuid4())
	state = session_manager.get_or_create_state(session_id=session_id, domain="general")
	state.add_user_message(request.message)
	agent = ExampleAgent(session_id=session_id)
	state = await agent.process(state)
	session_manager.save_state(state)
	assistant_messages = state.get_assistant_messages()
	assistant_text = assistant_messages[-1] if assistant_messages else ""
	return {
		"session_id": session_id,
		"message": assistant_text,
		"status": state.status,
		"is_task_complete": state.is_task_complete,
	}


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
	await websocket.accept()
	# Determine or create a session id
	session_id = websocket.query_params.get("session_id") or str(uuid.uuid4())
	logger.info(f"WebSocket connected: session_id={session_id}")
	
	# Prepare initial state and agent
	state = session_manager.get_or_create_state(session_id=session_id, domain="general")
	agent = ExampleAgent(session_id=session_id)
	
	# Send ready message
	await websocket.send_json({
		"type": "ready",
		"session_id": session_id,
		"message": "WebSocket connected. Send text messages to chat."
	})
	
	try:
		while True:
			# Receive message (supports raw text or JSON {content: ...})
			try:
				data = await websocket.receive_text()
				try:
					payload = json.loads(data)
					content = payload.get("content", "")
				except Exception:
					content = data
			except WebSocketDisconnect:
				logger.info(f"WebSocket disconnected: session_id={session_id}")
				break
			
			if not content:
				await websocket.send_json({"type": "error", "message": "Empty message"})
				continue
			
			# Add user message to state
			state.add_user_message(content)
			
			# Process with agent
			state = await agent.process(state)
			
			# Save state
			session_manager.save_state(state)
			
			# Send assistant response (last assistant message)
			assistant_messages = state.get_assistant_messages()
			assistant_text = assistant_messages[-1] if assistant_messages else ""
			await websocket.send_json({
				"type": "assistant_message",
				"content": assistant_text,
				"status": state.status,
				"is_task_complete": state.is_task_complete,
			})
			
	except Exception as e:
		logger.error(f"WebSocket error: {e}", exc_info=True)
		try:
			await websocket.send_json({"type": "error", "message": str(e)})
		except Exception:
			pass
		finally:
			try:
				await websocket.close()
			except Exception:
				pass
