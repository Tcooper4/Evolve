from fastapi import APIRouter, Request, HTTPException, WebSocket
import logging
from automation.notifications.notification_manager import NotificationManager
from automation.web.middleware import login_required
from automation.web.websocket import WebSocketHandler

# Initialize logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/notifications", tags=["notifications"])

# Initialize WebSocket handler
websocket_handler = None

def init_routes(notification_manager: NotificationManager, ws_handler: WebSocketHandler):
    global websocket_handler
    websocket_handler = ws_handler

    return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    user_id = request.state.user.get("username") if hasattr(request.state, "user") else None
    if not user_id:
        await websocket.close(code=4001)
        return
    
    await websocket_handler.handle_websocket(websocket, user_id)

@router.get("")
@login_required
async def get_notifications(
    request: Request,
    limit: int = 10,
    offset: int = 0,
    unread_only: bool = False
):
    try:
        user_id = request.state.user["username"]
        notifications = await notification_manager.get_user_notifications(
            user_id=user_id,
            limit=limit,
            offset=offset,
            unread_only=unread_only
        )
        
        return notifications
    except Exception as e:
        logger.error(f"Error getting notifications: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get notifications")

@router.post("/{notification_id}/read")
@login_required
async def mark_notification_as_read(request: Request, notification_id: str):
    try:
        user_id = request.state.user["username"]
        success = await notification_manager.mark_as_read(notification_id, user_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Notification not found")
        
        return None
    except Exception as e:
        logger.error(f"Error marking notification as read: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to mark notification as read")

@router.post("/read-all")
@login_required
async def mark_all_notifications_as_read(request: Request):
    try:
        user_id = request.state.user["username"]
        notifications = await notification_manager.get_user_notifications(user_id=user_id, unread_only=True)
        
        for notification in notifications:
            await notification_manager.mark_as_read(notification["id"], user_id)
        
        return None
    except Exception as e:
        logger.error(f"Error marking all notifications as read: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to mark all notifications as read")

@router.delete("/{notification_id}")
@login_required
async def delete_notification(request: Request, notification_id: str):
    try:
        user_id = request.state.user["username"]
        success = await notification_manager.delete_notification(notification_id, user_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Notification not found")
        
        return None
    except Exception as e:
        logger.error(f"Error deleting notification: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete notification")

@router.delete("")
@login_required
async def clear_all_notifications(request: Request):
    try:
        user_id = request.state.user["username"]
        notifications = await notification_manager.get_user_notifications(user_id=user_id)
        
        for notification in notifications:
            await notification_manager.delete_notification(notification["id"], user_id)
        
        return None
    except Exception as e:
        logger.error(f"Error clearing notifications: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to clear notifications") 