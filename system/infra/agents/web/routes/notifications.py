import logging
from datetime import datetime
from typing import Optional

from automation.notifications.notification_manager import (
    NotificationManager,
    NotificationPriority,
)
from automation.web.middleware import login_required
from automation.web.websocket import WebSocketHandler
from fastapi import APIRouter, HTTPException, Request, WebSocket

# Global notification manager instance
notif_manager_instance: NotificationManager = None

# Initialize logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/notifications", tags=["notifications"])

# Initialize WebSocket handler
websocket_handler_instance: WebSocketHandler = None


def init_routes(
    notification_manager: NotificationManager, ws_handler: WebSocketHandler
):
    global notif_manager_instance, websocket_handler_instance
    notif_manager_instance = notification_manager
    websocket_handler_instance = ws_handler

    return {
        "success": True,
        "message": "Initialization completed",
        "timestamp": datetime.now().isoformat(),
    }


@router.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    # Optionally validate user_id or perform auth here
    await websocket_handler_instance.handle_websocket(websocket, user_id)


@router.get("")
@login_required
async def get_notifications(
    request: Request,
    limit: int = 10,
    offset: int = 0,
    unread_only: bool = False,
    priority: Optional[NotificationPriority] = None,
    sort_by: str = "created_at",
    sort_order: str = "desc",
):
    try:
        user_id = request.state.user["username"]

        # Get all notifications first
        notifications = await notif_manager_instance.get_user_notifications(
            user_id=user_id,
            limit=1000,
            offset=0,
            unread_only=unread_only,  # Get all to filter
        )

        # Filter by priority if specified
        if priority:
            notifications = [
                n for n in notifications if n.get("priority") == priority.value
            ]

        # Sort notifications
        reverse = sort_order.lower() == "desc"
        if sort_by == "created_at":
            notifications.sort(key=lambda x: x.get("created_at", ""), reverse=reverse)
        elif sort_by == "priority":
            # Sort by priority (HIGH > MEDIUM > LOW)
            priority_order = {
                NotificationPriority.HIGH: 3,
                NotificationPriority.MEDIUM: 2,
                NotificationPriority.LOW: 1,
            }
            notifications.sort(
                key=lambda x: priority_order.get(
                    NotificationPriority(x.get("priority", "LOW")), 0
                ),
                reverse=reverse,
            )
        elif sort_by == "read":
            notifications.sort(key=lambda x: x.get("read", False), reverse=reverse)

        # Apply pagination
        total_count = len(notifications)
        notifications = notifications[offset : offset + limit]

        return {
            "notifications": notifications,
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total_count,
        }
    except Exception as e:
        logger.error(f"Error getting notifications: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get notifications")


@router.get("/priority/{priority}")
@login_required
async def get_notifications_by_priority(
    request: Request,
    priority: NotificationPriority,
    limit: int = 10,
    offset: int = 0,
    unread_only: bool = False,
    sort_by: str = "created_at",
    sort_order: str = "desc",
):
    """Get notifications filtered by priority level."""
    try:
        user_id = request.state.user["username"]

        # Get notifications by priority
        notifications = await notif_manager_instance.get_user_notifications(
            user_id=user_id,
            limit=1000,
            offset=0,
            unread_only=unread_only,  # Get all to filter
        )

        # Filter by priority
        priority_notifications = [
            n for n in notifications if n.get("priority") == priority.value
        ]

        # Sort notifications
        reverse = sort_order.lower() == "desc"
        if sort_by == "created_at":
            priority_notifications.sort(
                key=lambda x: x.get("created_at", ""), reverse=reverse
            )
        elif sort_by == "read":
            priority_notifications.sort(
                key=lambda x: x.get("read", False), reverse=reverse
            )

        # Apply pagination
        total_count = len(priority_notifications)
        priority_notifications = priority_notifications[offset : offset + limit]

        return {
            "notifications": priority_notifications,
            "priority": priority.value,
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total_count,
        }
    except Exception as e:
        logger.error(f"Error getting notifications by priority: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Failed to get notifications by priority"
        )


@router.get("/priority/{priority}/count")
@login_required
async def get_notification_count_by_priority(
    request: Request, priority: NotificationPriority, unread_only: bool = False
):
    """Get count of notifications by priority level."""
    try:
        user_id = request.state.user["username"]

        # Get notifications by priority
        notifications = await notif_manager_instance.get_user_notifications(
            user_id=user_id,
            limit=1000,
            offset=0,
            unread_only=unread_only,  # Get all to filter
        )

        # Filter by priority
        priority_notifications = [
            n for n in notifications if n.get("priority") == priority.value
        ]

        return {
            "priority": priority.value,
            "count": len(priority_notifications),
            "unread_only": unread_only,
        }
    except Exception as e:
        logger.error(f"Error getting notification count by priority: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Failed to get notification count by priority"
        )


@router.get("/priorities/summary")
@login_required
async def get_notification_priorities_summary(request: Request):
    """Get summary of notifications by priority levels."""
    try:
        user_id = request.state.user["username"]

        # Get all notifications
        notifications = await notif_manager_instance.get_user_notifications(
            user_id=user_id, limit=1000, offset=0, unread_only=False
        )

        # Count by priority
        priority_counts = {}
        unread_counts = {}

        for priority in NotificationPriority:
            priority_notifications = [
                n for n in notifications if n.get("priority") == priority.value
            ]
            priority_counts[priority.value] = len(priority_notifications)

            unread_notifications = [
                n for n in priority_notifications if not n.get("read", False)
            ]
            unread_counts[priority.value] = len(unread_notifications)

        return {
            "total_notifications": len(notifications),
            "priority_counts": priority_counts,
            "unread_counts": unread_counts,
            "priorities": [p.value for p in NotificationPriority],
        }
    except Exception as e:
        logger.error(f"Error getting notification priorities summary: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Failed to get notification priorities summary"
        )


@router.post("/priority/{priority}/read-all")
@login_required
async def mark_all_notifications_by_priority_as_read(
    request: Request, priority: NotificationPriority
):
    """Mark all notifications of a specific priority as read."""
    try:
        user_id = request.state.user["username"]

        # Get notifications by priority
        notifications = await notif_manager_instance.get_user_notifications(
            user_id=user_id, limit=1000, offset=0, unread_only=True
        )

        # Filter by priority and mark as read
        priority_notifications = [
            n for n in notifications if n.get("priority") == priority.value
        ]

        marked_count = 0
        for notification in priority_notifications:
            success = await notif_manager_instance.mark_as_read(
                notification["id"], user_id
            )
            if success:
                marked_count += 1

        return {
            "priority": priority.value,
            "marked_count": marked_count,
            "total_count": len(priority_notifications),
        }
    except Exception as e:
        logger.error(f"Error marking notifications by priority as read: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Failed to mark notifications by priority as read"
        )


@router.delete("/priority/{priority}")
@login_required
async def delete_all_notifications_by_priority(
    request: Request, priority: NotificationPriority
):
    """Delete all notifications of a specific priority."""
    try:
        user_id = request.state.user["username"]

        # Get notifications by priority
        notifications = await notif_manager_instance.get_user_notifications(
            user_id=user_id, limit=1000, offset=0, unread_only=False
        )

        # Filter by priority and delete
        priority_notifications = [
            n for n in notifications if n.get("priority") == priority.value
        ]

        deleted_count = 0
        for notification in priority_notifications:
            success = await notif_manager_instance.delete_notification(
                notification["id"], user_id
            )
            if success:
                deleted_count += 1

        return {
            "priority": priority.value,
            "deleted_count": deleted_count,
            "total_count": len(priority_notifications),
        }
    except Exception as e:
        logger.error(f"Error deleting notifications by priority: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Failed to delete notifications by priority"
        )


@router.post("/{notification_id}/read")
@login_required
async def mark_notification_as_read(request: Request, notification_id: str):
    try:
        user_id = request.state.user["username"]
        success = await notif_manager_instance.mark_as_read(notification_id, user_id)

        if not success:
            raise HTTPException(status_code=404, detail="Notification not found")

        return None
    except Exception as e:
        logger.error(f"Error marking notification as read: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Failed to mark notification as read"
        )


@router.post("/read-all")
@login_required
async def mark_all_notifications_as_read(request: Request):
    try:
        user_id = request.state.user["username"]
        notifications = await notif_manager_instance.get_user_notifications(
            user_id=user_id, unread_only=True
        )

        for notification in notifications:
            await notif_manager_instance.mark_as_read(notification["id"], user_id)

        return None
    except Exception as e:
        logger.error(f"Error marking all notifications as read: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Failed to mark all notifications as read"
        )


@router.delete("/{notification_id}")
@login_required
async def delete_notification(request: Request, notification_id: str):
    try:
        user_id = request.state.user["username"]
        success = await notif_manager_instance.delete_notification(
            notification_id, user_id
        )

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
        notifications = await notif_manager_instance.get_user_notifications(
            user_id=user_id
        )

        for notification in notifications:
            await notif_manager_instance.delete_notification(
                notification["id"], user_id
            )

        return None
    except Exception as e:
        logger.error(f"Error clearing notifications: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to clear notifications")
