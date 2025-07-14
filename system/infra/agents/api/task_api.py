from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from system.infra.agents.config.config import load_config
from system.infra.agents.core.orchestrator import Orchestrator

app = FastAPI(
    title="Automation Task API",
    description="API for managing automation tasks",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response


class TaskCreate(BaseModel):
    name: str = Field(..., description="Name of the task")
    task_type: str = Field(
        ..., description="Type of task (data_collection, model_training, etc.)"
    )
    priority: int = Field(default=1, description="Task priority (1-5)")
    parameters: Dict[str, Any] = Field(default={}, description="Task parameters")
    dependencies: List[str] = Field(
        default_factory=list, description="List of task IDs this task depends on"
    )


class TaskResponse(BaseModel):
    task_id: str
    name: str
    task_type: str
    status: str
    priority: int
    parameters: Dict[str, Any]
    dependencies: List[str]
    created_at: datetime
    updated_at: datetime
    error_message: Optional[str] = None


class TaskUpdate(BaseModel):
    status: Optional[str] = None
    priority: Optional[int] = None
    parameters: Optional[Dict[str, Any]] = None
    dependencies: Optional[List[str]] = None


# Dependency for getting orchestrator instance


def get_orchestrator():
    config = load_config()
    orchestrator = Orchestrator(config)
    return orchestrator


# Task creation endpoint


@app.post("/tasks", response_model=TaskResponse)
async def create_task(
    task: TaskCreate, orchestrator: Orchestrator = Depends(get_orchestrator)
):
    """Create a new task."""
    try:
        task_id = await orchestrator.create_task(
            name=task.name,
            task_type=task.task_type,
            priority=task.priority,
            parameters=task.parameters,
            dependencies=task.dependencies,
        )

        created_task = await orchestrator.get_task(task_id)
        return TaskResponse(
            task_id=created_task.task_id,
            name=created_task.name,
            task_type=created_task.task_type,
            status=created_task.status,
            priority=created_task.priority,
            parameters=created_task.parameters,
            dependencies=created_task.dependencies,
            created_at=created_task.created_at,
            updated_at=created_task.updated_at,
            error_message=created_task.error_message,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Get task by ID


@app.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: str = Path(..., description="The ID of the task to retrieve"),
    orchestrator: Orchestrator = Depends(get_orchestrator),
):
    """Get a specific task by ID."""
    try:
        task = await orchestrator.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return TaskResponse(
            task_id=task.task_id,
            name=task.name,
            task_type=task.task_type,
            status=task.status,
            priority=task.priority,
            parameters=task.parameters,
            dependencies=task.dependencies,
            created_at=task.created_at,
            updated_at=task.updated_at,
            error_message=task.error_message,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# List tasks with filtering


@app.get("/tasks", response_model=List[TaskResponse])
async def list_tasks(
    status: Optional[str] = Query(None, description="Filter by task status"),
    task_type: Optional[str] = Query(None, description="Filter by task type"),
    orchestrator: Orchestrator = Depends(get_orchestrator),
):
    """List all tasks with optional filtering."""
    try:
        tasks = await orchestrator.get_tasks()

        # Apply filters
        if status:
            tasks = [t for t in tasks if t.status == status]
        if task_type:
            tasks = [t for t in tasks if t.task_type == task_type]

        return [
            TaskResponse(
                task_id=t.task_id,
                name=t.name,
                task_type=t.task_type,
                status=t.status,
                priority=t.priority,
                parameters=t.parameters,
                dependencies=t.dependencies,
                created_at=t.created_at,
                updated_at=t.updated_at,
                error_message=t.error_message,
            )
            for t in tasks
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Update task


@app.patch("/tasks/{task_id}", response_model=TaskResponse)
async def update_task(
    task_update: TaskUpdate,
    task_id: str = Path(..., description="The ID of the task to update"),
    orchestrator: Orchestrator = Depends(get_orchestrator),
):
    """Update a task's properties."""
    try:
        task = await orchestrator.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        # Update task properties
        if task_update.status:
            task.status = task_update.status
        if task_update.priority:
            task.priority = task_update.priority
        if task_update.parameters:
            task.parameters.update(task_update.parameters)
        if task_update.dependencies:
            task.dependencies = task_update.dependencies

        await orchestrator.update_task(task)

        return TaskResponse(
            task_id=task.task_id,
            name=task.name,
            task_type=task.task_type,
            status=task.status,
            priority=task.priority,
            parameters=task.parameters,
            dependencies=task.dependencies,
            created_at=task.created_at,
            updated_at=task.updated_at,
            error_message=task.error_message,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Delete task


@app.delete("/tasks/{task_id}")
async def delete_task(
    task_id: str = Path(..., description="The ID of the task to delete"),
    orchestrator: Orchestrator = Depends(get_orchestrator),
):
    """Delete a task."""
    try:
        task = await orchestrator.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        await orchestrator.delete_task(task_id)
        return {"message": f"Task {task_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Execute task


@app.post("/tasks/{task_id}/execute")
async def execute_task(
    task_id: str = Path(..., description="The ID of the task to execute"),
    orchestrator: Orchestrator = Depends(get_orchestrator),
):
    """Execute a specific task."""
    try:
        task = await orchestrator.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        await orchestrator.execute_task(task)
        return {"message": f"Task {task_id} execution started"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Get task metrics


@app.get("/tasks/{task_id}/metrics")
async def get_task_metrics(
    task_id: str = Path(..., description="The ID of the task to get metrics for"),
    orchestrator: Orchestrator = Depends(get_orchestrator),
):
    """Get metrics for a specific task."""
    try:
        task = await orchestrator.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        metrics = await orchestrator.get_task_metrics(task_id)
        return metrics
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Get task dependencies


@app.get("/tasks/{task_id}/dependencies")
async def get_task_dependencies(
    task_id: str = Path(..., description="The ID of the task to get dependencies for"),
    orchestrator: Orchestrator = Depends(get_orchestrator),
):
    """Get dependencies for a specific task."""
    try:
        task = await orchestrator.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        dependencies = await orchestrator.get_task_dependencies(task_id)
        return {"dependencies": dependencies}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
