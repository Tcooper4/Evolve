import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import asyncio
from pathlib import Path
import json
from pydantic import BaseModel, Field, validator
from cachetools import TTLCache
from ratelimit import limits, sleep_and_retry
import croniter
import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.executors.pool import ThreadPoolExecutor

from ..core.models.task import Task, TaskStatus, TaskPriority, TaskType
from .automation_core import AutomationCore
from .automation_tasks import AutomationTasks
from .automation_workflows import AutomationWorkflows
from .automation_monitoring import AutomationMonitoring

logger = logging.getLogger(__name__)

class ScheduleConfig(BaseModel):
    """Configuration for scheduling."""
    timezone: str = Field(default="UTC")
    max_jobs: int = Field(default=1000)
    max_instances: int = Field(default=3)
    job_defaults: Dict[str, Any] = Field(default={
        "coalesce": True,
        "max_instances": 3,
        "misfire_grace_time": 60
    })
    threadpool_workers: int = Field(default=20)
    cache_size: int = Field(default=1000)
    cache_ttl: int = Field(default=3600)
    
    @validator('timezone')
    def validate_timezone(cls, v):
        try:
            pytz.timezone(v)
            return v
        except pytz.exceptions.UnknownTimeZoneError:
            raise ValueError(f"Invalid timezone: {v}")

class Schedule(BaseModel):
    """Schedule model."""
    id: str
    name: str
    description: str
    task_type: TaskType
    parameters: Dict[str, Any]
    cron_expression: Optional[str] = None
    interval_seconds: Optional[int] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    timezone: str = "UTC"
    enabled: bool = True
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    metadata: Dict[str, Any] = {}
    
    @validator('cron_expression', 'interval_seconds', 'start_date')
    def validate_schedule_type(cls, v, values):
        if not any([values.get('cron_expression'), 
                   values.get('interval_seconds'),
                   values.get('start_date')]):
            raise ValueError("Must specify either cron_expression, interval_seconds, or start_date")
        return v
    
    @validator('cron_expression')
    def validate_cron(cls, v):
        if v:
            try:
                croniter.croniter(v, datetime.now())
            except Exception as e:
                raise ValueError(f"Invalid cron expression: {str(e)}")
        return v

class AutomationScheduler:
    """Scheduling and task management functionality."""
    
    def __init__(
        self,
        core: AutomationCore,
        tasks: AutomationTasks,
        workflows: AutomationWorkflows,
        monitoring: AutomationMonitoring,
        config_path: str = "automation/config/scheduler.json"
    ):
        """Initialize scheduler."""
        self.core = core
        self.tasks = tasks
        self.workflows = workflows
        self.monitoring = monitoring
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.setup_scheduler()
        self.setup_cache()
        self.schedules: Dict[str, Schedule] = {}
        self.lock = asyncio.Lock()
        
    def _load_config(self, config_path: str) -> ScheduleConfig:
        """Load scheduler configuration."""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            return ScheduleConfig(**config_data)
        except Exception as e:
            logger.error(f"Failed to load scheduler config: {str(e)}")
            raise
            
    def setup_logging(self):
        """Configure logging."""
        log_path = Path("automation/logs")
        log_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / "scheduler.log"),
                logging.StreamHandler()
            ]
        )
        
    def setup_scheduler(self):
        """Setup APScheduler."""
        try:
            jobstores = {
                'default': MemoryJobStore()
            }
            
            executors = {
                'default': ThreadPoolExecutor(
                    max_workers=self.config.threadpool_workers
                )
            }
            
            self.scheduler = AsyncIOScheduler(
                jobstores=jobstores,
                executors=executors,
                timezone=pytz.timezone(self.config.timezone),
                job_defaults=self.config.job_defaults
            )
            
        except Exception as e:
            logger.error(f"Failed to setup scheduler: {str(e)}")
            raise
            
    def setup_cache(self):
        """Setup schedule caching."""
        self.cache = TTLCache(
            maxsize=self.config.cache_size,
            ttl=self.config.cache_ttl
        )
        
    @sleep_and_retry
    @limits(calls=100, period=60)
    async def create_schedule(
        self,
        name: str,
        description: str,
        task_type: TaskType,
        parameters: Dict[str, Any],
        cron_expression: Optional[str] = None,
        interval_seconds: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        timezone: str = "UTC",
        metadata: Dict[str, Any] = {}
    ) -> Schedule:
        """Create a new schedule."""
        try:
            schedule = Schedule(
                id=str(len(self.schedules) + 1),
                name=name,
                description=description,
                task_type=task_type,
                parameters=parameters,
                cron_expression=cron_expression,
                interval_seconds=interval_seconds,
                start_date=start_date,
                end_date=end_date,
                timezone=timezone,
                metadata=metadata
            )
            
            # Add to scheduler
            await self._add_to_scheduler(schedule)
            
            self.schedules[schedule.id] = schedule
            return schedule
            
        except Exception as e:
            logger.error(f"Failed to create schedule: {str(e)}")
            raise
            
    async def _add_to_scheduler(self, schedule: Schedule):
        """Add schedule to APScheduler."""
        try:
            if schedule.cron_expression:
                trigger = CronTrigger.from_crontab(
                    schedule.cron_expression,
                    timezone=pytz.timezone(schedule.timezone)
                )
            elif schedule.interval_seconds:
                trigger = IntervalTrigger(
                    seconds=schedule.interval_seconds,
                    timezone=pytz.timezone(schedule.timezone)
                )
            else:
                trigger = DateTrigger(
                    run_date=schedule.start_date,
                    timezone=pytz.timezone(schedule.timezone)
                )
                
            self.scheduler.add_job(
                self._execute_schedule,
                trigger=trigger,
                args=[schedule.id],
                id=schedule.id,
                replace_existing=True,
                end_date=schedule.end_date
            )
            
            # Update next run time
            schedule.next_run = self.scheduler.get_job(schedule.id).next_run_time
            
        except Exception as e:
            logger.error(f"Failed to add schedule to scheduler: {str(e)}")
            raise
            
    async def _execute_schedule(self, schedule_id: str):
        """Execute scheduled task."""
        try:
            schedule = self.schedules.get(schedule_id)
            if not schedule:
                raise ValueError(f"Schedule {schedule_id} not found")
                
            if not schedule.enabled:
                return
                
            # Create and execute task
            task = await self.tasks.schedule_task(
                name=schedule.name,
                description=schedule.description,
                task_type=schedule.task_type,
                parameters=schedule.parameters,
                priority=TaskPriority.NORMAL
            )
            
            # Update schedule
            schedule.last_run = datetime.now()
            schedule.next_run = self.scheduler.get_job(schedule_id).next_run_time
            
            # Record metrics
            await self.monitoring.record_metrics(
                "scheduled_task",
                {
                    "schedule_id": schedule_id,
                    "task_id": task.id,
                    "status": task.status.value
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to execute schedule {schedule_id}: {str(e)}")
            await self.monitoring.record_metrics(
                "scheduled_task_error",
                {
                    "schedule_id": schedule_id,
                    "error": str(e)
                }
            )
            
    async def update_schedule(
        self,
        schedule_id: str,
        **kwargs
    ) -> Schedule:
        """Update existing schedule."""
        try:
            schedule = self.schedules.get(schedule_id)
            if not schedule:
                raise ValueError(f"Schedule {schedule_id} not found")
                
            # Update fields
            for key, value in kwargs.items():
                if hasattr(schedule, key):
                    setattr(schedule, key, value)
                    
            schedule.updated_at = datetime.now()
            
            # Update scheduler
            await self._add_to_scheduler(schedule)
            
            return schedule
            
        except Exception as e:
            logger.error(f"Failed to update schedule: {str(e)}")
            raise
            
    async def delete_schedule(self, schedule_id: str):
        """Delete schedule."""
        try:
            schedule = self.schedules.get(schedule_id)
            if not schedule:
                raise ValueError(f"Schedule {schedule_id} not found")
                
            # Remove from scheduler
            self.scheduler.remove_job(schedule_id)
            
            # Remove from schedules
            del self.schedules[schedule_id]
            
        except Exception as e:
            logger.error(f"Failed to delete schedule: {str(e)}")
            raise
            
    async def get_schedule(self, schedule_id: str) -> Optional[Schedule]:
        """Get schedule by ID."""
        return self.schedules.get(schedule_id)
        
    async def get_schedules(
        self,
        enabled: Optional[bool] = None,
        task_type: Optional[TaskType] = None
    ) -> List[Schedule]:
        """Get schedules with optional filtering."""
        schedules = list(self.schedules.values())
        
        if enabled is not None:
            schedules = [s for s in schedules if s.enabled == enabled]
            
        if task_type:
            schedules = [s for s in schedules if s.task_type == task_type]
            
        return schedules
        
    async def start(self):
        """Start the scheduler."""
        try:
            self.scheduler.start()
            logger.info("Scheduler started")
        except Exception as e:
            logger.error(f"Failed to start scheduler: {str(e)}")
            raise
            
    async def shutdown(self):
        """Shutdown the scheduler."""
        try:
            self.scheduler.shutdown()
            logger.info("Scheduler shutdown")
        except Exception as e:
            logger.error(f"Failed to shutdown scheduler: {str(e)}")
            raise
            
    async def cleanup(self):
        """Cleanup resources."""
        try:
            # Shutdown scheduler
            await self.shutdown()
            
            # Clear caches
            self.cache.clear()
            self.schedules.clear()
            
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            raise 