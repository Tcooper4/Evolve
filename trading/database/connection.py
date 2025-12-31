"""
Database Connection Management

Handles SQLAlchemy database connections, session management, and initialization.
"""

import logging
import os
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

logger = logging.getLogger(__name__)

# Global variables
_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker] = None


def get_database_url() -> str:
    """
    Get database URL from environment variables.
    
    Returns:
        Database connection URL
    """
    # Try PostgreSQL first
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "trading")
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD", "")
    
    # If no password, try SQLite as fallback
    if not db_password:
        sqlite_path = os.getenv("SQLITE_PATH", "data/trading.db")
        logger.info(f"Using SQLite database at {sqlite_path}")
        return f"sqlite:///{sqlite_path}"
    
    # PostgreSQL connection
    db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    logger.info(f"Using PostgreSQL database at {db_host}:{db_port}/{db_name}")
    return db_url


def create_database_engine(database_url: Optional[str] = None) -> Engine:
    """
    Create SQLAlchemy database engine.
    
    Args:
        database_url: Optional database URL (defaults to get_database_url())
        
    Returns:
        SQLAlchemy engine
    """
    global _engine
    
    if _engine is not None:
        return _engine
    
    if database_url is None:
        database_url = get_database_url()
    
    # Engine configuration
    connect_args = {}
    if database_url.startswith("sqlite"):
        # SQLite-specific configuration
        connect_args = {"check_same_thread": False}
        engine = create_engine(
            database_url,
            connect_args=connect_args,
            poolclass=QueuePool,
            pool_pre_ping=True,
            echo=False,  # Set to True for SQL debugging
        )
    else:
        # PostgreSQL configuration
        engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=False,  # Set to True for SQL debugging
        )
    
    # Enable foreign key constraints for SQLite
    if database_url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    
    _engine = engine
    logger.info("Database engine created successfully")
    return engine


def get_session_factory() -> sessionmaker:
    """
    Get or create session factory.
    
    Returns:
        SQLAlchemy sessionmaker
    """
    global _SessionLocal
    
    if _SessionLocal is not None:
        return _SessionLocal
    
    engine = create_database_engine()
    _SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine
    )
    return _SessionLocal


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Get database session context manager.
    
    Yields:
        SQLAlchemy session
        
    Example:
        with get_db_session() as session:
            # Use session
            pass
    """
    session_factory = get_session_factory()
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_database(create_tables: bool = True) -> None:
    """
    Initialize database and create tables.
    
    Args:
        create_tables: Whether to create tables if they don't exist
    """
    try:
        from trading.database.models import Base
        
        engine = create_database_engine()
        
        if create_tables:
            # Create all tables
            Base.metadata.create_all(bind=engine)
            logger.info("Database tables created successfully")
        else:
            logger.info("Database connection initialized (tables not created)")
            
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def get_engine() -> Optional[Engine]:
    """
    Get the global database engine.
    
    Returns:
        SQLAlchemy engine or None if not initialized
    """
    return _engine


def close_database() -> None:
    """
    Close database connections.
    """
    global _engine, _SessionLocal
    
    if _engine:
        _engine.dispose()
        _engine = None
    
    _SessionLocal = None
    logger.info("Database connections closed")

