from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

from app.core.config import settings

# Fixed naming convention so Alembic autogenerate produces stable constraint
# names instead of DB-assigned ones that differ across environments.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


IS_SQLITE = settings.database_url.startswith("sqlite")

# Every streaming step holds its request session for the whole render (minutes, for video), so a
# handful of concurrent runs can hold a session each for a long time. On a real server this needs a
# generous QueuePool (pre-ping/recycle guard against an idle pooled connection getting dropped by
# the server or a NAT before we reuse it, which surfaces as "server closed the connection
# unexpectedly"). On SQLite a connection is just a cheap file handle — NullPool (no cap, no reuse)
# instead of leaving SQLAlchemy's QueuePool defaults (pool_size=5, max_overflow=10 = 15 total) in
# place, which silently caps concurrent long-held streaming sessions and times out the 16th request.
_POOL_ARGS = {"poolclass": NullPool} if IS_SQLITE else {
    "pool_pre_ping": True,
    "pool_recycle": 1800,
    "pool_size": 10,
    "max_overflow": 20,
}

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if IS_SQLITE else {},
    **_POOL_ARGS,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
