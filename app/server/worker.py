import asyncio
import logging

# import pytz
from apps.imagination.worker import update_imagination
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from server.config import Settings

# irst_timezone = pytz.timezone("Asia/Tehran")
logging.getLogger("apscheduler").setLevel(logging.WARNING)


async def worker():
    await update_imagination()
    
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        update_imagination, "interval", seconds=Settings.worker_update_time
    )

    scheduler.start()

    try:
        await asyncio.Event().wait()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        scheduler.shutdown()
