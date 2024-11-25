from typing import Literal

from metisai.async_metis import AsyncMetisBot
from metisai.metistypes import TaskResult

from server.config import Settings

from .engine import Engine, EnginesDetails
from .schemas import ImaginationEngines, ImaginationStatus


class DalleDetails(EnginesDetails):
    session_id: str


class Dalle(Engine):
    def __init__(self, item, **kwargs) -> None:
        super().__init__(item, **kwargs)
        if item:
            self.client = AsyncMetisBot(
                api_key=Settings.METIS_API_KEY,
                bot_id=ImaginationEngines.dalle.metis_bot_id,
            )

    async def result(self, **kwargs) -> DalleDetails:
        id = self._get_data("id")
        session_id = self._get_data("session_id")
        task = await self.client.retrieve_async_task(session_id, id)
        return await self._result_to_details(task, task_id=id, session_id=session_id)

    async def _request(self, **kwargs) -> DalleDetails:
        self.item.prompt += f", in a {self.item.aspect_ratio} aspect ratio"
        session = await self.client.create_session()
        res = await self.client.send_message_async(
            session, f"/imagine {self.item.prompt}"
        )
        task = await self.client.retrieve_async_task(session, res)
        return await self._result_to_details(
            task, task_id=res.taskId, session_id=session.id
        )

    def _status(self, status: Literal["RUNNING", "FINISHED", "FAILED"]):
        return {
            "RUNNING": ImaginationStatus.processing,
            "FINISHED": ImaginationStatus.completed,
            "FAILED": ImaginationStatus.error,
        }.get(status, ImaginationStatus.error)

    def validate(self, data):
        aspect_ratio_valid = data.aspect_ratio in self.engine.supported_aspect_ratios
        message = (
            f"aspect_ratio must be one of them {self.engine.supported_aspect_ratios}"
            if not aspect_ratio_valid
            else None
        )
        return aspect_ratio_valid, message

    async def _result_to_details(self, task: TaskResult, **kwargs):
        task_data = task.__dict__.copy()
        session_id = kwargs.get("session_id")
        task_id = kwargs.get("task_id")
        status = self._status(task.status)
        task_data.pop("status", None)
        return DalleDetails(
            **task_data,
            id=task_id,
            session_id=session_id,
            status=status,
            prompt=self.item.prompt,
            result=(
                {"uri": task.message.attachments[0].content} if task.message else None
            ),
        )
