from pydantic import BaseModel, field_validator

from .schemas import ImaginationStatus


# Required data
class EnginesDetails(BaseModel):
    id: str
    prompt: str
    status: ImaginationStatus
    percentage: int | None = None
    result: dict | None = None

    @field_validator("percentage", mode="before")
    def validate_percentage(cls, value):
        if value is None:
            return -1
        if isinstance(value, str):
            return int(value.replace("%", ""))
        if value < -1:
            return -1
        if value > 100:
            return 100
        return value


class Engine:
    def __init__(self, item, engine) -> None:
        self.item = item
        self.engine = engine

    # Get Result from service(client / API)
    async def result(self, **kwargs) -> EnginesDetails:
        pass

    # Validate schema
    def validate(self, data: BaseModel) -> tuple[bool, str | None]:
        return True, None

    # Send request to service(client / API)
    async def _request(self, **kwargs) -> EnginesDetails:
        pass

    # Get property from item meta_data
    # item.meta_data: It is a response sent from the service
    def _get_data(self, name: str, **kwargs):
        value = (self.item.meta_data or {}).get(name, None)
        if value is None:
            raise ValueError(f"Missing value {name}")
        return value

    # Get current request service(Convert service status to ImaginationStatus)
    def _status(self, status: str) -> ImaginationStatus:
        pass

    async def imagine(self, **kwargs):
        response = await self._request(**kwargs)
        return response

    # Convert service response to EnginesDetails
    async def _result_to_details(self, res) -> EnginesDetails:
        pass
