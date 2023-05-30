from typing import List, Union

from pydantic import BaseModel, Field


class Context(BaseModel):
    task: str = Field(...)
    domain: str = Field(...)
    context: Union[str, List[str]] = Field(...)


class FIBContext(BaseModel):
    context: str
    num_blank: int


class HistoryTextbookContext(BaseModel):
    task: str
    section: str


class FeedBack(BaseModel):
    task: str = Field(...)
    domain: str = Field(...)
    results: List[dict] = Field(...)
    time: str = Field(...)
