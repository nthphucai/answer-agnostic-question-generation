from typing import List, Optional, Union

from pydantic import BaseModel, Field


class ContextAnswer(BaseModel):
    context: str = Field(...)
    answer: str = Field(...)


class Context(BaseModel):
    context: str = Field(...)


class QuestgenRequestItem(BaseModel):
    task: str = Field(
        description="'simple-question' -> Return the question and the answer ; 'multiple-choices' -> Return the question and four options for the question (one is the answer)"
    )
    context: Union[str, List[str]] = Field(...)


class BookRequestItem(BaseModel):
    task: str = Field(
        default="simple-question",
        description="'simple-question' -> Return the question and the answer ; 'multiple-choices' -> Return the question and four options for the question (one is the answer)",
    )
    section: str = Field(
        default="Lớp 10>Xã hội nguyên thủy>Sự xuất hiện loài người và bầy người nguyên thủy>Sự xuất hiện loài người và đời sống bầy người nguyên thủy",
        description="Path to context",
    )


class FIBRequestItem(BaseModel):
    context: str = Field(
        description="The blank is created on this context, pass a empty string '' to get a random context",
        default="",
    )
    num_blank: int = Field(description="Number of blanks is created", default=6)


class FIBWithGivenWordItem(BaseModel):
    context: str = Field(
        description="The blank is created on this context, pass a empty string '' to get a random context",
        default="",
    )
    word: Optional[Union[str, List]] = Field(
        description="If specific word(s) is not passed, generate all possible blank as default",
        default=None,
    )


class QARequestItem(BaseModel):
    question: str = Field(...)


class FromFileRequestItem(BaseModel):
    task: str


class FeedBack(BaseModel):
    task: str
    domain: str
    results: List[dict] = Field(...)
    time: float = Field(...)


class UserFeedback(BaseModel):
    task: str = Field(...)
    domain: str = Field(...)
    data: dict = Field(...)
    time: float = Field(default=0)
    label: bool = Field(...)
    rating: int = Field(default=0)
    comment: str = Field(default="")
