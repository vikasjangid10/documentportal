from pydantic import BaseModel, Field
from typing import List, Optional,Dict, Any, Union


class Metadata(BaseModel):
    Summary: List[str] = Field(default_factory=list, description="Summary of the document.")
    Title:str
    Author: str
    DateCreated: str
    LastModifiedDate: str
    Publisher: str
    Language: str
    PageCount: Union[int, str]
    SentimentTone: str