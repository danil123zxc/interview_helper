from typing import Literal, Optional, List, Union
from pydantic import Field, BaseModel 
from langchain_core.documents import Document  

class ContextSchema(BaseModel):
    role: str = Field(..., description="The job role the user is applying for")
    resume: Union[str, Document, List[Union[Document, str]]] = Field(None, description="The user's resume or background information")
    experience_level: Literal["intern", "junior", "mid", "senior", "lead"] = Field(default="intern", description="The user's experience level")
    years_of_experience: Optional[int] = Field(default=None, description="The number of years of experience the user has")      
