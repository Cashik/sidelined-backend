from typing import List, Optional
from pydantic import BaseModel
from src import enums


class SocialMedia(BaseModel):
    name: Optional[str] = None
    social_id: str
    social_login: str
    social_media_type: enums.ProjectAccountStatusType
    


class Project(BaseModel):
    name: str
    description: Optional[str] = None
    url: Optional[str] = None
    keywords: List[str] = []
    social_media: List[SocialMedia] = []



caldera = Project(
    name="Caldera",
    url="https://caldera.xyz",
    keywords=["caldera", "caldera.com", "Caldera"],
    social_media=[
        SocialMedia(
            name="Caldera",
            social_id="1502716760486678528",
            social_login="Calderaxyz",
            social_media_type=enums.ProjectAccountStatusType.MEDIA
        ),
        SocialMedia(
            name="Matt Katz (CEO)",
            social_id="853899428012244992",
            social_login="0xkatz",
            social_media_type=enums.ProjectAccountStatusType.FOUNDER
        ),
        SocialMedia(
            name="Parker Jou (CTO)",
            social_id="1441265308455501824",
            social_login="theappletucker",
            social_media_type=enums.ProjectAccountStatusType.FOUNDER
        ),
    ]
)

projects_all = [
    caldera
]
