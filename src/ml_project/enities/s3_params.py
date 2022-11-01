from dataclasses import dataclass, field


@dataclass()
class AwsS3Params:
    bucket: str = field(default="dvcstorage")
    path: str = field(default="untracked")
    defaultfile: str = field(default="heart_cleveland_upload.csv")
    defaultout: str = field(default="src/ml_project/assets/raw")
