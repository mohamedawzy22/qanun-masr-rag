import os
import random
import re
import string

from utils.logger import get_logger

logger = get_logger(__name__)


class ProjectStorage:
    def __init__(self):
        self.base_dir = os.path.dirname(
            os.path.dirname(
                os.path.dirname(__file__),
            ),
        )

        self.files_dir = os.path.join(
            self.base_dir,
            "asset",
            "files",
        )

        logger.info(
            "Project storage initialized: files_dir=%s",
            self.files_dir,
        )

    def get_project_path(
        self,
        project_id: str,
    ) -> str:
        project_dir = os.path.join(
            self.files_dir,
            str(project_id),
        )

        os.makedirs(
            project_dir,
            exist_ok=True,
        )

        return project_dir

    @staticmethod
    def sanitize_file_name(
        file_name: str,
    ) -> str:
        file_name = file_name.strip()

        file_name = re.sub(
            r"[^\w.\- ]",
            "",
            file_name,
        )

        file_name = re.sub(
            r"\s+",
            "_",
            file_name,
        )

        return file_name or "uploaded_file"

    def generate_unique_file_name(
        self,
        file_name: str,
    ) -> str:
        """Generate a unique asset name without creating local directories."""
        random_key = self._generate_random_string()

        return f"{random_key}_{file_name}"

    def generate_unique_filepath(
        self,
        project_id: str,
        file_name: str,
    ) -> tuple[str, str]:
        project_path = self.get_project_path(
            project_id=project_id,
        )

        while True:
            random_key = self._generate_random_string()

            generated_file_name = f"{random_key}_{file_name}"

            file_path = os.path.join(
                project_path,
                generated_file_name,
            )

            if not os.path.exists(file_path):
                return (
                    file_path,
                    generated_file_name,
                )

    @staticmethod
    def _generate_random_string(
        length: int = 12,
    ) -> str:
        return "".join(
            random.choices(
                string.ascii_lowercase + string.digits,
                k=length,
            )
        )
