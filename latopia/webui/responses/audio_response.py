import os
from typing import *

from fastapi import HTTPException, status
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask


class AudioResponse(StreamingResponse):
    def __init__(
        self,
        path: str,
        status_code: int = 200,
        headers: Mapping[str, str] | None = None,
        media_type: str | None = None,
        background: BackgroundTask | None = None,
    ):
        self.file_path = path
        self.file_size = os.stat(path).st_size
        self.range_header = headers.get("range")

        headers = {
            "content-type": media_type,
            "accept-ranges": "bytes",
            "content-encoding": "identity",
            "content-length": str(self.file_size),
            "access-control-expose-headers": (
                "content-type, accept-ranges, content-length, "
                "content-range, content-encoding"
            ),
        }
        self.start = 0
        self.end = self.file_size - 1
        status_code = status.HTTP_200_OK

        if self.range_header is not None:
            self.start, self.end = self.get_range_header()
            size = self.end - self.start + 1
            headers["content-length"] = str(size)
            headers["content-range"] = f"bytes {self.start}-{self.end}/{self.file_size}"
            status_code = status.HTTP_206_PARTIAL_CONTENT

        super().__init__(
            self.send_bytes_range_requests(),
            status_code,
            headers,
            media_type,
            background,
        )

    def get_range_header(self) -> tuple[int, int]:
        def invalid_range():
            return HTTPException(
                status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
                detail=f"Invalid request range (Range:{self.range_header!r})",
            )

        try:
            h = self.range_header.replace("bytes=", "").split("-")
            start = int(h[0]) if h[0] != "" else 0
            end = int(h[1]) if h[1] != "" else self.file_size - 1
        except ValueError:
            raise invalid_range()

        if start > end or start < 0 or end > self.file_size - 1:
            raise invalid_range()
        return start, end

    def send_bytes_range_requests(self, chunk_size: int = 10_000):
        with open(self.file_path, "rb") as f:
            f.seek(self.start)
            while (pos := f.tell()) <= self.end:
                read_size = min(chunk_size, self.end + 1 - pos)
                yield f.read(read_size)
