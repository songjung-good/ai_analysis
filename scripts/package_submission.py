"""Package explicit submission artifacts only. Never traverses the source tree or .env."""
import argparse
import hashlib
import json
import os
import re
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse


def package_submission(*, repo_url: str, report: Path, traces: list[Path], campus: str,
                       class_name: str, names: str, run_id: str, output_dir: Path) -> Path:
    import pymupdf

    url = urlparse(repo_url)
    if url.scheme != "https" or url.hostname != "github.com" or url.username or url.password or len(url.path.strip("/").split("/")) < 2:
        raise ValueError("repo_url must be an HTTPS GitHub repository/branch link")
    for label, value in (("campus", campus), ("class_name", class_name), ("names", names)):
        if not re.fullmatch(r"[\w가-힣+-]+", value):
            raise ValueError(f"Invalid filename component: {label}")
    if not run_id.strip():
        raise ValueError("run_id is required to identify the actual execution")
    report = Path(report)
    if report.suffix.lower() != ".pdf":
        raise ValueError("report must be a PDF")
    with pymupdf.open(report) as pdf:
        text = "\n".join(page.get_text() for page in pdf)
        if not 1 <= pdf.page_count <= 10 or "SUMMARY" not in text or "REFERENCE" not in text:
            raise ValueError("Report must have SUMMARY, REFERENCE, and at most 10 pages")
        if "/ SAMPLE" in text or "레이아웃 검증용" in text:
            raise ValueError("A sample preview cannot be submitted as a final report")
        if "FINAL / PASS" not in text and "FINAL / NOT PASSED" not in text:
            raise ValueError("Finalize the evaluated report before packaging; a draft is not a final report")
        required = ("기술 성숙도 관점", "시장성 관점", "이해관계자 관점", "도메인 적용 관점", "RISKS")
        if any(heading not in text for heading in required):
            raise ValueError("Report is missing required perspectives or risks")
        if run_id not in re.findall(r"^trace_id: (.+)$", text, flags=re.MULTILINE):
            raise ValueError("Report trace_id does not match the supplied run_id")
        page_count = pdf.page_count
    if not traces:
        raise ValueError("At least one real LangSmith tracing PNG is required")
    artifacts = {"report.pdf": report.read_bytes()}
    from PIL import Image
    for index, trace in enumerate(traces, 1):
        path = Path(trace)
        with Image.open(path) as image:
            if image.format != "PNG":
                raise ValueError("Trace images must be PNG")
            image.verify()
        artifacts[f"tracing-{index}.png"] = path.read_bytes()
    artifacts["GITHUB_LINK.txt"] = (repo_url + "\n").encode()
    manifest = {"run_id": run_id, "report_pages": page_count, "github_url": repo_url,
                "files": {name: hashlib.sha256(data).hexdigest() for name, data in artifacts.items()},
                "note": "File validation only; confirm the PDF and real LangSmith traces describe the same evaluated run."}
    artifacts["manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=2).encode()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"Agent_{campus}_{class_name}_{names}.zip"
    if destination.exists():
        raise FileExistsError(f"Submission already exists: {destination}")
    fd, temporary_name = tempfile.mkstemp(prefix=".submission-", suffix=".zip", dir=output_dir)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in artifacts.items():
                archive.writestr(name, data)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination.resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo-url", "campus", "class-name", "names", "run-id"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--traces", type=Path, nargs="+", required=True, help="Actual trace PNGs in execution order")
    parser.add_argument("--output-dir", type=Path, default=Path("output/submission"))
    print(package_submission(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
