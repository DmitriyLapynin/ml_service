from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ai_domain.rag.file_loader import load_text_from_file
from ai_domain.rag.loaders.types import DocumentLike


def _doc(text: str, metadata: Dict[str, Any]) -> DocumentLike:
    return {"text": text, "metadata": metadata}


def _load_pdf(path: Path) -> List[DocumentLike]:
    loader = None
    try:
        from langchain_community.document_loaders import PyMuPDFLoader  # type: ignore

        loader = PyMuPDFLoader(str(path))
    except Exception:
        loader = None

    if loader is None:
        try:
            from langchain_community.document_loaders import PyPDFLoader  # type: ignore

            loader = PyPDFLoader(str(path))
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("PyMuPDFLoader or PyPDFLoader is required for PDF loading") from exc

    docs = loader.load()
    out: List[DocumentLike] = []
    for doc in docs:
        metadata = dict(getattr(doc, "metadata", {}) or {})
        metadata.setdefault("source", str(path))
        if "page_number" not in metadata and "page" in metadata:
            metadata["page_number"] = metadata.get("page")
        out.append(_doc(doc.page_content, metadata))
    return out


def _load_excel_with_pandas(path: Path) -> List[DocumentLike]:
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pandas is required for Excel loading") from exc

    sheets = pd.read_excel(path, sheet_name=None)
    out: List[DocumentLike] = []
    for sheet_name, frame in sheets.items():
        text = frame.to_csv(index=False)
        out.append(
            _doc(
                text,
                {
                    "source": str(path),
                    "sheet_name": sheet_name,
                    "row_count": int(frame.shape[0]),
                    "column_count": int(frame.shape[1]),
                },
            )
        )
    return out


def _load_excel_with_unstructured(path: Path) -> List[DocumentLike]:
    try:
        from langchain_community.document_loaders import UnstructuredExcelLoader  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("UnstructuredExcelLoader is required for Excel loading") from exc

    loader = UnstructuredExcelLoader(str(path))
    docs = loader.load()
    out: List[DocumentLike] = []
    for doc in docs:
        metadata = dict(getattr(doc, "metadata", {}) or {})
        metadata.setdefault("source", str(path))
        out.append(_doc(doc.page_content, metadata))
    return out


def load_documents(path: str | Path) -> List[DocumentLike]:
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix in {".txt", ".md", ".json", ".jsonl"}:
        return [_doc(load_text_from_file(file_path), {"source": str(file_path)})]

    if suffix == ".pdf":
        return _load_pdf(file_path)

    if suffix in {".xlsx", ".xls"}:
        try:
            return _load_excel_with_pandas(file_path)
        except Exception:
            return _load_excel_with_unstructured(file_path)

    return [_doc(load_text_from_file(file_path), {"source": str(file_path)})]
