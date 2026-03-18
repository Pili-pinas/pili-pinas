"""Pydantic request/response models for the Pili-Pinas API."""

from pydantic import BaseModel, ConfigDict, Field


class QueryRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "question": "What education bills has the Senate passed this year?",
            "source_type": "bill",
            "top_k": 5,
        }
    })

    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Question in English, Filipino, or Taglish.",
    )
    source_type: str | None = Field(
        None,
        description=(
            "Filter results to a specific document type. "
            "Options: `bill`, `law`, `news`, `profile`, `saln`, `election`. "
            "Omit to search across all types."
        ),
    )
    top_k: int = Field(
        5,
        ge=1,
        le=20,
        description="Number of document chunks to retrieve before generating the answer.",
    )


class SourceDoc(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "title": "Senate Bill No. 2765 — Basic Education Funding Act",
            "url": "https://senate.gov.ph/lis/bill_res.aspx?congress=19&q=SB02765",
            "source": "senate.gov.ph",
            "date": "2025-01-10",
            "score": 0.923,
        }
    })

    title: str = Field(description="Document title.")
    url: str = Field(description="Source URL — voters can verify the claim here.")
    source: str = Field(description="Domain of the source (e.g. senate.gov.ph).")
    date: str = Field(description="Publication date (YYYY-MM-DD).")
    score: float = Field(description="Similarity score (0–1). Higher = more relevant.")


class QueryResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "answer": (
                "The Senate passed the Basic Education Funding Act (SB 2765) on January 10, 2025, "
                "increasing per-pupil spending by 30% [senate.gov.ph]."
            ),
            "sources": [{
                "title": "Senate Bill No. 2765 — Basic Education Funding Act",
                "url": "https://senate.gov.ph/lis/bill_res.aspx?congress=19&q=SB02765",
                "source": "senate.gov.ph",
                "date": "2025-01-10",
                "score": 0.923,
            }],
            "query": "What education bills has the Senate passed this year?",
            "chunks_used": 1,
        }
    })

    answer: str = Field(description="AI-generated answer with inline source citations.")
    sources: list[SourceDoc] = Field(description="Documents used to generate the answer.")
    query: str = Field(description="The original question.")
    chunks_used: int = Field(description="Number of document chunks passed to the LLM.")


class ScrapeRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "sources": ["news"],
            "max_news": 20,
            "embed": True,
        }
    })

    sources: list[str] | None = Field(
        None,
        description=(
            "Data sources to scrape. `null` scrapes all sources. "
            "Valid values: `senate_bills`, `senators`, `gazette`, "
            "`house_bills`, `house_members`, `comelec`, `news`."
        ),
    )
    congresses: list[int] | None = Field(
        None,
        description=(
            "Congress numbers to scrape for bill sources. "
            "Defaults to current congress (20). "
            "Pass multiple for a backfill, e.g. `[17, 18, 19, 20]`."
        ),
    )
    election_years: list[int] | None = Field(
        None,
        description=(
            "Election years to scrape for COMELEC. "
            "Defaults to current year (2025). "
            "Pass multiple for a backfill, e.g. `[2016, 2019, 2022, 2025]`."
        ),
    )
    max_pages: int = Field(
        3,
        ge=1,
        le=2000,
        description="Maximum items to scrape per congress session for bill scrapers.",
    )
    max_news: int = Field(
        20,
        ge=1,
        le=200,
        description="Maximum articles to fetch per news source.",
    )
    max_laws: int = Field(
        50,
        ge=1,
        le=5000,
        description="Maximum laws to fetch from the Official Gazette.",
    )
    embed: bool = Field(
        True,
        description=(
            "If `true`, automatically runs the embedding pipeline after ingestion "
            "so new documents are immediately searchable."
        ),
    )
    resume: bool = Field(
        False,
        description=(
            "If `true`, resumes a previously interrupted scrape job by skipping "
            "sources that were already completed in the last run."
        ),
    )


class ScrapeJobStatus(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "job_id": "a3f1c2d4-1234-5678-abcd-ef0123456789",
            "status": "done",
            "started_at": "2025-01-15T10:00:00.000000",
            "finished_at": "2025-01-15T10:04:32.123456",
            "stats": {
                "ingestion": {
                    "sources": ["news"],
                    "counts": {"news_articles": 87},
                    "total_chunks": 87,
                },
                "embedding": {"news_articles": 87},
            },
            "error": None,
        }
    })

    job_id: str = Field(description="Unique job identifier.")
    status: str = Field(description="Job state: `pending` → `running` → `done` | `failed`.")
    started_at: str | None = Field(None, description="ISO timestamp when the job started.")
    finished_at: str | None = Field(None, description="ISO timestamp when the job finished.")
    stats: dict | None = Field(None, description="Ingestion and embedding counts, populated on completion.")
    error: str | None = Field(None, description="Error message if status is `failed`.")
