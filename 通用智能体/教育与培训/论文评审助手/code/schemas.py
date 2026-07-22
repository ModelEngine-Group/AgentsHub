from pydantic import BaseModel


class PaperReviewRequest(BaseModel):
    """论文评审请求"""
    content: str
    paper_type: str | None = None  # 可选：数模论文、学术论文、学位论文、课程论文


class PaperReviewResponse(BaseModel):
    """论文评审响应"""
    paper_type: str
    total_score: int
    grade: str
    review_report: str
    dimensions: list[dict]
