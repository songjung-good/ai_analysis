"""보고서 품질 평가 노드 (Hybrid = 규칙 검사 + LLM Judge).

흐름: report_generation -> evaluator -> route_after_eval
    pass                -> END
    rewrite             -> report_generation  (feedback 반영 재작성)
    research            -> supervisor         (target_views 재조사)
    report_attempts 상한 -> END                (미달 사유를 eval_result에 남김)

평가 4항목: groundedness / neutrality / bias_control / coverage
- 규칙(코드, 결정적)을 먼저 계산한다. 확실한 실패는 Judge 없이 바로 미달.
- 금지 표현처럼 오탐 가능한 규칙은 '경고'로만 Judge에 넘긴다.
- Judge는 초안 문단 + 인용 Evidence 발췌를 함께 보고 항목별로 판정한다.
- Judge 오류·파싱 실패는 통과로 보지 않는다(1회 재호출 후 미달).

입력 State
- report_draft: {섹션명: [{"text": str, "source_ids": [str]}]}   (report 노드가 저장)
- references:   list[Evidence | dict]  (source_id, title, url, excerpt)
- report_attempts: int
출력 State: eval_result, report_attempts, resume_after_research
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Literal, TypedDict
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from ..state import EvalItem as ItemVerdict
from ..state import EvalVerdict

logger = logging.getLogger(__name__)

# ── 계약: 보고서 섹션 이름은 report 담당과 합의한 값 ─────────────────────────
VIEW_SECTIONS: dict[str, str] = {  # Supervisor 관점명 -> ReportDraft 섹션명
    "technical": "technical",
    "market": "market",
    "stakeholder": "stakeholder",
    "domain": "domain",
}
SUMMARY_SECTION = "summary"
RISK_SECTION = "risks"
ITEMS = ("groundedness", "neutrality", "bias_control", "coverage")

# ── 규칙 임계값 (통합 실행 후 조정) ─────────────────────────────────────────
MAX_UNCITED_RATIO = 0.2      # 관점·리스크 섹션에서 출처 없는 문단 비율 상한
MIN_DOMAINS_PER_VIEW = 2     # 관점별 서로 다른 출처 도메인 수
MAX_SINGLE_SOURCE_SHARE = 0.5  # 전체 인용 중 한 출처가 차지하는 비중 상한
MAX_REPORT_ATTEMPTS = 3      # state.py 계약: 최초 초안 + 보완 2회, 3번째 미달에 종료
EXCERPT_CHARS = 500
BANNED_PHRASES = (
    "압도적", "무조건", "확실한 수익", "독보적", "반드시 투자", "최고의 투자", "리스크가 없",
    "실패할 수 없", "100% 성공",
)

Action = Literal["pass", "rewrite", "research"]


# ── 1. 규칙 검사 ────────────────────────────────────────────────────────────
class RuleResult(TypedDict):
    passed: bool          # False = 확실한 실패 (Judge 없이 미달)
    warning: bool         # True = Judge가 확인해야 할 의심
    reason: str
    views: list[str]      # 문제가 된 관점 (재조사 대상 후보)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, Mapping) else getattr(obj, key, default)


def _paragraphs(draft: Mapping[str, Any], section: str) -> list[Mapping[str, Any]]:
    items = draft.get(section) or []
    return [p if isinstance(p, Mapping) else _dump(p) for p in items]


def _dump(model: Any) -> dict[str, Any]:
    return model.model_dump() if hasattr(model, "model_dump") else dict(model)


def _host(url: str | None) -> str | None:
    if not url:
        return None
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host or None


def _source_key(ref: Any) -> str:
    """편향 판단용 출처 단위: 웹이면 도메인, 내부 문서면 제목."""
    return _host(_get(ref, "url")) or f"doc:{_get(ref, 'title', _get(ref, 'source_id'))}"


def check_groundedness(draft: Mapping[str, Any], refs: Mapping[str, Any]) -> RuleResult:
    sections = [*VIEW_SECTIONS.values(), RISK_SECTION]
    paragraphs = [(s, p) for s in sections for p in _paragraphs(draft, s)]
    if not paragraphs:
        return RuleResult(passed=False, warning=False, reason="관점·리스크 섹션에 문단이 없음", views=[])
    unknown = sorted({sid for _, p in paragraphs for sid in p.get("source_ids", []) if sid not in refs})
    if unknown:
        return RuleResult(passed=False, warning=False,
                          reason=f"REFERENCE에 없는 출처 인용: {', '.join(unknown[:5])}", views=[])
    uncited = [(s, p) for s, p in paragraphs if not p.get("source_ids")]
    ratio = len(uncited) / len(paragraphs)
    if ratio > MAX_UNCITED_RATIO:
        views = sorted({v for v, s in VIEW_SECTIONS.items() for sec, _ in uncited if sec == s})
        return RuleResult(passed=False, warning=False,
                          reason=f"출처 없는 문단 비율 {ratio:.0%} > {MAX_UNCITED_RATIO:.0%}", views=views)
    return RuleResult(passed=True, warning=False, reason=f"출처 없는 문단 비율 {ratio:.0%}", views=[])


def check_neutrality(draft: Mapping[str, Any]) -> RuleResult:
    hits = sorted({
        phrase
        for section in draft
        for p in _paragraphs(draft, section)
        for phrase in BANNED_PHRASES
        if phrase in str(p.get("text", ""))
    })
    if hits:  # 오탐 가능 -> 경고만, 판정은 Judge
        return RuleResult(passed=True, warning=True, reason=f"단정·과장 의심 표현: {', '.join(hits)}", views=[])
    return RuleResult(passed=True, warning=False, reason="금지 표현 없음", views=[])


def check_bias(draft: Mapping[str, Any], refs: Mapping[str, Any]) -> RuleResult:
    weak_views = []
    for view, section in VIEW_SECTIONS.items():
        keys = {_source_key(refs[sid]) for p in _paragraphs(draft, section)
                for sid in p.get("source_ids", []) if sid in refs}
        if len(keys) < MIN_DOMAINS_PER_VIEW:
            weak_views.append(view)
    if not _paragraphs(draft, RISK_SECTION):
        return RuleResult(passed=False, warning=False, reason="리스크·반대 근거 섹션이 비어 있음", views=weak_views)
    if weak_views:
        return RuleResult(passed=False, warning=False,
                          reason=f"출처 다양성 부족(서로 다른 출처 < {MIN_DOMAINS_PER_VIEW}): {', '.join(weak_views)}",
                          views=weak_views)
    cited = [_source_key(refs[sid]) for s in draft for p in _paragraphs(draft, s)
             for sid in p.get("source_ids", []) if sid in refs]
    top = max((cited.count(k) for k in set(cited)), default=0)
    share = top / len(cited) if cited else 1.0
    if share >= MAX_SINGLE_SOURCE_SHARE:
        return RuleResult(passed=False, warning=False,
                          reason=f"단일 출처 비중 {share:.0%} >= {MAX_SINGLE_SOURCE_SHARE:.0%}", views=[])
    return RuleResult(passed=True, warning=False, reason=f"단일 출처 최대 비중 {share:.0%}", views=[])


def check_coverage(draft: Mapping[str, Any]) -> RuleResult:
    empty = [view for view, section in VIEW_SECTIONS.items() if not _paragraphs(draft, section)]
    if not _paragraphs(draft, SUMMARY_SECTION):
        return RuleResult(passed=False, warning=False, reason="SUMMARY 없음", views=empty)
    if empty:
        return RuleResult(passed=False, warning=False, reason=f"비어 있는 관점: {', '.join(empty)}", views=empty)
    return RuleResult(passed=True, warning=False, reason="4관점·SUMMARY 존재", views=[])


def check_rules(draft: Mapping[str, Any], refs: Mapping[str, Any]) -> dict[str, RuleResult]:
    return {
        "groundedness": check_groundedness(draft, refs),
        "neutrality": check_neutrality(draft),
        "bias_control": check_bias(draft, refs),
        "coverage": check_coverage(draft),
    }


# ── 2. LLM Judge ────────────────────────────────────────────────────────────
class JudgeItem(BaseModel):
    passed: bool
    reason: str = Field(description="판정 근거 1~2문장. 문제 문단이 있으면 섹션명과 내용을 짚는다")
    problem_views: list[str] = Field(default_factory=list,
                                     description="근거 자체가 부족해 재조사가 필요한 관점 (technical/market/stakeholder/domain)")
    fix: list[str] = Field(default_factory=list, description="표현·인용 수정으로 해결 가능한 구체적 수정 지시")


class JudgeOutput(BaseModel):
    groundedness: JudgeItem
    neutrality: JudgeItem
    bias_control: JudgeItem
    coverage: JudgeItem


JUDGE_PROMPT = """너는 AI 스타트업 투자 평가 보고서의 품질 심사자다. 아래 4개 항목을 각각 pass/fail로 판정한다.

[평가 항목 정의]
1. groundedness: 각 문단의 주장이 인용한 출처 발췌로 실제로 뒷받침되는가. 발췌에 없는 수치·사실을 쓰면 fail.
2. neutrality: 투자 판정(invest/conditional/hold) 자체는 허용한다. 단, 판정은 코드가 계산한 점수와 출처로 근거를 제시해야 하고,
   근거 없는 단정·과장·홍보성 표현(예: 압도적, 무조건, 독보적)이 있으면 fail. 회사 자체 주장을 사실처럼 쓰면 fail.
3. bias_control: 유리한 근거만 골라 쓰지 않고 리스크·한계·반대 근거를 함께 서술하는가. 단일 출처나 회사 보도자료에 편중되면 fail.
4. coverage: 기술 성숙도(technical), 시장성(market), 이해관계자(stakeholder), 도메인 적용(domain) 4관점이
   각각 실질적인 내용으로 다뤄졌는가. 섹션은 있지만 일반론뿐이면 fail.

[판정 규칙]
- 근거 자체가 부족해 문장 수정으로 해결할 수 없으면 problem_views에 해당 관점을 넣는다.
- 표현·인용 연결로 해결 가능하면 fix에 구체적 수정 지시를 쓴다.
- 확신이 없으면 fail로 판정한다.

[코드 규칙 검사 결과 (참고, 경고 항목은 네가 최종 판단)]
{rules}

[보고서 초안: 섹션별 문단과 인용 ID]
{draft}

[인용된 출처 발췌]
{evidence}
"""


def _evidence_block(draft: Mapping[str, Any], refs: Mapping[str, Any]) -> str:
    cited = sorted({sid for s in draft for p in _paragraphs(draft, s) for sid in p.get("source_ids", [])})
    lines = []
    for sid in cited:
        ref = refs.get(sid)
        if ref is None:
            continue
        excerpt = str(_get(ref, "excerpt", ""))[:EXCERPT_CHARS]
        lines.append(f"[{sid}] {_get(ref, 'title', '')} ({_get(ref, 'url') or '내부 문서'})\n{excerpt}")
    return "\n\n".join(lines) or "(인용 없음)"


def build_judge_prompt(draft: Mapping[str, Any], refs: Mapping[str, Any],
                       rules: Mapping[str, RuleResult]) -> str:
    rule_text = "\n".join(
        f"- {k}: {'경고' if r['warning'] else ('통과' if r['passed'] else '실패')} - {r['reason']}"
        for k, r in rules.items()
    )
    draft_text = json.dumps({s: _paragraphs(draft, s) for s in draft}, ensure_ascii=False, indent=1)
    return JUDGE_PROMPT.format(rules=rule_text, draft=draft_text, evidence=_evidence_block(draft, refs))


def default_judge() -> Callable[[str], JudgeOutput]:
    from langchain_openai import ChatOpenAI

    model = os.getenv("OPENAI_JUDGE_MODEL") or os.getenv("OPENAI_MODEL")
    if not model:
        raise RuntimeError("OPENAI_JUDGE_MODEL or OPENAI_MODEL is not set")
    effort = os.getenv("OPENAI_REASONING_EFFORT", "low")
    kwargs: dict[str, Any] = {"reasoning_effort": effort} if effort else {"temperature": 0}
    llm = ChatOpenAI(model=model, **kwargs).with_structured_output(JudgeOutput)
    return lambda prompt: llm.invoke(prompt, config={"run_name": "quality_judge"})


def run_judge(judge: Callable[[str], JudgeOutput], prompt: str, retries: int = 1) -> JudgeOutput | None:
    for attempt in range(retries + 1):
        try:
            out = judge(prompt)
            return out if isinstance(out, JudgeOutput) else JudgeOutput.model_validate(out)
        except Exception as exc:  # noqa: BLE001 - 파싱·호출 실패는 미달로 귀결
            logger.warning("judge failed (attempt %d): %s", attempt + 1, exc)
    return None


# ── 3. 판정 종합 ────────────────────────────────────────────────────────────
def combine(rules: Mapping[str, RuleResult], judge: JudgeOutput | None) -> EvalVerdict:
    items: dict[str, ItemVerdict] = {}
    research_views: set[str] = set()
    research_hints: dict[str, list[str]] = {}
    feedback: list[str] = []

    for key in ITEMS:
        rule = rules[key]
        j: JudgeItem | None = getattr(judge, key) if judge else None
        if not rule["passed"]:
            items[key] = ItemVerdict(passed=False, reason=f"[규칙] {rule['reason']}")
            # 출처 부족·관점 공백은 근거 문제 -> 재조사, 그 외(인용 오류)는 재작성
            if key in ("bias_control", "coverage") and rule["views"]:
                research_views.update(rule["views"])
                for view in rule["views"]:
                    research_hints.setdefault(view, []).append(rule["reason"])
            else:
                feedback.append(f"{key}: {rule['reason']}")
        elif judge is None:
            items[key] = ItemVerdict(passed=False, reason="[Judge] 평가 실패(호출·파싱 오류)")
        else:
            items[key] = ItemVerdict(passed=j.passed, reason=f"[Judge] {j.reason}")
            if not j.passed:
                valid = [v for v in j.problem_views if v in VIEW_SECTIONS]
                research_views.update(valid)
                for view in valid:
                    research_hints.setdefault(view, []).append(j.reason)
                feedback.extend(f"{key}: {fix}" for fix in j.fix)
                if not valid and not j.fix:
                    feedback.append(f"{key}: {j.reason}")

    passed = all(v["passed"] for v in items.values())
    if passed:
        action: Action = "pass"
    elif research_views:
        action = "research"
    elif judge is None and all(r["passed"] for r in rules.values()):
        action = "rewrite"  # Judge만 실패 -> 다시 써서 재평가
        feedback.append("평가 재시도: 이전 Judge 호출이 실패함")
    else:
        action = "rewrite"
    verdict = EvalVerdict(
        passed=passed,
        items=items,
        action=action,
        target_views=sorted(research_views),
        feedback=feedback,
    )
    if action == "research":  # supervisor가 rework_hints로 그대로 넘길 수 있게 관점별 사유
        verdict["research_hints"] = {v: research_hints.get(v, []) for v in sorted(research_views)}
    return verdict


def _refs_by_id(references: Iterable[Any]) -> dict[str, Any]:
    return {str(_get(r, "source_id")): r for r in references or []}


# ── 4. 노드 & 라우팅 ────────────────────────────────────────────────────────
def evaluate(state: Mapping[str, Any], *, judge: Callable[[str], JudgeOutput] | None = None) -> dict[str, Any]:
    draft = state.get("report_draft") or {}
    draft = draft if isinstance(draft, Mapping) else _dump(draft)
    refs = _refs_by_id(state.get("references", []))
    attempt = int(state.get("report_attempts", 0))

    rules = check_rules(draft, refs)
    hard_fail = any(not r["passed"] for r in rules.values())
    judge_out = None
    if not hard_fail:  # 규칙 확실 실패면 Judge 비용을 쓰지 않는다
        judge_out = run_judge(judge or default_judge(), build_judge_prompt(draft, refs, rules))
    verdict = combine(rules, judge_out)
    if hard_fail:  # Judge를 안 돌린 항목은 "미평가"로 표시만 바꾸고 판정 자체는 유지
        for key, rule in rules.items():
            if rule["passed"]:
                verdict["items"][key] = ItemVerdict(passed=True, reason=f"[규칙] {rule['reason']} (Judge 생략)")
        verdict["passed"] = False

    logger.info(
        "evaluator decision",
        extra={"trace_id": state.get("trace_id"), "node": "evaluator", "decision": verdict["action"],
               "reason": {k: v["reason"] for k, v in verdict["items"].items()},
               "target_views": verdict["target_views"], "attempt": attempt},
    )
    update: dict[str, Any] = {
        "eval_result": verdict,
        "report_attempts": attempt + (0 if verdict["passed"] else 1),
    }
    if verdict["action"] == "research":
        # 재조사로 근거가 바뀌면 점수도 바뀔 수 있으므로 투자 판단부터 다시 거친다.
        update["resume_after_research"] = "investment_decision"
    return update


def run(state: Mapping[str, Any]) -> dict[str, Any]:
    return evaluate(state)


Route = Literal["end", "report_generation", "supervisor"]


def route_after_eval(state: Mapping[str, Any]) -> Route:
    """pass면 종료, 미달이면 action대로. 미달 횟수가 상한이면 무조건 종료(종료 보장)."""
    verdict = state.get("eval_result") or {}
    if verdict.get("passed"):
        return "end"
    if int(state.get("report_attempts", 0)) >= MAX_REPORT_ATTEMPTS:
        return "end"
    return "supervisor" if verdict.get("action") == "research" else "report_generation"
