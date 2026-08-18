"""Pipeline nodes — one async function per step of the video-factory flow.

Each node:
  - reads/writes the shared ``PipelineState``
  - emits SSE events to the frontend via ``emit(...)``
  - is wrapped in a Langfuse span via ``obs.span(...)`` so the whole graph run
    shows up as a single nested trace

Services (the Gemini client) are injected through LangGraph's ``configurable``
config, so nodes never touch module-level globals.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import datetime
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from loguru import logger

from app.core import observability as obs
from app.core.config import ROOT_DIR
from app.graph.state import PipelineState, emit, services
from app.models.script import ImageElement, PartOverview
from app.models.storyboard import ShotImagePlan
from app.services import character_store
from app.services import director_store
from app.services import gemini_client
from app.services import openai_image
from app.services import scene_store
from app.services import storage as _storage
from app.services import subject_ref_store
from app.services.image_fetch import fetch_many, load_reference
from app.services.image_search import search_images
from app.services.prompts import get_prompt
from app.services.script_render import render_markdown
from app.services.video_search import search_videos

_OUTPUTS_DIR = ROOT_DIR / "outputs"
_MEDIA_DIR = _OUTPUTS_DIR / "media"
_SEARCH_CACHE = _OUTPUTS_DIR / "cache" / "image_search.json"


def _slug(topic: str) -> str:
    return "".join(c if c.isalnum() or c in " _-" else "_" for c in topic)[:50].strip() or "run"


# The category shown in the SSE status line of storyboard / image_prompts / generate_images. These
# three lines are the ONLY place an operator can see which rule set a run is actually being built
# with, so they must cover every category — a two-way ternary printed "อาหาร" for "other" too.
_CATEGORY_LABEL = {"food": "อาหาร 🍽", "drink": "เครื่องดื่ม 🥤", "other": "ทั่วไป 📦"}


def _category_label(category: str) -> str:
    return _CATEGORY_LABEL.get(category, _CATEGORY_LABEL["food"])


# First-line marker a video summary uses to say "I watched it, it's not this topic" (BATCH_PROMPT step 0).
OFF_TOPIC_MARKER = "OFF_TOPIC"

_VALID_JOINS = {"continuous", "match_cut", "dissolve", "cut", "j_cut", "l_cut"}


def _norm_join(raw: dict) -> str:
    """Sanitize a shot's join_with_prev from the LLM. Insert shots are never 'continuous'
    (Veo can't extend a talking-host clip into a no-person insert — it 400s)."""
    j = str(raw.get("join_with_prev", "cut")).lower().strip()
    j = j if j in _VALID_JOINS else "cut"
    if j == "continuous" and str(raw.get("shot_kind", "")).lower() == "insert":
        j = "match_cut"
    return j


def _norm_shot_scale(v) -> str:
    """Normalize the classifier's shot_scale to 'closeup' | 'medium' | 'wide' | '' — the bg-lock router
    compares it with ==, so raw LLM casing/variants ('Close-up', 'CU', 'false') must not slip through."""
    s = str(v or "").strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    if s.startswith("close") or s == "cu":
        return "closeup"
    if s.startswith("med") or s == "ms":
        return "medium"
    if s.startswith("wide") or s.startswith("establish") or s == "ws":
        return "wide"
    return ""


def _as_bool(v) -> bool:
    """Coerce a classifier flag to bool — a JSON string "false" is truthy under bool(), so match explicitly."""
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"true", "1", "yes"}


_QTY_RE = re.compile(r"[0-9๐-๙½¼¾]")


def _split_qty(item: str) -> tuple[str, str]:
    """Split a canonical item string into (base name, quantity). The base is the text before
    the first digit — 'ครีมเทียม 1 ช้อนโต๊ะพูน' → ('ครีมเทียม', '1 ช้อนโต๊ะพูน'), 'น้ำแข็ง' → ('น้ำแข็ง', '')."""
    m = _QTY_RE.search(item)
    if not m or m.start() == 0:   # no digit, or a digit-led name ("2% นมสด") → whole string is the base
        return item.strip(), ""
    base = item[: m.start()].strip()
    return (base or item.strip()), item[m.start():].strip()


# A scene's on_screen_text is one dense line of captions joined by these (see SCRIPT_PROMPT's
# "separate multiple specs with · or •").
_OST_SPLIT_RE = re.compile(r"[·•|\n]+")


def _longest_common_run(a: str, b: str) -> int:
    """Length of the longest substring shared by `a` and `b`, in CHARACTERS.

    Character-level on purpose: Thai writes without spaces, so a caption like "โรยผงปรุงรสตามชอบ" has
    no tokens to compare — but the shot whose line is "จากนั้นโรยผงปรุงรสตามชอบลงไปเลยค่ะ" shares a
    long unbroken run with it, while its neighbours share almost nothing.
    """
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def _ost_relevance(caption: str, voice_over: str, motion: str) -> float:
    """How much of `caption` this shot's own script accounts for.

    The spoken line outweighs the motion description: a caption is a written echo of what is being
    SAID at that moment, and the motion text often names the same objects in every shot of a scene
    (three consecutive close-ups of the same hands) which would make it a poor discriminator alone.
    """
    parts = [p.strip() for p in _OST_SPLIT_RE.split(caption) if p.strip()] or [caption.strip()]
    return sum(_longest_common_run(p, voice_over) + 0.4 * _longest_common_run(p, motion) for p in parts)


def _dedupe_scene_on_screen_text(shots: list[dict]) -> int:
    """Within ONE scene, keep a repeated on_screen_text on the single shot it belongs to and blank
    the rest. Returns how many were blanked.

    STORYBOARD_PROMPT now forbids the repetition outright, but this is the enforcing half of the
    same "prompt asks, code guarantees" pattern the breakdown already uses for voice_over splits
    (rebalance_intro_vo) and per-item intro shots (_rebuild_intro_shots, whose comment says the LLM
    "can't be trusted to"). It also repairs boards authored before that rule existed.

    Why it matters downstream: every non-empty on_screen_text gets OMNI_ONSCREEN_BLOCK appended, so
    N shots carrying the same caption become N separately-generated clips each rendering its own
    copy of the title graphic — the viewer sees it pop in and out N times, never quite identically.
    """
    by_text: dict[str, list[int]] = {}
    for i, s in enumerate(shots):
        text = str(s.get("on_screen_text") or "").strip()
        if text:
            by_text.setdefault(text, []).append(i)

    blanked = 0
    for text, idxs in by_text.items():
        if len(idxs) < 2:
            continue
        # Highest relevance wins; ties go to the EARLIEST shot, which is also the right answer for a
        # scene-title caption that no single line happens to echo.
        keep = max(idxs, key=lambda i: (_ost_relevance(text, str(shots[i].get("voice_over") or ""),
                                                       str(shots[i].get("motion_description") or "")), -i))
        for i in idxs:
            if i != keep:
                shots[i]["on_screen_text"] = ""
                blanked += 1
    return blanked


def _clear_overview_on_screen_text(shots: list[dict], all_ingredients: list[str],
                                   all_equipment: list[str]) -> int:
    """Blank the caption on an ALL-ITEMS overview shot. Returns how many were blanked.

    `_rebuild_intro_shots` now builds the overview with no caption, but boards written before that
    carry every item's caption joined with " · " on it — which Omni renders as a wall of title text
    across the flat-lay. This is the repair half, so an existing board is fixed without paying for a
    fresh breakdown.

    An overview is recognised the same way the image path does it (see the `_cov_ing`/`_cov_eq`
    coverage test): the shot's own refs cover the WHOLE board list. A per-item intro shot names one
    item and is left alone, and so is any ordinary shot — a mid-recipe frame never carries all 20
    ingredients as refs."""
    ing, eq = set(all_ingredients or []), set(all_equipment or [])
    blanked = 0
    for s in shots:
        if not str(s.get("on_screen_text") or "").strip():
            continue
        covers = ((ing and ing <= set(s.get("ingredient_refs") or []))
                  or (eq and eq <= set(s.get("equipment_refs") or [])))
        if covers:
            s["on_screen_text"] = ""
            blanked += 1
    return blanked


def _find_intro_closing(vo: str) -> int:
    """Index of the intro scene's REAL closing sentence ("และนี่คือ[วัตถุดิบ/อุปกรณ์]ทั้งหมด...") or -1.
    The narration sometimes uses "และนี่คือ" as a mid-list connector ("และนี่คือ เหยือกตวง...") — matching
    the FIRST occurrence dumped half the items into the overview shot, so only an occurrence followed
    closely by "ทั้งหมด" counts as the closing."""
    start = 0
    while (idx := vo.find("และนี่คือ", start)) >= 0:
        if "ทั้งหมด" in vo[idx:idx + 40]:
            return idx
        start = idx + 1
    return -1


def _head(base: str) -> str:
    """The head noun used to locate an item inside prose — the first space-token of the base, so a
    descriptive canonical name ('น้ำแข็ง สำหรับเติมเต็มแก้ว') still matches the plain word ('น้ำแข็ง')
    the script actually speaks. Thai compound words carry no internal spaces, so the head stays specific.
    ponytail: first-token heuristic; a genuinely two-word head would over-shorten — fine for our lists."""
    parts = base.split()
    return parts[0] if parts else base


def _sibling_conflicts(head: str, all_heads: list[str]) -> list[str]:
    """Other heads that CONTAIN this head as a substring (น้ำ ⊂ น้ำแข็ง / น้ำตาล). Thai runs words
    together with no gaps, so a raw `head in text` match inside one of these is a false positive."""
    return [h for h in all_heads if h != head and head in h]


def _head_spoken(head: str, text: str, all_heads: list[str]) -> bool:
    """True if `head` is genuinely named in `text` — mask out any longer sibling head first, so
    'น้ำ' is not counted as spoken merely because 'น้ำแข็ง' appears."""
    masked = text
    for h in _sibling_conflicts(head, all_heads):
        masked = masked.replace(h, " ")
    return head in masked


def _find_head(body: str, head: str, cursor: int, conflicts: list[str]) -> int:
    """Next occurrence of `head` at/after `cursor` that is NOT the start of a longer sibling head
    (so 'น้ำ' binds to a real 'น้ำ …' mention, not to the 'น้ำ' inside 'น้ำแข็ง')."""
    i = body.find(head, cursor)
    while i >= 0:
        if not any(body.startswith(h, i) for h in conflicts):
            return i
        i = body.find(head, i + 1)
    return -1


def _intro_groups(canon: list[str]) -> list[dict]:
    """Distinct canonical items in order, merging same-base variants (hot/cold amounts of the SAME
    item become one group). Each group carries its base name, the full canonical strings, their
    quantities, and a display caption.
    ponytail: base = text before first digit; two genuinely different items sharing a prefix could
    over-merge — upgrade with an explicit base_name field on the canonical list if that ever bites."""
    groups: list[dict] = []
    by_base: dict[str, dict] = {}
    for it in canon:
        base, qty = _split_qty(it)
        g = by_base.get(base)
        if g is None:
            g = {"base": base, "items": [], "qtys": []}
            by_base[base] = g
            groups.append(g)
        g["items"].append(it)
        if qty and qty not in g["qtys"]:
            g["qtys"].append(qty)
    for g in groups:
        g["caption"] = f"{g['base']} {' / '.join(g['qtys'])}".strip() if g["qtys"] else g["base"]
    return groups


def _intro_fallback_prompt(kind: str, caption: str, aspect: str) -> str:
    """Deterministic prompt_img for an intro shot when no clean one could be harvested from the LLM
    (e.g. every candidate mentioned the host and was dropped). Thai item name inside an English
    scaffold — existing prompts already mix Thai fragments, and the Image Prompt step can rewrite
    it into a richer English version later. Never leaves a shot with an empty prompt."""
    if kind == "ingredient":
        return (f"A photorealistic insert shot of {caption}, presented in its own vessel, resting on a "
                f"light marble counter, the rest of the counter empty. Lit by soft natural light, a clean atmosphere. "
                f"Shot on 100mm macro, low 30-45 degree angle, shallow depth of field, {aspect} widescreen, cinematic.")
    return (f"A photorealistic insert shot of {caption}, resting directly on a light marble counter, "
            f"the rest of the counter empty. Lit by soft natural light, a clean atmosphere. "
            f"Shot on 85mm, low 30-45 degree hero angle, shallow depth of field, {aspect} widescreen, cinematic.")


def _intro_overview_fallback_prompt(kind: str, bases: list[str], aspect: str) -> str:
    what = "ingredients" if kind == "ingredient" else "kitchen tools"
    return (f"A photorealistic top-down flat lay of all the {what}: {', '.join(bases)}, neatly arranged "
            f"on a light marble counter. Lit by soft natural light, a clean atmosphere. "
            f"Shot on 28mm, straight top-down, deep focus, {aspect} widescreen, cinematic.")


# NO-DANGLING-LEAD-IN for the anchor split: an item's slice runs up to the NEXT item's name, so the
# announcer phrase introducing the next item ("ค่ะ ถัดมาเป็น", "นะคะ ชิ้นต่อไปคือ", a bare "และ") lands
# at the END of the previous item's line — that clip then speaks a hanging connector and Veo pads or
# ad-libs. The item's own sentence ends at its last polite particle; whatever follows is the next
# item's lead-in, gated on ending like an announcer (copula/connector) so a genuine trailing clause
# about the item itself is never moved.
_POLITE_END_RE = re.compile(r'(?:นะคะ|นะครับ|น่ะค่ะ|ค่ะ|คะ|ครับ|จ้ะ|จ้า)(?=\s|$)')
_LEADIN_END_RE = re.compile(r'(?:และ|หรือ|แต่|กับ|ก็|ต่อไป|จากนั้น|ถัดมา|ต่อด้วย|รวมถึง|ได้แก่|คือ|เป็น|มี|ด้วย|ใช้|สำหรับ|สุดท้าย)$')
_CONN_TOKENS = {"และ", "และก็", "แล้วก็", "หรือ", "กับ", "ก็", "ต่อด้วย"}


def _split_trailing_leadin(vo: str) -> tuple[str, str]:
    """Split an anchor slice into (line, trailing lead-in fragment). frag = "" when nothing to move —
    the slice already ends on its own sentence. ponytail: the residue "…ถัดมานะคะ" (announcer that
    itself ends on a polite particle) stays put — a complete utterance, Veo won't hang on it."""
    v = (vo or "").rstrip()
    if not v:
        return v, ""
    last = None
    for m in _POLITE_END_RE.finditer(v):
        last = m
    if last:
        head, frag = v[: last.end()], v[last.end():].strip()
    else:
        # no polite particle anywhere in the slice ("…ครึ่งช้อนชา และ") → peel trailing connector tokens
        toks = v.split()
        i = len(toks)
        while i > 1 and toks[i - 1] in _CONN_TOKENS:
            i -= 1
        head, frag = " ".join(toks[:i]), " ".join(toks[i:])
    if not frag or not head.strip() or not _LEADIN_END_RE.search(frag):
        return v, ""    # nothing after the particle, or the tail is a real clause about the item → keep
    return head.rstrip(), frag


_SCENE_KIND_MAP = {"intro_ingredients": "ingredient", "intro_equipment": "equipment"}


def _intro_kind(sc) -> str:
    """Return "ingredient" / "equipment" / "" for a scene, preferring the structured `scene_kind`
    field (language/category-agnostic) and falling back to the legacy Thai transition/name substring
    match for scripts generated before `scene_kind` existed."""
    kind = _SCENE_KIND_MAP.get(getattr(sc, "scene_kind", "") or "", "")
    if kind:
        return kind
    hay = f"{getattr(sc, 'transition', '')} {getattr(sc, 'name', '')}"
    return "ingredient" if "แนะนำวัตถุดิบ" in hay else ("equipment" if "แนะนำอุปกรณ์" in hay else "")


def _rebuild_intro_shots(shots_raw: list[dict], production: dict, kind: str, scene_vo: str = "",
                         aspect: str = "16:9") -> list[dict]:
    """Deterministically enforce the intro-scene structure the LLM keeps breaking: ONE insert shot per
    distinct canonical item (variants of the same base merged) + a final all-together overview. Each
    per-item shot's VISUAL fields (motion_description, image_subjects, refs, on_screen_text) are code-set
    from the canonical item so the image can never drift from what the shot is about. voice_over is
    ANCHOR-SPLIT from the scene's own voice_over — each item keeps the clause that NAMES it, so it never
    speaks a sibling's line and the shots' VOs concatenate back to the script VO verbatim (faithful).
    prompt_img is reused from the matching LLM shot. Returns raw shot dicts."""
    canon = [str(x) for x in (production.get("ingredients" if kind == "ingredient" else "equipment") or []) if str(x).strip()]
    if not canon:
        return shots_raw
    groups = _intro_groups(canon)

    def _bases_of(sh: dict) -> set[str]:
        out = set()
        for r in (sh.get("ingredient_refs" if kind == "ingredient" else "equipment_refs") or []):
            out.add(_split_qty(str(r))[0])
        for s in (sh.get("image_subjects") or []):
            out.add(_split_qty(str(s))[0])
        return out

    # prompt_img: reuse the LLM shot that depicted each item (ref/subject base match); the shot matching
    # the MOST groups (≥2) is the LLM's own overview and sources the overview image.
    overview_src, best = None, 1
    for sh in shots_raw:
        n = sum(1 for g in groups if g["base"] in _bases_of(sh))
        if n > best:
            best, overview_src = n, sh
    # Per-item prompt: harvest ONLY from a shot that depicts exactly ONE group (a true single-item
    # shot). The all-items overview shot matches many groups → skipped here (so its crowded prompt
    # isn't donated to one item), which also means a scene where the LLM emitted a lone all-items
    # shot leaves per-item prompts to the fallback rather than mis-assigning.
    for sh in shots_raw:
        matched = [g for g in groups if g["base"] in _bases_of(sh)]
        if len(matched) == 1 and not matched[0].get("prompt"):
            p = str(sh.get("prompt_img", "")).strip()
            # Intro shots are object-only: drop any harvested prompt that mentions the host/a person
            # (LLM drift) rather than carry the contamination forward.
            if p and not _prompt_has_person(p):
                matched[0]["prompt"] = p

    # voice_over: anchor-split the scene's own voice_over. Walk the canonical items with a forward
    # cursor, find each base name; item i's line = from its name to the next item's name (item 1 also
    # absorbs the lead-in). A trailing "และนี่คือ…" closing goes to the overview. Items the script never
    # named stay empty (coverage repair upstream prevents this).
    vo = scene_vo or ""
    close_at = _find_intro_closing(vo)
    body = vo[:close_at] if close_at >= 0 else vo
    overview_vo = vo[close_at:].strip() if close_at >= 0 else ""
    hits: list[tuple[dict, int]] = []
    heads = [_head(g["base"]) for g in groups]
    cursor = 0
    for g, head in zip(groups, heads):
        idx = _find_head(body, head, cursor, _sibling_conflicts(head, heads))
        if idx >= 0:
            hits.append((g, idx))
            cursor = idx + len(head)
    for k, (g, pos) in enumerate(hits):
        start = 0 if k == 0 else pos
        end = hits[k + 1][1] if k + 1 < len(hits) else len(body)
        g["vo"] = body[start:end].strip()
    # Shift each slice's trailing lead-in ("ค่ะ ถัดมาเป็น" / bare "และ") to the START of the next
    # slice — the words read in order stay verbatim, only the boundary moves. The last item's
    # fragment (if any) rides onto the overview/closing line.
    seq = [g for g, _ in hits]
    for k, g in enumerate(seq):
        line, frag = _split_trailing_leadin(g.get("vo", ""))
        if not frag:
            continue
        g["vo"] = line
        if k + 1 < len(seq):
            seq[k + 1]["vo"] = f"{frag} {seq[k + 1].get('vo', '')}".strip()
        else:
            overview_vo = f"{frag} {overview_vo}".strip()

    # Preserve the scene's opening transition — the LLM set it on its first shot; forcing "cut" made a
    # scripted dissolve/match-cut into the intro scene open with a hard cut. Rest of the shots: cut.
    open_join = str((shots_raw[0] if shots_raw else {}).get("join_with_prev", "cut")).lower().strip()
    if open_join not in {"cut", "dissolve", "match_cut", "fade"}:
        open_join = "cut"

    out: list[dict] = []
    for i, g in enumerate(groups, 1):
        cap = g["caption"]
        out.append({
            "no": i, "time": "", "shot_kind": "insert", "join_with_prev": open_join if i == 1 else "cut", "screen_direction": "neutral",
            "motion_description": f"{cap} วางนิ่งบนเคาน์เตอร์หินอ่อน กล้องค่อยๆ ดันเข้าหา",
            "voice_over": g.get("vo", ""),
            "on_screen_text": cap,
            "key_message": "",
            "prompt_img": g.get("prompt") or _intro_fallback_prompt(kind, cap, aspect),
            "image_subjects": [g["base"]],
            "ingredient_refs": list(g["items"]) if kind == "ingredient" else [],
            "equipment_refs": list(g["items"]) if kind == "equipment" else [],
        })
    all_items = [it for g in groups for it in g["items"]]
    all_bases = [g["base"] for g in groups]
    ov_prompt = str((overview_src or {}).get("prompt_img", "")).strip()
    if not ov_prompt or _prompt_has_person(ov_prompt):
        # object-only overview: never carry a host-contaminated prompt forward — and never leave it empty
        ov_prompt = _intro_overview_fallback_prompt(kind, all_bases, aspect)
    out.append({
        "no": len(groups) + 1, "time": "", "shot_kind": "insert", "join_with_prev": "cut", "screen_direction": "neutral",
        "motion_description": f"{', '.join(all_bases[:8])} วางเรียงรวมกันแบบ flat lay บนเคาน์เตอร์หินอ่อน กล้องค่อยๆ ดันเข้าหา",
        "voice_over": overview_vo,
        # No caption on the all-together shot. Every non-empty on_screen_text becomes an
        # OMNI_ONSCREEN_BLOCK instruction to render that text as a title graphic — and this one used
        # to be every item's caption joined with " · ", so a 20-ingredient recipe asked for a wall of
        # text across the frame. Empty instead sends OMNI_ONSCREEN_NONE_BLOCK ("no lettering
        # anywhere"), which is what a clean flat-lay wants. Per-item intro shots keep their caption.
        "on_screen_text": "",
        "key_message": "",
        "prompt_img": ov_prompt,
        "image_subjects": all_bases,
        "ingredient_refs": all_items if kind == "ingredient" else [],
        "equipment_refs": all_items if kind == "equipment" else [],
    })
    return out


async def _ensure_intro_coverage(svc, doc, topic: str, only_part: int | None = None) -> None:
    """Guarantee every canonical item is SPOKEN in each intro scene's voice_over, so the storyboard's
    faithful split yields one non-silent shot per item (no gap-generation downstream — the script stays
    the single source of truth). When items are missing, regenerate that scene's narration via LLM
    (varied wording); a deterministic backstop appends any item the LLM still drops.

    ``only_part`` limits repair to that part number — on a single-part regeneration we must not rewrite
    intro scenes in parts the user didn't touch (their narration may be hand-edited)."""
    production = doc.production.model_dump()
    char = production.get("character_desc", "")
    todo: list[tuple] = []   # (scene, kind, groups, heads) for scenes that need repair
    for part in doc.parts:
        if only_part is not None and getattr(part, "number", None) != only_part:
            continue
        for sc in part.scenes:
            kind = _intro_kind(sc)
            if not kind:
                continue
            canon = [str(x) for x in (production.get("ingredients" if kind == "ingredient" else "equipment") or []) if str(x).strip()]
            groups = _intro_groups(canon)
            if not groups:
                continue
            vo = sc.voice_over or ""
            heads = [_head(g["base"]) for g in groups]
            # Repair when any item isn't spoken OR there's no all-together closing (the overview shot
            # needs a line too). Closing = "และนี่คือ...ทั้งหมด" — same detection as the storyboard split.
            if all(_head_spoken(h, vo, heads) for h in heads) and _find_intro_closing(vo) >= 0:
                continue
            todo.append((sc, kind, groups, heads))
    if not todo:
        return

    async def _repair(sc, kind, groups):
        try:
            return await svc.gemini.complete_intro_narration([g["caption"] for g in groups], kind, topic, character_desc=char)
        except Exception as e:  # noqa: BLE001 — repair is best-effort; backstop still guarantees coverage
            logger.warning(f"intro coverage repair failed for scene {getattr(sc, 'scene_id', '?')}: {e}")
            return {}

    # Intro scenes are independent — repair them concurrently (was sequential per scene).
    results = await asyncio.gather(*(_repair(sc, kind, groups) for sc, kind, groups, _ in todo))
    for (sc, kind, groups, heads), res in zip(todo, results):
        if (res.get("voice_over") or "").strip():
            sc.voice_over = res["voice_over"].strip()
            if (res.get("on_screen_text") or "").strip():
                sc.on_screen_text = res["on_screen_text"].strip()
        # Backstop: any item STILL unspoken → append a short clause so no shot ends up silent.
        missing = [g for g, h in zip(groups, heads) if not _head_spoken(h, sc.voice_over or "", heads)]
        if missing:
            sc.voice_over = (f"{sc.voice_over or ''} " + " ".join(f"{g['caption']}ค่ะ" for g in missing)).strip()
            logger.info(f"intro coverage backstop added {len(missing)} item(s) to scene {getattr(sc, 'scene_id', '?')}")


def _load_search_cache() -> dict[str, list]:
    """Persistent {query: search_results} cache so the same keyword reuses the same
    images (deterministic refs) and we don't re-hit the search API. Best-effort."""
    try:
        return json.loads(_SEARCH_CACHE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — missing/corrupt cache is fine
        return {}


def _save_search_cache(cache: dict[str, list]) -> None:
    try:
        _SEARCH_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _SEARCH_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:  # noqa: BLE001 — caching is best-effort
        logger.warning(f"Failed to write image-search cache: {e}")


async def generate_queries(state: PipelineState, config: RunnableConfig) -> dict:
    topic = state["topic"]
    svc = services(config)
    with obs.span("node.generate_queries", input={"topic": topic}):
        emit({"type": "status", "message": f"Generating search queries for '{topic}'..."})
        queries = await svc.gemini.generate_query_variants(
            topic, n=svc.settings.video.search_query_variants
        )
    return {"queries": queries}


async def search(state: PipelineState, config: RunnableConfig) -> dict:
    queries = state["queries"]
    svc = services(config)
    with obs.span("node.search", input={"queries": queries}) as span:
        emit({"type": "status", "message": f"Searching YouTube with {len(queries)} queries in parallel..."})
        candidates = await search_videos(queries, svc.settings)
        span.update(output={"candidate_count": len(candidates)})

        if not candidates:
            emit({"type": "error", "message": "No videos found for this topic."})
            return {"pool": [], "confirmed": [], "seen_ids": set(), "round_num": 1, "error": "no_videos"}

        emit({"type": "status", "message": f"Found {len(candidates)} unique candidates. Filtering for relevance..."})

    return {
        "pool": candidates,
        "seen_ids": {v["id"] for v in candidates},
        "confirmed": [],
        "round_num": 1,
    }


async def relevance_filter(state: PipelineState, config: RunnableConfig) -> dict:
    svc = services(config)
    topic = state["topic"]
    target = state["target"]
    confirmed = list(state.get("confirmed", []))
    pool = state.get("pool", [])
    needed = target - len(confirmed)

    # First pass borrows the search node's "Filtering for relevance…" label; on a
    # top-up loop round_num is set, so announce the re-check explicitly.
    if state.get("round_num"):
        emit({"type": "status", "message": f"Re-checking relevance ({len(pool)} new candidates)..."})

    with obs.span("node.filter", input={"pool": len(pool), "needed": needed}) as span:
        kept = await svc.gemini.filter_videos(pool, topic, n=needed) if pool else []
        confirmed = confirmed + kept
        # Count consecutive rounds that add NO new relevant video — even though search keeps
        # finding candidates — so top-up can stop early (the rest of YouTube isn't on-topic).
        # (A fail-closed empty `kept` from a parse failure also counts as barren — rare, fine.)
        barren = 0 if kept else state.get("barren_rounds", 0) + 1
        span.update(output={"kept": len(kept), "confirmed_total": len(confirmed), "barren_rounds": barren})

    return {"confirmed": confirmed, "barren_rounds": barren}


async def topup(state: PipelineState, config: RunnableConfig) -> dict:
    """Search a fresh batch of candidates to fill the relevance gap.

    Higher temperature on the query variants makes the new queries diverge from
    earlier rounds so we actually surface unseen videos.
    """
    svc = services(config)
    topic = state["topic"]
    target = state["target"]
    seen_ids = set(state.get("seen_ids", set()))
    used_queries = list(state.get("queries", []))
    round_num = state.get("round_num", 1) + 1
    needed = target - len(state.get("confirmed", []))

    with obs.span("node.topup", input={"round": round_num, "needed": needed}) as span:
        emit({"type": "progress", "message": f"Only {len(state.get('confirmed', []))}/{target} confirmed. Searching for {needed} more..."})

        # Higher temperature + the queries already used → genuinely different queries this round
        # (the old code called defaults and re-surfaced the same near-miss pool).
        new_queries = await svc.gemini.generate_query_variants(
            topic, n=svc.settings.video.search_query_variants, temperature=0.9, avoid=used_queries,
        )
        found = await search_videos(new_queries, svc.settings)
        fresh = [v for v in found if v["id"] not in seen_ids]
        seen_ids.update(v["id"] for v in fresh)
        span.update(output={"fresh": len(fresh)})

        if not fresh:
            logger.info("No new candidates found in top-up round, stopping.")
        else:
            logger.info(f"Top-up round {round_num}: {len(fresh)} fresh candidates")

    # Accumulate the queries so the next round avoids all of them.
    all_queries = list(dict.fromkeys(used_queries + new_queries))
    return {"pool": fresh, "seen_ids": seen_ids, "round_num": round_num, "queries": all_queries}


async def analyze(state: PipelineState, config: RunnableConfig) -> dict:
    """Analyze each confirmed video concurrently (capped), streaming per-video progress.

    Second relevance gate: Gemini has now WATCHED each video. If a clip turns out off-topic
    (summary starts with the OFF_TOPIC marker from BATCH_PROMPT step 0), drop it — and rewrite
    `confirmed` to only the videos actually used, so `references` never lists a dropped/failed clip.
    """
    svc = services(config)
    topic = state["topic"]
    confirmed = state["confirmed"]
    total = len(confirmed)
    gate = svc.settings.pipeline.off_topic_gate
    concurrency = max(1, svc.settings.pipeline.analyze_concurrency)
    sem = asyncio.Semaphore(concurrency)

    with obs.span("node.analyze", input={"videos": total, "concurrency": concurrency}) as span:
        emit({"type": "references", "videos": [{"title": v["title"], "url": v["url"]} for v in confirmed]})
        emit({"type": "analyzing", "videos": [{"title": v["title"], "url": v["url"]} for v in confirmed]})

        results: dict[int, dict] = {}
        off_topic: dict[int, str] = {}

        async def _one(idx: int, video: dict) -> None:
            async with sem:
                try:
                    summary = await svc.gemini.summarize_video(video, topic, idx + 1, total)
                    notes = summary["notes"]
                    first_line = notes.lstrip().splitlines()[0].strip() if notes.strip() else ""
                    # Tolerate markdown/odd separators: "**OFF_TOPIC:**", "OFF-TOPIC: ..." must still match.
                    probe = first_line.upper().lstrip("*#> ").replace("-", "_")
                    if gate and probe.startswith(OFF_TOPIC_MARKER):
                        off_topic[idx] = first_line
                        logger.info(f"Video {idx + 1} dropped as off-topic: '{video['title'][:60]}' — {first_line}")
                        emit({"type": "analyzed", "index": idx, "success": False, "off_topic": True, "message": first_line})
                        return
                    results[idx] = summary
                    emit({"type": "analyzed", "index": idx, "success": True})
                except Exception as exc:  # noqa: BLE001 — one bad video must not kill the batch
                    logger.warning(f"Video {idx + 1} failed: {exc}")
                    emit({"type": "analyzed", "index": idx, "success": False})

        await asyncio.gather(*(_one(i, v) for i, v in enumerate(confirmed)))
        used = sorted(results)
        summaries = [results[i]["notes"] for i in used]
        used_videos = [confirmed[i] for i in used]
        video_pacing = [
            {
                "title": confirmed[i]["title"],
                "duration_seconds": results[i]["duration_seconds"],
                "phases": results[i]["phases"],
                "voice_samples": results[i].get("voice_samples", []),
                "presentation_style": results[i].get("presentation_style", ""),
            }
            for i in used
        ]
        span.update(output={"succeeded": len(summaries), "off_topic": len(off_topic),
                            "failed": total - len(summaries) - len(off_topic)})

        if not summaries:
            if off_topic and len(off_topic) == total:
                emit({"type": "error", "message": "All videos turned out to be off-topic after full analysis. Try rephrasing the topic."})
            else:
                emit({"type": "error", "message": "All analysis failed."})
            return {"summaries": [], "confirmed": [], "video_pacing": [], "error": "no_analysis"}

        # Re-emit references (final list) when some were dropped, so the UI reflects what's actually used.
        if len(used_videos) != total:
            emit({"type": "references", "videos": [{"title": v["title"], "url": v["url"]} for v in used_videos]})

    # Overwrite confirmed → routes builds references from it, so dropped/failed clips can't leak in.
    return {"summaries": summaries, "confirmed": used_videos, "video_pacing": video_pacing}


def _parse_mmss(t: str) -> int | None:
    try:
        mm, _, ss = t.strip().partition(":")
        return int(mm) * 60 + int(ss or 0)
    except (ValueError, AttributeError):
        return None


def _duration_budget_hint(cfg) -> str:
    """Build a synthesis-time pacing sentence from a Research-declared target runtime.

    Only fires in duration_mode="custom" — the user already knows the episode count/length
    up front (e.g. a fixed-format series) and set it before running Research. duration_mode
    "source" deliberately stays unscoped: it lets synthesis run free and derives the target
    duration FROM the result afterward, so scoping it here would defeat that mode's purpose.

    Without this, Script generation (which reads the same target) has to compress or pad
    content synthesis never budgeted for — the padding case is a real hallucination risk
    (Script inventing steps/tips that aren't in any source to fill time).
    """
    if getattr(cfg, "duration_mode", "source") != "custom":
        return ""
    parts = max(1, int(getattr(cfg, "parts", 1) or 1))
    lo_s, _, hi_s = str(getattr(cfg, "duration_per_part", "") or "").partition("-")
    lo, hi = _parse_mmss(lo_s), _parse_mmss(hi_s or lo_s)
    if not lo or not hi:
        return ""
    total_lo, total_hi = lo * parts, hi * parts
    return (
        f"TARGET LENGTH: this tutorial is being produced as {parts} part(s) totalling roughly "
        f"{_fmt_timecode(total_lo)}–{_fmt_timecode(total_hi)} (about {_fmt_timecode(lo)}–{_fmt_timecode(hi)} "
        "per part). Scope the synthesis to fit that budget — cover the base method's essential "
        "steps in full, but only include as many supplementary tips/refinements as comfortably fit. "
        "Do NOT invent extra steps, tips, or padding to stretch the content to this length — if the "
        "source material naturally covers less, keep it accurate and let it run shorter within the "
        "part's floor rather than fabricate filler."
    )


async def synthesize(state: PipelineState, config: RunnableConfig) -> dict:
    svc = services(config)
    topic = state["topic"]
    summaries = state["summaries"]
    confirmed = state.get("confirmed", [])  # analyze() already rewrote this to line up with `summaries`
    master_url = (state.get("master_url") or "").strip()
    master_index = None
    if master_url:
        for i, v in enumerate(confirmed):
            if v.get("url") == master_url:
                master_index = i
                break
        else:
            logger.warning(f"master_url {master_url!r} not among the confirmed/analyzed videos — ignoring")
            emit({
                "type": "progress",
                "message": f"⚠️ Master reference video ({master_url}) was not among the analyzed videos — proceeding without it.",
            })
    # No explicit (or mismatched) master from the user → auto-suggest one instead of falling back to
    # a free blend. Makes "one base method + supplementary tips" the DEFAULT, not something the user
    # must remember to opt into. Best-effort: any failure here just proceeds without a master, same
    # as before this existed.
    auto_suggested = False
    if master_index is None and len(summaries) >= 2:
        try:
            suggestion = await svc.gemini.select_master_video(summaries, topic, confirmed)
        except Exception as e:  # noqa: BLE001 — a failed suggestion must not block synthesis
            logger.warning(f"select_master_video failed, proceeding without an auto-suggested master: {e}")
            suggestion = {"index": None, "reason": ""}
        if suggestion.get("index") is not None:
            master_index = suggestion["index"]
            auto_suggested = True
            chosen = confirmed[master_index]
            emit({
                "type": "master_suggested",
                "index": master_index,
                "url": chosen.get("url", ""),
                "title": chosen.get("title", ""),
                "reason": suggestion.get("reason", ""),
            })
            emit({"type": "progress", "message": f"🎯 Auto-selected master reference: {chosen.get('title', '')} — {suggestion.get('reason', '')}"})

    duration_hint = _duration_budget_hint(state.get("script_config"))
    with obs.span("node.synthesize", input={"summaries": len(summaries), "master_index": master_index,
                                             "auto_suggested": auto_suggested, "duration_budgeted": bool(duration_hint)}) as span:
        emit({"type": "status", "message": "Synthesizing insights from all videos..."})
        synthesis = await svc.gemini.synthesize_from_summaries(
            summaries, topic, master_index=master_index, duration_hint=duration_hint,
        )
        span.update(output={"length": len(synthesis)})
        emit({"type": "result", "content": synthesis})
    # Structured ingredient/equipment lists, extracted HERE because every research flavour
    # (build_research, build_research_reuse, the full pipeline) ends at this node — the route then
    # seeds the run's Menu from them the moment research is saved, instead of waiting for the first
    # script generation. Best-effort: research must never fail over a missing shopping list; empty
    # lists simply mean no Menu seed, and the script step falls back to extracting for itself.
    ingredients, equipment = [], []
    try:
        emit({"type": "status", "message": "Extracting ingredient/equipment lists..."})
        items = await svc.gemini.extract_production_items(synthesis, topic)
        ingredients, equipment = items.get("ingredients") or [], items.get("equipment") or []
    except Exception:  # noqa: BLE001
        logger.exception("production-items extraction failed — research proceeds without lists")

    # Threaded to routes.py's `_aggregate_pacing`: when a master is set (explicit or auto-suggested),
    # duration_mode="source" should derive its target from the MASTER's own pacing, not an average
    # across every analyzed video — the script is built primarily from the master's method, so ITS
    # runtime/phase shape is the one that actually matters for pacing.
    return {"synthesis": synthesis, "master_index": master_index,
            "ingredients": ingredients, "equipment": equipment}


def _section_text(v) -> str:
    """Render a director section value to text — strings pass through, do_dont's
    {"do": [...], "dont": [...]} object becomes bullet lists."""
    if isinstance(v, dict):
        out = []
        if v.get("do"):
            out.append("ควรทำ:\n" + "\n".join(f"- {x}" for x in v["do"]))
        if v.get("dont"):
            out.append("ควรเลี่ยง:\n" + "\n".join(f"- {x}" for x in v["dont"]))
        return "\n".join(out)
    return str(v or "").strip()


def _director_block(sections: dict, keys: list[str]) -> str:
    """Format the chosen director sections into a prompt block ("" if none)."""
    parts = [f"### {k}\n{t}" for k in keys if (t := _section_text(sections.get(k)))]
    return "\n\n".join(parts)


def _brand_overrides(brand: dict | None) -> dict:
    """Non-empty brand identity fields that override ScriptConfig inputs (theme/mood/material/lighting/wps)."""
    if not brand:
        return {}
    out: dict = {}
    for k in ("theme", "mood", "material_palette", "lighting"):
        if str(brand.get(k) or "").strip():
            out[k] = brand[k].strip()
    try:
        wps = float(brand.get("words_per_second") or 0)
    except (TypeError, ValueError):
        wps = 0.0
    if wps > 0:
        out["words_per_second"] = wps
    return out


# "M:SS" / "MM:SS", e.g. "0:00", "12:05" — matches every example in the generation prompts.
_TIMECODE_RE = re.compile(r"^\d{1,3}:\d{2}$")
# A dash-style separator (hyphen, en/em dash, minus) between the bare shot type and its framing
# description, e.g. "Medium Shot — ...". shot_type must never be just the bare type (prompt rule 5b).
_SHOT_TYPE_SEP_RE = re.compile(r"[\-–—―−]")


def _scan_scene_issues(scenes) -> list[str]:
    """Best-effort structural scan of freshly-LLM-generated scenes (NOT a model validator — the
    same ScriptDocument model is re-validated for hand-edited/partial scripts on the storyboard
    and regenerate-part re-entry paths, which must stay permissive). Returns human-readable
    problem strings so a caller can warn the user right at generation time instead of the issue
    surfacing several steps later as an unexplained silent shot or malformed image prompt."""
    issues = []
    for sc in scenes:
        sid = getattr(sc, "scene_id", "?")
        if not (sc.voice_over or "").strip():
            issues.append(f"scene {sid}: voice_over is blank")
        for label, tc in (("timecode_start", sc.timecode_start), ("timecode_end", sc.timecode_end)):
            if not _TIMECODE_RE.match((tc or "").strip()):
                issues.append(f"scene {sid}: {label} {tc!r} is not 'M:SS'/'MM:SS'")
        if not _SHOT_TYPE_SEP_RE.search(sc.shot_type or ""):
            issues.append(f"scene {sid}: shot_type {sc.shot_type!r} is missing a '<type> — <framing>' separator")
    return issues


def _emit_scene_issues(issues: list[str], where: str) -> None:
    if not issues:
        return
    logger.warning(f"{where}: {len(issues)} structural issue(s) in generated scenes: {issues}")
    preview = "; ".join(issues[:3]) + (f" (+{len(issues) - 3} more)" if len(issues) > 3 else "")
    emit({"type": "progress", "message": f"⚠️ {where}: {len(issues)} scene(s) may need review — {preview}"})


def _lines_block(title: str, pairs: list[tuple[str, str]]) -> str:
    """A '### TITLE' block of 'Label: value' lines, skipping empties. "" if all empty."""
    body = [f"{label}: {v.strip()}" for label, v in pairs if str(v or "").strip()]
    return f"### {title}\n" + "\n".join(body) if body else ""


def _brand_script_block(brand: dict | None) -> str:
    """BRAND guidance for the script: concept/platform/voice/music + editing (transition) + camera (shot_type)."""
    if not brand:
        return ""
    return _lines_block("BRAND IDENTITY — follow closely (script tone, VO, transitions, shot types)", [
        ("Concept / tagline", brand.get("tagline", "")),
        ("Platform & format style", brand.get("platform_style", "")),
        ("Voice-over tone", brand.get("vo_tone", "")),
        ("Music style (guide the `music` field of every scene)", brand.get("music", "")),
        ("Editing style (drive `transition` between scenes)", brand.get("editing_style", "")),
        ("Camera / movement (drive `shot_type`)", brand.get("camera_movement", "")),
    ])


def _brand_storyboard_block(brand: dict | None) -> str:
    """BRAND guidance for the storyboard breakdown: editing → join_with_prev, camera → motion_description."""
    if not brand:
        return ""
    return _lines_block("BRAND MOTION — apply to every shot", [
        ("Editing style (shape `join_with_prev`: cut / dissolve, and beat length)", brand.get("editing_style", "")),
        ("Camera / movement (shape `motion_description`)", brand.get("camera_movement", "")),
    ])


def _brand_video_block(brand: dict | None) -> str:
    """BRAND guidance for video prompts — only the VO delivery (tone + pace) for talking shots."""
    if not brand:
        return ""
    wps = brand.get("words_per_second") or 0
    pace = f"~{wps} words/second (steady, unhurried)" if wps else ""
    return _lines_block("BRAND VOICE — for shots where the character speaks (has voice_over)", [
        ("Speaking tone / delivery", brand.get("vo_tone", "")),
        ("Speaking pace", pace),
    ])


async def generate_script(state: PipelineState, config: RunnableConfig) -> dict:
    """Produce the structured script. Saving/emitting happens after image enrichment."""
    svc = services(config)
    topic = state["topic"]
    cfg = state["script_config"]
    synthesis = state["synthesis"]
    voice_samples = state.get("voice_samples") or []
    presentation_notes = state.get("presentation_notes") or []

    # Active brand overrides script inputs (theme/mood/material/lighting/pace) and adds a BRAND block.
    brand = scene_store.active_brand()
    overrides = _brand_overrides(brand)
    if overrides:
        cfg = cfg.model_copy(update=overrides)
        emit({"type": "status", "message": f"Applying brand: {brand.get('name', '')}..."})
    brand_block = _brand_script_block(brand)

    # Optionally attach a saved director guide — narrative sections for the script.
    director_prompt = ""
    if cfg.director_id:
        director = director_store.get(cfg.director_id)
        if director:
            director_prompt = _director_block(
                director.sections,
                ["tone_mood", "narrative_vo", "hook_cta", "graphics_text_music", "do_dont"],
            )
            emit({"type": "status", "message": f"Applying director style: {director.title}..."})
        else:
            logger.warning(f"director_id {cfg.director_id!r} not found — generating without it")

    # The run's Menu is authoritative once it exists (seeded at research time, curated in Manage):
    # its lists go to the LLM as an instruction AND overwrite production afterwards, so an item the
    # operator deleted from the Menu cannot be resurrected out of the synthesis text.
    menu_locked = bool(state.get("menu_locked"))
    menu_ing = state.get("menu_ingredients") or []
    menu_eq = state.get("menu_equipment") or []
    menu_block = ""
    if menu_locked:
        menu_block = get_prompt("SCRIPT_MENU_LOCK_BLOCK").format(
            ingredients="\n".join(f"- {n}" for n in menu_ing) or "(none)",
            equipment="\n".join(f"- {n}" for n in menu_eq) or "(none)",
        )
        emit({"type": "status", "message": f"Using the run's Menu as canonical lists ({len(menu_ing)} ingredients, {len(menu_eq)} tools)..."})

    with obs.span("node.generate_script", input={"topic": topic, "parts": cfg.parts, "director": cfg.director_id}) as span:
        emit({"type": "status", "message": "Generating storyboard script..."})
        doc = await svc.gemini.generate_script(
            synthesis, topic, cfg, director_prompt=director_prompt, brand_block=brand_block,
            voice_samples=voice_samples, presentation_notes=presentation_notes,
            menu_block=menu_block,
        )
        # The prompt asked; this guarantees. Before _ensure_intro_coverage, because it and the
        # storyboard's _rebuild_intro_shots treat production as the canonical item list — forcing
        # first means the intro scenes/shots are repaired against the REAL lists.
        if menu_locked:
            doc.production.ingredients = list(menu_ing)
            doc.production.equipment = list(menu_eq)
        # Intro (ingredient/equipment) scenes: ensure every canonical item is spoken so the storyboard's
        # faithful split has one non-silent shot per item (script stays the single source of truth).
        await _ensure_intro_coverage(svc, doc, topic)
        _emit_scene_issues(_scan_scene_issues(sc for part in doc.parts for sc in part.scenes), "generate_script")
        span.update(output={"scenes": sum(len(p.scenes) for p in doc.parts)})

    # Show the script immediately as a markdown preview; the structured doc view
    # is built once image enrichment finishes (frontend shows a "Parsing script"
    # spinner over this preview meanwhile).
    emit({"type": "script", "content": render_markdown(doc)})
    emit({"type": "status", "message": "Parsing script into document view..."})

    return {"script_doc_obj": doc}


# Fallback Thai narration pace when no brand/config words_per_second is set — matches the
# default used by breakdown_scene() so intra-part timecode estimates stay consistent pipeline-wide.
_DEFAULT_WORDS_PER_SECOND = 2.3


def _fmt_timecode(total_seconds: float) -> str:
    total_seconds = max(0, round(total_seconds))
    return f"{total_seconds // 60}:{total_seconds % 60:02d}"


def _estimate_scene_seconds(sc, words_per_second: float) -> float:
    words = len((sc.voice_over or "").split())
    wps = words_per_second if words_per_second and words_per_second > 0 else _DEFAULT_WORDS_PER_SECOND
    return max(3.0, words / wps)  # floor so a near-empty voice_over doesn't collapse toward 0s


def _recompute_part_timecodes(part, words_per_second: float) -> None:
    """Re-time every scene in `part` back-to-back from 0:00 based on each scene's voice_over length.

    A part regenerated in isolation (only reading the OTHER parts as read-only context) has no
    reliable timecode anchor of its own — the LLM's per-scene guesses routinely overlap or drift.
    This keeps the part internally consistent immediately, without depending on the frontend to
    reconcile it after a later storyboard re-measure."""
    t = 0.0
    for sc in part.scenes:
        dur = _estimate_scene_seconds(sc, words_per_second)
        sc.timecode_start = _fmt_timecode(t)
        t += dur
        sc.timecode_end = _fmt_timecode(t)
    part.duration = _fmt_timecode(t)


async def regenerate_script_part(state: PipelineState, config: RunnableConfig) -> dict:
    """Regenerate ONE part of an existing script (staircase per-part auto-fit), keeping other parts.

    Returns the full ScriptDocument with only ``part_number`` replaced, its scene_ids renumbered
    ``N.1..N.k``, and its timecodes deterministically recomputed from voice_over length (see
    ``_recompute_part_timecodes``) — the part's OWN internal timing is fully backend-owned. What
    is NOT done here is reconciling this part's total duration against the storyboard's actual
    per-shot clip lengths across the WHOLE document — that still happens later, in storyboard.
    """
    svc = services(config)
    topic = state["topic"]
    cfg = state["script_config"]
    synthesis = state.get("synthesis", "")
    doc = state["script_doc_obj"]
    n = int(state["part_number"])
    target_duration = state.get("target_duration") or cfg.target_duration
    target_scenes = int(state.get("target_scenes") or 0)
    min_shots = int(state.get("min_shots") or 0)

    # Same brand/director resolution as generate_script so tone matches the original generation.
    brand = scene_store.active_brand()
    overrides = _brand_overrides(brand)
    if overrides:
        cfg = cfg.model_copy(update=overrides)
    brand_block = _brand_script_block(brand)
    director_prompt = ""
    if cfg.director_id:
        director = director_store.get(cfg.director_id)
        if director:
            director_prompt = _director_block(
                director.sections,
                ["tone_mood", "narrative_vo", "hook_cta", "graphics_text_music", "do_dont"],
            )

    # Menu-locked lists, same as generate_script. Doubly important here: `doc` comes from the
    # frontend request, so its production block can be stale — still carrying a ref the operator
    # deleted after the last full generation.
    menu_locked = bool(state.get("menu_locked"))
    menu_ing = state.get("menu_ingredients") or []
    menu_eq = state.get("menu_equipment") or []
    menu_block = ""
    if menu_locked:
        menu_block = get_prompt("SCRIPT_MENU_LOCK_BLOCK").format(
            ingredients="\n".join(f"- {x}" for x in menu_ing) or "(none)",
            equipment="\n".join(f"- {x}" for x in menu_eq) or "(none)",
        )

    with obs.span("node.regenerate_script_part", input={"topic": topic, "part": n}) as span:
        emit({"type": "status", "message": f"Regenerating part {n}..."})
        new_part = await svc.gemini.regenerate_script_part(
            doc, n, synthesis, topic, cfg, target_duration, target_scenes or None,
            min_shots=min_shots or None, director_prompt=director_prompt, brand_block=brand_block,
            menu_block=menu_block,
        )
        # merge/renumber: replace ONLY part n; renumber its scene_ids; recompute its timecodes.
        new_part.number = n
        for i, sc in enumerate(new_part.scenes, start=1):
            sc.scene_id = f"{n}.{i}"
        _recompute_part_timecodes(new_part, cfg.words_per_second)
        parts = list(doc.parts)
        parts[n - 1] = new_part
        overview = list(doc.overview)
        if n - 1 < len(overview):
            overview[n - 1] = PartOverview(number=n, title=new_part.title, summary=(new_part.description or overview[n - 1].summary))
        merged = doc.model_copy(update={"parts": parts, "overview": overview})
        if menu_locked:   # enforce before intro-coverage repairs against the canonical lists
            merged.production.ingredients = list(menu_ing)
            merged.production.equipment = list(menu_eq)
        await _ensure_intro_coverage(svc, merged, topic, only_part=n)
        _emit_scene_issues(_scan_scene_issues(new_part.scenes), f"regenerate_script_part (part {n})")
        span.update(output={"scenes": len(new_part.scenes)})

    emit({"type": "script", "content": render_markdown(merged)})
    return {"script_doc_obj": merged}


async def _validate_subjects(gemini, shots, topic: str, rounds: int) -> None:
    """LLM-validate each shot's image_subjects, regenerating invalid ones (max `rounds`).

    `shots` is a list of (id, shot) where id uniquely identifies the shot.
    """
    pending = [(sid, sh) for sid, sh in shots if sh.image_subjects]
    for r in range(1, rounds + 1):
        if not pending:
            break
        res = await gemini.validate_image_subjects(
            [{"id": sid, "motion_description": sh.motion_description, "on_screen_text": sh.on_screen_text,
              "prompt_img": sh.prompt_img, "image_subjects": sh.image_subjects} for sid, sh in pending],
            topic,
        )
        still = []
        for sid, sh in pending:
            v = res.get(sid)
            if not v:
                continue
            sh.image_subjects = v["image_subjects"]
            if not v["valid"]:
                still.append((sid, sh))
        logger.info(f"Subject validation round {r}: {len(pending) - len(still)} ok, {len(still)} re-checking")
        pending = still


async def _validate_queries(gemini, query_map: dict, topic: str, theme: str, rounds: int) -> dict:
    """LLM-validate each transformed query, regenerating invalid ones (max `rounds`)."""
    final = dict(query_map)
    remaining = set(final)
    for r in range(1, rounds + 1):
        if not remaining:
            break
        res = await gemini.validate_image_queries(
            [{"subject": s, "query": final[s]} for s in remaining], topic, theme
        )
        still = set()
        for s in list(remaining):
            v = res.get(s)
            if not v:
                continue
            if v["query"]:
                final[s] = v["query"]
            if not v["valid"]:
                still.add(s)
        logger.info(f"Query validation round {r}: {len(remaining) - len(still)} ok, {len(still)} re-checking")
        remaining = still
    return final


async def _validate_candidates(gemini, query_map: dict, results: dict, topic: str) -> dict:
    """LLM-validate candidates from their descriptions, dropping unsuitable ones (no retry)."""
    payload = [
        {"subject": s, "query": query_map.get(s, s),
         "candidates": [{"index": i, "description": c.get("description", "")} for i, c in enumerate(cands)]}
        for s, cands in results.items() if cands
    ]
    if not payload:
        return results
    keep = await gemini.validate_candidates(payload, topic)
    out: dict = {}
    for s, cands in results.items():
        idxs = keep.get(s)
        out[s] = [cands[i] for i in idxs if 0 <= i < len(cands)] if idxs is not None else cands
    return out


async def _vision_validate_candidates(
    gemini, results: dict, topic: str, api_key: str, gcfg
) -> dict:
    """Confirm the text-filtered candidates by actually LOOKING at the images.

    Downloads + downscales every surviving candidate (de-duped, concurrent), then
    asks Gemini which ones truly depict each subject, batching subjects so each
    multimodal call carries at most ``vision_batch_images`` images. Subjects whose
    images all failed to download, or that the model leaves unanswered, are kept
    as-is — vision only ever removes a candidate it could see and rejected.
    """
    # 1. Download every surviving candidate URL once.
    all_urls = [c["url"] for cands in results.values() for c in cands if c.get("url")]
    if not all_urls:
        return results
    emit({"type": "status", "message": f"Downloading {len(set(all_urls))} reference images..."})
    images = await fetch_many(all_urls, api_key=api_key, cfg=gcfg)

    # 2. Per subject, keep only candidates whose image downloaded (preserve order).
    sent: dict[str, list[dict]] = {}   # subject -> candidate dicts actually shown
    payload: list[dict] = []           # {subject, images:[{mime,data}]} for the LLM
    for subject, cands in results.items():
        shown, imgs = [], []
        for c in cands:
            blob = images.get(c.get("url"))
            if blob:
                shown.append(c)
                imgs.append(blob)
        if imgs:
            sent[subject] = shown
            payload.append({"subject": subject, "images": imgs})

    if not payload:
        return results  # nothing downloaded — keep text-filtered survivors

    # 3. Batch by image count so a single call never carries too many images.
    cap = max(1, gcfg.vision_batch_images)
    batches, cur, cur_n = [], [], 0
    for el in payload:
        n = len(el["images"])
        if cur and cur_n + n > cap:
            batches.append(cur)
            cur, cur_n = [], 0
        cur.append(el)
        cur_n += n
    if cur:
        batches.append(cur)

    keep: dict[str, list[int]] = {}
    for i, batch in enumerate(batches, 1):
        emit({"type": "status", "message": f"Reviewing images visually (batch {i}/{len(batches)})..."})
        keep.update(await gemini.vision_filter_candidates(batch, topic))

    # 4. Map keep-positions (into the SHOWN list) back to candidate dicts.
    out = dict(results)
    for subject, shown in sent.items():
        idxs = keep.get(subject)
        if idxs is None:
            out[subject] = shown            # unanswered → keep what we showed
        else:
            out[subject] = [shown[i] for i in idxs if 0 <= i < len(shown)]
    return out


async def fetch_images(state: PipelineState, config: RunnableConfig) -> dict:
    """Find reference images per STORYBOARD SHOT, with LLM validation, then save + emit.

    Pipeline (per shot): validate image_subjects → transform to queries → validate
    queries → search candidates → validate/drop candidates → assign. Subjects/queries
    can be regenerated up to `max_validation_rounds`; bad candidates are dropped.
    Reference images live on the storyboard shots (each shot's `prompt_img` is what
    actually generates its picture later). Best-effort: failures leave a shot without
    an image, never crash.
    """
    svc = services(config)
    topic = state["topic"]
    doc = state["script_doc_obj"]
    sb = state.get("storyboard_obj")
    gcfg = svc.settings.image_search
    api_key = svc.tavily_api_key

    # Flatten storyboard into (unique_id, shot) pairs.
    shots = [(f"{sc.scene_id}#{sh.no}", sh) for sc in sb.scenes for sh in sc.shots] if sb else []
    enabled = gcfg.enabled and bool(api_key) and bool(shots)

    with obs.span("node.fetch_images", input={"shots": len(shots), "enabled": enabled}) as span:
        if enabled:
            theme = state["script_config"].theme
            material_palette = state["script_config"].material_palette

            # 1. validate / regenerate image_subjects (per shot)
            if gcfg.validation:
                emit({"type": "status", "message": "Validating visual elements..."})
                await _validate_subjects(svc.gemini, shots, topic, gcfg.max_validation_rounds)

            # 2. unique elements across all shots (dedup → fewer searches)
            unique = sorted({s.strip() for _, sh in shots for s in sh.image_subjects if s.strip()})

            # 3. transform → query (1 LLM call), then validate / regenerate
            emit({"type": "status", "message": f"Optimizing image queries for {len(unique)} visual elements..."})
            query_map = await svc.gemini.transform_image_queries(unique, topic, theme, material_palette)
            if gcfg.validation:
                query_map = await _validate_queries(svc.gemini, query_map, topic, theme, gcfg.max_validation_rounds)

            # 4. search candidates per unique element (capped concurrency).
            # Persistent cache keyed by query → same keyword reuses the same images
            # and we skip the API call entirely (only non-empty results are cached).
            emit({"type": "status", "message": f"Searching reference images for {len(unique)} elements..."})
            sem = asyncio.Semaphore(max(1, gcfg.concurrency))
            results: dict[str, list[dict]] = {}
            cache = _load_search_cache()

            async def _one(subject: str) -> None:
                q = query_map.get(subject, subject)
                if cache.get(q):
                    results[subject] = cache[q]
                    return
                try:
                    async with sem:
                        results[subject] = await search_images(q, api_key=api_key, cfg=gcfg, limit=gcfg.candidates)
                except Exception as e:  # noqa: BLE001 — one failed search must not sink the whole node (best-effort)
                    logger.warning(f"image search failed for {subject!r} (that shot just gets no ref): {e}")
                    results[subject] = []

            await asyncio.gather(*(_one(s) for s in unique))

            # persist newly-searched (non-empty) results so next run reuses them
            new_hits = 0
            for s in unique:
                q = query_map.get(s, s)
                if results.get(s) and q not in cache:
                    cache[q] = results[s]
                    new_hits += 1
            if new_hits:
                _save_search_cache(cache)

            # 5a. cheap text-description filter drops obvious junk before we download
            if gcfg.validation:
                emit({"type": "status", "message": "Screening candidates by description..."})
                results = await _validate_candidates(svc.gemini, query_map, results, topic)

            # 5b. vision confirm — download survivors + let Gemini look at the images
            if gcfg.vision_validation:
                results = await _vision_validate_candidates(svc.gemini, results, topic, api_key, gcfg)

            # 6. assign results back to every shot (candidates = surviving URLs)
            for _, sh in shots:
                sh.image_results = [
                    ImageElement(subject=s, query=query_map.get(s, s),
                                 candidates=[c["url"] for c in results.get(s, [])])
                    for s in sh.image_subjects
                ]
                sh.img = next((c for el in sh.image_results for c in el.candidates), "")

            total_cand = sum(len(c) for c in results.values())
            found = sum(1 for _, sh in shots if sh.img)
            logger.info(f"Image search: {len(unique)} elements → {total_cand} candidates kept, {found}/{len(shots)} shots have an image")
            span.update(output={"elements": len(unique), "candidates": total_cand, "shots_with_image": found})
        else:
            for _, sh in shots:
                sh.image_results = []
                sh.img = ""
            if gcfg.enabled and not api_key:
                logger.warning("image_search enabled but TAVILY_API_KEY missing — skipping images")

        # Persist + emit the script (structured) and the storyboard (with images).
        _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = "".join(c if c.isalnum() or c in " _-" else "_" for c in topic)[:50].strip()

        (_OUTPUTS_DIR / f"{ts}_{slug}.json").write_text(doc.model_dump_json(indent=2), encoding="utf-8")
        emit({"type": "script", "content": render_markdown(doc)})
        emit({"type": "script_json", "content": doc.model_dump()})

        if sb is not None:
            (_OUTPUTS_DIR / f"{ts}_{slug}_storyboard.json").write_text(sb.model_dump_json(indent=2), encoding="utf-8")
            logger.info(f"Storyboard JSON saved with images: {ts}_{slug}_storyboard.json")
            emit({"type": "storyboard", "content": sb.model_dump()})

    return {"script_doc": doc.model_dump(), "storyboard": sb.model_dump() if sb is not None else None}


async def storyboard(state: PipelineState, config: RunnableConfig) -> dict:
    """Break every script scene into individual storyboard shots (+ image prompts).

    Faithful to the script: each scene's voice_over / on_screen_text is split across
    its shots, nothing added or removed. Each shot gets a `prompt_img` for later
    image generation plus `image_subjects` for reference-image search. Runs one LLM
    call per scene, concurrently (capped). Image search + save/emit happen in the
    next node (fetch_images), which works on these shots.
    """
    from app.models.storyboard import StoryboardDocument, StoryboardScene, StoryboardShot

    svc = services(config)
    topic = state["topic"]
    cfg = state["script_config"]
    if not getattr(cfg, "storyboard", True):
        return {}
    # Surface the category (same as image_prompts/generate_images) so a food/drink gate mishap is visible.
    emit({"type": "status", "message": f"สร้าง storyboard (ประเภท: {_category_label(cfg.category)})..."})
    logger.info(f"storyboard category={cfg.category}")

    doc = state["script_doc_obj"]
    production = doc.production.model_dump()
    # Optionally drop Theme from prompt_img generation so the scene comes purely from
    # scene_ref (avoids the Theme text fighting the scene_ref image at generation).
    if not svc.settings.image_gen.use_theme_in_prompt:
        production["theme"] = ""
    scenes = [sc for part in doc.parts for sc in part.scenes]
    # Per-part auto-fit: a target total shot count for this (usually 1-part) doc → soft per-scene target.
    target_shots = int(state.get("target_shots") or 0)
    per_scene_shots = max(1, round(target_shots / len(scenes))) if target_shots and scenes else 0
    sem = asyncio.Semaphore(max(1, svc.settings.pipeline.analyze_concurrency))
    results: dict[str, list[StoryboardShot]] = {}

    # Director guide — visual sections steer how scenes are broken into shots.
    director_block = ""
    if cfg.director_id:
        d = director_store.get(cfg.director_id)
        if d:
            director_block = _director_block(
                d.sections,
                ["tone_mood", "pacing_editing", "shots_framing", "graphics_text_music", "do_dont"],
            )
    # Active brand's editing/camera shape join_with_prev + motion_description (flows to video prompts).
    brand_block = _brand_storyboard_block(scene_store.active_brand())
    director_block = "\n\n".join(b for b in (director_block, brand_block) if b.strip())

    with obs.span("node.storyboard", input={"scenes": len(scenes)}) as span:
        emit({"type": "status", "message": f"Breaking {len(scenes)} scenes into storyboard shots..."})

        async def _one(sc) -> None:
            async with sem:
                # Ingredient/equipment-introduction scene → EVERY shot is an insert (items only, no host),
                # per the storyboard spec — enforced here because the LLM drifts to person shots.
                kind = _intro_kind(sc)
                intro_scene = bool(kind)
                # REL-1: one scene's breakdown failing (429/500/bad JSON) must not blow up the whole
                # storyboard step via gather — isolate it, emit a warning, leave that scene empty.
                try:
                    # Pass the WHOLE scene (every field) so the breakdown uses all of it. Drop the shot-count
                    # target on intro scenes — its uniform per-scene number pressures the LLM to GROUP items
                    # into fewer shots (the "2 ingredients, 1 visual" bug); item count drives the shot count here.
                    shots_raw = await svc.gemini.breakdown_scene(sc.model_dump(), production, topic, director_prompt=director_block, aspect=svc.settings.video_gen.aspect_ratio, clip_seconds=svc.settings.video_gen.duration_seconds, words_per_second=cfg.words_per_second, shot_target=(0 if intro_scene else per_scene_shots), intro_scene=intro_scene, category=cfg.category)
                    # Deterministically enforce 1-shot-per-item + all-together overview (LLM can't be trusted to),
                    # and faithfully anchor-split THIS scene's voice_over so each item speaks its own line.
                    if intro_scene:
                        shots_raw = _rebuild_intro_shots(shots_raw, production, kind, scene_vo=getattr(sc, "voice_over", ""),
                                                         aspect=svc.settings.video_gen.aspect_ratio)
                        # Semantic pass on the split lines: the LLM moves ANY trailing lead-in phrase
                        # (phrasings the deterministic suffix gate can't enumerate) onto the next line;
                        # rebalance_intro_vo returns None unless the words stay verbatim → fail-open.
                        vo_lines = [str(s.get("voice_over", "")) for s in shots_raw]
                        vo_anchors = [(s.get("image_subjects") or [""])[0] for s in shots_raw]
                        fixed_lines = await svc.gemini.rebalance_intro_vo(vo_lines, vo_anchors)
                        if fixed_lines:
                            for s, v in zip(shots_raw, fixed_lines):
                                s["voice_over"] = v
                    # A caption belongs to ONE shot (STORYBOARD_PROMPT's ON-SCREEN TEXT rule); this
                    # guarantees it, the same way rebalance_intro_vo guarantees the voice_over split.
                    if (_blanked := _dedupe_scene_on_screen_text(shots_raw)):
                        emit({"type": "progress",
                              "message": f"📝 ฉาก {sc.scene_id}: on-screen text ซ้ำ — เก็บไว้ช็อตที่เกี่ยวที่สุด เว้นว่าง {_blanked} ช็อต"})
                    results[sc.scene_id] = [
                        StoryboardShot(
                            no=int(s.get("no", i) or i),
                            time=str(s.get("time", "")),
                            shot_kind="insert" if (intro_scene or str(s.get("shot_kind", "")).lower() == "insert") else "person",
                            join_with_prev=_norm_join({**s, "shot_kind": "insert"} if intro_scene else s),
                            screen_direction=str(s.get("screen_direction", "")),
                            motion_description=str(s.get("motion_description", "")),
                            voice_over=str(s.get("voice_over", "")),
                            on_screen_text=str(s.get("on_screen_text", "")),
                            key_message=str(s.get("key_message", "")),
                            prompt_img=str(s.get("prompt_img", "")),
                            image_subjects=[x for x in (s.get("image_subjects") or []) if isinstance(x, str) and x.strip()],
                            ingredient_refs=[x for x in (s.get("ingredient_refs") or []) if isinstance(x, str) and x.strip()],
                            equipment_refs=[x for x in (s.get("equipment_refs") or []) if isinstance(x, str) and x.strip()],
                        )
                        for i, s in enumerate(shots_raw, 1)
                    ]
                except Exception as e:  # noqa: BLE001 — best-effort per scene
                    logger.warning(f"storyboard breakdown failed for scene {getattr(sc, 'scene_id', '?')}: {e}")
                    results[sc.scene_id] = []
                    emit({"type": "progress", "message": f"⚠️ ข้ามฉาก {getattr(sc, 'scene_id', '?')} — breakdown ล้มเหลว (ฉากอื่นทำต่อ)"})

        await asyncio.gather(*(_one(sc) for sc in scenes))

        # Normalize prompt_img consistency (character + setting) within each scene.
        if svc.settings.pipeline.normalize_prompt_img:
            emit({"type": "status", "message": "Normalizing prompt_img consistency across shots..."})
            norm_tasks = []
            for sc in scenes:
                shots_in_scene = results.get(sc.scene_id, [])
                # Person shots only — the normalizer aligns CHARACTER/setting, and feeding insert shots
                # made it inject the host + theme props into object-only frames (must stay person-free).
                shots_payload = [{"no": sh.no, "prompt_img": sh.prompt_img}
                                 for sh in shots_in_scene if sh.prompt_img and sh.shot_kind == "person"]
                if shots_payload:
                    norm_tasks.append((sc.scene_id, shots_payload))

            async def _normalize_one(scene_id: str, payload: list) -> None:
                async with sem:
                    fixed = await svc.gemini.normalize_prompt_img_consistency(payload, production, topic)
                    for sh in results.get(scene_id, []):
                        if sh.no in fixed:
                            sh.prompt_img = fixed[sh.no]

            await asyncio.gather(*(_normalize_one(sid, p) for sid, p in norm_tasks))

        def _scene_transition(shots: list) -> str:
            # The scene opens however its FIRST shot joins the previous scene's last shot.
            # A within-scene-only join ('continuous') isn't a scene-level transition → 'cut'.
            j = shots[0].join_with_prev if shots else "cut"
            return j if j in {"cut", "dissolve", "fade", "match_cut"} else "cut"

        sb = StoryboardDocument(
            title=doc.title,
            ingredients=list(doc.production.ingredients),
            equipment=list(getattr(doc.production, "equipment", []) or []),
            production=doc.production.model_dump(),   # carried so prompt_img can be regenerated without the script
            scenes=[StoryboardScene(scene_id=sc.scene_id, name=sc.name,
                                    transition_in=_scene_transition(results.get(sc.scene_id, [])),
                                    music=getattr(sc, "music", "") or "",
                                    shots=results.get(sc.scene_id, []))
                    for sc in scenes],
        )
        # Honest timeline: overwrite the LLM's padded scene timecodes with a cumulative estimate of the
        # ACTUAL rendered length. A standalone shot is one ~clip_seconds Veo clip; a CONTINUOUS run of
        # >= min_chain_shots is rendered as one extension take (opening clip + ~7s per extra shot).
        # (Estimate only — interpolation/use_last_frame=8s isn't known until Step 5; ffprobe gives the truth.)
        clip = max(1, svc.settings.video_gen.duration_seconds)
        ext = 7   # Veo extension adds ~7s per chained shot (Gemini API docs)
        min_chain = max(1, svc.settings.video_gen.min_chain_shots)
        def _ts(sec: int) -> str:
            return f"{sec // 60}:{sec % 60:02d}"
        _t = 0
        for scn in sb.scenes:
            shots = scn.shots
            i = 0
            while i < len(shots):
                j = i + 1
                while j < len(shots) and shots[j].join_with_prev == "continuous":  # group a continuous run
                    j += 1
                chained = (j - i) >= min_chain   # a long enough run renders as one extension take
                for k in range(i, j):
                    dur = clip if (k == i or not chained) else ext   # opening/standalone = clip; extension = +7s
                    shots[k].time = f"{_ts(_t)} - {_ts(_t + dur)}"
                    _t += dur
                i = j

        # Deterministic vessel pass (same block as the image_prompts node) — Step 3's visible prompt_img
        # must obey the vessel rules too, not only the Step 4/5 regens (the LLM occasionally ignores them).
        for scn in sb.scenes:
            for sh in scn.shots:
                if sh.shot_kind == "insert" and sh.prompt_img:
                    sh.prompt_img = _liquid_vessel_fix(sh.prompt_img, sh.ingredient_refs, cfg.category)
                    if cfg.category == "drink":
                        sh.prompt_img = _spoon_vessel_fix(sh.prompt_img, sh.ingredient_refs)
                if sh.prompt_img and cfg.category == "drink":
                    sh.prompt_img = _ml_from_spoon_fix(sh.prompt_img)
                # Every shot, process included: a pan/board/pot clause carries its own vessel word and
                # is skipped, so this only rescues food genuinely left on the bare counter.
                sh.prompt_img = _bare_counter_vessel_fix(sh.prompt_img, sh.ingredient_refs, cfg.category)

        total_shots = sum(len(s) for s in results.values())
        logger.info(f"Storyboard: {len(sb.scenes)} scenes → {total_shots} shots (~{_ts(_t)} est. runtime)")
        span.update(output={"scenes": len(sb.scenes), "shots": total_shots, "runtime_seconds": _t})

    return {"storyboard_obj": sb}


async def image_prompts(state: PipelineState, config: RunnableConfig) -> dict:
    """Re-generate every shot's prompt_img on an EXISTING storyboard (the "Image Prompt" step).

    Decoupled from the text breakdown: the shot set + voice_over/on_screen_text/motion stay untouched —
    only prompt_img is rewritten (then normalized for character/setting consistency). Needs the script's
    production spec, so the route seeds `script_doc_obj` into state alongside `storyboard_obj`.
    """
    svc = services(config)
    topic = state["topic"]
    sb = state["storyboard_obj"]
    doc = state.get("script_doc_obj")
    # Production spec: prefer the script (Step 2); else the copy embedded in the storyboard itself.
    production = doc.production.model_dump() if doc is not None else dict(sb.production or {})
    cat = getattr(state.get("script_config"), "category", "food")   # food|drink → drink-only prompt rules
    # Surface the category so a food/drink gate mishap is visible immediately (SSE + log).
    emit({"type": "status", "message": f"เขียน prompt รูป (ประเภท: {_category_label(cat)})..."})
    logger.info(f"image_prompts category={cat}")
    if not production:
        emit({"type": "status", "message": "No production spec (script) available — cannot regenerate image prompts."})
        return {}
    # Same as the storyboard node: optionally drop Theme so the scene comes purely from scene_ref.
    if not svc.settings.image_gen.use_theme_in_prompt:
        production["theme"] = ""
    aspect = svc.settings.video_gen.aspect_ratio
    sem = asyncio.Semaphore(max(1, svc.settings.pipeline.analyze_concurrency))

    with obs.span("node.image_prompts", input={"scenes": len(sb.scenes)}) as span:
        # dish_state timeline — compute each shot's ACTUAL dish/drink state across the WHOLE ordered
        # sequence FIRST (an ingredient joins the dish only at the step that adds it), so every prompt
        # below renders the food at its real stage, not the finished look pasted from dish_appearance.
        flat = [sh for sc in sb.scenes for sh in sc.shots]
        # NEW-1: a single-shot regen (the "↻ Prompt" button sends ONE shot) has no real timeline —
        # recomputing dish_state from one shot in isolation would overwrite the correct state that
        # the full-storyboard pass already set. Skip the recompute; the shot keeps the dish_state it
        # already carries (the frontend sends it back on the shot).
        if len(flat) > 1:
            ds_payload = [
                {"id": i, "motion_description": sh.motion_description,
                 "voice_over": sh.voice_over, "ingredient_refs": sh.ingredient_refs}
                for i, sh in enumerate(flat)
            ]
            emit({"type": "status", "message": "Computing dish-state timeline..."})
            try:
                ds = await svc.gemini.dish_state_timeline(
                    ds_payload, production, topic, research_summary=state.get("synthesis", "") or "")
            except Exception as e:  # noqa: BLE001 — dish_state is an enhancement; never crash the step over it
                logger.warning(f"dish_state timeline failed (shots keep their prior/empty state): {e}")
                ds = {}
            # NEW-6: a full recompute is authoritative — clear the state on shots the new timeline omitted
            # (an edited storyboard would otherwise keep a stale dish_state). Guard on `ds` being non-empty:
            # a failed timeline call returns {} above, and must NOT wipe every shot's existing state.
            if ds:
                for i, sh in enumerate(flat):
                    sh.dish_state = ds.get(i, "")

        emit({"type": "status", "message": f"Regenerating image prompts for {len(sb.scenes)} scenes..."})

        async def _one(scene) -> None:
            payload = [
                {"no": sh.no, "shot_kind": sh.shot_kind, "motion_description": sh.motion_description,
                 "voice_over": sh.voice_over, "on_screen_text": sh.on_screen_text,
                 "image_subjects": sh.image_subjects, "ingredient_refs": sh.ingredient_refs,
                 "dish_state": sh.dish_state}
                for sh in scene.shots
            ]
            if not payload:
                return
            async with sem:
                fixed = await svc.gemini.generate_image_prompts(payload, production, topic, aspect, category=cat,
                                                               extra_ref_notes=state.get("ref_notes") or [])
            for sh in scene.shots:
                if sh.no in fixed:
                    sh.prompt_img = fixed[sh.no]

        await asyncio.gather(*(_one(sc) for sc in sb.scenes))

        # Same character/setting consistency pass the storyboard node runs.
        if svc.settings.pipeline.normalize_prompt_img:
            emit({"type": "status", "message": "Normalizing prompt_img consistency across shots..."})

            async def _normalize_one(scene) -> None:
                # Person shots only — see storyboard node: inserts must stay person/prop-free.
                shots_payload = [{"no": sh.no, "prompt_img": sh.prompt_img}
                                 for sh in scene.shots if sh.prompt_img and sh.shot_kind == "person"]
                if not shots_payload:
                    return
                async with sem:
                    fixed = await svc.gemini.normalize_prompt_img_consistency(shots_payload, production, topic)
                for sh in scene.shots:
                    if sh.no in fixed:
                        sh.prompt_img = fixed[sh.no]

            await asyncio.gather(*(_normalize_one(sc) for sc in sb.scenes))

        # Deterministic vessel pass (runs LAST, after any normalization): a liquid ingredient shown in an
        # insert must be in a measuring cup, not a bowl (the LLM keeps writing bowl).
        for scene in sb.scenes:
            for sh in scene.shots:
                if sh.shot_kind == "insert" and sh.prompt_img:
                    sh.prompt_img = _liquid_vessel_fix(sh.prompt_img, sh.ingredient_refs, cat)
                    if cat == "drink":
                        sh.prompt_img = _spoon_vessel_fix(sh.prompt_img, sh.ingredient_refs)
                if sh.prompt_img and cat == "drink":   # person/process shots too: ml never pours from a spoon
                    sh.prompt_img = _ml_from_spoon_fix(sh.prompt_img)
                sh.prompt_img = _bare_counter_vessel_fix(sh.prompt_img, sh.ingredient_refs, cat)
                if sh.prompt_img:
                    sh.prompt_img = _milk_state_fix(sh.prompt_img, sh.dish_state)

        span.update(output={"scenes": len(sb.scenes), "shots": sum(len(s.shots) for s in sb.scenes)})

    return {"storyboard_obj": sb}


def _selected_scene(svc, scene_match: str | None) -> tuple[str, str]:
    """(image_src, display_name) for a shot's place ref: a configured scene the classifier matched
    (scene_match) or the default one; else the legacy single scene_ref (display "kitchen")."""
    scenes = scene_store.active_scenes()
    if scenes:
        sel = next((s for s in scenes if s.get("id") == (scene_match or "")), None) or scene_store.active_default_scene()
        if sel and sel.get("image"):
            return sel["image"], (sel.get("name") or sel.get("id") or "")
    return svc.settings.image_gen.scene_ref, "kitchen"


async def load_base_image_refs(svc, scene_match: str | None = None) -> list[dict]:
    """The character + place reference images applied to a shot. The place is the configured scene
    the classifier matched (scene_match) — falling back to the default scene / legacy scene_ref."""
    cfg_gen = svc.settings.image_gen
    icfg = svc.settings.image_search
    scene_src, scene_name = _selected_scene(svc, scene_match)
    scene_label = ("Background reference — the kitchen's IDENTITY (same fixtures, materials, layout and colours); "
                   "reproduce that identity, but the CAMERA ANGLE, crop and framing follow THIS shot's "
                   "description — do NOT copy the reference's angle")
    if scene_name and scene_name != "kitchen":
        scene_label += f" — {scene_name}"
    base_refs: list[dict] = []
    # The run's chosen host (falling back to the default character); config.yaml's character_ref is
    # only the floor for an install with no characters at all. This is the seam that made a second
    # host impossible: the scene already resolved per run, the character never did.
    char_src = character_store.active_character_ref() or cfg_gen.character_ref
    for src, label, max_px in (
        (char_src, "Character reference — the host's IDENTITY (face, hair, outfit, skin tone); keep her identical. For a HAND-ONLY shot take ONLY her hand, skin tone and jewelry from this — do NOT render her full body or face.", cfg_gen.character_ref_resize_px),
        (scene_src, scene_label, cfg_gen.ref_resize_px),
    ):
        ref = await load_reference(src, cfg=icfg, max_px=max_px) if src else None
        if ref:
            base_refs.append({"label": label, **ref})
    return base_refs


def _apply_text_rules(prompt: str, on_screen_text: str, cfg_gen) -> str:
    """Append the on-screen-text instruction (render it, or leave clean negative space)."""
    if cfg_gen.on_screen_text_in_image and on_screen_text.strip():
        # The image model draws the caption itself, so it can never be pixel-consistent across shots;
        # pin one fixed treatment (position/font style/legibility) so every shot looks the same family.
        return prompt + (
            '\n\nAlso render EXACTLY this Thai on-screen caption into the image (this text only, '
            f'no other words/labels/logos anywhere): "{on_screen_text}". Caption STYLE — keep it IDENTICAL '
            'on every shot: place it as a lower-third caption centered along the BOTTOM of the frame; a clean '
            'modern Thai sans-serif in a formal broadcast lower-third style, one consistent size and weight; '
            'white text with a thin dark outline/soft shadow so it stays legible on any background; tone tasteful '
            'and matching the scene mood. Spell the Thai correctly and keep it on one or two lines.')
    return prompt + (
        "\n\nIMPORTANT: Do NOT render any text, letters, words, captions, numbers, "
        "labels, logos, or lettered motion-graphics in the image. Even if the description "
        "above mentions on-screen text or a graphic label, show that spot as CLEAN EMPTY "
        "space (negative space) — leave room for a caption to be added later in editing. "
        "Render the scene, character and props only, with no written words anywhere. "
        "This ALSO applies to any reference photo of an ingredient, package, product or "
        "container provided alongside this prompt: match ONLY its shape, color and material — "
        "if the reference has printed text, a brand name, or a label on it, OMIT that text "
        "entirely and render the surface as plain/unlabeled instead of copying it."
    )


def _scene_host_rule(layout: str) -> str:
    """The host-position sentence(s) from a scene's `layout` text, or "" when it has none.

    SCENE_LAYOUT_PROMPT makes every AI-drafted layout state where a person belongs in
    camera-relative terms ("on the far side of the hob from the camera, …"), but the exact
    wording is the model's own — the verb varies (should/must) and hand-written layouts may
    phrase it differently. So: split into sentences, keep the ones that mention the host.
    "" (no such sentence) tells the caller to fall back to the generic COMPOSITION wording.
    """
    sentences = re.split(r"(?<=[.])\s+", layout or "")
    picked = [s.strip() for s in sentences if re.search(r"\bhost\b", s, re.IGNORECASE)]
    return " ".join(picked)


def assemble_full_prompt(svc, prompt: str, refs: list[dict]) -> str:
    """The exact text the image provider will receive for `prompt` + `refs` — what the UI shows/edits
    as the shot's 'full prompt' (ครบ). Mirrors _provider_generate's provider branch."""
    if svc.settings.image_gen.provider == "openai":
        return openai_image._ref_instruction(refs, prompt)
    return gemini_client._ref_instruction_gemini(refs, prompt)


async def _provider_generate(svc, prompt: str, refs: list[dict], prebuilt: str | None = None) -> bytes | None:
    """Send a finalized prompt + ordered refs to the configured image provider.
    prebuilt = an already-assembled full prompt to send verbatim (skips the provider's ref wrapper)."""
    cfg_gen = svc.settings.image_gen
    if cfg_gen.provider == "openai":
        return await openai_image.generate_image(
            prompt, refs, api_key=svc.openai_api_key,
            model=cfg_gen.openai.model, size=cfg_gen.openai.size, quality=cfg_gen.openai.quality,
            prebuilt=prebuilt,
        )
    return await svc.gemini.generate_image(prompt, references=refs, prebuilt=prebuilt)


async def _qc_generate(svc, prompt: str, refs: list[dict], shot_ctx: dict, topic: str) -> tuple[bytes | None, dict]:
    """C1 (output QC): generate → Gemini LOOKS at the result → regenerate on FAIL, up to
    image_gen.output_qc.max_regens extra attempts (the QC feedback is appended to the prompt).
    The LAST render is always kept — a shot that never passes ships flagged, never empty.
    Returns (image_bytes|None, {"qc_passed","qc_attempts","qc_issues"}); QC off → verdict is the
    permissive default and this is exactly one _provider_generate call."""
    qc_cfg = svc.settings.image_gen.output_qc
    data = await _provider_generate(svc, prompt, refs)
    info = {"qc_passed": True, "qc_attempts": 0, "qc_issues": []}
    if not (qc_cfg.enabled and data):
        return data, info
    verdict = await svc.gemini.qc_generated_image({"mime": "image/png", "data": data}, shot_ctx, topic)
    info["qc_attempts"] = 1

    def _feedback(v: dict) -> str:
        """The QC-FEEDBACK block for a failed verdict — appended on auto-retry, and surfaced on the
        plan (qc_feedback / qc_retry_prompt) so the USER can choose to attach it on a manual regen."""
        issues = "; ".join(v.get("issues") or []) or "the QC rules listed"
        hint = (v.get("fix_hint") or "").strip()
        return ("QC FEEDBACK — the previous render FAILED review because: "
                f"{issues}. {hint} Re-render this SAME shot correcting ONLY these problems; "
                "keep the composition, subjects, lighting and references otherwise identical.")

    regens = 0
    while not verdict.get("pass") and regens < max(0, qc_cfg.max_regens):
        regens += 1
        retry_prompt = prompt + "\n\n" + _feedback(verdict)
        info["qc_retry_prompt"] = retry_prompt   # surfaced on image_plan → UI shows what the model was sent
        logger.info(f"image QC: regen {regens}/{qc_cfg.max_regens} ({verdict.get('issues')})")
        data2 = await _provider_generate(svc, retry_prompt, refs)
        if not data2:
            break   # regen produced nothing — keep the previous render (flagged below)
        data = data2
        verdict = await svc.gemini.qc_generated_image({"mime": "image/png", "data": data}, shot_ctx, topic)
        info["qc_attempts"] += 1
    if not verdict.get("pass"):
        # judge-only (max_regens=0) or still failing after retries → hand the user the feedback
        # so their next manual Generate can attach it (ImageStepRequest.qc_feedback).
        info["qc_feedback"] = _feedback(verdict)
        info["qc_retry_prompt"] = prompt + "\n\n" + info["qc_feedback"]
    info["qc_passed"] = bool(verdict.get("pass"))
    info["qc_issues"] = [] if info["qc_passed"] else [str(i) for i in (verdict.get("issues") or [])]
    if not info["qc_passed"]:
        logger.warning(f"image QC: kept a FAILED render after {info['qc_attempts']} review(s): {info['qc_issues']}")
    return data, info


def _qc_ctx(shot_kind: str, framing: str, must_show: list[str], description: str, expect_text: bool) -> dict:
    """Shot-intent context for qc_generated_image (capped so an overview can't over-fail)."""
    return {"shot_kind": shot_kind, "framing": framing,
            "must_show": [s for s in must_show if isinstance(s, str) and s.strip()][:8],
            "description": (description or "").strip(), "expect_text": bool(expect_text)}


def _media_path(url: str) -> Path | None:
    """Map a media URL to a local file path.
    Handles /api/media/ local URLs and S3 URLs (reconstructed from AWS_URL prefix).
    Rejects path traversal attempts."""
    if not url:
        return None
    if "/api/media/" in url:
        rel = url.split("/api/media/", 1)[-1]
        try:
            p = (_MEDIA_DIR / rel).resolve()
            p.relative_to(_MEDIA_DIR.resolve())
        except (ValueError, OSError):
            logger.warning(f"rejected media path outside media dir: {rel!r}")
            return None
        return p
    # S3/CDN URL: reconstruct local path (Phase 2 always writes locally before uploading)
    from urllib.parse import unquote
    aws_base = os.getenv("AWS_URL", "").rstrip("/")
    if aws_base and url.startswith(aws_base + "/"):
        key = unquote(url[len(aws_base) + 1:])   # URL is percent-encoded; on-disk name is raw
        try:
            p = (_MEDIA_DIR / key).resolve()
            p.relative_to(_MEDIA_DIR.resolve())
            return p
        except (ValueError, OSError):
            return None
    return None


def _read_media_bytes(url: str) -> bytes | None:
    """Read raw bytes from a /api/media/ URL (local), S3 URL (local copy), or generic HTTP URL."""
    p = _media_path(url)
    if p and p.exists():
        return p.read_bytes()
    if (url or "").startswith(("http://", "https://")):
        import httpx
        try:
            r = httpx.get(url, timeout=30, follow_redirects=True)
            r.raise_for_status()
            return r.content
        except Exception as exc:
            logger.warning(f"_read_media_bytes fetch failed {url}: {exc}")
    return None


def _ref_src(url: str) -> str:
    """Resolve an app-relative ref URL to its file on disk so load_reference can read it:
    /api/media/… → outputs/media; /api/menus/…/subjects/…/preview → the menu subject photo;
    /api/brands/…/scenes/…/preview → the brand scene photo. http(s)/plain paths pass through.

    When S3 is configured, `_persist_media` uploads and deliberately keeps NO local copy — so an
    S3-URL shot.generated_img has no file for `_media_path` to find. Only take the local-path
    shortcut when that file actually exists; otherwise fall through to the original URL so
    load_reference() downloads it over HTTP instead of logging a false "not found"."""
    from urllib.parse import unquote
    u = (url or "").strip()
    mp = _media_path(u)
    if mp and mp.exists():
        return str(mp)
    m = re.match(r"^/api/menus/([^/]+)/subjects/([^/]+)/preview", u)
    if m:
        ref = subject_ref_store.subject_image_ref(unquote(m.group(1)), unquote(m.group(2)))
        return ref or u
    m = re.match(r"^/api/brands/([^/]+)/scenes/([^/]+)/preview", u)
    if m:
        ref = scene_store.scene_image_ref(unquote(m.group(1)), unquote(m.group(2)))
        return ref or u
    return u


async def render_shot_image(svc, prompt_img: str, on_screen_text: str, image_results, base_refs: list[dict], shot_kind: str = "person") -> bytes | None:
    """Render ONE shot's image (prompt_img + on-screen-text handling + subject refs on
    top of base character/scene refs). Returns PNG bytes or None. Shared by the batch
    node and the per-shot endpoint. `image_results` items may be dicts or ImageElement.

    shot_kind="insert" → an object/food close-up with no person, so the Character
    reference is dropped (else the host gets composited into an object shot)."""
    cfg_gen = svc.settings.image_gen
    icfg = svc.settings.image_search
    aspect = svc.settings.video_gen.aspect_ratio
    # Insert shots show only food/objects close-up → drop BOTH the character AND the wide
    # kitchen (scene) references; the eye-level wide kitchen forces a wrong perspective and
    # makes top-down objects float. Keep only subject reference photos (added below).
    if shot_kind == "insert":
        base_refs = []
        prompt = (f"{prompt_img}\n\nThis is an INSERT / product shot: a clean top-down (or close-up) of the "
                  f"food/objects resting FLAT on a plain white marble kitchen counter — fill the frame with them, "
                  f"NO person, NO wide room, nothing floats. Render as a {aspect} widescreen, cinematic frame.")
    else:
        # Reinforce the target aspect; the frame feeds Veo image-to-video so it must match the video aspect.
        prompt = (f"{prompt_img}\n\nRender as a {aspect} widescreen, cinematic storyboard frame; "
                  f"match the kitchen's identity and fixtures from the Background reference — the camera angle "
                  f"and framing follow this shot, not the reference's angle.")
    prompt = _apply_text_rules(prompt, on_screen_text, cfg_gen)
    refs = base_refs
    if cfg_gen.subject_refs_in_image and image_results and cfg_gen.max_subject_refs > 0:
        subj_refs: list[dict] = []
        for el in image_results:
            if len(subj_refs) >= cfg_gen.max_subject_refs:
                break
            cands = el.get("candidates") if isinstance(el, dict) else getattr(el, "candidates", [])
            subj = el.get("subject") if isinstance(el, dict) else getattr(el, "subject", "")
            url = next((c for c in (cands or []) if c), "")
            if not url:
                continue
            r = await load_reference(url, cfg=icfg, max_px=cfg_gen.ref_resize_px)
            if r:
                subj_refs.append({"label": f'Reference photo of "{subj}" — match this real object\'s appearance', **r})
        refs = base_refs + subj_refs
    return await _provider_generate(svc, prompt, refs)


_FIXTURES_FILE = _OUTPUTS_DIR / "kitchen_fixtures.json"


def load_kitchen_fixtures(svc) -> list[str]:
    """Fixture list used at GENERATION — the configured/edited file if present, else the static default.
    Never runs vision; extraction happens once when the scene image is configured (UI)."""
    try:
        if _FIXTURES_FILE.exists():
            data = json.loads(_FIXTURES_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [s for s in data if isinstance(s, str) and s.strip()]
    except Exception as e:  # noqa: BLE001
        logger.warning(f"kitchen_fixtures load failed: {e}")
    return list(svc.settings.image_gen.kitchen_fixtures)


def save_kitchen_fixtures(fixtures: list[str]) -> list[str]:
    clean = [s.strip() for s in (fixtures or []) if isinstance(s, str) and s.strip()]
    _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    _FIXTURES_FILE.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")
    return clean


async def extract_kitchen_fixtures_from_scene(svc) -> list[str]:
    """Vision-extract fixtures from the current scene_ref and store them (called when the scene image
    is configured). Returns the stored list (unchanged on failure)."""
    cfg_gen = svc.settings.image_gen
    src = (cfg_gen.scene_ref or "").strip()
    if not src:
        return load_kitchen_fixtures(svc)
    ref = await load_reference(src, cfg=svc.settings.image_search, max_px=cfg_gen.character_ref_resize_px)
    if not ref:
        return load_kitchen_fixtures(svc)
    try:
        extracted = await svc.gemini.extract_kitchen_fixtures(ref)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"kitchen_fixtures extract failed: {e}")
        return load_kitchen_fixtures(svc)
    return save_kitchen_fixtures(extracted) if extracted else load_kitchen_fixtures(svc)


# Person words in a (English) prompt_img → the shot actually shows the host/hand, so her identity ref must be
# fed even if the classifier marked the shot human-less (e.g. an ingredient intro where she still holds the item).
_PERSON_RE = re.compile(
    r"\b(chef|cook|host|hostess|woman|women|man|men|human|person|people|girl|boy|hand|hands|she|her|"
    r"barista|presenter|waiter|waitress)\b"
    r"|ผู้หญิง|ผู้ชาย|เชฟ|ครูพี่|พิธีกร", re.I)


def _prompt_has_person(text: str) -> bool:
    return bool(_PERSON_RE.search(text or ""))


# The three vessels the liquid rules move things between. Written once because the same phrase is
# needed by every fix below and by the flat-lay suffix — and because "measuring cup" needed a wording
# change in ONE place: it used to say "with measurement markings", and a clear glass with ml markings
# is exactly what the model renders as a laboratory beaker.
_MEASURING_CUP = ("in a plain clear glass measuring cup — NO numbers, NO measurement lines, "
                  "no pattern or logo on the glass")
_SMALL_CUP = "in a small plain cup (a round dish with no handle)"
_OIL_BOTTLE = "in its clear glass oil bottle"

# The LLM keeps putting LIQUID ingredients in the wrong vessel despite the prompt rule. Deterministic
# backstop: a liquid is identified by a VOLUME unit in its name (ml/มิลลิลิตร/ลิตร/cc); solids
# (กรัม/ช้อนโต๊ะ/หัว/ฟอง) are left alone. Which vessel it moves to depends on the CATEGORY — see
# _liquid_vessel_fix.
_LIQUID_UNIT_RE = re.compile(r"\d+\s*(?:ml|cc|มล\.?|มิลลิลิตร|ลิตร|litre?s?|liters?)\b", re.I)
_BOWL_VESSEL_RE = re.compile(r"\bin(?:to)?\s+an?\s+[a-z\s\-]*?bowl\b", re.I)
_CUP_VESSEL_RE = re.compile(r"\bin(?:to)?\s+an?\s+[a-z\s\-]*?measuring\s+cup\b", re.I)

# Substance, not unit. Oil is matched by NAME because a real recipe writes it without a volume
# ("น้ำมันปาล์มสำหรับทอด ปริมาณพอเหมาะ"), so the unit-based liquid test never sees it. น้ำมันหอย is
# oyster SAUCE, not oil — it belongs in the small-cup bucket with the other sauces.
_OIL_ITEM_RE = re.compile(r"น้ำมัน(?!หอย)|\boil\b", re.I)
# Plain water only, and spelled out: nearly every Thai liquid starts with น้ำ (น้ำปลา, น้ำมะนาว,
# น้ำมัน) and น้ำตาล is not even a liquid — a bare "น้ำ" test would sweep sugar into a measuring cup.
_WATER_ITEM_RE = re.compile(r"น้ำเปล่า|น้ำสะอาด|น้ำกรอง|น้ำอุ่น|น้ำร้อน|น้ำเย็น|น้ำแร่|\bwater\b", re.I)


def _is_liquid_item(name: str) -> bool:
    return bool(_LIQUID_UNIT_RE.search(name or ""))


def _is_oil_item(name: str) -> bool:
    return bool(_OIL_ITEM_RE.search(name or ""))


def _is_water_item(name: str) -> bool:
    """Plain water — checked AFTER oil by every caller, since น้ำมัน contains น้ำ."""
    return not _is_oil_item(name) and bool(_WATER_ITEM_RE.search(name or ""))


def _liquid_vessel_fix(prompt_img: str, ingredient_refs: list[str] | None, category: str = "food") -> str:
    """Put this shot's liquid ingredient in the vessel its CATEGORY calls for.

    A drink is mixed and measured in glassware, so every liquid there goes in the measuring cup — the
    rule this function has always enforced. Food is different and used to be forced into the same cup:
    sauces and stock belong in a small cup on a Thai cooking set, and oil is shown in the bottle it is
    sold in. Only plain water still earns the measuring cup.

    Note the food branch works in the OPPOSITE direction from the drink one: a bowl is already right,
    so it is left alone, and it is the measuring cup written by an older prompt that gets replaced."""
    refs = ingredient_refs or []
    if not prompt_img:
        return prompt_img
    if any(_is_oil_item(n) for n in refs):     # oil first: น้ำมัน contains น้ำ
        if category == "drink":
            return _BOWL_VESSEL_RE.sub(_MEASURING_CUP, prompt_img)
        return _CUP_VESSEL_RE.sub(_OIL_BOTTLE, _BOWL_VESSEL_RE.sub(_OIL_BOTTLE, prompt_img))
    if not any(_is_liquid_item(n) for n in refs):
        return prompt_img
    if category == "drink" or any(_is_water_item(n) for n in refs):
        return _BOWL_VESSEL_RE.sub(_MEASURING_CUP, prompt_img)
    return _CUP_VESSEL_RE.sub(_SMALL_CUP, prompt_img)


# Food left loose on the bare counter. The three prompt authors all carry a "food is never loose on the
# counter" rule now, and it fixed roughly half the cases on a measured re-run (2 of 4 known-bad shots) —
# the other half still wrote "700 grams of raw chicken mid-wings resting on a light-colored marble
# countertop". Same shape as the bowl/spoon fixes below: the prompt asks, this guarantees.
# Only fires when the clause has NO vessel word in front of it, so a cutting board, pan, pot, plate or
# cooling rack — all legitimate places for food to sit — is left alone, as are tool-only shots (no
# ingredient_refs), which may rest on the bare counter.
_BARE_SURFACE_RE = re.compile(
    r"\b(resting|sitting|placed|laid|lying|arranged|scattered|piled|set)\s+(?:directly\s+)?"
    r"on\s+(the|a)\s+([^,.;]{0,40}?(?:counter|countertop|surface|worktop|marble|granite)\w*)", re.I)
_VESSEL_WORD_RE = re.compile(
    r"\b(bowl|cup|plate|tray|board|pan|pot|glass|jar|dish|spoon|rack|colander|strainer|basket|container|sheet|bottle)\b",
    re.I)


def _bare_counter_vessel_fix(prompt_img: str, ingredient_refs: list[str] | None, category: str = "food") -> str:
    """Put food into a vessel when the prompt left it lying on the bare counter.

    Vessel by the ingredient's own nature AND its category, the same rule the prompts state: oil goes
    in its bottle, plain water (or any liquid in a DRINK) in the measuring cup, another food liquid in
    a small cup, a lone spoon-measured item on a measuring spoon, everything else in a mixing bowl.
    ponytail: one generic bowl for the whole "else" branch — a plate would suit a few whole-piece
    items better; revisit only if renders show the bowl looking wrong."""
    refs = ingredient_refs or []
    if not prompt_img or not refs:
        return prompt_img
    if any(_is_oil_item(n) for n in refs) and category != "drink":   # oil first: น้ำมัน contains น้ำ
        vessel = _OIL_BOTTLE
    elif any(_is_liquid_item(n) for n in refs):
        vessel = (_MEASURING_CUP if category == "drink" or any(_is_water_item(n) for n in refs)
                  else _SMALL_CUP)
    elif len(refs) == 1 and _is_spoon_item(refs[0]):
        vessel = "in a stainless steel measuring spoon"
    else:
        vessel = "in a stainless steel mixing bowl"

    def _sub(m: re.Match) -> str:
        # The match itself is checked too, not just the run-up: "resting on a light wooden cutting
        # board on the marble counter" reaches `marble` inside the same clause, and skipping only on
        # the preceding text would have stuffed a mixing bowl in front of a perfectly good board.
        if _VESSEL_WORD_RE.search(prompt_img[max(0, m.start() - 70):m.start()] + m.group(0)):
            return m.group(0)
        return f"{m.group(1)} {vessel} on {m.group(2)} {m.group(3)}"

    return _BARE_SURFACE_RE.sub(_sub, prompt_img)


# A bowl or cup that already holds an ingredient, with a colour or material stuck on it. The photo of
# that bowl is attached to the render, so the adjective can only disagree with it — measured on the
# ปีกไก่ทอดคลุกผง board, ONE mixing bowl was written "large white" in three shots and "stainless steel"
# in a fourth. Only bowl/cup: a plate, pan, pot, board or lidded container is usually where a shot
# MOVES the food, which makes it new to the frame and legitimately describable in full.
_VESSEL_ADJ_RE = re.compile(
    r"\b(?:white|black|clear|glass|stainless(?:\s+steel)?|steel|metal|metallic|ceramic|porcelain|"
    r"silver|grey|gray|beige|cream|blue|green|red|matte|glossy|wooden|wood)\s+"
    r"(?=(?:\w+\s+){0,2}(?:bowl|cup)\b)", re.I)


def _vessel_colour_fix(prompt_img: str) -> str:
    """Strip colour/material adjectives off a bowl or cup so the reference photo decides.

    The rule is also in the prompt (IMAGE_PROMPT_VESSEL_FROM_REF_BLOCK), but asking is not enough
    here: the shot JSON handed to the author carries its OWN previous prompt_img, and a concrete
    "a large white mixing bowl" sitting in the input beats an abstract instruction — the block alone
    only took 7 coloured vessels down to 5. Caller gates this to non-intro shots; an INTRODUCTION is
    where the colour is supposed to be written, since it creates the photo.

    ponytail: adjective strip, not a rewrite — "a large white mixing bowl" → "a large mixing bowl".
    The article is left as-is, so a stripped "an off-white cup" could read "an cup"; no colour in the
    list starts with a vowel sound, so it cannot happen with these words."""
    return _VESSEL_ADJ_RE.sub("", prompt_img or "")


# Drink-only sibling of _liquid_vessel_fix: a DRY ingredient measured by SPOON (ช้อนตวง/ช้อนโต๊ะ/ช้อนชา) belongs
# on a measuring spoon, not a bowl (the unit in the ingredient name says so). Gated to a SINGLE-ingredient shot
# (an intro/show frame) so an overview flat-lay mixing spoon + gram + piece units isn't blindly swapped.
_SPOON_UNIT_RE = re.compile(r"(?:ช้อน(?:ตวง|โต๊ะ|ชา)?|tbsp|tsp|tablespoons?|teaspoons?)", re.I)


def _is_spoon_item(name: str) -> bool:
    return bool(_SPOON_UNIT_RE.search(name or ""))


def _spoon_vessel_fix(prompt_img: str, ingredient_refs: list[str] | None) -> str:
    """Drink recipes only: if the shot's SOLE ingredient is spoon-measured and the prompt shows a bowl, swap it
    for a measuring spoon. Caller gates on category == 'drink' (grams/pieces keep their bowl)."""
    refs = ingredient_refs or []
    if not prompt_img or len(refs) != 1 or not _is_spoon_item(refs[0]):
        return prompt_img
    return _BOWL_VESSEL_RE.sub("in a stainless steel measuring spoon", prompt_img)


# Drink-only: the prompt writer keeps making ml-measured liquids POUR "from a measuring spoon"
# ("pouring 20 ml of evaporated milk from a small metal measuring spoon") — an ml amount doesn't fit a
# spoon and the render looks wrong. Swap the SOURCE vessel to a small glass measuring cup, only when the
# same clause carries an ml amount (a dry "1 tablespoon ... from a measuring spoon" is correct and untouched).
_ML_FROM_SPOON_RE = re.compile(
    r"(\b(?:pour|add|drizzl)\w*\s[^.]*?\d+\s*(?:ml|cc|มล\.?|มิลลิลิตร|millilit\w*)\b[^.]*?)"
    r"\bfrom\s+a[a-z\s\-]*\bmeasuring\s+spoon", re.I)


def _ml_from_spoon_fix(prompt_img: str) -> str:
    """Drink recipes only (caller gates): ml liquid poured 'from a ... measuring spoon' → from a measuring cup."""
    if not prompt_img:
        return prompt_img
    return _ML_FROM_SPOON_RE.sub(r"\1from a small plain clear glass measuring cup (no numbers, no lines)", prompt_img)


# dish_state is authoritative; when it has NO milk but the LLM still leaked "with milk" into the prompt
# (a pre-milk brew step), strip the milk phrase deterministically — the writers keep copying the finished
# "with milk" look from the stale prompt they rewrite. ponytail: milk-only; extend if other states leak.
_MILK_LEAK_RE = re.compile(r",?\s*\bwith\s+(?:condensed\s+|evaporated\s+|sweetened\s+|creamy\s+)*milk\b", re.I)


def _milk_state_fix(prompt_img: str, dish_state: str) -> str:
    if not prompt_img or not dish_state:
        return prompt_img
    ds = dish_state.lower()
    # dish genuinely has milk only if it mentions milk/cream and NOT as a negation ("no milk yet")
    has_milk = bool(re.search(r"milk|cream", ds)) and "no milk" not in ds and "without milk" not in ds
    if has_milk:
        return prompt_img
    return _MILK_LEAK_RE.sub("", prompt_img)


async def resolve_shot_refs(svc, plan: dict, ref_keywords: list[str], image_results, prev_ref: dict | None,
                            ing_refs: list[dict] | None = None, state_ref: dict | None = None,
                            cap: int | None = None,
                            scene_match: str | None = None, eq_refs: list[dict] | None = None,
                            gen_map: dict | None = None, allow_config: bool = True,
                            person_override: str = "", scene_override: str = "",
                            edit_state: bool = False):
    """Build the ordered, capped ref list + the refs_used metadata (for the UI), by priority:
    prev(use_ref) → person → kitchen → ingredient → equipment → state/dish → keyword.
    The character + kitchen refs are permanent identity anchors: sent on EVERY relevant shot (person → character,
    non-flatlay → kitchen) rather than piggybacked on the prev frame, so they can't drift from a prev close-up.
    edit_state (same-framing cooking continuation) reproduces the prev frame as its base and STILL gets the
    kitchen anchor — the prev frame alone let the room drift. flatlay (intro/overview) drops both anchors."""
    cfg_gen = svc.settings.image_gen
    icfg = svc.settings.image_search
    gtype = plan.get("image_generate_type", "new_generate")
    flatlay = plan.get("framing") == "flatlay"
    max_refs = cap or cfg_gen.max_total_refs   # overview (all-ingredients) shot overrides the normal cap
    refs: list[dict] = []
    used: list[dict] = []

    def _room() -> bool:
        return len(refs) < max_refs

    # route A (edit_state): same-framing continuation → the previous cooking frame IS the base. Put it FIRST
    # with the EDIT label so the image wrapper reproduces the WHOLE frame (background + counter + props) and
    # changes only the food — locking out invented set-dressing. Consumes state_ref (skips priority-4 below).
    if edit_state and state_ref:
        refs.append({"label": "Previous frame — keep it IDENTICAL except for the described changes",
                     "mime": state_ref["mime"], "data": state_ref["data"]})
        used.append({"kind": "state", "label": "prev step (edit base)",
                     "url": state_ref.get("_url", ""), "source": "generated"})
        state_ref = None

    # 1. prev frame (use_ref_img edit base — already carries person + kitchen)
    if gtype == "use_ref_img" and prev_ref:
        refs.append({"label": "Previous frame — keep it IDENTICAL except for the described changes", **prev_ref})
        used.append({"kind": "prev", "label": "previous frame", "url": "", "source": "prev"})

    base = None
    # Load base refs (character + scene) for EVERY non-flatlay shot — they are permanent identity anchors sent
    # directly, no longer piggybacked on the prev frame (a prev close-up that dropped the person/kitchen used to
    # make them drift). flatlay shots (intro/overview, top-down, no room) skip it.
    if not flatlay:
        base = await load_base_image_refs(svc, scene_match=scene_match)  # [character, scene picked for this shot]
    # Per-shot overrides: swap the character / scene entry for the user-picked image (SAME anchor label,
    # so the identity/background wording keeps working). Only substitutes a ref that would be used anyway.
    if base and person_override:
        _po = await load_reference(_ref_src(person_override), cfg=icfg, max_px=cfg_gen.character_ref_resize_px)
        if _po:
            base = [({"label": b["label"], **_po} if str(b["label"]).startswith("Character") else b) for b in base]
    if base and scene_override:
        _so = await load_reference(_ref_src(scene_override), cfg=icfg, max_px=cfg_gen.ref_resize_px)
        if _so:
            base = [({"label": b["label"], **_so} if str(b["label"]).startswith("Background") else b) for b in base]
    # 2. person (identity anchor) — ALWAYS on a shot that has the host (incl. her hand), even on edits, so her
    # face/identity can't drift from a prev close-up that dropped her. Non-human shots skip it.
    if base and plan.get("has_human") and _room():
        char_ref = next((r for r in base if str(r["label"]).startswith("Character")), None)
        if char_ref:
            refs.append(char_ref)
            # "config" (no override) still needs a REAL preview url — this is what generation
            # actually uses (load_base_image_refs reads settings.image_gen.character_ref
            # unconditionally), the UI thumbnail was just never pointed at it before.
            # Prefer the DIRECT S3 URL (Omni render fetches it straight, no preview-endpoint
            # indirection); fall back to the preview endpoint when the config ref is still a local
            # default (docs/teacher.png) that the render can't fetch by URL.
            _char = character_store.active_character_ref() or svc.settings.image_gen.character_ref or ""
            _char_url = _char if _char.startswith(("http://", "https://")) else "/api/refs/character/preview"
            used.append({"kind": "person", "label": "person",
                         "url": person_override or _char_url,
                         "source": "override" if person_override else "config"})

    # 3. kitchen scene_ref (identity anchor for the room) — ALWAYS on a non-flatlay shot so the background can't
    # drift from a prev close-up. edit_state used to skip it on the theory that the prev frame carries the room;
    # in practice the room DID drift on those continuations (reported from real renders), so the anchor now goes
    # on every route. Placed before ingredient/equipment so the identity anchors are never crowded out of the cap.
    if base and _room():
        scene_ref = next((r for r in base if str(r["label"]).startswith("Background")), None)
        if scene_ref:
            refs.append(scene_ref)
            # thumbnail must point at the scene actually fed (the resolved default when scene_match is
            # empty/unmatched), not raw scene_match — else the UI falls back to the legacy kitchen.
            _scenes = scene_store.active_scenes()
            _sel = next((s for s in _scenes if s.get("id") == (scene_match or "")), None) \
                   or (scene_store.active_default_scene() if _scenes else None)
            if scene_override:
                used.append({"kind": "kitchen", "label": "kitchen (เลือกเอง)", "url": scene_override, "source": "override"})
            else:
                # Prefer the brand scene's DIRECT S3 URL (_sel["image"] = asset.s3_url) so the Omni
                # render fetches it straight. Fall back to the scene_ref config (if it's an S3 URL) or
                # the preview endpoint (render resolves both via _resolve_ref_bytes).
                _scene_s3 = (_sel or {}).get("image") or ""
                _sref = svc.settings.image_gen.scene_ref or ""
                if _scene_s3.startswith(("http://", "https://")):
                    _scene_url = _scene_s3
                elif _sref.startswith(("http://", "https://")):
                    _scene_url = _sref
                else:
                    _scene_url = (scene_store.scene_preview_url(_sel["id"]) if _sel else "/api/refs/scene/preview")
                used.append({"kind": "kitchen", "label": _selected_scene(svc, scene_match)[1] or "kitchen",
                             "url": _scene_url, "source": "config"})

    # 4. ingredient refs (original form, first use)
    for r in (ing_refs or []):
        if not _room():
            break
        refs.append({"label": r["label"], "mime": r["mime"], "data": r["data"]})
        used.append({"kind": "ingredient", "label": r.get("_name", ""),
                     "url": r.get("_preview") or r.get("_url", ""), "source": r.get("_source", "")})

    # 4b. equipment refs (tools — reused as-is; equipment never changes state)
    for r in (eq_refs or []):
        if not _room():
            break
        refs.append({"label": r["label"], "mime": r["mime"], "data": r["data"]})
        used.append({"kind": "equipment", "label": r.get("_name", ""),
                     "url": r.get("_preview") or r.get("_url", ""), "source": r.get("_source", "")})

    # 5. state/dish ref — the dish in its CURRENT form (prev cooking step / latest dish image)
    if state_ref and _room():
        refs.append({"label": state_ref["label"], "mime": state_ref["mime"], "data": state_ref["data"]})
        used.append({"kind": "state", "label": "prev step", "url": state_ref.get("_url", ""), "source": "generated"})

    # 6. keyword stock photos (fill remaining)
    by_subject: dict[str, list] = {}
    for el in (image_results or []):
        subj = el.get("subject") if isinstance(el, dict) else getattr(el, "subject", "")
        cands = el.get("candidates") if isinstance(el, dict) else getattr(el, "candidates", [])
        by_subject[subj] = cands or []
    for kw in ref_keywords:
        if len(refs) >= max_refs:
            break
        # An already-introduced item → reuse its GENERATED intro image (name-only keyword vs qty-suffixed
        # gen_map key → bidirectional substring). config (Menu) is allowed ONLY on intro shots.
        _k = kw.strip().lower()
        gen = next((u for n, u in (gen_map or {}).items()
                    if u and n and (_k in n.strip().lower() or n.strip().lower() in _k)), "")
        if gen:
            mp = _media_path(gen)
            r = await load_reference(str(mp) if mp and mp.exists() else gen, cfg=icfg, max_px=cfg_gen.ref_resize_px)
            if r:
                refs.append({"label": f'Reference for "{kw}" — match this real object\'s appearance', **r})
                used.append({"kind": "keyword", "label": kw, "url": gen, "source": "generated"})
                continue
        cfg_subj = subject_ref_store.match_subject(kw) if allow_config else None
        if cfg_subj:
            r = await load_reference(cfg_subj["image"], cfg=icfg, max_px=cfg_gen.ref_resize_px)
            if r:
                refs.append({"label": f'Reference photo of "{kw}" — match this real object\'s appearance', **r})
                used.append({"kind": "keyword", "label": kw,
                             "url": subject_ref_store.subject_preview_url(cfg_subj["id"]), "source": "config"})
                continue
        url = next((c for c in by_subject.get(kw, []) if c), "")
        if not url:
            continue
        r = await load_reference(url, cfg=icfg, max_px=cfg_gen.ref_resize_px)
        if r:
            refs.append({"label": f'Reference photo of "{kw}" — match this real object\'s appearance', **r})
            used.append({"kind": "keyword", "label": kw, "url": url, "source": "search"})
    # Carry each ref's source url + kind onto the ref dict itself, so the image-generation trace can log
    # WHICH stored image was actually sent (the video path already does this — see the `refs` field on
    # omni.generate_video_multi). Without it a trace shows labels and raw bytes only, which makes
    # "did this regenerate use the ref I just uploaded?" unanswerable.
    # Safe to zip: every refs.append above is paired 1:1 with a used.append, so the lists stay aligned.
    # Underscore-prefixed keys are ignored by both image providers (they read label/mime/data).
    for _r, _u in zip(refs, used):
        _r["_url"] = _u.get("url", "")
        _r["_kind"] = _u.get("kind", "")
    return refs, used


async def generate_shot_dynamic(svc, shot: dict, prev_shot: dict | None, topic: str,
                                all_ingredients: list[str] | None = None,
                                ingredient_images: dict | None = None,
                                prev_state_url: str = "",
                                used_ingredients: list[str] | None = None,
                                all_equipment: list[str] | None = None,
                                equipment_images: dict | None = None,
                                extra_refs: list[str] | None = None,
                                ref_notes: list[str] | None = None,
                                removed_refs: list[str] | None = None,
                                ref_person: str = "",
                                ref_scene: str = "",
                                ref_state: str = "",
                                category: str = "food",
                                plan_only: bool = False,
                                prompt_override: str | None = None,
                                qc_feedback: str = ""):
    """Dynamic per-shot image flow (spec docs/plan_generate_image.md steps 1-4):
    classify → (reuse_prev | prompt+ref-plan → map refs) → generate.
    Returns (image_bytes | None, image_plan_dict, reused_bool). `shot` is a StoryboardShot dump
    (carries image_results); prev_shot likewise (carries generated_img).
    `plan_only=True` → run classify+plan+resolve (refs_used populated) but SKIP image generation
    (returns (None, plan, reused)) — used by the /steps/image_plan dry-run preview."""
    cfg_gen = svc.settings.image_gen
    aspect = svc.settings.video_gen.aspect_ratio
    if ref_state:
        prev_state_url = ref_state   # per-shot override: user-picked dish-state frame/upload wins over auto
    prev_img_url = (prev_shot or {}).get("generated_img", "") if cfg_gen.use_prev_as_ref else ""
    prev_shows_dish = bool(((prev_shot or {}).get("image_plan") or {}).get("shows_dish"))

    # 1) classify (pass configured scenes so the LLM can pick which one this shot's action belongs in)
    classify = await svc.gemini.classify_shot_image(shot, prev_shot if prev_img_url else None, topic, scenes=scene_store.active_scenes())
    plan = {**classify, "classified": True}
    plan["shot_scale"] = _norm_shot_scale(plan.get("shot_scale"))          # router reads this dict directly
    plan["same_framing_as_prev"] = _as_bool(plan.get("same_framing_as_prev"))
    gtype = plan.get("image_generate_type") or "new_generate"
    is_proc = bool(plan.get("is_process"))
    # Ingredient/equipment INTRODUCTION + all-items overview are ALWAYS a clean top-down flat-lay on
    # plain marble — never the kitchen room, never the host. Force it so a misclassified framing can't
    # add the kitchen scene_ref (priority 5) or the person ref.
    is_intro_product = (shot.get("shot_kind") == "insert" and not is_proc and not plan.get("shows_dish")
                        and (plan.get("is_overview") or shot.get("ingredient_refs") or shot.get("equipment_refs")))
    # Overview kind (single-kind: an ingredient overview shows ALL ingredients + NO tools, and vice-versa),
    # decided by the shot's own refs. A TRUE overview lists the ENTIRE canonical set in its refs (the
    # storyboard spec requires it) — verify that instead of trusting the classifier alone, so a mid-scene
    # partial multi-item shot never gets the full list swapped into its prompt/refs.
    overview = bool(plan.get("is_overview"))
    _cov_ing = bool(all_ingredients) and set(all_ingredients) <= set(shot.get("ingredient_refs") or [])
    _cov_eq = bool(all_equipment) and set(all_equipment) <= set(shot.get("equipment_refs") or [])
    ov_ing = overview and _cov_ing
    ov_eq = overview and not ov_ing and _cov_eq
    # Intro + overview shots are a clean top-down flat-lay of items on marble — never kitchen, never host.
    if is_intro_product or overview:
        plan["framing"], plan["shows_kitchen"], plan["has_human"] = "flatlay", False, False
    # Intro + overview shots are a FRESH flat-lay — never adapt a previous (often person) frame: use_ref_img
    # would carry the host into a no-person insert AND skip the config refs (only `prev` gets fed).
    if is_intro_product or overview:
        gtype, plan["image_generate_type"], plan["reuse_prev"] = "new_generate", "new_generate", False
    # config (Menu subject photos) / search seed ONLY the intro shots (+ overview fallback); every other
    # shot reuses the individually-generated intro images as its refs.
    allow_config = is_intro_product or overview
    # can't reuse/adapt a previous frame that doesn't exist → fall back to a fresh image
    if not prev_img_url and gtype in ("reuse_prev", "use_ref_img"):
        gtype, plan["image_generate_type"], plan["reuse_prev"] = "new_generate", "new_generate", False

    # reuse_prev → copy the previous shot's image, no generation (only non-process talk-only shots)
    if gtype == "reuse_prev" and not is_proc:
        plan["ref_keywords"] = []
        plan["refs_used"] = [{"kind": "prev", "label": "reused previous frame", "url": prev_img_url, "source": "prev"}]
        return (None if plan_only else _read_media_bytes(prev_img_url)), plan, True

    # Strict EDIT (swap one item, SAME framing) is for ingredient INTROs only. A cooking PROCESS shot
    # instead chains the dish STATE from the previous cooking frame, with a fresh framing + action.
    strict_edit = (gtype == "use_ref_img" and not is_proc)
    # state/dish ref: process shots (carry transformed food) + recap/non-process that SHOW the dish (bowl continuity)
    state_url = ""
    if is_proc or plan.get("shows_dish"):
        if prev_state_url:
            state_url = prev_state_url                       # cooking-step state (diced→boiled→...)
        elif plan.get("shows_dish") and prev_img_url and prev_shows_dish and not strict_edit:
            state_url = prev_img_url   # no cooking yet (e.g. hook): keep the finished/hero dish identical to the prev shot
    if not strict_edit:
        plan["image_generate_type"] = "process_chain" if is_proc else "new_generate"

    # 2) prompt + ref keywords
    if strict_edit:
        # Deterministic minimal EDIT instruction from the classified deltas — don't let the LLM
        # re-describe the whole scene (that's what made it hallucinate a fresh flat-lay).
        deltas = []
        if plan.get("ingredient_changed") and plan.get("ingredient_change"):
            deltas.append(f"the food/ingredient shown is now {plan['ingredient_change']}")
        if plan.get("equipment_changed") and plan.get("equipment_change"):
            deltas.append(f"the tool/equipment shown is now {plan['equipment_change']}")
        if plan.get("process_changed") and plan.get("process_change"):
            deltas.append(f"the action is now {plan['process_change']}")
        delta_txt = "; ".join(deltas) or "the single change this shot describes"
        prompt_img = ("Keep the previous frame EXACTLY the same — identical camera, crop, surface, bowls and "
                      f"lighting, and every other object. Change ONLY: {delta_txt}. Nothing else changes.")
        ref_keywords = [s for s in (shot.get("image_subjects") or []) if isinstance(s, str) and s.strip()]
        authored_img = prompt_img          # what this shot SHOWS — see the note at the else-branch snapshot
    else:
        pool = sorted({(el.get("subject") if isinstance(el, dict) else getattr(el, "subject", ""))
                       for el in (shot.get("image_results") or [])} - {""})
        # The scene the classifier picked, described (camera angle, fore/background, where the host
        # belongs). plan_shot_image is text-only and never sees the scene photo, so without this it
        # writes prompts blind to the geometry — which is how the host ended up beside the hob
        # instead of behind it. Empty until the scene has a layout written in the Brand panel.
        _sel_scene = next((sc for sc in scene_store.active_scenes()
                           if sc.get("id") == (plan.get("scene_match") or "")), None) \
            or scene_store.active_default_scene()
        planned = await svc.gemini.plan_shot_image(shot, plan, pool, topic, aspect, category=category,
                                                   scene_layout=(_sel_scene or {}).get("layout", ""),
                                                   extra_ref_notes=ref_notes,
                                                   # Intro/overview shots CREATE the photo every later shot is
                                                   # matched against, so they own the container's colour and
                                                   # material. Every other shot RECEIVES that photo, and a colour
                                                   # it writes can only contradict it — that is how one mixing
                                                   # bowl came out white in some shots and stainless in others.
                                                   vessel_from_ref=not (is_intro_product or overview))
        prompt_img = (planned.get("prompt_img") or shot.get("prompt_img") or "").strip()
        if not is_proc:   # intro/show shot → liquid ingredient goes in a measuring cup, not a bowl (base agrees with the flat-lay suffix)
            prompt_img = _liquid_vessel_fix(prompt_img, shot.get("ingredient_refs"), category)
            if category == "drink":   # a spoon-measured ingredient goes on a measuring spoon, not a bowl
                prompt_img = _spoon_vessel_fix(prompt_img, shot.get("ingredient_refs"))
        if category == "drink":   # any shot (incl. process): an ml liquid never pours from a measuring spoon
            prompt_img = _ml_from_spoon_fix(prompt_img)
        prompt_img = _bare_counter_vessel_fix(prompt_img, shot.get("ingredient_refs"), category)
        if not (is_intro_product or overview):
            # Same scope as the vessel_from_ref block above, enforced: an intro CREATES the photo and
            # writes the colour, every later shot RECEIVES it and must not name a different one.
            prompt_img = _vessel_colour_fix(prompt_img)
        # dish_state authoritative: strip milk the plan may have copied into a pre-milk step
        prompt_img = _milk_state_fix(prompt_img, shot.get("dish_state") or "")
        # Last point at which prompt_img is purely WHAT THE SHOT SHOWS. Everything appended from here on
        # (flat-lay rule, ingredient MANDATORY, "The host ONLY does...", composition, no-packaging,
        # isolated-closeup, aspect, fixtures) is an INSTRUCTION TO the model, not a description OF the
        # frame. The person check further down has to read this snapshot: scanning the assembled string
        # made the code treat its own guard rails as evidence — "The host ONLY does what the description
        # says" (a rule meant to stop invented props) set has_human on an object-only close-up, which
        # attached the character ref, put a stray hand in the render, and had Omni invent a whole person.
        authored_img = prompt_img
        ref_keywords = [k for k in (planned.get("ref_keywords") or []) if isinstance(k, str) and k.strip()]
        if plan.get("framing") == "flatlay":
            # Overview shot → use the FULL recipe list for its OWN kind (no guessing); else this shot's items.
            if ov_eq and all_equipment:
                items = ", ".join(all_equipment)
            elif overview and all_ingredients:
                items = ", ".join(all_ingredients)
            else:
                items = ", ".join(s for s in (shot.get("image_subjects") or []) if isinstance(s, str) and s.strip())
            # Tools/equipment rest DIRECTLY on the counter; only loose INGREDIENTS go in bowls.
            is_equip_flat = ov_eq or (bool(shot.get("equipment_refs")) and not shot.get("ingredient_refs"))
            if not items:
                only = ""
            elif is_equip_flat:
                only = (f" Show EXACTLY these tools/equipment: {items}. Each item rests DIRECTLY on the plain "
                        "marble counter — do NOT put any of them inside a bowl, on a plate, or in any container.")
            elif category == "drink":
                only = (f" Show EXACTLY these items: {items}. Put each ingredient in the vessel its recipe UNIT "
                        "implies — a measuring spoon (matching size) for a SPOON amount (ช้อนตวง/ช้อนโต๊ะ/ช้อนชา), "
                        "a PLAIN clear glass MEASURING CUP (no numbers, no measurement lines, no pattern) for a "
                        "LIQUID (water/milk/syrup/juice/oil/tea/coffee or anything by volume/ml), a bowl sized to "
                        "fit for a loose weight (กรัม) or whole pieces (หัว/ฟอง/ลูก/ใบ/แผ่น). Never force a "
                        "spooned amount into a bowl.")
            elif category == "other":
                # Not a recipe → no vessel doctrine to apply. The bowls/measuring-cup wording below
                # is food-specific and would be nonsense for a topic that isn't cooking; this text
                # is written inline rather than in CATEGORY_RULES, so category_block() cannot skip
                # it for us.
                only = f" Show EXACTLY these items: {items}."
            else:
                only = (f" Show EXACTLY these items: {items}. Put each DRY/solid ingredient in its own small bowl. "
                        "A LIQUID (sauce, stock, milk, syrup, juice, vinegar, or anything measured by volume/ml) "
                        "goes in its own SMALL PLAIN CUP — a round dish with no handle, NOT a measuring cup. Two "
                        "exceptions: PLAIN WATER goes in a plain clear glass measuring cup (no numbers, no "
                        "measurement lines, no pattern), and COOKING OIL is shown in its own clear glass oil "
                        "BOTTLE — never poured into a cup or a bowl.")
            # Ingredient INTRO (just showing, not a real manipulation) → no hand at all; the items just rest.
            # A genuine process (cutting/mixing/...) keeps the acting hand.
            hand = (" The host's hand performs the action."
                    if plan.get("is_process")
                    else " NO hand and NO person anywhere in the frame — the ingredients simply REST on the counter "
                         "(do NOT show a hand pointing at, holding or presenting them).")
            prompt_img += (" Top-down flat-lay on a plain white marble counter; objects rest FLAT, nothing floats."
                           + hand + only +
                           " Do NOT add ANY other food, ingredient, garnish, herb, parsley, garlic, vegetable, "
                           "spice, sauce, prop, plate, bowl, utensil, cloth or decoration that is not named above "
                           "— keep the rest of the counter completely clean and empty.")
        else:
            # eye-level shots: enforce the EXACT ingredient quantities (ingredient_refs carry "name + qty")
            # so the model doesn't add extra potatoes / a bowl of spares.
            qty = ", ".join(s for s in (shot.get("ingredient_refs") or []) if isinstance(s, str) and s.strip())
            if qty:
                prompt_img += (f" MANDATORY: The following ingredient(s) MUST be CLEARLY VISIBLE and physically "
                               f"present as the subject of this cooking action: {qty}. "
                               "Show them INSIDE the cooking vessel or ON the cooking surface — "
                               "e.g., potato chunks in the boiling water, chicken pieces in the pan, dough on the board. "
                               "Do NOT show only the vessel or cooking medium without the ingredient inside. "
                               "Do NOT add extra pieces, extra copies, a bowl of spares, or any other "
                               "food/ingredient beyond what this step actually uses.")
            if not plan.get("is_process"):
                # talk / greeting / recap shot — the host isn't cooking → don't let the model invent props/actions.
                prompt_img += (" The host ONLY does what the description says (e.g. talking, greeting, presenting). "
                               "Do NOT add food being cut or prepared, a cutting board, knife, vegetables, extra "
                               "ingredients, dishes, or any cooking action or prop that is not explicitly described "
                               "— keep the rest of the scene clean.")
            if plan.get("has_human"):
                # Wherever the host appears she belongs BEHIND the work surface, with that surface
                # between her and the camera. Previously gated on `not is_proc`, so cooking shots —
                # the ones at the hob, where getting this wrong is most visible — got no composition
                # rule at all and the model was free to put her off to one side.
                # The positioning sentence comes from the matched scene's `layout` when one names the
                # host's place — measured on the sink scene, handing the layout to the prompt AUTHOR
                # alone never got the fixture's name into the outgoing prompt ("work surface (counter
                # or hob…)" reached the model with no mention of a sink), so the rule is spliced in
                # deterministically here instead of hoping the author paraphrases it. Scenes with no
                # layout keep the old generic wording.
                _host_rule = _scene_host_rule((_sel_scene or {}).get("layout", ""))
                if not _host_rule:
                    _surface = "work surface (counter or hob, whichever this scene shows)" if is_proc else "counter"
                    _host_rule = (f"the host stands BEHIND the {_surface} — it runs across the FOREGROUND, "
                                  "between the host and the camera. NEVER place the host in front of it, "
                                  "off to one side of it, or between it and the camera.")
                prompt_img += (f" COMPOSITION: {_host_rule} She is a SOLID, separate "
                               "subject: her body must NOT clip through, merge into or intersect the counter, "
                               "fixtures or props — keep a clean silhouette with clear separation from the "
                               "background.")
        # C (bg-lock layer 2): a process shot that is NOT an intro/overview must not gain extra packaging /
        # props the model adds as "kitchen dressing". OFF for intro/overview (those SHOW the items).
        if is_proc and not (overview or is_intro_product):
            prompt_img += (" Do NOT add any product packaging, extra bowls, glasses, or ingredients on the "
                           "counter beyond what THIS step uses — keep the rest of the counter clean.")
        # route 4: a tight close-up on a NEW angle has no prior frame to lock the background → force an
        # isolated, uncluttered composition so the model can't fill the empty counter with themed props.
        if plan.get("shot_scale") == "closeup" and not plan.get("same_framing_as_prev"):
            # Only mention hands when the shot actually has a person — telling the model "the hands are in
            # frame" on an object-only close-up invites it to draw one. (This used to matter twice over: the
            # sentence also fed the person check below. That leak is closed now — it reads the authored
            # description instead of this assembled string — but the wording is still right on its own terms.)
            in_frame = ("the cup/bowl/vessel and the hands are" if plan.get("has_human")
                        else "ONLY the cup/bowl/vessel is")
            prompt_img += (f" ISOLATED close-up on a plain, uncluttered countertop — {in_frame} in frame; "
                           "no product packages, extra bowls, glasses, or other "
                           "ingredients anywhere in the frame or background.")
        prompt_img = f"{prompt_img}\n\nRender as a {aspect} widescreen, cinematic frame."

    # Drop kitchen fixtures (sink / faucet / stove ...) from keyword refs — they already exist in the
    # scene_ref; adding them as a new ref makes the model invent a second / mismatched fixture.
    # The fixture list is configured (extracted from scene_ref + editable in the UI).
    fixtures = load_kitchen_fixtures(svc)
    def _is_fixture(n: str) -> bool:
        nl = n.lower()
        return any(f and (f.lower() in nl or nl in f.lower()) for f in fixtures)
    used_fixtures = [k for k in ref_keywords if _is_fixture(k)]
    ref_keywords = [k for k in ref_keywords if not _is_fixture(k)]
    if used_fixtures:
        prompt_img += (f" The {', '.join(used_fixtures)} is the EXISTING one already in the kitchen "
                       "(Background reference) — use it as-is; do NOT add, move or invent a new one.")
    plan["ref_keywords"] = ref_keywords
    # If the AUTHORED description mentions a person (chef/host/hand), keep her identity ref even when the shot
    # was marked human-less (e.g. an intro where the host still holds the item) — else she renders with no
    # character reference and her face drifts. Skip flat-lays: their prompt carries a "NO hand/NO person"
    # instruction that would false-match, and they are enforced person-free anyway.
    # `authored_img`, NOT `prompt_img`: by this point prompt_img also carries the guard rails appended above,
    # and several of them name the host to forbid something ("The host ONLY does what the description says").
    # Reading those back made the shot human by mere mention. prompt_img itself is untouched and still goes to
    # the image model in full — only this one decision reads the narrower string.
    if not plan.get("has_human") and plan.get("framing") != "flatlay" and (
            shot.get("shot_kind") == "person" or _prompt_has_person(authored_img)):
        # The storyboard's own shot_kind is an explicit declaration that the host is in this shot —
        # when the classifier misses it, the person anchor used to vanish and her face drifted.
        plan["has_human"] = True

    # 3) map refs
    prev_ref = None
    if strict_edit:
        p = _media_path(prev_img_url)
        prev_src = str(p) if p and p.exists() else prev_img_url
        prev_ref = await load_reference(prev_src, cfg=svc.settings.image_search, max_px=cfg_gen.ref_resize_px) if prev_src else None
    # Process shot → STATE reference: the dish in its current cooked/cut form from the previous cooking frame.
    state_ref = None
    if state_url:
        sp = _media_path(state_url)
        sr = await load_reference(str(sp) if sp and sp.exists() else state_url, cfg=svc.settings.image_search, max_px=cfg_gen.ref_resize_px)
        if sr:
            state_ref = {**sr, "_url": state_url,
                         "label": ("State reference of the food — match the food's CURRENT form, colour and size "
                                   "shown here (e.g. cut pieces / cooked); the camera, framing and action follow the "
                                   "description below — do NOT copy the previous pose or hands.")}
    # Ingredient intro images — the ingredient in its ORIGINAL form. Used when it is being introduced or
    # ADDED for the first time (e.g. peeling the raw potato, adding the butter). For a process shot, an
    # ingredient that has ALREADY been worked on before (in `used_ingredients`) is now transformed → skip its
    # raw-intro ref (the state/dish ref carries its current form instead).
    used_set = set(used_ingredients or [])
    ing_refs: list[dict] = []
    imap = ingredient_images or {}
    # subject→candidates lookup from this shot's image_results for Tavily fallback
    _by_subj: dict[str, list] = {}
    for _el in (shot.get("image_results") or []):
        _subj = _el.get("subject") if isinstance(_el, dict) else getattr(_el, "subject", "")
        _cands = _el.get("candidates") if isinstance(_el, dict) else getattr(_el, "candidates", [])
        _by_subj[_subj] = _cands or []
    # For an overview, cover EVERY recipe ingredient (not just this shot's list) so the combined flat-lay
    # matches every single-ingredient shot; a normal shot uses its own ingredient_refs. (overview/ov_ing/
    # ov_eq were decided above.)
    ing_names = [] if ov_eq else (all_ingredients if (ov_ing and all_ingredients) else (shot.get("ingredient_refs") or []))
    for name in ing_names:
        if name in used_set and not overview:
            continue  # already worked/transformed in a prior step → state/prev ref carries its current form (not raw)
        gen_url = imap.get(name)                                          # generated intro image (reuse for continuity)
        cfg_subj = subject_ref_store.match_subject(name, "ingredient")    # configured Menu ref
        # config SEEDS an item's raw look on its FIRST appearance (per-item intro, or the first non-process shot
        # that shows it — incl. a shot where the host holds it) and wins over a stray generated; the OVERVIEW reuses
        # the generated singles so its flat-lay matches them; process/repeat shots reuse the generated single.
        seed_config = cfg_subj and not overview and (is_intro_product or (not gen_url and not is_proc))
        if seed_config:
            url, source, preview = cfg_subj["image"], "config", subject_ref_store.subject_preview_url(cfg_subj["id"])
        elif gen_url:
            url, source, preview = gen_url, "generated", ""
        elif allow_config:
            url, source, preview = next((c for c in _by_subj.get(name, []) if c), ""), "search", ""
        else:
            continue  # non-intro shot, no config seed: reuse the generated single only — no search fallback
        if not url:
            continue
        mp = _media_path(url)
        r = await load_reference(str(mp) if mp and mp.exists() else url, cfg=svc.settings.image_search, max_px=cfg_gen.ref_resize_px)
        if r:
            # The vessel used to be on the "copy nothing else" list, which is why an ingredient
            # introduced in a stainless bowl came back in a white one two shots later: the reference
            # was showing the right bowl while the label forbade reproducing it. It is part of the
            # ingredient's identity across the recipe, so it is copied now — the shot's own text
            # still wins when the step genuinely moves the food (into a pan, onto a board).
            label = (f'Reference for "{name}" (original form being added) — match its shape, size and colour, '
                     f"AND the vessel it sits in (the SAME bowl/cup/plate: same material, colour and shape), "
                     f"unless THIS shot's description puts it somewhere else; "
                     f"copy NOTHING else from this photo: do NOT reproduce its background, hands, "
                     f"any OTHER items, or any TEXT/labels burned into it"
                     if source in ("config", "generated") else
                     f'Reference photo of "{name}" — match this real object\'s appearance')
            ing_refs.append({**r, "_name": name, "_url": url, "_source": source, "_preview": preview, "label": label})
    # Equipment refs — tools this shot uses/introduces. Like ingredients but equipment NEVER changes state,
    # so it's always REUSED as-is (generated intro image → configured photo → search); no used_set/process skip.
    eqmap = equipment_images or {}
    eq_refs: list[dict] = []
    eq_names = [] if ov_ing else (all_equipment if (ov_eq and all_equipment) else (shot.get("equipment_refs") or []))
    for name in eq_names:
        gen_url = eqmap.get(name)
        cfg_subj = subject_ref_store.match_subject(name, "equipment")     # configured Menu ref
        # config seeds equipment on its FIRST appearance in ANY shot (tools never change state, so no is_proc
        # guard); a per-item intro prefers config over a stray generated; overview + repeats reuse the single.
        seed_config = cfg_subj and not overview and (is_intro_product or not gen_url)
        if seed_config:
            url, source, preview = cfg_subj["image"], "config", subject_ref_store.subject_preview_url(cfg_subj["id"])
        elif gen_url:
            url, source, preview = gen_url, "generated", ""
        elif allow_config:
            url, source, preview = next((c for c in _by_subj.get(name, []) if c), ""), "search", ""
        else:
            continue  # non-intro shot, no config seed: reuse the generated single only — no search fallback
        if not url:
            continue
        mp = _media_path(url)
        r = await load_reference(str(mp) if mp and mp.exists() else url, cfg=svc.settings.image_search, max_px=cfg_gen.ref_resize_px)
        if r:
            label = (f'Reference for the tool/equipment "{name}" — match its exact shape, material and colour; '
                     f"copy NOTHING else from this photo: do NOT reproduce its background, hands, any OTHER "
                     f"items, or any TEXT/labels burned into it — only the {name} itself"
                     if source in ("config", "generated") else
                     f'Reference photo of the tool "{name}" — match this real object\'s appearance')
            eq_refs.append({**r, "_name": name, "_url": url, "_source": source, "_preview": preview, "label": label})

    # Dedup: drop keyword refs already covered by an ingredient/equipment ref (same object) — that ref wins
    # (higher priority + carries quantity). Bidirectional substring since ingredient_refs carry a qty suffix
    # (e.g. "ผงชาเขียว 2 ช้อนโต๊ะ") while a keyword is name-only ("ผงชาเขียว").
    _ref_names = [str(r.get("_name", "")).strip().lower() for r in (ing_refs + eq_refs) if r.get("_name")]
    def _covered(kw: str) -> bool:
        k = kw.strip().lower()
        return any(n and (k in n or n in k) for n in _ref_names)
    # An overview is single-kind and fully defined by its expanded ingredient/equipment list — drop ALL
    # keyword refs so a cross-kind subject (e.g. a tool named in an ingredient-overview shot) can't slip in.
    ref_keywords = [] if overview else [k for k in ref_keywords if not _covered(k)]
    plan["ref_keywords"] = ref_keywords
    # overview / all-ingredients+equipment shot: feed EVERY item image as a ref (the per-item shots generated
    # them) so the combined flat-lay matches the singles — needs a cap above max_total_refs.
    overview_cap = (len(ing_refs) + len(eq_refs) + 2) if plan.get("is_overview") else None
    # edit_state (route A): same-framing continuation of a cooking step → reproduce the WHOLE previous frame,
    # change only the food (so it skips the separate kitchen ref). OFF for intro/overview (clean flat-lays).
    edit_state = bool(state_ref) and plan.get("same_framing_as_prev") and not (overview or is_intro_product)
    refs, used = await resolve_shot_refs(svc, plan, ref_keywords, shot.get("image_results"), prev_ref, ing_refs, state_ref,
                                         cap=overview_cap, scene_match=plan.get("scene_match"),
                                         eq_refs=eq_refs, gen_map={**imap, **eqmap}, allow_config=allow_config,
                                         person_override=ref_person, scene_override=ref_scene,
                                         edit_state=edit_state)
    for u in used:
        if u["kind"] == "prev":
            u["url"] = prev_img_url   # so the UI can show the previous frame that was adapted
    # Dedup by image identity: the same media reached via different slots (a tool whose generated intro is
    # also the prev/state frame, or two tool names mapped to one generated image) must appear once. Key on
    # the media URL; url-less refs (person) fall back to kind|label. refs[i] ↔ used[i] are index-aligned.
    seen, _refs, _used = set(), [], []
    for r, u in zip(refs, used):
        key = u.get("url") or f"{u['kind']}|{u.get('label', '')}"
        if key in seen:
            continue
        seen.add(key)
        _refs.append(r); _used.append(u)
    refs, used = _refs, _used
    # Manual ref overrides (user hand-edited the ref list in the UI): drop removed, append uploaded extras.
    # refs[i] ↔ used[i] are index-aligned, so filter both by the same indices.
    if removed_refs:
        _rm = set(removed_refs)
        keep = [i for i, u in enumerate(used) if f"{u['kind']}|{u.get('label', '')}" not in _rm]
        refs, used = [refs[i] for i in keep], [used[i] for i in keep]
    # Extras land AFTER the dedup pass above, so without this an extra that duplicates an auto ref
    # reached the model twice with CONTRADICTORY labels ("copy NOTHING else…" vs "take visual
    # cues"). Easy to hit now that the pickers offer exactly the images the resolver auto-attaches
    # (generated frames, Menu subjects). Exact URL match suffices: every picker flow hands back the
    # same string the resolver stores (generated_img / SubjectRef.image); the auto ref wins because
    # its label says precisely what to take. A fresh upload can never collide.
    _attached = {u.get("url") for u in used if u.get("url")} | {r.get("_url") for r in refs if r.get("_url")}
    for _i, u_url in enumerate(extra_refs or []):
        if u_url in _attached:
            continue
        # _ref_src also resolves /api/menus|brands/…/preview URLs (config-menu picks) to their files
        r = await load_reference(_ref_src(u_url), cfg=svc.settings.image_search, max_px=cfg_gen.ref_resize_px)
        if r:
            # The user's own words for this ref, when given. Attaching three images that all say
            # "additional reference image" leaves the model no way to tell them apart, while every
            # auto ref above states exactly what to take from it.
            _note = (ref_notes[_i].strip() if _i < len(ref_notes or []) else "")
            _label = (f"Additional reference image the user provided — {_note}" if _note else
                      "Additional reference image the user provided — take visual cues "
                      "(subject / style / composition) from it")
            refs.append({"label": _label, **r})
            used.append({"kind": "upload", "label": _note or "uploaded", "url": u_url, "source": "upload"})
    plan["refs_used"] = used

    # The EXACT full prompt the image model will receive (prompt_img + text rules + provider ref wrapper),
    # exposed so the UI can show/edit it as "ครบ". Editing it → generate that text verbatim (prompt_override).
    prompt = _apply_text_rules(prompt_img, shot.get("on_screen_text", ""), cfg_gen)
    if qc_feedback.strip():
        # user opted to attach the QC-FEEDBACK block from the failed review on this manual regen
        prompt = prompt + "\n\n" + qc_feedback.strip()
    plan["full_prompt"] = assemble_full_prompt(svc, prompt, refs)

    # 4) generate (skipped for dry-run preview — refs_used is already populated)
    if plan_only:
        return None, plan, False
    if prompt_override:
        # user-edited full prompt: send it verbatim (skip the QC prompt-munging loop) with the same refs
        data = await _provider_generate(svc, prompt, refs, prebuilt=prompt_override)
        return data, plan, False
    # C1: generate + vision-QC loop — the verdict rides on the plan (qc_passed/qc_attempts/qc_issues)
    # so the storyboard JSON + UI can show which shots shipped flagged. QC off → one plain generate.
    qc_ctx = _qc_ctx(
        shot_kind=str(shot.get("shot_kind") or ("person" if plan.get("has_human") else "insert")),
        framing=str(plan.get("framing") or plan.get("shot_scale") or ""),
        must_show=(shot.get("image_subjects") or shot.get("ingredient_refs") or shot.get("equipment_refs") or []),
        description=str(shot.get("motion_description") or "")[:1200],
        expect_text=bool(cfg_gen.on_screen_text_in_image and str(shot.get("on_screen_text", "")).strip()),
    )
    data, qc = await _qc_generate(svc, prompt, refs, qc_ctx, topic)
    plan.update(qc)
    return data, plan, False


async def generate_images(state: PipelineState, config: RunnableConfig) -> dict:
    """Generate an image per storyboard shot from its prompt_img (gemini image model).

    Images are written to outputs/media/<run>/ and served at /api/media/...; each
    shot's `generated_img` URL is emitted progressively so the frontend fills in
    pictures as they finish. Best-effort: a failed shot just has no image.
    """
    svc = services(config)
    sb = state.get("storyboard_obj")
    cfg_gen = svc.settings.image_gen
    if not (cfg_gen.enabled and sb):
        return {}
    if cfg_gen.provider == "openai" and not svc.openai_api_key:
        logger.warning("image_gen.provider=openai but OPENAI_API_KEY is not set — images will be skipped")

    topic = state["topic"]
    regenerate_all = state.get("regenerate_all", True)
    pairs = [(sc, sh) for sc in sb.scenes for sh in sc.shots if sh.prompt_img.strip()]
    if not pairs:
        return {}
    pairs_to_generate = pairs if regenerate_all else [(sc, sh) for sc, sh in pairs if not sh.generated_img]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _slug(topic)
    run_dir = _MEDIA_DIR / f"{ts}_{slug}"
    sem = asyncio.Semaphore(max(1, cfg_gen.concurrency))

    with obs.span("node.generate_images", input={"shots": len(pairs)}) as span:
        base_refs = await load_base_image_refs(svc)

        _cat = getattr(state.get("script_config"), "category", "food")
        emit({
            "type": "status",
            "message": f"Generating images for {len(pairs_to_generate)}/{len(pairs)} shots... "
                       f"(ประเภท: {_category_label(_cat)})",
        })
        logger.info(f"generate_images category={_cat} regenerate_all={regenerate_all} to_generate={len(pairs_to_generate)}/{len(pairs)}")
        run_dir.mkdir(parents=True, exist_ok=True)

        def _save(sc, sh, data) -> None:
            fname = f"{sc.scene_id}_{sh.no}.png"
            _key = f"{ts}_{slug}/{fname}"
            # S3 on → upload only, keep no local copy (disk doesn't grow); else write local.
            if _storage.is_configured():
                sh.generated_img = _storage.upload_bytes(_key, data, "image/png")
            else:
                (run_dir / fname).write_bytes(data)
                sh.generated_img = f"/api/media/{_key}"
            # Same clickable result link the per-shot routes attach (routes.py's /steps/image) — the bulk
            # path was the only one rendering images without one on its trace.
            obs.attach_media(sh.generated_img, "image")
            # image_plan rides along per shot (sh.image_plan is set before _save) — the frontend's
            # reuse chain reads is_process from it, and until the final step_result lands the board
            # otherwise has stale plans; a bulk STOPPED midway then under-built every later chain.
            emit({"type": "image_generated", "scene_id": sc.scene_id, "no": sh.no, "url": sh.generated_img,
                  "image_plan": sh.image_plan.model_dump() if sh.image_plan is not None else None})

        if cfg_gen.dynamic_image:
            # canonical ingredient/equipment → its generated image (intro shot); reused by later shots.
            # ⚠ TWIN LOGIC: frontend/src/steps/Step4Images.tsx `buildReuseChain` rebuilds this exact
            # walk (single-item-intro rule, is_process → state) for per-shot Generate/Preview — a rule
            # changed here must change there too, or per-shot and bulk silently resolve different refs
            # (that drift is precisely the bug fixed in fix/image-ref-chain).
            ing_map: dict[str, str] = {}
            eq_map: dict[str, str] = {}
            last_proc_img = ""   # most recent cooking-step image → dish STATE carried forward
            used_set: set[str] = set()   # ingredients already worked in a process shot (→ no raw-intro ref)
            for sc in sb.scenes:
                for sh in sc.shots:
                    if sh.generated_img:
                        # Only a SINGLE-item intro frame may become an item's reusable ref — a multi-item
                        # frame (overview / hero) carries other items + burned-in labels that the model
                        # would copy into later shots.
                        if len(sh.ingredient_refs) == 1:
                            ing_map.setdefault(sh.ingredient_refs[0], sh.generated_img)
                        eqr = getattr(sh, "equipment_refs", []) or []
                        if len(eqr) == 1:
                            eq_map.setdefault(eqr[0], sh.generated_img)
                        if getattr(sh.image_plan, "is_process", False):
                            last_proc_img = sh.generated_img
                            used_set.update(sh.ingredient_refs)

            # Process in timeline order (scenes then shots) so prev-frame edits + dish-state chaining work.
            for sc in sb.scenes:
                prev = None   # strict-edit base is within-scene; dish state chains globally via last_proc_img
                for sh in sc.shots:
                    if not sh.prompt_img.strip():
                        continue
                    if not regenerate_all and sh.generated_img:
                        # "missing only" — keep the existing image untouched, but still advance `prev` so
                        # later shots' prev-frame/dish-state chaining sees this shot as if freshly rendered.
                        prev = sh
                        continue
                    # REL-1: a single shot's generate failing (429 at shot 47/60) must NOT discard the 46
                    # already-rendered (paid-for) shots — isolate it, warn, keep going. prev still advances
                    # so the chain/dish-state continuity is unbroken.
                    try:
                        data, plan, _ = await generate_shot_dynamic(
                            svc, sh.model_dump(), prev.model_dump() if prev else None, topic,
                            all_ingredients=list(sb.ingredients), ingredient_images=ing_map,
                            prev_state_url=last_proc_img, used_ingredients=list(used_set),
                            all_equipment=list(getattr(sb, "equipment", []) or []), equipment_images=eq_map,
                            category=getattr(state.get("script_config"), "category", "food"),
                            # A hand-edited "ครบ" prompt survives Generate All — pre-refactor bulk
                            # was a client-side loop over the per-shot route, which always sent it;
                            # dropping it here meant one bulk run overwrote every manual fix.
                            prompt_override=(getattr(sh, "prompt_full", "") or None))
                        sh.image_plan = ShotImagePlan(**{k: v for k, v in plan.items() if k in ShotImagePlan.model_fields})
                        if data:
                            _save(sc, sh, data)
                            # first generated SINGLE-item intro wins (multi-item frames carry other items/labels)
                            if len(sh.ingredient_refs) == 1:
                                ing_map.setdefault(sh.ingredient_refs[0], sh.generated_img)
                            eqr = getattr(sh, "equipment_refs", []) or []
                            if len(eqr) == 1:
                                eq_map.setdefault(eqr[0], sh.generated_img)
                            if plan.get("is_process"):
                                last_proc_img = sh.generated_img
                                used_set.update(sh.ingredient_refs)   # transformed → no raw-intro ref next time
                    except Exception as e:  # noqa: BLE001 — best-effort per shot
                        logger.warning(f"image gen failed for shot {sc.scene_id}.{sh.no} (kept the other shots): {e}")
                        emit({"type": "progress", "message": f"⚠️ ข้ามรูปช็อต {sc.scene_id}.{sh.no} — สร้างไม่สำเร็จ (ช็อตอื่นทำต่อ)"})
                    prev = sh
        else:
            async def _one(sc, sh) -> None:
                async with sem:
                    data = await render_shot_image(svc, sh.prompt_img, sh.on_screen_text, sh.image_results, base_refs, sh.shot_kind)
                    if data:
                        _save(sc, sh, data)
            await asyncio.gather(*(_one(sc, sh) for sc, sh in pairs_to_generate))

        made = sum(1 for _, sh in pairs if sh.generated_img)
        _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        (_OUTPUTS_DIR / f"{ts}_{slug}_storyboard.json").write_text(sb.model_dump_json(indent=2), encoding="utf-8")
        logger.info(f"Generated images: {made}/{len(pairs)} shots")
        span.update(output={"generated": made, "total": len(pairs)})

    return {"storyboard": sb.model_dump()}


async def _validate_video_prompts(gemini, scenes, topic: str, next_img: dict | None = None, voice: str = "") -> None:
    """QA-check every shot's prompt_video, regenerating invalid ones (1 round).

    Passes shot_kind so the checker can enforce NO-INJECTION (no person/hand in an insert),
    voice_over so it can enforce AUDIO PRESENCE (every speaking shot names its spoken line + voice),
    and next_frame so it knows which ending-transition is sanctioned (won't delete it)."""
    next_img = next_img or {}
    # H1: shot.no restarts at 1 every scene, so keying the QA round-trip by it alone
    # collapses same-numbered shots across scenes and writes one scene's correction onto
    # all of them. Flatten with a document-unique index as the echo key ("no" is only an
    # opaque id the checker echoes back — a running index is safe).
    flat = [(sc, sh) for sc in scenes for sh in sc.shots if sh.prompt_video]
    payload = [
        {"no": i, "shot_kind": getattr(sh, "shot_kind", "person"),
         "voice_over": getattr(sh, "voice_over", ""),
         "prompt_img": sh.prompt_img, "prompt_video": sh.prompt_video,
         **({"next_frame": next_img[(sc.scene_id, sh.no)]}
            if next_img.get((sc.scene_id, sh.no)) else {})}
        for i, (sc, sh) in enumerate(flat)
    ]
    if not payload:
        return
    res = await gemini.validate_video_prompts(payload, topic, voice=voice)
    for i, (sc, sh) in enumerate(flat):
        v = res.get(i)
        if v and v.get("prompt_video"):
            sh.prompt_video = v["prompt_video"]
    invalid = sum(1 for v in res.values() if not v.get("valid"))
    logger.info(f"Video prompt validation: {len(res)} checked, {invalid} corrected")


async def generate_video_prompts(state: PipelineState, config: RunnableConfig) -> dict:
    """Write a Veo motion prompt (prompt_video) for every storyboard shot.

    One LLM call per scene (concurrent, capped). Each shot's still frame (prompt_img)
    plus its motion_description/voice_over become a short clip-motion prompt. The
    actual video is rendered later (per-shot, on demand) in the video step.
    """
    svc = services(config)
    sb = state.get("storyboard_obj")
    if not sb:
        return {}
    topic = state["topic"]
    cfg = state.get("script_config")
    vcfg = svc.settings.video_gen
    duration = vcfg.duration_seconds
    # Cinematic-craft block (fixed colour grade across all shots) — gated by config.
    grade = (vcfg.color_grade or "").strip() or "natural soft light, consistent cinematic color"
    # Narrator voice: pin gender/tone so Veo (which invents a voice PER CLIP) can't drift between shots.
    # Priority: explicit voice_desc (Script Settings) → brand vo_tone → derived from character_desc (carries
    # the host's gender/age even when no voice was configured) → generic default.
    brand = scene_store.active_brand() or {}
    _cd = (getattr(cfg, "character_desc", "") or "").strip()
    voice = ((getattr(cfg, "voice_desc", "") or "").strip()
             or (brand.get("vo_tone") or "").strip()
             or (f"a Thai narrator whose voice matches the host — {_cd}" if _cd else "a warm, friendly Thai narrator"))
    style_block = get_prompt("VIDEO_STYLE_BLOCK").format(grade=grade) if vcfg.cinematic_prompt else ""
    scenes = sb.scenes
    sem = asyncio.Semaphore(max(1, svc.settings.pipeline.analyze_concurrency))
    total = sum(len(sc.shots) for sc in scenes)

    # Frame mode is chosen HERE (at prompt time) so the wording fits how the clip will
    # be rendered. text-to-video / image-to-video (start frame) / first+last (start +
    # end on the NEXT shot's frame for a seamless join).
    use_image_seed = bool(state.get("use_image_seed", True))
    use_last_frame = bool(state.get("use_last_frame", False))
    # Target end frame per shot — ONLY for a next shot that truly continues this one (same scene AND
    # same shot_kind). An insert→person or cross-scene neighbor is a CUT, not a join; interpolating an
    # ingredient shot toward a person frame is exactly what made a host appear at the clip's end.
    flat = [(sc.scene_id, sh) for sc in scenes for sh in sc.shots]
    next_img: dict = {}
    for i, (sid, sh) in enumerate(flat):
        nxt = flat[i + 1] if i + 1 < len(flat) else None
        compatible = (
            nxt is not None and nxt[0] == sid
            and getattr(nxt[1], "shot_kind", "person") == getattr(sh, "shot_kind", "person")
        )
        next_img[(sid, sh.no)] = nxt[1].prompt_img if compatible else ""
    if not use_image_seed:
        mode_block = (
            "FRAME MODE — text-to-video: there is NO starting image. Each prompt_video must first "
            "establish the full scene (the setting, the subject exactly as described in prompt_img — "
            "a person appears ONLY if prompt_img contains one — and the props) and THEN the motion."
        )
    elif use_last_frame:
        mode_block = (
            "FRAME MODE — first+last frame: each shot's clip STARTS exactly from its still frame (prompt_img) "
            "and must END settling into that shot's `next_frame` field (the opening of the next shot). Write ONE "
            "continuous camera/subject motion that smoothly carries the opening composition into that ending "
            "composition, so the final frame lines up with the next shot for a seamless join. If `next_frame` is "
            "empty, just animate the motion with no forced ending."
        )
    else:
        mode_block = (
            "FRAME MODE — image-to-video: each shot's clip STARTS exactly from its still frame (prompt_img); "
            "describe only the motion that evolves from it (the composition may drift slightly, no hard cut)."
        )

    # Director guide — pacing/shots sections steer how motion prompts are written.
    director_block = ""
    if cfg and cfg.director_id:
        d = director_store.get(cfg.director_id)
        if d:
            director_block = _director_block(
                d.sections,
                ["tone_mood", "pacing_editing", "shots_framing"],
            )
    # Active brand's VO tone/pace steer the delivery of shots that have voice_over.
    brand_block = _brand_video_block(brand)
    director_block = "\n\n".join(b for b in (director_block, brand_block) if b.strip())

    # Voice config for the Omni narration block. The route already merged the global Character
    # config with any per-request override, so this only unpacks it — the literal defaults that
    # used to live here moved to config.yaml's `voice:` section, which is now the single floor.
    voice_cfg = {
        "language": state.get("vo_language"),
        "vo_pace": state.get("vo_pace"),
        "tone": state.get("vo_tone"),
        "style": state.get("vo_style"),
        "gender": state.get("vo_gender"),
    }

    with obs.span("node.generate_video_prompts", input={"scenes": len(scenes), "shots": total}) as span:
        emit({"type": "status", "message": f"Writing video prompts for {total} shots..."})

        failed_scenes: list[str] = []

        async def _one(sc) -> None:
            # Omni-only (POC parity): author each shot's prompt DIRECTLY for Omni (visual half + the
            # verbatim on-screen/narration/duration blocks) instead of a Veo-style prompt that later
            # needs LLM-adapting at render. Per-shot LLM call; scenes run concurrently under `sem`.
            async with sem:
                # Repair boards authored before the one-shot-per-caption rule existed: without this
                # the duplicates below become N clips each burning the same title graphic. Done here
                # rather than only at breakdown because this route persists a new storyboard version
                # anyway (media="drop_video"), so the change is visible and revertable.
                _shot_dicts = [{"on_screen_text": s.on_screen_text, "voice_over": s.voice_over,
                                "motion_description": s.motion_description,
                                "ingredient_refs": s.ingredient_refs, "equipment_refs": s.equipment_refs}
                               for s in sc.shots]
                # Same idea for the all-items overview shot, which older boards captioned with every
                # item at once. Runs FIRST so the emptied caption can't win the dedupe below.
                if (_ov := _clear_overview_on_screen_text(_shot_dicts, sb.ingredients, sb.equipment)):
                    emit({"type": "progress",
                          "message": f"📝 ฉาก {sc.scene_id}: ช็อตรวมทุกอย่าง — ตัด on-screen text ออก {_ov} ช็อต"})
                if (_blanked := _dedupe_scene_on_screen_text(_shot_dicts)):
                    emit({"type": "progress",
                          "message": f"📝 ฉาก {sc.scene_id}: on-screen text ซ้ำ — เก็บไว้ช็อตที่เกี่ยวที่สุด เว้นว่าง {_blanked} ช็อต"})
                if _ov or _blanked:
                    for s, d in zip(sc.shots, _shot_dicts):
                        s.on_screen_text = d["on_screen_text"]

                prev_sh = None
                for sh in sc.shots:
                    fields = {
                        "prompt_img": sh.prompt_img,
                        "motion_description": sh.motion_description,
                        "voice_over": sh.voice_over,
                        "on_screen_text": sh.on_screen_text,
                        "key_message": getattr(sh, "key_message", "") or "",
                        "time": sh.time,
                    }
                    man = (state.get("omni_manifests") or {}).get(f"{sc.scene_id}:{sh.no}", {})
                    # On continuous/match_cut the render seeds this clip with the PREVIOUS clip's last
                    # frame (see _omni_prev_frame_ref), so frame 0 is not this shot's still and the
                    # author cannot see what is in it — it gets told. On a `cut` the opening frame IS
                    # this shot's own still, which prompt_img already describes, so nothing is passed
                    # and that path behaves exactly as before. This loop is sequential within a scene,
                    # so prev_sh.prompt_video is the one just written a moment ago.
                    prev_ctx = ({"prompt_video": prev_sh.prompt_video,
                                 "motion": prev_sh.motion_description}
                                if prev_sh is not None and sh.join_with_prev in ("continuous", "match_cut")
                                else None)
                    # One failing shot must not kill the step (network / bad LLM output).
                    try:
                        pv, secs = await svc.gemini.generate_omni_prompt(
                            fields, vcfg.aspect_ratio, voice=voice_cfg,
                            image_manifest=man.get("listing", ""), source_header=man.get("header", ""),
                            usage_block=man.get("usage", ""), prev_context=prev_ctx)
                    except Exception as exc:  # noqa: BLE001 — isolate the shot, surface below
                        logger.warning(f"Omni prompt generation failed for {sc.scene_id}.{sh.no}: {exc}")
                        failed_scenes.append(f"{sc.scene_id}.{sh.no}")
                        prev_sh = sh   # still the previous shot for the next one, prompt or no prompt
                        continue
                    if pv:
                        sh.prompt_video = pv
                        # Only overwrite when the author actually produced an estimate — a parse
                        # failure must not wipe a length the user hand-corrected earlier.
                        if secs:
                            sh.target_seconds = secs
                    prev_sh = sh

        await asyncio.gather(*(_one(sc) for sc in scenes))

        # NO LLM re-validation here (Veo-era _validate_video_prompts is intentionally skipped):
        # it rewrote the WHOLE prompt and destroyed the code-assembled locked blocks — the
        # [# Sources]/[# References] header, per-ref lock rules, on-screen-text and narration
        # blocks — inlining the voice-over as prose. Those guarantees are now deterministic:
        # _normalize_omni_tags + code-built header/lock at author time, and _omni_apply_manifest
        # rebuilds the locked blocks from the actual images again at render time.

        made = sum(1 for sc in scenes for sh in sc.shots if sh.prompt_video)
        logger.info(f"Video prompts: {made}/{total} shots")
        # Surface shots left without a prompt_video now (else they die silently at the video step).
        if made < total or failed_scenes:
            missing = [f"{sc.scene_id}.{sh.no}" for sc in scenes for sh in sc.shots if not sh.prompt_video]
            emit({"type": "progress",
                  "message": f"⚠️ {total - made} shot(s) ไม่ได้ prompt_video: {', '.join(missing[:12])}"
                             + (f" (scene ที่ LLM พัง: {', '.join(failed_scenes)})" if failed_scenes else "")
                             + " — กด generate ซ้ำหรือพิมพ์เองใน Step ถัดไป"})
        span.update(output={"prompts": made, "total": total, "failed_scenes": failed_scenes})

    emit({"type": "storyboard", "content": sb.model_dump()})
    return {"storyboard": sb.model_dump()}


async def fail_no_relevant(state: PipelineState, config: RunnableConfig) -> dict:
    emit({"type": "error", "message": "No relevant videos found for this topic. Try rephrasing."})
    return {"error": "no_relevant"}


# ----------------------------------------------------------------------------- #
# Routing functions (conditional edges)
# ----------------------------------------------------------------------------- #

def after_search(state: PipelineState) -> str:
    return "filter" if state.get("pool") else "no_videos"


def after_filter(state: PipelineState, config: RunnableConfig) -> str:
    svc = services(config)
    confirmed = state.get("confirmed", [])
    target = state["target"]
    if len(confirmed) >= target:
        return "analyze"
    # Early stop: several rounds in a row found candidates but NONE were relevant → the rest of
    # YouTube isn't on-topic, so proceed with what we have instead of burning more top-up rounds.
    barren_cap = svc.settings.search.stop_after_barren_rounds
    if barren_cap > 0 and state.get("barren_rounds", 0) >= barren_cap:
        return "analyze" if confirmed else "no_relevant"
    if state.get("round_num", 1) >= svc.settings.search.max_search_rounds:
        return "analyze" if confirmed else "no_relevant"
    return "topup"


def after_topup(state: PipelineState) -> str:
    if state.get("pool"):
        return "filter"
    # exhausted — proceed with whatever we have, or fail if nothing
    return "analyze" if state.get("confirmed") else "no_relevant"


def after_analyze(state: PipelineState) -> str:
    return "synthesize" if state.get("summaries") else "end"


def after_synthesize(state: PipelineState) -> str:
    cfg = state.get("script_config")
    return "script" if (cfg and cfg.enabled) else "end"
