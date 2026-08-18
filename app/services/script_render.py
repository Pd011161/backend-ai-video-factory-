"""Render a ScriptDocument into the storyboard Markdown template.

This is the single source of truth for the Markdown layout. The LLM only emits
structured data; the human-readable script is built here so the two never drift.
Keep this in sync with SCRIPT_PROMPT's OUTPUT FORMAT section.
"""

from __future__ import annotations

from app.models.script import ScriptDocument


def render_markdown(draft: ScriptDocument) -> str:
    p = draft.production
    n_parts = p.parts_count or len(draft.parts)
    lines: list[str] = []

    lines.append("**VIDEO SCRIPT — พร้อมพัฒนาเป็น Storyboard**")
    lines.append("")
    lines.append(f"**{draft.title}**")
    lines.append("")

    # Production info table
    lines.append(f"| ตัวละครหลัก | {p.character_name} — {p.character_desc} |")
    lines.append("| :---- | :---- |")
    lines.append(f"| **Theme** | {p.theme} |")
    lines.append(f"| **อารมณ์หลัก** | {p.mood} |")
    lines.append(f"| **Material Palette** | {p.material_palette} |")
    lines.append(f"| **แสง** | {p.lighting} |")
    lines.append(f"| **โครงสร้าง** | {n_parts} ตอน (Part) · ความยาวต่อตอน {p.duration_per_part} |")
    lines.append("")

    # Overview
    lines.append(f"**ภาพรวมทั้ง {n_parts} ตอน**")
    lines.append("")
    for ov in draft.overview:
        lines.append(f"* PART {ov.number} — {ov.title} : {ov.summary}")
    lines.append("")

    # Parts + scenes
    for part in draft.parts:
        lines.append(f"**PART {part.number}**")
        lines.append("")
        lines.append(f"| {part.title} — {part.description}     •  ความยาวรวม ≈ {part.duration} นาที |")
        lines.append("| :---- |")
        lines.append("")
        for sc in part.scenes:
            lines.append(f"| ฉาก {sc.scene_id}  ·  {sc.name} ⏱ {sc.timecode_start} – {sc.timecode_end} |  |")
            lines.append("| :---- | :---- |")
            lines.append(f"| **เชื่อม** | {sc.transition} |")
            lines.append(f"| **ประเภทช็อต** | {sc.shot_type} |")
            lines.append(f"| **แนวทางภาพ / ภาพประกอบ** | {sc.visual_direction} |")
            lines.append(f"| **บทพูด (Voice Over)** | {sc.voice_over} |")
            lines.append(f"| **ข้อความบนหน้าจอ** | {sc.on_screen_text} |")
            lines.append(f"| **สาระสำคัญ / Key Message** | {sc.key_message} |")
            lines.append(f"| **ดนตรีประกอบ** | {sc.music} |")
            lines.append("")

    return "\n".join(lines).strip() + "\n"
