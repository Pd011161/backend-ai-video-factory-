from pydantic import BaseModel, Field


class ImageElement(BaseModel):
    """One atomic visual element of a scene plus its searched reference candidates.

    Populated by the image-search node, not the LLM.
    """

    subject: str = Field(description="The atomic visual element (from image_subjects)")
    query: str = Field(default="", description="Search query produced by transforming the subject")
    candidates: list[str] = Field(default_factory=list, description="Candidate reference image URLs")


class SceneEntry(BaseModel):
    scene_id: str = Field(description="Scene ID in part.scene format, e.g. '1.1', '2.3'")
    name: str = Field(description="Short descriptive scene name/title")
    transition: str = Field(
        description=(
            "เชื่อม — the scene's narrative role / transition label, e.g. "
            "'เปิดตัว / Hook', 'เชื่อมเข้าเนื้อหา', 'สรุป', 'Teaser/CTA'. "
            "Describes how this scene connects the story to the next."
        )
    )
    scene_kind: str = Field(
        default="",
        description=(
            "Machine-readable scene role, independent of `transition`/`name` wording or language: "
            "'intro_ingredients' for the dedicated ingredient-introduction scene, 'intro_equipment' "
            "for the dedicated equipment/tools-introduction scene, or '' for every other scene."
        ),
    )
    timecode_start: str = Field(description="Scene start timecode, e.g. '0:00'")
    timecode_end: str = Field(description="Scene end timecode, e.g. '0:35'")
    shot_type: str = Field(
        description=(
            "Camera shot type PLUS a concrete framing line, in the form "
            "'<Shot type> — <who/what is in the frame, their action, key props/position>'. "
            "NEVER the bare shot type. e.g. 'Medium Shot — ครูพี่เกศยืนหลังเคาน์เตอร์หินอ่อน "
            "มือถือถ้วยหมี่กรอบสำเร็จรูปยกโชว์', 'Close-up ถ้วยหมี่กรอบหมุนช้าๆ + Insert motion "
            "graphic ตัวเลข', 'Top-down (Flat Lay) วัตถุดิบทั้งหมด + Pan ช้าๆ'."
        )
    )
    visual_direction: str = Field(
        description=(
            "The frame described as several distinct visual beats (separate with ' / ' or "
            "newlines), NOT one action sentence: (1) setting + lighting, (2) key props / "
            "set dressing, (3) the focal subject + its action / position in frame, "
            "(4) any motion-graphic / insert / on-screen-number cues. Shootable without guessing."
        )
    )
    voice_over: str = Field(description="Voice over script in Thai, ready to record")
    on_screen_text: str = Field(description="On-screen text, captions, and motion graphic cues")
    key_message: str = Field(description="Core learning objective or key message for this scene")
    music: str = Field(description="Music mood, tempo, genre, and SFX notes")


class PartOverview(BaseModel):
    number: int = Field(description="Part number starting from 1")
    title: str = Field(description="Part title")
    summary: str = Field(description="One-line summary of part content")


class ScriptPart(BaseModel):
    number: int = Field(description="Part number starting from 1")
    title: str = Field(description="Part title")
    description: str = Field(
        description=(
            "Part subtitle/description — the part's theme plus the ordered list of "
            "steps/topics it covers, e.g. 'เปิดครัว & เตรียมความกรอบ · เกริ่นนำ · "
            "วัตถุดิบ · ไข่กรอบ · ทอดเส้นหมี่ · ทอดเครื่อง · เตรียมธัญพืช'."
        )
    )
    duration: str = Field(description="Estimated total duration, e.g. '6:20'")
    scenes: list[SceneEntry] = Field(description="All scenes in this part in order")


class ProductionSpec(BaseModel):
    character_name: str = Field(description="Main character name")
    character_desc: str = Field(description="Character description")
    theme: str = Field(description="Set and production theme")
    mood: str = Field(description="Emotional tone and mood")
    material_palette: str = Field(description="Material and color palette")
    lighting: str = Field(description="Lighting setup description")
    dish_appearance: str = Field(
        default="",
        description=(
            "Canonical appearance of the FINISHED dish in English — its final shape, color, "
            "texture and plating (e.g. 'golden-brown crispy fried mashed-potato sticks shaped "
            "like thick fries, crisp outside, soft inside, on a white plate'). Derived from the "
            "synthesis so every shot that shows the finished dish renders it the SAME way."
        ),
    )
    ingredients: list[str] = Field(
        default_factory=list,
        description=(
            "The COMPLETE ingredient list of the recipe, each entry a name + quantity in the "
            "topic's language (Thai if Thai), e.g. ['มันฝรั่งขนาดกลาง 2 หัว', 'เนยจืด 1 ช้อนโต๊ะ', "
            "'ไข่แดง 1 ฟอง']. Taken from the synthesis. Used to render the 'all ingredients' "
            "overview shot exactly (no guessing / no extra items)."
        ),
    )
    equipment: list[str] = Field(
        default_factory=list,
        description=(
            "The COMPLETE list of TOOLS / EQUIPMENT the tutorial physically uses, each a standalone "
            "item name in the topic's language (e.g. ['กระทะก้นลึก', 'กระชอน', 'ชามผสม', 'ไม้พาย']). "
            "Consumable ingredients do NOT belong here. Empty if the recipe uses no notable tools. "
            "Used to render the 'all equipment' overview shot and per-tool intro shots."
        ),
    )
    parts_count: int = Field(description="Total number of parts/episodes")
    duration_per_part: str = Field(description="Duration per part, e.g. '06:00-06:30'")


class ScriptDocument(BaseModel):
    """The script as pure structured data — the only thing the LLM produces.

    Markdown is NOT stored here. It is a derived view: render it on demand from
    these fields with ``app.services.script_render.render_markdown``. Keeping the
    document markdown-free means the saved JSON stays compact and the LLM output
    is half the size (the content isn't duplicated), so it doesn't hit the
    output-token limit.
    """

    title: str = Field(description="The main topic or recipe title")
    production: ProductionSpec
    overview: list[PartOverview] = Field(description="High-level overview of all parts")
    parts: list[ScriptPart] = Field(description="Complete script parts with all scenes")
