from __future__ import annotations

import json
import string
from pathlib import Path

from loguru import logger

from app.core.config import ROOT_DIR
from app.services.store_io import atomic_write_text

# ── Category rule blocks (food / drink) ─────────────────────────────────────────
# The base prompts carry only GENERIC rules; each prompt appends ONE category block via the
# {category_rules} placeholder ({finished_look_rule} in the script prompt), selected by
# ScriptConfig.category (manual toggle). Food and drink each get their own examples/rules so
# neither reads the other's text — add new per-category rules HERE, never in the base prompts.

FOOD_STORYBOARD_RULES = """
CATEGORY RULES — FOOD (apply together with the rules above):
  • TOOLS BY ACTION: Peeling → a peeler (and a bowl for the peels) — NO knife, NO cutting board. Cutting/slicing → a knife + cutting board. Mixing/mashing → a bowl + spoon/masher/fork. Frying → a pan/pot.
  • STATE CHAIN: the dish progresses raw → peeled → washed → cut → boiled → mashed → mixed → shaped → coated → fried; describe each ingredient at its CURRENT step — e.g. "มันฝรั่งที่ปอกและหั่นเป็นชิ้นสี่เหลี่ยมแล้ว" (peeled, cut potato), NEVER just "มันฝรั่ง 2 หัว" after it has been transformed.
  • PLATING: do NOT add garnish / herbs / sauce / side dishes on top of the dish unless the canonical look or the script explicitly says so — identical plating every time.
  • VESSELS: name them specifically like "ชามผสมสแตนเลส", "หม้อสแตนเลส", "เขียงไม้"; state contents precisely (e.g. "ใส่มันฝรั่งลงหม้อ" = pot with potato and NO water yet — the water comes in a later shot).
  • LIQUID VESSELS (food): a liquid ingredient shown on its own — sauce, stock, fish sauce, soy sauce, vinegar, milk, syrup, juice — goes in a SMALL PLAIN CUP (ถ้วยเล็ก: a round dish with no handle), NOT a measuring cup. Two exceptions: PLAIN WATER (น้ำเปล่า) may use a plain clear glass measuring cup, and COOKING OIL (น้ำมัน) is ALWAYS shown in its own clear glass oil BOTTLE (ขวดน้ำมัน) — never decanted into a cup or a bowl. Any measuring cup in shot is PLAIN: no numbers, no measurement lines, no pattern or logo.
  • ACTION EXAMPLE: peeling = "ปอกมันฝรั่งลูกหนึ่งไปได้บางส่วน เห็นเนื้อขาวที่ปอกแล้วและเปลือกที่เหลือ เปลือกบางส่วนหล่นบนเขียง อีกลูกยังไม่ปอกวางข้างๆ" — not a vague "peeling potatoes".
  • PROCESS SHOTS: name the food physically IN the vessel — "chunks of potato in the boiling water", "chicken pieces sizzling in the pan", "dough being kneaded on the board"."""

DRINK_STORYBOARD_RULES = """
CATEGORY RULES — DRINK (apply together with the rules above):
  • VESSEL FLOW: liquids are MIXED in the clear glass measuring cup through every prep step and poured into the SERVING GLASS only at the final serving step; state the drink's SPECIFIC type and its expected COLOUR at THIS step (matcha → bright green; Thai tea → orange; espresso → dark brown) so the liquid renders the correct colour, not a generic brown.
  • SPOON-MEASURED DRY: an ingredient measured by SPOON (ช้อนตวง/ช้อนโต๊ะ/ช้อนชา) is shown IN/ON that measuring spoon at its matching size — NOT a bowl; a bowl (sized to fit) only for a loose weight (กรัม) or whole pieces (หัว/ฟอง/ลูก/ใบ/แผ่น). Match each ingredient's vessel to the UNIT the recipe states.
  • SOURCE VESSEL: when ADDING an ingredient, it pours from the vessel its UNIT implies — an ml-measured liquid from its own small glass measuring cup (NEVER from a measuring spoon; an ml amount does not fit a spoon), a spoon-measured dry amount from that measuring spoon.
  • TOOLS BY ACTION: Stirring → a long spoon. Brewing/straining → the tea filter bag (+ tongs to lift or squeeze it). Pouring → the measuring cup/jug it was mixed in. Serving → the stated serving glass.
  • STATE CHAIN: the drink progresses powder → dissolved/brewed concentrate → sweetened/creamed mix → poured over ice / into the serving glass; describe its CURRENT colour, opacity and vessel — e.g. "dark green tea concentrate, no milk yet, in a glass measuring cup".
  • CONSISTENCY: the drink's colour, opacity and layers stay IDENTICAL across shots of the same step; NO milk / ice / topping before the step that actually adds it.
  • VESSELS: name them specifically like "แก้วตวงแก้วใส", "เหยือกตวง", "ช้อนตวงสแตนเลส", "แก้วเสิร์ฟ"; state contents precisely (e.g. "เทน้ำร้อนลงถุงชาในแก้วตวง" = hot water over the tea bag, nothing else added yet). Every measuring cup/jug is PLAIN glassware: NO numbers, NO measurement lines, no pattern or logo (markings render it as a laboratory beaker).
  • ACTION EXAMPLE: squeezing the tea bag = "her left hand steadies the bag's knotted top while her right hand squeezes the bag's lower half with metal tongs, dark green tea streaming down into the cup".
  • PROCESS SHOTS: name the liquid/ingredient IN the vessel — "the tea filter bag steeping in hot water, dark colour blooming", "condensed milk streaming from a small measuring cup into the green tea"."""

# The image-prompt writers mirror the storyboard bullets → the image_prompt slot reuses the same
# *_STORYBOARD_RULES block (see CATEGORY_RULES); no separate image-rules constant.

FOOD_PLAN_RULES = """
CATEGORY RULES — FOOD:
  • is_process WRONG: "A pot of boiling water on the stove, steam rising." RIGHT: "A pot of boiling water with potato chunks visible below the surface, steam rising." WRONG: "A frying pan with sizzling oil over high heat." RIGHT: "A frying pan with diced chicken pieces sizzling in the oil."
  • STATE EXAMPLES: "two peeled potatoes cut into small even cubes" (not "two potatoes"); "smooth mashed potato" (not "potato").
  • VESSELS: e.g. "a stainless steel mixing bowl", "a wooden cutting board"; a small bowl for a dry/solid ingredient shown alone.
  • LIQUID VESSELS: a lone liquid (sauce, stock, fish sauce, soy sauce, vinegar, milk, syrup, juice) sits in a small plain cup — a round dish with no handle, NOT a measuring cup. Plain water may use a plain clear glass measuring cup; cooking oil is always shown in its clear glass oil bottle, never poured into a cup or bowl. Any measuring cup is plain: no numbers, no measurement lines, no pattern."""

DRINK_PLAN_RULES = """
CATEGORY RULES — DRINK:
  • VESSEL FLOW: liquids are MIXED in the measuring cup through every prep step and poured into the SERVING GLASS only at the serving step; state the drink's specific type and its expected COLOUR at this step (matcha → bright green, Thai tea → orange, coffee → dark brown) so the liquid renders correctly.
  • SPOON-MEASURED DRY: a spoon-measured (ช้อนตวง/ช้อนโต๊ะ/ช้อนชา) ingredient sits IN/ON that measuring spoon — NOT a bowl; a bowl (sized to fit) only for grams or whole pieces.
  • SOURCE VESSEL: an ml-measured liquid pours from its own small glass measuring cup — NEVER from a measuring spoon; a spooned dry amount pours from that spoon.
  • is_process WRONG: "A glass measuring cup of hot water, steam rising." RIGHT: "The tea filter bag steeping in the glass measuring cup of hot water, dark green colour blooming out of the bag." WRONG: "Milk pouring into the tea." RIGHT: "Evaporated milk streaming from a small glass measuring cup into the dark green tea concentrate."
  • STATE EXAMPLES: "dark green tea concentrate, no milk yet, in a glass measuring cup"; "pale creamy green tea mix after the condensed milk dissolves".
  • CUP LOOK: every measuring cup/jug is PLAIN glassware — no numbers, no measurement lines, no pattern (markings render it as a laboratory beaker)."""

FOOD_FINISHED_LOOK = (
    " For a DISH, describe its shape, color, texture and plating (e.g. \"golden-brown crispy fried"
    " mashed-potato sticks shaped like thick fries, crisp outside and soft inside, on a white plate\")."
)

DRINK_FINISHED_LOOK = (
    " For a DRINK, describe the finished beverage's COLOUR, clarity/opacity, any layers or ice, and the"
    " serving glass (e.g. \"bright green iced matcha latte with a milk layer over ice cubes in a tall clear glass\")."
)

# Slots map to CONSTANT NAMES (not values) so category_block() resolves through the override
# store — editing FOOD_STORYBOARD_RULES on the web changes both the storyboard and image_prompt
# slots (both point at the same block).
CATEGORY_RULES = {
    "food": {"storyboard": "FOOD_STORYBOARD_RULES", "image_prompt": "FOOD_STORYBOARD_RULES",
             "image_plan": "FOOD_PLAN_RULES", "finished_look": "FOOD_FINISHED_LOOK"},
    "drink": {"storyboard": "DRINK_STORYBOARD_RULES", "image_prompt": "DRINK_STORYBOARD_RULES",
              "image_plan": "DRINK_PLAN_RULES", "finished_look": "DRINK_FINISHED_LOOK"},
    # Neither food nor drink. Deliberately empty rather than absent: "food" is NOT a neutral base —
    # it carries peel/chop/boil state chains, stainless bowls and plating language, so letting this
    # fall through to it told the model to cook a dish for a topic that isn't one.
    "other": {},
}


def category_block(category: str, slot: str) -> str:
    """The category-specific rule block for a prompt slot.

    "other" is a KNOWN category that maps to no block at all (see CATEGORY_RULES); the food fallback
    below is only for values that are genuinely unrecognised."""
    name = CATEGORY_RULES.get(category, CATEGORY_RULES["food"]).get(slot, "")
    return get_prompt(name) if name else ""


STORYBOARD_PROMPT = """\
You are a storyboard artist. Break ONE script scene into a sequence of individual camera SHOTS.

STAY 100% FAITHFUL TO THE SCRIPT — do NOT add or remove any content:
- The shots' voice_over fields, read in order, must reconstruct the scene's voice_over EXACTLY — same words, nothing added, nothing dropped, no rewriting. Split it at natural sentence/clause boundaries across the shots.
- NO DANGLING LEAD-IN (HARD RULE) — every shot's voice_over must END on a COMPLETE thought. NEVER end a shot on a connector word OR a lead-in PHRASE / clause that announces what comes next — e.g. "และ", "หรือ", "ต่อด้วย", "ชิ้นต่อไปคือ", "ถัดมานะคะเป็น", "แล้วก็มี", "สุดท้ายนะคะเป็น", "อีกอย่างที่ขาดไม่ได้คือ" (examples only — apply to ANY word or phrase of this kind, however long). Move the ENTIRE trailing lead-in phrase to the START of the NEXT shot's voice_over instead. Example — WRONG: shot 2 = "เกลือป่น 1 ช้อนโต๊ะ ค่ะ ชิ้นต่อไปคือ", shot 3 = "น้ำปลา 1 ช้อนโต๊ะ" · RIGHT: shot 2 = "เกลือป่น 1 ช้อนโต๊ะ ค่ะ", shot 3 = "ชิ้นต่อไปคือ น้ำปลา 1 ช้อนโต๊ะ". This only shifts the split boundary — the words read in order stay identical, so it does NOT add, drop or rewrite anything. (Why: each shot renders as its own clip; a clip that ends mid-thought makes the speech hang or the video model ad-lib extra words.)
- ON-SCREEN TEXT — EACH CAPTION ON THE ONE SHOT THAT SAYS IT (HARD RULE). Split the scene's on_screen_text at its separators (·, •, |, line breaks) into individual captions and keep each caption's wording EXACTLY as written — you are DISTRIBUTING the scene's captions, never rewriting or inventing them. For each caption, read every shot's voice_over and place it on the ONE shot whose line actually says it:
  - a caption naming an item, a quantity or an action → the shot whose voice_over speaks that item/quantity/action;
  - a caption naming the STEP itself (a heading like "ล้างไก่ด้วยเกลือ") → the shot that OPENS that step, normally the scene's first shot;
  - a caption no shot's line covers → leave it out entirely.
  ONE SHOT MAY HOLD SEVERAL CAPTIONS: if a single shot's voice_over covers two of them, that shot gets BOTH, joined with " · " in the scene's original order — do NOT spread them across shots just to give each shot something. Every shot whose line covers no caption gets on_screen_text: "" — most shots in a scene SHOULD be empty.
  NEVER CARRY A CAPTION FORWARD (this is the most common mistake). on_screen_text is THIS shot's own caption, not a running overlay that accumulates down the scene. Once a caption sits on its shot, it must not appear on any later shot. Example — scene captions "ล้างไก่ด้วยเกลือ · เกลือป่น 1 ชต. · ขยำ 2-3 นาที" over 5 shots whose voice_over is 1:"ขั้นตอนแรกสำคัญมากค่ะ!" 2:"เราจะนำปีกไก่ใส่ชามผสม" 3:"ใส่เกลือป่นหนึ่งช้อนโต๊ะลงไปเลยค่ะ" 4:"จากนั้นขยำและถูให้ทั่วประมาณสองถึงสามนาที" 5:"การล้างด้วยเกลือจะช่วยให้ไก่สะอาด" · WRONG: shot 1 "ล้างไก่ด้วยเกลือ", shot 2 "ล้างไก่ด้วยเกลือ", shot 3 "ล้างไก่ด้วยเกลือ · เกลือป่น 1 ชต.", shot 4 "ล้างไก่ด้วยเกลือ · เกลือป่น 1 ชต. · ขยำ 2-3 นาที", shot 5 the same again — every shot re-showing everything before it, and shots 1-2 showing a caption their line never mentions. · RIGHT: shot 1 "ล้างไก่ด้วยเกลือ" (the step heading, on the shot that opens the step), shot 2 "", shot 3 "เกลือป่น 1 ชต.", shot 4 "ขยำ 2-3 นาที", shot 5 "".
  (Why both halves matter: each shot renders as its own clip, so a caption repeated on three shots makes the same title graphic pop in and out three times; and a caption on a shot that never says it puts words on screen the narration does not support.)
- Put the scene's key_message on the shot where it fits best; other shots may leave key_message empty.
- The system assigns each shot's FINAL timecode from the clip count (one ~{duration}-second clip per shot), so do NOT pad or stretch shots to fill the scripted scene time {start}–{end} — just split the scene into the natural number of single-beat shots (estimate each from how much voice_over it carries). The `time` field you write is only a rough estimate; the real runtime = number of shots × clip length.
- CLIP SIZING — each shot is rendered as ONE ~{duration}-second video clip whose audio speaks that shot's voice_over verbatim, so every shot must be a SINGLE continuous beat and its voice_over must be a short line comfortably speakable in ~{duration} seconds (never more than ~8 seconds of speech){words_hint}. If a beat's spoken line is longer, SPLIT it into several consecutive shots. This makes shots shorter and more numerous — that is expected; still keep the time ranges contiguous and spanning the whole scene.{shot_hint}
- Use ALL of the scene's fields below (name, transition, shot_type, visual_direction, voice_over, on_screen_text, key_message, music) — do NOT drop information. motion_description must combine the scene's shot_type (with its framing) AND every visual beat in visual_direction (one concrete shot per entry); reflect the scene's music/mood and transition role where relevant. When a shot shows ingredients, materials, or tools/equipment, motion_description MUST show ONLY the items THAT THIS SHOT is about — read from BOTH this shot's voice_over slice AND its on_screen_text slice — and NAME each with its EXACT quantity (e.g. "มันฝรั่งขนาดกลาง 2 หัว วางบนเขียง"). Do NOT pull in items from other shots or from the whole scene; what is SHOWN must match what is SPOKEN in this shot. Only mention the tool(s) the ACTION actually uses (ปอก→ที่ปอก ไม่ต้องมีมีด/เขียง · หั่น→มีด+เขียง · ผสม→ชาม+ไม้พาย) — do NOT add idle tools the step does not use. NEVER a group word like "วัตถุดิบทั้งหมด", "อุปกรณ์ต่างๆ" or "วัตถุดิบที่จัดเรียง". EXCEPTION: only when THIS shot is an explicit OVERVIEW / flat-lay of the ENTIRE set (its own voice_over/on_screen_text refers to all items at once) does it list every item by name — "ปลายปีกไก่ เกลือ น้ำปลา น้ำตาลทราย พริกไทย ผงปรุงรส ผงปาปริก้า แป้งทอดกรอบ" (keep the main 6–8 if very long). The shots together must cover the FULL content of the scene.

For each shot, also write:
- prompt_img — ONE English paragraph that renders THIS shot as a clean, photorealistic {aspect} widescreen storyboard FRAME. This still will be ANIMATED into video next (and used as the first/last frame for video interpolation), so quality, clarity and consistency matter. Follow this structure: "A photorealistic <shot type / framing> of <subject>, <what they do in this shot>, set in <setting + key props>. Lit by <lighting>, <mood> atmosphere. Shot on <lens, e.g. 35mm, shallow depth of field>, {aspect} widescreen, cinematic." Rules:
  • CHARACTER (person shots only) — FIRST set this shot's `shows_face`: TRUE when the host's face/head is in frame (she talks to camera, or a medium/wide shot shows her head); FALSE when the camera is tight on her HANDS or the food / the process / over-the-shoulder (her face is NOT shown). shows_face=TRUE → refer to her with the SAME short visual descriptor in EVERY such shot (e.g. "a Thai woman chef in her 30s with a warm, friendly look"); no proper name, do NOT over-describe her exact face/hair. shows_face=FALSE → the SUBJECT is "the chef's hand"/"the chef's hands" (same skin tone, bracelet/watch) framed close on the action — write NO appearance descriptor at all (no age, no "friendly look", no hair, no body); a character reference image is supplied at generation and locks her identity, outfit and hand. Keep her outfit and the setting IDENTICAL across the scene. For shot_kind="insert", do NOT mention or describe the host AT ALL (no body, no hands).
  • COMPOSITION FOR ANIMATION — one clear focal subject, uncluttered; capture a natural moment of the shot's action with room for that motion to play out (headroom / space in the direction of movement); eye-level unless shot_type says otherwise. NO text, captions, numbers, logos or graphics anywhere in the frame.
  • SHOT GRAMMAR (lens + camera height) — choose the LENS and CAMERA HEIGHT by shot size and write them into the "Shot on <lens>" slot — do NOT default every shot to 35mm: wide/establishing → 24–28mm at eye level, deep focus; medium (host talks/acts) → 35mm at chest height; close-up of hands/action → 50mm slightly above counter level, shallow depth of field; insert/detail of food or tools → 85–100mm macro — straight top-down for a flat-lay, or a LOW 30–45° "hero" angle at counter height for appetizing food; finished-dish beauty shot → 85mm hero angle, very shallow focus. Subject sharp / background soft on close-ups; keep wides deeper.
  • DEPTH STAGING — where natural, compose in 3 layers: a soft out-of-focus FOREGROUND element at a frame edge (a bowl rim, herbs, rising steam), the subject in the MIDGROUND, and real BACKGROUND depth (the kitchen falling away behind) — avoid flat subject-against-wall compositions in person shots.
  • OBJECTS — if the shot shows ingredients/materials/tools, NAME in English ONLY the objects THIS shot is about (from this shot's own voice_over / on_screen_text), with their quantities — what is SHOWN must match what is SPOKEN in this shot. Do NOT include items from other shots or the whole scene UNLESS this shot is an explicit overview/flat-lay of the ENTIRE set (then list the main 6–8). NEVER a vague "various cooking ingredients and utensils", and NEVER use a generic filler subject like "showcasing cooking ingredients" — the structure's <subject> must be the SPECIFIC named item(s) of THIS shot.
  • TOOLS MATCH THE ACTION — include ONLY the tool(s) the shot's action actually uses, and NO others (see the CATEGORY RULES tool-action pairings). Do NOT add idle props (extra utensils, gadgets) that the action does not require, even if they are common kitchen items.
  • GROUNDING — every object RESTS on a real surface; nothing floats in mid-air. FOOD IS NEVER LOOSE ON THE COUNTER — every ingredient, raw item or finished dish sits IN or ON a vessel/work surface (a bowl, cup, plate, tray, board, pan/pot or rack), chosen per the CONTAINERS rule. Never "3-4 coriander roots resting on the marble counter" or "700 g of raw chicken wings on the countertop" — put them in the vessel their nature calls for. Tools and equipment MAY rest on the bare counter; food may not. Use the kitchen's EXISTING fixtures (sink, faucet, stove, cabinets, window) from the setting — do NOT invent or add new fixtures.
  • LIGHTING — soft, even, natural light; NO rim light, backlight glow, halo or white outline around the subject. The subject blends naturally into the scene, not cut out from it.
  • DISH CONSISTENCY — when the shot shows the FINISHED dish, describe it with this EXACT canonical look (verbatim) so every finished-dish shot is identical: "{dish}". When the shot shows an IN-PROGRESS stage, describe the dish's ACTUAL state at THAT step faithfully (per the CATEGORY RULES state chain), never the finished look. Do NOT add anything on top of the finished look unless the canonical look or the script explicitly says so (keep the presentation identical every time).
  • INGREDIENT STATE — describe EVERY ingredient's CURRENT physical state at THIS step, not the generic name. Once it has been transformed, say so — the state must match the actual step in the CATEGORY RULES state chain.
  • CONTAINERS / EQUIPMENT — name the vessel SPECIFICALLY (see the CATEGORY RULES vessel examples) and keep the SAME one across consecutive steps that use it (do not switch the in-progress dish to a different vessel/colour/size between shots). When a reference photo of an ingredient is supplied, its VESSEL is part of what must match: render the SAME bowl/cup/plate shown there (same material, colour, shape), not a new one — unless THIS shot moves the food somewhere else (into the pan, onto the board), which wins. The shot that INTRODUCES an ingredient or a tool (and the all-items overview) is where its container's colour and material are DECIDED — write them there. Any LATER shot reusing that same ingredient or tool names the container by TYPE only ("in its bowl") and states NO colour or material: the introduction's photo already fixes them, and restating one is how a single mixing bowl ends up white in some shots and stainless in others. (A vessel a later shot moves the food INTO — pan, pot, board, serving plate — is new to that frame and IS described in full.) EVERY ingredient visible in the frame is in a vessel — not loose on the counter — and when a shot SHOWS or measures one on its own it gets its OWN separate vessel: a small bowl for a DRY/solid ingredient, and for a LIQUID the vessel the CATEGORY RULES below name (they differ for food and for drink — follow them, not a default). Do NOT combine different ingredients into one vessel UNLESS the shot is explicitly about MIXING/combining them.
  • VESSEL CONTENTS — state exactly what is in a vessel at this step: an EMPTY vessel, the ingredient with nothing added yet, or the added liquid covering it — never add contents the step has not reached (see the CATEGORY RULES vessel examples).
  • ACTION REALISM — describe the action at a concrete, believable mid-stage, not a vague label (see the CATEGORY RULES action example).
  • INSERT shots (shot_kind="insert") — NO person and NO hands; frame ONLY the food/objects close-up or top-down on the counter surface. The structure's <subject> must be the SPECIFIC item(s) this shot shows (by name and quantity), NEVER a generic "cooking ingredients" / "various ingredients" / "all the ingredients" (that invites the model to add random extra items).
  • INGREDIENT-INTRODUCTION scene (transition = "แนะนำวัตถุดิบ") — its voice_over introduces ingredients ONE AT A TIME, so emit ONE insert shot PER ingredient: each shows that SINGLE ingredient ALONE in its own vessel (chosen per the CONTAINERS rule), shot_kind="insert", NO person / NO hand / NO pointing — the item just RESTS on the counter, the rest of the counter empty. STRICTLY ONE ingredient per shot — NEVER pair or group two DIFFERENT ingredients into one shot, even when the voice_over names them in one breath (split into two shots instead). EXCEPTION: the SAME ingredient listed twice for two amounts (hot vs cold, e.g. "ครีมเทียม 1 ช้อนโต๊ะ" and "ครีมเทียม 1 ช้อนโต๊ะพูน") is ONE shot showing that ingredient once, with BOTH canonical strings in its ingredient_refs. EVERY distinct ingredient in the canonical "Recipe ingredients" list gets its own shot — including plain ones like water and ice; skip NONE. The LAST shot of the scene is REQUIRED and is the all-together OVERVIEW: ALL the recipe's ingredients placed together (also insert, no hand) — and its ingredient_refs must list EVERY ingredient in the recipe and its equipment_refs MUST be empty (NO tools in an ingredient overview). VOICE SPLIT PER ITEM: the announcer phrase that introduces the NEXT ingredient ("ถัดมานะคะเป็น", "ชิ้นต่อไปคือ", "ต่อด้วย", "แล้วก็มี", …) belongs at the START of THAT next ingredient's shot — each item's shot ends with its own item (+ polite particle ค่ะ/นะคะ), NEVER with the lead-in for the next one.
  • EQUIPMENT-INTRODUCTION scene (transition = "แนะนำอุปกรณ์") — SAME as the ingredient-introduction scene but for TOOLS: emit ONE insert shot PER tool (each shows that SINGLE tool ALONE on the counter, shot_kind="insert", NO person / NO hand / NO pointing — it just RESTS), and its equipment_refs lists ONLY that one tool. STRICTLY ONE tool per shot — NEVER pair or group tools; EVERY item in the canonical "Tools / equipment" list gets its own shot, skip NONE. The LAST shot is REQUIRED and is the all-together OVERVIEW of ALL tools (also insert, no hand) — its equipment_refs must list EVERY tool and its ingredient_refs MUST be empty (NO ingredients in an equipment overview). VOICE SPLIT PER ITEM: the announcer phrase that introduces the NEXT tool ("ถัดมานะคะเป็น", "ชิ้นต่อไปคือ", "ต่อไปเป็น", …) belongs at the START of THAT next tool's shot — each tool's shot ends with its own item (+ polite particle), NEVER with the lead-in for the next one.
  • PROCESS shots (host physically manipulates the food/drink) — prompt_img MUST name the ingredient(s) being processed as PHYSICALLY VISIBLE and present in the scene (inside its vessel or on its surface), NOT just describe the vessel or action alone — NEVER an empty-vessel action shot (see the CATEGORY RULES process examples).
  • PROCESS FRAMING — a process shot frames the ACTION, not the host. Pick exactly ONE of two framings: (a) CLOSE-UP — the hands + tool + ingredient/vessel fill the frame (50mm, slightly above counter); or (b) MID SHOT CROPPED BELOW THE CHIN — torso / chest-down and hands visible, the face NOT in frame. NEVER show the host's face or full body in a process shot — her face appears ONLY in talking-to-camera / opening / closing shots. Write motion_description with the chosen framing explicit (e.g. "Close-up มือ..." / "ระดับอก ไม่เห็นใบหน้า").
  • Use the production spec below for setting, lighting and mood.
{category_rules}
- shot_kind — "insert" when NO part of the host is needed in the frame: a pure food/ingredient/tool shot (e.g. "butter sitting in a bowl on the counter", an ingredients flat-lay). "person" ONLY when the host genuinely acts with her HANDS or body — cutting, mixing, mashing, frying, holding/lifting an item, or talking to camera (her hand's skin tone & jewelry then stay consistent via the reference). IMPORTANT: a shot that merely INTRODUCES / SHOWS / names an ingredient or tool is "insert" — show the item simply PLACED / arranged on the counter, do NOT add a pointing hand. Use "person" for an ingredient shot only if the script describes a real hand action on it. When shot_kind="insert", motion_description AND prompt_img must describe ONLY the object(s) + a camera move (e.g. a slow push-in on the item) — NEVER the host, a hand, or 'ครูพี่เกศหยิบ...ขึ้นมาโชว์'; the item just rests and the camera reveals it (this text feeds the video-motion prompt next).
- image_subjects — a LIST of the ATOMIC subject OBJECTS visible in THIS shot (food, ingredients, tools, finished items), one standalone object per entry, IN THE TOPIC'S LANGUAGE (Thai if the topic is Thai, e.g. "เส้นหมี่แห้ง", "น้ำตาลมะพร้าว" — NOT English). Used for reference-image search. EXCLUDE people / characters / hands and the scene, setting, or background (kitchen, counter, room, environment, lighting). Include a tool ONLY if the shot's action actually uses it (per TOOLS MATCH THE ACTION — never idle tools). No camera jargon. May be empty if the shot shows no concrete object.
- ingredient_refs — a LIST of which RECIPE INGREDIENTS (from the canonical "Recipe ingredients" list below) this shot SHOWS or USES, copied VERBATIM from that list (exact same string). So a later cooking shot can reuse the exact image generated when the ingredient was first introduced. Include the ingredient whether the shot introduces it OR uses it in a step (e.g. a "mix in the butter" shot → ["เนยจืด 1 ช้อนโต๊ะ"]). INCLUDE the ingredient EVEN when the shot focuses on a vessel/action — if the potato is inside the boiling pot, list the potato; if mashed potato is in the bowl being mixed, list the potato. Empty ONLY if truly no listed ingredient is present. Use ONLY strings that appear in the canonical list — do not invent or rephrase. In the ingredient-introduction scene: a per-ingredient intro shot lists ONLY that one ingredient; the FINAL all-together overview shot lists EVERY ingredient in the recipe.
- equipment_refs — a LIST of which TOOLS / EQUIPMENT (from the canonical "Tools / equipment" list below) this shot SHOWS or USES, copied VERBATIM. Same idea as ingredient_refs but for tools: a tool NEVER changes state, so a later shot reuses the exact image generated when the tool was introduced. Include the tool whether the shot introduces it OR uses it in a step. Empty when no listed tool is present. Use ONLY strings from the canonical list. In the equipment-introduction scene: a per-tool intro shot lists ONLY that one tool; the FINAL overview lists EVERY tool.
- join_with_prev — how THIS shot cuts from the PREVIOUS shot, for editing the final film. Choose ONE:
  • "continuous" — SAME camera setup, location and host, the action flows on with NO cut (the two shots will be rendered as ONE unbroken take). Use only between consecutive "person" shots where the host keeps talking/acting in the same framing. NEVER for an "insert" shot, NEVER across a big angle/location change. VARIETY CAP — do NOT chain more than 3 shots "continuous" in a row on the exact same framing (e.g. a whole frying/mixing action told in one unbroken take reads as static, no editing rhythm). After 2-3 continuous shots of one action, cut ("cut"/"match_cut") to an insert/close-up/different angle of the SAME action, then optionally resume — this is what makes footage feel professionally edited instead of a static recording.
  • "match_cut" — a tight cut where the previous shot's last frame lines up with this shot's first frame (e.g. host lowers a bowl → cut to a top-down of that same bowl). Good for entering an insert/close-up of what was just shown.
  • "dissolve" — a soft cross-dissolve, for a gentle time/topic change (e.g. moving to a new step or a beauty shot).
  • "cut" — a clean hard cut (the default for most boundaries).
  • "j_cut" — a hard video cut, but THIS shot's audio (its dialogue/sound) STARTS a beat EARLY, over the tail of the previous shot — good when the host's next line leads into a new visual.
  • "l_cut" — a hard video cut, but the PREVIOUS shot's audio LINGERS a beat over the start of this shot — good when narration carries over into a cutaway/insert.
  The FIRST shot of the scene: its join_with_prev describes how the WHOLE scene opens from the previous scene (usually "cut"; "dissolve"/"fade-like" for a softer scene change). Default to "cut" when unsure.
- screen_direction — "left" | "right" | "neutral": the side of frame the subject FACES or the action MOVES toward. 180° RULE: keep it IDENTICAL across consecutive shots of the same setup — if the host faces left while chopping in one shot, she still faces left in the next; flipping sides between cuts disorients the viewer. Only a genuinely new setup (new scene / re-established wide) may change it. Use "neutral" for symmetric compositions, top-down flat-lays and object inserts.

LANGUAGE — CRITICAL: write motion_description, key_message, and image_subjects in the SAME language as the topic "{topic}" (Thai if the topic is Thai). English is allowed ONLY for transliterated technical terms (e.g. Medium Shot, Insert, Slow-mo, ASMR); NEVER write a whole field in English. voice_over and on_screen_text are split verbatim from the script — keep them exactly as written. EXCEPTION: prompt_img is always in English (it feeds an image model).

Production spec:
- Character (use ONLY as a short English visual descriptor in prompt_img; NEVER copy this name/bio verbatim): {character}
- Theme / setting: {theme}
- Lighting: {lighting}
- Mood: {mood}
- Material palette (surfaces, props, tableware — reflect these materials/colors): {material_palette}
- Finished dish (canonical look): {dish}
- Recipe ingredients (canonical list — copy into ingredient_refs VERBATIM when a shot shows/uses one): {ingredients}
- Tools / equipment (canonical list — copy into equipment_refs VERBATIM when a shot shows/uses one): {equipment}

Scene (topic "{topic}"):
{scene}

FINAL SELF-CHECK before returning: re-read EVERY shot's voice_over — if any ends mid-thought (a connector word or a phrase announcing the next item, e.g. "…ค่ะ ชิ้นต่อไปคือ" / "…นะคะ ถัดมาเป็น"), MOVE that trailing fragment to the START of the next shot's voice_over. No shot may end on a dangling lead-in. ALSO re-read every run of consecutive shots: if two shots in a row have a near-identical motion_description (same framing, same described action, nothing visibly progressing), rewrite one of them to show real progression — a push-in, a reframe, a new visible detail (steam rising, colour changing, a gesture change) — or change its join_with_prev to a cut into a different angle. A static talking-to-camera segment split into several shots (e.g. an outro/CTA) must NOT reuse the same motion_description sentence across those shots — vary the framing or add a beat (a hand gesture, a graphic appearing, a slight camera move) each time. FINALLY re-read every shot's on_screen_text AGAINST THAT SHOT'S OWN voice_over, in order: (a) drop any caption phrase that already appeared on an earlier shot — nothing carries forward, so a later shot must never repeat or extend an earlier shot's caption; (b) drop any remaining phrase this shot's own voice_over does not actually say, UNLESS it is the scene's step heading sitting on the shot that opens the step; (c) if the same text still sits on two shots, keep it on the one whose line speaks it and blank the other. A shot left with nothing gets "" — that is the expected result for most shots.

Return ONLY a raw JSON array, one object per shot, no markdown:
[{{"no": 1, "time": "{start} - ...", "shot_kind": "person", "shows_face": true, "join_with_prev": "cut", "screen_direction": "neutral", "motion_description": "...", "voice_over": "...", "on_screen_text": "...", "key_message": "...", "prompt_img": "...", "image_subjects": ["..."], "ingredient_refs": ["..."], "equipment_refs": ["..."]}}]
"""


# Coverage repair for an intro (ingredient/equipment) scene whose script voice_over didn't name every
# canonical item. Regenerate the WHOLE scene narration as one short line per item (in order) + a closing,
# so the storyboard's faithful split gives one non-silent shot per item. Wording varies each run.
INTRO_NARRATION_PROMPT = """\
You write the spoken narration for a Thai cooking video's {kind_th}-introduction scene, in the host's warm, friendly voice ({character}).
Rules:
- Write ONE short natural Thai sentence PER item, IN THE GIVEN ORDER. Say each item's NAME out loud EXACTLY as written (so it appears verbatim in the line), plus its quantity/spec.
- NEVER merge two different items into one sentence.
- END with ONE closing sentence that presents them all together: "และนี่คือ{kind_th}ทั้งหมดของเราค่ะ" (or a close variant containing "ทั้งหมด").
- The phrase "และนี่คือ" is RESERVED for that closing sentence ONLY — NEVER use it to introduce an individual item (use connectors like "ถัดมา", "ต่อไปเป็น", "ชิ้นต่อไปคือ", "ขาดไม่ได้คือ" instead).
- Keep each line short (speakable in ~6 seconds) and vary the phrasing naturally.
- on_screen_text: the item names + quantities, compact, separated by " · ".
Topic: "{topic}".
Items (in order): {items}
Return ONLY raw JSON, no markdown: {{"voice_over": "<all the sentences joined by spaces>", "on_screen_text": "<captions joined by ·>"}}
"""


# Semantic pass over an intro scene's per-shot voice_over lines: move any trailing lead-in fragment
# (a connector or announcer phrase that sets up the NEXT item) to the START of the next line. Covers
# every phrasing the deterministic shift's suffix gate can't enumerate; code REJECTS the output unless
# the concatenated words stay verbatim, so the model can only move boundaries, never rewrite.
INTRO_VO_REBALANCE_PROMPT = """\
You fix the SPLIT BOUNDARIES of a Thai cooking video narration that was cut into one spoken line per shot.
Each line is spoken as its OWN video clip, so every line must END on a COMPLETE thought.

Task: for each line, if it ENDS with a connector word or a lead-in phrase/clause that announces the NEXT
item (e.g. "และ", "ต่อด้วย", "ถัดมานะคะเป็น", "ชิ้นต่อไปคือ", "แล้วก็มี", "ตามด้วย", "เริ่มกันที่",
"อีกอย่างที่ขาดไม่ได้คือ", "สำหรับ...เราจะใช้" — ANY phrasing of this kind, however long), MOVE that ENTIRE
trailing fragment to the START of the NEXT line.

HARD RULES:
- Do NOT add, remove, translate or rewrite ANY word — you may ONLY move a trailing fragment across a line
  boundary. Reading all lines in order must reproduce the original text word-for-word.
- Keep the SAME number of lines, in the SAME order. A line that already ends on a complete thought stays as-is.
- Each line is spoken over the shot showing ITS item (listed below, same order) — the sentence NAMING that
  item must STAY in its own line. Move ONLY the trailing announcer fragment, NEVER the item's own sentence.
- A fragment that describes the line's OWN item (not an announcement of the next one) must NOT be moved.
- The last line has no next line — leave it unchanged.

Each line's item (same order; "" = no specific item): {items}

Lines (in order):
{lines}

Return ONLY raw JSON, no markdown: {{"lines": ["...", ...]}}
"""


# Auto-generate a voice-casting descriptor (Thai) that PINS the AI video model's narrator voice across every
# clip. Distilled from the character + VO tone into one short castable line → stored in ScriptConfig.voice_desc
# and folded into every prompt_video as {voice}.
VOICE_DESC_PROMPT = """\
You write a SINGLE voice-casting descriptor (in THAI) for the narrator of a Thai cooking video. It is used to
PIN an AI video model's generated voice so it stays IDENTICAL across every clip.

From the character and tone below, distill ONE short line (~8–20 words) naming: gender, apparent age, and the
delivery tone/energy (e.g. อบอุ่น / สดใส / สุขุม / เป็นกันเอง), the speaking pace, and an accent if implied.
Concrete and castable — e.g. "เสียงผู้หญิงไทยวัย 35 อบอุ่น สดใส เป็นกันเอง พูดชัดถ้อยชัดคำ จังหวะสบายๆ".

Character: {character_desc}
Voice-over tone: {vo_tone}

Return ONLY the descriptor line — no quotes, no label, no explanation, no markdown.
"""


# Re-generate ONLY prompt_img for an already-broken-down scene (the "Image Prompt" step), so prompts can be
# regenerated without re-running the storyboard text breakdown. The shot set/text is FIXED — write one
# prompt_img per shot. Rules mirror the prompt_img bullets in STORYBOARD_PROMPT (kept in sync by hand).
IMAGE_PROMPT_GEN_PROMPT = """\
You are a storyboard image-prompt writer. For EACH shot given below, write ONE English `prompt_img` that
renders THAT shot as a clean, photorealistic {aspect} widescreen storyboard FRAME for an image model.

DO NOT add, remove, reorder, merge or split shots — return EXACTLY one object per input shot, keyed by its `no`.
Use ONLY each shot's own fields (motion_description, voice_over, on_screen_text, shot_kind, image_subjects,
ingredient_refs) plus the production spec — do NOT invent content the shot does not describe.

prompt_img — ONE English paragraph. This still will be ANIMATED into video next (and used as the first/last
frame for video interpolation), so quality, clarity and consistency matter. Follow this structure: "A
photorealistic <shot type / framing> of <subject>, <what they do in this shot>, set in <setting + key props>.
Lit by <lighting>, <mood> atmosphere. Shot on <lens, e.g. 35mm, shallow depth of field>, {aspect} widescreen,
cinematic." Rules:
  • CHARACTER (person shots only) — FIRST set this shot's `shows_face`: TRUE when the host's face/head is in frame (she talks to camera, or a medium/wide shot shows her head); FALSE when the framing is tight on her HANDS or the food / the process / over-the-shoulder (the motion says มือ/hand — her face is NOT shown). shows_face=TRUE → refer to her with the SAME short visual descriptor in EVERY such shot (e.g. "a Thai woman chef in her 30s with a warm, friendly look"); no proper name, do NOT over-describe her face/hair. shows_face=FALSE → the SUBJECT is "the chef's hand"/"the chef's hands" (same skin tone, bracelet/watch) with the tool and the vessel ONLY — write NO appearance descriptor at all (no age, no "friendly look", no hair, no body); NEVER glue the descriptor onto the hand ("...friendly look's hand" is WRONG and makes the model render the whole host). A character reference image is supplied at generation and locks her identity, outfit and hand. Keep her outfit and the setting IDENTICAL across the scene. For shot_kind="insert", do NOT mention or describe the host AT ALL (no body, no hands).
  • COMPOSITION FOR ANIMATION — one clear focal subject, uncluttered; capture a natural moment of the shot's action with room for that motion to play out; eye-level unless the shot says otherwise. NO text, captions, numbers, logos or graphics anywhere in the frame.
  • SHOT GRAMMAR (lens + camera height) — choose the LENS and CAMERA HEIGHT by shot size for the "Shot on <lens>" slot — do NOT default every shot to 35mm: wide/establishing → 24–28mm eye level, deep focus; medium → 35mm chest height; close-up of hands/action → 50mm slightly above counter, shallow depth of field; insert/detail → 85–100mm macro (top-down flat-lay, or a low 30–45° hero angle for food); finished-dish beauty → 85mm hero angle, very shallow focus.
  • DEPTH STAGING — where natural, compose 3 layers: a soft out-of-focus FOREGROUND element at a frame edge, the MIDGROUND subject, real BACKGROUND depth — avoid flat subject-against-wall person shots.
  • OBJECTS — if the shot shows ingredients/materials/tools, NAME in English ONLY the objects THIS shot is about (from this shot's own voice_over / on_screen_text / image_subjects), with their quantities — what is SHOWN must match what is SPOKEN. Do NOT include items from other shots UNLESS this shot is an explicit overview/flat-lay of the ENTIRE set (then list the main 6–8). NEVER a vague "various cooking ingredients", and NEVER a generic filler subject — the <subject> must be the SPECIFIC named item(s) of THIS shot. NO EXTRA PROPS: the counter around the subject stays CLEAN — NEVER add unnamed props like "other small cups of ingredients", "a spoon nearby" or "various utensils"; if the source text stages other items, either NAME each one explicitly (from this shot's own refs) or DROP them.
  • TOOLS MATCH THE ACTION — include ONLY the tool(s) the shot's action actually uses (see the CATEGORY RULES tool-action pairings). Do NOT add idle props the action does not require.
  • GROUNDING — every object RESTS on a real surface; nothing floats. FOOD IS NEVER LOOSE ON THE COUNTER — every ingredient, raw item or finished dish sits IN or ON a vessel/work surface (a bowl, cup, plate, tray, board, pan/pot or rack), chosen per the CONTAINERS rule. Never "3-4 coriander roots resting on the marble counter" or "700 g of raw chicken wings on the countertop" — put them in the vessel their nature calls for. Tools and equipment MAY rest on the bare counter; food may not. Use the kitchen's EXISTING fixtures (sink, faucet, stove, cabinets, window) — do NOT invent or add new fixtures.
  • LIGHTING — soft, even, natural light; NO rim light, backlight glow, halo or white outline around the subject. The subject blends naturally into the scene.
  • DISH STATE (AUTHORITATIVE) — each shot carries `dish_state`: the dish/drink's ACTUAL physical state at THIS step, already computed across the whole recipe timeline. When `dish_state` is non-empty you MUST describe the dish/mixture EXACTLY as it says — its form, colour/opacity and vessel — and NEVER add anything it does not mention (no milk, garnish, sauce, cooking or later-stage look before the step that reaches it). Use the finished canonical look "{dish}" only as a fallback when `dish_state` is empty AND the shot clearly shows the finished served dish.
  • INGREDIENT STATE — describe EVERY ingredient's CURRENT physical state at THIS step, not the generic name (once transformed, say so — per the CATEGORY RULES state chain).
  • CONTAINERS / EQUIPMENT — name the vessel SPECIFICALLY (see the CATEGORY RULES vessel examples) and keep the SAME one across consecutive steps that use it. When a reference photo of an ingredient is supplied, its VESSEL is part of what must match: render the SAME bowl/cup/plate shown there (same material, colour, shape), not a new one — unless THIS shot moves the food somewhere else (into the pan, onto the board), which wins. The shot that INTRODUCES an ingredient or a tool (and the all-items overview) is where its container's colour and material are DECIDED — write them there. Any LATER shot reusing that same ingredient or tool names the container by TYPE only ("in its bowl") and states NO colour or material: the introduction's photo already fixes them, and restating one is how a single mixing bowl ends up white in some shots and stainless in others. (A vessel a later shot moves the food INTO — pan, pot, board, serving plate — is new to that frame and IS described in full.) EVERY ingredient visible in the frame is in a vessel — not loose on the counter — and when a shot SHOWS or measures one on its own it gets its OWN separate vessel: a small bowl for a DRY/solid ingredient, and for a LIQUID the vessel the CATEGORY RULES below name (they differ for food and for drink — follow them, not a default). Do NOT combine them UNLESS the shot is explicitly about mixing.
  • VESSEL CONTENTS — state exactly what is in a vessel at this step; never add contents the step has not reached.
  • ACTION MECHANICS — describe a hand/process action MECHANICALLY, never abstractly: which hand holds/does WHAT, WHERE the tool grips or contacts the object, and the VISIBLE result (see the CATEGORY RULES action example) — never just an abstract verb. Derive the mechanics from motion_description; if it lacks them, write the most natural mechanics for that action. Capture the action at a concrete, believable mid-stage.
  • GLOVES — IF the hand in this shot wears gloves, they are ALWAYS "white food-safe gloves". Write that exact colour every time; NEVER write bare "gloves" and NEVER any other colour — an unspecified colour drifts shot to shot. (This does not add gloves; it only fixes the colour when the shot already has them.)
  • INSERT shots (shot_kind="insert") — NO person and NO hands; frame ONLY the food/objects close-up or top-down on the counter. The <subject> must be the SPECIFIC item(s) this shot shows, NEVER a generic "all the ingredients".
  • INGREDIENT-INTRODUCTION shots — a per-ingredient intro shot shows that SINGLE ingredient ALONE in its own vessel (chosen per the CONTAINERS rule) (insert, NO person/hand/pointing — it just RESTS, the rest of the counter empty). The all-together OVERVIEW shot (ingredient_refs lists every recipe ingredient) places ALL ingredients together, also insert, no hand.
  • PROCESS shots (host physically manipulates food) — prompt_img MUST name the ingredient(s) being processed as PHYSICALLY VISIBLE inside the pot/pan/bowl or on the board, NOT just the vessel/action alone.
  • PROCESS FRAMING — frame the ACTION, not the host: either a CLOSE-UP (hands + tool + ingredient/vessel fill the frame) or a MID SHOT CROPPED BELOW THE CHIN (chest-down + hands, face NOT in frame). NEVER the host's face or full body in a process shot — her face only in talking-to-camera / opening / closing shots.
  • Use the production spec below for setting, lighting and mood.
{category_rules}

prompt_img is ALWAYS in English (it feeds an image model).

Production spec:
- Character (use ONLY as a short English visual descriptor; NEVER copy the name/bio verbatim): {character}
- Theme / setting: {theme}
- Lighting: {lighting}
- Mood: {mood}
- Material palette (surfaces, props, tableware — reflect these materials/colors): {material_palette}
- Finished dish (canonical look): {dish}
- Recipe ingredients (canonical list): {ingredients}

Shots (topic "{topic}"):
{shots}

Return ONLY a raw JSON array, one object per shot, no markdown:
[{{"no": 1, "shows_face": true, "prompt_img": "..."}}]
"""

SUBJECTS_VALIDATE_PROMPT = """\
You are a strict QA checker for image-subject extraction in a storyboard about "{topic}".

For each shot you get its id, motion_description, on_screen_text, prompt_img, and the extracted image_subjects (a list of atomic subject OBJECTS for image search). Check that image_subjects:
- contains ONLY concrete subject objects — food, ingredients, tools/equipment, finished items. NO camera jargon (Medium Shot, Top-down, Insert, Slow-mo, Pan, motion graphic).
- EXCLUDES people / characters / hands AND the scene, setting, or background (kitchen, counter, room, environment, lighting) — these are configured separately, so DROP any such entry.
- is ATOMIC: one standalone object per entry, not combined phrases.
- COVERS the concrete OBJECTS shown in this shot (read motion_description, on_screen_text AND prompt_img — details are sometimes only in one of them). An object-less shot may have an empty list.

For each shot return whether it is valid, and ALWAYS provide a corrected image_subjects list (drop people/setting/background, fix jargon & atomicity, add any missing object).

Return ONLY a raw JSON array, one object per shot, no markdown:
[{{"id": "1.1#1", "valid": true, "image_subjects": ["...", "..."]}}]

Shots:
{scenes}
"""

QUERIES_VALIDATE_PROMPT = """\
You are a strict QA checker for image-search queries (topic: "{topic}", theme: "{theme}").

For each item you get a visual element (subject) and a proposed image-search query. Check that the query:
- clearly targets the subject and would retrieve a relevant reference photo.
- has enough context to disambiguate a generic subject.
- describes a real photographable subject — NO camera jargon.

For each item return whether it is valid, and ALWAYS provide a corrected query (identical if already good, improved otherwise).

Return ONLY a raw JSON array, one object per item, no markdown:
[{{"subject": "...", "valid": true, "query": "..."}}]

Items:
{items}
"""

CANDIDATES_VALIDATE_PROMPT = """\
You are a strict relevance filter for reference images in a storyboard about "{topic}".

For each visual element you get its subject, the search query, and candidate images (index + description text from image search). Decide which candidates actually depict the subject and are usable as a reference. DROP candidates that are irrelevant, off-topic, generic logos/text, or clearly mismatched.

For each element return the indices to KEEP (may be empty if none fit).

Return ONLY a raw JSON array, one object per element, no markdown:
[{{"subject": "...", "keep": [0, 2]}}]

Elements:
{elements}
"""

CANDIDATES_VISION_PROMPT = """\
You are a strict relevance filter for reference images in a storyboard about "{topic}".

Below are several visual elements. For EACH element you are shown its subject and the
candidate images themselves, each labeled "image[i]" right before the image. LOOK at
every image and decide which ones actually depict the subject and would work as a clean
reference photo. DROP images that are off-topic, the wrong object, generic logos/text,
collages, watermarked stock, or clearly mismatched.

For each element return the indices to KEEP (may be empty if none fit).

Return ONLY a raw JSON array, one object per element, in the SAME order, no markdown:
[{{"subject": "...", "keep": [0, 2]}}]
"""

IMAGE_QUERY_PROMPT = """\
You turn short visual elements into image-search queries that each retrieve a clean, relevant reference photo.

Topic: "{topic}"
Theme / style: "{theme}"
Material palette: "{material_palette}"

For EACH element below, write ONE concise image-search query:
- Add just enough context from the topic/theme so a generic element is unambiguous (e.g. a bare "marble counter" should become a kitchen marble counter in this theme).
- For surfaces/props/containers/utensils, bias the query toward the material palette above (e.g. wood, granite, copper, stainless steel) so retrieved photos match the production's look. Do NOT force the palette onto food/ingredients themselves.
- Prefer English keywords (web image search returns more results); keep dish/proper names as-is.
- Describe a real, photographable subject — no camera jargon.

Return ONLY a raw JSON array of objects, one per element, in the same order. No markdown, no explanation.
Format: [{{"subject": "<original element>", "query": "<search query>"}}]

Elements:
{subjects}
"""

BATCH_PROMPT = """\
You are extracting detailed instructional notes from a set of videos about the topic: "{topic}" (video {n} of {total}).
This video's total runtime is {duration}.

STEP 0 — RELEVANCE CHECK (do this FIRST, before taking any notes):
Watch enough of the video to judge whether it actually TEACHES or DEMONSTRATES the exact topic "{topic}".
The video is NOT relevant if it is:
- a different dish/subject that merely has a similar name or shares keywords
- about growing, harvesting, or industrially producing an ingredient, rather than making "{topic}" itself
- a review, vlog, mukbang/eating show, gameplay, or news piece that does not teach the method
If the video is NOT relevant, reply with EXACTLY one line and nothing else:
OFF_TOPIC: <one short sentence explaining what the video is actually about>

Otherwise, watch the video carefully and extract everything a learner would need to know, including:
- Step-by-step instructions and procedures
- Exact quantities, measurements, durations, temperatures, or settings mentioned (e.g. 2 tablespoons, 180 degrees C, 500 grams, 10 minutes)
- Tools, ingredients, materials, or software required
- Tips, warnings, and common mistakes to avoid
- Any variations or alternatives mentioned

Be precise and thorough. These notes will be used to write a complete tutorial.

Also, separately, pay attention to TWO more things about the presenter, distinct from the instructional content:
1. Exactly HOW they TALK — their natural speaking style: word choice, filler words, exclamations, sentence
   rhythm, how they address the viewer.
2. HOW THEY RUN THE SHOW — their presentation flow/technique, independent of word choice: how they open before
   getting into the method, whether/how they build anticipation or ask the viewer a question to keep them
   engaged, how they transition between steps (a quick recap? straight cut? a teaser for what's next?), whether
   they periodically recap progress, how they handle a mistake or a tricky step on camera, and how they close.
   This is about STRUCTURE and PACING of the presentation, not the specific words used.

After the notes, on a NEW LINE, output EXACTLY this marker followed by a compact JSON object (no markdown, no code fence):
---PHASE_BREAKDOWN---
{{"intro": <int>, "prep": <int>, "main": <int>, "finish": <int>, "voice_samples": ["...", "...", "..."], "presentation_style": "..."}}
"intro"/"prep"/"main"/"finish" are the APPROXIMATE percentage (0-100) of this video's total runtime spent on:
intro/hook, ingredient & tool prep, the main hands-on process, and finishing/plating/wrap-up — they should
roughly sum to 100. Estimate from the pacing you actually observed; this does not need to be exact.
"voice_samples" is a list of 2-4 SHORT lines (one short sentence or exclamation each) quoted or closely
paraphrased VERBATIM from what the presenter actually says out loud, in the ORIGINAL language of the video —
picked to show their natural talking style (how they open, react, or transition), not instructional content
you already captured in the notes above.
"presentation_style" is 2-4 sentences, in English, ANALYTICALLY describing item 2 above (how they run the
show — NOT a quote, a description of the technique) — e.g. "Opens by showing the finished dish before
explaining anything. Poses a rhetorical question before the trickiest step to build anticipation. Recaps the
last 2-3 actions in one short line before starting a new step. Closes by inviting the viewer to try it and
share results." Be specific to what THIS video actually does, not generic.
Both fields: empty ("" / []) if the audio/narration isn't usable (e.g. no host talking, music-only, or a
different language than the topic). Skip this whole marker block if the video was OFF_TOPIC.
"""

SYNTHESIS_HEADER = """\
You are a professional course instructor and technical writer. \
Using the research notes below extracted from multiple videos on "{topic}", \
write a single, complete, detailed tutorial that teaches this topic from start to finish.

LANGUAGE — CRITICAL: write the ENTIRE tutorial in the SAME language as the topic "{topic}". \
If the topic is Thai, every heading, sentence and step MUST be in Thai prose — do NOT write the \
document, or whole sentences/sections, in English. English is allowed ONLY for transliterated \
technical terms, units, or proper/brand names that are normally kept in English (e.g. กรัม/gram, \
CTA, Slow-mo). Never output the whole tutorial in English.

{method_block}

NUTRITION & SAFETY — while synthesizing, actively watch for and incorporate:
- Food-safety-critical details: safe minimum cooking temperature/time for meat, poultry, seafood or eggs; \
cross-contamination avoidance (raw vs cooked surfaces/utensils); safe handling or storage notes if the source \
notes mention them. State these explicitly as part of the relevant step — do not bury or omit them.
- Nutritional reasonableness: do not inflate quantities of salt, sugar, or oil beyond what the notes actually \
call for. If the notes offer a lighter/healthier variation or substitution, include it as a brief tip.
Do NOT invent food-safety or nutrition facts the notes don't support — surface only what's actually there, or \
generally well-established basic cooking-safety practice (e.g. correct doneness for poultry).

Formatting rules:
- Write in Markdown format using headings (##, ###), numbered steps, bullet points, and **bold** for key terms.
- Use ## for major sections (Introduction, Requirements, Steps, Tips, Common Mistakes).
- Use numbered lists (1. 2. 3.) for sequential procedures.
- Use bullet points (-) for non-sequential items like ingredients, tools, or tips.
- Include all specific measurements, quantities, temperatures, and timings exactly as mentioned in the notes.
- Write in the same language as the topic "{topic}" (see the LANGUAGE rule above) — English only for transliterated terms, never the whole text.

The tutorial must feel like it was written by an expert teaching a student directly. \
It should be comprehensive, clear, and immediately actionable. \
Cover everything from preparation through completion, including tips and common mistakes.

Research notes:

"""

QUERY_VARIANTS_PROMPT = """\
Generate {n} distinct YouTube search queries for researching HOW TO MAKE / HOW TO DO: "{topic}"

Rules — every query MUST obey all of these:
- Target the EXACT same subject "{topic}" — the same dish/skill/item. Only the phrasing may vary \
(how-to, tutorial, recipe, step-by-step, secret tips, common mistakes).
- Do NOT drift to adjacent subjects: no similarly-named dishes or look-alike items, \
no growing/harvesting/factory production of raw ingredients, no reviews/vlogs/eating shows/history.
- Write the queries in the SAME language as the topic. At most ONE query may be in English, \
and only if the topic has a well-known English name.
{avoid_block}
Return ONLY a raw JSON array of {n} strings. No explanation. No markdown.
Example: ["วิธีทำ X แบบละเอียด", "X สูตรโบราณ เคล็ดลับ", "สอนทำ X ทีละขั้นตอน"]\
"""

# Appended to QUERY_VARIANTS_PROMPT at top-up time — the queries already used, to avoid repeating them.
QUERY_AVOID_BLOCK = """\
- These queries were already used; produce queries that are meaningfully DIFFERENT from all of them \
(different wording/angle, same exact subject):
{used_queries}
"""

DIRECTOR_EXTRACT_PROMPT = """\
You are a veteran video director and creative producer.

Watch this reference video carefully (title: "{title}") and reverse-engineer its
DIRECTORIAL STYLE so another team can reproduce the same professional feel in a
brand-new video on a different topic. Focus on HOW it is made, not on its specific
subject matter.

Return ONLY a raw JSON object (no markdown, no commentary). All VALUES in Thai,
concrete and prescriptive (e.g. "ตัดทุก 2–3 วินาที", "เปิดด้วยโคลสอัพเสียง ASMR") so each
is directly usable. Do NOT mention the video's specific topic/recipe — only its reusable
directorial craft. Keys:
- "summary": โทน/จุดเด่นของสไตล์วิดีโอนี้ในประโยคเดียว
- "tone_mood": บุคลิก/อารมณ์โดยรวม จังหวะความรู้สึกที่ผู้ชมควรได้รับ
- "pacing_editing": ความเร็วในการตัด ความยาวช็อตเฉลี่ย การใช้ transition / jump cut / b-roll
- "shots_framing": ประเภทช็อตที่ใช้บ่อย (close-up วัตถุดิบ, top-down, medium ฯลฯ) การเคลื่อนกล้อง
- "narrative_vo": โครงการเล่าเรื่อง น้ำเสียง/สำนวนผู้บรรยาย วิธีอธิบายขั้นตอน
- "hook_cta": วิธีเปิดเรื่องให้ติดใน 3–5 วินาทีแรก และวิธีปิดท้าย/ชวนติดตาม
- "graphics_text_music": สไตล์ตัวอักษร/motion graphic การวางข้อความบนจอ แนวดนตรี/SFX
- "do_dont": object {{"do": [...], "dont": [...]}} — แต่ละอันเป็น list ของประโยคสั้นๆ
  ("do" = สิ่งที่ควรทำเพื่อคงสไตล์นี้, "dont" = สิ่งที่จะทำให้หลุดสไตล์)

ทุก value เป็น string ภาษาไทย ยกเว้น "do_dont" ที่เป็น object ตามรูปแบบด้านล่าง
Format: {{"summary": "...", "tone_mood": "...", "pacing_editing": "...", "shots_framing": "...", "narrative_vo": "...", "hook_cta": "...", "graphics_text_music": "...", "do_dont": {{"do": ["..."], "dont": ["..."]}}}}
"""

SCRIPT_PROMPT = """\
You are a Senior Learning Experience Designer.

Analyze the following tutorial about "{topic}" and produce a complete, storyboard-ready video script.

You will return a single structured JSON object (schema enforced by the tool):
`title`, `production`, `overview`, and `parts` (each part has `description` and a
list of `scenes`). Fill EVERY field for EVERY scene — do not summarise or skip
scenes. The reference template below shows how this structured data will be laid
out for the viewer, so you know exactly what each field should contain. You do
NOT write Markdown yourself — only the structured fields.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRODUCTION SPEC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Parts: {parts}
- {duration_part_note}
- Character: {character_name} — {character_desc}
- Theme: {theme}
- Mood: {mood}
- Material palette: {material_palette}
- Lighting: {lighting}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT (copy this structure exactly)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**VIDEO SCRIPT — พร้อมพัฒนาเป็น Storyboard**

**{{topic title}}**

| ตัวละครหลัก | {{character_name}} — {{character_desc}} |
| :---- | :---- |
| **Theme** | {{theme}} |
| **อารมณ์หลัก** | {{mood}} |
| **Material Palette** | {{material_palette}} |
| **แสง** | {{lighting}} |
| **โครงสร้าง** | {parts} ตอน (Part) · ความยาวต่อตอน {duration_per_part} |

**ภาพรวมทั้ง {parts} ตอน**

* PART 1 — {{part 1 title}} : {{one-line summary of part 1 content}}
* PART 2 — {{part 2 title}} : {{one-line summary of part 2 content}}
[... repeat for each part ...]

**PART 1**

| {{Part 1 title}} — {{subtitle: ordered list of steps/topics covered, separated by ·}}     •  ความยาวรวม ≈ {{duration}} นาที |
| :---- |

| ฉาก 1.1  ·  {{Scene name}} ⏱ 0:00 – 0:35 |  |
| :---- | :---- |
| **เชื่อม** | {{Scene's narrative role / transition, e.g. เปิดตัว / Hook, เชื่อมเข้าเนื้อหา, สรุป, Teaser/CTA}} |
| **ประเภทช็อต** | {{Shot type + framing description}} |
| **แนวทางภาพ / ภาพประกอบ** | {{Detailed visual/camera direction — actionable for a camera operator}} |
| **บทพูด (Voice Over)** | {{Natural conversational Thai VO — ready to record directly}} |
| **ข้อความบนหน้าจอ** | {{On-screen text, captions, lower thirds, or motion graphic cues}} |
| **สาระสำคัญ / Key Message** | {{Core learning objective or takeaway for this scene}} |
| **ดนตรีประกอบ** | {{Music mood, tempo, genre, and any SFX notes}} |

| ฉาก 1.2  ·  {{Scene name}} ⏱ 0:35 – 1:15 |  |
| :---- | :---- |
| **เชื่อม** | ... |
| **ประเภทช็อต** | ... |
[... continue all scenes in Part 1 ...]

**PART 2**
[... same structure ...]

**PART 3**
[... same structure ...]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NARRATIVE CRAFT (this is what separates a professional script from a generic AI one — apply it
throughout, not just at the hook)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- HOOK, scene 1.1 — lead with PAYOFF, never SETUP, but still greet the viewer and name the dish —
  just not as the FIRST line. The scene needs BOTH: (1) an opening beat that is NOT a lesson
  announcement (see below), AND (2) shortly after, a brief greeting introducing the host by name
  and clearly naming what dish/topic this video teaches — viewers still need to know who's talking
  and what they're watching, this information must never be dropped. Do NOT open by announcing what
  the video will teach ("วันนี้เรามาทำ...", "มาเริ่มกันเลยค่ะ") as the very FIRST line — open on the most
  interesting single beat instead: the finished dish's most striking visual/sound, a bold specific
  claim, or a question that creates a curiosity gap. THEN, within the same scene, greet the viewer
  and name the dish plainly (e.g. "สวัสดีค่ะทุกคน ครูพี่เกศเองนะคะ วันนี้จะพาทำ <ชื่อเมนู>" — reworded so it
  doesn't sound like a copy-pasted template, but the greeting + dish name must appear). Before
  writing the opening beat, silently consider 3 different opening angles for THIS specific topic (a striking
  visual moment / a bold claim / a relatable problem it solves) and commit to whichever is sharpest
  and most specific to this dish — never settle for the first generic option. A hook that would
  work for any recipe is not sharp enough, but a hook that never says what's being made is broken.
- SENTENCE RHYTHM — vary sentence length on purpose across each scene's voice_over: short punchy
  line → one longer sentence that builds/explains → short line again. Uniform medium-length
  sentences back to back read as robotic. A short fragment used deliberately ("ได้ที่แล้วค่ะ.") can
  land harder than a full sentence — use that occasionally, not every scene.
- OPEN LOOPS — within each part, plant at least one small unresolved question or bit of tension
  early (e.g. a step that looks like it might go wrong, an ingredient whose purpose isn't obvious
  yet, "รอดูนะคะว่าทำไมต้องใส่ตอนนี้") and resolve it later in the SAME part. This keeps the viewer
  watching for the answer instead of the video being one flat stream of instructions.
- AVOID THESE AI TELLS as the OPENING LINE of scene 1.1 specifically (this is not a style
  suggestion, treat it as a hard constraint like the quantity rules below) — they are fine LATER in
  the same scene once the payoff beat has already landed, e.g. right after the greeting:
  • "วันนี้เรามาทำ...", "วันนี้ครูมี...มาฝาก", "มาเริ่มกันเลยค่ะ", "มาดูกันว่า...", or any other line whose
    JOB is to announce that a lesson/menu is about to happen, used as the FIRST line — cut straight
    to the payoff first instead (see the rewritten hook example below). Do NOT over-apply this and
    drop the greeting/dish-name entirely — see the HOOK rule above, both parts are required.
  • decorative adjectives with nothing concrete backing them — "อร่อยสุดๆ", "พิเศษสุดๆ", "สุดฟิน",
    "จัดจ้านถึงใจ", "เด็ดมากๆ" — replace with ONE specific sensory detail (a sound, a texture, a visible
    change) instead of an intensity claim.
  • "ไม่ใช่แค่...แต่ยัง..." constructions, and restating the same point in two different phrasings
    back to back.
  • generic transitions like "ต่อไปเรามาดู..." when a sharper, more specific transition is possible.
  When in doubt, cut the generic line and replace it with one concrete, dish-specific sensory detail.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WRITING DEPTH & STYLE (match this quality for every scene)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- voice_over: the FULL spoken line in {character_name}'s persona and voice (per {character_desc}) — 2–4 natural, warm, directly-recordable Thai sentences. VARY sentence length within these 2-4 sentences (short → longer → short, per NARRATIVE CRAFT's SENTENCE RHYTHM rule above) — do not write 2-4 sentences that are all the same medium length, that is the #1 tell of generic AI narration. Carry the character's goal/angle through the hook, key messages and closing. Say important quantities out loud using the EXACT numbers from the tutorial. When a scene introduces or uses ingredients, materials, or tools/equipment, the SPOKEN voice_over must NAME each item explicitly out loud with its EXACT quantity/spec from the tutorial (e.g. "เส้นหมี่แห้ง 200 กรัม น้ำตาลมะพร้าว 3 ช้อนโต๊ะ และกระทะก้นลึก") — never refer to them vaguely as "ส่วนผสมทั้งหมด" or "อุปกรณ์ที่เตรียมไว้", and never push these details to on_screen_text only. For such an introduce/prep scene you MAY exceed 4 sentences as needed to read the full list aloud.
  SPECIAL — the dedicated INGREDIENT-INTRODUCTION scene (its transition OR its name is "แนะนำวัตถุดิบ"): introduce
  the ingredients ONE AT A TIME — ONE ingredient's name + quantity per sentence, in order — then END with ONE
  closing sentence presenting ALL ingredients together (e.g. "และนี่คือวัตถุดิบทั้งหมดของเราค่ะ"). COVER EVERY
  ingredient in the recipe's ingredient list by NAME — skip NONE (even plain ones like water and ice); each item's
  exact name must appear SPOKEN in the voice_over. This lets the storyboard split into one clean single-item shot
  per ingredient + a final all-together overview shot. NEVER cram TWO DIFFERENT ingredients into one sentence or
  "one breath" (e.g. do NOT write "น้ำร้อน 180 มล. ครีมเทียม 1 ช้อนโต๊ะ" together) — each distinct ingredient is
  its OWN sentence. EXCEPTION: when the SAME ingredient is used in two amounts (e.g. hot vs cold — ครีมเทียม 1
  ช้อนโต๊ะ สำหรับเย็น / 1 ช้อนโต๊ะพูน สำหรับร้อน), keep it in ONE sentence for that ingredient and state BOTH
  amounts there — it is still one item, one shot.
  NATURAL DELIVERY (CRITICAL — real cooking-show hosts do NOT read a checklist): do NOT cycle through the same
  handful of connector words for every single item ("ถัดมา" → "ต่อไปเป็น" → "ขาดไม่ได้เลย" → "ต่อไปเป็น" → ...) —
  that repeating cadence IS an AI tell by itself, even though each individual line looks fine. Instead: some
  items get NO connector word at all (just the name, the way a person naturally continues after a beat); some
  get a short natural aside about why it matters or how it's used instead of a bare sequencing word (e.g. "อันนี้
  ต้องใช้ก้นลึกหน่อยนะคะ เพราะน้ำมันต้องท่วม" instead of "ถัดมาคือกระทะก้นลึก"); vary WHICH sequencing word you use
  on the rare item that needs one, never the same one twice in a row. Mentally read the whole scene back as
  spoken Thai — if it sounds like a list being read aloud rather than a person talking, rewrite it. The phrase
  "และนี่คือ" is RESERVED for the closing sentence ONLY — never use it to introduce an individual item.
  The ingredients are just PLACED on the counter — do NOT write the host pointing at / holding / presenting
  them with her hand.
  SPECIAL — the dedicated EQUIPMENT-INTRODUCTION scene (its transition OR its name is "แนะนำอุปกรณ์"): SAME pattern
  but for TOOLS — introduce the tools ONE AT A TIME (one short sentence per tool), COVER EVERY tool in the equipment
  list by NAME (skip none), then END with ONE closing sentence presenting ALL tools together ("และนี่คือ" reserved
  for that closing only). NEVER cram two different tools into one sentence. The same NATURAL DELIVERY rule above
  applies here just as strictly — do not let this scene read as a checklist either; vary connectors, skip them
  where a real host would, add a brief natural aside for at least a couple of tools. Each tool is just PLACED on
  the counter (no hand / no pointing).
- shot_type (ประเภทช็อต): the shot type PLUS a concrete framing line — "<type> — <subject + action + key props/position in frame>". NEVER the bare type (not just "Medium Shot to Close-up"). e.g. "Medium Shot — ครูพี่เกศยืนหลังเคาน์เตอร์หินอ่อน มือถือถ้วยหมี่กรอบสำเร็จรูปยกโชว์", "Close-up ถ้วยหมี่กรอบหมุนช้าๆ + Insert motion graphic ตัวเลข", "Top-down (Flat Lay) วัตถุดิบทั้งหมด + Pan ช้าๆ / มือชี้". Always name who/what is on screen.
- visual_direction (แนวทางภาพ / ภาพประกอบ): describe the FRAME as four distinct visual beats, NOT one action sentence — separate beats with " / " or newlines. Cover: (1) setting + lighting, (2) key props / set dressing, (3) the focal subject and its action / where it sits in frame, (4) MOTION ARC — temporal sequence describing how the scene unfolds start→middle→end (e.g. "เริ่มด้วย close-up เม็ดเกลือตกลงช้าๆ → กล้อง pull-back เผยถาด → จบที่ wide shot ไอน้ำลอยขึ้น"). e.g. "(1) ครัวโทนอุ่น แสงธรรมชาติจากหน้าต่างด้านข้าง / (2) เคาน์เตอร์หินอ่อนขาว พร็อพทองเหลืองและไม้อ่อน / (3) ถ้วยหมี่กรอบสีส้มสดบนจานไม้ เป็นจุดเด่นกลางเฟรม / (4) MOTION ARC: กล้องนิ่งในช็อตแรก → push-in ช้าๆ เน้นถ้วย → จบที่ close-up ไอร้อนลอยขึ้น". A camera operator must execute it without guessing. When the scene shows ingredients/materials or tools/equipment, a beat MUST list the SPECIFIC items by name (the same items named in this scene's voice_over / on_screen_text) — e.g. "วางปีกกลางไก่ เกลือ น้ำปลา ผงปาปริก้า แป้งทอดกรอบ เรียงในถ้วยแก้ว" — never a vague "วัตถุดิบทั้งหมด" or "อุปกรณ์ต่างๆ". When a scene simply INTRODUCES / shows ingredients or tools (not an actual hand action), frame them PLACED / arranged on the counter (a clean top-down flat-lay) — do NOT direct the host to point at them. For the ingredient-introduction scene specifically, each beat shows ONE ingredient resting ALONE on the counter with NO hand and NO pointing (just placed), and the FINAL beat places ALL ingredients together.
- on_screen_text: short caption-style text that pulls the EXACT measurements, quantities and step labels from the tutorial (grams, ml, minutes, temperatures). NOT a copy of voice_over. Keep it compact — separate multiple specs with · or •, e.g. "เส้นหมี่ 200 ก. · น้ำตาลมะพร้าว 350 ก. · เกลือ ½ ชต.".
- transition (เชื่อม): a short label naming the scene's job (e.g. เปิดตัว / Hook, แนะนำวัตถุดิบ, เคล็ดลับ, ทดสอบ, ปิดตอน + Teaser).
- scene_kind: set to "intro_ingredients" for the dedicated ingredient-introduction scene, "intro_equipment" for the dedicated equipment/tools-introduction scene, or "" (empty) for every other scene — this is a fixed machine-readable tag, independent of what language `transition`/`name` are written in.
- key_message: one concise line — the takeaway of the scene.
- music: mood + tempo + SFX; vary it scene to scene and add SFX cues that fit the action (e.g. Pop, Whoosh, ASMR, Sting, Ding).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GOLD-STANDARD EXAMPLE SCENES (DIFFERENT topic — a Thai snack course)
Match this DEPTH, VOICE and field shape. Do NOT copy this content — write about "{topic}" using the tutorial below.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[scene 1.1] transition: เปิดตัว / Hook | timecode: 0:00–0:35
  shot_type: Medium Shot — ครูพี่เกศยืนหลังเคาน์เตอร์หินอ่อน มือถือถ้วยหมี่กรอบสำเร็จรูปยกโชว์
  visual_direction: (1) ครัวโทนอุ่น แสงธรรมชาติจากหน้าต่างด้านข้างเต็มเฟรม / (2) เคาน์เตอร์หินอ่อนขาว พร็อพทองเหลืองและไม้อ่อน / (3) ถ้วยหมี่กรอบสีส้มสดบนจานไม้ เป็นจุดเด่นกลางเฟรม / (4) MOTION ARC: เริ่มด้วย medium shot ครูยืน → กล้อง push-in ช้าๆ เน้นถ้วยหมี่ → จบที่ close-up ถ้วยพร้อมไอร้อนลอยขึ้น
  voice_over: กรุบ! ฟังเสียงนี้ให้ชัดๆ นะคะ — นี่คือหมี่กรอบที่ทิ้งไว้ทั้งวันก็ยังกรอบเป๊ะ ไม่คืนตัว สวัสดีค่ะทุกคน ครูพี่เกศเองนะคะ วันนี้จะพาทำ "หมี่กรอบธัญพืช" สูตรที่หลายคนทำแล้วมันเยิ้ม เพราะพลาดขั้นตอนเดียวที่คนส่วนใหญ่มองข้าม เดี๋ยวครูจะบอกให้หมดเปลือกค่ะ!
  on_screen_text: หมี่กรอบธัญพืช — สูตรทำขายได้จริง
  key_message: เปิดด้วย payoff (เสียงกรุบ + คำอ้างที่จับต้องได้) ไม่ใช่การประกาศว่าจะสอนอะไร แล้วปลูกคำถามค้างใจ ("พลาดขั้นตอนเดียว") ที่จะคลี่คลายทีหลังในตอนเดียวกัน
  music: Acoustic อุ่น สดใส จังหวะกลางๆ เปิดอารมณ์เป็นกันเอง

[scene 1.3] transition: แนะนำวัตถุดิบ | scene_kind: intro_ingredients | timecode: 1:05–2:00
  shot_type: Insert (Top-down) วัตถุดิบวางเดี่ยวทีละชิ้นบนหินอ่อน ไม่มีมือ ไม่ชี้ → ปิดท้าย Flat Lay วางรวมทุกชิ้นในเฟรมเดียว
  visual_direction: (1) Flat lay บนหินอ่อน แสงสตูดิโอนวล / (2) แต่ละช็อตวางวัตถุดิบ "ชิ้นเดียว" ในถ้วยของตัวเอง วางเฉยๆ ไม่มีมือ ไม่ชี้ ป้ายชื่อ+ปริมาณเด้งตาม / (3) โฟกัสวัตถุดิบชิ้นนั้นกลางเฟรม เคาน์เตอร์ที่เหลือสะอาดว่าง / (4) MOTION ARC: ตัดทีละชิ้น เส้นหมี่แห้ง → น้ำตาลมะพร้าว → มะขามเปียก → ... → กลุ่มธัญพืช → จบด้วย flat-lay วางรวมทุกชิ้นพร้อมกัน
  voice_over: เริ่มจากเส้นหมี่แห้งสองร้อยกรัมค่ะ ต่อด้วยหอมแขกซอยหนึ่งร้อยห้าสิบกรัม จากนั้นน้ำตาลมะพร้าวสามร้อยห้าสิบกรัม น้ำตาลทรายหนึ่งร้อยห้าสิบกรัม มะขามเปียกสี่สิบกรัม น้ำกระเทียมดองหกสิบมิลลิลิตร ซอสพริกหนึ่งร้อยกรัม เกลือครึ่งช้อนโต๊ะ และสีผสมอาหารสีส้มแดงเล็กน้อย ปิดท้ายด้วยกลุ่มธัญพืชพรีเมียม กล้วยตาก ลูกเกด เม็ดมะม่วงหิมพานต์ อัลมอนด์ และถั่วลิสง และนี่คือวัตถุดิบทั้งหมดของเราค่ะ
  on_screen_text: เส้นหมี่แห้ง 200 ก. · หอมแขก 150 ก. · น้ำตาลมะพร้าว 350 ก. · น้ำตาลทราย 150 ก. · มะขามเปียก 40 ก. · น้ำกระเทียมดอง 60 มล. · ซอสพริก 100 ก. · เกลือ ½ ชต. · สีส้มแดง ¼ ชช.
  key_message: รู้จักวัตถุดิบครบถ้วน + ชูจุดขาย "ธัญพืชพรีเมียม"
  music: จังหวะสนุกขึ้นเล็กน้อย + SFX ป๊อปตอนป้ายเด้ง

{tools_example}[scene 1.6] transition: ทอดเส้นหมี่ + ทดสอบน้ำมัน | timecode: 3:20–4:30
  shot_type: Medium close ที่กระทะ + Slow-motion ตอนเส้นฟู
  visual_direction: (1) กระทะก้นลึกบนเตาไฟแรง น้ำมันเดือดพอง แสงร้อนจากเตา / (2) เส้นหมี่แห้งสีขาว กระชอนด้ามยาว ชามพักน้ำมัน / (3) มือครูหย่อนเส้นหมี่ลงน้ำมัน เส้นฟูขึ้นทันที → กระชอนตักออก พักในชาม / (4) MOTION ARC: เริ่ม medium close น้ำมันเดือด → Slow-mo เส้นฟู (เน้น 1–2 วินาที) → cut กลับ normal speed ตักพัก
  voice_over: ตั้งน้ำมันให้ท่วม ใช้ไฟแรงนะคะ วิธีเช็กว่าน้ำมันร้อนพอ ให้หย่อนเส้นหมี่ลงไปนิดเดียว ถ้ามันฟูขึ้นทันทีแล้วกลายเป็นสีขาว แปลว่าได้ที่แล้วค่ะ! ทอดทีละน้อยนะคะ อย่าใส่ทีละเยอะ พอเย็นแล้วบิเบาๆ ให้เป็นชิ้นพอดีคำ จะคลุกซอสง่าย
  on_screen_text: ไฟแรง • น้ำมันท่วม · ทดสอบ: หย่อนเส้น → ฟูทันที = ได้ที่ · ทอดทีละน้อย
  key_message: เทคนิคทอดเส้นหมี่ให้ฟูสวย กรอบ ไม่อมน้ำมัน
  music: จังหวะขึ้น สนุก + Whoosh ตอน Slow-mo เน้นความฟู

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. EVERY scene MUST fill all fields: scene_id, name, transition (เชื่อม), scene_kind, timecode_start, timecode_end, shot_type, visual_direction (แนวทางภาพ), voice_over (บทพูด), on_screen_text (ข้อความบนหน้าจอ), key_message (สาระสำคัญ), music (ดนตรีประกอบ).
2. Scene numbering uses PART.SCENE notation in scene_id: '1.1', '1.2' … '2.1', '2.2' … etc.
3. Each scene MUST have a running timecode (timecode_start / timecode_end) that accumulates correctly within each part (part resets to 0:00). Estimate realistically based on content length.
4. Voice Over MUST be full spoken Thai in {character_name}'s persona (per {character_desc}) — 2–4 natural, warm sentences per scene, directly recordable (a scene that introduces/preps ingredients or tools may run longer to name them all aloud with quantities). No English.
5. visual_direction (แนวทางภาพ) must be specific enough for a camera operator to execute without guessing.
5b. shot_type (ประเภทช็อต) MUST be "<shot type> — <framing: who/what + action + key props>", never the bare shot type alone.
6. Each part MUST have a `description` = short theme title + the ordered step list it covers (separated by ·) + this part's duration (WITHIN the min–max per-part window above), e.g. "เปิดครัว & เตรียมความกรอบ — เกริ่นนำ · วัตถุดิบ · ทอดเส้นหมี่ · เตรียมธัญพืช • ความยาว ≈ 6:20 นาที".
7. Fill `production` (from the spec above) and `overview` (one entry per part) before the parts. In `production`, ALSO set `dish_appearance`: a short ENGLISH description of the FINISHED dish/drink's final look inferred from the tutorial.{finished_look_rule} This is the canonical finished-dish look used to keep every storyboard frame's dish identical. ALSO set `ingredients`: the COMPLETE ingredient list from the synthesis, each entry a name + quantity in the topic's language (e.g. "มันฝรั่งขนาดกลาง 2 หัว", "เนยจืด 1 ช้อนโต๊ะ"). List every ingredient the recipe uses, none missing, no duplicates. ALSO set `equipment`: the COMPLETE list of TOOLS / EQUIPMENT the tutorial physically uses (standalone item names in the topic's language, e.g. "กระทะก้นลึก", "กระชอน", "ชามผสม") — NO consumable ingredients here; empty [] if the recipe uses no notable tools.
8. Output ONLY the structured fields — no preamble, no explanation, no extra commentary.
9. Do not omit any scene. Every part in `parts` must contain all of its scenes in order.
10. Break each part into about 6–9 scenes (~30–70 seconds each) so they add up to the part's target duration and every step in the part's description is shown by at least one scene.
11. Scene 1.1 is a Hook (see NARRATIVE CRAFT above — payoff first, not a setup announcement) AND, within that
    same scene, MUST also include a brief greeting introducing {character_name} and plainly name the
    dish/topic being taught — never as the very first line, but never omitted either. Every part after the
    first OPENS with a short Recap-and-intro scene, and EVERY part ENDS with a Teaser/CTA scene (the final
    part's last scene is the outro + call to action).
12. Use the EXACT quantities, times and temperatures from the tutorial in both voice_over and on_screen_text — never invent or round numbers.
13. Write EVERY text field in the same language as the topic "{topic}" (Thai if the topic is Thai) — transition, visual_direction, voice_over, on_screen_text, key_message, music, descriptions. English is allowed ONLY for transliterated technical terms (e.g. Medium Shot, Insert, ASMR, CTA); never write a whole field in English.
14. {tools_rule}
15. BANNED PHRASES — scan every voice_over line before returning: it must contain NONE of "วันนี้เรามาทำ",
    "วันนี้...มาฝาก", "มาเริ่มกันเลยค่ะ", "มาดูกันว่า", or a decorative-only intensity word ("สุดฟิน",
    "จัดจ้านถึงใจ", "อร่อยสุดๆ", "พิเศษสุดๆ", "เด็ดมากๆ") with no concrete detail attached. If found, rewrite
    that line per NARRATIVE CRAFT above before returning.
16. OPEN LOOP — each part must plant at least one small unresolved question or tension within its
    first 2-3 scenes (a hinted mistake, an ingredient whose purpose isn't explained yet, an
    unresolved claim) and resolve it in a LATER scene of that SAME part. A part that is purely
    linear step-by-step with no planted-and-resolved thread fails this rule.

Tutorial content:
"""

# Rule 14 variants — picked by ScriptConfig.include_tools_scene (injected as {tools_rule}).
TOOLS_RULE_BOTH = (
    'If the tutorial involves BOTH consumable ingredients/materials AND physical tools/equipment, give them '
    'TWO SEPARATE scenes — one "แนะนำวัตถุดิบ" (ingredients only) and one "แนะนำอุปกรณ์" (tools/equipment only) — '
    "each scene's voice_over, on_screen_text and visual_direction covering ONLY its own group "
    "(see gold example scenes 1.3 and 1.4). ORDER: the ingredients scene (\"แนะนำวัตถุดิบ\") MUST come BEFORE the "
    "equipment scene (\"แนะนำอุปกรณ์\"), and the two must sit ADJACENT near the start (same intro part) — introduce "
    "the ingredients first, then the tools. If the topic has no real tools, a single ingredients scene is fine."
)

TOOLS_RULE_INGREDIENTS_ONLY = (
    'Create ONLY an ingredients/materials introduction scene ("แนะนำวัตถุดิบ") — do NOT create a separate '
    '"แนะนำอุปกรณ์" / tools/equipment scene at all. A tool may be mentioned briefly inside the relevant '
    "step where it is used, but it must NEVER get its own dedicated introduction scene."
)

# Gold example's tools scene — injected as {tools_example} ONLY when include_tools_scene=True.
TOOLS_GOLD_SCENE = """\
[scene 1.4] transition: แนะนำอุปกรณ์ | scene_kind: intro_equipment | timecode: 2:00–2:30   (← ingredients & tools are SEPARATE scenes)
  shot_type: Top-down (Flat Lay) กระทะก้นลึก กระชอน ตะแกรง ชามผสม วางเรียง + มือครูพี่เกศชี้
  visual_direction: (1) Flat lay บนเคาน์เตอร์หินอ่อน แสงนวลสม่ำเสมอ / (2) กระทะก้นลึก กระชอนด้ามยาว ตะแกรงพักน้ำมัน ชามผสมใบใหญ่ ทัพพีสองอัน วางเรียงสวยงาม / (3) มือครูชี้ทีละชิ้น ป้ายชื่ออุปกรณ์เด้งตาม / (4) MOTION ARC: กล้องนิ่ง top-down → pan ช้าๆ ตามมือครูที่ชี้ → จบที่ wide shot เห็นอุปกรณ์ครบชุด
  voice_over: อุปกรณ์ที่ต้องเตรียมมีกระทะก้นลึกสำหรับทอด กระชอนกับตะแกรงไว้ตักและพักให้สะเด็ดน้ำมัน ชามผสมใบใหญ่ และทัพพีสองอันไว้คลุกซอสค่ะ เตรียมให้พร้อมก่อนเริ่มจะทำงานลื่นมากๆ
  on_screen_text: กระทะก้นลึก · กระชอน · ตะแกรง · ชามผสมใบใหญ่ · ทัพพี 2 อัน
  key_message: เตรียมอุปกรณ์ให้พร้อมก่อนลงมือ
  music: จังหวะสนุก ต่อเนื่องจากฉากวัตถุดิบ + SFX ป๊อปตอนป้ายเด้ง

"""


# Pull the structured ingredient/equipment lists out of a research synthesis, so the run's Menu can
# be seeded the moment research finishes instead of waiting for the first script generation. The
# entry format deliberately mirrors SCRIPT_PROMPT rule 7 (name + quantity in the topic's language)
# so names authored here match what the script's production block used to produce — the image step's
# fuzzy subject matching depends on that.
PRODUCTION_ITEMS_EXTRACT_PROMPT = """\
You extract the shopping and equipment lists from a cooking/drink tutorial about "{topic}".

Return ONLY a compact JSON object, no code fence, no commentary:
{{"ingredients": ["..."], "equipment": ["..."]}}

Rules:
- `ingredients`: the COMPLETE ingredient list, each entry a name + exact quantity in the topic's
  language (e.g. "มันฝรั่งขนาดกลาง 2 หัว", "เนยจืด 1 ช้อนโต๊ะ"). Use the EXACT quantities from the
  tutorial — never invent or round numbers. Every ingredient the recipe uses, none missing,
  no duplicates.
- `equipment`: the COMPLETE list of TOOLS / EQUIPMENT the tutorial physically uses (standalone item
  names in the topic's language, e.g. "กระทะก้นลึก", "กระชอน", "ชามผสม") — NO consumable
  ingredients here. Empty [] if the recipe uses no notable tools.

Tutorial:
{synthesis}"""


# Appended to the script prompt (never a {placeholder} inside SCRIPT_PROMPT — an overridden template
# would silently drop a new slot) when the run's Menu already holds its ingredient/equipment lists.
# The Menu is operator-curated: items deleted there must NOT come back from the synthesis text. The
# prompt asks; generate_script's code then overwrites production with these exact lists regardless.
SCRIPT_MENU_LOCK_BLOCK = """

CANONICAL PRODUCTION LISTS (operator-confirmed — the tutorial text below does NOT override them):
Set `production.ingredients` to EXACTLY this list — same items, same order, verbatim strings.
Add NOTHING (even if the tutorial mentions more items) and remove NOTHING:
{ingredients}
Set `production.equipment` the same way, from EXACTLY this list:
{equipment}
The ingredient/equipment introduction scenes' voice_over, on_screen_text and visual_direction must
name ONLY items from these lists.
"""


# Regenerate ONE part of an existing multi-part script (staircase per-part auto-fit). Keeps the same
# show/persona; uses the full overview + the previous part as read-only context so the new part stays
# continuous and doesn't re-cover other parts. Returns a single ScriptPart (schema enforced).
SCRIPT_REGENERATE_PART_PROMPT = """\
You are the same Senior Learning Experience Designer who wrote this multi-part video script about "{topic}".
Your job now: REGENERATE **only Part {part_number}** — rewrite its scenes so this part fits its target
duration, stays continuous with the rest of the script, and does NOT repeat what the other parts cover.

Return a SINGLE structured JSON object (schema enforced): a ScriptPart with `number`, `title`,
`description`, `duration`, and `scenes` (every field of every scene filled). Return ONLY this one part —
no other parts, no preamble, no commentary.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRODUCTION SPEC (keep identical — same show/persona)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Character: {character_name} — {character_desc}
- Theme: {theme}
- Mood: {mood}
- Material palette: {material_palette}
- Lighting: {lighting}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FULL STORY MAP — every part (READ-ONLY; this is the fixed plan — cover ONLY Part {part_number}'s own scope)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{overview_block}
{prev_block}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THIS PART ({part_number}) — TARGET
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{duration_sentence}{scenes_sentence} Each scene is assembled from ~{clip_s}s clips (one per shot).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Fill EVERY field of EVERY scene: scene_id, name, transition (เชื่อม), scene_kind ("intro_ingredients" for
   the dedicated ingredient-introduction scene, "intro_equipment" for the dedicated equipment/tools
   scene, else ""), timecode_start, timecode_end, shot_type ("<type> — <who/what + action + key props>",
   never the bare type), visual_direction (four distinct visual beats incl. a MOTION ARC), voice_over
   (full natural Thai in {character_name}'s voice, 2–4 sentences; when a scene introduces/uses ingredients
   or tools, NAME each out loud with its EXACT quantity/spec — never "ส่วนผสมทั้งหมด"), on_screen_text
   (compact specs · separated), key_message, music.
2. scene_id MUST be "{part_number}.1", "{part_number}.2", … in order. timecodes accumulate from 0:00
   WITHIN this part (a global pass re-times later — just estimate realistically).
3. CONTINUITY: Part {part_number} must flow naturally from the previous part shown above (open with a short
   recap-and-intro when it is NOT Part 1) and END with a Teaser/CTA leading into what comes next.
4. NO REPETITION: do NOT re-introduce any ingredient, tool, or step already covered by another part in the
   story map above. Cover ONLY Part {part_number}'s own scope.
5. `description` = this part's short theme title + its ordered step list (separated by ·) + " • ความยาว ≈ {{mm:ss}} นาที".
6. Use the EXACT quantities/times/temperatures from the tutorial — never invent or round. Write every field
   in the topic's language (Thai if Thai); English only for technical terms (Medium Shot, Insert, CTA).
7. {tools_rule}
8. Output ONLY the ScriptPart structured fields.

Tutorial content (source of truth for facts/quantities):
"""

FILTER_PROMPT = """\
User intent: "{topic}"

You are a strict intent-aware relevance filter for YouTube video titles.
Judge EACH title INDEPENDENTLY: does this video teach or demonstrate the EXACT subject of the user's intent?

A video is relevant ONLY if it directly serves the user's intent — not just shares keywords.
EXCLUDE, even when the title shares words with the intent:
- Look-alike subjects with similar names — e.g. intent 'จุ๋ยก๊วย (水粿)' -> exclude 'ไชเถ่าก๊วย/ขนมผักกาด (菜頭粿)' \
even though both are steamed Teochew cakes; intent 'learn Three.js' -> exclude React/Vue tutorials that merely mention 3D
- Adjacent stages of the supply chain — e.g. intent 'การชงชาเขียว (brewing green tea)' -> exclude \
tea farming, leaf picking, or factory tea production
- Non-instructional content — reviews, vlogs, mukbang/eating shows, gameplay, tier lists, news
- Intent is 'how to MAKE a survivor game' -> exclude gameplay, reviews, walkthroughs
- Intent is 'cook fried rice' -> exclude restaurant reviews or food vlogs about fried rice

RULES:
- Judge each video on its own merits. There is NO quota — keeping 3 truly relevant videos is BETTER \
than keeping {n} where some are wrong. When uncertain, set relevant to false.
- Give a short reason (one clause) for every verdict.

Titles:
{candidates}

Reply with ONLY a raw JSON array of objects, one per title, no explanation, no markdown:
[{{"index": 0, "relevant": true, "reason": "direct tutorial for the exact dish"}}, {{"index": 1, "relevant": false, "reason": "different dish (radish cake), name merely similar"}}]\
"""

SELECT_MASTER_PROMPT = """\
Topic: "{topic}"

You have WATCHED and taken notes on {n} tutorial videos about this topic. Pick the ONE video whose
method should become the MASTER/base for a new tutorial — every OTHER video will only contribute
small supplementary tips on top of this one's method, so the choice matters: pick the video whose
core steps, sequence and quantities are the most complete, clear and reliable to build directly on.

Judge each candidate on:
- Completeness — covers the full process end-to-end, no missing steps or vague quantities
- Clarity — steps are concrete and unambiguous, easy to follow in order
- Practicality — realistic for an average home cook/viewer to reproduce, not a stunt/pro-only method
- Reliability signals — precise measurements/timings/temperatures over vague gestures

Candidates:
{candidates}

Reply with ONLY a raw JSON object, no explanation, no markdown:
{{"index": 0, "reason": "one short clause — why this one's method is the strongest base"}}\
"""

# DEAD (Veo era) — no live caller. Its only reader, GeminiClient.generate_video_prompt, is itself
# uncalled: the graph's generate_video_prompts node builds OMNI_VIDEO_PROMPT_V2_PROMPT instead.
# Kept because test_prompt_rules.py still asserts against it; a rule added here does NOT reach the
# model (test_appearance_rule.py exists because exactly that mistake was made once).
VIDEO_PROMPT_PROMPT = """\
<role>
You are an elite Cinematographer AI writing VIDEO-GENERATION motion prompts (`prompt_video`) for a text/image-to-video model (Veo).
Your job: take a still frame (described by `prompt_img`) and describe how it COMES ALIVE over a {duration}-second clip.
</role>

<input_context>
Topic: "{topic}"
{director_block}{mode_block}{style_block}
</input_context>

<critical_rules>
1. STRICT ADHERENCE: The `prompt_video` is a CLOSED set. ONLY subjects, props, actions, and settings present in `prompt_img` or `motion_description` may appear. NEVER invent an extra event, gesture, ingredient, prop, or camera cut.
2. STATE LOCK: Liquid color/clarity and vessel/tool shapes stay EXACTLY as they are in the still frame (`prompt_img`). They change ONLY through explicit on-screen action (e.g., "water stays completely clear"). Without this, the model drifts states.
3. PHYSICS RULES: Gravity applies. Smooth dynamics (no instant speed jumps). Liquids pour naturally. Heat/Smoke/Fire exist ONLY if a heat source is stated (hot water steams, but never catches fire). Solid objects NEVER melt, morph, or stretch.
4. INSERT SHOTS: If `shot_kind` is "insert", NO PERSON and NO HANDS may enter the frame at ANY moment. The action is driven entirely by camera movement and natural physics. Any voice_over MUST be explicitly phrased as an off-screen narrator.
5. NO DIALOGUE HALLUCINATION: The spoken dialogue is AUDIO ONLY. A food/drink merely MENTIONED in the `voice_over` (e.g. "ชาเขียว") MUST NOT appear visually unless explicitly poured or present in `prompt_img`.
6. AUDIO LAYERS: You MUST direct 3 sound layers on EVERY shot: (1) Dialogue (`voice_over`), (2) Music (background score), (3) Ambience/SFX (natural action sounds).
7. NO SPONTANEOUS APPEARANCE: every person AND every object visible at ANY moment must either be ALREADY in frame at the first second, OR enter along a VISIBLE path — a person walks into frame; a hand reaches in to pick up or set down; the camera pans/dollies to reveal something that was already sitting there. NEVER let a person or object simply be there in a later moment when it was absent earlier — no fade-in, no popping into existence while the camera moves, no "extra person/prop present once the shot pushes in".
   THIS OVERRIDES the motion_description's wording. When `join_with_prev` is "continuous" or "match_cut", the clip OPENS on the PREVIOUS shot's ending frame (look at the shot listed just before this one in the scene). If that previous shot ended with NO person / without a given prop, then even if THIS shot's motion_description says the host "is now behind the counter" or a prop "is on the counter", you MUST open WITHOUT them and stage their ENTRANCE — rewrite "she is now behind the counter" as "she walks into frame and takes her place behind the counter", rewrite "the packet sits on the counter" as "a hand sets the packet on the counter". Only when the previous shot ALREADY had them may they be present from frame one.
   · WRONG (person): shot 1 is a wide EMPTY kitchen; shot 2 (continuous) opens with the host already standing behind the counter. RIGHT: shot 2 opens on the empty counter and the host walks into frame.
   · WRONG (object): a seasoning packet absent in the previous frame, then simply resting on the counter here. RIGHT: it was already on the counter in the previous shot, OR a hand carries/sets it into frame now.
8. GLOVES: if the hand wears gloves they are "white food-safe gloves" and STAY that colour the whole clip — never recolour or swap them mid-shot.
</critical_rules>

<step_by_step_instructions>
For EACH shot in the scene, you MUST follow these steps to construct your `prompt_video`:

STEP 1: ANALYZE THE FRAME (Contextualize)
- Read `shot_kind` ("person" vs "insert").
- Identify all elements in `prompt_img` and `dish_state` to form your STATE LOCK.
- Determine the required SCREEN DIRECTION (if any).

STEP 2: DEFINE MOTION & CAMERA
- Write a concrete camera move (e.g., slow push-in, gentle handheld, locked-off).
- Describe the subject action (cooking action, liquid settling) for {duration} seconds based on `motion_description`.
- Incorporate `join_with_prev` logic:
  · "continuous": smooth continuation of the previous clip's energy.
  · "match_cut": the previous clip settles into THIS shot's exact opening composition.
  · "dissolve": calm opening, soft transition.
  · "cut": standalone shot. Hold the tail (end on the SAME framing you opened on; do not pull back to a new scene).

STEP 3: CONSTRUCT AUDIO DESIGN
- Layer 1 (Dialogue): Finish the prompt_video with the exact spoken line.
  · "person" shot + FACE shown: 'The host, on camera, speaks the line herself — her lips move in sync with the words — in a {voice} voice: "<verbatim voice_over>".'
  · "person" shot + FACE NOT shown OR "insert" shot: 'An off-screen narrator, in a {voice} voice, says: "<verbatim voice_over>".'
  · Empty voice_over: "No dialogue."
- Layer 2 (Music): Direct the background score matching the scene's mood.
- Layer 3 (SFX): Add natural ambience sounds for the action.

STEP 4: FINALIZE
Combine all steps into ONE tight paragraph in English. End the paragraph with this exact sentence:
"Only the subjects, actions and sounds described here appear — nothing else is added, no scene change, no camera cut."
</step_by_step_instructions>

Return ONLY a raw JSON array, one object per shot. No markdown, no extra text.
IMPORTANT: Output a "reasoning" field showing your Step 1-3 thought process before the final "prompt_video".
[
  {{
    "no": 1,
    "reasoning": "Step 1: ..., Step 2: ..., Step 3: ...",
    "prompt_video": "<FINAL ONE PARAGRAPH PROMPT>"
  }}
]

Shots (one scene):
{scene}
"""

# Cinematic-craft block folded into VIDEO_PROMPT_PROMPT when video_gen.cinematic_prompt is on.
# {grade} is the fixed colour-grade token (video_gen.color_grade or a production-derived fallback)
# repeated across every shot so the whole film keeps ONE consistent look.
VIDEO_STYLE_BLOCK = """\
<cinematic_guidelines>
CINEMATIC CRAFT — build your `prompt_video` like a real shot list:
- Camera move: Name a real move (dolly/push-in, handheld, static, pan, rack focus).
- Shot size: Match the still frame (wide shot, close-up, macro).
- Lens: Carry over the LENS stated in the `prompt_img` (wide 24–28mm / close-up 50mm / macro 85–100mm).
- Lighting & grade: KEEP THIS IDENTICAL in every shot: {grade}.
- Finish the visual description of every `prompt_video` with the word "cinematic".
</cinematic_guidelines>
"""

# DEAD (Veo era) — no live caller. GeminiClient.validate_video_prompts is reached only from
# nodes._validate_video_prompts, which nothing calls: see the "NO LLM re-validation here" comment in
# nodes.generate_video_prompts for why that pass was removed.
VIDEO_PROMPT_VALIDATE_PROMPT = """\
<role>
You are a strict QA Checker AI for video-generation prompts.
Topic: "{topic}" | Voice: {voice}
</role>

<validation_rules>
For each shot, check if `prompt_video` violates ANY of these rules:
1. INSERT SHOT VIOLATION: If `shot_kind` is "insert", it MUST NOT contain ANY person or hand. The voice_over MUST be attributed to an OFF-SCREEN narrator.
2. HALLUCINATION & STRICT ADHERENCE: `prompt_video` must NOT introduce new objects, people, or events missing from `prompt_img`.
3. STATE DRIFT: Missing explicit state locks for liquids or key tools (e.g. failing to state that water remains clear).
4. PHYSICS VIOLATION: Objects melting, teleporting, or heat/fire existing without a heat source.
5. TAIL INVENTION: A "cut" shot ending by inventing a new scene or pulling back instead of holding its frame.
6. MISSING AUDIO / DIALOGUE MISMATCH: Missing the three audio layers, or forcing lip-sync on a shot where the face isn't visible.
7. MISSING ADHERENCE CLAUSE: Must end with "Only the subjects, actions and sounds described here appear...".
8. SPONTANEOUS APPEARANCE: a person or object is present in a later beat but neither in frame at the open nor shown ENTERING by a visible path (walking in, a hand reaching in, the camera revealing something already there). Especially: a person materialising once the camera pushes in on a previously empty setting, or a prop simply being there in a later moment.
</validation_rules>

<step_by_step_instructions>
For EACH shot:
STEP 1: Identify Violations. Compare `prompt_video` against the `validation_rules` above.
STEP 2: Rewrite Strategy. Determine how to fix the violations while keeping the original intent. If it's already perfect, no change is needed.
STEP 3: Finalize. Output the corrected `prompt_video`.
</step_by_step_instructions>

Return ONLY a raw JSON array, one object per shot. No markdown.
[
  {{
    "no": 1,
    "valid": false,
    "reasoning": "Step 1: Found insert shot violation..., Step 2: ...",
    "prompt_video": "<CORRECTED PROMPT>"
  }}
]

Shots:
{shots}
"""

PROMPT_IMG_CONSISTENCY_PROMPT = """\
You are a visual-consistency editor for storyboard image prompts.

The shots below all belong to the SAME scene, and every one of them is a PERSON shot — the host
(or her hand) is part of the frame. Each must depict the SAME character in the SAME setting —
only the action, framing, and props change.

Production spec (the reference to normalize AGAINST):
- Character: {character}
- Theme / setting: {theme}
- Lighting: {lighting}

For each shot, check its prompt_img:
1. Does it depict the SAME character? DERIVE one SHORT ENGLISH visual descriptor FROM the "Character"
   line of the production spec above (condense its gender, age, role and vibe — e.g. a spec of a warm
   35-year-old Thai female cooking teacher → "a Thai woman chef in her 30s with a warm, friendly look";
   a different spec yields a correspondingly different descriptor) and use that SAME wording in every
   shot. NEVER paste the spec's bio verbatim (and never in Thai). If a shot's prompt_img is missing the
   character entirely (drifted to objects-only), add the character back performing the shot's action.
   Hand-only framings stay hand-only ("the host's hand ...") — set `shows_face`=false and NEVER (re)introduce
   an appearance descriptor on them (no age, no "friendly look", no hair, no body); the hand's skin tone +
   bracelet is enough (the character reference image locks identity). A face/talking shot keeps its descriptor.
2. Does it reference the SAME setting / environment?
3. Does it keep the SAME lighting style?

If any of these drift, rewrite the prompt_img so it matches — preserving the shot's
unique action, framing, and subject objects. If it already matches, return it unchanged.

STRICT LIMITS:
- NEVER add objects, ingredients, tools, props or set dressing that the prompt_img does not already
  name — not even items from the theme/production spec. Consistency means aligning what is THERE,
  not inserting new things.
- Keep every named subject object and its quantity EXACTLY as written.

Return ONLY a raw JSON array, one object per shot, no markdown:
[{{"no": 1, "shows_face": true, "prompt_img": "..."}}]

Shots:
{shots}
"""


IMAGE_DISH_STATE_PROMPT = """You track the CURRENT physical state of the dish/drink through a cooking
tutorial, shot by shot, so every image later renders the food at its ACTUAL stage — never the finished
look too early. Topic: {topic}.

The FINISHED dish/drink's canonical look (the END state): "{dish}"
Full ingredient list: {ingredients}
{research_block}
The shots below are the ENTIRE tutorial IN ORDER (id increases with time). Reason over the WHOLE sequence:
track what has been added to / done to the dish so far, cumulatively.

For EACH shot output `dish_state` = a SHORT ENGLISH phrase describing the dish/drink's physical state AT
THAT MOMENT — its form, colour/opacity, and the vessel it's in
(e.g. "concentrated dark-green tea, no milk yet, in a glass measuring cup";
"smooth mashed potato in a stainless bowl"; "golden fried sticks on a white plate").

HARD RULES:
- State moves FORWARD ONLY. Never show a later stage early (no milk/sauce/garnish/cooking before the shot
  that actually adds or does it); never revert to an earlier stage.
- An ingredient joins the dish ONLY at the shot where it is physically ADDED/COMBINED — NOT at a shot that
  merely displays, points at, or introduces it.
- The final serving shot(s) use the canonical finished look above.
- If a shot shows NO dish/mixture at all (a lone raw-ingredient display, a bare tool, or the host talking
  with nothing in hand), output dish_state = "" for it.

Return ONLY a raw JSON array, one object per shot, no markdown:
[{{"id": 0, "dish_state": "..."}}, ...]

Shots (in order):
{shots}
"""


# ── Dynamic per-shot image flow (classify → prompt+ref) ─────────────────────────

# Drafts a scene's `layout` from its reference photo (Brand panel → ✨ button). The image model
# already receives that photo, but nothing ever told it WHERE the host belongs in it — so a stove
# shot with the hob in the foreground had the host rendered off to one side instead of behind it.
# The answer is fixed per scene, so it is written once here rather than re-derived on every shot.
# Output is shown to the operator to edit and save; it is never stored automatically.
SCENE_LAYOUT_PROMPT = """\
You are looking at the reference photo of a kitchen scene used to generate cooking-video frames.

Describe its LAYOUT so an image-generation prompt writer — who will NEVER see this photo — can
place the host and the props correctly in it. Cover exactly these four points, in this order:

1. CAMERA — where the camera sits: height (low / eye level / high / top-down), angle (straight on,
   oblique from the left or right, roughly how many degrees), and how close it is.
2. DEPTH — what occupies the FOREGROUND (nearest the camera), the MIDDLE, and the BACKGROUND. Name
   the real fixtures you can see (hob, counter, sink, cabinets, window, wall).
3. THE HOST — where a person must stand to be correctly placed in this scene.
   State it as CAMERA-RELATIVE DEPTH, never as a flat direction. Use the form "on the FAR side of
   <fixture> from the camera, with <fixture> between them and the lens" — the reader has no photo
   and cannot resolve "in front of the hob", which reads equally as standing between the camera and
   the hob (wrong) or working at it (right). Never write "in front of" about the host at all.
   Then name the positions that are WRONG here, in the same camera-relative terms (e.g. "never
   between the camera and the hob; never off to the left or right of it").
4. PROP SPACE — which surfaces are free for ingredients, bowls and tools to rest on.

Rules:
- Describe ONLY what is actually visible. Never invent a fixture, and never guess at what lies
  outside the frame.
- Plain prose, 3-6 sentences, English. No headings, no bullet list, no markdown.
- Write about the PLACE, not about a person: this photo has no host in it, and the description is
  reused for every shot in the scene.
- Do NOT describe lighting mood, colour grading or style — other parts of the system own those.

Return ONLY the description."""


IMAGE_CLASSIFY_PROMPT = """You classify how to generate the image for ONE storyboard shot,
deciding whether it can REUSE or ADAPT the immediately-previous shot's image so a scene
stays visually consistent. Topic: {topic}.

PREVIOUS shot (the frame already generated just before this one; may be empty if this is
the first shot of the scene — has_prev={has_prev}):
{prev_shot}

THIS shot to generate:
{shot}

Decide and output a JSON OBJECT with these fields:
- has_human (bool): the host/person (incl. only her hand acting) appears in THIS shot.
- framing: "flatlay" if the shot is a top-down / overhead / flat-lay of food/objects on the counter
  (even if a hand points into frame), else "eye_level" (a normal eye-level shot of the host or the kitchen).
- shot_scale: how much of the room is in frame — "closeup" (tight on hands / food / a single object; the
  kitchen room is NOT visible behind), "medium" (the host roughly waist-up with some kitchen behind), or
  "wide" (the full kitchen scene / room is clearly visible).
- same_framing_as_prev (bool): TRUE only if has_prev AND this shot keeps the SAME camera angle, scale and
  composition as the previous shot (only the action / food state advances, no cut to a new angle). FALSE when
  the angle/scale changes or there is no previous shot.
- has_ingredients (bool): food / ingredients are shown.
- is_overview (bool): TRUE if this shot shows/introduces the WHOLE SET of ingredients OR the whole set of
  tools/equipment at once (an overview flat-lay of "all the ingredients/tools we'll use"), rather than one
  specific item.
- is_process (bool): the host PHYSICALLY manipulates the food (cutting, peeling, mixing, mashing, pouring,
  frying, kneading, shaping, ...). POINTING AT / showing / presenting / introducing an ingredient is NOT a
  process — set is_process=false for those (the ingredient is just being displayed).
- shows_dish (bool): the IN-PROGRESS dish / mixture (the thing being cooked so far — e.g. the mashed-potato
  mixture in its bowl) is visible in this shot, even in the background or as a held bowl (covers recap /
  summary shots that show the dish). Lets the shot reuse the dish's previous image so the bowl stays the same.
- shows_kitchen (bool): TRUE if the kitchen room / background is VISIBLE in this frame (e.g. an eye-level shot of
  the host standing in the kitchen — TRUE even for a close-up of the host as long as the kitchen is still seen
  behind her). FALSE for a tight close-up of only hands / food / objects where the kitchen room is NOT visible,
  a top-down flat-lay, or a plain / black-background product shot. (Used so the NEXT shot can re-add the kitchen
  reference when the previous frame didn't carry the kitchen background.)
- reuse_prev (bool): TRUE only if has_prev AND the camera angle / framing / setting do NOT change and
  the ACTION is the same as the previous shot — only the spoken words (voice_over) differ. Then the exact
  previous image can be reused with no new generation.
- use_prev_as_ref (bool): TRUE only if has_prev AND reuse_prev is FALSE AND the angle/framing/setting do
  NOT change (e.g. same flat-lay, same standing pose) but some content differs (a different ingredient,
  tool, or a small action). The previous image is then a strong reference to adapt.
- image_generate_type: "reuse_prev" if reuse_prev else "use_ref_img" if use_prev_as_ref else "new_generate".
- ingredient_changed (bool) + ingredient_change (str): for use_ref_img — does the ingredient shown differ
  from the previous shot, and exactly what is it now (name + quantity, from THIS shot's voice_over/on_screen_text).
- equipment_changed (bool) + equipment_change (str): same, for tools/equipment.
- process_changed (bool) + process_change (str): same, for the action/step.
Leave the *_change strings empty when not use_ref_img or nothing changed.
{scenes_block}
Return ONLY the raw JSON object, no markdown."""


IMAGE_PROMPT_PLAN_PROMPT = """You write the image-generation prompt for ONE storyboard shot AND pick
which reference photos to use — TOGETHER so they stay consistent. Topic: {topic}.

THIS shot:
{shot}

Classification: {classify}

Available reference-photo subjects already searched for this scene (pick from these by exact name):
{keyword_pool}

Rules by image_generate_type:
- "use_ref_img": the PREVIOUS shot's image is the base. Write prompt_img as an EDIT instruction: keep the
  exact same camera, composition, surface, lighting and any unchanged objects from the reference frame, and
  change ONLY what the classification says changed (the new ingredient/equipment/process — name it with its
  quantity). Do NOT re-describe the whole kitchen. Pick ref_keywords ONLY for the NEW/changed object(s).
- "new_generate": write a full standalone prompt_img for this shot from its motion_description / voice_over /
  on_screen_text. Photorealistic, {aspect} widescreen, objects rest on a real surface (nothing floats), soft
  even lighting (no rim/halo). Show ONLY the items this shot is about — NO EXTRA PROPS: the counter around the
  subject stays CLEAN; NEVER add unnamed props like "other small cups of ingredients", "a spoon nearby" or
  "various utensils" (name each item explicitly from this shot's refs, or drop it). Pick ref_keywords for this
  shot's real objects (most important first).
  ACTION MECHANICS for hand/process shots: describe the action MECHANICALLY, never abstractly — which hand
  holds/does WHAT, WHERE the tool grips or contacts the object, and the VISIBLE result (see the CATEGORY
  RULES action example) — never just an abstract verb. Derive from motion_description;
  if it lacks the mechanics, write the most natural mechanics for that action.
  A hand-only shot (the motion says มือ/hand) writes "the chef's hand" as the SUBJECT and does NOT describe
  her body/face — NEVER glue the descriptor to 's hand ("...friendly look's hand" is WRONG); frame the hand,
  tool and vessel only.
  CRITICAL for is_process=true shots: the ingredient(s) being cooked/processed MUST appear EXPLICITLY
  in prompt_img as PHYSICALLY INSIDE or ON the cooking vessel/surface. Do NOT write a prompt that
  describes only the cooking action, vessel, or liquid without naming the ingredient inside
  (see the CATEGORY RULES WRONG/RIGHT examples).
  Every ingredient listed in ingredient_refs that this shot processes must be named as VISIBLE.
  PROCESS FRAMING for is_process shots: frame the ACTION, not the host — either a CLOSE-UP (hands +
  tool + ingredient/vessel fill the frame) or a MID SHOT CROPPED BELOW THE CHIN (chest-down + hands,
  face NOT in frame). NEVER render the host's face or full body in a process shot.

STATE & REFERENCE CONSISTENCY (apply to every prompt that shows food or a vessel):
- DISH STATE (AUTHORITATIVE): if THIS shot's `dish_state` field is non-empty, the dish/drink/mixture MUST be
  rendered EXACTLY as `dish_state` describes (its form, colour/opacity and vessel) — it already accounts for
  what has been added so far. NEVER add anything `dish_state` does not mention (no milk, garnish, sauce, or any
  later-stage/finished look before the step that reaches it).
- State each ingredient's CURRENT physical form EXPLICITLY and accurately for THIS step — NOT the generic
  name (see the CATEGORY RULES state examples). Match the state described in motion_description and `dish_state`.
- Reference images of ingredients / the dish / the bowl WILL be supplied at generation. The prompt MUST say
  the food and its container should MATCH THE REFERENCE IMAGE exactly — same shape, size, colour and the
  SAME bowl/pot. e.g. "...matching the cut potato shown in the reference image" / "...in the same stainless
  mixing bowl as the reference".
- Name the container/vessel SPECIFICALLY and keep it consistent; do not invent a different vessel.
  FOOD IS NEVER LOOSE ON THE COUNTER: every ingredient, raw item or finished dish in the frame sits IN or
  ON a vessel/work surface (a bowl, cup, plate, tray, board,
  pan/pot or rack) — never "3-4 coriander roots resting on the marble counter" or "700 g of raw
  chicken wings on the countertop". Tools and equipment MAY rest on the bare counter; food may not.
  A lone DRY/solid ingredient gets a small bowl; a lone LIQUID gets the vessel the CATEGORY RULES below name
  (food and drink differ — follow them rather than defaulting to one glass for everything).
- State a vessel's contents PRECISELY: an EMPTY vessel, the ingredient with nothing added yet, or the added
  liquid covering it — whichever the step is. Do not add contents the step has not reached.
- GLOVES: IF the hand in this shot wears gloves, they are ALWAYS "white food-safe gloves" — write that exact
  colour every time; NEVER bare "gloves" and NEVER another colour (an unspecified colour drifts shot to shot).
  This does not add gloves; it only fixes the colour when the shot already has them.
{category_rules}

HOST POSITION (any shot showing the host): always frame her BEHIND the work surface this scene shows —
counter, hob or sink — with that surface running across the FOREGROUND between the host and the camera.
When a SCENE LAYOUT block is present its host rule is the authority: repeat its placement in prompt_img
using its own fixture name. NEVER write "standing in front of" the surface or any wording that puts the
host between the surface and the camera. The host is a SOLID, separate subject — her body must NOT clip
through, merge into or intersect the surface, fixtures or props; keep a clean silhouette with clear
separation between her and the background.

prompt_img must be ENTIRELY in ENGLISH — it feeds an image model. Translate any Thai from the shot or
production fields into English (lighting, mood, item names — quantities stay numeric). Refer to the host ONLY
as a SHORT English visual descriptor (e.g. "a Thai woman chef in her 30s with a warm, friendly look") — NEVER
paste the character's name/bio text verbatim, and NO person's real name. Output a JSON OBJECT:
{{"prompt_img": "...", "ref_keywords": ["<exact subject>", ...]}}
Return ONLY the raw JSON object, no markdown."""


IMAGE_PROMPT_EXTRA_REFS_BLOCK = """

REFERENCE PHOTOS THE USER ATTACHED — these will be sent to the image model alongside your prompt,
and the user described each one:
{notes}

Describe those things in the prompt so the render actually uses them, matching each one's shape,
colour and material. Refer to them by WHAT THEY ARE ("the ceramic bowl", "the seasoning packet"),
never by an image number — the attachments are ordered after references you cannot see here, so any
number you write would point at the wrong photo."""


IMAGE_PROMPT_VESSEL_FROM_REF_BLOCK = """

VESSEL COMES FROM THE REFERENCE — this is not an introduction shot. Every ingredient and tool it
lists was already shown earlier, and those photos ride along with your prompt, so a PHOTO decides
what each container looks like, not your words.

Name a container an ingredient is ALREADY sitting in by TYPE only: "its bowl", "the mixing bowl",
"the measuring cup it came in". Do NOT give it a colour, material or finish ("a large white mixing
bowl", "a stainless steel bowl", "a clear glass cup") — the photo already fixes those, and a word
that disagrees with it makes the render pick one at random. This is how one mixing bowl turned
white in some shots and stainless in others.

The shot you are given carries its OWN earlier `prompt_img`, and that draft very likely names a
colour ("a large white mixing bowl"). It is a draft, not an instruction: drop the colour and the
material when you rewrite it, and keep only the container's type and size.

This covers ONLY containers that are already holding something. A vessel THIS shot moves the food
into — a pan, a pot, a cutting board, a serving plate, a lidded container — is new to the frame:
describe it fully, colour and material included, as usual."""


IMAGE_PROMPT_REVISE_PROMPT = """You revise the FULL prompt that was sent to an image model, because the
picture it produced was wrong. You are given that exact prompt and the complaint about its output.

Return the SAME prompt with the complaint fixed and nothing else touched.

This prompt is not prose you may rewrite freely. It is an assembled document: shot description,
scene-layout geometry, text/negative rules, and a numbered manifest describing each reference image
attached to the render. Every one of those parts is load-bearing — a rule dropped here is a rule the
image model never sees. So:

- Keep every sentence you are not fixing WORD FOR WORD. Do not reword, reorder, tidy, shorten or
  "improve" anything the complaint did not mention.
- Keep the structure: the same blocks, headings, numbering and reference manifest, in the same
  order. Never renumber or drop a reference image.
- Change only what the complaint requires — usually a phrase or a sentence, not a paragraph.
- Describe the CORRECTED state rather than forbidding the mistake. An image model renders what it
  reads, and often renders the very thing it is told to avoid: "both hands resting on the counter"
  works, "no missing hands" does not.
- If the complaint is vague or you cannot tell which part it refers to, make the smallest change you
  are confident about and leave the rest untouched.

THE PROMPT THAT WAS USED:
{full_prompt}

WHAT WAS WRONG WITH THE PICTURE IT PRODUCED:
{feedback}

Return ONLY the revised prompt text — the whole thing, ready to send as-is. No preamble, no
explanation of what you changed, no markdown fences."""


IMAGE_OUTPUT_QC_PROMPT = """You are a strict quality-control inspector for AI-GENERATED storyboard frames
(topic: "{topic}"). You are shown ONE generated image, followed by the shot's intent below.
Decide whether this render is USABLE. Judge only what matters for the final video — do NOT fail
for taste, minor styling, or slight softness.

FAIL the image if ANY of these is true:
1. PERSON IN AN INSERT — shot_kind is "insert" but any person, face, body part or hand is visible.
2. MISSING SUBJECT — one of the REQUIRED items below is not clearly visible in the frame.
3. FOREIGN OBJECT — a prominent food/ingredient/tool/prop appears that the intent does not mention
   (a garnish, extra vegetables, a second bowl of spares, extra measuring cups/spoons/bowls beyond the
   named items, an appliance that doesn't belong).
4. ARTIFACT — malformed anatomy (extra/fused fingers, warped hands/face), objects floating in mid-air,
   impossible geometry, duplicated utensils, melted/garbled textures.
5. BURNED-IN TEXT — {text_rule}
6. WRONG FRAMING — the intent calls for a top-down flat-lay but the render is eye-level (or vice versa),
   or the required camera angle is clearly wrong.
7. IMPLAUSIBLE ACTION — the pictured action cannot physically produce the described result: the tool is not
   actually engaged with the object (e.g. tongs merely holding a bag's top while liquid streams out with no
   squeeze), liquid pouring with no source, or a hand in an impossible grip.

Shot intent:
- shot_kind: {shot_kind}
- expected framing: {framing}
- REQUIRED items that must be visible: {must_show}
- description: {description}

Return ONLY a raw JSON object, no markdown:
{{"pass": true, "issues": ["<short issue>", ...], "fix_hint": "<ONE sentence of prompt guidance that would fix the render; empty if pass>"}}
"""


VIDEO_OUTPUT_QC_PROMPT = """You are a strict quality-control inspector for AI-GENERATED video clips
(topic: "{topic}"). WATCH the clip, then judge it against the shot's intent below. Judge only what
matters — do NOT fail for taste, minor styling, soft focus or compression.

FAIL the clip if ANY of these is true AT ANY MOMENT (check the ENDING especially — drift concentrates there):
1. STATE DRIFT — a liquid changes colour/clarity/contents when the intent locks it (e.g. clear water turning
   into tea, a drink appearing inside an empty bag/cup on its own, ice/milk materializing).
2. OBJECT MORPH — a vessel or tool changes its shape, size, material or turns into a different object
   (a filter bag warping into a different bag, a cup becoming a bowl, handles appearing/disappearing).
3. CONTAINMENT VIOLATION — contents pass through a vessel (powder falling THROUGH a bag into the cup),
   or something added to a vessel ends up outside it.
4. FOREIGN ELEMENT / SCENE CHANGE — a person/hand appears in a shot that must not have one, a new prop or
   subject enters, or the clip cuts/pans to a different scene or setting (especially in the final seconds).
4b. SPONTANEOUS APPEARANCE — a person or object that was NOT in frame earlier is simply present later without
   ever being shown to enter (no walking in, no hand reaching in, no camera reveal of something already there).
   This applies EVEN when the shot is allowed to contain people: a host appearing behind the counter only after
   the camera pushes in on a previously empty kitchen, or a prop popping onto the counter between beats, both FAIL.
5. BURNED-IN TEXT — readable subtitles, captions, labels or logos rendered into the frames.
6. BROKEN PHYSICS — something ignites/smokes with no heat source, floats, teleports, or duplicate limbs/objects.

Shot intent:
- shot_kind: {shot_kind} ("insert" = NO person or hand may ever be visible)
- the motion prompt this clip was rendered from (the ground truth for what may appear):
{prompt_video}

Return ONLY a raw JSON object, no markdown:
{{"pass": true, "issues": ["<short issue>", ...], "fix_hint": "<ONE sentence of prompt guidance that would prevent these defects on a re-render; empty if pass>"}}
"""

IMAGE_EDIT_PROMPT_PROMPT = """\
<role>
You write ONE clear English prompt for an image-edit model that is being handed the CURRENT image
plus, sometimes, extra reference images. Output the PROMPT ONLY — no preamble, no markdown fences.
</role>

<context>
What the base image currently is:
{original_desc}

Attached images, in the order the model receives them:
- image 1: the CURRENT image — the one being edited
{refs_manifest}

What the user wants changed:
{instruction}
</context>

<rules>
- ENGLISH, plain prose, concise. Describe the desired RESULT, not the steps.
- Name ONLY what changes. Do NOT re-describe the whole scene — the model can see the current image,
  and re-describing it invites redrawing parts that were never meant to change.
- End with "Keep everything else the same."
- If any extra images are listed above, take the new object's appearance (shape / colour / detail)
  from the right one and say so by its number — e.g. "matching the bowl in image 2". Refer to an
  attached image ONLY by "image N" using the numbers above; never invent a number that is not listed.
- Preserve the user's intent EXACTLY — do not invent a bigger or different change than they asked for.
{mode_rule}</rules>

Return ONLY the finished prompt.
"""

OMNI_EDIT_PROMPT_PROMPT = """\
You write SHORT edit instructions for Gemini Omni Flash's conversational video-edit mode
(Interactions API). GOLDEN RULE: a short, direct instruction beats a long descriptive one — Omni
edit already SEES the clip (and, on turn 1, the reference images); it does not need the scene
re-described, only told WHAT TO CHANGE.

Turn the user's rough note below into ONE short edit instruction:
- ONE or TWO sentences, imperative mood ("Remove the spoon in the last second", "Make the camera
  push in slower"). Never a paragraph, never multiple unrelated changes in one instruction.
- Name ONLY what changes. Do NOT restate what should stay the same, do NOT re-describe the scene,
  subject, lighting or camera setup — Omni already has the clip; over-describing risks it "redrawing"
  parts that were never meant to change.
- Preserve the user's intent EXACTLY — do not invent a bigger or different change than they asked for.
- The ONE exception to "do not restate what stays the same": when the note implies the rest of the
  clip should be left alone, close with the exact sentence "Keep everything else the same." That is a
  scope marker Omni responds to, not a description of the scene — do not expand on it.
- Output plain text only, no quotes, no markdown, no preamble ("Here is the instruction:" etc.).

Reference images attached to this turn (in this order):
{refs_manifest}

- If any are listed above, refer to one by its role tag <IMAGE_REF_0>, <IMAGE_REF_1>, … at the spot it
  is used — e.g. "put the hat <IMAGE_REF_0> on the person". Use the tag, never a phrase like "the
  attached image". Do not describe the reference's contents at length, and never cite a tag that is
  not listed.

User's rough note: "{instruction}"

Return ONLY the finished short instruction.
"""

# Appended in code to every edit instruction (edit_omni_video) — NOT left to the LLM above, because
# the ✨ polish button is optional: a hand-typed instruction goes straight to Omni and never passes
# through OMNI_EDIT_PROMPT_PROMPT at all. Same reasoning as the narration/duration blocks further
# down — a contract that must hold every time cannot depend on a model, or on a button being pressed.
# Ported from the POC's edit prompt, where it was the one rule v2's rewrite dropped.
# Skipped when the instruction already contains it, so the ✨ path doesn't get it twice.
OMNI_EDIT_SCOPE_BLOCK = """\
Keep everything else the same."""

OMNI_ADAPT_PROMPT_PROMPT = """\
You adapt a Veo-style structured video-generation prompt into Gemini Omni Flash's expected style
(Interactions API). The two models want the SAME creative intent described in a DIFFERENT shape:

- Veo prompt (below): a rigid multi-section brief — SCENE SPECIFICATION / CAMERA / LIGHTING /
  SUBJECT / MOTION / AUDIO / HARD CONSTRAINTS / NEGATIVE CONSTRAINTS / STORYBOARD / OUTPUT GOAL —
  written for a model that accepts an explicit negative-prompt list and section headers.
- Omni has NO negative-prompt concept and no duration knob — everything must read as ONE flowing
  natural-language paragraph (or a couple of short paragraphs), with anything that must NOT happen
  folded into a plain sentence instead of a bullet list under a "NEGATIVE CONSTRAINTS" heading.

Rewrite the Veo prompt below into Omni style, preserving EVERY concrete fact it describes — subject,
setting, camera move, lighting, state-lock details (what must stay visually unchanged), the audio
layers (dialogue/narration line, music, ambience) verbatim, and any negative constraints (translate
each into a plain sentence, e.g. "the camera does not cut or pan" instead of a "Camera: no cuts..."
bullet). Do NOT invent new content, drop any described element, or change the dialogue/narration
text. Do NOT add section headers, markdown, or bullet points — output must read as prose.

Veo-style prompt to adapt:
{veo_prompt}

Return ONLY the adapted Omni-style prompt text — no preamble, no markdown fences.
"""

# ── Omni video prompt V2 (ported from POC) — authored directly for Omni, NOT via the Veo→adapt path.
# The LLM writes the VISUAL half only (OMNI_VIDEO_PROMPT_V2_PROMPT); the on-screen-text, narration and
# duration blocks below are appended VERBATIM in code (generate_omni_prompt) so those contracts can never
# be paraphrased away. Voice direction (language/gender/pace/tone/style) is filled from the voice config.
OMNI_VOICE_BLOCK = """\
Off-screen narration:

"{voice_over}"

Speech constraints:
- The narration MUST reproduce the script EXACTLY as written.
- Do not add, omit, replace, reorder, paraphrase, improvise, or repeat any words.
- Every word must be spoken exactly once and in the original order.
- A space may be written inside a compound to mark where one word ends and the next begins (e.g. "ก็ อดใจ" means ก็ + อดใจ, never ก็อด + ใจ). Read across such a space in one continuous flow — it marks a word boundary, not a pause.
- Do not stutter, restart, self-correct, or repeat any phrase.
- Prioritize script fidelity over expressive variation.
- Do not mispronounce any word. A playful or casual tone does NOT mean speaking incorrectly — deliver every word accurately.

Voice direction:
- Language: {language}.
- Voice: a clearly {gender} voice.
- Speaking pace: approximately {vo_pace} {language} words per second.
- Tone: {tone}
- Style: {style}
- Pronunciation should be clear and easy to understand.
- Do not rush, drag, overact, or insert unnecessary pauses."""

# Thai is written without spaces between words, so Omni has to guess where each word ends — and it
# sometimes guesses wrong. "ก็อดใจไม่ไหว" is ก็ + อดใจ, but ก็อด is also a real word (the
# transliteration of "god"), so the model reads ก็อด + ใจ. Writing the boundary explicitly fixes it:
# an A/B/baseline render test (backend/test/test_omni_thai_reading.py, scored by ear — see below)
# had the spaced variant read ตาก ลม and ก็ อดใจ correctly where both other variants failed.
#
# ONE RULE PER LINE, `term = replacement`. The replacement is substituted into the narration on its
# way into OMNI_VOICE_BLOCK and nowhere else, so the DB text and the burned-in subtitles keep the
# normal unspaced spelling. Right-hand side is free-form: a space for a boundary, or a respelling
# (ปรากฏการณ์ = ปรา-กด-กาน) for a word that is simply hard to read. `#` comments, blanks ignored.
#
# Add ONLY words confirmed wrong by ear. The same test showed unambiguous words (น้ำปลาร้า,
# ทอดมันปลา) already read fine, and no automated check can score this — ก็อด-ใจ and ก็-อดใจ are
# spelled identically, so both whisper and Gemini transcribe them to the same string.
THAI_READING_RULES = """\
# term = replacement
ก็อดใจ = ก็ อดใจ
ตากลม = ตาก ลม
"""

# Appended VERBATIM by code at the end of the LLM's visual half (any LLM-written variant is stripped
# first) — the adherence guard the Veo-era template/validator used to enforce; Omni has no negative
# prompt, so this closing sentence is the only "generate nothing beyond the script" contract.
OMNI_ADHERENCE_BLOCK = """\
Only the subjects, actions and sounds described here appear — nothing else is added, no scene change, no camera cut."""

# Omni renders ~4-10s and has no duration knob, so length is a prompt-text lever only. `{target_length}`
# is filled per shot with the author's own estimate ("Target length: about 6.5 seconds.") — a broad
# range on its own made the model drift toward the maximum. Empty string when nothing estimated it,
# which leaves exactly the old generic wording.
OMNI_DURATION_BLOCK = """\
{target_length}The narration decides the clip's length — the clip lasts exactly as long as the ENTIRE line needs to be spoken to completion at a natural pace. Choose the shortest natural duration (about 4 to 10 seconds) that fits the narration — do not default to the maximum; a short line needs only a short clip, not 10 seconds.

End the clip right after the final spoken word, letting the voice settle into silence naturally — the audio must never be chopped off mid-sound. Do not add lingering holds, idle motion, freeze frames, ending animation, or a picture fade-out."""

# Same contract for a clip with no spoken line. The name must end in one of _PROMPT_SUFFIXES or
# _DEFAULTS never collects it and get_prompt() raises KeyError — which is exactly what the earlier
# OMNI_DURATION_BLOCK_NO_VO spelling did to every shot without a voice over.
OMNI_DURATION_NO_VO_BLOCK = """\
{target_length}The action decides the clip's length — it lasts exactly as long as the described motion needs to play out comfortably. Choose the shortest natural duration (about 4 to 10 seconds) that fits the action — do not default to the maximum; a simple action needs only a short clip, not 10 seconds.

End the clip right after the final action, letting the audio settle into silence naturally — it must never be chopped off mid-sound. Do not add lingering holds, idle motion, freeze frames, ending animation, or a picture fade-out."""

OMNI_ONSCREEN_BLOCK = """\
On-screen text:

"{on_screen_text}"

On-screen text direction:
- Render this text EXACTLY as written, spelled correctly and fully legible.
- Present it as a modern floating title graphic: clean contemporary sans-serif, crisp edges, sitting in the scene's real space with lighting, shadow and depth that match the shot — a believable motion graphic, not a flat pasted caption.
- Ease it in and out smoothly; it must not pop or flicker.
- PLACEMENT IS A HARD RULE: place it in the UPPER or MIDDLE area of the frame, in empty space, with a comfortable safe margin from the edges.
- NEVER place it in the BOTTOM THIRD of the frame — that band is reserved for subtitles added after generation.
- It must not cover the subject's face, hands, or the main action.
- Show this text only; do not add any other words, captions, logos or watermarks."""

OMNI_ONSCREEN_NONE_BLOCK = """\
On-screen text: NONE.

There is no on-screen text in this clip. Do not render any text, caption, title, subtitle, label, sign copy, watermark, logo or lettering anywhere in the frame at any moment."""

OMNI_VIDEO_PROMPT_V2_PROMPT = """\
<role>
You are a video-prompt author for Gemini Omni Flash (image+text-to-video, Interactions API).
Turn one storyboard shot into ONE rich, natural-language Omni video prompt for a single continuous clip.
Omni is not Veo: there is no negative-prompt field, no duration knob, no last-frame interpolation —
write everything, including things to avoid, as plain prose inside the prompt.
You write the VISUAL half only: the on-screen-text block, the narration block and the duration block
are appended by the system after your text — never write them yourself (see rules 4, 5 and 6).
</role>

<image_manifest>
These reference images are attached to this shot, in THIS order. Refer to each ONLY by its simple
`@ImageN` token (e.g. `@Image3`), inline, at the spot where you describe that subject:
{image_manifest}

TAG RULES:
- Reference each image ONLY by its @ImageN token exactly as listed (e.g. @Image2), written PLAIN —
  never wrapped in backticks, quotes or any markdown. Do NOT write <IMAGE_REF_k>, <FIRST_FRAME> or any
  angle-bracket tag yourself — the system inserts the correct Omni role tag from your @ImageN. Never
  invent an @ImageN not listed above.
- REFERENCE EVERY `@ImageN` LISTED ABOVE AT LEAST ONCE — never drop one. In particular:
  - the SETTING/background image MUST appear with an explicit STRUCTURE LOCK — e.g. "the setting stays
    exactly as `@ImageN`, same layout and structure, no new or altered fixtures" — the background must
    never drift or gain objects. The lock applies to whatever part of the setting is actually in frame:
    on a tight insert or close-up, say the background stays as `@ImageN` where visible and leave it at
    that. NEVER stage the shot in that image, open on it, cut to it, or widen/pan to bring it into view.
  - the PERSON image (when one is listed) MUST appear with an IDENTITY LOCK — same face, hair, outfit
    including fabric pattern, texture, color and print as `@ImageN`, wherever the host is on screen.
  - the OWN-FRAME image (labelled "own generated frame") is the look/composition to MATCH (subject,
    props, framing).
- The OPENING FRAME image (labelled "OPENING FRAME"), IF listed, owns frame 0 and is NON-NEGOTIABLE.
  Your VERY FIRST SENTENCE must state it outright, then describe the motion that grows OUT of it. HOW
  it is opened on depends on what that image's own manifest line says — follow it, do not mix the two:
  - "…open on this exact frame" → a LITERAL opening. E.g.: "The video begins on the exact still image
    @ImageN — frame 0 IS that image, pixel for pixel: identical framing, crop, camera angle, lens,
    lighting, colours, and every object in the same position." Do NOT re-stage, re-frame, re-light,
    re-render, zoom, or reinterpret it — no new composition of the same scene, no "similar" shot.
  - "…open on its composition, framing and camera angle … match cut" → a COMPOSITIONAL opening. Lock
    the visual STRUCTURE, not the pixels. E.g.: "The video opens matching the composition of @ImageN —
    the same framing, camera angle, shot size and placement of the subject within the frame, so the
    cut lands without a visual jump." The subject, setting and content MAY differ from that image;
    only the structure carries over. Do NOT say "pixel for pixel" or "identical" here, and do NOT
    describe that image's contents as if they were this shot's.
  If NO opening-frame image is listed, do not invent one — open naturally on the scene described.
  ONLY that image can be frame 0. The other attached images are locks applied INSIDE the shot; never
  write an opening that begins on one of them, cuts to one, or treats one as a frame of this clip.
- Do NOT write a `[# Sources]` header line yourself — the system adds the authoritative one.
</image_manifest>

<shot>
Base image prompt (the intended still frame): {prompt_img}
Motion & description: {motion}
What the OPENING FRAME already shows: {opening_frame}
Voice over (spoken line — for context only; the system appends the narration block): {voice_over}
Key message (intent, not shown): {key_message}
Aspect ratio: {aspect_ratio}
</shot>

<how_to_write>
1. SINGLE CONTINUOUS SCENE: describe ONE unbroken shot. Include the phrase "In a single continuous shot" and "No scene cuts".
2. SUBJECT + SETTING + WEAVE @ImageN: describe the scene from the base image prompt and WEAVE every `@ImageN` from the manifest above INLINE, exactly where you describe/act that subject and following its label — e.g. show the host as `@Image2` (identity lock), keep the setting per `@Image3` (structure lock, no new fixtures, applied to whatever part of it this framing actually shows), match this shot's own frame `@Image1`. NEVER omit the setting or person image — but referencing the setting means locking the background you can see, NOT staging the shot inside that image or opening on it. Use only the `@ImageN` tokens listed; never write angle-bracket tags.
3. CAMERA + MOTION: one real camera move + the action from "Motion & description". Natural physics, smooth motion, no morphing.
4. AUDIO — MUSIC AND AMBIENCE ONLY: describe fitting background music and subtle ambience. Do NOT write the spoken line, a narration paragraph, speech constraints, or any voice direction (language / pace / tone / style) — the system appends that block verbatim after your text. Never mention the voice over sentence itself.
5. NO ON-SCREEN TEXT TALK: do NOT describe any on-screen text, caption, title card, label or lettering, and do NOT say whether text is present or absent — the system appends the on-screen-text block after your text. Just describe the scene itself.
6. TIMECODES (OPTIONAL): if the shot has multiple distinct actions that should sync with narration, you MAY use relative timecodes as loose beats — e.g. [0-2s] … [2s-end]. If the shot is a single continuous action or a mood/pan, skip timecodes and just describe the flow. When used, timecodes are LOOSE beats, not hard cuts — let motion flow naturally across them; do not cram multiple actions into a narrow slot. Do NOT describe how the clip ends or fades — the system appends the duration contract after your text. The total length goes in the `target_seconds` field, never in the prose.
7. NEGATIVES AS PROSE: fold "do not add / no extra objects / no scene change / no camera cut / no captions unless specified" into sentences — never a bullet list of negatives, never a separate field.
8. SAME-LOCATION CONTINUITY: assume the ENTIRE clip happens in ONE physical place unless "Motion & description" explicitly says the scene moves. Let consecutive actions CONTINUE IN PLACE, flowing on naturally ("continues into", "then", "next the hands…"). Do NOT use the words "transition", "move to", "another area", or "travel to" anywhere.
9. PHYSICAL REALISM: every action obeys real-world physics and logic. Objects NEVER pop into existence, teleport, float without support, or vanish — if something must leave the frame, move it out naturally. Body parts must stay anatomically correct (no extra fingers, no limbs bending the wrong way). Gravity, inertia and contact always behave as in the real world — nothing defies physics.
10. NOTHING APPEARS OUT OF NOWHERE: everything on screen must be accounted for from the start. There are only two ways, and YOU choose which when you write the shot:
    (a) IT IS ALREADY THERE at [0s], established in the opening framing.
    (b) IT ARRIVES through an action the viewer can see.
    PEOPLE AND HANDS: a person is either standing there from the first second, or walks into frame; a hand is either already in shot, or reaches in from off-camera. Never let a person or a hand simply be present in a later moment when an earlier moment did not have them — and a camera move must not "discover" one: pushing in from a wide empty kitchen must not find a host the wide framing did not contain.
    OBJECTS (tools, ingredients, props): they are either sitting in the scene from the first second, or a hand visibly brings them in from off-camera. An ingredient must never materialise in a hand that was already on screen empty, and a prop must never simply be resting on the counter in a later beat when an earlier beat showed that counter bare.
    THE TWO CHOICES MUST AGREE. If the salt is not on the counter at the open, do NOT open on the chef's empty hand and have salt appear in it — open without the hand, and bring the hand in already holding the salt. Decide what the first second contains, then keep every later element consistent with that decision.
    WHEN (a) IS NOT YOURS TO CHOOSE: check the OPENING FRAME's line in the manifest. If it is "this shot's own generated frame", you composed it — the base image prompt above says what is in it, and (a) is available. But if it is "the previous clip's last frame", you did NOT compose that frame and cannot place anything into it, so (a) is closed to you: use (b) and have the host, the hand and every prop ARRIVE on screen — "a hand reaches into frame holding the measuring spoon, then pours…", never "her hand is holding the spoon" as the opening state.
</how_to_write>

<clip_length>
You also decide how long this clip runs. Omni has no duration setting — the only thing that controls
length is what the prompt says, and a vague range makes it drift long, so give a concrete number.

Estimate `target_seconds` from THIS shot's own script:
- With a voice over: how long the ENTIRE line takes to speak at a natural pace — count the line's
  NON-SPACE CHARACTERS, divide by {chars_per_second} characters per second, then add about half a
  second of settle at the end. Count characters, NOT words: Thai is written without spaces between
  words, so word-counting a Thai line reads it as two or three words and under-estimates it several
  times over. The clip must never be shorter than its narration.
- With no voice over: how long the described action needs to play out comfortably, once.
Then sanity-check against the motion: a single gesture or a slow pan is short; several distinct
beats need more. Choose the SHORTEST length that fits — do not pad toward the maximum.
Allowed range: 4 to 10 seconds. Round to one decimal.
</clip_length>

<output_format>
Return ONLY a raw JSON object, no markdown fences, no preamble:
{{"target_seconds": <number>, "prompt": "<the finished prompt text>"}}
The prompt text goes in the string as-is — do not escape it beyond normal JSON string escaping, and
do not add a preamble, headings or fences inside it.
</output_format>
"""

KITCHEN_FIXTURES_PROMPT = """Look at this kitchen reference image. List EVERY fixed fixture, built-in
structure and large appliance that is VISIBLE in it — e.g. sink, faucet, stove/hob, oven, range hood,
cabinets, drawers, countertop, window, backsplash, wall shelf, microwave, refrigerator, rice cooker, etc.
EXCLUDE loose/movable props, small utensils, food and ingredients (only things that are PART OF the kitchen).
For each fixture give a short name in BOTH Thai and English (two separate entries).
Return ONLY a raw JSON array of strings, no markdown, e.g.
["sink","อ่างล้างจาน","faucet","ก๊อกน้ำ","stove","เตา","window","หน้าต่าง"]."""


# ── Prompt override registry ─────────────────────────────────────────────────────
# Every template above is editable at runtime from the web. _DEFAULTS auto-collects the module
# constants; overrides live in outputs/prompt_overrides.json and win over defaults. get_prompt(name)
# is the single read path — call sites use it instead of the bare constant so an edit takes effect
# on the next generation with no restart.

_PROMPT_SUFFIXES = ("_PROMPT", "_RULES", "_BLOCK", "_LOOK", "_HEADER")

_DEFAULTS: dict[str, str] = {
    _k: _v
    for _k, _v in dict(globals()).items()
    if isinstance(_v, str) and _k.isupper() and _k.endswith(_PROMPT_SUFFIXES)
}

PROMPT_NAMES = frozenset(_DEFAULTS)

_OVERRIDES_FILE = ROOT_DIR / "outputs" / "prompt_overrides.json"


def _load_overrides() -> dict[str, str]:
    try:
        data = json.loads(_OVERRIDES_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception as exc:  # malformed file → ignore, never fatal
        logger.warning("prompt_overrides.json unreadable, ignoring: {}", exc)
        return {}
    # keep only known names with string values (drop stale entries silently)
    return {k: v for k, v in data.items() if k in _DEFAULTS and isinstance(v, str)}


_OVERRIDES: dict[str, str] = _load_overrides()


def get_prompt(name: str) -> str:
    """The active text for a prompt template — the override if set, else the built-in default."""
    return _OVERRIDES.get(name) or _DEFAULTS[name]


def _placeholders(text: str) -> set[str]:
    """The {field} names in a format string. Raises ValueError on unbalanced braces."""
    return {field for _, field, _, _ in string.Formatter().parse(text) if field is not None}


def validate_override(name: str, text: str) -> str | None:
    """None if `text` is a safe replacement for prompt `name`, else a human-readable reason."""
    if name not in _DEFAULTS:
        return f"unknown prompt '{name}'"
    try:
        used = _placeholders(text)
    except ValueError:
        return "unbalanced { or } — use {{ and }} for a literal brace"
    extra = used - _placeholders(_DEFAULTS[name])
    if extra:
        allowed = _placeholders(_DEFAULTS[name])
        return ("unknown placeholder(s): " + ", ".join("{" + p + "}" for p in sorted(extra))
                + ". allowed: " + (", ".join("{" + p + "}" for p in sorted(allowed)) or "(none)"))
    return None


def _group_of(name: str) -> str:
    if name.endswith(("_RULES", "_LOOK")):
        return "rules"
    if name.startswith(("STORYBOARD", "INTRO")):
        return "storyboard"
    if name.startswith(("IMAGE", "PROMPT_IMG")):
        return "image"
    if name.startswith("VIDEO"):
        return "video"
    if name.startswith("SCRIPT"):
        return "script"
    return "research"


def list_prompts() -> list[dict]:
    """All editable prompts with their active text, placeholders and override state."""
    return [
        {
            "name": name,
            "group": _group_of(name),
            "text": get_prompt(name),
            "is_overridden": name in _OVERRIDES,
            "placeholders": sorted(_placeholders(_DEFAULTS[name])),
        }
        for name in sorted(_DEFAULTS)
    ]


def set_override(name: str, text: str) -> str | None:
    """Persist an override; returns None on success or a validation-error reason (no change)."""
    err = validate_override(name, text)
    if err:
        return err
    _OVERRIDES[name] = text
    atomic_write_text(_OVERRIDES_FILE, json.dumps(_OVERRIDES, ensure_ascii=False, indent=2))
    return None


def reset_override(name: str) -> None:
    """Drop the override for `name` (back to the built-in default)."""
    if _OVERRIDES.pop(name, None) is not None:
        atomic_write_text(_OVERRIDES_FILE, json.dumps(_OVERRIDES, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    # ponytail: self-check — placeholders round-trip, and validation guards the risky cases
    for _n, _t in _DEFAULTS.items():
        assert _placeholders(get_prompt(_n)) == _placeholders(_t), _n
    assert validate_override("STORYBOARD_PROMPT", "plain text, no fields") is None
    assert validate_override("STORYBOARD_PROMPT", "{totally_made_up}") is not None
    assert validate_override("STORYBOARD_PROMPT", "unbalanced {") is not None
    assert validate_override("does_not_exist", "x") is not None
    print(f"prompts self-check OK — {len(_DEFAULTS)} templates")
