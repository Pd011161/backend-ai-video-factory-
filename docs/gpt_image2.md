# GPT Image 2 — เทคนิค Prompt / Generate / Edit (คู่มือเชิงลึกสำหรับ plan ระบบ)

> **Reference ทางการ:** https://developers.openai.com/api/docs/guides/image-generation
> Model: `gpt-image-2` (snapshot `gpt-image-2-2026-04-21`) · จุดเด่น: text rendering ดีขึ้น + รับรูป input แบบ high-fidelity อัตโนมัติ
> 2 ทางเรียก: **Images API** (`/v1/images/generations`, `/v1/images/edits`) · **Responses API** (`image_generation` tool — multi-turn)
> โปรเจกต์นี้ใช้อยู่แล้วผ่าน [app/services/openai_image.py](app/services/openai_image.py) (ดูข้อ 15)

---

## 1. เลือก API ให้ถูกงาน
| งาน | ใช้ | เหตุผล |
|---|---|---|
| gen/edit จบใน 1 request | **Images API** (`images.generate` / `images.edit`) | ตรงไปตรงมา คุม param ครบ ← โปรเจกต์นี้ใช้ตัวนี้ |
| แก้ต่อเนื่องหลายรอบ / คุยโต้ตอบ | **Responses API** (`tools=[{"type":"image_generation"}]`) | มี `previous_response_id` ต่อ turn + auto-revise prompt |

> Responses API ต้องเรียกผ่านโมเดลหลัก (เช่น `gpt-5.x`) แล้วโมเดลจะเรียก image tool ให้ · ผลรูปอยู่ใน output ชนิด `image_generation_call` (field `result`)

---

# GENERATE

## 2. Parameters (generation) — ครบ
| param | ค่า | default |
|---|---|---|
| `model` | `gpt-image-2` | required |
| `prompt` | ข้อความ (ยาวได้) | required |
| `n` | 1–10 | 1 |
| `size` | `auto` · `1024x1024` · `1536x1024` (landscape) · `1024x1536` (portrait) · หรือ custom | `auto` |
| `quality` | `auto` · `low` · `medium` · `high` | `auto` |
| `output_format` | `png` · `jpeg` · `webp` | `png` |
| `output_compression` | 0–100 (เฉพาะ jpeg/webp) | — |
| `background` | `auto` · `opaque` | `auto` |
| `moderation` | `auto` · `low` | `auto` |
| `stream` | `true`/`false` | `false` |
| `partial_images` | 0–3 (เฉพาะตอน stream) | — |

**ข้อจำกัด `size`:** ด้านยาวสุด ≤ 3840px · ทั้งสองด้านเป็น**ทวีคูณของ 16** · aspect ratio ≤ 3:1
> ⚠️ **gpt-image-2 ไม่รองรับพื้นหลังโปร่งใส** — อย่าส่ง `background:"transparent"` (ใช้ได้แค่ `auto`/`opaque`)

## 3. Prompting (generation) — ยิ่งละเอียดยิ่งดี
- บรรยายให้ครบ: **subject → style → mood/บรรยากาศ → รายละเอียดฉาก**
- ระบุสไตล์ชัด: `photorealistic`, `children's book drawing`, `ink wash`, ฯลฯ
- ตัวอย่าง doc: `A children's book drawing of a veterinarian using a stethoscope to listen to the heartbeat of a baby otter.`
- **text ในรูป**: gpt-image-2 เรนเดอร์ตัวอักษรได้ดีขึ้น แต่การวางตำแหน่งเป๊ะๆ ยังไม่ 100% → ระบุข้อความในเครื่องหมายคำพูด

---

# EDIT

## 4. Parameters (`images.edit`)
| param | รายละเอียด |
|---|---|
| `image` | ไฟล์เดียว **หรือ array หลายไฟล์** (PNG/JPEG/WebP, <50MB) |
| `mask` | (optional) PNG มี alpha channel — ชี้บริเวณที่จะแก้ |
| `prompt` | บรรยายสิ่งที่ต้องการเปลี่ยน |
| `size` / `quality` / `n` / `output_format` / `output_compression` / `moderation` | เหมือน generation |

## 5. Mask / inpainting — ทำงานยังไง
- **บริเวณโปร่งใส (alpha)= พื้นที่ที่ให้แก้** · **บริเวณทึบ = คงไว้เดิม**
- เป็น **prompt-based**: โมเดลใช้ mask เป็น "ไกด์" อาจไม่ตามรูปทรง mask เป๊ะ 100%
- ถ้าส่ง `image` หลายรูป → **mask มีผลกับรูปแรกเท่านั้น**
- `image` + `mask` ต้อง **format และมิติเท่ากัน**
- เติม alpha ให้ mask ขาว-ดำ (PIL):
```python
from PIL import Image
from io import BytesIO
mask = Image.open("mask.png").convert("L")
rgba = mask.convert("RGBA"); rgba.putalpha(mask)     # ส่วนดำ→โปร่งใส = จุดที่แก้
buf = BytesIO(); rgba.save(buf, "PNG"); mask_bytes = buf.getvalue()
```

## 6. Multi-image / reference edit (ไม่ต้อง mask)
ส่ง `image=[...]` หลายรูป → โมเดล**รวม element จากทุกรูป**เป็นภาพใหม่ (เช่น รวมสินค้าหลายชิ้นเป็นตะกร้าของขวัญ)
> โปรเจกต์นี้ใช้ท่านี้: fold role ของแต่ละรูปเข้าไปในตัว prompt เพราะ **OpenAI edit ไม่มี label ต่อรูป** (ดู `_ref_instruction` ใน openai_image.py)

## 7. `input_fidelity` — สำคัญกับ gpt-image-2
- **gpt-image-2 คง fidelity ของรูป input อัตโนมัติ (สูงเสมอ) → อย่าส่ง `input_fidelity`**
- (เฉพาะรุ่นเก่ากว่าถึงมี param นี้ให้คุมความละเอียดที่คงจากรูปต้นฉบับ) · แลกด้วย input image tokens สูงขึ้น

## 8. Prompting (edit) — สั้น + เจาะจง + คงของเดิม
- บอกเฉพาะ**สิ่งที่จะเปลี่ยน** อ้างอิงกับภาพเดิม
- เติมท้ายเพื่อคงส่วนที่เหลือ: **`Keep everything else the same`**
- โหมด "แก้จากเฟรมก่อน" (ที่โปรเจกต์ใช้): *"reproduce identically, apply ONLY the change: <delta>"* — กันโมเดลเปลี่ยน framing/พื้นหลัง

## 9. Multi-turn (Responses API) — iterate ต่อเนื่อง
```python
r1 = client.responses.create(model="gpt-5.6",
    input="Generate a gray tabby cat hugging an otter with an orange scarf",
    tools=[{"type":"image_generation"}])
r2 = client.responses.create(model="gpt-5.6",
    previous_response_id=r1.id, input="Now make it look realistic",
    tools=[{"type":"image_generation"}])
```
- แต่ละ turn ต่อยอดผลก่อนหน้า โดยไม่ต้อง re-upload
- โมเดลจะ **auto-revise prompt** ให้ → อ่านของจริงที่ใช้ได้จาก field `revised_prompt`

---

# ผลลัพธ์ / stream / ข้อจำกัด

## 10. Response format
- รูปกลับมาเป็น **base64** เสมอ (`b64_json` ใน Images API / `result` ใน Responses API) — **ไม่มี URL**
- decode: `base64.b64decode(resp.data[0].b64_json)`

## 11. Streaming (partial images)
- ตั้ง `stream=True` + `partial_images=1–3` → เห็นภาพร่างระหว่างทาง
- event: `image_generation.partial_image` (Images API) / `response.image_generation_call.partial_image` (Responses API)

## 12. Limits ที่ต้องรู้
- ไฟล์ input < **50MB** · รองรับ **PNG / JPEG / WebP**
- latency: prompt ซับซ้อนอาจนานถึง ~2 นาที
- ต้องทำ **Organization Verification** ใน developer console ก่อนใช้โมเดลตระกูล gpt-image
- จุดอ่อน: การวาง text เป๊ะ, ความ consistent ของตัวละครข้ามหลายรูป, การคุม composition แบบเป๊ะตำแหน่ง → มี regenerate เป็น safety net

## 13. Moderation
- prompt/output ถูกกรองตาม policy · ถ้าโดนบล็อก error `code = "moderation_blocked"`
- มี `moderation_details`: `moderation_stage` (`input`/`output`/`unknown`) + `categories[]`
```python
try:
    client.images.generate(model="gpt-image-2", prompt="...")
except openai.BadRequestError as e:
    if e.code != "moderation_blocked": raise
    d = (e.body or {}).get("moderation_details") or {}
    stage, cats = d.get("moderation_stage"), d.get("categories") or []
    # stage=="input" → แก้ prompt/รูปแล้วลองใหม่ ; stage=="output" → เปลี่ยน prompt แล้ว gen ใหม่
```

---

# 14. API quick reference

**Generate (Images API)**
```python
from openai import OpenAI; import base64
client = OpenAI()
r = client.images.generate(model="gpt-image-2",
    prompt="A children's book drawing of a veterinarian ...",
    size="1024x1024", quality="low", output_format="png")
open("out.png","wb").write(base64.b64decode(r.data[0].b64_json))
```

**Edit + mask**
```python
r = client.images.edit(model="gpt-image-2",
    image=open("lounge.png","rb"), mask=open("mask.png","rb"),
    prompt="A sunlit indoor lounge area with a pool containing a flamingo")
open("out.png","wb").write(base64.b64decode(r.data[0].b64_json))
```

**Multi-image edit (reference)**
```python
r = client.images.edit(model="gpt-image-2",
    image=[open("a.png","rb"), open("b.png","rb"), open("c.png","rb")],
    prompt="Photorealistic gift basket containing all items in the references, with a ribbon.")
open("out.png","wb").write(base64.b64decode(r.data[0].b64_json))
```

**Stream**
```python
stream = client.images.generate(model="gpt-image-2", prompt="...",
    stream=True, partial_images=2)
for ev in stream:
    if ev.type == "image_generation.partial_image":
        open(f"p{ev.partial_image_index}.png","wb").write(base64.b64decode(ev.b64_json))
```

**curl (generate / edit)**
```bash
curl -s https://api.openai.com/v1/images/generations \
  -H "Authorization: Bearer $OPENAI_API_KEY" -H "Content-type: application/json" \
  -d '{"model":"gpt-image-2","prompt":"...","size":"1024x1024"}' \
  | jq -r '.data[0].b64_json' | base64 --decode > out.png

curl -s https://api.openai.com/v1/images/edits \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F "model=gpt-image-2" -F "image[]=@lounge.png" -F "mask=@mask.png" \
  -F 'prompt=... a flamingo' | jq -r '.data[0].b64_json' | base64 --decode > out.png
```

---

# 15. หมายเหตุ integrate โปรเจกต์นี้
โค้ดที่มีอยู่: [app/services/openai_image.py](app/services/openai_image.py) · `generate_image()`
- **มี ref → `client.images.edit(model, image=files, prompt, size, quality)`** · **ไม่มี ref → `client.images.generate(...)`**
- **ไม่ส่ง `input_fidelity`** (gpt-image-2 คงให้อัตโนมัติ — ตรงกับข้อ 7)
- refs หลายรูป **ไม่มี label ต่อรูป** → fold บทบาทเข้า prompt ผ่าน `_ref_instruction()`; ถ้า ref แรก label ขึ้นต้น `"Previous frame"` → เข้าโหมด edit "reproduce identically, apply ONLY the change"
- `quality` default `low` (ช่วยเรื่อง edit ช้า) · ผล b64 → decode เก็บ bytes · usage เก็บผ่าน `record_usage("openai.generate_image", ...)`
- เปิด/เลือก provider ที่ `config.yaml › image_gen.provider: openai` (ดู flow เต็มใน [docs/plan_generate_image.md](docs/plan_generate_image.md))

## References
- https://developers.openai.com/api/docs/guides/image-generation
- https://developers.openai.com/api/docs/models/gpt-image-2
