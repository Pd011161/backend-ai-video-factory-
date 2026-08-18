# Gemini Omni Flash — เทคนิค Prompt / Edit / Reference (คู่มือเชิงลึก)

> **Reference ทางการ:** https://ai.google.dev/gemini-api/docs/omni
> Model: `gemini-omni-flash-preview` (preview) · Interactions API · 720p · 3–10s · 16:9 / 9:16 · ~$0.10/วิ

---

## 1. โครงสร้าง prompt ที่ดี (generation)
ใส่ให้ครบ 4 ชั้น: **subject/สภาพแวดล้อม → action → กล้อง/การเคลื่อนไหว → แสง/mood/style**
หลักการที่ doc ย้ำ: *ละเอียดเข้าไว้*
- "Be extremely detailed in your descriptions of characters and environments. Apply costume design principles to characters. Be very specific about the people, items and objects in the scene."
- "Include plenty of appropriate detail in the background elements to make the scene feel realistic and natural."
- "Consider micro-detail, expression and timing to create a very rich, detailed but entirely natural scene."

ตัวอย่างสั้น:
- `A marble rolling fast on a chain reaction style track, continuous smooth shot.`
- `A futuristic city with neon lights and flying cars, cyberpunk style`

## 2. กล้อง / cinematic
Omni ไม่ต้องใช้ศัพท์กล้องแบบ Veo — เน้น**บรรยายสิ่งที่เกิด**มากกว่าศัพท์เทคนิค แต่วลีคุมกล้องใส่ได้ เช่น `continuous smooth shot`, `Continuous, unbroken handheld shot`

## 3. Timing / ลำดับเหตุการณ์ (จุดแข็ง — คุมจังหวะได้ละเอียด)
**แบบภาษาธรรมชาติ:**
- `After 3 seconds, a woman enters the scene.`
- `At 5s the chorus starts in the background audio.`
- `Every 2s cut to a new frame.`
- `Every half a second (12 frames at 24fps) change the scene to a new location.`

**แบบ timecode:**
```
[0-3s] A person is walking
[3-6s] They stop and turn around
[6-10s] They start running
```

## 4. บังคับ "ฉากเดียวต่อเนื่อง" (สำคัญมากถ้าไม่อยากได้คัต)
ต้องใส่วลีใดวลีหนึ่ง: `In a single unbroken scene` · `In a single continuous shot` · `No scene cuts`
ตัวอย่างเต็ม:
> `Continuous, unbroken handheld shot of a fluffy tabby cat sitting on a sunny windowsill, looking out into a leafy garden. The cat's tail twitches slowly, and its ears rotate slightly toward ambient noises. Sunbeams illuminate dust motes in the air.`

## 5. เสียง / ดนตรี (Omni ใส่ audio ในตัว — สั่งได้ใน prompt)
- ดนตรี: `Include calm background music` · `The video has a high energy techno beat` · `The audio is a low tinny radio broadcast in the background, playing a song`
- sound design: `Sound design: Gentle breeze, distant bird chirps.`
- ปิดเสียงพูด: `No dialogue`

ตัวอย่างรวม: `...Sunbeams illuminate dust motes in the air. Sound design: Gentle breeze, distant bird chirps. No dialogue.`

## 6. เรนเดอร์ตัวอักษรในวิดีโอ (Omni ทำ text ให้อ่านออกได้)
- `One word on the screen at a time: 'did, you, know, that, Omni, can, do, awesome, text?' Each word appears for 1s with a different animated style. No dialogue.`
- `There is a street sign that says: 'This is an AI generation by Omni', there is a storefront that says: 'All you need AI', there's a car with the number plate: 'OMN111'`

## 7. เอา element ที่ไม่ต้องการออก (ไม่มี negative prompt — ใส่ในประโยคปกติ)
`No dialogue` · `No embellishments` · `No extra sound effects` · หรือสั่งตรงๆ `Do not do X`

---

# อ้างอิง Image Input

## 8. Role tag ของรูป (หัวใจของการอ้างอิงรูป)
- **`<FIRST_FRAME>`** = ใช้รูปเป็น**เฟรมแรกจริง**ของวิดีโอ (ภาพนี้จะปรากฏเป็นจุดเริ่ม)
  - `<FIRST_FRAME> a woman is walking`
- **`<IMAGE_REF_N>`** = ใช้รูปเป็น**อ้างอิง style/subject** (index เริ่ม 0, ไม่ปรากฏตรงๆ แต่กำหนดหน้าตา/สไตล์)
  - `in the style of <IMAGE_REF_0> a woman <IMAGE_REF_1> is walking`

> **First frame ≠ Reference**: first frame = จุดเริ่มวิดีโอตามภาพเป๊ะ · reference = ชี้นำ subject/style โดยไม่โผล่เป็นภาพนั้นตรงๆ

## 9. อ้างอิงหลายรูปพร้อมกัน (multi-image)
**แบบ tag ตรงๆ (แนะนำ)** — ผูกกับ timecode ได้:
```
[0-3s] A studio fashion sequence. Starting with woman <IMAGE_REF_0>,
she is holding <IMAGE_REF_1> [3-6s] Then we see the man <IMAGE_REF_2>
holding <IMAGE_REF_3> [6-10s] And finally another woman <IMAGE_REF_4>
who is holding <IMAGE_REF_5> while walking.
```
**แบบประกาศ source ชัดเจน** (`[# Sources ...] [# References ...]`):
```
[# Sources <FIRST_FRAME>@Image1] [# References <IMAGE_REF_0>@Image2]
a woman <IMAGE_REF_0> is walking. Use Image1 as the starting frame.
Use Image2 as a reference for the video generation.
```

## 10. Image-to-video — ใช้รูปเป็นไกด์ ไม่ให้รูปโผล่
> `turn this into realistic footage, using the drawing only as a guide for movement, do not show the drawing in the final video`

subject consistency = ส่งหลายรูปใน input array แล้วบรรยาย action (เช่น รูปแมว + รูปไหมพรม → `A cat playfully batting at a ball of yarn.`)

---

# อ้างอิง Video Input

## 11. ข้อจำกัดของ video reference (ต้องรู้)
- คลิป ref ≤ 3 วินาที: schema รับ **แต่โมเดลยัง process ไม่ถูกต้อง** (อย่าพึ่งพา)
- อ้างอิง**หลายวิดีโอ = ไม่รองรับ** (performance ตก)
- **ไม่มี** video interpolation / extension (ต่างจาก Veo)

## 12. ส่งวิดีโอเข้าไปแก้ (ผ่าน Files API)
```json
{"type": "document", "uri": video_file.uri}
```
flow: `client.files.upload()` → poll จน `state == "ACTIVE"` → ส่งเป็น `document` + prompt edit

ตัวอย่าง edit: `When the person touches the mirror, make the mirror ripple beautifully like liquid, and the person's arm turns into reflective mirror material`
> ⚠️ แก้วิดีโอที่**อัปโหลดเอง**ไม่รองรับใน EEA/สวิส/UK (แต่แก้วิดีโอที่ Omni gen เองได้)

---

# เทคนิค EDIT วิดีโอ (conversational)

## 13. กฎทองของ edit: **prompt ยิ่งสั้น ยิ่งดี**
> "Simple prompts work best for video editing. Overly descriptive prompts can lead to unintended changes."

ตัวอย่าง edit สั้นๆ ที่ได้ผล:
- `Make this video anime`
- `Put a fashionable hat on this person`
- `Change the lighting to be more dramatic`
- `Change the text on the sign to say "Omni Flash"`
- `Make the violin invisible.`

## 14. คู่ "อย่าทำ vs ทำ" (สั้นชนะยาวเสมอ)
| ❌ Avoid (ยาวเกิน) | ✅ Simplify |
|---|---|
| `...add a small black cat that runs from the right side of the screen, jumps onto his lap, and then he starts to stroke its head while looking down.` | `Add a cat that jumps onto his lap, he begins to pet it. Keep everything else the same.` |
| `Please remove the cell phone that the person is holding... and fill in the background so it looks like they are just holding their hand empty.` | `Make the phone invisible. Keep everything else the same.` |

## 15. รักษาส่วนที่ไม่อยากให้เปลี่ยน
เติมท้าย: **`Keep everything else the same`** — โมเดลจะคงทุกอย่างที่ไม่ได้พูดถึง

## 16. แก้ต่อเนื่องหลาย turn (`previous_interaction_id`)
```python
res1 = client.interactions.create(model="gemini-omni-flash-preview", input="A woman playing violin outdoors.")
res2 = client.interactions.create(model="gemini-omni-flash-preview",
    previous_interaction_id=res1.id, input="Make the violin invisible.")
```
แต่ละ turn ต่อยอดผลก่อนหน้า + คง element ที่ไม่ได้เอ่ยถึง โดยไม่ต้อง re-upload

---

# Best Practices รวม
1. **generate = ละเอียด, edit = สั้น** (หลักที่ขัดกันของสองโหมด)
2. อยากได้ช็อตเดียว → ใส่ `In a single continuous shot` / `No scene cuts`
3. คุมจังหวะด้วย timecode `[0-3s]` หรือภาษาธรรมชาติ `At 5s...`
4. เสียง/ดนตรีสั่งใน prompt ได้เลย, ปิดเสียงพูดด้วย `No dialogue`
5. negative = เขียนในประโยค (`Do not...` / `No...`) ไม่มี negative-prompt param
6. รูป: `<FIRST_FRAME>` = จุดเริ่ม, `<IMAGE_REF_N>` = อ้างอิง — หลายรูปผูก timecode ได้
7. วิดีโอ >4MB → `response_format={"type":"video","delivery":"uri"}` แล้ว poll
8. gen sync เร็วสุด: `background=false, store=false, stream=false`
9. EN รองรับเต็ม, ภาษาอื่นผลไม่แน่นอน

# API ย่อ (อ้างอิงเร็ว)
```python
from google import genai; import base64
client = genai.Client()
r = client.interactions.create(
    model="gemini-omni-flash-preview",
    input=[{"type":"image","data":b64,"mime_type":"image/png"},
           {"type":"text","text":"<FIRST_FRAME> ... prompt ..."}],
    response_format={"type":"video","aspect_ratio":"16:9"})
open("out.mp4","wb").write(base64.b64decode(r.output_video.data))
```
task ชัดเจน: `generation_config={"video_config":{"task":"image_to_video"}}` (text_to_video|image_to_video|reference_to_video|edit)

## หมายเหตุ integrate โปรเจกต์นี้
SDK google-genai 2.8.0 มี `client.interactions` + `output_video` (property) แล้ว · โค้ดมี `_is_omni()`/`_generate_video_omni()` + fallback continuous→Veo ใน `app/services/gemini_client.py` · Omni ไม่มี extension/last-frame/duration-param → pipeline ที่คุม duration ต่อช็อตต้องปรับ

## References
- https://ai.google.dev/gemini-api/docs/omni
- https://ai.google.dev/gemini-api/docs/video
- https://ai.google.dev/gemini-api/docs/pricing
