"""REQUANT-1 (copy of tools/fnmx/make_eval_inputs.py, output folder changed): build vision test images and the KL text sample (first N chars of klgate code.txt + zh.txt)."""
import sys
from PIL import Image, ImageDraw, ImageFont
out = r"C:\ClaudeWorkspace\models\requant1\eval"
# image 1: shapes + English text
im = Image.new("RGB", (640, 400), "white"); d = ImageDraw.Draw(im)
d.rectangle([40, 40, 220, 220], fill=(220, 30, 30))
d.ellipse([300, 40, 480, 220], fill=(30, 60, 220))
f = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 56)
d.text((40, 270), "WHIRL 42", fill="black", font=f)
im.save(out + r"\img_shapes_en.png")
# image 2: Traditional Chinese + English sign
im = Image.new("RGB", (720, 420), (250, 245, 230)); d = ImageDraw.Draw(im)
fz = ImageFont.truetype(r"C:\Windows\Fonts\msjh.ttc", 52)
fe = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 40)
d.text((40, 40), "營業時間：週一至週五", fill="black", font=fz)
d.text((40, 120), "上午 9:30 – 下午 6:00", fill="black", font=fz)
d.text((40, 220), "Closed on Sundays", fill=(20, 20, 140), font=fe)
d.text((40, 290), "Tel: 02-2345-6789", fill=(20, 20, 140), font=fe)
im.save(out + r"\img_sign_zh.png")
# KL text: mixed code + zh from the fixed klgate corpus
src = r"C:\ClaudeWorkspace\whirl-work\klgate\corpus"
code = open(src + r"\code.txt", encoding="utf-8").read()
zh = open(src + r"\zh.txt", encoding="utf-8").read()
n_code, n_zh = int(sys.argv[1]) if len(sys.argv) > 1 else 70000, int(sys.argv[2]) if len(sys.argv) > 2 else 40000
open(out + r"\kl_mix.txt", "w", encoding="utf-8", newline="\n").write(code[:n_code] + "\n\n" + zh[:n_zh])
print("ok")
