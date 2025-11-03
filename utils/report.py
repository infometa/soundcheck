
import os

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

def generate_report(rt60,c50,img="data/plots/ir.png",out="data/reports/report.pdf"):
    if not os.path.exists(img):
        raise FileNotFoundError(f"Impulse response plot not found at {img}")
    out_dir=os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir,exist_ok=True)
    c=canvas.Canvas(out,pagesize=A4)
    c.setFont("Helvetica",14)
    c.drawString(50,800,"会议室声学测试报告")
    c.setFont("Helvetica",12)
    c.drawString(50,770,f"RT60: {rt60:.3f}s")
    c.drawString(50,750,f"C50: {c50:.2f} dB")
    c.drawImage(img,50,400,500,250)
    c.save()
