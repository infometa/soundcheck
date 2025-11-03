
from core.device import choose_device
from core.sweep import generate_sweep
from core.record import play_and_record
from core.sync import sync_and_trim
from core.ir import extract_ir
from core.metrics import RT60,C50
from core.reflections import reflections
from utils.plot import plot_ir
from utils.report import generate_report

choose_device()
sig,inv=generate_sweep()
rec=play_and_record(sig)
rec2=sync_and_trim(rec,sig)
ir=extract_ir(rec2,inv)

rt=RT60(ir); c=C50(ir)
ref=reflections(ir)
plot_ir(ir,48000,ref)
generate_report(rt,c)

print("✅ 完成")
print("RT60:",rt)
print("C50:",c)
print("📁 波形: data/plots/ir.png")
print("📄 报告: data/reports/report.pdf")
