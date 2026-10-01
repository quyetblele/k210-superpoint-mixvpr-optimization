import _bootstrap
from spk210.settings import SUPERPOINT
from spk210.io import put
status={'status':'BLOCKED','reason':'No qualified K210 firmware/transport adapter for the new SuperPoint candidate in this workspace. Package is not evidence of execution on hardware.','required':'board/firmware identity, supported kmodel runtime, verified I/O contract, real output and timing/memory capture','actual_board_latency':'NOT_MEASURED','actual_peak_SRAM':'NOT_MEASURED'}
put(SUPERPOINT/'results/board_status.json',status)
print(status['reason'])
raise SystemExit(2)
