import _bootstrap
import argparse,json,shutil
from spk210.onnx_export import selected_candidate,deployment_dir
from spk210.io import sha,put
p=argparse.ArgumentParser();p.add_argument('--candidate');a=p.parse_args();candidate=selected_candidate(a.candidate);dest=deployment_dir(candidate)
compile_report=json.loads((dest/'compile.json').read_text());quality=json.loads((dest/'int8_quality.json').read_text())
assert compile_report['gencode']=='PASS' and quality['status']=='MEASURED'
assert sha(dest/'model.kmodel')==compile_report['kmodel_sha256']==quality['kmodel_sha256']
folder=dest/'package';folder.mkdir(exist_ok=True)
names=['model.kmodel','export.json','onnx_parity.json','calibration.json','compile.json','int8_quality.json']
for name in names:shutil.copy2(dest/name,folder/name)
put(folder/'manifest.json',{'status':'ENGINEERING_PACKAGE_NOT_BOARD_QUALIFIED','candidate':candidate,'files':{name:sha(folder/name) for name in names},'quality_acceptance':'not automatically approved by packaging','board_latency':'NOT_MEASURED','actual_peak_SRAM':'NOT_MEASURED'})
print(folder)
