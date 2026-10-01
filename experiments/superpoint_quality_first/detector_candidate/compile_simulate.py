"""Actual pinned nncase K210 execution; raw outputs only, no fake quantization."""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
import json,hashlib,time,traceback,resource
from pathlib import Path
from importlib.metadata import version
import numpy as np,onnx,nncase,_nncase
OUT=Path(__file__).parent;BASE=Path('/home/quyet/training_recovery/sp_ab')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(name,x):
    t=OUT/(name+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(OUT/name)
def main():
    report={'backend':'actual nncase K210 Simulator','quantization':'uint8 weights and activations, 8-bit PTQ; FP32 external I/O','nncase':version('nncase'),'_nncase':_nncase.__version__,'INT8_QUALITY':'NOT_RUN','board_latency':'NOT_MEASURED','actual_SRAM_peak':'NOT_MEASURED'}
    stage='identity';start=time.time()
    try:
        e=json.loads((OUT/'export.json').read_text());assert e['status']=='PASS';assert sha(e['checkpoint']['path'])==e['checkpoint']['sha256']
        assert sha(OUT/'pilot_canonical.onnx')==e['canonical_onnx_sha256']
        c=json.loads((OUT/'calibration.json').read_text());assert sha(OUT/'calibration_train.npy')==c['tensor_sha256']
        assert version('nncase')=='1.8.0.20220929' and _nncase.__version__=='1.8.0-55be52f'
        report['checkpoint_sha256']=e['checkpoint']['sha256'];report['calibration_sha256']=c['tensor_sha256']
        stage='checker';onnx.checker.check_model(onnx.load(str(OUT/'pilot_canonical.onnx')));report[stage]='PASS'
        options=nncase.CompileOptions();options.target='k210';options.quant_type='uint8';options.w_quant_type='uint8'
        options.dump_dir=str(OUT/'compiler_dump');options.dump_asm=True
        compiler=nncase.Compiler(options)
        stage='import';compiler.import_onnx((OUT/'pilot_canonical.onnx').read_bytes(),nncase.ImportOptions());report[stage]='PASS'
        stage='ptq';a=np.load(OUT/'calibration_train.npy');q=nncase.PTQTensorOptions();q.samples_count=len(a);q.set_tensor_data(a.tobytes());compiler.use_ptq(q);report[stage]='PASS'
        stage='compile';compiler.compile();report[stage]='PASS'
        stage='gencode';blob=compiler.gencode_tobytes();assert blob;(OUT/'pilot_trained.kmodel').write_bytes(blob);report[stage]='PASS'
        report['kmodel_sha256']=sha(OUT/'pilot_trained.kmodel');report['kmodel_bytes']=len(blob);write('deployment.json',report)
        stage='simulation';sim=nncase.Simulator();sim.load_model(blob)
        assert sim.get_input_tensor(0).dtype==np.dtype('float32'),'unknown input quantization mapping'
        for j in range(2):assert sim.get_output_tensor(j).dtype==np.dtype('float32'),'unknown output quantization mapping'
        folder=OUT/'sim_outputs';folder.mkdir(exist_ok=True)
        protocol=json.loads((BASE/'protocol.json').read_text());evidence=[]
        for i in range(len(protocol['DEV'])):
            source=BASE/'dev'/('%03d.npz'%i)
            with np.load(source) as z:pair=z['pair'].copy()
            values=[[],[]]
            for image in pair:
                sim.set_input_tensor(0,nncase.RuntimeTensor.from_numpy(np.ascontiguousarray(image[None],dtype=np.float32)));sim.run()
                for j,shape in enumerate(e['outputs']):
                    value=sim.get_output_tensor(j).to_numpy().copy();assert list(value.shape)==shape and np.isfinite(value).all();values[j].append(value[0])
            output=folder/('%03d.npz'%i);np.savez(output,logits=np.stack(values[0]),desc=np.stack(values[1]))
            evidence.append({'index':i,'input_cache_sha256':sha(source),'output_sha256':sha(output)})
            if i%7==0:print('REAL_KMODEL',i,len(protocol['DEV']),flush=True)
        report[stage]='PASS';report['INT8_QUALITY']='OUTPUTS_READY_FOR_EVALUATOR';write('simulation_manifest.json',evidence)
    except Exception as ex:
        report[stage]='BLOCKED' if stage=='simulation' else 'FAIL';report['INT8_QUALITY']='BLOCKED';report['reason']=str(ex);report['traceback']=traceback.format_exc()
    report['elapsed_s']=time.time()-start;report['rss_peak_MiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    write('deployment.json',report);print(json.dumps(report),flush=True)
    if report.get('simulation')!='PASS':raise SystemExit(1)
if __name__=='__main__':main()
