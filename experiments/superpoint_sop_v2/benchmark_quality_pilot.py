"""Laptop-only timing of trained FP32 network + local postprocessing, batch1."""
import time,json,resource
import numpy as np
import torch
import quality_pilot_v2 as q

@torch.inference_mode()
def run():
    results=[]
    for width in q.WIDTHS:
        for res in q.RES:
            name=width+'_'+res+('_cw' if res=='r2' else '')
            ck=q.ROOT/name/'best.pt'; c=torch.load(ck,weights_only=False)
            model=q.Model(q.WIDTHS[width]).eval();model.load_state_dict(c['model'])
            samples=[]
            # DEV timing only; no model/threshold adjustment. Images preloaded, decode excluded.
            for i in range(14):
                with np.load(q.ROOT/res/'dev'/f'{i:03d}.npz') as z:
                    samples.append((torch.from_numpy(z['pair'][:1].copy()),z['masks'][:1].copy()))
            for i in range(5):a,b=model(samples[0][0]);q.features(a,b,samples[0][1],res)
            network=[]; total=[]
            for repeat in range(3):
                for x,m in samples:
                    t=time.perf_counter();a,b=model(x);u=time.perf_counter();q.features(a,b,m,res);v=time.perf_counter()
                    network.append((u-t)*1000);total.append((v-t)*1000)
            results.append({'candidate':name,'checkpoint_sha256':q.sha(ck),'network_median_ms':float(np.median(network)),'network_and_postprocess_median_ms':float(np.median(total)),'network_and_postprocess_p90_ms':float(np.percentile(total,90)),'samples':len(total)})
    q.put(q.ROOT/'laptop_timing.json',{'scope':'WSL laptop CPU FP32,2threads,batch1,warmup5,repetitions3; preloaded14DEV images. Excludes image decode, retrieval, matching, PnP and data transfer. This is NOT K210 timing or end-to-end localization latency.','results':results,'code_sha256':q.sha(__file__)})
    print(json.dumps(results),flush=True)
if __name__=='__main__':run()
