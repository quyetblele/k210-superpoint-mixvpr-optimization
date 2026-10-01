"""One candidate training stage; atomic resume, disk teacher logits, DEV-only selection."""
import json, time, resource
import numpy as np
from .runtime import torch, ops
import torch.nn.functional as F
from .settings import ROOT, WIDTHS, SEED, STEPS
from .io import sha, put, save, guard
from .protocol import protocol
from .model import Model
from .metrics import evaluate, key

def train(name, res):
    p = protocol()
    guard()
    assert (ROOT / 'cache_done.json').exists()
    dest = ROOT / (name + '_' + res + ('_cw' if res == 'r2' else ''))
    dest.mkdir(exist_ok=True)
    if (dest / 'report.json').exists():
        return
    init = ROOT / f'{name}_init.pt'
    ph = sha(ROOT / 'protocol.json')
    torch.manual_seed(SEED)
    model = Model(WIDTHS[name])
    model.load_state_dict(torch.load(init, weights_only=True))
    opt = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=1e-05)
    cache = np.load(ROOT / res / 'train/teacher_logits.npy', mmap_mode='r')
    plan = np.load(ROOT / 'plan.npy')
    state = {'step': 0, 'best_step': 0, 'evaluations': [], 'losses': []}

    def checkpoint():
        save(dest / 'latest.pt', {'model': model.state_dict(), 'optimizer': opt.state_dict(), 'scheduler': sched.state_dict(), 'state': state, 'protocol_sha256': ph})

    def ev(step):
        m = evaluate(model, res, p['rows']['dev'])
        state['evaluations'].append({'step': step, 'metrics': m})
        if 'best' not in state or key(m) > key(state['best']):
            state.update(best=m, best_step=step)
            save(dest / 'best.pt', {'model': model.state_dict(), 'step': step, 'metrics': m, 'widths': WIDTHS[name], 'resolution': res, 'protocol_sha256': ph, 'initialization_sha256': sha(init)})
        put(dest / 'progress.json', state)
        print('DEV', dest.name, step, json.dumps(m['macro']), flush=True)
    if (dest / 'latest.pt').exists():
        c = torch.load(dest / 'latest.pt', weights_only=False)
        assert c['protocol_sha256'] == ph
        model.load_state_dict(c['model'])
        opt.load_state_dict(c['optimizer'])
        sched.load_state_dict(c['scheduler'])
        state = c['state']
    else:
        ev(0)
        checkpoint()
    started = time.time()
    try:
        for step in range(state['step'], STEPS):
            if step % 25 == 0:
                guard()
            opt.zero_grad(set_to_none=True)
            values = []
            for i in plan[step]:
                with np.load(ROOT / res / 'train' / f'{i:03d}.npz') as z:
                    (a, b) = model(torch.from_numpy(z['pair']))
                    xy = torch.from_numpy(z['points'])
                    cell = torch.from_numpy(z['cell'])
                ta = torch.from_numpy(cache[i].astype(np.float32))
                kl = F.kl_div(a.log_softmax(1), ta.softmax(1), reduction='none').sum(1)
                det = (kl * cell).sum() / cell.sum().clamp_min(1)
                d = ops.sample_descriptors(xy, F.normalize(b, dim=1), 8).transpose(1, 2)
                sim = d[0] @ d[1].T / 0.1
                target = torch.arange(len(sim))
                geo = (F.cross_entropy(sim, target) + F.cross_entropy(sim.T, target)) / 2
                loss = (det + geo) / 2
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss')
                loss.backward()
                values.append([det.item(), geo.item()])
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            if not torch.isfinite(grad):
                raise RuntimeError('Nonfinite gradient')
            opt.step()
            sched.step()
            state['step'] = step + 1
            state['losses'].append({'step': step + 1, 'detector': float(np.mean(values, axis=0)[0]), 'geometry': float(np.mean(values, axis=0)[1])})
            if (step + 1) % 25 == 0:
                checkpoint()
                print('TRAIN', dest.name, step + 1, flush=True)
            if (step + 1) % 100 == 0:
                ev(step + 1)
                checkpoint()
    except BaseException:
        checkpoint()
        raise
    put(dest / 'report.json', {'status': 'COMPLETE', 'state': state, 'elapsed_s': time.time() - started, 'peak_process_RSS_MiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 'protocol_sha256': ph, 'plan_sha256': sha(ROOT / 'plan.npy'), 'init_sha256': sha(init), 'checkpoint_sha256': sha(dest / 'best.pt'), 'INT8_quality': 'NOT_EVALUATED'})
