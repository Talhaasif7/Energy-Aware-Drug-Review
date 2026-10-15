"""Measured-domain energy utilities for Linux/Colab revision benchmarks.

No CodeCarbon return values or TDP proxies are used. A missing meter produces
an explicit missing domain, never a made-up complete-system energy value.
"""
from __future__ import annotations
import platform
import subprocess
import threading
import time
import numpy as np


class BoardTracer:
    def __init__(self,interval=.1):
        self.interval=interval;self.readings=[];self.stop_event=threading.Event()
    def start(self):
        self.readings=[];self.stop_event.clear()
        def loop():
            while not self.stop_event.is_set():
                try:
                    r=subprocess.run(['nvidia-smi','--id=0','--query-gpu=power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3,check=True)
                    self.readings.append((time.perf_counter(),float(r.stdout.strip())))
                except (ValueError,subprocess.SubprocessError,OSError):pass
                self.stop_event.wait(self.interval)
        self.thread=threading.Thread(target=loop,daemon=True);self.thread.start()
    def stop(self,t0,t1):
        self.stop_event.set();self.thread.join(timeout=4)
        rows=[(t,p) for t,p in self.readings if t0<=t<=t1]
        if len(rows)<2:return {'status':'unavailable_insufficient_power_samples','energy_j':None,'samples':rows}
        ts=np.array([r[0] for r in rows]);pw=np.array([r[1] for r in rows]);gap=(ts[0]-t0)+(t1-ts[-1])
        if gap>max(.5,.05*(t1-t0)):return {'status':'unavailable_inadequate_window_coverage','energy_j':None,'boundary_extension_s':gap,'samples':rows}
        integral=np.trapezoid if hasattr(np,'trapezoid') else np.trapz
        # Endpoint extension is limited and disclosed; actual timestamps account
        # for nvidia-smi query overhead, rather than assuming 10 Hz samples.
        energy=float(integral(np.r_[pw[0],pw,pw[-1]],np.r_[t0,ts,t1]))
        return {'status':'measured_board_trace','energy_j':energy,'samples':rows,
                'boundary_extension_s':float(gap),'actual_mean_sample_interval_s':float(np.diff(ts).mean())}


def runtime_metadata():
    import torch,transformers
    return {'platform':platform.platform(),'python':platform.python_version(),'torch':torch.__version__,
            'transformers':transformers.__version__,'numpy':np.__version__,'torch_threads':torch.get_num_threads(),
            'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}


def probabilities(model,tokenizer,texts,device,max_length,use_fp16=False):
    import torch
    enc=tokenizer(list(texts),padding='max_length',truncation=True,max_length=max_length,return_tensors='pt')
    enc={k:v.to(device) for k,v in enc.items()}
    with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.float16,enabled=use_fp16):
        p=torch.softmax(model(**enc).logits.float(),dim=1)[:,1]
    return p.cpu().numpy().astype(np.float64)


def infer_all(model,tokenizer,texts,device,batch_size=64,max_length=128,use_fp16=False):
    parts=[probabilities(model,tokenizer,texts[i:i+batch_size],device,max_length,use_fp16) for i in range(0,len(texts),batch_size)]
    return np.concatenate(parts)


def measure_case(model,tokenizer,texts,device,batch_size,max_length,config,rapl=None,operation='pipeline'):
    """Measure variable corpus text batches, including tokenization in timing.

    CUDA: board trace only. CPU: RAPL package only. Neither is a wall-power
    complete-system measurement. All raw traces and actual window lengths persist.
    """
    import torch
    if not len(texts) or batch_size<1 or max_length<1:raise ValueError('Nonempty texts and positive batch/length required')
    tracer=None
    use_gpu=str(device).startswith('cuda')
    def one(i):
        batch=[str(texts[(i+j)%len(texts)]) for j in range(batch_size)]
        if operation=='tokenization':
            return tokenizer(batch,padding='max_length',truncation=True,max_length=max_length,return_tensors='pt')
        return probabilities(model,tokenizer,batch,device,max_length,use_gpu)
    try:
        end=time.perf_counter()+config['energy_warmup_s'];i=0
        while time.perf_counter()<end:one(i);i=(i+batch_size)%len(texts)
        if use_gpu:torch.cuda.synchronize()
        repeats=[]
        for rep in range(config['energy_repeats']):
            if rep and config.get('energy_cooldown_s',0)>0:
                time.sleep(config['energy_cooldown_s'])
            tracer=BoardTracer() if use_gpu else None
            if tracer:tracer.start()
            before=rapl.read() if rapl else None
            t0=time.perf_counter();n=0
            while time.perf_counter()-t0<config['energy_measure_s']:
                one(i);i=(i+batch_size)%len(texts);n+=batch_size
            if use_gpu:torch.cuda.synchronize()
            t1=time.perf_counter();after=rapl.read() if rapl else None
            board=tracer.stop(t0,t1) if tracer else None
            joules=(board['energy_j'] if board else rapl.delta_j(before,after) if rapl else None)
            elapsed=t1-t0
            repeats.append({'repeat':rep,'elapsed_s':elapsed,'sentences':n,'throughput_sps':n/elapsed,
                'energy_j':joules,'gross_j_1000':joules*1000/n if joules is not None else None,
                'average_power_w':joules/elapsed if joules is not None else None,
                'counter_before_uj':before,'counter_after_uj':after,'board_trace':board})
        valid=[r for r in repeats if r['gross_j_1000'] is not None]
        if len(valid)!=len(repeats):return {'status':'missing_energy_domain','batch_size':batch_size,'max_length':max_length,'repeats':repeats}
        gross=[r['gross_j_1000'] for r in valid];med=valid[int(np.argsort(gross)[len(valid)//2])]
        cv=float(np.std(gross,ddof=1)/np.mean(gross)*100) if len(gross)>1 else None
        return {'status':'measured','scope':'GPU board; tokenization included in elapsed time, host energy unmeasured' if use_gpu else 'CPU RAPL package; tokenizer and forward pass included; RAM/whole-system excluded',
                'operation':operation,'batch_size':batch_size,'max_length':max_length,'gross_j_1000':med['gross_j_1000'],
                'throughput_sps':med['throughput_sps'],'load_w':med['average_power_w'],'cv_pct':cv,
                'repeatability_pass':bool(cv<=10) if cv is not None else None,'summary':'coupled median-energy repeat','repeats':repeats,
                'complete_system_energy_j':None,'host_energy_measured':not use_gpu,'wall_meter_available':False}
    except (torch.OutOfMemoryError,RuntimeError) as exc:
        if tracer:
            tracer.stop_event.set();tracer.thread.join(timeout=4)
        if 'out of memory' not in str(exc).lower():raise
        if use_gpu:torch.cuda.empty_cache()
        return {'status':'not_evaluated_OOM','batch_size':batch_size,'max_length':max_length,'reason':str(exc)}


def calibrated_arrays(y_cal,p_cal,p_test,p_external):
    from sklearn.isotonic import IsotonicRegression
    from metrics_utils import TemperatureScaler
    ts=TemperatureScaler().fit(y_cal,p_cal)
    iso=IsotonicRegression(out_of_bounds='clip').fit(p_cal,y_cal)
    return {'uncal':(p_cal,p_test,p_external),'temp':tuple(ts.transform(p) for p in [p_cal,p_test,p_external]),
            'iso':tuple(iso.predict(p) for p in [p_cal,p_test,p_external])}
