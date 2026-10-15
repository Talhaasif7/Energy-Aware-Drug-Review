"""Create an upload-ready Colab bundle and notebook from checked-in sources."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
ROOT=Path(__file__).resolve().parents[1]


def main():
    files=['scripts/revision_colab.py','scripts/revision_hardware.py','scripts/metrics_utils.py','configs/revision_config.json',
           'data/01_primary_adr_detection/dev_psytar/psytar_harmonised.csv',
           'data/01_primary_adr_detection/external_val_cadec/cadec_harmonised.csv']
    manifest={'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'working_tree_sources':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}}
    bundle=ROOT/'revision_colab_bundle.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        for name in files:z.write(ROOT/name,name)
        z.writestr('source_manifest.json',json.dumps(manifest,indent=2))
    def cell(kind,source):return {'cell_type':kind,'metadata':{},'source':source.splitlines(True),**({'outputs':[],'execution_count':None} if kind=='code' else {})}
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'colab':{'name':'AIH Round 1 Revision.ipynb'},'kernelspec':{'name':'python3','display_name':'Python 3'}},'cells':[
        cell('markdown','# AIH Round 1 revision — Colab T4\nSelect Runtime → Change runtime type → T4 GPU. Upload `revision_colab_bundle.zip` when prompted. This trains ten grouped splits for three encoders and saves primary-seed checkpoints. Energy is GPU-board only; host and wall energy remain unmeasured. Run duration depends on the runtime; results are written incrementally.\n'),
        cell('code','%pip install -q --upgrade regex "transformers>=4.40,<5" safetensors pandas numpy scipy scikit-learn\n'),
        cell('code','from google.colab import files\nfrom pathlib import Path\nimport zipfile, os\nuploaded=files.upload()\nname=next((n for n in uploaded if n.endswith(".zip")),None)\nif name is None: raise ValueError("Upload revision_colab_bundle.zip")\nroot=Path("/content/aih_revision"); root.mkdir(exist_ok=True)\nwith zipfile.ZipFile(name) as archive:\n    for member in archive.namelist():\n        if not (root/member).resolve().is_relative_to(root.resolve()): raise ValueError("Unsafe archive path")\n    archive.extractall(root)\nos.chdir(root)\nprint("Ready:",root)\n'),
        cell('code','import subprocess, sys\n# Defaults: seeds 42,123,456,1001..1007; DistilBERT, PubMedBERT, BERT-tiny.\n# For a quick first session use ["--seeds","42"] below, then resume remaining seeds.\nextra=[]\nwith open("revision_colab_run.log","w") as log:\n    process=subprocess.Popen([sys.executable,"-u","scripts/revision_colab.py"]+extra,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)\n    for line in process.stdout:\n        print(line,end=""); log.write(line); log.flush()\n    code=process.wait()\nif code: raise RuntimeError(f"Revision run exited {code}; retain existing outputs and rerun only incomplete seeds")\n'),
        cell('markdown','Download both ZIP files. `revision_predictions.zip` contains predictions and board-energy surfaces; extract it at the local repository root. Give `revision_checkpoints.zip` to the Linux operator and extract at that repository root too. Checkpoints exceed GitHub normal-file limits: transfer the archive separately; do not commit weights. Colab local storage is temporary—download archives before disconnecting.\n')
    ]}
    (ROOT/'AIH_Revision_Colab.ipynb').write_text(json.dumps(notebook,indent=2),encoding='utf-8')
    print(f'Created {bundle.name} ({bundle.stat().st_size/1e6:.2f} MB) and AIH_Revision_Colab.ipynb')


if __name__=='__main__':main()
