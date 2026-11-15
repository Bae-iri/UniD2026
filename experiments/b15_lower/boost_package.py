"""Package inference sources, fixed weights and audit reports without raw data."""
import argparse,hashlib,json,shutil
from pathlib import Path
from importlib.metadata import distribution


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    args.output.mkdir(parents=True,exist_ok=False)
    release=args.repo/'experiments/b15_lower/final_push_v1'
    for p in release.iterdir():
        if p.is_file() and p.suffix in ['.csv','.json','.md','.txt','.py']:shutil.copyfile(p,args.output/p.name)
    shutil.copyfile(args.artifact,args.output/'model.joblib')
    shutil.copyfile(args.weights,args.output/'resnet18_weights.pt')
    dist=distribution('torchvision')
    license_path=next(p for p in dist.files if str(p).endswith('.dist-info/LICENSE'))
    shutil.copyfile(dist.locate_file(license_path),args.output/'TORCHVISION_LICENSE')
    base=args.output/'base';base.mkdir()
    shutil.copyfile(args.base/'preprocessing.json',base/'preprocessing.json')
    for seed in [42,43,44]:
        dest=base/f'seed_{seed}/B';dest.mkdir(parents=True)
        shutil.copyfile(args.base/f'seed_{seed}/B/best.pt',dest/'best.pt')
    names=['boost_predict.py','boost_multimodal.py','boost_model.py','boost_family.py','boost_visual.py',
           'infer.py','pipeline.py','reproduce.py','validate.py']
    lib=args.output/'src/experiments/b15_lower';lib.mkdir(parents=True)
    for name in names:shutil.copyfile(args.repo/'experiments/b15_lower'/name,lib/name)
    shutil.copytree(args.repo/'experiments/b15_lower/core',lib/'core',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(args.repo/'abc_reference',args.output/'src/abc_reference',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (args.output/'README.md').write_text('''# 재현용 모델 묶음

원본 학습 자료나 행별 예측 캐시 없이 포함된 가중치만으로 실행한다.
Python 3.10.12 환경에서 검사했다. 기본 정책은 두 보정기가 동의할 때만 수정하는 합의 정책이다.
이미지 인코더는 TorchVision ResNet18 IMAGENET1K_V1의 고정 가중치다.
가중치 SHA256은 manifest.json에 기록했다. 인코더는 이 대회의 정답으로 학습하지 않았다.

```bash
python test.py --input /path/to/test.csv --image_root /path/to/test \\
  --sample_submission /path/to/sample_submission.csv --output submission.csv --device cpu
```

출력 파일이 이미 있으면 덮어쓰지 않는다. GPU 인코더를 쓰려면 --device cuda를 지정한다.
Test 전체에서 CPU/GPU 출력이 동일함을 확인했다.
실제 이미지로 prefix 검사까지 실행하려면 --audit-prefix를 추가한다.
대안인 확률 평균 정책은 --policy average50으로 재현한다.
검증 결과와 한계는 REPORT.md에 있다. 비공개 성능을 보장하지 않는다.
''')
    paths=sorted(p for p in args.output.rglob('*') if p.is_file())
    manifest=dict(files={str(p.relative_to(args.output)):sha(p) for p in paths},
                  training_data_included=False,prediction_cache_required=False,policy='agreement50',
                  encoder_origin='https://download.pytorch.org/models/resnet18-f37072fd.pth')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive=shutil.make_archive(str(args.output),'zip',root_dir=args.output)
    print(json.dumps(dict(archive=archive,sha256=sha(Path(archive)),files=len(paths)+1)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['repo','artifact','base','weights','output']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
