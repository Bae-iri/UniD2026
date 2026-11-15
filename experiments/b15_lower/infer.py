"""Run the archived B checkpoints and J096/S3 directly on images and sensors."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from abc_reference import abc_fusion_experiment as abc
from pipeline import predict
from reproduce import load_artifact


class Runtime:
    def __init__(self, base, upper, lower, device='cpu'):
        self.artifact = load_artifact(upper, lower)
        self.device = torch.device(device)
        self.models = []
        meta = json.loads((Path(base)/'preprocessing.json').read_text())
        self.prep = abc.SensorPreprocessor.__new__(abc.SensorPreprocessor)
        for key in ['median', 'mean', 'std']:
            setattr(self.prep, key, pd.Series(meta[key], index=abc.NUMERIC))
        for seed in [42, 43, 44]:
            model = abc.FusionModel('B').to(self.device)
            checkpoint = torch.load(Path(base)/f'seed_{seed}/B/best.pt', map_location=self.device, weights_only=True)
            model.load_state_dict(checkpoint['state'])
            self.models.append(model.eval())

    @torch.inference_mode()
    def probabilities(self, frame, image_root):
        root = Path(image_root).resolve()
        q, gas = self.prep.transform(frame)
        output = [[] for _ in self.models]
        paths = []
        for value in frame.image_path:
            relative = Path(str(value).replace('\\', '/'))
            resolved = (root/relative).resolve()
            if relative.is_absolute() or not resolved.is_relative_to(root):
                raise ValueError('Image path must stay inside image_root')
            paths.append(resolved)
        # Fixed-size batches reduce shape-dependent floating point differences.
        for start in range(0, len(frame), 128):
            stop = min(start+128, len(frame))
            size = stop-start
            packed = np.zeros((128, 4, 128, 128), dtype=np.uint8)
            packed[:size] = np.stack([abc.common.read_image(p, 128, 4)[0] for p in paths[start:stop]])
            qq = np.zeros((128, q.shape[1]), dtype=np.float32)
            gg = np.zeros((128, *gas.shape[1:]), dtype=np.float32)
            qq[:size], gg[:size] = q[start:stop], gas[start:stop]
            tensor = torch.from_numpy(packed).to(self.device)
            inputs = (tensor[:, :3].float()/255., tensor[:, 3:4],
                      torch.from_numpy(qq).to(self.device), torch.from_numpy(gg).to(self.device))
            for j, model in enumerate(self.models):
                output[j].append(model(*inputs).softmax(-1)[:size].cpu().numpy())
        return np.stack([np.concatenate(values) for values in output])

    def labels(self, frame, image_root):
        p = self.probabilities(frame, image_root)
        return predict(self.artifact, frame, p)['S3'], p


def main(args):
    torch.set_num_threads(4)
    frame = pd.read_csv(args.input)
    if not frame.sample_id.is_unique:
        raise ValueError('Duplicate sample ID')
    runtime = Runtime(args.base, args.upper_artifact, args.lower_artifact, args.device)
    labels, _ = runtime.labels(frame, args.image_root)
    table = pd.DataFrame({'sample_id': frame.sample_id, 'label': labels.astype(int)})
    if args.sample_submission:
        sample = pd.read_csv(args.sample_submission)
        assert sample.sample_id.is_unique and set(sample.sample_id) == set(table.sample_id)
        table = table.set_index('sample_id').loc[sample.sample_id].reset_index()
    table.to_csv(args.output, index=False, mode='x')
    print('Wrote', len(table), 'rows:', args.output, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ['input', 'image-root', 'base', 'upper-artifact', 'lower-artifact', 'output']:
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--sample-submission', type=Path)
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    with threadpool_limits(limits=2):
        main(ap.parse_args())
