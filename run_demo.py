import soundfile as sf
import json
import glob
import os
from tqdm import tqdm
from segfeat.inference import SavedSegmentor
import argparse

def main(args):
    with open(args.model_config_file, 'r') as config_file:
        config = json.load(config_file)

    saved_model = SavedSegmentor(config, args.device, args.chkpt_path)

    if os.path.isdir(args.file_or_dir):
        files = glob.glob(os.path.join(args.file_or_dir, '*.wav'))
    else:
        files = [args.file_or_dir]

    result = []

    for file_id, filename in tqdm(enumerate(files)):
        wav, sr = sf.read(filename)

        chunks = saved_model(wav, sr)        

        ret = {
            'filename': filename,
            'data': chunks
        }

        result.append(ret)

    with open(args.output_path, 'w') as out:
        json.dump(result, out, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phoneme Segmentation Demo')
    parser.add_argument('--model_config_file', help='Path to Model Config JSON file. Parameters in that file are a subset of parameters in main.py', 
                        type=str, default='configs/model_params.json')
    parser.add_argument('--file_or_dir', type=str, help='directory with wav files or single wav file')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--chkpt_path', type=str, default='segmentor.ckpt')
    parser.add_argument('--output_path', type=str, default='results/segmentation.json', help='where to save the results of segmentation')

    args = parser.parse_args()

    main(args)


        