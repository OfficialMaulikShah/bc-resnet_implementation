# BC-ResNet Keyword Spotting

This repository contains the Qualcomm AI Research implementation of **Broadcasted Residual Learning for Efficient Keyword Spotting (BC-ResNet)**, together with a small real-time microphone demo for testing a trained keyword spotter.

**Paper:** Byeonggeun Kim, Simyung Chang, Jinkyu Lee, Dooyong Sung, *Broadcasted Residual Learning for Efficient Keyword Spotting*, Interspeech 2021.

The original implementation supports the **Google Speech Commands (GSC) v1 and v2** datasets. This version is set up for a simple hackathon prototype:

```text
Microphone → audio preprocessing → BC-ResNet → keyword confidence → DETECTED
```

The prototype can also train a custom binary keyword detector. The current target keyword is **`marvin`**, which is already present in the Google Speech Commands dataset, so no custom recordings are required.

## Getting Started

### Prerequisites

The original Qualcomm implementation requires:

- Python >= 3.6
- PyTorch >= 1.7.1
- `tqdm`
- `requests`

The original paper/repository uses an older PyTorch environment. If the pinned versions are difficult to install on a modern Python/Ubuntu setup, use a compatible modern PyTorch/torchaudio environment instead and keep the model/preprocessing code unchanged.

### Installation

The original environment is:

```bash
conda create -n bcresnet python=3.6
conda activate bcresnet
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 -c pytorch
conda install tqdm requests
```

For the real-time microphone demo, also install `sounddevice`:

```bash
pip install sounddevice
```

On Ubuntu, `sounddevice` requires the system PortAudio library. If you see:

```text
OSError: PortAudio library not found
```

install it with:

```bash
sudo apt update
sudo apt install portaudio19-dev libportaudio2
```

Then verify that the microphone is visible:

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

## Training on Google Speech Commands v1

For the hackathon prototype, we use **Speech Commands v1** (`--ver 1`). The dataset is downloaded and prepared automatically by `main.py`.

### 1. Test the training pipeline first

Start with only a few epochs to make sure the environment, dataset, and model work:

```bash
python main.py --ver 1 --tau 1 --gpu 0 --download --epochs 5 --save astra.pt
```

### 2. Run the actual training

Once the short test succeeds:

```bash
python main.py --ver 1 --tau 1 --gpu 0 --epochs 30 --save astra.pt
```

You do **not** need `--download` on subsequent runs because the dataset is already in `data/`.

For example:

```bash
python main.py --ver 1 --tau 1 --gpu 0 --epochs 30 --save stop_bcresnet.pt
```

### Command-line options

| Option | Meaning | Example |
|---|---|---|
| `--ver` | Google Speech Commands version | `--ver 1` |
| `--tau` | BC-ResNet model size | `--tau 1` |
| `--gpu` | GPU device ID | `--gpu 0` |
| `--download` | Download and prepare the dataset | `--download` |
| `--epochs` | Number of training epochs | `--epochs 30` |
| `--batch-size` | Training batch size | `--batch-size 100` |
| `--save` | Output checkpoint filename | `--save astra.pt` |

### What `--ver` means

- `--ver 1` → Google Speech Commands **v1**
- `--ver 2` → Google Speech Commands **v2**

The standard examples use v1, while the current custom `marvin` detector can be trained on v2 as described above.

### What `--tau` means

`tau` controls the BC-ResNet model size. For the first prototype, use:

```bash
--tau 1
```

This is the smallest model and is useful for getting the complete pipeline running quickly. Larger values can be tested later.


## Custom Keyword: MARVIN

This project supports a custom binary classification mode that trains BC-ResNet to distinguish **`marvin`** from **other spoken words** in the Google Speech Commands dataset.

Because `marvin` is already included in the dataset, you do **not** need to record your own training clips. The custom mode uses the raw dataset splits so that `marvin` remains a distinct positive class instead of being folded into the standard `_unknown_` class.

### Train on Google Speech Commands v2

To download Speech Commands **v2** and train the custom `marvin` detector:

```bash
python main.py \
    --ver 2 \
    --download \
    --custom \
    --keyword marvin \
    --tau 1 \
    --gpu 0 \
    --epochs 30 \
    --save marvin_bcresnet_v2.pt
```

For a quick pipeline test, use only one epoch first:

```bash
python main.py --ver 2 --download --custom --keyword marvin --tau 1 --gpu 0 --epochs 1 --save marvin_test.pt
```

After the dataset has been downloaded, `--download` is not required for subsequent training runs.

### Class imbalance

The dataset contains many more clips that are **not `marvin`** than clips that are `marvin`. Training directly on every negative clip can cause the model to become biased toward the `OTHER` class. Overall accuracy can then look high even if the model is poor at detecting `marvin`.

The custom dataset loader handles this automatically using `--negative-ratio`. By default, it keeps at most **2 negative examples for every 1 positive (`marvin`) example**:

```text
MARVIN : OTHER
   1   :   2
```

The default is:

```bash
--negative-ratio 2
```

You can use a 1:1 balance instead with:

```bash
--negative-ratio 1
```

For example:

```bash
python main.py \
    --ver 2 \
    --download \
    --custom \
    --keyword marvin \
    --negative-ratio 1 \
    --tau 1 \
    --gpu 0 \
    --epochs 30 \
    --save marvin_bcresnet_v2.pt
```

A 1:2 ratio is a good starting point because it provides more negative examples without allowing the negative class to overwhelm the positive class.

### Run the MARVIN detector

After training, the checkpoint stores the custom keyword, so `--keyword` does not need to be specified during inference.

For a WAV file:

```bash
python demo.py --checkpoint marvin_bcresnet_v2.pt --file your_audio.wav
```

For microphone inference:

```bash
python demo.py --checkpoint marvin_bcresnet_v2.pt
```

The model performs binary classification:

```text
MARVIN  → positive / detected
OTHER   → negative / not detected
```

## Dataset and Classes

The standard training setup creates 12 classes:

```text
_silence_
_unknown_
down
go
left
no
off
on
right
stop
up
yes
```

For example, the label used by the code for `yes` is `11`, while `stop` is `9`.

The dataset is stored under:

```text
data/
└── speech_commands_v0.01/
```

The training/validation/test splits and the 12-class version are generated automatically by `main.py`.

## Real-Time Keyword Demo

After training, `demo.py` can run BC-ResNet in either of two modes:

1. **Microphone mode** — continuously listens and detects the selected keyword.
2. **Audio-file mode** — accepts a local audio file such as `.wav` and classifies it once.

### Microphone mode

For example, to detect `yes`:

```bash
python demo.py --checkpoint astra.pt --keyword yes
```

For `stop`:

```bash
python demo.py --checkpoint astra.pt --keyword stop
```

Microphone mode requires `sounddevice`. On Ubuntu, install its PortAudio system dependency if necessary:

```bash
pip install sounddevice
sudo apt install portaudio19-dev libportaudio2
```

You can test that the microphone is visible with:

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

### Audio-file mode

You can also upload/provide an audio file and have the model recognize the keyword without using the microphone. This mode does **not** require `sounddevice` or PortAudio.

For a WAV file:

```bash
python demo.py --checkpoint astra.pt --keyword yes --file my_audio.wav
```

For example:

```bash
python demo.py --checkpoint astra.pt --keyword stop --file recordings/stop.wav
```

The demo automatically:

- loads the audio file with `torchaudio`
- converts stereo/multi-channel audio to mono
- resamples audio to 16 kHz if necessary
- pads audio shorter than 1 second
- truncates audio longer than 1 second
- applies the same log-mel preprocessing used by BC-ResNet
- prints the model's top prediction, target confidence, and inference time

Example output when the target word is detected:

```text
Top prediction : YES
Target score   : 94.8%
Inference      : 38.2 ms

✓ DETECTED: YES
```

If the selected keyword is not detected:

```text
Top prediction : NO
Target score   : 3.1%
Inference      : 37.9 ms

Not detected.
```

**Tip:** For the easiest first test, use file mode. You can record a short `.wav` clip of yourself saying the target word and run it through the trained model before troubleshooting live microphone input.

The demo displays information similar to:

```text
========================================
       BC-RESNET KEYWORD SPOTTER
========================================

Keyword : YES
Model   : BC-ResNet-1.0
Device  : cuda:0
Threshold: 80%

Listening...
```

When the keyword is detected:

```text
✓ DETECTED: YES
confidence=94.8%
inference=38.2 ms
```

The demo uses a confidence threshold and a short cooldown to prevent a single spoken word from producing repeated detections.

### Demo options

```bash
python demo.py --help
```

Important options include:

```text
--checkpoint    Path to the trained checkpoint
--keyword       Keyword to detect
--file          Audio file to classify instead of using the microphone
--threshold     Detection confidence threshold (default: 0.80)
--cooldown      Seconds to wait after a detection (default: 1.0)
--gpu           GPU device ID
--tau           Model size, only needed for old state-dict-only checkpoints
```

Example with a stricter threshold:

```bash
python demo.py --checkpoint astra.pt --keyword yes --threshold 0.90
```

## Recommended Hackathon Workflow

The recommended order is:

```text
1. Set up BC-ResNet
        ↓
2. Download Google Speech Commands v1
        ↓
3. Train BC-ResNet-1
        ↓
4. Verify validation/test accuracy
        ↓
5. Run microphone demo
        ↓
6. Say the selected keyword
        ↓
7. Show ✓ DETECTED
        ↓
8. Use the custom `marvin` binary detector when a single-word target is required
```

### Minimum viable demo

The minimum working prototype is simply:

```text
🎤 Microphone
     ↓
  BC-ResNet
     ↓
  keyword?
     ↓
✓ DETECTED
```

A website or graphical interface is optional. The terminal demo is enough to validate the complete ML pipeline before spending time on UI.

## Troubleshooting

### `OSError: PortAudio library not found`

Install the Ubuntu system libraries:

```bash
sudo apt update
sudo apt install portaudio19-dev libportaudio2
```

Then test:

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

### No microphone appears

Run:

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

Check that an input device is listed. If multiple microphones are available, the default system input device can be changed in Ubuntu's sound settings.

### CUDA/GPU issues

To check whether PyTorch sees your GPU:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

If CUDA is unavailable, the code can fall back to CPU, although training and inference may be slower.

## Project Files

```text
main.py             Training and evaluation
bcresnet.py         BC-ResNet model architecture
subspectralnorm.py  Sub-Spectral Normalization
utils.py            Dataset, preprocessing, and augmentation utilities
demo.py             Real-time microphone keyword detection
```

## Reference

If you find this work useful for your research, please cite:

```text
@inproceedings{kim21l_interspeech,
  author={Byeonggeun Kim and Simyung Chang and Jinkyu Lee and Dooyong Sung},
  title={{Broadcasted Residual Learning for Efficient Keyword Spotting}},
  year={2021},
  booktitle={Proc. Interspeech 2021},
  pages={4538--4542},
  doi={10.21437/Interspeech.2021-383}
}
```
