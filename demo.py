"""Simple real-time one-keyword BC-ResNet demo.

Usage:
    python demo.py --checkpoint astra.pt --keyword stop

The model is trained on the 12-class Google Speech Commands setup used by the
original repository. This demo watches one selected class and displays
DETECTED when its confidence crosses the chosen threshold.
"""

import argparse
import time

import numpy as np
import torch
import torchaudio

from bcresnet import BCResNets
from utils import Preprocess, SR, label_dict

try:
    import sounddevice as sd
except ImportError as exc:
    raise SystemExit(
        "Missing microphone package. Install it with: pip install sounddevice"
    ) from exc


ID_TO_LABEL = {v: k for k, v in label_dict.items()}


def load_model(checkpoint_path, device, tau_override=None):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        tau = checkpoint.get("tau", tau_override or 1)
    else:
        # Backwards compatibility with the original astra.pt format, which
        # contained only model.state_dict().
        state_dict = checkpoint
        tau = tau_override or 1

    model = BCResNets(int(float(tau) * 8)).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, float(tau)


def record_one_second():
    """Record exactly one second of mono 16 kHz microphone audio."""
    audio = sd.rec(
        int(SR),
        samplerate=SR,
        channels=1,
        dtype="float32",
        blocking=True,
    )
    return torch.from_numpy(audio.T.copy()).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description="Real-time BC-ResNet keyword demo")
    parser.add_argument("--checkpoint", default="astra.pt")
    parser.add_argument("--keyword", default="stop", choices=sorted(label_dict.keys()))
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--cooldown", type=float, default=1.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--tau", type=float, default=None,
                        help="Only needed when loading an old state-dict-only checkpoint")
    args = parser.parse_args()

    if args.keyword in ("_silence_", "_unknown_"):
        raise SystemExit("Choose a real keyword such as stop, yes, no, go, left, or right.")

    device = torch.device(
        "cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu"
    )
    print("Loading model on", device)

    model, tau = load_model(args.checkpoint, device, args.tau)
    target_id = label_dict[args.keyword]

    # No augmentation during live inference. The training pipeline already
    # teaches the model about background noise; the demo should only compute
    # the log-mel representation.
    preprocess = Preprocess(noise_loc=None, device=device)
    dummy_labels = torch.zeros(1, dtype=torch.long, device=device)

    print()
    print("=" * 48)
    print("          BC-RESNET KEYWORD SPOTTER")
    print("=" * 48)
    print("Keyword :", args.keyword.upper())
    print("Model   : BC-ResNet-%.1f" % tau)
    print("Device  :", device)
    print("Threshold: %.0f%%" % (args.threshold * 100))
    print("=" * 48)
    print("Speak the keyword once per second window.")
    print("Press Ctrl+C to stop.\n")

    last_detection = 0.0

    try:
        while True:
            print("\rListening...                              ", end="", flush=True)
            audio = record_one_second()
            audio = audio.to(device)

            start = time.perf_counter()
            with torch.inference_mode():
                features = preprocess(
                    audio,
                    labels=dummy_labels,
                    augment=False,
                    is_train=False,
                )
                logits = model(features)
                probabilities = torch.softmax(logits, dim=-1)[0]
                confidence = float(probabilities[target_id].item())
                predicted_id = int(torch.argmax(probabilities).item())
            latency_ms = (time.perf_counter() - start) * 1000.0

            now = time.monotonic()
            predicted_name = ID_TO_LABEL.get(predicted_id, "unknown")

            if confidence >= args.threshold and predicted_id == target_id and now - last_detection >= args.cooldown:
                print(
                    "\r\033[1;32m✓ DETECTED: %s\033[0m  confidence=%5.1f%%  inference=%5.1f ms"
                    % (args.keyword.upper(), confidence * 100, latency_ms)
                )
                last_detection = now
            else:
                print(
                    "\rListening...  top=%-5s  target=%5.1f%%  inference=%5.1f ms"
                    % (predicted_name, confidence * 100, latency_ms),
                    end="\n",
                )

    except KeyboardInterrupt:
        print("\n\nDemo stopped.")


if __name__ == "__main__":
    main()
