"""BC-ResNet keyword spotting demo.

Live microphone:
    python demo.py --checkpoint astra.pt --keyword yes

Audio file:
    python demo.py --checkpoint astra.pt --keyword yes --file my_audio.wav

The file mode accepts an audio file, converts it to mono 16 kHz, pads/truncates
it to one second, and runs the same BC-ResNet preprocessing used for inference.
"""

import argparse
import time

import torch
import torchaudio

from bcresnet import BCResNets
from utils import Preprocess, SR, label_dict

ID_TO_LABEL = {v: k for k, v in label_dict.items()}


def load_model(checkpoint_path, device, tau_override=None):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        tau = checkpoint.get("tau", tau_override or 1)
        num_classes = checkpoint.get("num_classes", 12)
        custom = checkpoint.get("custom", False)
        saved_keyword = checkpoint.get("keyword")
    else:
        state_dict = checkpoint
        tau = tau_override or 1
        num_classes = 12
        custom = False
        saved_keyword = None

    model = BCResNets(int(float(tau) * 8), num_classes=int(num_classes)).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, float(tau), bool(custom), saved_keyword, int(num_classes)


def prepare_audio_file(path):
    """Load an audio file and convert it to the 1-second, 16 kHz model input."""
    audio, sample_rate = torchaudio.load(path)

    # Convert stereo/multi-channel audio to mono.
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    # Match the 16 kHz sampling rate used by Google Speech Commands.
    if sample_rate != SR:
        audio = torchaudio.functional.resample(audio, sample_rate, SR)

    # BC-ResNet's GSC input is exactly one second.
    if audio.shape[-1] < SR:
        audio = torch.nn.functional.pad(audio, (0, SR - audio.shape[-1]))
    elif audio.shape[-1] > SR:
        audio = audio[..., :SR]

    return audio.unsqueeze(0)  # [batch, channel, time]


def record_one_second(sd):
    """Record exactly one second of mono 16 kHz microphone audio."""
    audio = sd.rec(
        int(SR),
        samplerate=SR,
        channels=1,
        dtype="float32",
        blocking=True,
    )
    return torch.from_numpy(audio.T.copy()).unsqueeze(0)


def predict(model, preprocess, audio, dummy_labels, device, target_id):
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
    return predicted_id, confidence, latency_ms


def main():
    parser = argparse.ArgumentParser(description="BC-ResNet keyword spotting demo")
    parser.add_argument("--checkpoint", default="astra.pt")
    parser.add_argument("--keyword", default=None, help="Target keyword. In custom mode this is usually 'marvin'.")
    parser.add_argument("--file", default=None, help="Audio file to classify instead of using the microphone")
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--cooldown", type=float, default=1.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--tau", type=float, default=None,
        help="Only needed when loading an old state-dict-only checkpoint",
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu"
    )

    print("Loading model on", device)
    model, tau, custom, saved_keyword, num_classes = load_model(args.checkpoint, device, args.tau)

    if custom:
        keyword = (args.keyword or saved_keyword or "marvin").lower()
        target_id = 1
        id_to_label = {0: "other", 1: keyword}
    else:
        keyword = (args.keyword or "stop").lower()
        if keyword not in label_dict or keyword in ("_silence_", "_unknown_"):
            raise SystemExit(
                "For a standard 12-class checkpoint, choose a keyword such as stop, yes, no, go, left, or right."
            )
        target_id = label_dict[keyword]
        id_to_label = ID_TO_LABEL

    preprocess = Preprocess(noise_loc=None, device=device, keyword_mode=custom)
    dummy_labels = torch.zeros(1, dtype=torch.long, device=device)

    print()
    print("=" * 52)
    print("              BC-RESNET KEYWORD SPOTTER")
    print("=" * 52)
    print("Keyword  :", keyword.upper())
    print("Mode     :", "custom binary" if custom else "12-class GSC")
    print("Model    : BC-ResNet-%.1f" % tau)
    print("Device   :", device)
    print("Threshold: %.0f%%" % (args.threshold * 100))
    print("=" * 52)

    if args.file:
        # File mode does not import sounddevice, so it also works without
        # the system PortAudio library installed.
        print("File     :", args.file)
        print("Processing audio...\n")
        audio = prepare_audio_file(args.file)
        predicted_id, confidence, latency_ms = predict(
            model, preprocess, audio, dummy_labels, device, target_id
        )
        predicted_name = id_to_label.get(predicted_id, "unknown")

        print("Top prediction : %s" % predicted_name.upper())
        print("Target score   : %.1f%%" % (confidence * 100))
        print("Inference      : %.1f ms" % latency_ms)

        if predicted_id == target_id and confidence >= args.threshold:
            print("\n\033[1;32m✓ DETECTED: %s\033[0m" % keyword.upper())
        else:
            print("\nNot detected.")
        return

    try:
        import sounddevice as sd
    except ImportError as exc:
        raise SystemExit(
            "Microphone mode requires sounddevice. Install it with: pip install sounddevice\n"
            "On Ubuntu also install: sudo apt install portaudio19-dev libportaudio2\n"
            "For an audio file, use --file path/to/audio.wav instead."
        ) from exc
    except OSError as exc:
        raise SystemExit(
            "Could not initialize PortAudio. On Ubuntu install:\n"
            "  sudo apt install portaudio19-dev libportaudio2\n"
            "Or use file mode: --file path/to/audio.wav\n\n"
            f"Original error: {exc}"
        ) from exc

    print("Listening... Speak the keyword once per second window.")
    print("Press Ctrl+C to stop.\n")

    last_detection = 0.0
    try:
        while True:
            print("\rListening...                              ", end="", flush=True)
            audio = record_one_second(sd)
            predicted_id, confidence, latency_ms = predict(
                model, preprocess, audio, dummy_labels, device, target_id
            )

            now = time.monotonic()
            predicted_name = id_to_label.get(predicted_id, "unknown")

            if (
                confidence >= args.threshold
                and predicted_id == target_id
                and now - last_detection >= args.cooldown
            ):
                print(
                    "\r\033[1;32m✓ DETECTED: %s\033[0m  confidence=%5.1f%%  inference=%5.1f ms"
                    % (keyword.upper(), confidence * 100, latency_ms)
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
