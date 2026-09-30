# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import os
import shutil
from argparse import ArgumentParser
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from bcresnet import BCResNets
from utils import (
    CustomKeywordDataset,
    DownloadDataset,
    Padding,
    Preprocess,
    SpeechCommand,
    SplitDataset,
)


class Trainer:
    def __init__(self):
        parser = ArgumentParser(description="Train BC-ResNet on Google Speech Commands.")
        parser.add_argument("--ver", default=1, help="Google Speech Commands version 1 or 2", type=int, choices=[1, 2])
        parser.add_argument("--tau", default=1, help="Model size", type=float, choices=[1, 1.5, 2, 3, 6, 8])
        parser.add_argument("--gpu", default=0, help="GPU device id", type=int)
        parser.add_argument("--download", help="Download and prepare the dataset", action="store_true")
        parser.add_argument("--epochs", default=30, help="Number of training epochs", type=int)
        parser.add_argument("--batch-size", default=100, help="Training batch size", type=int)
        parser.add_argument("--save", default="astra.pt", help="Output checkpoint path")

        # Custom binary keyword mode.
        parser.add_argument(
            "--custom",
            action="store_true",
            help="Train a binary target-keyword-vs-other model using the raw GSC split.",
        )
        parser.add_argument(
            "--keyword",
            default="marvin",
            help="Keyword to detect in --custom mode (e.g. marvin).",
        )
        parser.add_argument(
            "--negative-ratio",
            default=2,
            type=int,
            help="Maximum number of negative examples per positive example in --custom mode.",
        )
        args = parser.parse_args()

        if args.custom and not args.keyword.strip():
            raise SystemExit("--keyword cannot be empty in --custom mode.")

        self.__dict__.update(vars(args))
        self.keyword = self.keyword.lower().strip()
        self.device = torch.device(
            "cuda:%d" % self.gpu if torch.cuda.is_available() else "cpu"
        )
        print("device:", self.device)

        self._load_data()
        self._load_model()

    def __call__(self):
        total_epoch = self.epochs
        warmup_epoch = min(5, max(1, total_epoch // 6))
        init_lr = 1e-1
        lr_lower_limit = 0

        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0, weight_decay=1e-3, momentum=0.9
        )
        n_step_warmup = len(self.train_loader) * warmup_epoch
        total_iter = len(self.train_loader) * total_epoch
        iterations = 0

        for epoch in range(total_epoch):
            self.model.train()
            running_loss = 0.0

            for inputs, labels in tqdm(
                self.train_loader, desc="epoch %d, iters" % (epoch + 1)
            ):
                iterations += 1

                if iterations < n_step_warmup:
                    lr = init_lr * iterations / max(1, n_step_warmup)
                else:
                    lr = lr_lower_limit + 0.5 * (init_lr - lr_lower_limit) * (
                        1
                        + np.cos(
                            np.pi
                            * (iterations - n_step_warmup)
                            / max(1, total_iter - n_step_warmup)
                        )
                    )

                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                inputs = self.preprocess_train(inputs, labels, augment=True)
                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, labels)

                loss.backward()
                optimizer.step()
                self.model.zero_grad()

                running_loss += loss.item()

            print(
                "epoch %d/%d - loss: %.4f - lr: %.5f"
                % (
                    epoch + 1,
                    total_epoch,
                    running_loss / max(1, len(self.train_loader)),
                    lr,
                )
            )

            with torch.no_grad():
                self.model.eval()
                valid_acc = self.Test(
                    self.valid_dataset, self.valid_loader, augment=True
                )
            print("valid acc: %.3f%%" % valid_acc)

        test_acc = self.Test(self.test_dataset, self.test_loader, augment=False)
        print("test acc: %.3f%%" % test_acc)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "tau": self.tau,
            "ver": self.ver,
            "num_classes": 2 if self.custom else 12,
            "custom": self.custom,
            "keyword": self.keyword if self.custom else None,
            "test_accuracy": float(test_acc),
        }
        torch.save(checkpoint, self.save)
        print("Saved checkpoint:", self.save)
        print("End.")

    def Test(self, dataset, loader, augment):
        true_count = 0.0
        num_testdata = float(len(dataset))

        for inputs, labels in loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            inputs = self.preprocess_test(
                inputs, labels=labels, is_train=False, augment=augment
            )
            outputs = self.model(inputs)
            prediction = torch.argmax(outputs, dim=-1)
            true_count += torch.sum(prediction == labels).detach().cpu().numpy()

        return true_count / num_testdata * 100.0

    def _download_data(self):
        """Download and create the standard Qualcomm GSC split structure."""
        if not os.path.isdir("./data"):
            os.mkdir("./data")

        base_dir = "./data/speech_commands_v0.01"
        url = (
            "https://storage.googleapis.com/download.tensorflow.org/data/"
            "speech_commands_v0.01.tar.gz"
        )
        url_test = (
            "https://storage.googleapis.com/download.tensorflow.org/data/"
            "speech_commands_test_set_v0.01.tar.gz"
        )

        if self.ver == 2:
            base_dir = base_dir.replace("v0.01", "v0.02")
            url = url.replace("v0.01", "v0.02")
            url_test = url_test.replace("v0.01", "v0.02")

        test_dir = base_dir.replace("commands", "commands_test_set")

        old_dirs = glob(base_dir.replace("commands_", "commands_*"))
        for old_dir in old_dirs:
            if os.path.isdir(old_dir):
                shutil.rmtree(old_dir)

        if os.path.isdir(test_dir):
            shutil.rmtree(test_dir)
        os.mkdir(test_dir)
        DownloadDataset(test_dir, url_test)

        if os.path.isdir(base_dir):
            shutil.rmtree(base_dir)
        os.mkdir(base_dir)
        DownloadDataset(base_dir, url)
        SplitDataset(base_dir)
        print("Done...")

        return base_dir

    def _load_data(self):
        print("Check Google Speech Commands dataset v1 or v2 ...")

        base_dir = "./data/speech_commands_v0.01"
        if self.ver == 2:
            base_dir = base_dir.replace("v0.01", "v0.02")

        if self.download:
            base_dir = self._download_data()

        if not os.path.isdir(base_dir):
            raise SystemExit(
                "Dataset not found. Run with --download first, for example:\n"
                "  python main.py --ver 1 --download --custom --keyword marvin"
            )

        noise_dir = "%s/_background_noise_" % base_dir
        transform = transforms.Compose([Padding()])

        if self.custom:
            # IMPORTANT: use the original class directories created by SplitDataset,
            # not the 12-class directories. Otherwise 'marvin' has already been
            # folded into _unknown_.
            split_root = base_dir + "_split"
            train_dir = os.path.join(split_root, "train")
            valid_dir = os.path.join(split_root, "valid")
            test_dir = os.path.join(split_root, "test")

            for required_dir in (train_dir, valid_dir, test_dir):
                if not os.path.isdir(required_dir):
                    raise SystemExit(
                        f"Missing GSC split directory: {required_dir}. "
                        "Run again with --download."
                    )

            self.train_dataset = CustomKeywordDataset(
                train_dir,
                self.keyword,
                transform=transform,
                negative_ratio=self.negative_ratio,
                seed=42,
            )
            self.valid_dataset = CustomKeywordDataset(
                valid_dir,
                self.keyword,
                transform=transform,
                negative_ratio=self.negative_ratio,
                seed=43,
            )
            self.test_dataset = CustomKeywordDataset(
                test_dir,
                self.keyword,
                transform=transform,
                negative_ratio=self.negative_ratio,
                seed=44,
            )
        else:
            train_dir = "%s/train_12class" % base_dir
            valid_dir = "%s/valid_12class" % base_dir
            test_dir = base_dir.replace("commands", "commands_test_set")

            self.train_dataset = SpeechCommand(train_dir, self.ver, transform=transform)
            self.valid_dataset = SpeechCommand(valid_dir, self.ver, transform=transform)
            self.test_dataset = SpeechCommand(test_dir, self.ver, transform=transform)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )
        self.valid_loader = DataLoader(
            self.valid_dataset, batch_size=self.batch_size, num_workers=0
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=0
        )

        print(
            "train/valid/test: %d/%d/%d"
            % (len(self.train_dataset), len(self.valid_dataset), len(self.test_dataset))
        )

        specaugment = self.tau >= 1.5
        frequency_masking_para = {1: 0, 1.5: 1, 2: 3, 3: 5, 6: 7, 8: 7}

        self.preprocess_train = Preprocess(
            noise_dir,
            self.device,
            specaug=specaugment,
            frequency_masking_para=frequency_masking_para[self.tau],
            keyword_mode=self.custom,
        )
        self.preprocess_test = Preprocess(
            noise_dir,
            self.device,
            keyword_mode=self.custom,
        )

    def _load_model(self):
        if self.custom:
            print(
                "model: BC-ResNet-%.1f | binary keyword classifier | keyword='%s'"
                % (self.tau, self.keyword)
            )
            self.model = BCResNets(int(self.tau * 8), num_classes=2).to(self.device)
        else:
            print(
                "model: BC-ResNet-%.1f on Google Speech Commands v0.0%d"
                % (self.tau, self.ver)
            )
            self.model = BCResNets(int(self.tau * 8), num_classes=12).to(self.device)


if __name__ == "__main__":
    Trainer()()
