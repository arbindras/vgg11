
import torch
import torch.nn as nn
import gdown

from .classification import VGG11Classifier
from .localization import VGG11Localizer
from .segmentation import VGG11UNet


class MultiTaskPerceptionModel(nn.Module):
    """
    Multi-task perception model that composes VGG11Classifier, VGG11Localizer,
    and VGG11UNet and loads each from its own checkpoint.

    Using the actual trained model classes guarantees the architecture matches
    the saved state dicts exactly, so weight loading is correct.
    """

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        classifier_path: str = "classifier.pth",
        localizer_path: str = "localizer.pth",
        unet_path: str = "unet.pth",
    ):
        super().__init__()

        # -------------------------
        # 📥 Download weights
        # -------------------------
        gdown.download(id="1VrusgIINSRfmrx6WHxrWQsVHu68SJIvT", output=classifier_path, quiet=False)
        gdown.download(id="1-WYAwJGEQCW7q-VIkgLBfFf17PI6zvv9", output=localizer_path, quiet=False)
        gdown.download(id="13YwIqn8Wn4Ic4FOp2DQYge_nePH8SB97", output=unet_path, quiet=False)

        # -------------------------
        # 🧠 Individual models
        # Each has the EXACT architecture that was used during training,
        # so load_state_dict will match layer names and shapes.
        # -------------------------
        self.classifier = VGG11Classifier(num_classes=num_breeds, in_channels=in_channels)
        self.localizer  = VGG11Localizer(in_channels=in_channels)
        self.segmenter  = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)

        classifier_path="classifier.pth"
        localizer_path="localizer.pth"
        unet_path="unet.pth"

        # -------------------------
        # 🔥 Load weights
        # -------------------------
        self._load(self.classifier, classifier_path, "classifier")
        self._load(self.localizer,  localizer_path,  "localizer")
        self._load(self.segmenter,  unet_path,       "UNet")

    # ============================================================
    # 🔥 Weight loading
    # ============================================================
    def _load(self, model: nn.Module, path: str, name: str) -> None:
        """Load a checkpoint into *model*, supporting both raw and wrapped state dicts."""
        try:
            sd = torch.load(path, map_location="cpu")
            # train.py wraps the state dict under "model_state"
            if isinstance(sd, dict) and "model_state" in sd:
                sd = sd["model_state"]
            model.load_state_dict(sd, strict=True)
            print(f"✅ Loaded {name} weights")
        except Exception as e:
            print(f"⚠️ {name} strict load failed ({e}), trying strict=False")
            try:
                sd = torch.load(path, map_location="cpu")
                if isinstance(sd, dict) and "model_state" in sd:
                    sd = sd["model_state"]
                model.load_state_dict(sd, strict=False)
                print(f"⚠️ Loaded {name} weights (partial)")
            except Exception as e2:
                print(f"❌ {name} load completely failed: {e2}")

    # ============================================================
    # 🚀 Forward
    # ============================================================
    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: [B, C, H, W] input images.
        Returns:
            dict with keys:
              "classification" → [B, num_breeds] logits
              "localization"   → [B, 4] normalized (cx, cy, w, h) in [0, 1]
              "segmentation"   → [B, seg_classes, H, W] logits
        """
        cls_out = self.classifier(x)
        loc_out = self.localizer(x)
        seg_out = self.segmenter(x)


        return {
            "classification": cls_out,
            "localization":   loc_out,
            "segmentation":   seg_out,
        }