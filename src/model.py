import torch
import torch.nn as nn
import torchaudio
from torchvision.models import resnet18, ResNet18_Weights

class AudioResNet(nn.Module):
    def __init__(self, num_classes=3, sample_rate=16000): # Default to 3 classes
        super(AudioResNet, self).__init__()
        
        # 1. MelSpectrogram (GPU-accelerated)
        self.spec_layer = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            win_length=400,
            hop_length=160,
            n_mels=128,
            f_min=0,
            f_max=4000,
            power=2.0
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # 2. Backbone
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # 3. Modify Input (1 Channel)
        original_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size, 
            stride=original_conv.stride,
            padding=original_conv.padding, 
            bias=False
        )
        # Average weights to preserve initialization
        with torch.no_grad():
            self.backbone.conv1.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)

        # 4. Modify Output
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.spec_layer(x)
        x = self.amplitude_to_db(x)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.backbone(x)
        return x

# --- Verification Block ---
if __name__ == "__main__":
    # This block only runs if you type 'python src/model.py'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AudioResNet(num_classes=3).to(device)
    
    # Create a dummy input (Batch Size 2, 16000 samples)
    dummy_input = torch.randn(2, 16000).to(device)
    output = model(dummy_input)
    
    print(f"Model initialized on {device}")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape} (Expected: [2, 3])")
    print("SUCCESS: Model is ready for 3-class training.")