import torch
from torch import nn
from torch.nn import functional as F


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                         nn.GroupNorm(8, cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                         nn.GroupNorm(8, cout), nn.ReLU(inplace=True))


class SmallNet(nn.Module):
    def __init__(self, channels, classes, kind):
        super().__init__()
        self.kind = kind
        self.features = nn.Sequential(block(channels, 32), nn.MaxPool2d(2),
                                      block(32, 64), nn.MaxPool2d(2), block(64, 128))
        self.head = nn.Conv2d(128, classes, 1)

    def forward(self, x):
        z = self.head(self.features(x))
        if self.kind == "classification":
            return z.mean((2, 3))
        return F.interpolate(z, x.shape[-2:], mode="bilinear", align_corners=False)


class Decoder(nn.Module):
    def __init__(self, cin, skip, cout):
        super().__init__()
        self.layers = block(cin + skip, cout)

    def forward(self, x, skip):
        x = F.interpolate(x, skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.layers(torch.cat([x, skip], dim=1))


class ResUNet(nn.Module):
    def __init__(self, encoder, classes):
        super().__init__()
        self.encoder = encoder
        self.d4, self.d3 = Decoder(512, 256, 256), Decoder(256, 128, 128)
        self.d2, self.d1 = Decoder(128, 64, 64), Decoder(64, 64, 32)
        self.head = nn.Conv2d(32, classes, 1)

    def forward(self, x):
        e = self.encoder
        c1 = e.relu(e.bn1(e.conv1(x)))
        c2 = e.layer1(e.maxpool(c1))
        c3 = e.layer2(c2)
        c4 = e.layer3(c3)
        c5 = e.layer4(c4)
        y = self.d1(self.d2(self.d3(self.d4(c5, c4), c3), c2), c1)
        return F.interpolate(self.head(y), x.shape[-2:], mode="bilinear", align_corners=False)


def build_model(recipe, channels, classes, kind, load_pretrained=True):
    if recipe.model == "small":
        return SmallNet(channels, classes, kind)
    from torchvision.models import resnet18, resnet34, ResNet18_Weights, ResNet34_Weights
    pretrained = recipe.pretrained and load_pretrained
    if recipe.model == "resnet18":
        encoder = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    else:
        encoder = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
    if channels != 3:
        old = encoder.conv1
        new = nn.Conv2d(channels, 64, 7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                # Average RGB weights and scale for the new channel count.
                new.weight.copy_(old.weight.mean(1, keepdim=True).repeat(1, channels, 1, 1) * 3 / channels)
        encoder.conv1 = new
    if kind == "classification":
        encoder.fc = nn.Linear(encoder.fc.in_features, classes)
        return encoder
    encoder.fc = nn.Identity()
    return ResUNet(encoder, classes)


def loss_value(logits, target, weights=None, dice=False):
    ce = F.cross_entropy(logits, target, weight=weights)
    if not dice:
        return ce
    classes = logits.shape[1]
    probabilities = logits.float().softmax(1)
    truth = F.one_hot(target, classes).permute(0, 3, 1, 2).float()
    intersection = (probabilities * truth).sum((0, 2, 3))
    denominator = probabilities.sum((0, 2, 3)) + truth.sum((0, 2, 3))
    # Include all classes; foreground IoU is reported independently.
    return ce + (1 - (2 * intersection + 1) / (denominator + 1)).mean()
