import torch
import torch.nn as nn
import torchvision.models as models

from src.schema.model import ModelFamily

from src.loading.models.mobilenet.model import MobileNetV3Small

family_to_model = {
    ModelFamily.VGG: lambda weights: models.vgg11_bn(weights=weights),
    ModelFamily.RESNET: lambda weights: models.resnet18(weights=weights),
    ModelFamily.EFFICIENTNET: lambda weights: models.efficientnet_b0(weights=weights),
    ModelFamily.MOBILENET: lambda weights: models.mobilenet_v3_small(weights=weights),
    ModelFamily.DENSENET: lambda weights: models.densenet121(weights=weights),
    ModelFamily.REGNET: lambda weights: models.regnet_y_400mf(weights=weights),
    ModelFamily.SQUEEZENET: lambda weights: models.squeezenet1_0(weights=weights),
    ModelFamily.MOBILENETV3: lambda weights: MobileNetV3Small(pretrained=True),
}

default_weights = {
    ModelFamily.VGG: models.VGG11_BN_Weights.IMAGENET1K_V1,
    ModelFamily.RESNET: models.ResNet18_Weights.IMAGENET1K_V1,
    ModelFamily.EFFICIENTNET: models.EfficientNet_B0_Weights.IMAGENET1K_V1,
    ModelFamily.MOBILENET: models.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
    ModelFamily.DENSENET: models.DenseNet121_Weights.IMAGENET1K_V1,
    ModelFamily.REGNET: models.RegNet_Y_400MF_Weights.IMAGENET1K_V1,
    ModelFamily.SQUEEZENET: models.SqueezeNet1_0_Weights.IMAGENET1K_V1,
    ModelFamily.MOBILENETV3: models.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
}

def suggest_output_layer(model, num_classes=10):
    # print("Top-level modules and their types:")
    # for name, module in model.named_children():
    #     print(f"  {name}: {module.__class__.__name__}")
    
    # print("\nSuggestion for replacing classifier:")
    
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        # print("→ Replace `model.fc` with:")
        # print(f"  model.fc = nn.Linear({model.fc.in_features}, {num_classes})")
        pass
        
    elif hasattr(model, "classifier"):
        classifier = model.classifier
        if isinstance(classifier, nn.Sequential):
            # Look for last Linear or Conv2d
            last_idx = -1
            for i in reversed(range(len(classifier))):
                if isinstance(classifier[i], (nn.Linear, nn.Conv2d)):
                    last_idx = i
                    break
            if last_idx >= 0:
                layer = classifier[last_idx]
                if isinstance(layer, nn.Linear):
                    # print(f"→ Replace `model.classifier[{last_idx}]` with:")
                    # print(f"  model.classifier[{last_idx}] = nn.Linear({layer.in_features}, {num_classes})")
                    pass
                elif isinstance(layer, nn.Conv2d):
                    # print(f"→ Replace `model.classifier[{last_idx}]` with:")
                    # print(f"  model.classifier[{last_idx}] = nn.Conv2d({layer.in_channels}, {num_classes}, kernel_size=1)")
                    pass
        elif isinstance(classifier, nn.Linear):
            # print("→ Replace `model.classifier` with:")
            # print(f"  model.classifier = nn.Linear({classifier.in_features}, {num_classes})")
            pass
        elif isinstance(classifier, nn.Conv2d):
            # print("→ Replace `model.classifier` with:")
            # print(f"  model.classifier = nn.Conv2d({classifier.in_channels}, {num_classes}, kernel_size=1)")
            pass
        else:
            # print("Unrecognized classifier structure.")
            pass
    else:
        # print("Unable to detect a known classifier layer to replace.")
        pass

def load_pretrained_model(family: ModelFamily, num_classes: int, pretrained: bool = True):
    try:
        if pretrained:
            weights = default_weights[family]
        else:
            weights = None
        
        model = family_to_model[family](weights)
        
        if family == ModelFamily.VGG:
            model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
        elif family == ModelFamily.RESNET:
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        elif family == ModelFamily.EFFICIENTNET:
            model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
        elif family == ModelFamily.MOBILENET:
            model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)
        elif family == ModelFamily.DENSENET:
            model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
        elif family == ModelFamily.REGNET:
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        elif family == ModelFamily.SQUEEZENET:
            model.classifier[1] = torch.nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        elif family == ModelFamily.MOBILENETV3:
            model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)
        else:
            raise ValueError(f"Model head replacement not implemented for {family}")
        
        return model

    except KeyError:
        raise ValueError(f"Unknown model family: {family}")
