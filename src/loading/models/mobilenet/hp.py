from src.loading.models.mobilenet.utils import make_divisible

class SqueezeExcitationHP:
    def __init__(self, squeeze_factor: int, activation: str):
        self.squeeze_factor = squeeze_factor
        self.activation = activation

    def to_dict(self):
        return {
            "squeeze_factor": self.squeeze_factor,
            "activation": self.activation
        }

    @classmethod
    def from_dict(cls, d):
        return cls(squeeze_factor=d["squeeze_factor"], activation=d["activation"])
        
class ConvBNActivationHP:
    def __init__(self, kernel_size: int, stride: int, activation: str, channels: int = None):
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation = activation
        
    def to_dict(self):
        return {
            "channels": self.channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "activation": self.activation
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            kernel_size=d["kernel_size"],
            stride=d["stride"],
            activation=d["activation"],
            channels=d.get("channels")
        )

class InvertedResidualHP:
    def __init__(self, expand_channels: int, use_se: bool, se_hp: SqueezeExcitationHP, conv_bn_activation_hp: ConvBNActivationHP):
        self.expanded_channels = make_divisible(expand_channels, 8)
        self.use_se = use_se
        self.se_hp = se_hp
        self.conv_bn_activation_hp = conv_bn_activation_hp
        
    def to_dict(self):
        return {
            "expanded_channels": self.expanded_channels,
            "use_se": self.use_se,
            "se_hp": self.se_hp.to_dict() if self.se_hp else None,
            "conv_bn_activation_hp": self.conv_bn_activation_hp.to_dict()
        }

    @classmethod
    def from_dict(cls, d):
        se = d.get("se_hp")
        return cls(
            expand_channels=d["expanded_channels"],
            use_se=d["use_se"],
            se_hp=SqueezeExcitationHP.from_dict(se) if se else None,
            conv_bn_activation_hp=ConvBNActivationHP.from_dict(d["conv_bn_activation_hp"])
        )

class ClassifierHP:
    def __init__(self, neurons: int, activation: str, dropout_rate: float):
        self.neurons = neurons
        self.activation = activation
        self.dropout_rate = dropout_rate
        
    def to_dict(self):
        return {
            "neurons": self.neurons,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            neurons=d["neurons"],
            activation=d["activation"],
            dropout_rate=d["dropout_rate"]
        )

class MobileNetHP:
    def __init__(self, initial_conv_hp: ConvBNActivationHP, inverted_residual_hps: list[InvertedResidualHP], last_conv_upsample: int, last_conv_hp: ConvBNActivationHP, classifier_hp: ClassifierHP):
        self.initial_conv_hp = initial_conv_hp
        self.inverted_residual_hps = inverted_residual_hps
        self.last_conv_upsample = last_conv_upsample
        self.last_conv_hp = last_conv_hp
        self.classifier_hp = classifier_hp
        
    def to_dict(self):
        return {
            "initial_conv_hp": self.initial_conv_hp.to_dict(),
            "inverted_residual_hps": [ir_hp.to_dict() for ir_hp in self.inverted_residual_hps],
            "last_conv_upsample": self.last_conv_upsample,
            "last_conv_hp": self.last_conv_hp.to_dict(),
            "classifier_hp": self.classifier_hp.to_dict()
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            initial_conv_hp=ConvBNActivationHP.from_dict(d["initial_conv_hp"]),
            inverted_residual_hps=[InvertedResidualHP.from_dict(ir) for ir in d["inverted_residual_hps"]],
            last_conv_upsample=d["last_conv_upsample"],
            last_conv_hp=ConvBNActivationHP.from_dict(d["last_conv_hp"]),
            classifier_hp=ClassifierHP.from_dict(d["classifier_hp"])
        )

    def get_flattened_representation(self):
        representation = {}

        # Initial Conv
        representation['initial_conv_hp_channels'] = self.initial_conv_hp.channels
        representation['initial_conv_hp_kernel_size'] = self.initial_conv_hp.kernel_size
        representation['initial_conv_hp_stride'] = self.initial_conv_hp.stride
        for act in ['Hardswish', 'ReLU']:
            representation[f'initial_conv_hp_activation_{act}'] = int(self.initial_conv_hp.activation == act)

        # Inverted Residual Blocks
        for idx, block in enumerate(self.inverted_residual_hps):
            prefix = f'block_{idx}'

            representation[f'{prefix}_expanded_channels'] = block.expanded_channels
            representation[f'{prefix}_use_se'] = int(block.use_se)

            # Squeeze Excitation
            if block.use_se:
                representation[f'{prefix}_se_squeeze_factor'] = block.se_hp.squeeze_factor
                for se_act in ['Hardsigmoid', 'Sigmoid']:
                    representation[f'{prefix}_se_activation_{se_act}'] = int(block.se_hp.activation == se_act)
            else:
                representation[f'{prefix}_se_squeeze_factor'] = 0
                representation[f'{prefix}_se_activation_NONE'] = 1

            # Convolution
            representation[f'{prefix}_channels'] = block.conv_bn_activation_hp.channels
            representation[f'{prefix}_kernel_size'] = block.conv_bn_activation_hp.kernel_size
            representation[f'{prefix}_stride'] = block.conv_bn_activation_hp.stride

            for act in ['Hardswish', 'ReLU']:
                representation[f'{prefix}_activation_{act}'] = int(block.conv_bn_activation_hp.activation == act)

        # Last Conv
        representation['last_conv_upsample'] = self.last_conv_upsample
        representation['last_conv_hp_channels'] = self.last_conv_hp.channels
        representation['last_conv_hp_kernel_size'] = self.last_conv_hp.kernel_size
        representation['last_conv_hp_stride'] = self.last_conv_hp.stride
        for act in ['Hardswish', 'ReLU']:
            representation[f'last_conv_hp_activation_{act}'] = int(self.last_conv_hp.activation == act)

        # Classifier
        representation['classifier_hp_neurons'] = self.classifier_hp.neurons
        representation['classifier_hp_dropout_rate'] = self.classifier_hp.dropout_rate
        for act in ['Hardswish', 'ReLU']:
            representation[f'classifier_hp_activation_{act}'] = int(self.classifier_hp.activation == act)

        return representation.values()
        
original_hp = MobileNetHP(
    initial_conv_hp=ConvBNActivationHP(channels=16, kernel_size=3, stride=2, activation="Hardswish"),
    inverted_residual_hps=[
        InvertedResidualHP(
            expand_channels=16,
            use_se=True,
            se_hp=SqueezeExcitationHP(squeeze_factor=4, activation="Hardsigmoid"),
            conv_bn_activation_hp=ConvBNActivationHP(channels=16, kernel_size=3, stride=2, activation="ReLU"),
        ),
        InvertedResidualHP(
            expand_channels=72,
            use_se=False,
            se_hp=None,
            conv_bn_activation_hp=ConvBNActivationHP(channels=24, kernel_size=3, stride=2, activation="ReLU"),
        ),
        InvertedResidualHP(
            expand_channels=88,
            use_se=False,
            se_hp=None,
            conv_bn_activation_hp=ConvBNActivationHP(channels=24, kernel_size=3, stride=1, activation="Hardswish"),
        ),
        InvertedResidualHP(
            expand_channels=96,
            use_se=True,
            se_hp=SqueezeExcitationHP(squeeze_factor=4, activation="Hardsigmoid"),
            conv_bn_activation_hp=ConvBNActivationHP(channels=40, kernel_size=5, stride=2, activation="Hardswish"),
        ),
        InvertedResidualHP(
            expand_channels=240,
            use_se=True,
            se_hp=SqueezeExcitationHP(squeeze_factor=4, activation="Hardsigmoid"),
            conv_bn_activation_hp=ConvBNActivationHP(channels=40, kernel_size=5, stride=1, activation="Hardswish"),
        ),
        InvertedResidualHP(
            expand_channels=240,
            use_se=True,
            se_hp=SqueezeExcitationHP(squeeze_factor=4, activation="Hardsigmoid"),
            conv_bn_activation_hp=ConvBNActivationHP(channels=40, kernel_size=5, stride=1, activation="Hardswish"),
        ),
        InvertedResidualHP(
            expand_channels=120,
            use_se=True,
            se_hp=SqueezeExcitationHP(squeeze_factor=4, activation="Hardsigmoid"),
            conv_bn_activation_hp=ConvBNActivationHP(channels=48, kernel_size=5, stride=1, activation="Hardswish"),
        ),
        InvertedResidualHP(
            expand_channels=144,
            use_se=True,
            se_hp=SqueezeExcitationHP(squeeze_factor=4, activation="Hardsigmoid"),
            conv_bn_activation_hp=ConvBNActivationHP(channels=48, kernel_size=5, stride=1, activation="Hardswish"),
        ),
        InvertedResidualHP(
            expand_channels=288,
            use_se=True,
            se_hp=SqueezeExcitationHP(squeeze_factor=4, activation="Hardsigmoid"),
            conv_bn_activation_hp=ConvBNActivationHP(channels=96, kernel_size=5, stride=1, activation="Hardswish"),
        ),
        InvertedResidualHP(
            expand_channels=576,
            use_se=True,
            se_hp=SqueezeExcitationHP(squeeze_factor=4, activation="Hardsigmoid"),
            conv_bn_activation_hp=ConvBNActivationHP(channels=96, kernel_size=5, stride=1, activation="Hardswish"),
        ),
        InvertedResidualHP(
            expand_channels=576,
            use_se=True,
            se_hp=SqueezeExcitationHP(squeeze_factor=4, activation="Hardsigmoid"),
            conv_bn_activation_hp=ConvBNActivationHP(channels=96, kernel_size=5, stride=1, activation="Hardswish"),
        ),
    ],
    last_conv_upsample=6,
    last_conv_hp=ConvBNActivationHP(channels=96*6, kernel_size=1, stride=1, activation="Hardswish"),
    classifier_hp=ClassifierHP(neurons=1024, activation="Hardswish", dropout_rate=0.2)
)
