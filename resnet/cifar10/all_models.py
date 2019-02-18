from resnet.cifar10.models import resnet, densenet, resnet_single_channel

MODELS = {
        # "Deep Residual Learning for Image Recognition"
        'resnet20': resnet.ResNet20,
        'resnet32': resnet.ResNet32,
        'resnet44': resnet.ResNet44,
        'resnet56': resnet.ResNet56,
        'resnet110': resnet.ResNet110,
        'resnet1202': resnet.ResNet1202,

        # "Deep Residual Learning for Image Recognition" - For single-channel images
        'resnet20-1': resnet_single_channel.ResNet20,
        'resnet32-1': resnet_single_channel.ResNet32,
        'resnet44-1': resnet_single_channel.ResNet44,
        'resnet56-1': resnet_single_channel.ResNet56,
        'resnet110-1': resnet_single_channel.ResNet110,
        'resnet1202-1': resnet_single_channel.ResNet1202,

        # "Wide Residual Networks"
        'wrn-40-4': resnet.WRN_40_4,
        'wrn-16-4': resnet.WRN_16_4,
        'wrn-16-8': resnet.WRN_16_8,
        'wrn-28-10': resnet.WRN_28_10,

        # Based on "Identity Mappings in Deep Residual Networks"
        'preact8': resnet.PreActResNet8,
        'preact14': resnet.PreActResNet14,
        'preact20': resnet.PreActResNet20,
        'preact56': resnet.PreActResNet56,
        'preact164-basic': resnet.PreActResNet164Basic,

        # "Identity Mappings in Deep Residual Networks"
        'preact110': resnet.PreActResNet110,
        'preact164': resnet.PreActResNet164,
        'preact1001': resnet.PreActResNet1001,

        # Based on "Deep Networks with Stochastic Depth"
        'stochastic56': resnet.StochasticResNet56,
        'stochastic56-08': resnet.StochasticResNet56_08,
        'stochastic110': resnet.StochasticResNet110,
        'stochastic1202': resnet.StochasticResNet1202,
        'stochastic152-svhn': resnet.StochasticResNet152SVHN,
        'resnet152-svhn': resnet.ResNet152SVHN,

        # "Aggregated Residual Transformations for Deep Neural Networks"
        'resnext29-8-64': lambda num_classes=10: resnet.ResNeXt29(8, 64, num_classes=num_classes),  # noqa: E501
        'resnext29-16-64': lambda num_classes=10: resnet.ResNeXt29(16, 64, num_classes=num_classes),  # noqa: E501

        # "Densely Connected Convolutional Networks"
        'densenetbc100': densenet.DenseNetBC100,
            'densenetbc250': densenet.DenseNetBC250,
        'densenetbc190': densenet.DenseNetBC190,

        # Kuangliu/pytorch-cifar
        'resnet18': resnet.ResNet18,
        'resnet50': resnet.ResNet50,
        'resnet101': resnet.ResNet101,
        'resnet152': resnet.ResNet152,
}

