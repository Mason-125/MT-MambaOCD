# Mamba config
model = dict(
    type='RamanClassifier',
    backbone=dict(
        type='MTMamba',
        d_input=900,    # Input Dimensions
        d_model=900,    # Entering the mamba dimension
        n_layer=5,      # The number of layers in the model
        d_state=16,     # SSM status extended parameters
        d_conv=4,       # Local convolution width
        expand=2,       # Block extension parameters
    ),
    # neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='MultiTaskLinearClsHead',
        labels_f=[2, 3, 3],
        num_classes=3,
        in_channels=900,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,)
    )
)
