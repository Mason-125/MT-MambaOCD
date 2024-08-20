# dataset settings
dataset_type = 'RamanSpectral'  # Dataset name

# Training data pipelines
train_pipeline = [
    dict(type='LoadDataFromFile', id='ID', labels=['labels1', 'labels2', 'labels3']),
    # dict(type='RemoveBaseline', roi=[[-29, 4090]], method='drPLS', lam=10 ** 5, p=0.05),
    # dict(type='Smoothing', method="whittaker", window_length=5, polyorder=2),
    dict(type='Normalize', method='z-score'),  # Normalization
    dict(type='DataToFloatTensor', keys=['spectrum']),  # data is converted to torch.Tensor
    dict(type='ToTensor', keys=['labels']),  # labels are converted to torch.Tensor
]

# Test data pipeline
test_pipeline = [
    dict(type='LoadDataFromFile', id='ID', labels=['labels1', 'labels2', 'labels3']),
    # dict(type='RemoveBaseline', roi=[[-29, 4090]], method='drPLS', lam=10 ** 5, p=0.05),
    # dict(type='Smoothing', method="whittaker", window_length=5, polyorder=2),
    dict(type='Normalize', method='z-score'),
    dict(type='DataToFloatTensor', keys=['spectrum']),
]

data = dict(
    samples_per_gpu=64,  # Batch size for a single GPU
    workers_per_gpu=2,  # Number of threads per GPU
    train=dict(
        type=dataset_type,
        data_size=(0, 0.7),
        file_path='data/oral_tnm/oral_tnm_data.csv',
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_size=(0.7, 0.9),
        file_path='data/oral_tnm/oral_tnm_data.csv',
        pipeline=test_pipeline,
        test_mode=True
    ),
    test=dict(
        type=dataset_type,
        data_size=(0, 1),
        file_path='data/oral_tnm/oral_tnm_data.csv',
        pipeline=test_pipeline,
        test_mode=True
    )
)

evaluation = dict(  # Calculate accuracy
    interval=1,
    metric=['precision', 'f1_score'],
    metric_options={'topk': (1,)},
    save_best="auto",
    start=1
)
