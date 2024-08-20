# Optimizer Configuration
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.002)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=0.01,
    warmup_by_epoch=True)
runner = dict(type='EpochMultiRunner',  # The type of runner to be used
              max_epochs=300)           # Total number of runner rounds