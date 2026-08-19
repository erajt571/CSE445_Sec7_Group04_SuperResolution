# Custom evaluation set

This folder contains an additional evaluation-only image set used to test how the trained SRCNN behaves on content that is visually different from the original project images.

- `source_images/` - original custom source images.
- `HR_256/` - standardized 256x256 reference images.
- `LR_x4/` - 64x64 low-resolution inputs created with bicubic downsampling and a light Gaussian blur.

These images are not used to train or validate the model. They are a stress-test set only. The corresponding Bicubic/SRCNN comparison outputs are stored in `support/new_results/`.
