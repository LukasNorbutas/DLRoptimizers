## dlr 

## Purpose

Keras optimizers that support discriminative learning rates. Optimizers are edited Keras optimizers that take an additional
input param_lrs, which is a dictionary that contains model's parameters and learning_rate multipliers.
Utils contain dlr.get_lr_multipliers function that creates such an input dictionary based on an input model and a learning rate.

NOTE: get_lr_multipliers() currently works with ImageLearner class (not keras.Model), that contains ImageLearner.base_model (class of transfer learning model used, e.g. keras.applications.Xception) and ImageLearner.model (final keras.Model, built on ImageLearner.base_model).

## Example

```python
  # Create a learner
  learner = ImageLearner(
      path=TEMP_DIR/"efficientnetb0_v1",
      data=data_container,
      base_model=efn.EfficientNetB0,
      input_shape=IMG_DIMS,
      dropout=0.5,
      l1=3e-5,
      l2=3e-4,
      load=True,
  )
  
  # Create a parameter: lr_multiplier dictionary
  my_dict = dlr.get_lr_multipliers(learner, (1e-4,), params=True)
  print(my_dict)
```
```
  > {'conv1_conv_11/kernel:0': 0.3, 'conv1_conv_11/bias:0': 0.3, 'conv1_bn_11/gamma:0': 0.3,
    [...]
    batch_normalization_11/moving_variance:0': 1, 'dense_11/kernel:0': 1, 'dense_11/bias:0': 1}
```
```python
  # Pass the dictionary to a DLR optimizer 
  learner.model.compile(optimizer=dlr.DLR_Adam(learning_rate=1e-4, param_lrs=my_dict),
                       loss=keras.losses.sparse_categorical_crossentropy,
                       metrics=[keras.metrics.sparse_categorical_accuracy])
```
