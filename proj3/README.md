# Command Line Usage

main usage `python3 main.py <batch_size> <num_epochs> <decode_method> <feature_type>`

The possible values for `<decode_method>` are `greedy` and `ctc`

The possible values for `<feature_type>` are `discrete` and `mfcc`

For example, to train an ASR model with batch size 32, 50 epochs, ctc decoding and discrete features

`python3 main.py 32 50 ctc discrete`
