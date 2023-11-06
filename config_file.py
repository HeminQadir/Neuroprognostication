import ml_collections

def get_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': 1000})
    config.hidden_size = 768
    config.input_length = 30000

    config.window_size = 5  # in munites
    
    config.step_size = 3     # in munites
    config.num_classes = 2
    config.resampling_frequency  = 100

    config.in_channels = 2
    
    config.train_batch_size = 2
    config.eval_batch_size = 2
    config.learning_rate = 1e-4 
    config.num_steps = 20000
    config.eval_every = 500
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 2
    config.transformer.num_layers = 2
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config