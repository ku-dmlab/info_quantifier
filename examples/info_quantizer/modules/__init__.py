from .transformer_layers import TransformerEncoderLayer, TransformerDecoderLayer

import importlib
import os

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        importlib.import_module('examples.info_quantizer.modules.' + module_name)