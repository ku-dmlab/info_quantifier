import logging
from typing import Any, Dict, Iterator, List

import torch
from torch import nn

from transformers import MarianTokenizer, MarianMTModel

logger = logging.getLogger(__name__)


class MarianInterface(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        src = cfg.source_lang
        trg = cfg.target_lang
        
        self.tokenizer = MarianTokenizer.from_pretrained(cfg.hf_model_name)
        self.model = MarianMTModel.from_pretrained(cfg.hf_model_name)
        
        self.decoder = self.model.get_decoder()
        self.decoder.embed_dim = self.decoder.embed_tokens.embedding_dim
        self.encoder = self.model.get_decoder()

    @property
    def device(self):
        return self._float_tensor.device
    
    def cuda(self):
        self.model.cuda()

    def generate(
        self, 
        tokens: torch.Tensor,
        num_beams: int = 1,
        num_return_sequences: int = 1,
        output_hidden_states: bool = True,
    ) -> List[str]:
        gen_out = self.model.generate(
            input_ids=tokens,
            return_dict_in_generate=True, 
            output_hidden_states=output_hidden_states,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences
        )
        return gen_out

    def sample(
        self, sentences: List[str], beam: int = 1, verbose: bool = False, **kwargs
    ) -> List[str]:
        if isinstance(sentences, str):
            return self.sample([sentences], beam=beam, verbose=verbose, **kwargs)[0]
        tokenized_sents = self.encode(sentences)
        hyp_tokens = self.model.generate(
            **tokenized_sents, 
            return_dict_in_generate=True, 
            output_hidden_states=True
        )
        '''
        "sequences"
        "encoder_hidden_states"
        "decoder_hidden_states"
        '''
        
        hyp_strs = self.decode(hyp_tokens['sequences'])
        return {
            "sequences": hyp_tokens['sequences'],
            "hyp_strs": hyp_strs,
        }

    def encode(self, sentences: List[str]) -> Dict[str, torch.Tensor]:
        return self.tokenizer(sentences, return_tensors='pt', padding=True)

    def decode(self, tokens: torch.Tensor) -> List[str]:
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
    
    @torch.no_grad()
    def get_src_features(self, tokens: torch.Tensor) -> torch.Tensor:
        encoder_out = self.encoder(tokens, output_hidden_states=True, return_dict=True)
        '''
        "last_hidden_state" B X T X D
        "hidden_states" [B X T X D] * Layers(7)
        '''
        return encoder_out["last_hidden_state"]

    @torch.no_grad()
    def get_trg_features(self, tokens: torch.Tensor) -> torch.Tensor:
        decoder_out = self.decoder(tokens, output_hidden_states=True, return_dict=True)
        '''
        "last_hidden_state" B X T X D
        "hidden_states" [B X T X D] * Layers(7)
        "past_key_values" 
        '''
        return decoder_out["last_hidden_state"]