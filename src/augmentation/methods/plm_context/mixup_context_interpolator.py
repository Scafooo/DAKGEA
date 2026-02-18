"""Context-Aware Interpolator (wrapper around MixupT5XLInterpolator)."""

from src.augmentation.methods.plm.mixup_t5_xl_interpolator import MixupT5XLInterpolator
from transformers.modeling_outputs import BaseModelOutput
import torch
import re

class MixupContextInterpolator(MixupT5XLInterpolator):
    """Interpolator that handles context-injected prompts."""

    def interpolate_pair(
        self,
        val1: str,
        val2: str,
        predicate: str = "attribute",
        alpha: float = 0.5,
        context1: str = "generic",
        context2: str = "generic",
    ) -> tuple[str, str]:
        """Interpolate between two values with CONTEXT.
        
        Args:
            val1: First value
            val2: Second value
            predicate: Predicate name
            alpha: Mixing ratio
            context1: Context string for val1
            context2: Context string for val2
            
        Returns:
            (generated_from_1, generated_from_2)
        """
        self.model.eval()
        
        # Clean predicate
        p_name = predicate.replace("<", "").replace(">", "").lower().replace('_', ' ')
            
        # Format prompts with context
        prompt1 = f"context: {context1} | generate variation <{p_name}>: {val1}"
        prompt2 = f"context: {context2} | generate variation <{p_name}>: {val2}"
        
        # Tokenize
        inputs = self.tokenizer([prompt1, prompt2], return_tensors="pt", padding=True, truncation=True, max_length=self.max_len_in).to(self.device)
        
        with torch.no_grad():
            # Get encoder
            # For PeftModel, we go down to the base model
            encoder = self.model.base_model.model.get_encoder()
            enc_out = encoder(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
            
            H_A, H_B = enc_out.last_hidden_state[0:1], enc_out.last_hidden_state[1:2]
            
            # Mixup
            H_mix_A = (1.0 - alpha) * H_A + alpha * H_B
            H_mix_B = alpha * H_A + (1.0 - alpha) * H_B
            
            if self.latent_noise_std > 0:
                H_mix_A += torch.randn_like(H_mix_A) * self.latent_noise_std
                H_mix_B += torch.randn_like(H_mix_B) * self.latent_noise_std

            # Shared attention mask (OR of both masks)
            m_f = (inputs.attention_mask[0:1] | inputs.attention_mask[1:2]).repeat(2, 1)
            H_f = torch.cat([H_mix_A, H_mix_B], dim=0)
            
            # Generate
            out_ids = self.model.generate(
                encoder_outputs=BaseModelOutput(last_hidden_state=H_f),
                attention_mask=m_f,
                max_new_tokens=64,
                do_sample=True,
                temperature=self.gen_temperature,
                num_beams=self.gen_num_beams,
                repetition_penalty=self.gen_repetition_penalty,
                top_p=self.gen_top_p
            )

        res_a = self._clean_output(self.tokenizer.decode(out_ids[0], skip_special_tokens=True)) or val1
        res_b = self._clean_output(self.tokenizer.decode(out_ids[1], skip_special_tokens=True)) or val2
        
        return res_a, res_b