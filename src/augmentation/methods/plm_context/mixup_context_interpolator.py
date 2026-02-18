"""Context-Aware Interpolator (wrapper around MixupT5XLInterpolator)."""

from src.augmentation.methods.plm.mixup_t5_xl_interpolator import MixupT5XLInterpolator
import torch

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
        # We need access to the underlying encode/decode logic.
        # Since MixupT5XLInterpolator doesn't expose context in its interpolate_pair,
        # we have to reconstruct the prompt here and call generate directly if possible,
        # OR override the encoding step.
        
        # MixupT5XLInterpolator.interpolate_pair does:
        # 1. format input prompt
        # 2. encode
        # 3. mix embeddings
        # 4. generate
        
        # We need to inject context into step 1.
        
        # Let's override the input formatting logic by passing the FULL prompt as 'val'
        # if the base class allows flexible inputs.
        
        # The base class constructs prompt: f"generate variation <{predicate}>: {text}"
        # We want: f"context: {ctx} | generate variation <{predicate}>: {text}"
        
        # HACK: We can "trick" the base class if we can't modify it.
        # But base class formats strict strings.
        # We must reimplement interpolate_pair logic here.
        
        self.model.eval()
        
        # Clean predicate
        if "/" in predicate:
            p_name = predicate.split("/")[-1].split("#")[-1].lower()
        else:
            p_name = predicate.lower()
            
        # Format prompts with context
        prompt1 = f"context: {context1} | generate variation <{p_name}>: {val1}"
        prompt2 = f"context: {context2} | generate variation <{p_name}>: {val2}"
        
        # Encode
        # Note: We access self.tokenizer and self.model from parent
        with torch.no_grad():
            # Encoder inputs
            enc1 = self.tokenizer(prompt1, return_tensors="pt", max_length=self.max_len_in, truncation=True).input_ids.to(self.device)
            enc2 = self.tokenizer(prompt2, return_tensors="pt", max_length=self.max_len_in, truncation=True).input_ids.to(self.device)
            
            # Get embeddings
            emb1 = self.model.encoder(input_ids=enc1).last_hidden_state
            emb2 = self.model.encoder(input_ids=enc2).last_hidden_state
            
            # Mixup (same logic as base)
            # Align lengths
            len1, len2 = emb1.size(1), emb2.size(1)
            target_len = max(len1, len2)
            
            emb1_pad = self._pad_embeddings(emb1, target_len)
            emb2_pad = self._pad_embeddings(emb2, target_len)
            
            # Add noise
            noise1 = torch.randn_like(emb1_pad) * self.latent_noise_std
            noise2 = torch.randn_like(emb2_pad) * self.latent_noise_std
            
            h1_prime = (1 - alpha) * emb1_pad + alpha * emb2_pad + noise1
            h2_prime = alpha * emb1_pad + (1 - alpha) * emb2_pad + noise2
            
            # Decode
            # We need to construct dummy encoder outputs for generate
            # This relies on T5's generate accepting encoder_outputs
            
            from transformers.modeling_outputs import BaseModelOutput
            
            out1 = self.model.generate(
                encoder_outputs=BaseModelOutput(last_hidden_state=h1_prime),
                do_sample=True,
                temperature=self.gen_temperature,
                max_length=64,
                top_p=0.9
            )
            
            out2 = self.model.generate(
                encoder_outputs=BaseModelOutput(last_hidden_state=h2_prime),
                do_sample=True,
                temperature=self.gen_temperature,
                max_length=64,
                top_p=0.9
            )
            
            res1 = self.tokenizer.decode(out1[0], skip_special_tokens=True)
            res2 = self.tokenizer.decode(out2[0], skip_special_tokens=True)
            
            return res1, res2

    def _pad_embeddings(self, emb, target_len):
        """Helper for padding (copied from base logic if not protected)."""
        curr_len = emb.size(1)
        if curr_len < target_len:
            pad_size = target_len - curr_len
            # Pad with zeros
            padding = torch.zeros(emb.size(0), pad_size, emb.size(2), device=emb.device)
            return torch.cat([emb, padding], dim=1)
        return emb
