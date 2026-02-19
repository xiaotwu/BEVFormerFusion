# compat_registries.py — minimal CustomBaseTransformerLayer that forwards extra kwargs
from mmcv.cnn.bricks.transformer import BaseTransformerLayer, TRANSFORMER_LAYER
import torch

from mmcv.cnn.bricks.transformer import BaseTransformerLayer, TRANSFORMER_LAYER
import torch

@TRANSFORMER_LAYER.register_module()
class CustomBaseTransformerLayer(BaseTransformerLayer):
    """Forwards kwargs through to attention modules and lets self_attn and cross_attn
    use different key/value sources:
      - self_attn (TSA):     from kwargs['tsa_key'], kwargs['tsa_value'] if provided
      - cross_attn (SCA):    from kwargs['sca_key'], kwargs['sca_value'] or top-level key/value
    """
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):

        # one-time debug
        if not hasattr(self, "_dbg_called_once"):
            print("[CBTL] CustomBaseTransformerLayer.forward called; "
                  f"kwargs keys: {list(kwargs.keys())}")
            self._dbg_called_once = True

        # always set identity if not provided
        if identity is None:
            identity = query

        attn_index = norm_index = ffn_index = 0

        for layer in self.operation_order:
            if layer == 'self_attn':
                # --- TSA gets its own K/V if provided ---
                k_self = kwargs.pop('tsa_key', None)
                v_self = kwargs.pop('tsa_value', None)

                # fallbacks: preserve explicit key/value if caller set them; else default to query
                if k_self is None:
                    k_self = key if key is not None else query
                if v_self is None:
                    v_self = value if value is not None else k_self

                attn_mask_i = (attn_masks[attn_index]
                               if (attn_masks is not None and len(attn_masks) > attn_index)
                               else None)

                query = self.attentions[attn_index](
                    query=query,
                    key=k_self,
                    value=v_self,
                    identity=identity,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_mask_i,
                    query_key_padding_mask=query_key_padding_mask,
                    key_padding_mask=query_key_padding_mask,
                    **kwargs  # includes prev_bev, bev_h, bev_w, ref_2d, etc.
                )
                attn_index += 1

                if not torch.isfinite(query).all():
                    print("[ENC/LAYER] attention output non-finite -> zeroing")
                    query = torch.nan_to_num(query, nan=0.0, posinf=0.0, neginf=0.0)

            elif layer == 'cross_attn':
                # --- SCA uses image features (either from kwargs or top-level key/value) ---
                k_cross = kwargs.get('sca_key', key)
                v_cross = kwargs.get('sca_value', value)

                attn_mask_i = (attn_masks[attn_index]
                               if (attn_masks is not None and len(attn_masks) > attn_index)
                               else None)

                query = self.attentions[attn_index](
                    query=query,
                    key=k_cross,
                    value=v_cross,
                    identity=identity,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_mask_i,
                    key_padding_mask=key_padding_mask,
                    **kwargs
                )
                attn_index += 1

                if not torch.isfinite(query).all():
                    print("[ENC/LAYER] attention output non-finite -> zeroing")
                    query = torch.nan_to_num(query, nan=0.0, posinf=0.0, neginf=0.0)

            elif layer == 'ffn':
                query = self.ffns[ffn_index](query, identity=identity)
                ffn_index += 1

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            else:
                raise ValueError(f'Unsupported layer type: {layer}')

            # residual for next sub-layer
            identity = query

        return query




