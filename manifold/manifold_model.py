# Reference to https://github.com/johannbrehmer/manifold-flow

import torch.nn as nn
import torch
import glog as logger
from torch.nn import functional as F

from manifold.various import product, create_alternating_binary_mask
from manifold.distributions import StandardNormal, RescaledNormal
from manifold import transforms
import manifold.nn as nn_
from manifold.image_transforms import create_image_transform


class BaseFlow(nn.Module):
    """ Abstract base flow class """

    def forward(self, x, context=None):
        raise NotImplementedError

    def encode(self, x, context=None):
        raise NotImplementedError

    def decode(self, u, context=None):
        raise NotImplementedError

    def project(self, x, context=None):
        return self.decode(self.encode(x, context), context)

    def log_prob(self, x, context=None):
        raise NotImplementedError

    def sample(self, u=None, n=1, context=None):
        raise NotImplementedError

    def _report_model_parameters(self):
        """ Reports the model size """

        all_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        size = all_params * (32 / 8)  # Bytes
        logger.info("Model has %.1f M parameters (%.1f M trainable) with an estimated size of %.1f MB", all_params / 1e6, trainable_params / 1.0e6, size / 1.0e6)


def create_vector_transform(device, x_dim, flow_steps, coupling_type, spline_bin, spline_bound, hidden_features, context_features=None, dropout_probability=0, use_batch_norm=False, LU_linear=True):        
    if LU_linear:
        _create_vector_linear_transform = lambda: transforms.CompositeTransform([transforms.RandomPermutation(device, features=x_dim), transforms.LULinear(x_dim, identity_init=True)], device=device)
    else:
        _create_vector_linear_transform = lambda: transforms.CompositeTransform([transforms.RandomPermutation(device, features=x_dim)], device=device)
    
    transform_net_create_fn = lambda in_features, out_features: nn_.ResidualNet(
        in_features=in_features,
        out_features=out_features,
        hidden_features=hidden_features,
        context_features=context_features,
        num_blocks=2,
        activation=F.relu,
        dropout_probability=dropout_probability,
        use_batch_norm=use_batch_norm,
    )

    if coupling_type=='rational_quadratic_spline':
        coupling_create_fn = lambda i: transforms.PiecewiseRationalQuadraticCouplingTransform(
                            device=device,
                            mask=create_alternating_binary_mask(x_dim, even=(i % 2 == 0)),
                            transform_net_create_fn=transform_net_create_fn,
                            num_bins=spline_bin,
                            tails="linear",
                            tail_bound=spline_bound,
                            apply_unconditional_transform=False,
                        )
    elif coupling_type=='affine':

        coupling_create_fn = lambda i: transforms.AffineCouplingTransform(
                            device=device,
                            mask=create_alternating_binary_mask(x_dim, even=(i % 2 == 0)),
                            transform_net_create_fn=transform_net_create_fn,
                        )

    transform = transforms.CompositeTransform(
        [
            transforms.CompositeTransform(
                [
                    _create_vector_linear_transform(),
                    coupling_create_fn(i)
                ], device=device
            )
            for i in range(flow_steps)
        ]
        + [_create_vector_linear_transform()], device=device
    )

    return transform

class ManifoldFlow(BaseFlow):
    """ Manifold-based flow (base class for FOM, M-flow, PIE) """

    def __init__(self, device, args, data_vector_len, latent_dim, condition_dim, pie_epsilon=1.0e-2, apply_context_to_outer=True, clip_pie=False):
        super(ManifoldFlow, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.apply_context_to_outer = apply_context_to_outer
        self.data_vector_len = data_vector_len
        self.total_latent_dim = product(latent_dim)

        assert self.total_latent_dim < self.data_vector_len

        self.manifold_latent_distribution = StandardNormal((self.total_latent_dim,))
        self.orthogonal_latent_distribution = RescaledNormal(
            (self.data_vector_len - self.total_latent_dim,), std=pie_epsilon, clip=None if not clip_pie else clip_pie * pie_epsilon
        )

        if args.outer_image:
            self.outer_transform = create_image_transform(
                                    device,
                                    args.img_size[0],
                                    args.img_size[1],
                                    args.img_size[2],
                                    levels=args.levels,
                                    hidden_channels=100,
                                    steps_per_level=args.outerlayers//args.levels,
                                    num_res_blocks=2,
                                    alpha=0.05,
                                    num_bits=8,
                                    preprocessing="glow",
                                    dropout_prob=args.dropout,
                                    multi_scale=True,
                                    tail_bound=args.splinerange,
                                    num_bins=args.splinebins,
                                    coupling_layer_type=args.coupling_type,
                                    postprocessing="partial_nsf" if args.intermediatensf else "partial_mlp",
                                    postprocessing_layers=args.linlayers,
                                    postprocessing_channel_factor=args.linchannelfactor,
                                    use_actnorm=args.actnorm,
                                    use_batchnorm=args.batchnorm,
                                    context_features=condition_dim if apply_context_to_outer else None,
                                )
        else:
            self.outer_transform = create_vector_transform(device, x_dim=self.data_vector_len, flow_steps=args.outerlayers, coupling_type=args.coupling_type,
                                                        spline_bin=args.splinebins, spline_bound=args.splinerange, hidden_features=args.hidden_features,
                                                        context_features=condition_dim if apply_context_to_outer else None,
                                                        dropout_probability=args.dropout, use_batch_norm=args.dropout,
                                                        LU_linear=args.LU_linear)
        self.projection = transforms.ProjectionSplit(device, self.data_vector_len, self.total_latent_dim)
        self.inner_transform = create_vector_transform(device, x_dim=self.total_latent_dim, flow_steps=args.innerlayers, coupling_type=args.coupling_type,
                                                       spline_bin=args.splinebins, spline_bound=args.splinerange, hidden_features=args.hidden_features,
                                                       context_features=condition_dim,
                                                       dropout_probability=args.dropout, use_batch_norm=args.dropout,
                                                       LU_linear=args.LU_linear)

        self._report_model_parameters()

    def forward(self, x, mode="mf", context=None, return_hidden=False):
        """
        Transforms data point to latent space, evaluates likelihood, and transforms it back to data space.

        mode can be "mf" (calculating the exact manifold density based on the full Jacobian), "pie" (calculating the density in x), "slice"
        (calculating the density on x, but projected onto the manifold), or "projection" (calculating no density at all).
        """

        assert mode in ["mf", "pie", "slice", "projection", "pie-inv", "mf-fixed-manifold"]

        if mode == "mf" and not x.requires_grad:
            x.requires_grad = True

        # Encode
        u, h_manifold, h_orthogonal, log_det_outer, log_det_inner = self._encode(x, context)

        # Decode
        x_reco, inv_log_det_inner, inv_log_det_outer, inv_jacobian_outer, h_manifold_reco = self._decode(u, mode=mode, context=context)

        # Log prob
        log_prob = self._log_prob(mode, u, h_orthogonal, log_det_inner, log_det_outer, inv_log_det_inner, inv_log_det_outer, inv_jacobian_outer)

        if return_hidden:
            return x_reco, log_prob, u, torch.cat((h_manifold, h_orthogonal), -1)
        return x_reco, log_prob, u

    def encode(self, x, context=None):
        """ Transforms data point to latent space. """

        u, _, _, _, _ = self._encode(x, context=context)
        return u

    def decode(self, u, u_orthogonal=None, context=None):
        """ Decodes latent variable to data space."""

        x, _, _, _, _ = self._decode(u, mode="projection", u_orthogonal=u_orthogonal, context=context)
        return x

    def log_prob(self, x, mode="mf", context=None):
        """ Evaluates log likelihood for given data point."""

        return self.forward(x, mode, context)[1]

    def sample(self, u=None, n=1, context=None, sample_orthogonal=False):
        """
        Generates samples from model.

        Note: this is PIE / MF sampling! Cannot sample from slice of PIE efficiently.
        """

        if u is None:
            u = self.manifold_latent_distribution.sample(n, context=None).to(self.device)
        u_orthogonal = self.orthogonal_latent_distribution.sample(n, context=None).to(self.device) if sample_orthogonal else None
        x = self.decode(u, u_orthogonal=u_orthogonal, context=context)
        return x

    def _encode(self, x, context=None):
        # Encode
        h, log_det_outer = self.outer_transform(x, full_jacobian=False, context=context if self.apply_context_to_outer else None)

        h_manifold, h_orthogonal = self.projection(h)
        u, log_det_inner = self.inner_transform(h_manifold, full_jacobian=False, context=context)

        return u, h_manifold, h_orthogonal, log_det_outer, log_det_inner

    def _decode(self, u, mode, u_orthogonal=None, context=None):
        if mode == "mf" and not u.requires_grad:
            u.requires_grad = True

        h, inv_log_det_inner = self.inner_transform.inverse(u, full_jacobian=False, context=context)

        x, inv_log_det_outer, inv_jacobian_outer, h = self._decode_h(h, mode, u_orthogonal, context)

        return x, inv_log_det_inner, inv_log_det_outer, inv_jacobian_outer, h

    def _decode_h(self, h_manifold, mode, h_orthogonal=None, context=None):
        if h_orthogonal is not None:
            h = self.projection.inverse(h_manifold, orthogonal_inputs=h_orthogonal)
        else:
            h = self.projection.inverse(h_manifold)

        if mode in ["pie", "slice", "projection", "mf-fixed-manifold"]:
            x, inv_log_det_outer = self.outer_transform.inverse(h, full_jacobian=False, context=context if self.apply_context_to_outer else None)
            inv_jacobian_outer = None
        else:
            x, inv_jacobian_outer = self.outer_transform.inverse(h, full_jacobian=True, context=context if self.apply_context_to_outer else None)
            inv_log_det_outer = None

        return x, inv_log_det_outer, inv_jacobian_outer, h

    def _log_prob(self, mode, u, h_orthogonal, log_det_inner, log_det_outer, inv_log_det_inner, inv_log_det_outer, inv_jacobian_outer):
        if mode == "pie":
            log_prob = self.manifold_latent_distribution._log_prob(u, context=None)
            log_prob = log_prob + self.orthogonal_latent_distribution._log_prob(h_orthogonal, context=None)
            log_prob = log_prob + log_det_outer + log_det_inner

        elif mode == "pie-inv":
            log_prob = self.manifold_latent_distribution._log_prob(u, context=None)
            log_prob = log_prob + self.orthogonal_latent_distribution._log_prob(h_orthogonal, context=None)
            log_prob = log_prob - inv_log_det_outer - inv_log_det_inner

        elif mode == "slice":
            log_prob = self.manifold_latent_distribution._log_prob(u, context=None)
            log_prob = log_prob + self.orthogonal_latent_distribution._log_prob(torch.zeros_like(h_orthogonal), context=None)
            log_prob = log_prob - inv_log_det_outer - inv_log_det_inner

        elif mode == "mf":
            # inv_jacobian_outer is dx / du, but still need to restrict this to the manifold latents
            inv_jacobian_outer = inv_jacobian_outer[:, :, : self.latent_dim]
            # And finally calculate log det (J^T J)
            jtj = torch.bmm(torch.transpose(inv_jacobian_outer, -2, -1), inv_jacobian_outer)

            log_prob = self.manifold_latent_distribution._log_prob(u, context=None)
            log_prob = log_prob - 0.5 * torch.slogdet(jtj)[1] - inv_log_det_inner

        elif mode == "mf-fixed-manifold":
            log_prob = self.manifold_latent_distribution._log_prob(u, context=None)
            log_prob = log_prob + self.orthogonal_latent_distribution._log_prob(h_orthogonal, context=None)
            log_prob = log_prob + log_det_outer + log_det_inner

        else:
            log_prob = None

        return log_prob

    def _report_model_parameters(self):
        """ Reports the model size """
        super()._report_model_parameters()
        inner_params = sum(p.numel() for p in self.inner_transform.parameters())
        outer_params = sum(p.numel() for p in self.outer_transform.parameters())
        logger.info("  Outer transform: %.1f M parameters", outer_params / 1.0e06)
        logger.info("  Inner transform: %.1f M parameters", inner_params / 1.0e06)
