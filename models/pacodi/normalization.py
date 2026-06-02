class InstanceNormalizationMixin:
    def instance_normalize(self, x, stats=None, inverse=False):
        if not inverse:
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.instance_norm_eps)
            return (x - mean) / std, (mean, std)

        if stats is None:
            return x
        mean, std = stats
        return x * std.to(x.device) + mean.to(x.device)
