import torch
import torch.nn.functional as F


def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adain(content_feat, style_feat, alpha=1.0):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    stylized_feat = normalized_feat * style_std.expand(size) + style_mean.expand(size)
    blended_feat = alpha * stylized_feat + (1.0 - alpha) * content_feat
    return blended_feat


def calc_content_loss(gen_feat, target_feat):
    return F.mse_loss(gen_feat, target_feat)


def calc_style_loss(gen_feats, style_feats):
    style_loss = torch.tensor(0.0, dtype=torch.float32, device=gen_feats[0].device)
    for gen_feat, style_feat in zip(gen_feats, style_feats):
        gen_mean, gen_std = calc_mean_std(gen_feat)
        style_mean, style_std = calc_mean_std(style_feat)
        style_loss += F.mse_loss(gen_mean, style_mean) + F.mse_loss(gen_std, style_std)
    return style_loss