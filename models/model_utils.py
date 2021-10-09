from models.pytorch_i3d import InceptionI3d


def setup_inception3d(num_class):
    return InceptionI3d(num_class, in_channels=3)


def setup_model(args, num_class, pretrain_ckpt=None, default_ckpt=None):
    if pretrain_ckpt:
        model = setup_inception3d(num_class)

        model.load_state_dict(pretrain_ckpt['ckpt'])
    else:
        model = setup_inception3d(2000)
        if pretrain_ckpt:
            model.load_state_dict(default_ckpt)

        model.replace_logits(num_class)

    return model
