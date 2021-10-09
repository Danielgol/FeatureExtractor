import argparse


def get_options():
    parser = argparse.ArgumentParser(description='Training backbone.')

    # path
    parser.add_argument('--root_cslr', type=str, required=True, help='the path to phoenix-2014 video dir.')
    parser.add_argument('--root_slt', type=str, required=True, help='the path to phoenix-2014-T video dir.')
    parser.add_argument('--save_ckpt_path', type=str, default=None, required=True, help='the path to save checkpoints.')
    parser.add_argument('--train_split', type=str, default=None, required=True, help='the path to train split file.')
    parser.add_argument('--test_split', type=str, default=None, required=True, help='the path to test split file.')
    parser.add_argument('--vocab_file', type=str, default=None, required=True, help='the path to vocabulary file.')
    parser.add_argument('--default_ckpt', type=str, default=None, required=True, help='the path to the initial i3d model from WLASL and MSASL.')
    parser.add_argument('--pretrain_ckpt', type=str, default=None, required=False, help='the path to the finetuned i3d checkpoints.')

    # network
    parser.add_argument('--num_class', default=1208, type=int, help='the number of classes / labels.')

    # dataset
    parser.add_argument('--is_balance', action='store_true', help='if specified, use resampling to balance the dataset.')

    # optimization
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate.')
    parser.add_argument('--weight_decay', default=1e-8, type=float, help='weight decay')
    parser.add_argument('--patience', default=5, type=int, help='the number of epochs before decreasing learning rate.')
    parser.add_argument('--lr_factor', default=0.3, type=float, help='the rate to decrease learning rate')
    parser.add_argument('--max_steps', default=1e7, type=float, help='the maximal number of steps.')

    # hardware-related
    parser.add_argument('--max_frames', default=400, type=int, help='the maximal number of frames in a batch. For a sole I3D model, ~200 frames per 11 GB GPU memory.')
    parser.add_argument('--num_workers', default=12, type=int, help='the number of processes to read data.')
    parser.add_argument('--pin_memory', action='store_true', help='if specified, use pin_memory=True in the dataloader.')

    # logging
    parser.add_argument('--logging_interval', default=100, type=int, help='logging training per N iterations.')
    parser.add_argument('--logging_root', default='./train_log', type=str, help='directory to save logging file.')

    # evaluation
    parser.add_argument('--eval_on_the_fly', action='store_true', help='if specified, run validation during training.')

    return parser.parse_args()