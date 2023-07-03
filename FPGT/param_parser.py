"""Parsing the parameters."""
import argparse

import torch

if torch.cuda.is_available():
    device = torch.device('cuda:2')
else:
    device = torch.device('cpu')


def parameter_parser():
    parser = argparse.ArgumentParser(description="Run FPGT.")

    # information settings
    parser.add_argument('--data-path', type=str, default='datasets/NYC-G/', help='data save parent path')
    parser.add_argument('--map-path', type=str, default='datasets/NYC-G/map/', help='map save parent path')
    parser.add_argument('--poi-features', type=str, default='datasets/NYC-G/map/poi_features2idx.csv')
    parser.add_argument('--user-map', type=str, default='datasets/NYC-G/map/user_id2idx_dict.csv')
    parser.add_argument('--poi-map', type=str, default='datasets/NYC-G/map/poi_id2idx_dict.csv')
    parser.add_argument('--hotness-map', type=str, default='datasets/NYC-G/map/hotness2idx_dict.csv')
    parser.add_argument('--region-map', type=str, default='datasets/NYC-G/map/region2idx_dict.csv')

    # dataset path settings
    parser.add_argument('--data-train', type=str, default='datasets/NYC-G/NYC_train.csv', help='training data path')
    parser.add_argument('--data-val', type=str, default='datasets/NYC-G/NYC_val.csv', help='Validation data path')
    parser.add_argument('--data-test', type=str, default='datasets/NYC-G/NYC_test.csv', help='test data path')


    # dimension settings
    parser.add_argument('--user-embed-dim', type=int, default=128, help='user embedding dimensions')
    parser.add_argument('--region-embed-dim', type=int, default=128, help='region embedding dimensions')
    parser.add_argument('--hotness-embed-dim', type=int, default=16, help='hotness embedding dimensions')
    parser.add_argument('--checkin-embed-dim', type=int, default=272, help='checkin embedding dimensions')

    # Transformer settings
    parser.add_argument('--hid', type=int, default=2048, help='Transformer hid dim')
    parser.add_argument('--layers', type=int, default=1, help='Num of TransformerLayer')
    parser.add_argument('--head', type=int, default=4, help='Num of heads in multi-head-attention')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate for transformer')

    # parameter settings
    parser.add_argument('--batch', type=int, default=20, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--lr-scheduler-factor', type=float, default=0.1, help='Learning rate scheduler factor')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--region_split', type=int, default=100, help='alpha')
    parser.add_argument('--hotness_split', type=int, default=10, help='beta')

    # other settings
    parser.add_argument('--save-weights', action='store_true', default=True, help='whether save the model')
    parser.add_argument('--save-embeds', action='store_true', default=False, help='whether save the embeddings')
    parser.add_argument('--store_code', default=True, help='whether save the code')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--r_name', default='exp', help='save to project/name')
    parser.add_argument('--r_not_cover', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=3407, help='Random seed')
    parser.add_argument('--device', type=str, default=device, help='Set running device')
    parser.add_argument('--early_stop', type=int, default=10, help='Shut when no improve over setting batch')
    parser.add_argument('--performance_check', type=int, default=30, help='Ensure the model at least train 30 epochs')

    return parser.parse_args()
