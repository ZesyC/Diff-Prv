import argparse
import ast


MODEL_NAME = 'CFM'
_MODEL_ALIASES = {
    'cfm': MODEL_NAME,
    'optimized': MODEL_NAME,
    'flowmatching_optimized': MODEL_NAME,
}


def _parse_dims(value):
    if isinstance(value, list):
        dims = value
    else:
        try:
            dims = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise argparse.ArgumentTypeError("--dims must look like '[1000]'") from exc

    if not isinstance(dims, list) or not all(isinstance(dim, int) and dim > 0 for dim in dims):
        raise argparse.ArgumentTypeError('--dims must be a list of positive integers')
    return dims


def _parse_bool(value):
    if isinstance(value, bool):
        return value

    value = value.lower()
    if value in {'1', 'true', 't', 'yes', 'y'}:
        return True
    if value in {'0', 'false', 'f', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError('boolean value expected')


def ParseArgs():
    parser = argparse.ArgumentParser(description='CFM model params')

    parser.add_argument('--data', default='baby', choices=['baby', 'tiktok', 'sports', 'sport'], type=str, help='dataset name')
    parser.add_argument('--epoch', default=50, type=int, help='number of epochs')
    parser.add_argument('--gpu', default='0', type=str, help='GPU id')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch', default=1024, type=int, help='batch size')
    parser.add_argument('--tstBat', default=256, type=int, help='number of users in a testing batch')
    parser.add_argument('--reg', default=1e-5, type=float, help='weight decay regularizer')
    parser.add_argument('--latdim', default=64, type=int, help='embedding size')
    parser.add_argument('--gnn_layer', default=1, type=int, help='number of GNN layers')
    parser.add_argument('--topk', default=20, type=int, help='K of top-K recommendation')
    parser.add_argument('--model_type', default=MODEL_NAME, type=str, help='CFM optimized model. Aliases: optimized, flowmatching_optimized')
    parser.add_argument('--ssl_reg', default=1e-1, type=float, help='contrastive learning weight')
    parser.add_argument('--temp', default=0.5, type=float, help='temperature in contrastive learning')
    parser.add_argument('--tstEpoch', default=1, type=int, help='test frequency while training')
    parser.add_argument('--seed', type=int, default=421, help='random seed')

    parser.add_argument('--keepRate', default=1.0, type=float, help='ratio of graph edges to keep')
    parser.add_argument('--dims', type=_parse_dims, default=[1000], help="velocity MLP hidden dims, e.g. '[1000]'")
    parser.add_argument('--d_emb_size', type=int, default=10, help='time embedding size')
    parser.add_argument('--norm', type=_parse_bool, default=False, help='normalize CFM input vectors')
    parser.add_argument('--steps', type=int, default=5, help='Euler solver steps for CFM sampling')

    parser.add_argument('--noise_scale', type=float, default=0.1)
    parser.add_argument('--noise_min', type=float, default=0.0001)
    parser.add_argument('--noise_max', type=float, default=0.02)
    parser.add_argument('--sampling_steps', type=int, default=0)

    parser.add_argument('--rebuild_k', type=int, default=1, help='top-k edges rebuilt per user/modal')
    parser.add_argument('--e_loss', type=float, default=0.1, help='MSI loss weight')
    parser.add_argument('--ris_lambda', type=float, default=0.5, help='residual item semantic weight')
    parser.add_argument('--ris_adj_lambda', type=float, default=0.2, help='rebuilt adjacency residual weight')
    parser.add_argument('--trans', type=int, default=0, choices=[0, 1, 2], help='0: matrix projection, 1: linear, 2: mixed')
    parser.add_argument('--cl_method', type=int, default=0, choices=[0, 1], help='0: modal-vs-modal, 1: modal-vs-main')
    parser.add_argument('--gate_dim', type=int, default=32, help='hidden dim of ModalGatingNetwork')
    parser.add_argument('--gate_reg', type=float, default=0.0, help='entropy regularization weight for gating')
    parser.add_argument('--modal_cond', type=int, default=1, choices=[0, 1], help='1: enable modal-conditioned CFM, 0: disable')
    parser.add_argument('--cfm_lambda', type=float, default=0.1, help='contrastive flow matching weight')
    parser.add_argument('--cross_fm_weight', type=float, default=0.01, help='cross-modal flow matching alignment weight')

    parsed = parser.parse_args()
    if parsed.data == 'sport':
        parsed.data = 'sports'

    model_type = _MODEL_ALIASES.get(parsed.model_type.lower())
    if model_type is None:
        parser.error("--model_type only supports CFM/optimized. Old alias 'flowmatching_optimized' is still accepted.")
    parsed.model_type = model_type
    return parsed


args = ParseArgs()
