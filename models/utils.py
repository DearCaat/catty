import torch
from torch.utils.checkpoint import checkpoint
from itertools import chain
import numpy as np

def patchify(imgs,patch_size):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = patch_size[0] if type(patch_size) in (list,tuple) else patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

def unpatchify(x,patch_size):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """

    p = patch_size[0] if type(patch_size) in (list,tuple) else patch_size
    h = w = int(x.shape[1]**.5)
    assert h * w == x.shape[1]
    
    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    return imgs

# concat the impl of simmim and mae
class MaskGenerator:
    def __init__(self,input_size=224, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6,use_mae=False):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        self.use_mae = use_mae

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = mask_patch_size // model_patch_size
        self.mask_token_count = int(self.rand_size ** 2)
        self.mask_count = int(np.ceil(self.mask_token_count * mask_ratio))
        self.keep_count = self.mask_token_count - self.mask_count
    def __call__(self,x,mask_token=None):
        N, L, D = x.shape

        noise = torch.rand(N,self.mask_token_count,device=x.device)

        ids_shuffle = torch.argsort(noise,dim=1)
        ids_restore = torch.argsort(ids_shuffle,dim=1)
        # ids_keep = ids_shuffle[:,:self.keep_count]

        mask = torch.ones([N, self.mask_token_count],device=x.device)
        mask[:, :self.keep_count] = 0

        mask = torch.gather(mask, dim=1, index=ids_restore)

        mask = mask.view((-1,self.rand_size,self.rand_size))
        mask = mask.repeat_interleave(self.scale,1).repeat_interleave(self.scale,2).contiguous()

        if self.use_mae:
            x_masked = x[(mask == 0).flatten(1)].view(N,-1,D)
        else:
            mask_token = mask_token.expand(N, L, -1)
            w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)

            x_masked = x * (1 - w) + mask_token * w
        return x_masked,mask

def random_masking(x,mask_token,mask_patch_size,model_patch_size,mask_ratio,use_mae):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence

    original copyright/license mae, thanks to the authors
    """
    N, L, D = x.shape  # batch, length, dim
    print(x.size())
    # for block-random mask, impl from simmim
    rand_size = L ** .5
    scale = mask_patch_size // model_patch_size
    token_count = L
    mask_count = int(np.ceil(token_count * mask_ratio))
    
    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
    
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :mask_count]

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :mask_count] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    # the mae only fed the unmasked patch to encoder, but simmim use the all.
    if use_mae:
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    else:
        mask_token = mask_token.expand(N, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
        x_masked = x * (1 - w) + mask_token * w

    return x_masked, mask, ids_restore

# copy from timm https://github.com/rwightman/pytorch-image-models/blob/9e12530433f38c536e9a5fdfe5c9f455638a3e8a/timm/models/helpers.py#L690
# the timm==0.5.4 doesn't have this function
def checkpoint_seq(
        functions,
        x,
        every=1,
        flatten=False,
        skip_last=False,
        preserve_rng_state=True
):
    r"""A helper function for checkpointing sequential models.
    Sequential models execute a list of modules/functions in order
    (sequentially). Therefore, we can divide such a sequence into segments
    and checkpoint each segment. All segments except run in :func:`torch.no_grad`
    manner, i.e., not storing the intermediate activations. The inputs of each
    checkpointed segment will be saved for re-running the segment in the backward pass.
    See :func:`~torch.utils.checkpoint.checkpoint` on how checkpointing works.
    .. warning::
        Checkpointing currently only supports :func:`torch.autograd.backward`
        and only if its `inputs` argument is not passed. :func:`torch.autograd.grad`
        is not supported.
    .. warning:
        At least one of the inputs needs to have :code:`requires_grad=True` if
        grads are needed for model inputs, otherwise the checkpointed part of the
        model won't have gradients.
    Args:
        functions: A :class:`torch.nn.Sequential` or the list of modules or functions to run sequentially.
        x: A Tensor that is input to :attr:`functions`
        every: checkpoint every-n functions (default: 1)
        flatten (bool): flatten nn.Sequential of nn.Sequentials
        skip_last (bool): skip checkpointing the last function in the sequence if True
        preserve_rng_state (bool, optional, default=True):  Omit stashing and restoring
            the RNG state during each checkpoint.
    Returns:
        Output of running :attr:`functions` sequentially on :attr:`*inputs`
    Example:
        >>> model = nn.Sequential(...)
        >>> input_var = checkpoint_seq(model, input_var, every=2)
    """
    def run_function(start, end, functions):
        def forward(_x):
            for j in range(start, end + 1):
                _x = functions[j](_x)
            return _x
        return forward

    if isinstance(functions, torch.nn.Sequential):
        functions = functions.children()
    if flatten:
        functions = chain.from_iterable(functions)
    if not isinstance(functions, (tuple, list)):
        functions = tuple(functions)

    num_checkpointed = len(functions)
    if skip_last:
        num_checkpointed -= 1
    end = -1
    for start in range(0, num_checkpointed, every):
        end = min(start + every - 1, num_checkpointed - 1)
        x = checkpoint(run_function(start, end, functions), x, preserve_rng_state=preserve_rng_state)
    if skip_last:
        return run_function(end + 1, len(functions) - 1, functions)(x)
    return x