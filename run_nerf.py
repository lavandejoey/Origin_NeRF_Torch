import logging
import os
import time

import imageio
from tqdm import tqdm, trange

from load_LINEMOD import load_LINEMOD_data
from load_blender import load_blender_data
from load_deepvoxels import load_dv_data
from load_llff import load_llff_data
from run_nerf_helpers import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

# 配置日志记录器
log_file = "log-{}.txt".format(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)


def batchify(fn, chunk):
    """构建一个适用于较小批次的'fn'的版本。
    @:param: fn (function): 要应用的函数。
    @:param: chunk (int): 每个批次的大小。
    @:returns: function: 适用于较小批次的'fn'的版本。
    """
    if chunk is None:
        return fn

    def ret(inputs):
        """将输入张量分成小的minibatch以避免内存不足。"""
        # torch.cat()函数用于连接两个张量，torch.cat((tensor1,tensor2),dim=0)表示按照行的方向进行拼接，dim=1表示按照列的方向进行拼接
        return torch.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024 * 64):
    """准备输入并应用网络 'fn'。
    @:param: inputs (torch.Tensor): 输入张量，形状为 [batch_size, ... , input_ch]。
    @:param: viewdirs (torch.Tensor): 观察方向张量，形状为 [batch_size, ... , input_ch_views]。
    @:param: fn (torch.nn.Module): 要应用的网络模型。
    @:param: embed_fn (function): 将输入张量嵌入到特征空间中的函数。
    @:param: embeddirs_fn (function): 将观察方向张量嵌入到特征空间中的函数。
    @:param: netchunk (int): 网络处理数据块大小。
    @:returns: torch.Tensor: 经过网络 fn 处理后的输出张量，形状与输入张量相同。
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024 * 32, **kwargs):
    """将射线分成小的minibatch以避免内存不足。
    @:param: rays_flat (torch.Tensor): 扁平化的射线张量，形状为[N, 8]，N为射线数量，8表示射线起点、方向和长度。
    @:param: chunk (int): minibatch的大小，默认为1024*32。
    @:param: **kwargs: 传递给render_rays函数的其他参数。
    @:returns: dict: 包含所有渲染结果的字典，每个键值对为（名称，张量）。
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i + chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K, chunk=1024 * 32, rays=None, c2w=None, ndc=True,
           near=0., far=1., use_viewdirs=False, c2w_staticcam=None, **kwargs):
    """渲染光线
    @:param: H (int): 图像高度。
    @:param: W (int): 图像宽度。
    @:param: K (torch.Tensor): 相机内参矩阵，形状为[3, 3]。
    @:param: chunk (int): minibatch的大小，默认为1024*32。
    @:param: rays (torch.Tensor): 射线张量，形状为[2, batch_size, 3]，batch_size为射线数量，3表示射线起点、方向和长度。
    @:param: c2w (torch.Tensor): 相机到世界坐标系的变换矩阵，形状为[3, 4]。
    @:param: ndc (bool): 如果为True，则表示射线起点、方向在NDC坐标系中。
    @:param: near (float or torch.Tensor): 射线最近距离。
    @:param: far (float or torch.Tensor): 射线最远距离。
    @:param: use_viewdirs (bool): 如果为True，则使用空间中点的观察方向。
    @:param: c2w_staticcam (torch.Tensor): 如果不为None，则使用此变换矩阵。
    @:param: **kwargs: 传递给render_rays函数的其他参数。
    @:returns: dict: 包含所有渲染结果的字典，每个键值对为（名称，张量）。
    """

    if c2w is not None:
        # 特殊情况下渲染整个图像
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # 使用提供的射线批次
        rays_o, rays_d = rays

    if use_viewdirs:
        # 将射线方向作为输入
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # 特殊情况下可视化视图方向的效果
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape  # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):
    """渲染路径
    @:param: render_poses (list): 相机位姿列表。
    @:param: hwf (list): 图像高度、宽度和焦距。
    @:param: K (torch.Tensor): 相机内参矩阵，形状为[3, 3]。
    @:param: chunk (int): minibatch的大小，默认为1024*32。
    @:param: render_kwargs (dict): 传递给render函数的参数。
    @:param: gt_imgs (list): 真实图像列表。
    @:param: savedir (str): 保存渲染结果的路径。
    @:param: render_factor (int): 渲染因子。
    @:returns: list: 渲染结果列表。
    """
    H, W, focal = hwf  # focal为焦距

    if render_factor != 0:
        # 为了加速，渲染下采样的图像
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i == 0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args):
    """创建NeRF的多层感知器模型
    @:param: args (argparse.Namespace): 参数。
    @:returns: torch.nn.Module: NeRF模型。
    """
    # 获取embedding函数和输入通道数(from run_nerf_helpers.py)
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    # 初始化视点embedding函数和输入通道数
    input_ch_views = 0  # 输入通道数
    embeddirs_fn = None  # embedding函数
    if args.use_viewdirs:  # 如果使用视点方向，获取embedding函数和输入通道数
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)

    # 输出通道数（如果使用fine模型，则输出通道数为5，否则为4）
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]

    # 创建NeRF模型
    model = NeRF(D=args.netdepth, W=args.netwidth,  # 网络深度和宽度
                 input_ch=input_ch, output_ch=output_ch,  # 输入输出通道数
                 skips=skips,  # 跳层
                 input_ch_views=input_ch_views,  # 视点输入通道数
                 use_viewdirs=args.use_viewdirs).to(device)  # 是否使用视点方向
    grad_vars = list(model.parameters())  # 获取需要优化的参数

    # 如果使用了fine模型，则创建fine模型
    model_fine = None
    if args.N_importance > 0:  # 如果使用了fine模型
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,  # fine模型的网络深度和宽度
                          input_ch=input_ch, output_ch=output_ch,  # 输入输出通道数
                          skips=skips,  # 跳层
                          input_ch_views=input_ch_views,  # 视点输入通道数
                          use_viewdirs=args.use_viewdirs).to(device)  # 是否使用视点方向
        grad_vars += list(model_fine.parameters())

    # 定义网络查询函数
    network_query_fn = lambda inputs, viewdirs, network_fn: run_network(inputs,  # 输入图像坐标
                                                                        viewdirs,  # 视点方向
                                                                        network_fn,  # 网络函数
                                                                        embed_fn=embed_fn,  # embedding函数
                                                                        embeddirs_fn=embeddirs_fn,  # 视点embedding函数
                                                                        netchunk=args.netchunk)  # minibatch大小

    # 创建优化器
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))  # Adam优化器

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # 加载检查点
    if args.ft_path is not None and args.ft_path != 'None':  # 如果指定了fine-tune的路径，则加载该路径下的检查点
        ckpts = [args.ft_path]
    else:  # 否则加载basedir/expname下的检查点
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # 加载模型
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }

    # 只有LLFF格式的前向数据才适用NDC
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """转换模型的预测值到语义上的有意义的值
    @:param raw: [num_rays, num_samples along ray, 4]. 模型的预测值
    @:param z_vals: [num_rays, num_samples along ray]. 集成时间
    @:param rays_d: [num_rays, 3]. 每个射线的方向
    @:param raw_noise_std: 标准差
    @:param white_bkgd: 是否使用白色背景
    @:param pytest: 是否使用pytest
    @:return rgb_map: [num_rays, 3]. 射线的估计RGB颜色
    @:return disp_map: [num_rays]. 射线的估计深度
    @:return acc_map: [num_rays]. 射线的估计透明度
    @:return weights: [num_rays, num_samples along ray]. 射线的权重
    @:return depth_map: [num_rays]. 射线的估计深度
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
        """体积渲染：从光线批次中渲染图像
        @:param ray_batch: [batch_size, ...]. 所有必要的信息，包括：光线原点、光线方向、最小距离、最大距离和单位化的视图方向
        @:param network_fn: 模型，用于预测每个空间点的RGB和密度
        @:param network_query_fn: 用于将查询传递给network_fn的函数
        @:param N_samples: int. 每条光线采样的次数
        @:param retraw: bool. 如果为True，则包括模型的原始、未处理的预测
        @:param lindisp: bool. 如果为True，则以相反深度的线性方式采样，而不是以深度采样
        @:param perturb: float, 0 or 1. 如果非零，则每条光线在时间上以分层随机点采样
        @:param N_importance: int. 每条光线额外采样的次数。这些样本仅传递给network_fine
        @:param network_fine: "fine" 用于优化的网络：如果不为None，则使用它来重新采样光线
        @:param white_bkgd: bool. 如果为True，则将背景设置为白色
        @:param raw_noise_std: float. 如果不为0，则在网络输出上添加噪声
        @:param verbose: bool. 如果为True，则打印有关渲染的信息
        @:param pytest: bool. 如果为True，则使用固定的随机种子
        @:return ret:
            rgb0: rgb_map: [num_rays, 3]. 估算出的射线的RGB颜色。来自于精细模型。
            disp0: disp_map: [num_rays]. 差距图。1/深度。
            acc0: acc_map: [num_rays]. 沿着每条射线累积的不透明度。来自于精细模型。
            z_std: [num_rays]. 每个样本的沿射线距离的标准偏差。N_rays = ray_batch.shape[0]
        """
        N_rays = ray_batch.shape[0]
        rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]
        viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
        bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
        near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

        t_vals = torch.linspace(0., 1., steps=N_samples)
        if not lindisp:
            z_vals = near * (1. - t_vals) + far * (t_vals)
        else:
            z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

        z_vals = z_vals.expand([N_rays, N_samples])

        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape)

            # Pytest, overwrite u with numpy's fixed random numbers
            if pytest:
                np.random.seed(0)
                t_rand = np.random.rand(*list(z_vals.shape))
                t_rand = torch.Tensor(t_rand)

            z_vals = lower + (upper - lower) * t_rand

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

        #     raw = run_network(pts)
        raw = network_query_fn(pts, viewdirs, network_fn)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd,
                                                                     pytest=pytest)

        if N_importance > 0:
            rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)
            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :,
                                                                None]  # [N_rays, N_samples + N_importance, 3]

            run_fn = network_fn if network_fine is None else network_fine
            #         raw = run_network(pts, fn=run_fn)
            raw = network_query_fn(pts, viewdirs, run_fn)

            rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd,
                                                                         pytest=pytest)

        ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
        if retraw:
            ret['raw'] = raw
        if N_importance > 0:
            ret['rgb0'] = rgb_map_0
            ret['disp0'] = disp_map_0
            ret['acc0'] = acc_map_0
            ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

        for k in ret:
            if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
                print(f"! [Numerical Error] {k} contains nan or inf.")

        return ret


def config_parser():
    """解析命令行参数和flag"""
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024 * 32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024 * 64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')
    # logging/saving options
    parser.add_argument("--i_print", type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img", type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video", type=int, default=50000,
                        help='frequency of render_poses video saving')

    return parser


def train():
    """训练函数"""
    parser = config_parser()
    args = parser.parse_args()

    # 加载数据
    K = None
    if args.dataset_type == 'llff':  # 如果是llff格式的数据集
        # 加载图片、相机位姿、相机焦距、渲染相机位姿、测试图片的索引
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        # 获取相机参数
        hwf = poses[0, :3, -1]
        # 从poses数组中提取出旋转和平移信息
        poses = poses[:, :3, :4]
        print(f"Loaded llff, img:{images.shape}, render poses{render_poses.shape}, hwf:{hwf}, datadir:{args.datadir}")

        # 如果i_test不是一个列表，将其转化为列表形式
        if not isinstance(i_test, list):
            i_test = [i_test]

        # 根据llffhold的值来分割数据集
        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        # 确定训练集、验证集和测试集的索引
        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])

        # 定义场景的空间范围（near和far）
        print('DEFINING BOUNDS')
        if args.no_ndc:  # 如果不使用归一化坐标系，使用数据集中深度范围的90%作为near，深度范围的最大值作为far
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
        else:  # 否则near为0，far为1
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res,
                                                                                    args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
        near = hemi_R - 1.
        far = hemi_R + 1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # 将相机内参转换为正确的数据类型
    H, W, focal = hwf  # 相机成像高、宽、焦距
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    # 如果K为空，使用默认相机内参(LINEMOD 数据带有相机内参矩阵K)
    if K is None:
        K = np.array([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])

    # 如果需要渲染测试集中的图片，则使用测试集的相机位姿
    if args.render_test:
        render_poses = np.array(poses[i_test])

    # 创建日志目录，将命令行参数和配置文件写入日志目录中
    basedir = args.basedir  # 日志目录
    expname = args.expname  # 实验名称
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)  # 如果目录不存在，则创建一个日志目录
    f = os.path.join(basedir, expname, 'args.txt')  # 存储命令行参数的文本文件的路径
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))  # 将每个命令行参数和其值写入文本文件中
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')  # 存储配置文件的文本文件的路径
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())  # 将配置文件内容写入文本文件中

    # 创建NERF模型
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    # 定义场景空间范围
    bds_dict = {
        'near': near,
        'far': far,
    }
    # 将场景空间范围加入渲染参数
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # 将测试数据移动到GPU上
    render_poses = torch.Tensor(render_poses).to(device)

    # 如果只是从已经训练好的模型中进行渲染，则直接进行渲染
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # 使用测试集的数据进行渲染
                images = images[i_test]
            else:
                # 默认使用较为平滑的渲染路径
                images = None

            # 定义渲染结果的保存目录
            testsavedir = os.path.join(basedir, expname,
                                       'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            # 渲染路径，并将结果保存在testsavedir中
            rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images,
                                  savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)

            # 将渲染结果保存为mp4格式视频
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # 如果使用随机光线批处理，则准备光线批处理张量
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # 从训练集中随机选择N_rand个图像
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:, :3, :4]], 0)  # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:, None]], 1)  # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])  # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0)  # train images only
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])  # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)  # [N, H, W, ro+rd+rgb, 3]
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    # 将训练数据移动到GPU上
    if use_batching:
        images = torch.Tensor(images).to(device)  # [N, H, W, 3]
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)

    N_iters = 200000 + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))

    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch + N_rand]  # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3, :4]

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H // 2 * args.precrop_frac)
                    dW = int(W // 2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                            torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW)
                        ), -1)
                    if i == start:
                        print(
                            f"[Config] Center cropping of size {2 * dH} x {2 * dW} is enabled until iter {args.precrop_iters}")
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)),
                                         -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####
        rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                        verbose=i < 10, retraw=True,
                                        **render_kwargs_train)

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        trans = extras['raw'][..., -1]
        loss = img_loss
        psnr = mse2psnr(img_loss)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        # 设置衰减率和衰减步数
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        # 根据全局步数计算新的学习率
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        # 更新优化器中所有参数组的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        # 注意：重要提示！
        ### 更新学习率 ###
        衰减率 = 0.1
        衰减步数 = args.lrate_decay * 1000
        新学习率 = args.lrate * (衰减率 ** (global_step / 衰减步数))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time() - time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i % args.i_video == 0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test,
                            gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')

        if i % args.i_print == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
        """
            print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))

            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('psnr', psnr)
                tf.contrib.summary.histogram('tran', trans)
                if args.N_importance > 0:
                    tf.contrib.summary.scalar('psnr0', psnr0)


            if i%args.i_img==0:

                # Log a rendered validation view to Tensorboard
                img_i=np.random.choice(i_val)
                target = images[img_i]
                pose = poses[img_i, :3,:4]
                with torch.no_grad():
                    rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                        **render_kwargs_test)

                psnr = mse2psnr(img2mse(rgb, target))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                    tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                    tf.contrib.summary.image('disp', disp[tf.newaxis,...,tf.newaxis])
                    tf.contrib.summary.image('acc', acc[tf.newaxis,...,tf.newaxis])

                    tf.contrib.summary.scalar('psnr_holdout', psnr)
                    tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])


                if args.N_importance > 0:

                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                        tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
                        tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
                        tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])
        """

        global_step += 1


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
