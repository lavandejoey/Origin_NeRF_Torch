import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)  # 检测梯度异常

# Misc
img2mse = lambda x, y: torch.mean((x - y) ** 2)  # 计算均方误差
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]))  # 计算PSNR
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)  # 将图像转换为8位图像


# Positional encoding (section 5.1)
class Embedder:
    """
    给定空间位置(x, y, z)对应得到高纬度的编码位置p=(p1, p2, p3,...,pk)使MLP易学习
    Embed(x,y,z)=[sin(2^0 πx),cos(2^0 πx),...,sin(2^(K-1) πx),cos(2^(K-1) πx),
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    # 创建embedding函数
    def create_embedding_fn(self):
        # 定义空的embedding函数列表和输出维度
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        # 如果包含输入，则添加一个恒等函数，并更新输出维度
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d
        # 根据输入参数设置embedding函数的各种参数
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)  # 对数采样
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)  # 等间隔采样
        # 添加不同频率的周期函数到embedding函数列表中，并累加输出维度
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                # 使用lambda函数定义embedding函数，每个embedding函数将输入的每个维度乘上不同的频率，再使用周期函数进行变换
                # 将得到的结果添加到embedding函数列表中
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d
        # 将embedding函数列表和输出维度分别赋值给类的两个成员变量
        self.embed_fns = embed_fns
        self.out_dim = out_dim

    # 对输入进行embedding操作
    def embed(self, inputs):
        # 遍历embedding函数列表，将输入的每一维度都分别传递给相应的embedding函数
        # 并将所有函数的输出按最后一维拼接起来，形成最终的输出
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    """定义embedding函数
    @param multires: 频率数量
    @param i: 输入通道数
    @return embedding函数和输出维度
    """
    # 如果i为-1，返回恒等映射和输入通道数为3
    if i == -1:
        return nn.Identity(), 3
    # 定义embedding函数的参数
    embed_kwargs = {
        'include_input': True,  # 是否包含输入信息
        'input_dims': 3,  # 输入通道数
        'max_freq_log2': multires - 1,  # 最大频率
        'num_freqs': multires,  # 频率数量
        'log_sampling': True,  # 是否使用对数采样
        'periodic_fns': [torch.sin, torch.cos],  # 周期函数
    }
    # 创建Embedder对象并定义embedding函数
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)  # embedding函数, lambda输入x，输出eo.embed(x)
    return embed, embedder_obj.out_dim


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """定义NeRF的MLP模型
        @param D: MLP的深度
        @param W: MLP的宽度
        @param input_ch: 输入点的通道数
        @param input_ch_views: 输入视角的通道数
        @param output_ch: 输出的通道数
        @param skips: 跳层的索引
        @param use_viewdirs: 是否使用视角
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        # 对点的全连接层
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                                        range(D - 1)])
        # 对视角的全连接层
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:  # 若使用视角，则分别计算透明度、颜色和视角特征
            self.feature_linear = nn.Linear(W, W)  # 视角特征
            self.alpha_linear = nn.Linear(W, 1)  # 透明度
            self.rgb_linear = nn.Linear(W // 2, 3)  # 颜色
        else:  # 不使用视角，则直接计算输出
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        """前向传播: 输入点和视角，输出透明度、颜色和视角特征
        @param x: 输入
        @return 输出
        """
        # 拆分输入的点和视角信息
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts  # 将输入的点信息赋值给h
        for i, l in enumerate(self.pts_linears):  # 遍历点的全连接层
            h = self.pts_linears[i](h)  # 计算全连接层的输出
            h = F.relu(h)  # 使用ReLU激活函数
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)  # 将输入和输出拼接起来

        if self.use_viewdirs:  # 若使用视角
            # 分别计算透明度、颜色和视角特征
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):  # 遍历视角的全连接层
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)  # 将颜色和透明度拼接起来
        else:
            # 直接计算输出
            outputs = self.output_linear(h)

        return outputs

    def load_weights_from_keras(self, weights):
        """将从 Keras 模型中获取的权重加载到当前 PyTorch 模型中。
        @param weights: numpy.ndarray，包含从 Keras 模型获取的权重的 numpy 数组。
        """
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        # 加载点的全连接层
        for i in range(self.D):
            idx_pts_linears = 2 * i
            # 从 numpy 数组中读取当前层的权重，然后将它们转换为 torch.Tensor 类型，并将其赋值给当前层的权重。
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears + 1]))
        # 加载特征的全连接层
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear + 1]))
        # 加载视角的全连接层
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears + 1]))
        # 加载颜色的全连接层
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear + 1]))
        # 加载透明度的全连接层
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear + 1]))


def get_rays(H, W, K, c2w):
    """获取射线
    @param H: 图像的高度
    @param W: 图像的宽度
    @param K: 相机内参矩阵
    @param c2w: 相机坐标系到世界坐标系的变换矩阵
    @return 射线的起点和方向
    """
    # 生成网格：i, j 的取值范围是 [0, W-1] 和 [0, H-1]，步长为 1。
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W),
                          torch.linspace(0, H - 1, H))  # pytorch的网格有indexing='ij'。
    i = i.t()
    j = j.t()
    # 将像素坐标转换为归一化的相机坐标：x = (i - cx) / fx, y = (j - cy) / fy, z = -1
    dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)
    # 将射线方向从相机坐标系旋转到世界坐标系
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)  # 点积，等于： [c2w.dot(dir) for dir in dirs]
    # 将相机坐标系的原点转换到世界坐标系, 它是所有射线的起点
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    """生成射线（视线）的起点和方向，以用于渲染图像。
    @param H (int): 图像高度
    @param W (int): 图像宽度
    @param K (np.ndarray): 相机内参矩阵，形状为[3, 3]，包含焦距、主点和相机畸变参数。
    @param c2w (np.ndarray): 相机到世界坐标系的变换矩阵，形状为[4, 4]。
    @return rays_o, rays_d (np.ndarray): 射线起点和方向，形状为[H, W, 3]。
    """
    # 生成网格: i, j 的取值范围是 [0, W-1] 和 [0, H-1]，步长为 1。
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexin0g='xy')
    dirs = np.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -np.ones_like(i)], -1)
    # 旋转射线方向从相机坐标系到世界坐标系
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)  # 点积，等于： [c2w.dot(dir) for dir in dirs]
    # 转换相机坐标系的原点到世界坐标系，它是所有射线的起点
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    """将射线从相机坐标系转换到 NDC 坐标系。
    @param H: 图像的高度
    @param W: 图像的宽度
    @param focal: 相机焦距
    @param near: 相机近平面
    @param rays_o: 射线的起点，形状为 [N_rays, 3]
    @param rays_d: 射线的方向，形状为 [N_rays, 3]
    @return rays_o, rays_d (np.ndarray): 射线起点和方向，形状为 [N_rays, 3]
    """
    """将射线的起点移动到近平面
           near + rays_o,z
    t = - —————————————————
              rays_d,z
    rays_o,new = rays_o + t ⋅ rays_d"""
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d
    """射线起点在归一化设备坐标系中的坐标为：
                     o_x
    o_x,new = - ————————————— ⋅ 1/o_z 
                 focal ⋅ W/2
                     o_y
    o_y,new = - ————————————— ⋅ 1/o_z 
                 focal ⋅ H/2
                   2 · near
    o_z,new = 1 + —————————— 
                     o_z"""
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]
    """射线方向在归一化设备坐标系中的坐标为：
                      1         d_x    o_x
    d_x,new = - ———————————— ⋅ (———— - ————)
                 focal ⋅ W/2     d_z    o_z
                      1         d_y    o_y
    d_y,new = - ———————————— ⋅ (———— - ————)
                 focal ⋅ H/2     d_z    o_z
                 2 · near
    d_z,new = − ——————————
                   o_z"""
    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    # 将新的射线起点 [o0, o1, o2] 沿着最后一个维度（即维度 -1）拼接起来，生成一个形状为 [N_rays, 3] 的张量
    rays_o = torch.stack([o0, o1, o2], -1)
    # 将新的射线方向 [d0, d1, d2] 沿着最后一个维度拼接起来，生成一个形状为 [N_rays, 3] 的张量
    rays_d = torch.stack([d0, d1, d2], -1)
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    """取样概率密度函数
    :@param bins: 概率密度函数的离散值，形状为 [N_bins]
    :@param weights: 概率密度函数的权重，形状为 [N_bins]
    :@param N_samples: 取样的数量
    :@param det: 是否使用确定性取样
    :@param pytest: 是否使用固定的随机数
    :@return samples (torch.Tensor): 取样结果，形状为 [N_samples]
    """
    # 获取概率密度函数PDF和累积分布函数CDF
    weights = weights + 1e-5  # 防止NaN
    pdf = weights / torch.sum(weights, -1, keepdim=True) # 计算概率密度函数
    cdf = torch.cumsum(pdf, -1)  # 计算累积分布函数
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # 添加 0 到 cdf 第一个元素之间的值，方便进行二分搜索

    # 采取统一的样本
    if det:  # 确定性取样: 从0到1等间隔取样
        u = torch.linspace(0., 1., steps=N_samples) # 等间隔取样
        u = u.expand(list(cdf.shape[:-1]) + [N_samples]) # 将 u 扩展成与 cdf 相同的形状
    else:  # 随机取样: 从0到1均匀取样
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples]) # 在 [0,1) 中随机取样

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:  # 使用固定的随机数取样，主要用于测试
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # 构造一个二元组，包含每个样本的区间的下界和上界

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)  # 从 cdf 中按照 inds_g 的值进行取值
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)  # 从 bins 中按照 inds_g 的值进行取值

    denom = (cdf_g[..., 1] - cdf_g[..., 0])  # 计算区间长度
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)  # 将非法值设为 1，避免分母为 0
    t = (u - cdf_g[..., 0]) / denom  # 计算每个样本在区间中的位置
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])  # 根据区间中的位置计算每个样本的值

    return samples
