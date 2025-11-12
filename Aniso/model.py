"""Definitions of AnisoNet models."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn.models.gate_points_2101 import smooth_cutoff, tp_path_exists
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import FullyConnectedNet, Gate
from e3nn.o3 import FullyConnectedTensorProduct, TensorProduct
from e3nn.util.jit import compile_mode
from e3nn.io import CartesianTensor

from torch_scatter import scatter_mean,scatter

from Aniso.data import *
class Convolution(torch.nn.Module):
    r"""equivariant convolution

    Parameters
    ----------
    irreps_in : e3nn.o3.Irreps
        representation of the input node features

    irreps_node_attr : e3nn.o3.Irreps
        representation of the node attributes

    irreps_edge_attr : e3nn.o3.Irreps
        representation of the edge attributes

    irreps_out : e3nn.o3.Irreps or None
        representation of the output node features

    number_of_basis : int
        number of basis on which the edge length are projected

    radial_layers : int
        number of hidden layers in the radial fully connected network

    radial_neurons : int
        number of neurons in the hidden layers of the radial fully connected network

    num_neighbors : float
        typical number of nodes convolved over
    """

    def __init__(
        self,
        irreps_in,
        irreps_node_attr,
        irreps_edge_attr,
        irreps_out,
        number_of_basis,
        radial_layers,
        radial_neurons,
        num_neighbors,

        irreps_crystal_attr,
        #irreps_lattice_feature,
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_out = o3.Irreps(irreps_out)
        self.num_neighbors = num_neighbors

        self.irreps_crystal_attr = o3.Irreps(irreps_crystal_attr)
        #self.irreps_lattice_feature = o3.Irreps(irreps_lattice_feature)
        
        # 三路 tensor product 组合节点特征
        self.sc_attr = FullyConnectedTensorProduct(self.irreps_in, self.irreps_node_attr, self.irreps_out)
        self.sc_crystal = FullyConnectedTensorProduct(self.irreps_in, self.irreps_crystal_attr, self.irreps_out)
        #self.sc_lattice = FullyConnectedTensorProduct(self.irreps_in, self.irreps_lattice_feature, self.irreps_out)

        self.lin1_attr = FullyConnectedTensorProduct(self.irreps_in, self.irreps_node_attr, self.irreps_in)
        self.lin1_crystal = FullyConnectedTensorProduct(self.irreps_in, self.irreps_crystal_attr, self.irreps_in)
        #self.lin1_lattice = FullyConnectedTensorProduct(self.irreps_in, self.irreps_lattice_feature, self.irreps_in)

        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(self.irreps_in):
            for j, (_, ir_edge) in enumerate(self.irreps_edge_attr):
                for ir_out in ir_in * ir_edge:
                    if ir_out in self.irreps_out:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        instructions = [(i_1, i_2, p[i_out], mode, train) for i_1, i_2, i_out, mode, train in instructions]

        tp = TensorProduct(
            self.irreps_in,
            self.irreps_edge_attr,
            irreps_mid,
            instructions,
            internal_weights=False,
            shared_weights=False,
        )
        self.fc = FullyConnectedNet(
            [number_of_basis] + radial_layers * [radial_neurons] + [tp.weight_numel], torch.nn.functional.silu
        )
        self.tp = tp

        # self.lin2_attr = FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attr, self.irreps_out)
        # self.lin2_crystal = FullyConnectedTensorProduct(irreps_mid, self.irreps_crystal_attr, self.irreps_out)
        self.lin2_attr = FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attr, irreps_mid)
        self.lin2_crystal = FullyConnectedTensorProduct(irreps_mid, self.irreps_crystal_attr, irreps_mid)


        self.lin3 = FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attr, self.irreps_out)

        

        
    def forward(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedded,node_crystal_attr,sym_mask) -> torch.Tensor:
        
        weight = self.fc(edge_length_embedded)

        x = node_input

        s = (
            self.sc_attr(node_input, node_attr)
            + self.sc_crystal(node_input, node_crystal_attr)
            #+ self.sc_lattice(node_input, node_lattice_feature)
        )

        x = (
            self.lin1_attr(node_input, node_attr)
            + self.lin1_crystal(node_input, node_crystal_attr)
            #+ self.lin1_lattice(node_input, node_lattice_feature)
        )

        edge_features = self.tp(x[edge_src], edge_attr, weight)
        x = scatter(edge_features, edge_dst, dim=0, dim_size=x.shape[0]).div(self.num_neighbors**0.5)

        x = (
            self.lin2_attr(x, node_attr)
            + self.lin2_crystal(x, node_crystal_attr)
            #+ self.lin2_lattice(x, node_lattice_feature)
        )
        
        x = (self.lin3(x, sym_mask))

        c_s, c_x = math.sin(math.pi / 8), math.cos(math.pi / 8)
        m = self.sc_attr.output_mask
        c_x = (1 - m) + c_x * m
        return c_s * s + c_x * x  


class CustomCompose(torch.nn.Module):
    """Custom compose for gate and convolution."""

    def __init__(self, first, second):
        super().__init__()
        self.first = first
        self.second = second
        self.irreps_in = self.first.irreps_in
        self.irreps_out = self.second.irreps_out

    def forward(self, *input):
        """Apply gate and convolution."""
        x = self.first(*input)
        self.first_out = x.clone()
        x = self.second(x)
        self.second_out = x.clone()
        return x


class E3nnModel(torch.nn.Module):
    r"""equivariant neural network.

    Parameters
    ----------
    irreps_in : e3nn.o3.Irreps or None
        representation of the input features
        can be set to `None if nodes don't have input features
    irreps_hidden : e3nn.o3.Irreps
        representation of the hidden features
    irreps_out : e3nn.o3.Irreps
        representation of the output features
    irreps_node_attr : e3nn.o3.Irreps or None
        representation of the nodes attributes
        can be set to `None if nodes don't have attributes
    irreps_edge_attr : e3nn.o3.Irreps
        representation of the edge attributes
        the edge attributes are :math:h(r) Y(\vec r / r)
        where :math:h is a smooth function that goes to zero at `max_radius
        and :math:Y are the spherical harmonics polynomials
    layers : int
        number of gates (non linearities)
    max_radius : float
        maximum radius for the convolution
    number_of_basis : int
        number of basis on which the edge length are projected
    radial_layers : int
        number of hidden layers in the radial fully connected network
    radial_neurons : int
        number of neurons in the hidden layers of the radial fully connected network
    num_neighbors : float
        typical number of nodes at a distance `max_radius
    num_nodes : float
        typical number of nodes in a graph.
    """

    def __init__(
        self,
        in_dim,
        em_dim,
        in_attr_dim,
        em_attr_dim,
        crystal_dim,
        em_crystal_dim,
       
        irreps_out,
        
        layers,
        mul,
        lmax,
        max_radius,
        number_of_basis=10,
        radial_layers=1,
        radial_neurons=100,
        num_neighbors=1.0,
        num_nodes=1.0,
        reduce_output=True,
        same_em_layer=False,
    ) -> None:
        super().__init__()
        self.mul = mul
        self.lmax = lmax
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.num_neighbors = num_neighbors
        self.num_nodes = num_nodes
        self.reduce_output = reduce_output
        self.same_em_layer = same_em_layer


        #输入节点特征表示，维度为 em_dim，表示的是 em_dim 个 0e（标量，偶宇称）的不可约表示。
        self.irreps_in = o3.Irreps(str(em_dim) + "x0e")
        #节点属性表示，和 node_attr 一一对应，同样是 0e 类型
        self.irreps_node_attr = o3.Irreps(str(em_attr_dim) + "x0e")

        self.irreps_crystal_attr = o3.Irreps(f"{em_crystal_dim}x0e")
        #self.irreps_lattice_feature = o3.Irreps(f"{lattice_dim}x0e")

        #定义隐藏层的表示结构：每个角动量量子数 l=0...lmax，和奇偶性 p=±1，乘以 mul 表示重复多少次。
        self.irreps_hidden = o3.Irreps(
            [(self.mul, (L, p)) for L in range(lmax + 1) for p in [-1, 1]]
        )

        #最终输出表示结构，可自定义
        self.irreps_out = o3.Irreps(irreps_out)
        #边特征的角动量表示
        self.irreps_edge_attr = o3.Irreps.spherical_harmonics(lmax)

        #节点特征嵌入层 将输入特征从 in_dim 映射到 em_dim 维，变成 em_dim x 0e 的表示。
        self.em = nn.Linear(in_dim, em_dim)
        
        #可选：给节点属性单独一个线性嵌入层（如果 same_em_layer=False）。
        if same_em_layer is False:
            self.em_attr = nn.Linear(in_attr_dim, em_attr_dim)
        
        #sym_mask和节点特征的维度相同
        self.em_crys = nn.Linear(48, em_attr_dim)

        self.em_crystal= nn.Linear(crystal_dim, em_crystal_dim)

        irreps = self.irreps_in if self.irreps_in is not None else o3.Irreps("0e")

        #用于给 scalar features（l=0） 设置非线性激活函数（用于 Gate 里的 scalars）
        act = {
        1: torch.nn.functional.silu,  # 如果是偶（even parity, +1） => 使用 silu 激活（平滑ReLU）
        -1: torch.tanh,               # 如果是奇（odd parity, -1） => 使用 tanh（奇函数，反演对称）
        }
        #用于设置 gate 中的控制变量 的激活函数：
        act_gates = {
        1: torch.sigmoid,  # 控制偶对称的 gated 输出 => sigmoid ∈ [0,1]
        -1: torch.tanh     # 控制奇对称的 gated 输出 => tanh，满足反演变号的对称性
        }

        # 模块列表容器
        self.layers = torch.nn.ModuleList()

        #在layers参数的控制下：循环构造每一层的 图卷积 + 门控非线性层
        for _ in range(layers):
            #筛选出可以通过张量积路径生成的标量成分，这个结果会作为当前网络层中 Gate 模块里的 scalar 分支的输出表示（对应 Gate 中的 scalar 门控）
            irreps_scalars = o3.Irreps([(mul, ir) for mul, ir in self.irreps_hidden if ir.l == 0 and tp_path_exists(irreps, self.irreps_edge_attr, ir)])
            #筛选出可以通过张量积路径生成的张量成分 (  irreps_gates ,方向性张量)
            irreps_gated = o3.Irreps([(mul, ir) for mul, ir in self.irreps_hidden if ir.l > 0 and tp_path_exists(irreps, self.irreps_edge_attr, ir)])
            
            #确保 gate 控制通道 一定是标量（l=0），只不过奇偶性可能不同
            ir = "0e" if tp_path_exists(irreps, self.irreps_edge_attr, "0e") else "0o"
            #为方向性张量构造对应数量的 gate 控制标量通道（irreps_gates）
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated])
            #遍历之前筛选好的 irreps_gated，也就是所有要被 gate 控制的向量/张量特征
            #对每个 (mul, _)，我们用 相同的 multiplicity（重复次数） 和刚才选定的 "0e" 或 "0o" 构造新的 scalar irrep
            #最终构建出与 irreps_gated 形状一致的标量 irrep 结构（只是表示类型从张量换成了标量）

            #门控激活函数模块
            gate = Gate(
                irreps_scalars,
                [act[ir.p] for _, ir in irreps_scalars],  # scalar
                irreps_gates,
                [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated,  # gated tensors
            )
            conv = Convolution(
                irreps,
                self.irreps_node_attr,
                self.irreps_edge_attr,

                #节点特征的输出表示，在该卷积层后可能还存在新的gate+卷积 层 ，则该输出为 gate 的输入
                gate.irreps_in,
                #gate.irreps_in 是由 Gate 类内部自动计算并设置的属性，用于描述 Gate 接受的特征结构，你无需手动定义

                number_of_basis,
                radial_layers,
                radial_neurons,
                num_neighbors,
                
                irreps_crystal_attr=self.irreps_crystal_attr,
                #irreps_lattice_feature=self.irreps_lattice_feature,
            )

            #当前层输出的特征结构（Irreps）被作为下一层的输入特征结构
            irreps = gate.irreps_out
            #当前这一层的“卷积 + 激活”模块组合起来，加到整个神经网络的层结构中
            self.layers.append(CustomCompose(conv, gate))

        #这段代码是紧跟在前面构建多个“卷积 + Gate”的主干层之后，用于收尾：
        #添加最后一层卷积（没有 Gate 激活）
        self.layers.append(
            Convolution(
                irreps,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                self.irreps_out,
                number_of_basis,
                radial_layers,
                radial_neurons,
                num_neighbors,

                irreps_crystal_attr=self.irreps_crystal_attr,
                #irreps_lattice_feature=self.irreps_lattice_feature,
            )
        )

    def forward(self, data) -> torch.Tensor:
        # print("\n=== 首个 batch ===")
        # print(data)                # 会显示 keys: batch.node_input, batch.edge_index, ...
        # print("batch.node_input:", data.node_input.shape)
        # print("====================\n")
        #exit(0)
        """Evaluate the network.

        Parameters
        ----------
        data : torch_geometric.data.Data or dict
            data object containing
            - `pos the position of the nodes (atoms)
            - `x the input features of the nodes, optional
            - `z the attributes of the nodes, for instance the atom type, optional
            - `batch the graph to which the node belong, optional.
        """
        # if data is None:
        #     raise ValueError("Forward received None batch")
        # if isinstance(data, tuple):
        #     data = data[0]
        # # 确保是 Aniso.data.Batch
        # assert hasattr(data, "node_input"), f"Expected Batch, got {type(data)}"
        
        node_input = F.relu(self.em(data.node_input))

        if self.same_em_layer:
            node_attr = F.relu(self.em(data.node_attr))
        else:
            node_attr = F.relu(self.em_attr(data.node_attr))

        node_crystal_attr = F.relu(self.em_crystal(data.node_crystal_attr))  # shape: [N, 7]
        #node_lattice_feature = data.node_lattice_feature  # shape: [N, 9]

        edge_src, edge_dst = data.edge_index
        edge_vec = data.edge_vec
        
        sym_attr = F.relu(self.em_crys(data.sym_mask))
        sym_mask = sym_attr[data.batch]

        ##将边方向编码成 球谐函数，用作旋转等变的卷积核
        edge_sh = o3.spherical_harmonics(
            self.irreps_edge_attr, edge_vec, True, normalization="component"
        )
        ##边长（用于构造径向函数） 计算每条边的欧几里得距离（标量），用于径向处理
        edge_length = edge_vec.norm(dim=1) 
        #print('==================breakpoint2=====================')
        ##构造径向嵌入
        edge_length_embedded = soft_one_hot_linspace(
            x=edge_length,
            start=0.0,
            end=self.max_radius,
            number=self.number_of_basis,
            basis="gaussian",
            cutoff=False,
        ).mul(self.number_of_basis**0.5)
        #乘以（self.number_of_basis**0.5） √N 是为了确保径向嵌入向量的 整体幅度（norm）不随 basis 数量 N 增大而变小，保持数值稳定性，让网络更容易训练
        #即mul(self.number_of_basis**0.5) 是一种数值标准化策略，用来补偿 soft-one-hot 嵌入维度变高带来的“模长稀释”，确保每条边的嵌入保持恒定尺度，进而提升 GNN 模型训练的数值稳定性和效率。

        #构造的是边的最终属性特征（edge_attr），用在图卷积中传递邻居信息。
        edge_attr = smooth_cutoff(edge_length / self.max_radius)[:, None] * edge_sh
        #smooth_cutoff() 是一个平滑的 0~1 函数（近似激活范围），将边方向特征（球谐）乘以 cutoff 掩码，保证远距离边的影响自动衰减为 0（物理上合理）
        #构造每条边的方向（通过球谐函数）+ 距离衰减（通过 cutoff）特征，使得图卷积既有空间对称性，又有物理上合理的距离控制，是等变图神经网络的核心特征构建方式之一
        #print('==================breakpoint3=====================')
        for lay in self.layers:
            node_input = lay(
                node_input,
                node_attr,
                edge_src,
                edge_dst,
                edge_attr,
                edge_length_embedded,

                node_crystal_attr,
                #node_lattice_feature,
                sym_mask,
            )
        #用于将节点特征转换为图级特征
        #对同一个图（batch 中的一个图）内的所有节点特征 node_input 取平均（mean pooling），得到每个图的一个整体表示（Graph-level embedding）
        #不关心每个节点具体输出，只关注图整体的预测。
        # if self.reduce_output:
        #     return scatter_mean(node_input, data.batch, dim=0)

        # return node_input
        # 5) 图级 readout
        # 对于 "ij=ji" => 对称二阶张量 => 6个独立分量
    
        #node_input=scatter_mean(node_input, data.batch, dim=0)  # [B, 6]
        ct = CartesianTensor("ij=ji")
        # (a) 使用 ct.to_cartesian(...) 将 [B,6] 不可约表示 => [B,3,3]
        cart_pred = ct.to_cartesian(scatter_mean(node_input, data.batch, dim=0))
        # 添加类型转换：将 cart_pred 转换为 Double 类型
        cart_pred = cart_pred.double()
        cart_pred = apply_sym_mask(cart_pred, data.sym_mask)
        return cart_pred