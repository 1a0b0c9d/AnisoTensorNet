"""Classes for loading in a dataset."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from ase import Atom, Atoms
from ase.neighborlist import neighbor_list
from torch.utils.data import Dataset
from tqdm.auto import tqdm

import numpy as np
from pymatgen.optimization.neighbors import find_points_in_spheres
from Aniso.symmat import symmetry_operations
@dataclass
class Data:
    """
    Class to contain graph attributes.

    N and M are the number of nodes and edges in the graph, respectively.

    Parameters
    ----------
    node_input : Tensor
        The node features as a (N, n_node_feats) Tensor.
    edge_index : Tensor
        The edge src and dst as a (2, M) Tensor.
    edge_vec : LongTensor
        The edge vectors as a (M, 3) Tensor.
    dielectric_tensor : Tensor
        The full dielectric tensor of the material, as a (3, 3) Tensor.

    """
    """
    与旧版差异：
    - 去掉了 dielectric_tensor / dielectric_scalar（不再监督张量）
    - 新增 target_n：折射率监督信号（float64）
    """
    node_input: torch.Tensor       # [N, in_dim]
    node_attr: torch.Tensor        # [N, in_attr_dim]
    edge_index: torch.LongTensor   # [2, M]
    edge_vec: torch.Tensor         # [M, 3]

    target_n: torch.Tensor         # [] 或 [1]，我们存标量（后续 collate 成 [B]）

    idx: int
    node_crystal_attr: torch.Tensor  # [N, 7]  crystal-system one-hot per node

    sym_mask: torch.Tensor         # [1, 48]  （保持与旧版一致的形状/设备迁移逻辑）
    diel_mask: torch.Tensor        # [1, 3, 3]

    def to(self, device, non_blocking=False):
        """放到计算设备上。idx 是纯整数不搬。"""
        for k, v in self.__dict__.items():
            if k == "idx":
                continue
            self.__dict__[k] = v.to(device=device, non_blocking=non_blocking)


@dataclass
class Batch:
    """
    Class to contain batched graph attributes.

    N and M are the number of nodes and edges across all batched graphs,
    respectively.

    G is the number of graphs in the batch.

    Parameters
    ----------
    node_input : Tensor
        The node features as a (N, n_node_feats) Tensor.
    edge_index : Tensor
        The edge src and dst as a (2, M) Tensor.
    edge_vec : LongTensor
        The edge vectors as a (M, 3) Tensor.
    dielectric_tensor : Tensor
        The full dielectric tensor of the material, as a (3, 3) Tensor.
    batch : LongTensor
        The graph to which each node belongs, as a (N, ) Tensor.
    """

    node_input: torch.Tensor
    node_attr: torch.Tensor
    edge_index: torch.LongTensor
    edge_vec: torch.Tensor

    target_n: torch.Tensor         # [B]  ← 简化损失计算

    batch: torch.LongTensor        # [N]  每个节点对应的图id
    idx: torch.LongTensor          # [B]  图级样本索引

    node_crystal_attr: torch.Tensor
    sym_mask: torch.Tensor
    diel_mask: torch.Tensor

    crystal_system_idx: torch.LongTensor
    
    def to(self, device, non_blocking=False):
        for k, v in self.__dict__.items():
            self.__dict__[k] = v.to(device=device, non_blocking=non_blocking)

def collate_fn(dataset):
    """
    Collate a list of Data objects and return a Batch.

    Parameters
    ----------
    dataset : MaterialsDataset
        The dataset to batch.

    Returns
    -------
    Batch
        A batched dataset.
    """
    batch = Batch([], [], [], [], [], [], [], [], [], [],[])
    base_idx = 0
    crystal_system_idx = []
    
    for i, data in enumerate(dataset):
        batch.node_input.append(data.node_input)
        batch.node_attr.append(data.node_attr)
        batch.edge_index.append(data.edge_index + base_idx)
        batch.edge_vec.append(data.edge_vec)

        batch.target_n.append(data.target_n)  # 标量

        batch.idx.append(data.idx)
        batch.node_crystal_attr.append(data.node_crystal_attr)
        batch.sym_mask.append(data.sym_mask)
        batch.diel_mask.append(data.diel_mask)

        batch.batch.extend([i] * len(data.node_input))
        base_idx += len(data.node_input)
        
        cs_idx = data.node_crystal_attr[0].argmax().item()
        crystal_system_idx.append(cs_idx)

    return Batch(
        node_input=torch.cat(batch.node_input),
        node_attr=torch.cat(batch.node_attr),
        edge_index=torch.cat(batch.edge_index, dim=-1),
        edge_vec=torch.cat(batch.edge_vec),

        target_n=torch.stack(batch.target_n).to(torch.double),  # [B]，double 精度

        batch=torch.LongTensor(batch.batch),
        idx=torch.LongTensor(batch.idx),

        node_crystal_attr=torch.cat(batch.node_crystal_attr),
        sym_mask=torch.cat(batch.sym_mask),
        diel_mask=torch.cat(batch.diel_mask),
        
        crystal_system_idx=torch.LongTensor(crystal_system_idx),

    )

class BaseDataset(Dataset):
    """Dataset of materials properties.

    Parameters
    ----------
    filename : str or DataFrame
        The path to the dataset or a pandas dataframe. If supplying a pandas Dataframe
        then the dataset is expected to contain two columns: "structure" containing
        ASE Atoms objects or pymatgen Structure objects and "target" containing the
        target to predict. If passing a filename, the file is expected to be in json
        format, containing a list of dictionaries, each with the keys "positions"
        (cartesian atomic positions), "cell" (cell lattice parameters), "numbers"
        (atomic numbers of the atoms).
    cutoff : float
        The cutoff radius for searching for neighbors.
    """
    """
    适配折射率任务的数据集。
    期望传入：预处理后的 pandas.DataFrame,包含:
    - structure (ASE Atoms)
    - n (float64)
    - crystal_system (str)

    假设若 structure 不是 ASE Atoms,会尝试 .to_ase_atoms() 兜底转换（预处理已统一）。
    """
    def __init__(self, df, cutoff=5, symprec=0.01, graph_type="cutoff",device=None):
        self.cutoff = cutoff
        self.symprec = symprec
        self.device = device
        self.data = []

        num_nodes = 0
        num_neighbors = 0

        self.structures = []

        # self.dielectric_tensors = []
        # self.dielectric_scalars = []
        # #存储每个结构对应的晶体系统类型
        # #一个和结构数一样长的列表，每个元素都是字符串，表示该结构所属的晶体系统。
        # self.crystal_systems = []

        # if isinstance(filename, (Path, str)):
        #     if Path(filename).suffix == ".gz":
        #         with gzip.open(filename) as f:
        #             jsondata = json.load(f)
        #     else:
        #         with open(filename) as f:
        #             jsondata = json.load(f)

        #     for entry in jsondata:
        #         self.structures.append(entry_to_atoms(entry))
                
        #         self.dielectric_tensors.append(torch.Tensor(entry["dielectric_tensor"]))
        #         self.dielectric_scalars.append(entry["dielectric_scalar"])
        #         self.crystal_systems.append(entry["crystal_system"])
        # else:
        #     self.structures = filename["structure"].values.tolist()
        #     if not isinstance(self.structures[0], Atoms):
        #         # presume we have pymatgen structure objects and try and convert
        #         self.structures = [a.to_ase_atoms() for a in self.structures]
            
        #     self.dielectric_tensors = filename['dielectric_tensor'].values.tolist()
        #     self.dielectric_scalars = filename['dielectric_scalar'].values.tolist()
        #     self.crystal_systems = filename["crystal_system"].tolist()
        # ---- 校验输入为 DataFrame，而不是路径 ----
        if isinstance(df, (str, Path)):
            raise ValueError("折射率任务期望直接传入预处理后的 DataFrame（含 structure/n/crystal_system）。")

        # ---- 取列 ----
        self.structures = df["structure"].values.tolist()
        self.targets_n = df["n"].astype(np.float64).values.tolist()  # 明确 float64
        self.crystal_systems = df["crystal_system"].tolist()

        # ---- 尝试统一为 ASE Atoms（一般预处理已转好）----
        if not isinstance(self.structures[0], Atoms):
            self.structures = [
                (a.to_ase_atoms() if hasattr(a, "to_ase_atoms") else a) for a in self.structures
            ]
            assert isinstance(self.structures[0], Atoms), "structure 需为 ASE Atoms。"
        #图数据构造 为每个结构，提取它的索引 + 结构信息 + 目标值 + 晶体系统 
        #enumerate它在 zip 的基础上再加上了样本索引idx  self.structures, self.dielectric_tensors, self.crystal_systems→ 构造图 idx, (atoms, dielectric_tensor, crystal_sys)
        for idx, (atoms, n_val, crystal_sys) in tqdm(
            enumerate(zip(self.structures, self.targets_n, self.crystal_systems)),
            total=len(self.structures),
        ):
            data = atoms_to_data(
                atoms=atoms,
                cutoff=self.cutoff,
                symprec=self.symprec,
                idx=idx,
                graph_type=graph_type,
                crystal_system=crystal_sys,
                target_n=n_val,
                device=device,
            )
            num_nodes += len(data.node_input)
            num_neighbors += len(data.edge_vec)
            self.data.append(data)

        # ---- 统计（可用于模型归一化/超参参考）----
        self.num_neighbors = num_neighbors / num_nodes
        self.num_nodes = num_nodes / len(self.data)

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]


def entry_to_atoms(entry):
    """Convert a dataset entry to atoms object."""
    return Atoms(
        positions=entry["positions"],
        cell=entry["cell"],
        numbers=entry["numbers"],
        pbc=[True, True, True],
    )


def atoms_to_data(
    atoms,
    cutoff,
    symprec,
    idx: int = 0,
    graph_type: str = "cutoff",
    crystal_system=None,
    target_n: float | None = None,
    device = None,
):
    """Convert an atoms object to a data object.

    Parameters
    ----------
    atoms : Atoms
        An Ase atoms object.
    cutoff : float
        The cutoff radius for neighbor finding.
    idx : int
        Index of the sample.
    type : str
        kwarg to specify how to construct the data. (cutoff / voronoi)

    Returns
    -------
    Data
        A custom Data object.
    """
    """
    将 ASE Atoms 转成 Data：
    - 邻接：cutoff 半径搜索（保留原逻辑）
    - 特征：元素 one-hot（type + mass），crystal_system one-hot 复制到每个节点
    - 对称：sym_mask / diel_mask（继续用于中间张量的对称性约束）
    - 监督：target_n（float64 标量）
    """
    if graph_type == "cutoff":
        # construct edge_src, edge_dst, edge_vec using max cut-off.
        try:
            positions = atoms.get_positions()
            cell = np.array(atoms.get_cell())

            #find_points_in_spheres()函数返回值
            #edge_src  中心原子索引（在 center_coords 中）
            #edge_dst  邻居原子索引（在 all_coords 中）
            #images 所有邻居点对应的周期偏移量（单位晶胞个数），shape: (n, 3)
            edge_src, edge_dst, images, _ = find_points_in_spheres(
                positions,
                positions,
                r=cutoff,
                pbc=np.array([1, 1, 1], dtype=int),
                lattice=cell,
                tol=1.0e-8,
            )
            #然后通过 images @ cell 加回跨晶胞的向量偏移，构造真实的空间边向量
            edge_vec = positions[edge_dst] - positions[edge_src] + images @ cell
        except ImportError:
            # fall back on slower ase algorithm if pymatgen not installed
            edge_src, edge_dst, edge_vec = neighbor_list(
                "ijD", atoms, cutoff=cutoff, self_interaction=True)
            
    natom = atoms.get_number_of_atoms()
    if natom < 100:
        sym_mask = torch.tensor(check_symmetry_parallel(atoms, symprec, device))
    else:
        sym_mask = torch.tensor(check_symmetry_serial(atoms, symprec, device))

    diel_mask = cal_diel_mask(atoms, sym_mask, device).unsqueeze(0)
    sym_mask = torch.tensor(sym_mask, dtype=torch.double).unsqueeze(0)


    node_input = torch.Tensor([type_onehot(i) for i in atoms.numbers])
    node_attr = torch.Tensor([mass_onehot(i) for i in atoms.numbers])

    #n_atoms = len(atoms)

    #构成一个 (n_atoms, 7) 的张量  每个原子都获得相同的晶体系统信息。
    node_crystal_attr = torch.Tensor([onehot_crystal_system(crystal_system)] * natom)
    
    # ---- 监督：折射率 n（float64 标量）----
    if target_n is None:
        raise ValueError("target_n 不能为空（折射率监督）。")
    target_n = torch.tensor(float(target_n), dtype=torch.double)  # 标量
    
    return Data(
        node_input=node_input,
        node_attr=node_attr,

        #把边的起点 (edge_src) 和终点 (edge_dst) 拼成一个 edge_index 张量，用于图神经网络的边结构表示
        edge_index=torch.stack(
            [torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0),
        
        #edge_vec=torch.Tensor(edge_vec.tolist()),
        edge_vec = torch.from_numpy(edge_vec).to(torch.double),

        target_n=target_n,

        idx=idx,
        node_crystal_attr=node_crystal_attr,
        
        sym_mask=sym_mask.detach().cpu(), # transfer to cpu
        diel_mask=diel_mask.detach().cpu(),
    )

specie_am = [Atom(z).mass for z in range(1, 119)]
am_onehot = torch.diag(torch.tensor(specie_am)).tolist()

def type_onehot(number: int, max_number: int = 118):
    """Onehot encode an atom number into the type encoding."""
    embedding = [0.0] * max_number
    embedding[number - 1] = 1.0
    return embedding


def mass_onehot(number: int):
    """One hot encode an atom number into the mass encoding."""
    return am_onehot[number - 1]

# 晶体系统 one-hot 编码表
crystal_systems = ["Triclinic", "Monoclinic", "Orthorhombic", "Tetragonal", "Trigonal", "Hexagonal", "Cubic"]
crystal_system_map = {name: i for i, name in enumerate(crystal_systems)}

def onehot_crystal_system(name):
    encoding = [0.0] * len(crystal_systems)
    if name in crystal_system_map:
        encoding[crystal_system_map[name]] = 1.0
    return encoding

def is_invariant(atom, symmat, symprec: float = 0.1):
    '''
    Check if the structure is invariant under symmetry operation
    '''
    positions = atom.get_positions()
    transformed_positions = positions @ symmat
    cell = np.array(atom.get_cell())

    species = atom.get_atomic_numbers()

    atom.set_positions(transformed_positions)
    transformed_fractions = atom.get_scaled_positions()

    atom.set_positions(positions)
    fractions = atom.get_scaled_positions()

    transformed_fractions -= transformed_fractions[0]
    for site in fractions:
        translated_fractions = fractions - site
        dist_map = (translated_fractions[np.newaxis, :, :] - transformed_fractions[:, np.newaxis, :] + 0.5 ) % 1 - 0.5
        dist_map = dist_map @ cell
        dist_map = np.sqrt(np.sum(dist_map ** 2, (2)))
        min_idx = np.argmin(dist_map, 0)
        min_dist = np.min(dist_map, 0)
        # print('--------------------\n')
        # print('\n'.join([" ".join(["{0:.2f}".format(i) for i in j]) for j in dist_map]))
        # print('--------------------\n')
        # print(" ".join(["{0:.2f}".format(i) for i in min_dist]))
        # print(symprec)
        if np.sum(min_dist) > symprec:
            continue
        new_species = species[min_idx]
        # print(min_idx)
        # print(new_species - species)
        if np.sum(new_species - species) > 0:
            continue

        return 1.0
    return 0.0
        
def check_symmetry_parallel(atom, symprec: float = 0.1, device = None):

    cell = torch.tensor(atom.get_cell(), device=device)
    species = torch.tensor(atom.get_atomic_numbers(), device=device)
    
    positions = torch.tensor(atom.get_positions(), device=device)
    positions = positions - positions[0]   # shift the first atom to the origin
    natom, _ = positions.shape
    
    symmats = torch.tensor(symmetry_operations, dtype=torch.double, device=device)
    transformed_positions = torch.matmul(positions, symmats) # transformed postion in cartesian coordinates

    positions = positions.repeat(48, natom, 1, 1) # (48, natom, natom, 3): (Nsym, Nshift, Natom, 3)
    fractions = positions @ torch.inverse(cell)
    transformed_positions = transformed_positions.repeat(natom, 1, 1, 1).transpose(0, 1) # (48, natom, natom, 3): (Nsym, Nshift, Natom, 3)
    transformed_fractions = transformed_positions @ torch.inverse(cell) # to fractional coordinates

    fractions_shifts = fractions.transpose(1, 2)
    fractions = fractions - fractions_shifts # shift different atoms to the origin

    dist_map = (fractions[:, :, :, torch.newaxis, :] - transformed_fractions[:, :, torch.newaxis, :, :] + 0.5) % 1 - 0.5
    dist_map = dist_map @ cell
    dist_map = torch.sqrt(torch.sum(dist_map ** 2, -1)) # (Nsym, Nshift, Natom1, Natom2)
    values, indices = torch.min(dist_map, -1) # find minimal (Nsym, Nshift)
    new_species = species[indices]

    species_diff = torch.sum(torch.abs(new_species - species), -1)
    dist_min, _ = torch.max(values, -1)
    crit = (species_diff < 0.1 ) & (dist_min < symprec) # with same species on same sites
    crit = torch.sum(crit, 1)

    sym_mask = torch.sign(crit)
    return sym_mask

def check_symmetry_serial(atom, symprec: float = 0.1, device = None):

    cell = torch.tensor(atom.get_cell(), device=device)
    species = torch.tensor(atom.get_atomic_numbers(), device=device)
    
    positions = torch.tensor(atom.get_positions(), device=device)
    positions0 = positions - positions[0]   # shift the first atom to the origin
    natom, _ = positions0.shape

    sym_mask = []
    for symmetry_operation in symmetry_operations:
        symmats = torch.tensor(symmetry_operation, dtype=torch.double, device=device).unsqueeze(0)
        transformed_positions = torch.matmul(positions0, symmats) # transformed postion in cartesian coordinates

        positions = positions0.repeat(1, natom, 1, 1) # (48, natom, natom, 3): (Nsym, Nshift, Natom, 3)
        fractions = positions @ torch.inverse(cell)
        transformed_positions = transformed_positions.repeat(natom, 1, 1, 1).transpose(0, 1) # (48, natom, natom, 3): (Nsym, Nshift, Natom, 3)
        transformed_fractions = transformed_positions @ torch.inverse(cell) # to fractional coordinates

        fractions_shifts = fractions.transpose(1, 2)
        fractions = fractions - fractions_shifts # shift different atoms to the origin

        dist_map = fractions[:, :, :, torch.newaxis, :] - transformed_fractions[:, :, torch.newaxis, :, :]
        dist_map = torch.sqrt(torch.sum(dist_map ** 2, -1)) # (Nsym, Nshift, Natom1, Natom2)

        values, indices = torch.min(dist_map, -1) # find minimal (Nsym, Nshift)
        new_species = species[indices]

        species_diff = torch.sum(torch.abs(new_species - species), -1)
        dist_min = torch.sum(values, -1)

        crit = (species_diff < 0.1 ) & (dist_min < symprec) # with same species on same sites
        crit = torch.sum(crit, 1)
        sym_mask.append(torch.sign(crit))

    sym_mask = torch.concat(sym_mask)

    return sym_mask

#静态掩码
def cal_diel_mask(atoms, sym_mask, device):
    symtensor = torch.tensor(symmetry_operations, dtype=torch.double, device=device)
    diel_mask = torch.ones_like(symtensor)
    diel_mask = torch.matmul(diel_mask, symtensor)
    symtensor = torch.transpose(symtensor, 1, 2)
    diel_mask = torch.matmul(symtensor, diel_mask)
    diel_mask = sym_mask[:, torch.newaxis] * diel_mask.view(-1, 9)
    diel_mask = torch.sum(diel_mask, 0)
    diel_mask = diel_mask.view(3, 3) / torch.max(diel_mask)
    return diel_mask

#符合对称性的介电张量输出
def apply_sym_mask(diel_tensors, sym_mask):
    symtensor = torch.tensor(symmetry_operations, dtype=torch.double, device=diel_tensors.device)
    diel_tensors = torch.matmul(diel_tensors[:, torch.newaxis, :, :], symtensor)
    symtensor = torch.transpose(symtensor, 1, 2)
    diel_tensors = torch.matmul(symtensor, diel_tensors)
    sym_mask = sym_mask.repeat(3, 3, 1, 1).transpose(0, 2).transpose(1, 3)
    diel_tensors = sym_mask * diel_tensors
    diel_tensors = torch.mean(diel_tensors, 1)
    return diel_tensors