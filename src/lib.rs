extern crate consprob;

pub use consprob::*;
pub use petgraph::{Graph, Directed, Outgoing};
pub use petgraph::graph::{DefaultIx, NodeIndex};

pub type Col = Vec<Base>;
pub type Cols = Vec<Col>;
pub type PosMaps<T> = Vec<T>;
pub type PosMapSets<T> = Vec<PosMaps<T>>;
#[derive(Debug)]
pub struct MeaSeqAlign<T> {
  pub cols: Cols,
  pub pos_map_sets: PosMapSets<T>,
  pub rna_ids: RnaIds,
  pub ea: Mea,
}
pub type RnaIds = Vec<RnaId>;
pub type MeaSeqAlignPair<'a, T> = (&'a MeaSeqAlign<T>, &'a MeaSeqAlign<T>);
pub type SparseMeaMat = HashMap<RnaIdPair, Mea>;
pub type ProgressiveTree = Graph<RnaId, Mea>;
pub type ClusterSizes = HashMap<RnaId, usize>;
pub type NodeIndexes = HashMap<RnaId, NodeIndex<DefaultIx>>;

pub struct MeaCss<T> {
  pub bpa_pos_pairs: PosPairs<T>,
  pub ea: Mea,
}

impl<T> MeaCss<T> {
  pub fn new() -> MeaCss<T> {
    MeaCss {
      bpa_pos_pairs: PosPairs::<T>::new(),
      ea: 0.,
    }
  }
}

impl<T> MeaSeqAlign<T> {
  pub fn new() -> MeaSeqAlign<T> {
    MeaSeqAlign {
      cols: Cols::new(),
      pos_map_sets: PosMapSets::<T>::new(),
      rna_ids: RnaIds::new(),
      ea: 0.,
    }
  }
}

pub const GAP: Char = '-' as Char;

pub fn consalign<T>(
  fasta_records: &FastaRecords,
  gamma: Prob,
  align_prob_mat_pairs_with_rna_id_pairs: &AlignProbMatPairsWithRnaIdPairs<T>,
) -> MeaSeqAlign<T>
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Sync + Send + Display,
{
  let num_of_rnas = fasta_records.len();
  let mut mea_mat = SparseMeaMat::default();
  let mut progressive_tree = ProgressiveTree::new();
  let mut cluster_sizes = ClusterSizes::default();
  let mut node_indexes = NodeIndexes::default();
  for i in 0 .. num_of_rnas {
    let ref seq = fasta_records[i].seq;
    let converted_seq = convert_seq(seq, i);
    for j in i + 1 .. num_of_rnas {
      let ref seq_2 = fasta_records[j].seq;
      let converted_seq_2 = convert_seq(seq_2, j);
      let pair_seq_align = get_mea_align(&(&converted_seq, &converted_seq_2), align_prob_mat_pairs_with_rna_id_pairs, gamma);
      mea_mat.insert((i, j), pair_seq_align.ea);
    }
    let node_index = progressive_tree.add_node(i);
    cluster_sizes.insert(i, 1);
    node_indexes.insert(i, node_index);
  }
  let mut new_cluster_id = num_of_rnas;
  while mea_mat.len() > 1 {
    let mut max = NEG_INFINITY;
    let mut argmax = (0, 0);
    for (cluster_id_pair, &ea) in &mea_mat {
      if ea > max {
        argmax = *cluster_id_pair;
        max = ea;
      }
    }
    let cluster_size_pair = (
      cluster_sizes.remove(&argmax.0).unwrap(),
      cluster_sizes.remove(&argmax.1).unwrap(),
    );
    let new_cluster_size = cluster_size_pair.0 + cluster_size_pair.1;
    cluster_sizes.insert(new_cluster_id, new_cluster_size);
    mea_mat.remove(&argmax);
    for &i in node_indexes.keys() {
      if i == argmax.0 || i == argmax.1 {
        continue;
      }
      let cluster_id_pair = if i < argmax.0 {(i, argmax.0)} else {(argmax.0, i)};
      let obtained_ea = mea_mat.remove(&cluster_id_pair).unwrap();
      let cluster_id_pair_2 = if i < argmax.1 {(i, argmax.1)} else {(argmax.1, i)};
      let obtained_ea_2 = mea_mat.remove(&cluster_id_pair_2).unwrap();
      let new_ea = (cluster_size_pair.0 as Mea * obtained_ea + cluster_size_pair.1 as Mea * obtained_ea_2) / new_cluster_size as Mea;
      mea_mat.insert((i, new_cluster_id), new_ea);
    }
    let new_node = progressive_tree.add_node(new_cluster_id);
    node_indexes.insert(new_cluster_id, new_node);
    let edge_len = max / 2.;
    let argmax_node_pair = (
      node_indexes.remove(&argmax.0).unwrap(),
      node_indexes.remove(&argmax.1).unwrap(),
    );
    progressive_tree.add_edge(new_node, argmax_node_pair.0, edge_len);
    progressive_tree.add_edge(new_node, argmax_node_pair.1, edge_len);
    new_cluster_id += 1;
  }
  let root = node_indexes[&(new_cluster_id - 1)];
  let mut mea_seq_align = recursive_mea_seq_align(&progressive_tree, root, align_prob_mat_pairs_with_rna_id_pairs, &fasta_records, gamma);
  for col in mea_seq_align.cols.iter_mut() {
    let mut pairs: Vec<(Base, RnaId)> = col.iter().zip(mea_seq_align.rna_ids.iter()).map(|(&x, &y)| (x, y)).collect();
    pairs.sort_by_key(|x| x.1);
    *col = pairs.iter().map(|x| x.0).collect();
  }
  for pos_maps in mea_seq_align.pos_map_sets.iter_mut() {
    let mut pairs: Vec<(T, RnaId)> = pos_maps.iter().zip(mea_seq_align.rna_ids.iter()).map(|(&x, &y)| (x, y)).collect();
    pairs.sort_by_key(|x| x.1);
    *pos_maps = pairs.iter().map(|x| x.0).collect();
  }
  mea_seq_align.rna_ids.sort();
  mea_seq_align
}

pub fn recursive_mea_seq_align<T>(progressive_tree: &ProgressiveTree, node: NodeIndex<DefaultIx>, align_prob_mat_pairs_with_rna_id_pairs: &AlignProbMatPairsWithRnaIdPairs<T>, fasta_records: &FastaRecords, gamma: Prob) -> MeaSeqAlign<T>
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Sync + Send + Display,
{
  let num_of_rnas = fasta_records.len();
  let rna_id = *progressive_tree.node_weight(node).unwrap();
  if rna_id < num_of_rnas {
    let ref seq = fasta_records[rna_id].seq;
    convert_seq(seq, rna_id)
  } else {
    let mut neighbors = progressive_tree.neighbors_directed(node, Outgoing).detach();
    let child = neighbors.next_node(progressive_tree).unwrap();
    let child_mea_seq_align = recursive_mea_seq_align(progressive_tree, child, align_prob_mat_pairs_with_rna_id_pairs, fasta_records, gamma);
    let child_2 = neighbors.next_node(progressive_tree).unwrap();
    let child_mea_seq_align_2 = recursive_mea_seq_align(progressive_tree, child_2, align_prob_mat_pairs_with_rna_id_pairs, fasta_records, gamma);
    get_mea_align(&(&child_mea_seq_align, &child_mea_seq_align_2), align_prob_mat_pairs_with_rna_id_pairs, gamma)
  }
}

pub fn convert_seq<T>(seq: &Seq, rna_id: RnaId) -> MeaSeqAlign<T>
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Sync + Send + Display,
{
  let mut converted_seq = MeaSeqAlign::new();
  let seq_len = seq.len();
  converted_seq.cols = seq[1 .. seq_len - 1].iter().map(|&x| vec![x]).collect();
  converted_seq.pos_map_sets = (1 .. seq.len() - 1).map(|x| vec![T::from_usize(x).unwrap()]).collect();
  converted_seq.rna_ids = vec![rna_id];
  converted_seq
}

pub fn get_mea_align<'a, T>(seq_align_pair: &MeaSeqAlignPair<'a, T>, align_prob_mat_pairs_with_rna_id_pairs: &AlignProbMatPairsWithRnaIdPairs<T>, gamma: Prob) -> MeaSeqAlign<T>
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Sync + Send + Display,
{
  let gamma_plus_1 = gamma + 1.;
  let seq_align_len_pair = (seq_align_pair.0.cols.len(), seq_align_pair.1.cols.len());
  let mut mea_mat = vec![vec![0.; seq_align_len_pair.1 + 1]; seq_align_len_pair.0 + 1];
  let mut align_prob_mat_avg = mea_mat.clone();
  let ref rna_ids = seq_align_pair.0.rna_ids;
  let ref rna_ids_2 = seq_align_pair.1.rna_ids;
  let ref pos_map_sets = seq_align_pair.0.pos_map_sets;
  let ref pos_map_sets_2 = seq_align_pair.1.pos_map_sets;
  for i in 0 .. seq_align_len_pair.0 + 1 {
    for j in 0 .. seq_align_len_pair.1 + 1 {
      let mut mea = 0.;
      if i > 0 && j > 0 {
        let ref pos_maps = pos_map_sets[i - 1];
        let ref pos_maps_2 = pos_map_sets_2[j - 1];
        let mut align_prob_sum = 0.;
        let mut count = 0;
        for (&rna_id, &pos) in rna_ids.iter().zip(pos_maps.iter()) {
          for (&rna_id_2, &pos_2) in rna_ids_2.iter().zip(pos_maps_2.iter()) {
            let ordered_rna_id_pair = if rna_id < rna_id_2 {(rna_id, rna_id_2)} else {(rna_id_2, rna_id)};
            let ref align_prob_mat = align_prob_mat_pairs_with_rna_id_pairs[&ordered_rna_id_pair].align_prob_mat;
            let pos_pair = (pos, pos_2);
            match align_prob_mat.get(&pos_pair) {
              Some(&align_prob) => {
                align_prob_sum += align_prob;
                count += 1;
              }, None => {},
            }
          }
        }
        let align_prob_avg = if count > 0 {align_prob_sum / count as Prob} else {0.};
        align_prob_mat_avg[i][j] = align_prob_avg;
        let ea = mea_mat[i - 1][j - 1] + gamma_plus_1 * align_prob_avg - 1.;
        if ea > mea {
          mea = ea;
        }
      }
      if i > 0 {
        let ea = mea_mat[i - 1][j];
        if ea > mea {
          mea = ea;
        }
      }
      if j > 0 {
        let ea = mea_mat[i][j - 1];
        if ea > mea {
          mea = ea;
        }
      }
      mea_mat[i][j] = mea;
    }
  }
  let mut new_mea_seq_align = MeaSeqAlign::new();
  let (mut i, mut j) = (seq_align_len_pair.0, seq_align_len_pair.1);
  let mut mea;
  new_mea_seq_align.ea = mea_mat[i][j];
  let rna_num_pair = (seq_align_pair.0.rna_ids.len(), seq_align_pair.1.rna_ids.len());
  let mut new_rna_ids = seq_align_pair.0.rna_ids.clone();
  let mut rna_ids_append = seq_align_pair.1.rna_ids.clone();
  new_rna_ids.append(&mut rna_ids_append);
  new_mea_seq_align.rna_ids = new_rna_ids;
  while i > 0 || j > 0 {
    mea = mea_mat[i][j];
    if i > 0 && j > 0 {
      let align_prob_avg = align_prob_mat_avg[i][j];
      let ea = mea_mat[i - 1][j - 1] + gamma_plus_1 * align_prob_avg - 1.;
      if ea == mea {
        let mut new_col = seq_align_pair.0.cols[i - 1].clone();
        let mut col_append = seq_align_pair.1.cols[j - 1].clone();
        new_col.append(&mut col_append);
        new_mea_seq_align.cols.insert(0, new_col);
        let mut new_pos_map_sets = seq_align_pair.0.pos_map_sets[i - 1].clone();
        let mut pos_map_sets_append = seq_align_pair.1.pos_map_sets[j - 1].clone();
        new_pos_map_sets.append(&mut pos_map_sets_append);
        new_mea_seq_align.pos_map_sets.insert(0, new_pos_map_sets);
        i -= 1;
        j -= 1;
        continue;
      }
    }
    if i > 0 {
      let ea = mea_mat[i - 1][j];
      if ea == mea {
        let mut new_col = seq_align_pair.0.cols[i - 1].clone();
        let mut col_append = vec![PSEUDO_BASE; rna_num_pair.1];
        new_col.append(&mut col_append);
        new_mea_seq_align.cols.insert(0, new_col);
        let mut new_pos_map_sets = seq_align_pair.0.pos_map_sets[i - 1].clone();
        let mut pos_map_sets_append = vec![T::zero(); rna_num_pair.1];
        new_pos_map_sets.append(&mut pos_map_sets_append);
        new_mea_seq_align.pos_map_sets.insert(0, new_pos_map_sets);
        i -= 1;
        continue;
      }
    }
    if j > 0 {
      let ea = mea_mat[i][j - 1];
      if ea == mea {
        let mut new_col = vec![PSEUDO_BASE; rna_num_pair.0];
        let mut col_append = seq_align_pair.1.cols[j - 1].clone();
        new_col.append(&mut col_append);
        new_mea_seq_align.cols.insert(0, new_col);
        let mut new_pos_map_sets = vec![T::zero(); rna_num_pair.0];
        let mut pos_map_sets_append = seq_align_pair.1.pos_map_sets[j - 1].clone();
        new_pos_map_sets.append(&mut pos_map_sets_append);
        new_mea_seq_align.pos_map_sets.insert(0, new_pos_map_sets);
        j -= 1;
      }
    }
  }
  new_mea_seq_align
}

pub fn revert_char(c: Base) -> u8 {
  match c {
    A => BIG_A,
    C => BIG_C,
    G => BIG_G,
    U => BIG_U,
    PSEUDO_BASE => {GAP},
    _ => {assert!(false); GAP},
  }
}

pub fn consalifold<T>(mix_bpp_mat: &ProbMat, gamma: Prob, sa: &MeaSeqAlign<T>) -> MeaCss<T>
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer,
{
  let sa_len = sa.cols.len();
  let mut mea_mat = vec![vec![0.; sa_len]; sa_len];
  let sa_len = T::from_usize(sa_len).unwrap();
  let gamma_plus_1 = gamma + 1.;
  for sub_sa_len in range_inclusive(T::one(), sa_len) {
    for i in range_inclusive(T::zero(), sa_len - sub_sa_len) {
      let j = i + sub_sa_len - T::one();
      let (long_i, long_j) = (i.to_usize().unwrap(), j.to_usize().unwrap());
      if i == j {
        continue;
      }
      let mut mea = mea_mat[long_i + 1][long_j];
      let ea = mea_mat[long_i][long_j - 1];
      if ea > mea {
        mea = ea;
      }
      let ea = mea_mat[long_i + 1][long_j - 1] + gamma_plus_1 * mix_bpp_mat[long_i][long_j] - 1.;
      if ea > mea {
        mea = ea;
      }
      for k in long_i .. long_j {
        let ea = mea_mat[long_i][k] + mea_mat[k + 1][long_j];
        if ea > mea {
          mea = ea;
        }
      }
      mea_mat[long_i][long_j] = mea;
    }
  }
  let mut mea_css = MeaCss::new();
  let mut pos_pair_stack = vec![(T::zero(), sa_len - T::one())];
  while pos_pair_stack.len() > 0 {
    let pos_pair = pos_pair_stack.pop().expect("Failed to pop an element of a vector.");
    let (i, j) = pos_pair;
    if j <= i {continue;}
    let (long_i, long_j) = (i.to_usize().unwrap(), j.to_usize().unwrap());
    let mea = mea_mat[long_i][long_j];
    if mea == mea_mat[long_i + 1][long_j] {
      pos_pair_stack.push((i + T::one(), j));
    } else if mea == mea_mat[long_i][long_j - 1] {
      pos_pair_stack.push((i, j - T::one()));
    } else if mea == mea_mat[long_i + 1][long_j - 1] + gamma_plus_1 * mix_bpp_mat[long_i][long_j] - 1. {
      pos_pair_stack.push((i + T::one(), j - T::one()));
      mea_css.bpa_pos_pairs.push(pos_pair);
    } else {
      for k in range(i, j) {
        let long_k = k.to_usize().unwrap();
        if mea == mea_mat[long_i][long_k] + mea_mat[long_k + 1][long_j] {
          pos_pair_stack.push((i, k));
          pos_pair_stack.push((k + T::one(), j));
          break;
        }
      }
    }
  }
  mea_css.ea = mea_mat[0][sa_len.to_usize().unwrap() - 1];
  mea_css
}
