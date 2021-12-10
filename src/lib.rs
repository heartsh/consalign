extern crate consprob;

pub use consprob::*;
pub use petgraph::{Graph, Directed, Outgoing};
pub use petgraph::graph::{DefaultIx, NodeIndex};
pub use rand::seq::SliceRandom;
pub use rand::Rng;

pub type Col = Vec<Base>;
pub type Cols = Vec<Col>;
pub type PosMaps<T> = Vec<T>;
pub type PosMapSets<T> = Vec<PosMaps<T>>;
#[derive(Debug)]
pub struct MeaSeqAlign<T> {
  pub cols: Cols,
  pub rightmost_bp_cols_with_cols: ColsWithCols<T>,
  pub right_bp_col_sets_with_cols: ColSetsWithCols<T>,
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
pub type ColPairs<T> = HashSet<PosPair<T>>;
pub type ColsWithCols<T> = HashMap<T, T>;
pub type MeaMatsWithPosPairs<T> = HashMap<PosPair<T>, SparseProbMat<T>>;
pub type ColSetsWithCols<T> = HashMap<T, PosProbSeq<T>>;
pub type PosProbSeq<T> = Vec<(T, Prob)>;

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

impl<T: Clone + Copy + Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Sync + Send + Display> MeaSeqAlign<T> {
  pub fn new() -> MeaSeqAlign<T> {
    MeaSeqAlign {
      cols: Cols::new(),
      rightmost_bp_cols_with_cols: ColsWithCols::<T>::default(),
      right_bp_col_sets_with_cols: ColSetsWithCols::<T>::default(),
      pos_map_sets: PosMapSets::<T>::new(),
      rna_ids: RnaIds::new(),
      ea: 0.,
    }
  }

  pub fn copy_subset(&mut self, mea_seq_align: &MeaSeqAlign<T>, indexes: &Vec<usize>, prob_mat_sets: &ProbMatSets<T>, min_bpp: Prob) {
    let cols_len = mea_seq_align.cols.len();
    let num_of_rnas = mea_seq_align.cols[0].len();
    let col_gaps_only = vec![PSEUDO_BASE; num_of_rnas];
    let mut cols_subset = Cols::new();
    let mut pos_map_sets_subset = PosMapSets::new();
    for i in 0 .. cols_len {
      let ref col = mea_seq_align.cols[i];
      if *col == col_gaps_only {
        continue;
      }
      let ref pos_maps = mea_seq_align.pos_map_sets[i];
      let mut col_subset = Col::new();
      let mut pos_maps_subset = PosMaps::new();
      for (j, &c) in col.iter().enumerate() {
        let pos_map = pos_maps[j];
        if indexes.contains(&j) {
          col_subset.push(c);
          pos_maps_subset.push(pos_map);
        }
      }
      cols_subset.push(col_subset);
      pos_map_sets_subset.push(pos_maps_subset);
    }
    self.cols = cols_subset.clone();
    self.pos_map_sets = pos_map_sets_subset.clone();
    let mut rna_ids_subset = RnaIds::new();
    for (i, &r) in mea_seq_align.rna_ids.iter().enumerate() {
      if indexes.contains(&i) {
        rna_ids_subset.push(r);
      }
    }
    self.rna_ids = rna_ids_subset;
    self.set_right_bp_info(prob_mat_sets, min_bpp);
    self.ea = 0.;
  }

  pub fn set_right_bp_info(&mut self, prob_mat_sets: &ProbMatSets<T>, min_bpp: Prob) {
    let sa_len = self.cols.len();
    for i in 0 .. sa_len {
      let ref pos_maps = self.pos_map_sets[i];
      for j in i + 1 .. sa_len {
        let ref pos_maps_2 = self.pos_map_sets[j];
        let pos_map_pairs: Vec<(T, T)> = pos_maps.iter().zip(pos_maps_2.iter()).map(|(&x, &y)| (x, y)).collect();
        let mut bpp_sum = 0.;
        let mut count = 0;
        for (pos_map_pair, &rna_id) in pos_map_pairs.iter().zip(self.rna_ids.iter()) {
          let ref bpp_mat = prob_mat_sets[rna_id].bpp_mat;
          match bpp_mat.get(pos_map_pair) {
            Some(&bpp) => {
              bpp_sum += bpp;
              count += 1;
            }, None => {},
          }
        }
        let bpp_avg = if count > 0 {bpp_sum / count as Prob} else {0.};
        if bpp_avg >= min_bpp {
          let (i, j) = (T::from_usize(i).unwrap() + T::one(), T::from_usize(j).unwrap() + T::one());
          match self.right_bp_col_sets_with_cols.get_mut(&i) {
            Some(right_bp_cols) => {
              right_bp_cols.push((j, bpp_avg));
            }, None => {
              let mut right_bp_cols = PosProbSeq::<T>::new();
              right_bp_cols.push((j, bpp_avg));
              self.right_bp_col_sets_with_cols.insert(i, right_bp_cols);
            },
          }
        }
      }
      let i = T::from_usize(i).unwrap() + T::one();
      match self.right_bp_col_sets_with_cols.get(&i) {
        Some(right_bp_cols) => {
          let max = right_bp_cols.iter().map(|x| x.0).max().unwrap();
          self.rightmost_bp_cols_with_cols.insert(i, max);
        }, None => {},
      }
    }
  }
}

pub const GAP: Char = '-' as Char;
pub const MAX_ITER_REFINE: usize = 100;

pub fn consalign<T>(
  fasta_records: &FastaRecords,
  align_prob_mat_pairs_with_rna_id_pairs: &AlignProbMatPairsWithRnaIdPairs<T>,
  offset_4_max_gap_num: T,
  prob_mat_sets: &ProbMatSets<T>,
  min_bpp: Prob,
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
    let ref sparse_bpp_mat = prob_mat_sets[i].bpp_mat;
    let converted_seq = convert_seq(seq, i, sparse_bpp_mat);
    for j in i + 1 .. num_of_rnas {
      let ref seq_2 = fasta_records[j].seq;
      let ref sparse_bpp_mat_2 = prob_mat_sets[j].bpp_mat;
      let converted_seq_2 = convert_seq(seq_2, j, sparse_bpp_mat_2);
      let pair_seq_align = get_mea_align(&(&converted_seq, &converted_seq_2), align_prob_mat_pairs_with_rna_id_pairs, offset_4_max_gap_num, prob_mat_sets, min_bpp);
      mea_mat.insert((i, j), pair_seq_align.ea);
    }
    let node_index = progressive_tree.add_node(i);
    cluster_sizes.insert(i, 1);
    node_indexes.insert(i, node_index);
  }
  let mut new_cluster_id = num_of_rnas;
  while mea_mat.len() > 0 {
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
  let mut mea_seq_align = recursive_mea_seq_align(&progressive_tree, root, align_prob_mat_pairs_with_rna_id_pairs, &fasta_records, offset_4_max_gap_num, prob_mat_sets, min_bpp);
  for i in 0 .. MAX_ITER_REFINE {
    iter_refine_seq_align(&mut mea_seq_align, align_prob_mat_pairs_with_rna_id_pairs, offset_4_max_gap_num, prob_mat_sets, min_bpp);
  }
  for col in mea_seq_align.cols.iter_mut() {
    let mut pairs: Vec<(Base, RnaId)> = col.iter().zip(mea_seq_align.rna_ids.iter()).map(|(&x, &y)| (x, y)).collect();
    pairs.sort_by_key(|x| x.1.clone());
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

pub fn iter_refine_seq_align<T>(mea_seq_align: &mut MeaSeqAlign<T>, align_prob_mat_pairs_with_rna_id_pairs: &AlignProbMatPairsWithRnaIdPairs<T>, offset_4_max_gap_num: T, prob_mat_sets: &ProbMatSets<T>, min_bpp: Prob)
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Sync + Send + Display,
{
  let mut indexes: Vec<usize> = (0 .. mea_seq_align.rna_ids.len()).collect();
  let indexes_len = indexes.len();
  let mut rng = rand::thread_rng();
  indexes.shuffle(&mut rng);
  let split_at = rng.gen_range(1 .. indexes_len);
  let indexes_remain = indexes.split_off(split_at);
  let mut split_pair = (MeaSeqAlign::<T>::new(), MeaSeqAlign::<T>::new());
  split_pair.0.copy_subset(mea_seq_align, &indexes, prob_mat_sets, min_bpp);
  split_pair.1.copy_subset(mea_seq_align, &indexes_remain, prob_mat_sets, min_bpp);
  let tmp_align = get_mea_align(&(&split_pair.0, &split_pair.1), align_prob_mat_pairs_with_rna_id_pairs, offset_4_max_gap_num, prob_mat_sets, min_bpp);
  if tmp_align.ea > mea_seq_align.ea {
    *mea_seq_align = tmp_align;
  }
}

pub fn recursive_mea_seq_align<T>(progressive_tree: &ProgressiveTree, node: NodeIndex<DefaultIx>, align_prob_mat_pairs_with_rna_id_pairs: &AlignProbMatPairsWithRnaIdPairs<T>, fasta_records: &FastaRecords, offset_4_max_gap_num: T, prob_mat_sets: &ProbMatSets<T>, min_bpp: Prob) -> MeaSeqAlign<T>
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Sync + Send + Display,
{
  let num_of_rnas = fasta_records.len();
  let rna_id = *progressive_tree.node_weight(node).unwrap();
  if rna_id < num_of_rnas {
    let ref seq = fasta_records[rna_id].seq;
    let ref sparse_bpp_mat = prob_mat_sets[rna_id].bpp_mat;
    convert_seq(seq, rna_id, sparse_bpp_mat)
  } else {
    let mut neighbors = progressive_tree.neighbors_directed(node, Outgoing).detach();
    let child = neighbors.next_node(progressive_tree).unwrap();
    let child_mea_seq_align = recursive_mea_seq_align(progressive_tree, child, align_prob_mat_pairs_with_rna_id_pairs, fasta_records, offset_4_max_gap_num, prob_mat_sets, min_bpp);
    let child_2 = neighbors.next_node(progressive_tree).unwrap();
    let child_mea_seq_align_2 = recursive_mea_seq_align(progressive_tree, child_2, align_prob_mat_pairs_with_rna_id_pairs, fasta_records, offset_4_max_gap_num, prob_mat_sets, min_bpp);
    get_mea_align(&(&child_mea_seq_align, &child_mea_seq_align_2), align_prob_mat_pairs_with_rna_id_pairs, offset_4_max_gap_num, prob_mat_sets, min_bpp)
  }
}

pub fn convert_seq<T>(seq: &Seq, rna_id: RnaId, sparse_bpp_mat: &SparseProbMat<T>) -> MeaSeqAlign<T>
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Sync + Send + Display,
{
  let mut converted_seq = MeaSeqAlign::new();
  let seq_len = seq.len();
  converted_seq.cols = seq[1 .. seq_len - 1].iter().map(|&x| vec![x]).collect();
  converted_seq.pos_map_sets = (1 .. seq.len() - 1).map(|x| vec![T::from_usize(x).unwrap()]).collect();
  converted_seq.rna_ids = vec![rna_id];
  for (pos_pair, &bpp) in sparse_bpp_mat {
    let (i, j) = *pos_pair;
    match converted_seq.right_bp_col_sets_with_cols.get_mut(&i) {
      Some(right_bp_cols) => {
        right_bp_cols.push((j, bpp));
      }, None => {
        let mut right_bp_cols = PosProbSeq::<T>::new();
        right_bp_cols.push((j, bpp));
        converted_seq.right_bp_col_sets_with_cols.insert(i, right_bp_cols);
      },
    }
    for (&i, right_bp_cols) in &converted_seq.right_bp_col_sets_with_cols {
      let max = right_bp_cols.iter().map(|x| x.0).max().unwrap();
      converted_seq.rightmost_bp_cols_with_cols.insert(i, max);
    }
  }
  converted_seq
}

pub fn get_mea_align<'a, T>(seq_align_pair: &MeaSeqAlignPair<'a, T>, align_prob_mat_pairs_with_rna_id_pairs: &AlignProbMatPairsWithRnaIdPairs<T>, offset_4_max_gap_num: T, prob_mat_sets: &ProbMatSets<T>, min_bpp: Prob) -> MeaSeqAlign<T>
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Sync + Send + Display,
{
  let seq_align_len_pair = (seq_align_pair.0.cols.len(), seq_align_pair.1.cols.len());
  let max_gap_num = offset_4_max_gap_num
    + T::from_usize(max(seq_align_len_pair.0, seq_align_len_pair.1) - min(seq_align_len_pair.0, seq_align_len_pair.1))
      .unwrap();
  let max_gap_num_4_il = max(
    min(max_gap_num, T::from_usize(MAX_GAP_NUM_4_IL).unwrap()),
    T::from_usize(MIN_GAP_NUM_4_IL).unwrap(),
  );
  let mut align_prob_mat_avg = SparseProbMat::<T>::default();
  let ref rna_ids = seq_align_pair.0.rna_ids;
  let ref rna_ids_2 = seq_align_pair.1.rna_ids;
  let ref pos_map_sets = seq_align_pair.0.pos_map_sets;
  let ref pos_map_sets_2 = seq_align_pair.1.pos_map_sets;
  let seq_align_len_pair = (
    T::from_usize(seq_align_len_pair.0).unwrap(),
    T::from_usize(seq_align_len_pair.1).unwrap(),
  );
  let pseudo_col_quadruple = (
    T::zero(),
    seq_align_len_pair.0 + T::one(),
    T::zero(),
    seq_align_len_pair.1 + T::one(),
  );
  let max = max(max_gap_num, max_gap_num_4_il);
  for i in range_inclusive(T::one(), seq_align_len_pair.0) {
    let long_i = i.to_usize().unwrap();
    let ref pos_maps = pos_map_sets[long_i - 1];
    for j in range_inclusive(T::one(), seq_align_len_pair.1) {
      let col_pair = (i, j);
      if !is_min_gap_ok(&col_pair, &pseudo_col_quadruple, max) {continue;}
      let long_j = j.to_usize().unwrap();
      let ref pos_maps_2 = pos_map_sets_2[long_j - 1];
      let mut align_prob_sum = 0.;
      let mut count = 0;
      for (&rna_id, &pos) in rna_ids.iter().zip(pos_maps.iter()) {
        for (&rna_id_2, &pos_2) in rna_ids_2.iter().zip(pos_maps_2.iter()) {
          let ordered_rna_id_pair = if rna_id < rna_id_2 {(rna_id, rna_id_2)} else {(rna_id_2, rna_id)};
          let ref align_prob_mat = align_prob_mat_pairs_with_rna_id_pairs[&ordered_rna_id_pair].align_prob_mat;
          let pos_pair = if rna_id < rna_id_2 {(pos, pos_2)} else {(pos_2, pos)};
          match align_prob_mat.get(&pos_pair) {
            Some(&align_prob) => {
              align_prob_sum += align_prob;
              count += 1;
            }, None => {},
          }
        }
      }
      let align_prob_avg = if count > 0 {align_prob_sum / count as Prob} else {0.};
      align_prob_mat_avg.insert(col_pair, align_prob_avg);
    }
  }
  let mut mea_mats_with_col_pairs = MeaMatsWithPosPairs::default();
  for i in range_inclusive(T::one(), seq_align_len_pair.0).rev() {
    match seq_align_pair.0.rightmost_bp_cols_with_cols.get(&i) {
      Some(&j) => {
        for k in range_inclusive(T::one(), seq_align_len_pair.1).rev() {
          let col_pair_left = (i, k);
          if !is_min_gap_ok(&col_pair_left, &pseudo_col_quadruple, max_gap_num_4_il) {continue;}
          match seq_align_pair.1.rightmost_bp_cols_with_cols.get(&k) {
            Some(&l) => {
              let col_quadruple = (i, j, k, l);
              let mea_mat = get_mea_mat(&mea_mats_with_col_pairs, max_gap_num, max_gap_num_4_il, &pseudo_col_quadruple, &align_prob_mat_avg, &col_quadruple, true);
              update_mea_mats_with_col_pairs(&mut mea_mats_with_col_pairs, &col_pair_left, seq_align_pair, &mea_mat, align_prob_mat_pairs_with_rna_id_pairs, max_gap_num_4_il, &pseudo_col_quadruple, &align_prob_mat_avg);
            }, None => {},
          }
        }
      }, None => {},
    }
  }
  let mea_mat = get_mea_mat(&mea_mats_with_col_pairs, max_gap_num, max_gap_num_4_il, &pseudo_col_quadruple, &align_prob_mat_avg, &pseudo_col_quadruple, false);
  let mut new_mea_seq_align = MeaSeqAlign::new();
  new_mea_seq_align.ea = mea_mat[&(pseudo_col_quadruple.1 - T::one(), pseudo_col_quadruple.3 - T::one())];
  let rna_num_pair = (seq_align_pair.0.rna_ids.len(), seq_align_pair.1.rna_ids.len());
  let mut new_rna_ids = seq_align_pair.0.rna_ids.clone();
  let mut rna_ids_append = seq_align_pair.1.rna_ids.clone();
  new_rna_ids.append(&mut rna_ids_append);
  new_mea_seq_align.rna_ids = new_rna_ids;
  traceback(&mut new_mea_seq_align, seq_align_pair, max_gap_num, max_gap_num_4_il, &pseudo_col_quadruple, &pseudo_col_quadruple, &mea_mats_with_col_pairs, 0, &align_prob_mat_avg);
  new_mea_seq_align.set_right_bp_info(prob_mat_sets, min_bpp);
  new_mea_seq_align
}

pub fn traceback <'a, T>(new_mea_seq_align: &mut MeaSeqAlign<T>, seq_align_pair: &MeaSeqAlignPair<'a, T>, max_gap_num: T, max_gap_num_4_il: T, pseudo_col_quadruple: &PosQuadruple<T>, col_quadruple: &PosQuadruple<T>, mea_mats_with_col_pairs: &MeaMatsWithPosPairs<T>, offset: usize, align_prob_mat_avg: &SparseProbMat<T>)
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Sync + Send + Display,
{
  let rna_num_pair = (seq_align_pair.0.rna_ids.len(), seq_align_pair.1.rna_ids.len());
  let mut mea;
  let mea_mat = get_mea_mat(&mea_mats_with_col_pairs, max_gap_num, max_gap_num_4_il, &pseudo_col_quadruple, &align_prob_mat_avg, &col_quadruple, col_quadruple != pseudo_col_quadruple);
  let (i, j, k, l) = *col_quadruple;
  let (mut u, mut v) = (j - T::one(), l - T::one());
  while u > i || v > k {
    let col_pair = (u, v);
    mea = mea_mat[&col_pair];
    let (long_u, long_v) = (u.to_usize().unwrap(), v.to_usize().unwrap());
    if u > i && v > k {
      let align_prob_avg = align_prob_mat_avg[&col_pair];
      let col_pair_4_match = (u - T::one(), v - T::one());
      let ea = mea_mat[&col_pair_4_match] + align_prob_avg;
      if ea == mea {
        let mut new_col = seq_align_pair.0.cols[long_u - 1].clone();
        let mut col_append = seq_align_pair.1.cols[long_v - 1].clone();
        new_col.append(&mut col_append);
        new_mea_seq_align.cols.insert(offset, new_col);
        let mut new_pos_map_sets = seq_align_pair.0.pos_map_sets[long_u - 1].clone();
        let mut pos_map_sets_append = seq_align_pair.1.pos_map_sets[long_v - 1].clone();
        new_pos_map_sets.append(&mut pos_map_sets_append);
        new_mea_seq_align.pos_map_sets.insert(offset, new_pos_map_sets);
        u = u - T::one();
        v = v - T::one();
        continue;
      }
      let mut is_basepair_match_found = false;
      match mea_mats_with_col_pairs.get(&col_pair) {
        Some(mea_mat_4_bpas) => {
          for (col_pair_left, mea_4_bpa) in mea_mat_4_bpas {
            if !(i < col_pair_left.0 && k < col_pair_left.1) {continue;}
            let col_pair_4_match = (col_pair_left.0 - T::one(), col_pair_left.1 - T::one());
            match mea_mat.get(&col_pair_4_match) {
              Some(&ea) => {
                let ea = ea + mea_4_bpa;
                if ea == mea {
                  let mut new_col = seq_align_pair.0.cols[long_u - 1].clone();
                  let mut col_append = seq_align_pair.1.cols[long_v - 1].clone();
                  new_col.append(&mut col_append);
                  new_mea_seq_align.cols.insert(offset, new_col);
                  let mut new_pos_map_sets = seq_align_pair.0.pos_map_sets[long_u - 1].clone();
                  let mut pos_map_sets_append = seq_align_pair.1.pos_map_sets[long_v - 1].clone();
                  new_pos_map_sets.append(&mut pos_map_sets_append);
                  new_mea_seq_align.pos_map_sets.insert(offset, new_pos_map_sets);
                  let long_col_pair_left = (col_pair_left.0.to_usize().unwrap(), col_pair_left.1.to_usize().unwrap());
                  let mut new_col = seq_align_pair.0.cols[long_col_pair_left.0 - 1].clone();
                  let mut col_append = seq_align_pair.1.cols[long_col_pair_left.1 - 1].clone();
                  new_col.append(&mut col_append);
                  new_mea_seq_align.cols.insert(offset, new_col);
                  let mut new_pos_map_sets = seq_align_pair.0.pos_map_sets[long_col_pair_left.0 - 1].clone();
                  let mut pos_map_sets_append = seq_align_pair.1.pos_map_sets[long_col_pair_left.1 - 1].clone();
                  new_pos_map_sets.append(&mut pos_map_sets_append);
                  new_mea_seq_align.pos_map_sets.insert(offset, new_pos_map_sets);
                  traceback(new_mea_seq_align, seq_align_pair, max_gap_num, max_gap_num_4_il, pseudo_col_quadruple, &(col_pair_left.0, u, col_pair_left.1, v), mea_mats_with_col_pairs, offset + 1, align_prob_mat_avg);
                  u = col_pair_4_match.0;
                  v = col_pair_4_match.1;
                  is_basepair_match_found = true;
                  break;
                }
              }, None => {},
            }
          }
        }, None => {},
      }
      if is_basepair_match_found {
        continue;
      }
    }
    if u > i {
      match mea_mat.get(&(u - T::one(), v)) {
        Some(&ea) => {
          if ea == mea {
            let mut new_col = seq_align_pair.0.cols[long_u - 1].clone();
            let mut col_append = vec![PSEUDO_BASE; rna_num_pair.1];
            new_col.append(&mut col_append);
            new_mea_seq_align.cols.insert(offset, new_col);
            let mut new_pos_map_sets = seq_align_pair.0.pos_map_sets[long_u - 1].clone();
            let mut pos_map_sets_append = vec![T::zero(); rna_num_pair.1];
            new_pos_map_sets.append(&mut pos_map_sets_append);
            new_mea_seq_align.pos_map_sets.insert(offset, new_pos_map_sets);
            u = u - T::one();
            continue;
          }
        }, None => {},
      }
    }
    if v > k {
      match mea_mat.get(&(u, v - T::one())) {
        Some(&ea) => {
          if ea == mea {
            let mut new_col = vec![PSEUDO_BASE; rna_num_pair.0];
            let mut col_append = seq_align_pair.1.cols[long_v - 1].clone();
            new_col.append(&mut col_append);
            new_mea_seq_align.cols.insert(offset, new_col);
            let mut new_pos_map_sets = vec![T::zero(); rna_num_pair.0];
            let mut pos_map_sets_append = seq_align_pair.1.pos_map_sets[long_v - 1].clone();
            new_pos_map_sets.append(&mut pos_map_sets_append);
            new_mea_seq_align.pos_map_sets.insert(offset, new_pos_map_sets);
            v = v - T::one();
          }
        }, None => {},
      }
    }
  }
}

pub fn get_mea_mat<T>(mea_mats_with_col_pairs: &MeaMatsWithPosPairs<T>, max_gap_num: T, max_gap_num_4_il: T, pseudo_col_quadruple: &PosQuadruple<T>, align_prob_mat_avg: &SparseProbMat<T>, col_quadruple: &PosQuadruple<T>, is_internal: bool) -> SparseProbMat<T>
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Sync + Send + Display,
{
  let (i, j, k, l) = *col_quadruple;
  let mut mea_mat = SparseProbMat::<T>::default();
  for u in range(i, j) {
    for v in range(k, l) {
      let col_pair = (u, v);
      if !is_min_gap_ok(&col_pair, pseudo_col_quadruple, if is_internal {max_gap_num_4_il} else {max_gap_num}) {continue;}
      if u == i && v == k {
        mea_mat.insert(col_pair, 0.);
        continue;
      }
      let mut mea = NEG_INFINITY;
      match mea_mats_with_col_pairs.get(&col_pair) {
        Some(mea_mat_4_bpas) => {
          for (col_pair_left, mea_4_bpa) in mea_mat_4_bpas {
            let col_pair_4_match = (col_pair_left.0 - T::one(), col_pair_left.1 - T::one());
            if !(i < col_pair_left.0 && k < col_pair_left.1) {continue;}
            match mea_mat.get(&col_pair_4_match) {
              Some(&ea) => {
                let ea = ea + mea_4_bpa;
                if ea > mea {
                  mea = ea;
                }
              }, None => {},
            }
          }
        }, None => {},
      }
      let col_pair_4_match = (u - T::one(), v - T::one());
      match mea_mat.get(&col_pair_4_match) {
        Some(&ea) => {
          let align_prob_avg = align_prob_mat_avg[&col_pair];
          let ea = ea + align_prob_avg;
          if ea > mea {
            mea = ea;
          }
        }, None => {},
      }
      let col_pair_4_insert = (u - T::one(), v);
      match mea_mat.get(&col_pair_4_insert) {
        Some(&ea) => {
          if ea > mea {
            mea = ea;
          }
        }, None => {},
      }
      let col_pair_4_insert_2 = (u, v - T::one());
      match mea_mat.get(&col_pair_4_insert_2) {
        Some(&ea) => {
          if ea > mea {
            mea = ea;
          }
        }, None => {},
      }
      mea_mat.insert(col_pair, mea);
    }
  }
  mea_mat
}

pub fn update_mea_mats_with_col_pairs<'a, T>(mea_mats_with_col_pairs: &mut MeaMatsWithPosPairs<T>, col_pair_left: &PosPair<T>, seq_align_pair: &MeaSeqAlignPair<'a, T>, mea_mat: &SparseProbMat<T>, align_prob_mat_pairs_with_rna_id_pairs: &AlignProbMatPairsWithRnaIdPairs<T>, max_gap_num_4_il: T, pseudo_col_quadruple: &PosQuadruple<T>, align_prob_mat_avg: &SparseProbMat<T>)
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Sync + Send + Display,
{
  let (i, k) = *col_pair_left;
  let (long_i, long_k) = (i.to_usize().unwrap(), k.to_usize().unwrap());
  let ref rna_ids = seq_align_pair.0.rna_ids;
  let ref rna_ids_2 = seq_align_pair.1.rna_ids;
  let ref pos_map_sets = seq_align_pair.0.pos_map_sets;
  let ref pos_map_sets_2 = seq_align_pair.1.pos_map_sets;
  let ref right_bp_cols = seq_align_pair.0.right_bp_col_sets_with_cols[&i];
  let ref right_bp_cols_2 = seq_align_pair.1.right_bp_col_sets_with_cols[&k];
  let align_prob_avg_left = align_prob_mat_avg[&col_pair_left];
  for &(j, bpp_avg) in right_bp_cols.iter() {
    for &(l, bpp_avg_2) in right_bp_cols_2.iter() {
      let col_pair_right = (j, l);
      if !is_min_gap_ok(&col_pair_right, &pseudo_col_quadruple, max_gap_num_4_il) {continue;}
      let align_prob_avg_right = align_prob_mat_avg[&col_pair_right];
      let basepair_align_prob_avg = bpp_avg + bpp_avg_2 + align_prob_avg_left + align_prob_avg_right;
      let mea_4_bpa = basepair_align_prob_avg + mea_mat[&(j - T::one(), l - T::one())];
      match mea_mats_with_col_pairs.get_mut(&col_pair_right) {
        Some(mea_mat_4_bpas) => {
          mea_mat_4_bpas.insert(*col_pair_left, mea_4_bpa);
        }, None => {
          let mut mea_mat_4_bpas = SparseProbMat::default();
          mea_mat_4_bpas.insert(*col_pair_left, mea_4_bpa);
          mea_mats_with_col_pairs.insert(col_pair_right, mea_mat_4_bpas);
        },
      }
    }
  }
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
