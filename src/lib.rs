extern crate consprob;
extern crate num_cpus;
extern crate petgraph;
extern crate rand;

pub mod trained_feature_scores;

pub use consprob::*;
pub use petgraph::{Graph, Directed, Outgoing};
pub use petgraph::graph::{DefaultIx, NodeIndex};
pub use rand::seq::SliceRandom;
pub use rand::Rng;
pub use std::process::{Command, Output};
pub use std::fs::remove_file;
pub use std::fs::create_dir;
pub use std::io::{BufRead, BufWriter};
pub use std::fs::File;
pub use std::env;

pub type Col = Vec<Base>;
pub type Cols = Vec<Col>;
pub type PosMaps<T> = Vec<T>;
pub type PosMapSets<T> = Vec<PosMaps<T>>;
pub type PosMapSetPair<T> = (PosMaps<T>, PosMaps<T>);
pub type PosMapSetPairs<T> = Vec<PosMapSetPair<T>>;
#[derive(Clone)]
pub struct MeaStructAlign<T> {
  pub cols: Cols,
  pub rightmost_bp_cols_with_cols: ColsWithCols<T>,
  pub right_bp_col_sets_with_cols: ColSetsWithCols<T>,
  pub pos_map_sets: PosMapSets<T>,
  pub rna_ids: RnaIds,
  pub ea: Mea,
  pub bp_col_pairs: SparsePosMat<T>,
  pub mix_bpp_mat: SparseProbMat<T>,
  pub align_probs_avg: Probs,
}
#[derive(Clone)]
pub struct ReadStructAlign<T> {
  pub cols: Cols,
  pub bp_col_pairs: SparsePosMat<T>,
  pub pos_map_sets: PosMapSets<T>,
  pub fasta_records: FastaRecords,
}
pub type SparsePosMat<T> = HashSet<PosPair<T>>;
pub type RnaIds = Vec<RnaId>;
pub type MeaStructAlignPair<'a, T> = (&'a MeaStructAlign<T>, &'a MeaStructAlign<T>);
pub type SparseMeaMat = HashMap<RnaIdPair, Mea>;
pub type ProgressiveTree = Graph<RnaId, Mea>;
pub type ClusterSizes = HashMap<RnaId, usize>;
pub type NodeIndexes = HashMap<RnaId, NodeIndex<DefaultIx>>;
pub type ColPairs<T> = HashSet<PosPair<T>>;
pub type ColsWithCols<T> = HashMap<T, T>;
pub type MeaMatsWithPosPairs<T> = HashMap<PosPair<T>, SparseProbMat<T>>;
pub type ColSetsWithCols<T> = HashMap<T, PosProbSeq<T>>;
pub type PosProbSeq<T> = Vec<(T, Prob)>;
#[derive(Clone)]
pub struct TrainDatumPosterior<T> {
  pub fasta_records: FastaRecords,
  pub observed_struct_align: ReadStructAlign<T>,
  pub argmax: MeaStructAlign<T>,
  pub align_prob_mats_with_rna_id_pairs: ProbMatsWithRnaIdPairs<T>,
  pub bpp_mats: SparseProbMats<T>,
  pub sa_file_path: String,
}
pub type TrainDataPosterior<T> = Vec<TrainDatumPosterior<T>>;
#[derive(Clone, Debug)]
pub struct FeatureCountsPosterior {
  pub basepair_count_posterior: FeatureCount,
  pub align_count_posterior: FeatureCount,
}
pub type Accs = Vec<(FeatureCountsPosterior, Prob)>;
pub type SparseProbMats<T> = Vec<SparseProbMat<T>>;
pub type ProbMatsWithRnaIdPairs<T> = HashMap<RnaIdPair, SparseProbMat<T>>;

impl<T: Hash + Clone + Unsigned + PrimInt + FromPrimitive + Integer + Ord + Sync + Send + Display> TrainDatumPosterior<T> {
  pub fn origin() -> TrainDatumPosterior<T> {
    TrainDatumPosterior {
      fasta_records: FastaRecords::new(),
      observed_struct_align: ReadStructAlign::<T>::new(),
      argmax: MeaStructAlign::<T>::new(),
      align_prob_mats_with_rna_id_pairs: ProbMatsWithRnaIdPairs::<T>::default(),
      bpp_mats: SparseProbMats::<T>::default(),
      sa_file_path: String::new(),
    }
  }

  pub fn new(input_file_path: &Path, min_bpp: Prob, offset_4_max_gap_num: T, thread_pool: &mut Pool, mix_weight: Prob) -> TrainDatumPosterior<T> {
    let read_struct_align = read_sa_from_stockholm_file(input_file_path);
    let (prob_mat_sets, align_prob_mat_pairs_with_rna_id_pairs) = consprob::<T>(
      thread_pool,
      &read_struct_align.fasta_records,
      min_bpp,
      offset_4_max_gap_num,
      false,
      true,
      mix_weight,
    );
    let align_prob_mats_with_rna_id_pairs: ProbMatsWithRnaIdPairs<T> = align_prob_mat_pairs_with_rna_id_pairs.iter().map(|(key, x)| (*key, x.align_prob_mat.clone())).collect();
    let bpp_mats: SparseProbMats<T> = prob_mat_sets.iter().map(|x| x.bpp_mat.clone()).collect();
    let parent = input_file_path.parent().unwrap();
    let prefix = input_file_path.file_stem().unwrap();
    let sa_file_path = String::from(parent.join(&format!("{}_consalign.aln", prefix.to_str().unwrap())).to_str().unwrap());
    TrainDatumPosterior {
      fasta_records: read_struct_align.fasta_records.clone(),
      observed_struct_align: read_struct_align,
      argmax: MeaStructAlign::<T>::new(),
      align_prob_mats_with_rna_id_pairs: align_prob_mats_with_rna_id_pairs,
      bpp_mats: bpp_mats,
      sa_file_path: sa_file_path,
    }
  }
}

impl<T: Clone + Copy + Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Sync + Send + Display + Eq> MeaStructAlign<T> {
  pub fn new() -> MeaStructAlign<T> {
    MeaStructAlign {
      cols: Cols::new(),
      rightmost_bp_cols_with_cols: ColsWithCols::<T>::default(),
      right_bp_col_sets_with_cols: ColSetsWithCols::<T>::default(),
      pos_map_sets: PosMapSets::<T>::new(),
      rna_ids: RnaIds::new(),
      ea: 0.,
      bp_col_pairs: SparsePosMat::<T>::default(),
      mix_bpp_mat: SparseProbMat::<T>::default(),
      align_probs_avg: Probs::new(),
    }
  }

  pub fn set_right_bp_info(&mut self, bpp_mats: &SparseProbMats<T>) {
    let sa_len = self.cols.len();
    for i in 0 .. sa_len {
      let ref pos_maps = self.pos_map_sets[i];
      for j in i + 1 .. sa_len {
        let ref pos_maps_2 = self.pos_map_sets[j];
        let pos_map_pairs: Vec<(T, T)> = pos_maps.iter().zip(pos_maps_2.iter()).map(|(&x, &y)| (x, y)).collect();
        let mut bpp_sum = 0.;
        for (pos_map_pair, &rna_id) in pos_map_pairs.iter().zip(self.rna_ids.iter()) {
          let ref bpp_mat = bpp_mats[rna_id];
          match bpp_mat.get(pos_map_pair) {
            Some(&bpp) => {
              bpp_sum += bpp;
            }, None => {},
          }
        }
        let (i, j) = (T::from_usize(i).unwrap() + T::one(), T::from_usize(j).unwrap() + T::one());
        match self.right_bp_col_sets_with_cols.get_mut(&i) {
          Some(right_bp_cols) => {
            right_bp_cols.push((j, bpp_sum));
          }, None => {
            let mut right_bp_cols = PosProbSeq::<T>::new();
            right_bp_cols.push((j, bpp_sum));
            self.right_bp_col_sets_with_cols.insert(i, right_bp_cols);
          },
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

  pub fn set_mix_bpp_mat(&mut self, bpp_mats: &SparseProbMats<T>, bpp_mat_alifold: &SparseProbMat<T>, feature_scores: &FeatureCountsPosterior) {
    let sa_len = self.cols.len();
    let num_of_rnas = self.rna_ids.len();
    for i in 0 .. sa_len {
      let ref pos_maps = self.pos_map_sets[i];
      let short_i = T::from_usize(i).unwrap();
      for j in i + 1 .. sa_len {
        let short_j = T::from_usize(j).unwrap();
        let pos_pair = (short_i, short_j);
        let bpp_alifold = match bpp_mat_alifold.get(&pos_pair) {
          Some(&bpp_alifold) => {
            bpp_alifold
          }, None => {0.},
        };
        let ref pos_maps_2 = self.pos_map_sets[j];
        let pos_map_pairs: Vec<(T, T)> = pos_maps.iter().zip(pos_maps_2.iter()).map(|(&x, &y)| (x, y)).collect();
        let mut bpp_sum = 0.;
        for (pos_map_pair, &rna_id) in pos_map_pairs.iter().zip(self.rna_ids.iter()) {
          let ref bpp_mat = bpp_mats[rna_id];
          match bpp_mat.get(pos_map_pair) {
            Some(&bpp) => {
              bpp_sum += bpp;
            }, None => {},
          }
        }
        let mut bpp_avg = bpp_sum / num_of_rnas as Prob;
        if bpp_alifold > NEG_INFINITY {
          bpp_avg = (bpp_avg + bpp_alifold) / 2.;
        }
        let weight = feature_scores.basepair_count_posterior * bpp_avg - 1.;
        if weight >= 0. {
          let pos_pair = (T::from_usize(i).unwrap(), T::from_usize(j).unwrap());
          self.mix_bpp_mat.insert(pos_pair, weight);
        }
      }
    }
  }

  pub fn set_align_probs_avg(&mut self, align_prob_mats_with_rna_id_pairs: &ProbMatsWithRnaIdPairs<T>) {
    let sa_len = self.cols.len();
    let num_of_rnas = self.rna_ids.len();
    for i in 0 .. sa_len {
      let ref pos_maps = self.pos_map_sets[i];
      let mut align_prob_sum = 0.;
      for j in 0 .. num_of_rnas {
        let rna_id = self.rna_ids[j];
        let pos = pos_maps[j];
        for k in j + 1 .. num_of_rnas {
          let rna_id_2 = self.rna_ids[k];
          let pos_2 = pos_maps[k];
          let ordered_rna_id_pair = if rna_id < rna_id_2 {(rna_id, rna_id_2)} else {(rna_id_2, rna_id)};
          let ref align_prob_mat = align_prob_mats_with_rna_id_pairs[&ordered_rna_id_pair];
          let pos_pair = if rna_id < rna_id_2 {(pos, pos_2)} else {(pos_2, pos)};
          match align_prob_mat.get(&pos_pair) {
            Some(&align_prob) => {
              align_prob_sum += align_prob;
            }, None => {},
          }
        }
      }
      self.align_probs_avg.push(align_prob_sum);
    }
  }
}

impl<T: Clone + Copy + Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Sync + Send + Display + Hash> ReadStructAlign<T> {
  pub fn new() -> ReadStructAlign<T> {
    ReadStructAlign {
      cols: Cols::new(),
      bp_col_pairs: SparsePosMat::<T>::default(),
      pos_map_sets: PosMapSets::<T>::new(),
      fasta_records: FastaRecords::new(),
    }
  }
}

impl FeatureCountsPosterior {
  pub fn new(init_val: FeatureCount) -> FeatureCountsPosterior {
    FeatureCountsPosterior {
      basepair_count_posterior: init_val,
      align_count_posterior: init_val,
    }
  }
}

pub const GAP: Char = '-' as Char;
pub const MAX_ITER_REFINE: usize = 20;
pub const TRAINED_FEATURE_SCORE_SETS_FILE_PATH_POSTERIOR: &'static str = "../src/trained_feature_scores.rs";
pub const MIN_LOG_GAMMA_BASEPAIR: i32 = 0;
pub const MIN_LOG_GAMMA_ALIGN: i32 = 2;
pub const MAX_LOG_GAMMA_BASEPAIR: i32 = 5;
pub const MAX_LOG_GAMMA_ALIGN: i32 = MAX_LOG_GAMMA_BASEPAIR;
pub const BRACKET_PAIRS: [(char, char); 9] = [('(', ')'), ('<', '>'), ('{', '}'), ('[', ']'), ('A', 'a'), ('B', 'b'), ('C', 'c'), ('D', 'd'), ('E', 'e'), ];
pub const DEFAULT_OFFSET_4_MAX_GAP_NUM_ALIGN: usize = 4 * DEFAULT_OFFSET_4_MAX_GAP_NUM;
pub const DEFAULT_MIN_BPP_ALIGN: Prob = 0.04;

pub fn consalign<T>(
  fasta_records: &FastaRecords,
  align_prob_mats_with_rna_id_pairs: &ProbMatsWithRnaIdPairs<T>,
  bpp_mats: &SparseProbMats<T>,
  feature_scores: &FeatureCountsPosterior,
  sa_file_path: &Path,
) -> MeaStructAlign<T>
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
    let ref sparse_bpp_mat = bpp_mats[i];
    let converted_seq = convert_seq(seq, i, sparse_bpp_mat);
    for j in i + 1 .. num_of_rnas {
      let ref seq_2 = fasta_records[j].seq;
      let ref sparse_bpp_mat_2 = bpp_mats[j];
      let converted_seq_2 = convert_seq(seq_2, j, sparse_bpp_mat_2);
      let pair_struct_align = get_mea_align(&(&converted_seq, &converted_seq_2), align_prob_mats_with_rna_id_pairs, bpp_mats, &feature_scores);
      mea_mat.insert((i, j), pair_struct_align.ea);
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
  let mut mea_struct_align = recursive_mea_struct_align(&progressive_tree, root, align_prob_mats_with_rna_id_pairs, &fasta_records, bpp_mats, &feature_scores);
  let bpp_mat_alifold = get_bpp_mat_alifold(&mea_struct_align, sa_file_path, fasta_records);
  mea_struct_align.set_mix_bpp_mat(bpp_mats, &bpp_mat_alifold, &feature_scores);
  consalifold(&mut mea_struct_align);
  for col in mea_struct_align.cols.iter_mut() {
    let mut pairs: Vec<(Base, RnaId)> = col.iter().zip(mea_struct_align.rna_ids.iter()).map(|(&x, &y)| (x, y)).collect();
    pairs.sort_by_key(|x| x.1.clone());
    *col = pairs.iter().map(|x| x.0).collect();
  }
  for pos_maps in mea_struct_align.pos_map_sets.iter_mut() {
    let mut pairs: Vec<(T, RnaId)> = pos_maps.iter().zip(mea_struct_align.rna_ids.iter()).map(|(&x, &y)| (x, y)).collect();
    pairs.sort_by_key(|x| x.1);
    *pos_maps = pairs.iter().map(|x| x.0).collect();
  }
  mea_struct_align.rna_ids.sort();
  mea_struct_align
}

pub fn recursive_mea_struct_align<T>(progressive_tree: &ProgressiveTree, node: NodeIndex<DefaultIx>, align_prob_mats_with_rna_id_pairs: &ProbMatsWithRnaIdPairs<T>, fasta_records: &FastaRecords, bpp_mats: &SparseProbMats<T>, feature_scores: &FeatureCountsPosterior) -> MeaStructAlign<T>
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Sync + Send + Display,
{
  let num_of_rnas = fasta_records.len();
  let rna_id = *progressive_tree.node_weight(node).unwrap();
  if rna_id < num_of_rnas {
    let ref seq = fasta_records[rna_id].seq;
    let ref sparse_bpp_mat = bpp_mats[rna_id];
    convert_seq(seq, rna_id, sparse_bpp_mat)
  } else {
    let mut neighbors = progressive_tree.neighbors_directed(node, Outgoing).detach();
    let child = neighbors.next_node(progressive_tree).unwrap();
    let child_mea_struct_align = recursive_mea_struct_align(progressive_tree, child, align_prob_mats_with_rna_id_pairs, fasta_records, bpp_mats, feature_scores);
    let child_2 = neighbors.next_node(progressive_tree).unwrap();
    let child_mea_struct_align_2 = recursive_mea_struct_align(progressive_tree, child_2, align_prob_mats_with_rna_id_pairs, fasta_records, bpp_mats, feature_scores);
    get_mea_align(&(&child_mea_struct_align, &child_mea_struct_align_2), align_prob_mats_with_rna_id_pairs, bpp_mats, feature_scores)
  }
}

pub fn convert_seq<T>(seq: &Seq, rna_id: RnaId, sparse_bpp_mat: &SparseProbMat<T>) -> MeaStructAlign<T>
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Sync + Send + Display,
{
  let mut converted_seq = MeaStructAlign::new();
  let seq_len = seq.len();
  converted_seq.cols = seq[1 .. seq_len - 1].iter().map(|&x| vec![x]).collect();
  converted_seq.pos_map_sets = (1 .. seq.len() - 1).map(|x| vec![T::from_usize(x).unwrap()]).collect();
  converted_seq.rna_ids = vec![rna_id];
  converted_seq.align_probs_avg = vec![1.; seq_len - 2];
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
  }
  for (&i, right_bp_cols) in &converted_seq.right_bp_col_sets_with_cols {
    let max = right_bp_cols.iter().map(|x| x.0).max().unwrap();
    converted_seq.rightmost_bp_cols_with_cols.insert(i, max);
  }
  converted_seq
}

pub fn get_mea_align<'a, T>(struct_align_pair: &MeaStructAlignPair<'a, T>, align_prob_mats_with_rna_id_pairs: &ProbMatsWithRnaIdPairs<T>, bpp_mats: &SparseProbMats<T>, feature_scores: &FeatureCountsPosterior) -> MeaStructAlign<T>
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Sync + Send + Display,
{
  let struct_align_len_pair = (struct_align_pair.0.cols.len(), struct_align_pair.1.cols.len());
  let rna_num_pair = (struct_align_pair.0.rna_ids.len(), struct_align_pair.1.rna_ids.len());
  let num_of_rnas = rna_num_pair.0 + rna_num_pair.1;
  let denom = (num_of_rnas * (num_of_rnas - 1) / 2) as Prob;
  let mut align_weight_mat = SparseProbMat::<T>::default();
  let ref rna_ids = struct_align_pair.0.rna_ids;
  let ref rna_ids_2 = struct_align_pair.1.rna_ids;
  let ref pos_map_sets = struct_align_pair.0.pos_map_sets;
  let ref pos_map_sets_2 = struct_align_pair.1.pos_map_sets;
  let struct_align_len_pair = (
    T::from_usize(struct_align_len_pair.0).unwrap(),
    T::from_usize(struct_align_len_pair.1).unwrap(),
  );
  let pseudo_col_quadruple = (
    T::zero(),
    struct_align_len_pair.0 + T::one(),
    T::zero(),
    struct_align_len_pair.1 + T::one(),
  );
  for i in range_inclusive(T::one(), struct_align_len_pair.0) {
    let long_i = i.to_usize().unwrap();
    let ref pos_maps = pos_map_sets[long_i - 1];
    for j in range_inclusive(T::one(), struct_align_len_pair.1) {
      let col_pair = (i, j);
      let long_j = j.to_usize().unwrap();
      let mut align_prob_sum = 0.;
      let ref pos_maps_2 = pos_map_sets_2[long_j - 1];
      for (&rna_id, &pos) in rna_ids.iter().zip(pos_maps.iter()) {
        for (&rna_id_2, &pos_2) in rna_ids_2.iter().zip(pos_maps_2.iter()) {
          let ordered_rna_id_pair = if rna_id < rna_id_2 {(rna_id, rna_id_2)} else {(rna_id_2, rna_id)};
          let ref align_prob_mat = align_prob_mats_with_rna_id_pairs[&ordered_rna_id_pair];
          let pos_pair = if rna_id < rna_id_2 {(pos, pos_2)} else {(pos_2, pos)};
          match align_prob_mat.get(&pos_pair) {
            Some(&align_prob) => {
              align_prob_sum += align_prob;
            }, None => {},
          }
        }
      }
      let align_weight = feature_scores.align_count_posterior * align_prob_sum / denom - 1.;
      if align_weight >= 0. {
        align_weight_mat.insert(col_pair, align_weight);
      }
    }
  }
  let mut mea_mats_with_col_pairs = MeaMatsWithPosPairs::default();
  for i in range_inclusive(T::one(), struct_align_len_pair.0).rev() {
    match struct_align_pair.0.rightmost_bp_cols_with_cols.get(&i) {
      Some(&j) => {
        for k in range_inclusive(T::one(), struct_align_len_pair.1).rev() {
          let col_pair_left = (i, k);
          if !align_weight_mat.contains_key(&col_pair_left) {continue;}
          match struct_align_pair.1.rightmost_bp_cols_with_cols.get(&k) {
            Some(&l) => {
              let col_quadruple = (i, j, k, l);
              let mea_mat = get_mea_mat(&mea_mats_with_col_pairs, &align_weight_mat, &col_quadruple);
              update_mea_mats_with_col_pairs(&mut mea_mats_with_col_pairs, &col_pair_left, struct_align_pair, &mea_mat, &align_weight_mat, feature_scores);
            }, None => {},
          }
        }
      }, None => {},
    }
  }
  let mea_mat = get_mea_mat(&mea_mats_with_col_pairs, &align_weight_mat, &pseudo_col_quadruple);
  let mut new_mea_struct_align = MeaStructAlign::new();
  new_mea_struct_align.ea = mea_mat[&(pseudo_col_quadruple.1 - T::one(), pseudo_col_quadruple.3 - T::one())];
  new_mea_struct_align.ea += struct_align_pair.0.align_probs_avg.iter().map(|&x| feature_scores.align_count_posterior * x / denom - 1.).sum::<Mea>();
  new_mea_struct_align.ea += struct_align_pair.1.align_probs_avg.iter().map(|&x| feature_scores.align_count_posterior * x / denom - 1.).sum::<Mea>();
  let mut new_rna_ids = struct_align_pair.0.rna_ids.clone();
  let mut rna_ids_append = struct_align_pair.1.rna_ids.clone();
  new_rna_ids.append(&mut rna_ids_append);
  new_mea_struct_align.rna_ids = new_rna_ids;
  let mut bp_pos_map_set_pairs = PosMapSetPairs::<T>::new();
  traceback(&mut new_mea_struct_align, struct_align_pair, &pseudo_col_quadruple, &pseudo_col_quadruple, &mea_mats_with_col_pairs, 0, &align_weight_mat, &mut bp_pos_map_set_pairs, feature_scores);
  let sa_len = new_mea_struct_align.cols.len();
  let col_gaps_only = vec![PSEUDO_BASE; num_of_rnas];
  for i in (0 .. sa_len).rev() {
    let ref col = new_mea_struct_align.cols[i];
    if *col == col_gaps_only {
      new_mea_struct_align.cols.remove(i);
      new_mea_struct_align.pos_map_sets.remove(i);
    }
  }
  let sa_len = new_mea_struct_align.cols.len();
  new_mea_struct_align.ea /= sa_len as Mea;
  for bp_pos_map_set_pair in &bp_pos_map_set_pairs {
    for i in 0 .. sa_len {
      let ref pos_maps = new_mea_struct_align.pos_map_sets[i];
      if *pos_maps != bp_pos_map_set_pair.0 {
        continue;
      }
      let short_i = T::from_usize(i).unwrap();
      for j in i + 1 .. sa_len {
        let ref pos_maps_2 = new_mea_struct_align.pos_map_sets[j];
        if *pos_maps_2 == bp_pos_map_set_pair.1 {
          let short_j = T::from_usize(j).unwrap();
          new_mea_struct_align.bp_col_pairs.insert((short_i, short_j));
          break;
        }
      }
    }
  }
  new_mea_struct_align.set_right_bp_info(bpp_mats);
  new_mea_struct_align.set_align_probs_avg(align_prob_mats_with_rna_id_pairs);
  new_mea_struct_align
}

pub fn traceback <'a, T>(new_mea_struct_align: &mut MeaStructAlign<T>, struct_align_pair: &MeaStructAlignPair<'a, T>, pseudo_col_quadruple: &PosQuadruple<T>, col_quadruple: &PosQuadruple<T>, mea_mats_with_col_pairs: &MeaMatsWithPosPairs<T>, offset: usize, align_weight_mat: &SparseProbMat<T>, bp_pos_map_set_pairs: &mut PosMapSetPairs<T>, feature_scores: &FeatureCountsPosterior)
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Sync + Send + Display,
{
  let rna_num_pair = (struct_align_pair.0.rna_ids.len(), struct_align_pair.1.rna_ids.len());
  let mut mea;
  let mea_mat = get_mea_mat(&mea_mats_with_col_pairs, &align_weight_mat, &col_quadruple);
  let (i, j, k, l) = *col_quadruple;
  let (mut u, mut v) = (j - T::one(), l - T::one());
  while u > i || v > k {
    let col_pair = (u, v);
    mea = mea_mat[&col_pair];
    let (long_u, long_v) = (u.to_usize().unwrap(), v.to_usize().unwrap());
    if u > i && v > k {
      match align_weight_mat.get(&col_pair) {
        Some(&align_prob_avg) => {
          let col_pair_4_match = (u - T::one(), v - T::one());
          let ea = mea_mat[&col_pair_4_match] + align_prob_avg;
          if ea == mea {
            let mut new_col = struct_align_pair.0.cols[long_u - 1].clone();
            let mut col_append = struct_align_pair.1.cols[long_v - 1].clone();
            new_col.append(&mut col_append);
            new_mea_struct_align.cols.insert(offset, new_col);
            let mut new_pos_map_sets = struct_align_pair.0.pos_map_sets[long_u - 1].clone();
            let mut pos_map_sets_append = struct_align_pair.1.pos_map_sets[long_v - 1].clone();
            new_pos_map_sets.append(&mut pos_map_sets_append);
            new_mea_struct_align.pos_map_sets.insert(offset, new_pos_map_sets);
            u = u - T::one();
            v = v - T::one();
            continue;
          }
        }, None => {},
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
                  let mut new_col = struct_align_pair.0.cols[long_u - 1].clone();
                  let mut col_append = struct_align_pair.1.cols[long_v - 1].clone();
                  new_col.append(&mut col_append);
                  new_mea_struct_align.cols.insert(offset, new_col);
                  let mut new_pos_map_sets = struct_align_pair.0.pos_map_sets[long_u - 1].clone();
                  let mut pos_map_sets_append = struct_align_pair.1.pos_map_sets[long_v - 1].clone();
                  new_pos_map_sets.append(&mut pos_map_sets_append);
                  new_mea_struct_align.pos_map_sets.insert(offset, new_pos_map_sets.clone());
                  let long_col_pair_left = (col_pair_left.0.to_usize().unwrap(), col_pair_left.1.to_usize().unwrap());
                  let mut new_col = struct_align_pair.0.cols[long_col_pair_left.0 - 1].clone();
                  let mut col_append = struct_align_pair.1.cols[long_col_pair_left.1 - 1].clone();
                  new_col.append(&mut col_append);
                  new_mea_struct_align.cols.insert(offset, new_col);
                  let mut new_pos_map_sets_2 = struct_align_pair.0.pos_map_sets[long_col_pair_left.0 - 1].clone();
                  let mut pos_map_sets_append = struct_align_pair.1.pos_map_sets[long_col_pair_left.1 - 1].clone();
                  new_pos_map_sets_2.append(&mut pos_map_sets_append);
                  new_mea_struct_align.pos_map_sets.insert(offset, new_pos_map_sets_2.clone());
                  bp_pos_map_set_pairs.push((new_pos_map_sets_2, new_pos_map_sets));
                  traceback(new_mea_struct_align, struct_align_pair, pseudo_col_quadruple, &(col_pair_left.0, u, col_pair_left.1, v), mea_mats_with_col_pairs, offset + 1, align_weight_mat, bp_pos_map_set_pairs, feature_scores);
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
            let mut new_col = struct_align_pair.0.cols[long_u - 1].clone();
            let mut col_append = vec![PSEUDO_BASE; rna_num_pair.1];
            new_col.append(&mut col_append);
            new_mea_struct_align.cols.insert(offset, new_col);
            let mut new_pos_map_sets = struct_align_pair.0.pos_map_sets[long_u - 1].clone();
            let mut pos_map_sets_append = vec![T::zero(); rna_num_pair.1];
            new_pos_map_sets.append(&mut pos_map_sets_append);
            new_mea_struct_align.pos_map_sets.insert(offset, new_pos_map_sets);
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
            let mut col_append = struct_align_pair.1.cols[long_v - 1].clone();
            new_col.append(&mut col_append);
            new_mea_struct_align.cols.insert(offset, new_col);
            let mut new_pos_map_sets = vec![T::zero(); rna_num_pair.0];
            let mut pos_map_sets_append = struct_align_pair.1.pos_map_sets[long_v - 1].clone();
            new_pos_map_sets.append(&mut pos_map_sets_append);
            new_mea_struct_align.pos_map_sets.insert(offset, new_pos_map_sets);
            v = v - T::one();
          }
        }, None => {},
      }
    }
  }
}

pub fn get_mea_mat<'a, T>(mea_mats_with_col_pairs: &MeaMatsWithPosPairs<T>, align_weight_mat: &SparseProbMat<T>, col_quadruple: &PosQuadruple<T>) -> SparseProbMat<T>
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Sync + Send + Display,
{
  let (i, j, k, l) = *col_quadruple;
  let mut mea_mat = SparseProbMat::<T>::default();
  for u in range(i, j) {
    for v in range(k, l) {
      let col_pair = (u, v);
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
      match align_weight_mat.get(&col_pair) {
        Some(&align_prob_avg) => {
          match mea_mat.get(&col_pair_4_match) {
            Some(&ea) => {
              let ea = ea + align_prob_avg;
              if ea > mea {
                mea = ea;
              }
            }, None => {},
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
      if mea > NEG_INFINITY {
        mea_mat.insert(col_pair, mea);
      }
    }
  }
  mea_mat
}

pub fn update_mea_mats_with_col_pairs<'a, T>(mea_mats_with_col_pairs: &mut MeaMatsWithPosPairs<T>, col_pair_left: &PosPair<T>, struct_align_pair: &MeaStructAlignPair<'a, T>, mea_mat: &SparseProbMat<T>, align_weight_mat: &SparseProbMat<T>, feature_scores: &FeatureCountsPosterior)
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Sync + Send + Display,
{
  let num_of_rnas = (struct_align_pair.0.rna_ids.len() + struct_align_pair.1.rna_ids.len()) as Prob;
  let (i, k) = *col_pair_left;
  let ref right_bp_cols = struct_align_pair.0.right_bp_col_sets_with_cols[&i];
  let ref right_bp_cols_2 = struct_align_pair.1.right_bp_col_sets_with_cols[&k];
  let align_weight_left = align_weight_mat[&col_pair_left];
  for &(j, weight) in right_bp_cols.iter() {
    for &(l, weight_2) in right_bp_cols_2.iter() {
      let col_pair_right = (j, l);
      if !align_weight_mat.contains_key(&col_pair_right) {continue;}
      let align_weight_right = align_weight_mat[&col_pair_right];
      let bpp_avg = (weight + weight_2) / num_of_rnas;
      let bpp_weight = feature_scores.basepair_count_posterior * bpp_avg - 1.;
      if bpp_weight < 0. {continue;}
      let basepair_align_prob_avg = bpp_weight + align_weight_left + align_weight_right;
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

pub fn consalifold<T>(sa: &mut MeaStructAlign<T>)
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer,
{
  sa.bp_col_pairs = SparsePosMat::<T>::default();
  let sa_len = sa.cols.len();
  let mut mea_mat = vec![vec![0.; sa_len]; sa_len];
  let sa_len = T::from_usize(sa_len).unwrap();
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
      match sa.mix_bpp_mat.get(&(i, j)) {
        Some(&mix_bpp) => {
          let ea = mea_mat[long_i + 1][long_j - 1] + mix_bpp;
          if ea > mea {
            mea = ea;
          }
        }, None => {},
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
    } else {
      match sa.mix_bpp_mat.get(&pos_pair) {
        Some(&mix_bpp) => {
          if mea == mea_mat[long_i + 1][long_j - 1] + mix_bpp {
            pos_pair_stack.push((i + T::one(), j - T::one()));
            sa.bp_col_pairs.insert(pos_pair);
            continue;
          }
        }, None => {},
      }
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
  sa.ea = mea_mat[0][sa_len.to_usize().unwrap() - 1];
}

pub fn constrain_posterior<'a, T>(
  thread_pool: &mut Pool,
  train_data: &mut TrainDataPosterior<T>,
  output_file_path: &Path,
)
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Sync + Send + Display,
{
  let mut max_acc = NEG_INFINITY;
  let mut argmax = FeatureCountsPosterior::new(NEG_INFINITY);
  let mut accs = Vec::new();
  for log_gamma_basepair in MIN_LOG_GAMMA_BASEPAIR .. MAX_LOG_GAMMA_BASEPAIR + 1 {
    let basepair_count_posterior = (2. as Prob).powi(log_gamma_basepair) + 1.;
    for log_gamma_align in MIN_LOG_GAMMA_ALIGN .. MAX_LOG_GAMMA_ALIGN + 1 {
      let align_count_posterior = (2. as Prob).powi(log_gamma_align) + 1.;
      let mut feature_scores = FeatureCountsPosterior::new(0.);
      feature_scores.basepair_count_posterior = basepair_count_posterior;
      feature_scores.align_count_posterior = align_count_posterior;
      thread_pool.scoped(|scope| {
        let ref ref_2_feature_scores = feature_scores;
        for train_datum in train_data.iter_mut() {
          let ref fasta_records = train_datum.fasta_records;
          let ref align_prob_mats_with_rna_id_pairs = train_datum.align_prob_mats_with_rna_id_pairs;
          let ref bpp_mats = train_datum.bpp_mats;
          let ref mut argmax = train_datum.argmax;
          let sa_file_path = Path::new(&train_datum.sa_file_path);
          scope.execute(move || {
            *argmax = consalign::<T>(
              fasta_records,
              align_prob_mats_with_rna_id_pairs,
              bpp_mats,
              ref_2_feature_scores,
              sa_file_path,
            );
          });
        }
      });
      let mcc = get_mcc(train_data);
      let sps = get_sps(train_data);
      println!("basepair_score: {}, align_score: {}, mcc: {}, sps: {}", feature_scores.basepair_count_posterior, feature_scores.align_count_posterior, mcc, sps);
      let acc = (mcc * sps).sqrt();
      accs.push((feature_scores.clone(), acc));
      if acc > max_acc {
        max_acc = acc;
        argmax = feature_scores;
      }
    }
  }
  write_feature_scores_trained_posterior(&argmax);
  write_accs(&accs, output_file_path);
}

pub fn get_mcc<T>(train_data: &TrainDataPosterior<T>) -> Prob
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Sync + Send + Display,
{
  let (mut pt, mut nt, mut pf, mut nf): (Prob, Prob, Prob, Prob) = (0., 0., 0., 0.);
  for train_datum in train_data {
    let ref ref_struct_align = train_datum.observed_struct_align;
    let ref argmax = train_datum.argmax;
    let ref fasta_records = train_datum.fasta_records;
    let num_of_rnas = train_datum.observed_struct_align.cols[0].len();
    for i in 0 .. num_of_rnas {
      let mut mat_ref = SparsePosMat::<T>::default();
      for bp_col_pair in &ref_struct_align.bp_col_pairs {
        let pos_map_pair = (ref_struct_align.pos_map_sets[bp_col_pair.0.to_usize().unwrap()][i], ref_struct_align.pos_map_sets[bp_col_pair.1.to_usize().unwrap()][i]);
        mat_ref.insert(pos_map_pair);
      }
      let mut mat_max = SparsePosMat::<T>::default();
      for bp_col_pair in &argmax.bp_col_pairs {
        let pos_map_pair = (argmax.pos_map_sets[bp_col_pair.0.to_usize().unwrap()][i], argmax.pos_map_sets[bp_col_pair.1.to_usize().unwrap()][i]);
        mat_max.insert(pos_map_pair);
      }
      let ref fasta_record = fasta_records[i];
      let seq_len = fasta_record.seq.len() + 2;
      for i in 1 .. seq_len - 1 {
        let short_i = T::from_usize(i).unwrap();
        for j in i + 1 .. seq_len - 1 {
          let short_j = T::from_usize(j).unwrap();
          let pos_pair = (short_i, short_j);
          let bin_ref = mat_ref.contains(&pos_pair);
          let bin_max = mat_max.contains(&pos_pair);
          if bin_ref == bin_max {
            if bin_max {
              pt += 1.;
            } else {
              nt += 1.;
            }
          } else {
            if bin_max {
              pf += 1.;
            } else {
              nf += 1.;
            }
          }
        }
      }
    }
  }
  let mcc = (pt * nt - pf * nf) / ((pt + pf) * (pt + nf) * (nt + pf) * (nt + nf)).sqrt();
  mcc
}

pub fn get_sps<T>(train_data: &TrainDataPosterior<T>) -> Prob
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Sync + Send + Display,
{
  let (mut tp, mut total): (Prob, Prob) = (0., 0.);
  for train_datum in train_data {
    let ref ref_struct_align = train_datum.observed_struct_align;
    let ref argmax = train_datum.argmax;
    let sa_len_ref = train_datum.observed_struct_align.cols.len();
    let sa_len_max = train_datum.argmax.cols.len();
    let num_of_rnas = train_datum.observed_struct_align.cols[0].len();
    for i in 0 .. num_of_rnas {
      for j in i + 1 .. num_of_rnas {
        let mut mat_ref = SparsePosMat::<T>::default();
        let mut mat_max = SparsePosMat::<T>::default();
        for k in 0 .. sa_len_ref {
          let ref pos_maps_ref = ref_struct_align.pos_map_sets[k];
          let pos_map_ref = pos_maps_ref[i];
          let pos_map_ref_2 = pos_maps_ref[j];
          let pos_map_pair_ref = (pos_map_ref, pos_map_ref_2);
          mat_ref.insert(pos_map_pair_ref);
        }
        for k in 0 .. sa_len_max {
          let ref pos_maps_max = argmax.pos_map_sets[k];
          let pos_map_max = pos_maps_max[i];
          let pos_map_max_2 = pos_maps_max[j];
          let pos_map_pair_max = (pos_map_max, pos_map_max_2);
          mat_max.insert(pos_map_pair_max);
        }
        for pos_pair_ref in &mat_ref {
          for pos_pair_max in &mat_max {
            if *pos_pair_ref == *pos_pair_max {
              tp += 1.;
            }
          }
          total += 1.;
        }
      }
    }
  }
  let sps = tp / total;
  sps
}

pub fn read_sa_from_stockholm_file<T>(stockholm_file_path: &Path) -> ReadStructAlign<T>
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Sync + Send + Display,
{
  let mut sa = ReadStructAlign::<T>::new();
  let reader_2_stockholm_file = BufReader::new(File::open(stockholm_file_path).unwrap());
  let mut seqs = Vec::<Seq>::new();
  let mut bp_col_pairs = SparsePosMat::<T>::default();
  let mut seq_ids = SeqIds::new();
  for string in reader_2_stockholm_file.lines() {
    let string = string.unwrap();
    if string.starts_with("#=GC SS_cons") {
      let substrings: Vec<&str> = string.split_whitespace().collect();
      let ref css = substrings[2];
      let mut stack = Vec::new();
      for (left, right) in BRACKET_PAIRS {
        for (i, c) in css.chars().enumerate() {
          if c == left {
            stack.push(i);
          } else if c == right {
            let i = T::from_usize(i).unwrap();
            let pos = stack.pop().unwrap();
            let pos = T::from_usize(pos).unwrap();
            bp_col_pairs.insert((pos, i));
          }
        }
      }
    }
    if string.len() == 0 || string.starts_with("#") {
      continue;
    } else if string.starts_with("//") {
      break;
    }
    let substrings: Vec<&str> = string.split_whitespace().collect();
    let seq_id = substrings[0];
    seq_ids.push(SeqId::from(seq_id));
    let seq = substrings[1];
    let seq = seq.chars().map(|x| convert_sa_char(x as u8)).collect();
    seqs.push(seq);
  }
  sa.bp_col_pairs = bp_col_pairs;
  let align_len = seqs[0].len();
  for i in 0 .. align_len {
    let col = seqs.iter().map(|x| x[i]).collect();
    sa.cols.push(col);
  }
  let num_of_rnas = sa.cols[0].len();
  let mut seq_lens = vec![0 as usize; num_of_rnas];
  sa.pos_map_sets = vec![vec![T::zero(); num_of_rnas]; align_len];
  let mut fasta_records = vec![FastaRecord::origin(); num_of_rnas];
  for i in 0 .. align_len {
    for j in 0 .. num_of_rnas {
      let base = sa.cols[i][j];
      if base != PSEUDO_BASE {
        fasta_records[j].seq.push(base);
        seq_lens[j] += 1;
        sa.pos_map_sets[i][j] = T::from_usize(seq_lens[j]).unwrap();
      }
    }
  }
  for i in 0 .. num_of_rnas {
    fasta_records[i].seq.insert(0, PSEUDO_BASE);
    fasta_records[i].seq.push(PSEUDO_BASE);
    fasta_records[i].fasta_id = seq_ids[i].clone();
  }
  sa.fasta_records = fasta_records;
  sa 
}

pub fn write_feature_scores_trained_posterior(feature_scores: &FeatureCountsPosterior) {
  let mut writer_2_trained_feature_scores_file = BufWriter::new(File::create(TRAINED_FEATURE_SCORE_SETS_FILE_PATH_POSTERIOR).unwrap());
  let buf_4_writer_2_trained_feature_scores_file = format!("use FeatureCountsPosterior;\nimpl FeatureCountsPosterior {{\npub fn load_trained_score_params() -> FeatureCountsPosterior {{\nFeatureCountsPosterior {{\nbasepair_count_posterior: {}f32, align_count_posterior: {}f32\n}}\n}}\n}}", feature_scores.basepair_count_posterior, feature_scores.align_count_posterior);
  let _ = writer_2_trained_feature_scores_file.write_all(buf_4_writer_2_trained_feature_scores_file.as_bytes());
}

pub fn write_accs(accs: &Accs, output_file_path: &Path) {
  let mut writer_2_output_file = BufWriter::new(File::create(output_file_path).unwrap());
  let mut buf_4_writer_2_output_file = String::new();
  for acc in accs {
    buf_4_writer_2_output_file.push_str(&format!("{},{},{}\n", acc.0.basepair_count_posterior, acc.0.align_count_posterior, acc.1));
  }
  let _ = writer_2_output_file.write_all(buf_4_writer_2_output_file.as_bytes());
}

pub fn get_bpp_mat_alifold<T>(sa: &MeaStructAlign<T>, sa_file_path: &Path, fasta_records: &FastaRecords) -> SparseProbMat<T>
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Sync + Send + Display,
{
  let mut writer_2_sa_file = BufWriter::new(File::create(sa_file_path.clone()).unwrap());
  let mut buf_4_writer_2_sa_file = format!("CLUSTAL format sequence alignment\n\n");
  let sa_len = sa.cols.len();
  let fasta_ids: Vec<FastaId> = sa.rna_ids.iter().map(|&x| fasta_records[x].fasta_id.clone()).collect();
  let max_seq_id_len = fasta_ids.iter().map(|x| x.len()).max().unwrap();
  for (i, &rna_id) in sa.rna_ids.iter().enumerate() {
    let ref seq_id = fasta_records[rna_id].fasta_id;
    buf_4_writer_2_sa_file.push_str(seq_id);
    let mut clustal_row = vec![' ' as Char; max_seq_id_len - seq_id.len() + 2];
    let mut sa_row = (0 .. sa_len).map(|x| {revert_char(sa.cols[x][i])}).collect::<Vec<Char>>();
    clustal_row.append(&mut sa_row);
    let clustal_row = unsafe {from_utf8_unchecked(&clustal_row)};
    buf_4_writer_2_sa_file.push_str(&clustal_row);
    buf_4_writer_2_sa_file.push_str("\n");
  }
  let _ = writer_2_sa_file.write_all(buf_4_writer_2_sa_file.as_bytes());
  let _ = writer_2_sa_file.flush();
  let sa_file_prefix = sa_file_path.file_stem().unwrap().to_str().unwrap();
  let arg = format!("--id-prefix={}", sa_file_prefix);
  let args = vec!["-p", sa_file_path.to_str().unwrap(), &arg, "--noPS", "--noDP"];
  let _ = run_command("RNAalifold", &args, "Failed to run RNAalifold");
  let mut bpp_mat_alifold = SparseProbMat::<T>::default();
  let cwd = env::current_dir().unwrap();
  let output_file_path = cwd.join(String::from(sa_file_prefix) + "_0001_ali.out");
  let output_file = BufReader::new(File::open(output_file_path.clone()).unwrap());
  for (k, line) in output_file.lines().enumerate() {
    if k == 0 {continue;}
    let line = line.unwrap();
    if !line.starts_with(" ") {continue;}
    let substrings: Vec<&str> = line.split_whitespace().collect();
    let i = T::from_usize(substrings[0].parse().unwrap()).unwrap() - T::one();
    let j = T::from_usize(substrings[1].parse().unwrap()).unwrap() - T::one();
    let mut bpp = String::from(substrings[3]);
    bpp.pop();
    let bpp = 0.01 * bpp.parse::<Prob>().unwrap();
    if bpp == 0. {continue;}
    bpp_mat_alifold.insert((i, j), bpp);
  }
  let _ = remove_file(sa_file_path);
  let _ = remove_file(output_file_path);
  bpp_mat_alifold
}

pub fn run_command(command: &str, args: &[&str], expect: &str) -> Output {
  Command::new(command).args(args).output().expect(expect)
}
