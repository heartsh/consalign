extern crate consprob_trained;
extern crate petgraph;

pub use consprob_trained::*;
pub use petgraph::graph::{DefaultIx, NodeIndex};
pub use petgraph::{Directed, Graph, Outgoing};
pub use std::env;
pub use std::fs::create_dir;
pub use std::fs::remove_file;
pub use std::fs::File;
pub use std::io::{BufRead, BufWriter};
pub use std::process::{Command, Output};

pub type PosMapSetPair<T> = (PosMaps<T>, PosMaps<T>);
pub type PosMapSetPairs<T> = Vec<PosMapSetPair<T>>;
#[derive(Clone)]
pub struct MeaStructAlign<T, U> {
  pub struct_align: StructAlign<T, U>,
  pub rightmost_bp_cols_with_cols: ColsWithCols<U>,
  pub right_bp_col_sets_with_cols: ColSetsWithCols<U>,
  pub rna_ids: RnaIds,
  pub sps: Mea,
}
pub type RnaIds = Vec<RnaId>;
pub type MeaStructAlignPair<'a, T, U> = (&'a MeaStructAlign<T, U>, &'a MeaStructAlign<T, U>);
pub type SparseMeaMat = HashMap<RnaIdPair, Mea>;
pub type ProgressiveTree = Graph<RnaId, Mea>;
pub type ClusterSizes = HashMap<RnaId, usize>;
pub type NodeIndexes = HashMap<RnaId, NodeIndex<DefaultIx>>;
pub type ColPairs<T> = HashSet<PosPair<T>>;
pub type ColsWithCols<T> = HashMap<T, T>;
pub type MeaMatsWithPosPairs<T> = HashMap<PosPair<T>, SparseProbMat<T>>;
pub type ColSetsWithCols<T> = HashMap<T, PosProbSeq<T>>;
pub type PosProbSeq<T> = Vec<(T, Prob)>;
#[derive(Clone, Debug)]
pub struct FeatureCountsPosterior {
  pub basepair_count_posterior: FeatureCount,
  pub align_count_posterior: FeatureCount,
}
pub type SparseProbMats<T> = Vec<SparseProbMat<T>>;
pub type SparseProbMatsWithRnaIdPairs<T> = HashMap<RnaIdPair, SparseProbMat<T>>;
pub type ProbMatsWithRnaIdPairs = HashMap<RnaIdPair, ProbMat>;
pub type ProbSetPairsWithRnaIdPairs = HashMap<RnaIdPair, ProbSetPair>;
pub type SparseMeaAlignMat<T, U> = HashMap<RnaIdPair, MeaStructAlign<T, U>>;
pub type ProbSets = Vec<Probs>;
pub type SparseProbs<T> = HashMap<T, Prob>;
pub type MeaSetsWithPoss<T> = HashMap<T, SparseProbs<T>>;
pub type AlignShell<T> = HashMap<T, PosPair<T>>;
pub type Seqs = Vec<Seq>;

pub type Inputs4WrappedConsalign<'a> = (
  &'a mut Pool,
  &'a FastaRecords,
  &'a Path,
  &'a Path,
  Prob,
  Prob,
  ScoringModel,
  TrainType,
  bool,
  Prob,
  Prob,
  bool,
);

pub type Inputs4Traceback<'a, T, U> = (
  &'a mut MeaStructAlign<T, U>,
  &'a MeaStructAlignPair<'a, T, U>,
  &'a PosQuadruple<U>,
  &'a MeaMatsWithPosPairs<U>,
  &'a SparseProbMat<U>,
  &'a mut PosMapSetPairs<T>,
  &'a FeatureCountsPosterior,
  &'a AlignShell<U>,
);

pub type Inputs4RecursiveMeaStructAlign<'a, T> = (
  &'a ProgressiveTree,
  NodeIndex<DefaultIx>,
  &'a SparseProbMatsWithRnaIdPairs<T>,
  &'a SparseProbMats<T>,
  &'a FastaRecords,
  &'a FeatureCountsPosterior,
  &'a ProbSetPairsWithRnaIdPairs,
  bool,
);

#[derive(Clone)]
pub struct SeqAlign<T> {
  pub pos_map_sets: PosMapSets<T>,
  pub seqs: Seqs,
}

#[derive(Clone)]
pub struct StructAlign<T, U> {
  pub seq_align: SeqAlign<T>,
  pub bp_pos_pairs: SparsePosMat<U>,
  pub unpaired_poss: SparsePoss<U>,
}

impl<T, U> Default for MeaStructAlign<T, U>
where
  T: HashIndex,
  U: HashIndex,
{
  fn default() -> Self {
    Self::new()
  }
}

impl<T, U> MeaStructAlign<T, U>
where
  T: HashIndex,
  U: HashIndex,
{
  pub fn new() -> MeaStructAlign<T, U> {
    MeaStructAlign {
      struct_align: StructAlign::<T, U>::new(),
      rightmost_bp_cols_with_cols: ColsWithCols::<U>::default(),
      right_bp_col_sets_with_cols: ColSetsWithCols::<U>::default(),
      rna_ids: RnaIds::new(),
      sps: 0.,
    }
  }

  pub fn set_right_bp_info(
    &mut self,
    bpp_mats: &SparseProbMats<T>,
    feature_scores: &FeatureCountsPosterior,
  ) {
    let sa_len = self.struct_align.seq_align.pos_map_sets.len();
    let num_of_rnas = self.rna_ids.len();
    for i in 0..sa_len {
      let pos_maps = &self.struct_align.seq_align.pos_map_sets[i];
      for j in i + 1..sa_len {
        let pos_maps_2 = &self.struct_align.seq_align.pos_map_sets[j];
        let pos_map_pairs: Vec<(T, T)> = pos_maps
          .iter()
          .zip(pos_maps_2.iter())
          .map(|(&x, &y)| (x, y))
          .collect();
        let mut bpp_sum = 0.;
        for (pos_map_pair, &rna_id) in pos_map_pairs.iter().zip(self.rna_ids.iter()) {
          let bpp_mat = &bpp_mats[rna_id];
          if let Some(&bpp) = bpp_mat.get(pos_map_pair) {
            bpp_sum += bpp;
          }
        }
        let (i, j) = (
          U::from_usize(i).unwrap() + U::one(),
          U::from_usize(j).unwrap() + U::one(),
        );
        let bpp_avg = bpp_sum / num_of_rnas as Prob;
        let weight = feature_scores.basepair_count_posterior * bpp_avg - 1.;
        if weight >= 0. {
          match self.right_bp_col_sets_with_cols.get_mut(&i) {
            Some(right_bp_cols) => {
              right_bp_cols.push((j, weight));
            }
            None => {
              let right_bp_cols = vec![(j, weight)];
              self.right_bp_col_sets_with_cols.insert(i, right_bp_cols);
            }
          }
        }
      }
      let i = U::from_usize(i).unwrap() + U::one();
      if let Some(right_bp_cols) = self.right_bp_col_sets_with_cols.get(&i) {
        let max = right_bp_cols.iter().map(|x| x.0).max().unwrap();
        self.rightmost_bp_cols_with_cols.insert(i, max);
      }
    }
  }

  pub fn set_sps(
    &mut self,
    align_prob_mats_with_rna_id_pairs: &SparseProbMatsWithRnaIdPairs<T>,
    insert_prob_set_pairs_with_rna_id_pairs: &ProbSetPairsWithRnaIdPairs,
  ) {
    let sa_len = self.struct_align.seq_align.pos_map_sets.len();
    let num_of_rnas = self.rna_ids.len();
    let (mut pt, mut total): (Mea, Mea) = (0., 0.);
    for i in 0..num_of_rnas {
      let rna_id = self.rna_ids[i];
      for j in i + 1..num_of_rnas {
        let rna_id_2 = self.rna_ids[j];
        let ordered_rna_id_pair = if rna_id < rna_id_2 {
          (rna_id, rna_id_2)
        } else {
          (rna_id_2, rna_id)
        };
        let align_prob_mat = &align_prob_mats_with_rna_id_pairs[&ordered_rna_id_pair];
        let insert_prob_set_pair = &insert_prob_set_pairs_with_rna_id_pairs[&ordered_rna_id_pair];
        for k in 0..sa_len {
          let pos_maps = &self.struct_align.seq_align.pos_map_sets[k];
          let pos = pos_maps[i];
          let pos_2 = pos_maps[j];
          let pos_pair = if rna_id < rna_id_2 {
            (pos, pos_2)
          } else {
            (pos_2, pos)
          };
          if pos_pair.0 != T::zero() || pos_pair.1 != T::zero() {
            if pos_pair.0 != T::zero() && pos_pair.1 != T::zero() {
              if let Some(&align_prob) = align_prob_mat.get(&pos_pair) {
                pt += align_prob;
              }
            } else if pos_pair.1 == T::zero() {
              let insert_prob = insert_prob_set_pair.0[pos_pair.0.to_usize().unwrap()];
              pt += insert_prob;
            } else if pos_pair.0 == T::zero() {
              let insert_prob = insert_prob_set_pair.1[pos_pair.1.to_usize().unwrap()];
              pt += insert_prob;
            }
            total += 1.;
          }
        }
      }
    }
    let sps = pt / total;
    self.sps = sps;
  }

  pub fn sort(&mut self) {
    for pos_maps in self.struct_align.seq_align.pos_map_sets.iter_mut() {
      let mut pairs: Vec<(T, RnaId)> = pos_maps
        .iter()
        .zip(self.rna_ids.iter())
        .map(|(&x, &y)| (x, y))
        .collect();
      pairs.sort_by_key(|x| x.1);
      *pos_maps = pairs.iter().map(|x| x.0).collect();
    }
    self.rna_ids.sort();
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

impl<T: HashIndex> Default for SeqAlign<T> {
  fn default() -> Self {
    Self::new()
  }
}

impl<T: HashIndex> SeqAlign<T> {
  pub fn new() -> SeqAlign<T> {
    SeqAlign {
      pos_map_sets: PosMapSets::<T>::default(),
      seqs: Seqs::new(),
    }
  }
}

impl<T: HashIndex, U: HashIndex> Default for StructAlign<T, U> {
  fn default() -> Self {
    Self::new()
  }
}

impl<T: HashIndex, U: HashIndex> StructAlign<T, U> {
  pub fn new() -> StructAlign<T, U> {
    StructAlign {
      seq_align: SeqAlign::<T>::new(),
      bp_pos_pairs: SparsePosMat::<U>::default(),
      unpaired_poss: SparsePoss::<U>::default(),
    }
  }
}

pub const GAP: Char = b'-';
pub const MIN_LOG_GAMMA_BASEPAIR: i32 = 0;
pub const MIN_LOG_GAMMA_ALIGN: i32 = MIN_LOG_GAMMA_BASEPAIR;
pub const MAX_LOG_GAMMA_BASEPAIR: i32 = 3;
pub const MAX_LOG_GAMMA_ALIGN: i32 = 7;
pub const BRACKET_PAIRS: [(char, char); 9] = [
  ('(', ')'),
  ('<', '>'),
  ('{', '}'),
  ('[', ']'),
  ('A', 'a'),
  ('B', 'b'),
  ('C', 'c'),
  ('D', 'd'),
  ('E', 'e'),
];
pub const DEFAULT_MIN_BPP_ALIGN: Prob = 2. * DEFAULT_MIN_BPP;
pub const DEFAULT_MIN_ALIGN_PROB_ALIGN: Prob = 2. * DEFAULT_MIN_ALIGN_PROB;
pub const DEFAULT_MIN_BPP_ALIGN_TURNER: Prob = DEFAULT_MIN_BPP;
pub const DEFAULT_MIN_ALIGN_PROB_ALIGN_TURNER: Prob = DEFAULT_MIN_ALIGN_PROB;
pub const MIX_COEFF: Prob = 0.5;
pub const TRAINED_FEATURE_SCORE_SETS_FILE_PATH_POSTERIOR: &str = "../src/trained_feature_scores.rs";
pub const BASEPAIR_COUNT_POSTERIOR_ALIFOLD: Prob = 2.;
pub const DEFAULT_SCORING_MODEL: &str = "ensemble";
pub enum ScoringModel {
  Ensemble,
  Turner,
  Trained,
}
pub const OUTPUT_DIR_PATH: &str = "assets/sampled_trnas";

pub fn build_guide_tree<T>(
  fasta_records: &FastaRecords,
  align_prob_mats_with_rna_id_pairs: &SparseProbMatsWithRnaIdPairs<T>,
  feature_scores: &FeatureCountsPosterior,
) -> (ProgressiveTree, NodeIndex<DefaultIx>)
where
  T: HashIndex,
{
  let num_of_rnas = fasta_records.len();
  let mut mea_mat = SparseMeaMat::default();
  let mut progressive_tree = ProgressiveTree::new();
  let mut cluster_sizes = ClusterSizes::default();
  let mut node_indexes = NodeIndexes::default();
  for i in 0..num_of_rnas {
    for j in i + 1..num_of_rnas {
      let rna_id_pair = (i, j);
      let align_prob_mat = &align_prob_mats_with_rna_id_pairs[&rna_id_pair];
      let mea = align_prob_mat
        .values()
        .filter(|&x| feature_scores.align_count_posterior * x - 1. >= 0.)
        .sum();
      mea_mat.insert(rna_id_pair, mea);
    }
  }
  for i in 0..num_of_rnas {
    let node_index = progressive_tree.add_node(i);
    cluster_sizes.insert(i, 1);
    node_indexes.insert(i, node_index);
  }
  let mut new_cluster_id = num_of_rnas;
  while !mea_mat.is_empty() {
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
      let cluster_id_pair = if i < argmax.0 {
        (i, argmax.0)
      } else {
        (argmax.0, i)
      };
      let obtained_ea = mea_mat.remove(&cluster_id_pair).unwrap();
      let cluster_id_pair_2 = if i < argmax.1 {
        (i, argmax.1)
      } else {
        (argmax.1, i)
      };
      let obtained_ea_2 = mea_mat.remove(&cluster_id_pair_2).unwrap();
      let new_ea = (cluster_size_pair.0 as Mea * obtained_ea
        + cluster_size_pair.1 as Mea * obtained_ea_2)
        / new_cluster_size as Mea;
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
  (progressive_tree, root)
}

pub fn consalign<T, U>(
  fasta_records: &FastaRecords,
  align_prob_mats_with_rna_id_pairs: &SparseProbMatsWithRnaIdPairs<T>,
  bpp_mats: &SparseProbMats<T>,
  feature_scores: &FeatureCountsPosterior,
  insert_prob_set_pairs_with_rna_id_pairs: &ProbSetPairsWithRnaIdPairs,
) -> MeaStructAlign<T, U>
where
  T: HashIndex,
  U: HashIndex,
{
  let (progressive_tree, root) = build_guide_tree(
    fasta_records,
    align_prob_mats_with_rna_id_pairs,
    feature_scores,
  );
  recursive_mea_struct_align((
    &progressive_tree,
    root,
    align_prob_mats_with_rna_id_pairs,
    bpp_mats,
    fasta_records,
    feature_scores,
    insert_prob_set_pairs_with_rna_id_pairs,
    true,
  ))
}

pub fn recursive_mea_struct_align<T, U>(
  inputs: Inputs4RecursiveMeaStructAlign<T>,
) -> MeaStructAlign<T, U>
where
  T: HashIndex,
  U: HashIndex,
{
  let (
    progressive_tree,
    node,
    align_prob_mats_with_rna_id_pairs,
    bpp_mats,
    fasta_records,
    feature_scores,
    insert_prob_set_pairs_with_rna_id_pairs,
    is_final,
  ) = inputs;
  let num_of_rnas = fasta_records.len();
  let rna_id = *progressive_tree.node_weight(node).unwrap();
  if rna_id < num_of_rnas {
    let seq = &fasta_records[rna_id].seq;
    convert_seq(seq, rna_id, bpp_mats, feature_scores)
  } else {
    let mut neighbors = progressive_tree.neighbors_directed(node, Outgoing).detach();
    let child = neighbors.next_node(progressive_tree).unwrap();
    let child_2 = neighbors.next_node(progressive_tree).unwrap();
    let child_mea_struct_align = recursive_mea_struct_align((
      progressive_tree,
      child,
      align_prob_mats_with_rna_id_pairs,
      bpp_mats,
      fasta_records,
      feature_scores,
      insert_prob_set_pairs_with_rna_id_pairs,
      false,
    ));
    let child_mea_struct_align_2 = recursive_mea_struct_align((
      progressive_tree,
      child_2,
      align_prob_mats_with_rna_id_pairs,
      bpp_mats,
      fasta_records,
      feature_scores,
      insert_prob_set_pairs_with_rna_id_pairs,
      false,
    ));
    get_mea_align(
      &(&child_mea_struct_align, &child_mea_struct_align_2),
      align_prob_mats_with_rna_id_pairs,
      bpp_mats,
      feature_scores,
      insert_prob_set_pairs_with_rna_id_pairs,
      is_final,
    )
  }
}

pub fn convert_seq<T, U>(
  seq: &Seq,
  rna_id: RnaId,
  bpp_mats: &SparseProbMats<T>,
  feature_scores: &FeatureCountsPosterior,
) -> MeaStructAlign<T, U>
where
  T: HashIndex,
  U: HashIndex,
{
  let mut converted_seq = MeaStructAlign::new();
  converted_seq.struct_align.seq_align.pos_map_sets = (1..seq.len() - 1)
    .map(|x| vec![T::from_usize(x).unwrap()])
    .collect();
  converted_seq.rna_ids = vec![rna_id];
  converted_seq.set_right_bp_info(bpp_mats, feature_scores);
  converted_seq
}

pub fn get_mea_align<T, U>(
  struct_align_pair: &MeaStructAlignPair<T, U>,
  align_prob_mats_with_rna_id_pairs: &SparseProbMatsWithRnaIdPairs<T>,
  bpp_mats: &SparseProbMats<T>,
  feature_scores: &FeatureCountsPosterior,
  insert_prob_set_pairs_with_rna_id_pairs: &ProbSetPairsWithRnaIdPairs,
  is_final: bool,
) -> MeaStructAlign<T, U>
where
  T: HashIndex,
  U: HashIndex,
{
  let struct_align_len_pair = (
    struct_align_pair
      .0
      .struct_align
      .seq_align
      .pos_map_sets
      .len(),
    struct_align_pair
      .1
      .struct_align
      .seq_align
      .pos_map_sets
      .len(),
  );
  let rna_num_pair = (
    struct_align_pair.0.rna_ids.len(),
    struct_align_pair.1.rna_ids.len(),
  );
  let num_of_rnas = rna_num_pair.0 + rna_num_pair.1;
  let denom = (rna_num_pair.0 * rna_num_pair.1) as Prob;
  let mut align_weight_mat = SparseProbMat::<U>::default();
  let rna_ids = &struct_align_pair.0.rna_ids;
  let rna_ids_2 = &struct_align_pair.1.rna_ids;
  let pos_map_sets = &struct_align_pair.0.struct_align.seq_align.pos_map_sets;
  let pos_map_sets_2 = &struct_align_pair.1.struct_align.seq_align.pos_map_sets;
  let struct_align_len_pair = (
    U::from_usize(struct_align_len_pair.0).unwrap(),
    U::from_usize(struct_align_len_pair.1).unwrap(),
  );
  let pseudo_col_quadruple = (
    U::zero(),
    struct_align_len_pair.0 + U::one(),
    U::zero(),
    struct_align_len_pair.1 + U::one(),
  );
  let mut align_shell = AlignShell::<U>::default();
  for i in range_inclusive(U::one(), struct_align_len_pair.0) {
    let long_i = i.to_usize().unwrap();
    let pos_maps = &pos_map_sets[long_i - 1];
    for j in range_inclusive(U::one(), struct_align_len_pair.1) {
      let col_pair = (i, j);
      let long_j = j.to_usize().unwrap();
      let mut align_prob_sum = 0.;
      let pos_maps_2 = &pos_map_sets_2[long_j - 1];
      for (&rna_id, &pos) in rna_ids.iter().zip(pos_maps.iter()) {
        for (&rna_id_2, &pos_2) in rna_ids_2.iter().zip(pos_maps_2.iter()) {
          let ordered_rna_id_pair = if rna_id < rna_id_2 {
            (rna_id, rna_id_2)
          } else {
            (rna_id_2, rna_id)
          };
          let align_prob_mat = &align_prob_mats_with_rna_id_pairs[&ordered_rna_id_pair];
          let pos_pair = if rna_id < rna_id_2 {
            (pos, pos_2)
          } else {
            (pos_2, pos)
          };
          if let Some(&align_prob) = align_prob_mat.get(&pos_pair) {
            align_prob_sum += align_prob;
          }
        }
      }
      let align_weight = feature_scores.align_count_posterior * align_prob_sum / denom - 1.;
      if align_weight >= 0. {
        align_weight_mat.insert(col_pair, align_weight);
        match align_shell.get_mut(&(i - U::one())) {
          Some(col_pair_2) => {
            col_pair_2.0 = col_pair_2.0.min(j - U::one());
            col_pair_2.1 = col_pair_2.1.max(j - U::one());
          }
          None => {
            align_shell.insert(i - U::one(), (j - U::one(), j - U::one()));
          }
        }
        match align_shell.get_mut(&i) {
          Some(col_pair_2) => {
            col_pair_2.0 = col_pair_2.0.min(j);
            col_pair_2.1 = col_pair_2.1.max(j);
          }
          None => {
            align_shell.insert(i, (j, j));
          }
        }
      }
    }
  }
  for i in range_inclusive(U::zero(), struct_align_len_pair.0).rev() {
    let touch = if align_shell.contains_key(&(i + U::one())) {
      align_shell[&(i + U::one())].0
    } else {
      U::zero()
    };
    match align_shell.get_mut(&i) {
      Some(col_pair) => {
        col_pair.0 = col_pair.0.min(touch);
      }
      None => {
        align_shell.insert(i, (touch, U::zero()));
      }
    }
  }
  for i in range_inclusive(U::one(), struct_align_len_pair.0) {
    let touch = if align_shell.contains_key(&(i - U::one())) {
      align_shell[&(i - U::one())].1
    } else {
      U::zero()
    };
    match align_shell.get_mut(&i) {
      Some(col_pair) => {
        col_pair.1 = col_pair.1.max(touch);
      }
      None => {
        align_shell.insert(i, (U::zero(), touch));
      }
    }
  }
  for i in range_inclusive(U::one(), struct_align_len_pair.0) {
    if let Some(col_pair) = align_shell.get_mut(&i) {
      col_pair.0 = col_pair.0.min(col_pair.1);
    }
  }
  align_shell.get_mut(&U::zero()).unwrap().0 = U::zero();
  align_shell.get_mut(&struct_align_len_pair.0).unwrap().1 = struct_align_len_pair.1;
  let mut mea_mats_with_col_pairs = MeaMatsWithPosPairs::default();
  for i in range_inclusive(U::one(), struct_align_len_pair.0).rev() {
    if let Some(&j) = struct_align_pair.0.rightmost_bp_cols_with_cols.get(&i) {
      for k in range_inclusive(U::one(), struct_align_len_pair.1).rev() {
        let col_pair_left = (i, k);
        if !align_weight_mat.contains_key(&col_pair_left) {
          continue;
        }
        if let Some(&l) = struct_align_pair.1.rightmost_bp_cols_with_cols.get(&k) {
          let col_quadruple = (i, j, k, l);
          let mea_mat = get_mea_mat(
            &mea_mats_with_col_pairs,
            &align_weight_mat,
            &col_quadruple,
            &align_shell,
          );
          update_mea_mats_with_col_pairs(
            &mut mea_mats_with_col_pairs,
            &col_pair_left,
            struct_align_pair,
            &mea_mat,
            &align_weight_mat,
          );
        }
      }
    }
  }
  let mut new_mea_struct_align = MeaStructAlign::new();
  let mut new_rna_ids = struct_align_pair.0.rna_ids.clone();
  let mut rna_ids_append = struct_align_pair.1.rna_ids.clone();
  new_rna_ids.append(&mut rna_ids_append);
  new_mea_struct_align.rna_ids = new_rna_ids;
  let mut bp_pos_map_set_pairs = PosMapSetPairs::<T>::new();
  traceback((
    &mut new_mea_struct_align,
    struct_align_pair,
    &pseudo_col_quadruple,
    &mea_mats_with_col_pairs,
    &align_weight_mat,
    &mut bp_pos_map_set_pairs,
    feature_scores,
    &align_shell,
  ));
  let sa_len = new_mea_struct_align
    .struct_align
    .seq_align
    .pos_map_sets
    .len();
  let pos_maps_with_gaps_only = vec![T::zero(); num_of_rnas];
  for i in (0..sa_len).rev() {
    let pos_maps = &new_mea_struct_align.struct_align.seq_align.pos_map_sets[i];
    if *pos_maps == pos_maps_with_gaps_only {
      new_mea_struct_align
        .struct_align
        .seq_align
        .pos_map_sets
        .remove(i);
    }
  }
  if is_final {
    let sa_len = new_mea_struct_align
      .struct_align
      .seq_align
      .pos_map_sets
      .len();
    for bp_pos_map_set_pair in &bp_pos_map_set_pairs {
      for i in 0..sa_len {
        let pos_maps = &new_mea_struct_align.struct_align.seq_align.pos_map_sets[i];
        if *pos_maps != bp_pos_map_set_pair.0 {
          continue;
        }
        let short_i = U::from_usize(i).unwrap();
        for j in i + 1..sa_len {
          let pos_maps_2 = &new_mea_struct_align.struct_align.seq_align.pos_map_sets[j];
          if *pos_maps_2 == bp_pos_map_set_pair.1 {
            let short_j = U::from_usize(j).unwrap();
            new_mea_struct_align
              .struct_align
              .bp_pos_pairs
              .insert((short_i, short_j));
            break;
          }
        }
      }
    }
    new_mea_struct_align.set_sps(
      align_prob_mats_with_rna_id_pairs,
      insert_prob_set_pairs_with_rna_id_pairs,
    );
  } else {
    new_mea_struct_align.set_right_bp_info(bpp_mats, feature_scores);
  }
  new_mea_struct_align
}

pub fn get_mea_mat<T>(
  mea_mats_with_col_pairs: &MeaMatsWithPosPairs<T>,
  align_weight_mat: &SparseProbMat<T>,
  col_quadruple: &PosQuadruple<T>,
  align_shell: &AlignShell<T>,
) -> SparseProbMat<T>
where
  T: HashIndex,
{
  let (i, j, k, l) = *col_quadruple;
  let mut mea_mat = SparseProbMat::<T>::default();
  for u in range(i, j) {
    let (begin, end) = align_shell[&u];
    let begin = begin.max(k);
    let end = end.min(l - T::one());
    for v in range_inclusive(begin, end) {
      let col_pair = (u, v);
      if u == i && v == k {
        mea_mat.insert(col_pair, 0.);
        continue;
      }
      let mut mea = NEG_INFINITY;
      if let Some(mea_mat_4_bpas) = mea_mats_with_col_pairs.get(&col_pair) {
        for (col_pair_left, mea_4_bpa) in mea_mat_4_bpas {
          let col_pair_4_match = (col_pair_left.0 - T::one(), col_pair_left.1 - T::one());
          if !(i < col_pair_left.0 && k < col_pair_left.1) {
            continue;
          }
          if let Some(&ea) = mea_mat.get(&col_pair_4_match) {
            let ea = ea + mea_4_bpa;
            if ea > mea {
              mea = ea;
            }
          }
        }
      }
      let col_pair_4_match = (u - T::one(), v - T::one());
      if let Some(&align_prob_avg) = align_weight_mat.get(&col_pair) {
        if let Some(&ea) = mea_mat.get(&col_pair_4_match) {
          let ea = ea + align_prob_avg;
          if ea > mea {
            mea = ea;
          }
        }
      }
      let col_pair_4_insert = (u - T::one(), v);
      if let Some(&ea) = mea_mat.get(&col_pair_4_insert) {
        if ea > mea {
          mea = ea;
        }
      }
      let col_pair_4_insert_2 = (u, v - T::one());
      if let Some(&ea) = mea_mat.get(&col_pair_4_insert_2) {
        if ea > mea {
          mea = ea;
        }
      }
      if mea > NEG_INFINITY {
        mea_mat.insert(col_pair, mea);
      }
    }
  }
  mea_mat
}

pub fn update_mea_mats_with_col_pairs<T, U>(
  mea_mats_with_col_pairs: &mut MeaMatsWithPosPairs<U>,
  col_pair_left: &PosPair<U>,
  struct_align_pair: &MeaStructAlignPair<T, U>,
  mea_mat: &SparseProbMat<U>,
  align_weight_mat: &SparseProbMat<U>,
) where
  T: HashIndex,
  U: HashIndex,
{
  let (i, k) = *col_pair_left;
  let right_bp_cols = &struct_align_pair.0.right_bp_col_sets_with_cols[&i];
  let right_bp_cols_2 = &struct_align_pair.1.right_bp_col_sets_with_cols[&k];
  let align_weight_left = align_weight_mat[col_pair_left];
  for &(j, weight) in right_bp_cols.iter() {
    for &(l, weight_2) in right_bp_cols_2.iter() {
      let col_pair_right = (j, l);
      if !align_weight_mat.contains_key(&col_pair_right) {
        continue;
      }
      let align_weight_right = align_weight_mat[&col_pair_right];
      let basepair_align_prob_avg = weight + weight_2 + align_weight_left + align_weight_right;
      let mea_4_bpa = basepair_align_prob_avg + mea_mat[&(j - U::one(), l - U::one())];
      match mea_mats_with_col_pairs.get_mut(&col_pair_right) {
        Some(mea_mat_4_bpas) => {
          mea_mat_4_bpas.insert(*col_pair_left, mea_4_bpa);
        }
        None => {
          let mut mea_mat_4_bpas = SparseProbMat::default();
          mea_mat_4_bpas.insert(*col_pair_left, mea_4_bpa);
          mea_mats_with_col_pairs.insert(col_pair_right, mea_mat_4_bpas);
        }
      }
    }
  }
}

pub fn traceback<T, U>(inputs: Inputs4Traceback<T, U>)
where
  T: HashIndex,
  U: HashIndex,
{
  let (
    new_mea_struct_align,
    struct_align_pair,
    col_quadruple,
    mea_mats_with_col_pairs,
    align_weight_mat,
    bp_pos_map_set_pairs,
    _feature_scores,
    align_shell,
  ) = inputs;
  let rna_num_pair = (
    struct_align_pair.0.rna_ids.len(),
    struct_align_pair.1.rna_ids.len(),
  );
  let mut mea;
  let mea_mat = get_mea_mat(
    mea_mats_with_col_pairs,
    align_weight_mat,
    col_quadruple,
    align_shell,
  );
  let (i, j, k, l) = *col_quadruple;
  let (mut u, mut v) = (j - U::one(), l - U::one());
  while u > i || v > k {
    let col_pair = (u, v);
    mea = mea_mat[&col_pair];
    let (long_u, long_v) = (u.to_usize().unwrap(), v.to_usize().unwrap());
    if u > i && v > k {
      if let Some(&align_prob_avg) = align_weight_mat.get(&col_pair) {
        let col_pair_4_match = (u - U::one(), v - U::one());
        let ea = mea_mat[&col_pair_4_match] + align_prob_avg;
        if ea == mea {
          let mut new_pos_map_sets =
            struct_align_pair.0.struct_align.seq_align.pos_map_sets[long_u - 1].clone();
          let mut pos_map_sets_append =
            struct_align_pair.1.struct_align.seq_align.pos_map_sets[long_v - 1].clone();
          new_pos_map_sets.append(&mut pos_map_sets_append);
          new_mea_struct_align
            .struct_align
            .seq_align
            .pos_map_sets
            .insert(0, new_pos_map_sets);
          u = u - U::one();
          v = v - U::one();
          continue;
        }
      }
      let mut is_basepair_match_found = false;
      if let Some(mea_mat_4_bpas) = mea_mats_with_col_pairs.get(&col_pair) {
        for (col_pair_left, mea_4_bpa) in mea_mat_4_bpas {
          if !(i < col_pair_left.0 && k < col_pair_left.1) {
            continue;
          }
          let col_pair_4_match = (col_pair_left.0 - U::one(), col_pair_left.1 - U::one());
          if let Some(&ea) = mea_mat.get(&col_pair_4_match) {
            let ea = ea + mea_4_bpa;
            if ea == mea {
              let mut new_pos_map_sets =
                struct_align_pair.0.struct_align.seq_align.pos_map_sets[long_u - 1].clone();
              let mut pos_map_sets_append =
                struct_align_pair.1.struct_align.seq_align.pos_map_sets[long_v - 1].clone();
              new_pos_map_sets.append(&mut pos_map_sets_append);
              new_mea_struct_align
                .struct_align
                .seq_align
                .pos_map_sets
                .insert(0, new_pos_map_sets.clone());
              traceback((
                new_mea_struct_align,
                struct_align_pair,
                &(col_pair_left.0, u, col_pair_left.1, v),
                mea_mats_with_col_pairs,
                align_weight_mat,
                bp_pos_map_set_pairs,
                _feature_scores,
                align_shell,
              ));
              let long_col_pair_left = (
                col_pair_left.0.to_usize().unwrap(),
                col_pair_left.1.to_usize().unwrap(),
              );
              let mut new_pos_map_sets_2 = struct_align_pair.0.struct_align.seq_align.pos_map_sets
                [long_col_pair_left.0 - 1]
                .clone();
              let mut pos_map_sets_append = struct_align_pair.1.struct_align.seq_align.pos_map_sets
                [long_col_pair_left.1 - 1]
                .clone();
              new_pos_map_sets_2.append(&mut pos_map_sets_append);
              new_mea_struct_align
                .struct_align
                .seq_align
                .pos_map_sets
                .insert(0, new_pos_map_sets_2.clone());
              bp_pos_map_set_pairs.push((new_pos_map_sets_2, new_pos_map_sets));
              u = col_pair_4_match.0;
              v = col_pair_4_match.1;
              is_basepair_match_found = true;
              break;
            }
          }
        }
      }
      if is_basepair_match_found {
        continue;
      }
    }
    if u > i {
      if let Some(&ea) = mea_mat.get(&(u - U::one(), v)) {
        if ea == mea {
          let mut new_pos_map_sets =
            struct_align_pair.0.struct_align.seq_align.pos_map_sets[long_u - 1].clone();
          let mut pos_map_sets_append = vec![T::zero(); rna_num_pair.1];
          new_pos_map_sets.append(&mut pos_map_sets_append);
          new_mea_struct_align
            .struct_align
            .seq_align
            .pos_map_sets
            .insert(0, new_pos_map_sets);
          u = u - U::one();
          continue;
        }
      }
    }
    if v > k {
      if let Some(&ea) = mea_mat.get(&(u, v - U::one())) {
        if ea == mea {
          let mut new_pos_map_sets = vec![T::zero(); rna_num_pair.0];
          let mut pos_map_sets_append =
            struct_align_pair.1.struct_align.seq_align.pos_map_sets[long_v - 1].clone();
          new_pos_map_sets.append(&mut pos_map_sets_append);
          new_mea_struct_align
            .struct_align
            .seq_align
            .pos_map_sets
            .insert(0, new_pos_map_sets);
          v = v - U::one();
        }
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
    _ => {
      panic!();
    }
  }
}

pub fn consalifold<T>(
  mix_bpp_mat: &SparseProbMat<T>,
  sa_len: T,
  basepair_count_posterior: Prob,
) -> SparsePosMat<T>
where
  T: HashIndex,
{
  let mix_bpp_mat: SparseProbMat<T> = mix_bpp_mat
    .iter()
    .filter(|x| basepair_count_posterior * x.1 - 1. >= 0.)
    .map(|x| (*x.0, *x.1))
    .collect();
  let mut mea_sets_with_cols = MeaSetsWithPoss::default();
  let mut right_bp_cols_with_cols = ColSetsWithCols::<T>::default();
  for (col_pair, &mix_bpp) in &mix_bpp_mat {
    match right_bp_cols_with_cols.get_mut(&col_pair.0) {
      Some(cols) => {
        cols.push((col_pair.1, mix_bpp));
      }
      None => {
        let cols = vec![(col_pair.1, mix_bpp)];
        right_bp_cols_with_cols.insert(col_pair.0, cols);
      }
    }
  }
  let mut rightmost_bp_cols_with_cols = ColsWithCols::<T>::default();
  for (&i, cols) in &right_bp_cols_with_cols {
    let max = cols.iter().map(|x| x.0).max().unwrap();
    rightmost_bp_cols_with_cols.insert(i, max);
  }
  for i in range_inclusive(T::one(), sa_len).rev() {
    if let Some(&j) = rightmost_bp_cols_with_cols.get(&i) {
      let col_pair = (i, j);
      let meas = get_meas(&mea_sets_with_cols, &col_pair);
      update_mea_sets_with_cols(
        &mut mea_sets_with_cols,
        col_pair.0,
        &meas,
        &right_bp_cols_with_cols,
      );
    }
  }
  let pseudo_col_pair = (T::zero(), sa_len + T::one());
  let mut bp_col_pairs = SparsePosMat::<T>::default();
  traceback_alifold(&mut bp_col_pairs, &pseudo_col_pair, &mea_sets_with_cols);
  bp_col_pairs
}

pub fn traceback_alifold<T>(
  bp_col_pairs: &mut SparsePosMat<T>,
  col_pair: &PosPair<T>,
  mea_sets_with_cols: &MeaSetsWithPoss<T>,
) where
  T: HashIndex,
{
  let mut mea;
  let meas = get_meas(mea_sets_with_cols, col_pair);
  let (i, j) = *col_pair;
  let mut k = j - T::one();
  while k > i {
    mea = meas[&k];
    let ea = meas[&(k - T::one())];
    if ea == mea {
      k = k - T::one();
    }
    if let Some(meas_4_bps) = mea_sets_with_cols.get(&k) {
      for (&col_left, mea_4_bp) in meas_4_bps {
        if i >= col_left {
          continue;
        }
        let col_4_bp = col_left - T::one();
        let ea = meas[&col_4_bp];
        let ea = ea + mea_4_bp;
        if ea == mea {
          let col_pair = (col_left, k);
          traceback_alifold(bp_col_pairs, &col_pair, mea_sets_with_cols);
          let col_pair = (col_left - T::one(), k - T::one());
          bp_col_pairs.insert(col_pair);
          k = col_4_bp;
          break;
        }
      }
    }
  }
}

pub fn update_mea_sets_with_cols<T>(
  mea_sets_with_cols: &mut MeaSetsWithPoss<T>,
  i: T,
  meas: &SparseProbs<T>,
  right_bp_cols_with_cols: &ColSetsWithCols<T>,
) where
  T: HashIndex,
{
  let right_bp_cols = &right_bp_cols_with_cols[&i];
  for &(j, weight) in right_bp_cols {
    let mea_4_bp = weight + meas[&(j - T::one())];
    match mea_sets_with_cols.get_mut(&j) {
      Some(meas_4_bps) => {
        meas_4_bps.insert(i, mea_4_bp);
      }
      None => {
        let mut meas_4_bps = SparseProbs::default();
        meas_4_bps.insert(i, mea_4_bp);
        mea_sets_with_cols.insert(j, meas_4_bps);
      }
    }
  }
}

pub fn get_meas<T>(mea_sets_with_cols: &MeaSetsWithPoss<T>, col_pair: &PosPair<T>) -> SparseProbs<T>
where
  T: HashIndex,
{
  let (i, j) = *col_pair;
  let mut meas = SparseProbs::<T>::default();
  for k in range(i, j) {
    if k == i {
      meas.insert(k, 0.);
      continue;
    }
    let mut mea = meas[&(k - T::one())];
    if let Some(meas_4_bps) = mea_sets_with_cols.get(&k) {
      for (&l, mea_4_bp) in meas_4_bps {
        if i >= l {
          continue;
        }
        let ea = meas[&(l - T::one())];
        let ea = ea + mea_4_bp;
        if ea > mea {
          mea = ea;
        }
      }
    }
    meas.insert(k, mea);
  }
  meas
}

pub fn run_command(command: &str, args: &[&str], expect: &str) -> Output {
  Command::new(command).args(args).output().expect(expect)
}

pub fn get_bpp_mat_alifold<T, U>(
  sa: &MeaStructAlign<T, U>,
  sa_file_path: &Path,
  fasta_records: &FastaRecords,
  output_dir_path: &Path,
) -> SparseProbMat<U>
where
  T: HashIndex,
  U: HashIndex,
{
  let cwd = env::current_dir().unwrap();
  let sa_file_path = cwd.join(sa_file_path);
  let mut writer_2_sa_file = BufWriter::new(File::create(sa_file_path.clone()).unwrap());
  let mut buf_4_writer_2_sa_file = "CLUSTAL format sequence alignment\n\n".to_string();
  let sa_len = sa.struct_align.seq_align.pos_map_sets.len();
  let fasta_ids: Vec<FastaId> = sa
    .rna_ids
    .iter()
    .map(|&x| fasta_records[x].fasta_id.clone())
    .collect();
  let max_seq_id_len = fasta_ids.iter().map(|x| x.len()).max().unwrap();
  for (i, &rna_id) in sa.rna_ids.iter().enumerate() {
    let seq_id = &fasta_records[rna_id].fasta_id;
    buf_4_writer_2_sa_file.push_str(seq_id);
    let mut clustal_row = vec![b' '; max_seq_id_len - seq_id.len() + 2];
    let seq = &sa.struct_align.seq_align.seqs[i];
    let mut sa_row = (0..sa_len)
      .map(|x| {
        let pos_map = sa.struct_align.seq_align.pos_map_sets[x][i]
          .to_usize()
          .unwrap();
        if pos_map == 0 {
          GAP
        } else {
          revert_char(seq[pos_map])
        }
      })
      .collect::<Vec<Char>>();
    clustal_row.append(&mut sa_row);
    let clustal_row = unsafe { from_utf8_unchecked(&clustal_row) };
    buf_4_writer_2_sa_file.push_str(clustal_row);
    buf_4_writer_2_sa_file.push('\n');
  }
  let _ = writer_2_sa_file.write_all(buf_4_writer_2_sa_file.as_bytes());
  let _ = writer_2_sa_file.flush();
  let sa_file_prefix = sa_file_path.file_stem().unwrap().to_str().unwrap();
  let arg = format!("--id-prefix={sa_file_prefix}");
  let args = vec![
    "-p",
    sa_file_path.to_str().unwrap(),
    &arg,
    "--noPS",
    "--noDP",
  ];
  let _ = env::set_current_dir(output_dir_path);
  let _ = run_command("RNAalifold", &args, "Failed to run RNAalifold");
  let _ = env::set_current_dir(cwd);
  let mut bpp_mat_alifold = SparseProbMat::<U>::default();
  let output_file_path = output_dir_path.join(String::from(sa_file_prefix) + "_0001_ali.out");
  let output_file = BufReader::new(File::open(output_file_path.clone()).unwrap());
  for (k, line) in output_file.lines().enumerate() {
    if k == 0 {
      continue;
    }
    let line = line.unwrap();
    if !line.starts_with(' ') {
      continue;
    }
    let substrings: Vec<&str> = line.split_whitespace().collect();
    let i = U::from_usize(substrings[0].parse().unwrap()).unwrap();
    let j = U::from_usize(substrings[1].parse().unwrap()).unwrap();
    let mut bpp = String::from(substrings[3]);
    bpp.pop();
    let bpp = 0.01 * bpp.parse::<Prob>().unwrap();
    if bpp == 0. {
      continue;
    }
    bpp_mat_alifold.insert((i, j), bpp);
  }
  let _ = remove_file(sa_file_path);
  let _ = remove_file(output_file_path);
  bpp_mat_alifold
}

pub fn get_mix_bpp_mat<T, U>(
  sa: &MeaStructAlign<T, U>,
  bpp_mats: &SparseProbMats<T>,
  bpp_mat_alifold: &SparseProbMat<U>,
  disable_alifold: bool,
) -> SparseProbMat<U>
where
  T: HashIndex,
  U: HashIndex,
{
  let mut mix_bpp_mat = SparseProbMat::<U>::default();
  let sa_len = sa.struct_align.seq_align.pos_map_sets.len();
  let num_of_rnas = sa.rna_ids.len();
  for i in 0..sa_len {
    let pos_maps = &sa.struct_align.seq_align.pos_map_sets[i];
    let short_i = U::from_usize(i).unwrap();
    for j in i + 1..sa_len {
      let short_j = U::from_usize(j).unwrap();
      let pos_pair = (short_i + U::one(), short_j + U::one());
      let bpp_alifold = if disable_alifold {
        NEG_INFINITY
      } else {
        match bpp_mat_alifold.get(&pos_pair) {
          Some(&bpp_alifold) => bpp_alifold,
          None => 0.,
        }
      };
      let pos_maps_2 = &sa.struct_align.seq_align.pos_map_sets[j];
      let pos_map_pairs: Vec<(T, T)> = pos_maps
        .iter()
        .zip(pos_maps_2.iter())
        .map(|(&x, &y)| (x, y))
        .collect();
      let mut bpp_sum = 0.;
      for (pos_map_pair, &rna_id) in pos_map_pairs.iter().zip(sa.rna_ids.iter()) {
        let bpp_mat = &bpp_mats[rna_id];
        if let Some(&bpp) = bpp_mat.get(pos_map_pair) {
          bpp_sum += bpp;
        }
      }
      let bpp_avg = bpp_sum / num_of_rnas as Prob;
      let mix_bpp = if disable_alifold {
        bpp_avg
      } else {
        MIX_COEFF * bpp_avg + (1. - MIX_COEFF) * bpp_alifold
      };
      if mix_bpp > 0. {
        mix_bpp_mat.insert(pos_pair, mix_bpp);
      }
    }
  }
  mix_bpp_mat
}

pub fn get_insert_prob_set_pair<T>(
  align_prob_mat: &SparseProbMat<T>,
  seq_len_pair: &(usize, usize),
) -> ProbSetPair
where
  T: HashIndex,
{
  let mut insert_prob_set_pair = (vec![1.; seq_len_pair.0], vec![1.; seq_len_pair.1]);
  for (&(i, j), align_prob) in align_prob_mat {
    let (i, j) = (i.to_usize().unwrap(), j.to_usize().unwrap());
    insert_prob_set_pair.0[i] -= align_prob;
    insert_prob_set_pair.1[j] -= align_prob;
  }
  insert_prob_set_pair
}

pub fn wrapped_consalign<T, U>(
  inputs: Inputs4WrappedConsalign,
) -> (MeaStructAlign<T, U>, FeatureCountsPosterior)
where
  T: HashIndex,
  U: HashIndex,
{
  let (
    thread_pool,
    fasta_records,
    output_dir_path,
    input_file_path,
    min_bpp,
    min_align_prob,
    scoring_model,
    train_type,
    disable_alifold,
    min_bpp_turner,
    min_align_prob_turner,
    disable_transplant,
  ) = inputs;
  let mut align_feature_score_sets = AlignFeatureCountSets::new(0.);
  if disable_transplant {
    align_feature_score_sets.transfer();
  } else {
    copy_feature_count_sets_align(
      &mut align_feature_score_sets,
      &FeatureCountSets::load_trained_score_params(),
    );
  }
  let seqs = fasta_records.iter().map(|x| &x.seq[..]).collect();
  let (prob_mat_sets_turner, align_prob_mat_pairs_with_rna_id_pairs_turner) =
    if matches!(scoring_model, ScoringModel::Ensemble)
      || matches!(scoring_model, ScoringModel::Turner)
    {
      consprob::<T>(
        thread_pool,
        &seqs,
        min_bpp_turner,
        min_align_prob_turner,
        false,
        true,
        &align_feature_score_sets,
      )
    } else {
      (
        ProbMatSets::<T>::default(),
        AlignProbMatSetsWithRnaIdPairs::<T>::default(),
      )
    };
  let align_prob_mats_with_rna_id_pairs_turner: SparseProbMatsWithRnaIdPairs<T> =
    align_prob_mat_pairs_with_rna_id_pairs_turner
      .iter()
      .map(|(key, x)| (*key, x.align_prob_mat.clone()))
      .collect();
  let bpp_mats_turner: SparseProbMats<T> = prob_mat_sets_turner
    .iter()
    .map(|x| x.bpp_mat.clone())
    .collect();
  drop(prob_mat_sets_turner);
  drop(align_prob_mat_pairs_with_rna_id_pairs_turner);
  let (prob_mat_sets_trained, align_prob_mat_pairs_with_rna_id_pairs_trained) =
    if matches!(scoring_model, ScoringModel::Ensemble)
      || matches!(scoring_model, ScoringModel::Trained)
    {
      consprob_trained::<T>(
        thread_pool,
        &seqs,
        min_bpp,
        min_align_prob,
        false,
        true,
        train_type,
      )
    } else {
      (
        ProbMatSets::<T>::default(),
        AlignProbMatSetsWithRnaIdPairs::<T>::default(),
      )
    };
  let align_prob_mats_with_rna_id_pairs_trained: SparseProbMatsWithRnaIdPairs<T> =
    align_prob_mat_pairs_with_rna_id_pairs_trained
      .iter()
      .map(|(key, x)| (*key, x.align_prob_mat.clone()))
      .collect();
  let bpp_mats_trained: SparseProbMats<T> = prob_mat_sets_trained
    .iter()
    .map(|x| x.bpp_mat.clone())
    .collect();
  drop(prob_mat_sets_trained);
  drop(align_prob_mat_pairs_with_rna_id_pairs_trained);
  let num_of_fasta_records = fasta_records.len();
  let mut bpp_mats_fused = vec![SparseProbMat::<T>::new(); num_of_fasta_records];
  let mut align_prob_mats_with_rna_id_pairs_fused = SparseProbMatsWithRnaIdPairs::<T>::default();
  let mut insert_prob_set_pairs_with_rna_id_pairs = ProbSetPairsWithRnaIdPairs::default();
  for rna_id_1 in 0..num_of_fasta_records {
    for rna_id_2 in rna_id_1 + 1..num_of_fasta_records {
      let rna_id_pair = (rna_id_1, rna_id_2);
      align_prob_mats_with_rna_id_pairs_fused.insert(rna_id_pair, SparseProbMat::<T>::default());
      insert_prob_set_pairs_with_rna_id_pairs.insert(rna_id_pair, (Probs::new(), Probs::new()));
    }
  }
  if matches!(scoring_model, ScoringModel::Ensemble) {
    thread_pool.scoped(|scope| {
      for (bpp_mat_fused, bpp_mat_turner, bpp_mat_trained) in multizip((
        bpp_mats_fused.iter_mut(),
        bpp_mats_turner.iter(),
        bpp_mats_trained.iter(),
      )) {
        scope.execute(move || {
          *bpp_mat_fused = bpp_mat_turner
            .iter()
            .map(|(pos_pair, &bpp)| (*pos_pair, 0.5 * bpp))
            .collect();
          for (pos_pair, bpp_trained) in bpp_mat_trained {
            let bpp_trained = 0.5 * bpp_trained;
            match bpp_mat_fused.get_mut(pos_pair) {
              Some(bpp_fused) => {
                *bpp_fused += bpp_trained;
              }
              None => {
                bpp_mat_fused.insert(*pos_pair, bpp_trained);
              }
            }
          }
        });
      }
    });
    thread_pool.scoped(|scope| {
      for (rna_id_pair, align_prob_mat_fused) in align_prob_mats_with_rna_id_pairs_fused.iter_mut()
      {
        let align_prob_mat_turner = &align_prob_mats_with_rna_id_pairs_turner[rna_id_pair];
        let align_prob_mat_trained = &align_prob_mats_with_rna_id_pairs_trained[rna_id_pair];
        scope.execute(move || {
          *align_prob_mat_fused = align_prob_mat_turner
            .iter()
            .map(|(pos_pair, &align_prob)| (*pos_pair, 0.5 * align_prob))
            .collect();
          for (pos_pair, align_prob_trained) in align_prob_mat_trained {
            let align_prob_trained = 0.5 * align_prob_trained;
            match align_prob_mat_fused.get_mut(pos_pair) {
              Some(align_prob_fused) => {
                *align_prob_fused += align_prob_trained;
              }
              None => {
                align_prob_mat_fused.insert(*pos_pair, align_prob_trained);
              }
            }
          }
        });
      }
    });
  } else if matches!(scoring_model, ScoringModel::Turner) {
    bpp_mats_fused = bpp_mats_turner.clone();
    align_prob_mats_with_rna_id_pairs_fused = align_prob_mats_with_rna_id_pairs_turner.clone();
  } else {
    bpp_mats_fused = bpp_mats_trained.clone();
    align_prob_mats_with_rna_id_pairs_fused = align_prob_mats_with_rna_id_pairs_trained.clone();
  }
  thread_pool.scoped(|scope| {
    for (rna_id_pair, insert_prob_set_pair) in insert_prob_set_pairs_with_rna_id_pairs.iter_mut() {
      let align_prob_mat = &align_prob_mats_with_rna_id_pairs_fused[rna_id_pair];
      let seq_len_pair = (
        fasta_records[rna_id_pair.0].seq.len(),
        fasta_records[rna_id_pair.1].seq.len(),
      );
      scope.execute(move || {
        *insert_prob_set_pair = get_insert_prob_set_pair(align_prob_mat, &seq_len_pair);
      });
    }
  });
  if !output_dir_path.exists() {
    let _ = create_dir(output_dir_path);
  }
  let input_file_prefix = input_file_path.file_stem().unwrap().to_str().unwrap();
  let sa_file_path = output_dir_path.join(&format!("{input_file_prefix}.aln"));
  let mut candidates = Vec::new();
  for log_gamma_align in MIN_LOG_GAMMA_ALIGN..MAX_LOG_GAMMA_ALIGN + 1 {
    let align_count_posterior = (2. as Prob).powi(log_gamma_align) + 1.;
    for log_gamma_basepair in MIN_LOG_GAMMA_BASEPAIR..MAX_LOG_GAMMA_BASEPAIR + 1 {
      let basepair_count_posterior = (2. as Prob).powi(log_gamma_basepair) + 1.;
      let mut feature_scores = FeatureCountsPosterior::new(align_count_posterior);
      feature_scores.basepair_count_posterior = basepair_count_posterior;
      candidates.push((feature_scores, MeaStructAlign::new()));
    }
  }
  thread_pool.scoped(|scope| {
    let ref_2_align_prob_mats_with_rna_id_pairs = &align_prob_mats_with_rna_id_pairs_fused;
    let ref_2_bpp_mats = &bpp_mats_fused;
    let ref_2_insert_prob_set_pairs_with_rna_id_pairs = &insert_prob_set_pairs_with_rna_id_pairs;
    for candidate in &mut candidates {
      scope.execute(move || {
        candidate.1 = consalign::<T, U>(
          fasta_records,
          ref_2_align_prob_mats_with_rna_id_pairs,
          ref_2_bpp_mats,
          &candidate.0,
          ref_2_insert_prob_set_pairs_with_rna_id_pairs,
        );
      });
    }
  });
  let mut sa = MeaStructAlign::new();
  let mut feature_scores = FeatureCountsPosterior::new(0.);
  for candidate in &candidates {
    let tmp_sa = &candidate.1;
    if tmp_sa.sps > sa.sps {
      feature_scores = candidate.0.clone();
      sa = tmp_sa.clone();
    }
  }
  sa.sort();
  sa.struct_align.seq_align.seqs = fasta_records.iter().map(|x| x.seq.clone()).collect();
  let bpp_mat_alifold = if disable_alifold {
    SparseProbMat::<U>::default()
  } else {
    get_bpp_mat_alifold(&sa, &sa_file_path, fasta_records, output_dir_path)
  };
  let mix_bpp_mat = get_mix_bpp_mat(&sa, &bpp_mats_fused, &bpp_mat_alifold, disable_alifold);
  let sa_len = sa.struct_align.seq_align.pos_map_sets.len();
  let sa_len = U::from_usize(sa_len).unwrap();
  sa.struct_align.bp_pos_pairs =
    consalifold(&mix_bpp_mat, sa_len, BASEPAIR_COUNT_POSTERIOR_ALIFOLD);
  (sa, feature_scores)
}
